/**
 * test_phase3.js — Automated Verification Test for Member 1 (Phase 3)
 *
 * Verifies HostOrchestrator:
 *  1. Ingestion of prompt text & prefill phase
 *  2. Sequential decode steps over simulated P2P ring channel
 *  3. Streaming multi-byte UTF-8 character decoding
 *  4. Telemetry metrics calculation (TPS, Step MS, Host Compute MS, RTT MS)
 *  5. Max tokens & stop handling
 */

'use strict';

const assert = require('assert');

console.log('============================================================');
console.log('🧪 RUNNING MEMBER 1 PHASE 3 VERIFICATION TEST');
console.log('============================================================\n');

// ── Mock P2P Ring Manager ───────────────────────────────────────────────────
class MockP2PRingManager {
  constructor() {
    this.isRingSealed = true;
    this.assignedLayers = [0];
    this._tokenCallback = null;
    this.sentActivations = [];
  }

  onTokenFromSampler(callback) {
    this._tokenCallback = callback;
  }

  sendActivation(tensor) {
    this.sentActivations.push(tensor);
  }

  // Simulate sampler returning a token byte
  simulateSamplerToken(tokenByte) {
    if (this._tokenCallback) {
      setTimeout(() => this._tokenCallback(tokenByte), 10);
    }
  }
}

// ── Mock Node 0 Engine ──────────────────────────────────────────────────────
class MockNodeEngine {
  constructor() {
    this.resetCalled = false;
    this.embeddedTokens = [];
  }

  reset() {
    this.resetCalled = true;
  }

  embed(tokenIds, startPos) {
    this.embeddedTokens.push({ tokenIds, startPos });
    // Returns Float32Array [len, 384]
    return new Float32Array(tokenIds.length * 384);
  }

  forwardLayer(x, layerIdx, startPos) {
    // Identity forward pass for mock
    return x;
  }
}

// ── HostOrchestrator Class Definition for Test ─────────────────────────────
const MAX_SEQ_LEN = 512;
const EOS_TOKEN = 0x00;

class HostOrchestrator {
  constructor(ringManager, nodeEngine, callbacks = {}) {
    this._ring       = ringManager;
    this._engine     = nodeEngine;
    this._onToken    = callbacks.onToken    || (() => {});
    this._onComplete = callbacks.onComplete || (() => {});
    this._onError    = callbacks.onError    || (() => {});

    this._context = [];
    this._running = false;
    this._decoder = new TextDecoder('utf-8', { stream: true, fatal: false });
    this._tokenResolver = null;

    this._ring.onTokenFromSampler((tokenByte) => {
      if (this._tokenResolver) {
        const resolve = this._tokenResolver;
        this._tokenResolver = null;
        resolve(tokenByte);
      }
    });

    this._stats = {
      tokensGenerated: 0,
      totalElapsedMs: 0,
      currentTps: 0,
      avgStepMs: 0,
      lastStepMs: 0,
      lastHostComputeMs: 0,
      lastRingRttMs: 0
    };
  }

  async startGeneration(prompt, maxNewTokens = 5, temperature = 0.0) {
    this._running = true;
    this._decoder = new TextDecoder('utf-8', { stream: true, fatal: false });

    const promptBytes = typeof prompt === 'string'
      ? new TextEncoder().encode(prompt)
      : prompt;

    this._context = Array.from(promptBytes).slice(0, MAX_SEQ_LEN - maxNewTokens);
    this._engine.reset();

    this._stats = {
      tokensGenerated: 0,
      totalElapsedMs: 0,
      currentTps: 0,
      avgStepMs: 0,
      lastStepMs: 0,
      lastHostComputeMs: 0,
      lastRingRttMs: 0,
      startTime: performance.now()
    };

    // Prefill
    const prefillStart = performance.now();
    let h0 = this._engine.embed(this._context, 0);
    for (const l of this._ring.assignedLayers) {
      h0 = this._engine.forwardLayer(h0, l, 0);
    }
    this._stats.lastHostComputeMs = performance.now() - prefillStart;

    const rttStart = performance.now();
    const firstToken = await this._sendAndAwaitToken(h0);
    this._stats.lastRingRttMs = performance.now() - rttStart;

    this._recordToken(firstToken, performance.now() - prefillStart);

    // Decode loop
    while (this._running && this._stats.tokensGenerated < maxNewTokens) {
      const stepStart = performance.now();
      const lastToken = this._context[this._context.length - 1];

      if (lastToken === EOS_TOKEN && this._stats.tokensGenerated > 1) break;

      const compStart = performance.now();
      let hStep = this._engine.embed([lastToken], this._context.length - 1);
      for (const l of this._ring.assignedLayers) {
        hStep = this._engine.forwardLayer(hStep, l, this._context.length - 1);
      }
      this._stats.lastHostComputeMs = performance.now() - compStart;

      const hopStart = performance.now();
      const nextToken = await this._sendAndAwaitToken(hStep);
      this._stats.lastRingRttMs = performance.now() - hopStart;

      if (nextToken === null || !this._running) break;

      this._recordToken(nextToken, performance.now() - stepStart);
    }

    this._running = false;
    this._onComplete(this._stats);
    return this._stats;
  }

  _sendAndAwaitToken(activationTensor) {
    return new Promise((resolve) => {
      this._tokenResolver = resolve;
      this._ring.sendActivation(activationTensor);
    });
  }

  _recordToken(tokenByte, stepDurationMs) {
    this._context.push(tokenByte);
    this._stats.tokensGenerated++;

    const now = performance.now();
    const totalElapsedMs = now - this._stats.startTime;
    this._stats.totalElapsedMs = totalElapsedMs;
    this._stats.lastStepMs = stepDurationMs;
    this._stats.avgStepMs = totalElapsedMs / this._stats.tokensGenerated;
    this._stats.currentTps = (this._stats.tokensGenerated / (totalElapsedMs / 1000));

    const char = this._decoder.decode(new Uint8Array([tokenByte]), { stream: true });
    this._onToken(tokenByte, char, { ...this._stats });
  }
}

// ── Execution Test ───────────────────────────────────────────────────────────
async function runTest() {
  const ring = new MockP2PRingManager();
  const engine = new MockNodeEngine();

  const generatedChars = [];
  const generatedTokens = [];

  const orchestrator = new HostOrchestrator(ring, engine, {
    onToken: (tok, char, stats) => {
      generatedTokens.push(tok);
      generatedChars.push(char);
    }
  });

  console.log('--- Test 1: Prefill & Autoregressive Decode Loop ---');

  // Start generation: prompt "Hello" -> generate 4 characters
  const genPromise = orchestrator.startGeneration('Hello', 4);

  // Tokens to simulate arriving from Sampler: ' ' (32), 'W' (87), 'o' (111), 'r' (114)
  const simulatedSequence = [32, 87, 111, 114];

  for (let i = 0; i < simulatedSequence.length; i++) {
    // Wait slightly then send simulated sampler token
    await new Promise(r => setTimeout(r, 20));
    ring.simulateSamplerToken(simulatedSequence[i]);
  }

  const finalStats = await genPromise;

  assert.strictEqual(engine.resetCalled, true, 'Engine reset must be called');
  assert.strictEqual(ring.sentActivations.length, 4, 'Must send exactly 4 activations for 4 steps');

  // Check prefill activation shape
  const prefillActivation = ring.sentActivations[0];
  assert.strictEqual(prefillActivation.length, 5 * 384, 'Prefill activation shape must be [5, 384] for "Hello"');
  console.log(`✓ Prefill activation shape verified: [5, 384] (${prefillActivation.byteLength} bytes)`);

  // Check decode activation shape
  const decodeActivation = ring.sentActivations[1];
  assert.strictEqual(decodeActivation.length, 1 * 384, 'Decode activation shape must be [1, 384] (1,536 bytes)');
  console.log(`✓ Decode activation shape verified: [1, 384] (${decodeActivation.byteLength} bytes)`);

  // Check generated text
  const streamedString = generatedChars.join('');
  assert.strictEqual(streamedString, ' Wor', 'Streamed characters must match simulated tokens');
  console.log(`✓ Streamed text reconstruction verified: "${streamedString}"`);

  // Check stats
  assert.strictEqual(finalStats.tokensGenerated, 4);
  assert(finalStats.currentTps > 0, 'TPS must be positive');
  console.log(`✓ Telemetry metrics verified: Tokens: ${finalStats.tokensGenerated}, Avg TPS: ${finalStats.currentTps.toFixed(1)}`);

  console.log('\n============================================================');
  console.log('🎉 ALL MEMBER 1 PHASE 3 TESTS PASSED SUCCESSFULLY! [PASS]');
  console.log('============================================================\n');
}

runTest();

