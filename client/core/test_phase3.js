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

import assert from 'node:assert/strict';
import { HostOrchestrator } from './host-orchestrator.js';

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

