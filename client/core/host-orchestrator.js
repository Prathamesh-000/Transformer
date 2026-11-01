/**
 * host-orchestrator.js — Pipeline Driver & Autoregressive Loop for Node 0 (Phone 1)
 *
 * Member 1 Deliverable (Phase 3):
 * Coordinates the full distributed generation cycle from the Host phone:
 *
 * 1. Prefill Phase:
 *    - Ingests prompt text -> converts to UTF-8 byte array [b₀, b₁, ... b_{P-1}]
 *    - Runs Node 0's assigned layers (Embedding + Layer 0 or Layers 0-1)
 *    - Caches K, V in Node 0's KV-cache
 *    - Sends activation tensor h₀ [P, 384] downstream via P2P ring
 *    - Awaits first sampled byte token from the final Sampler node
 *
 * 2. Decode Loop (1 token per step):
 *    - Embeds single new token at sequence position startPos
 *    - Runs Node 0's assigned layers (attending over cached K, V)
 *    - Sends h₀ [1, 384] (1,536 bytes) downstream via P2P ring
 *    - Awaits next byte token from Sampler node
 *    - Streams decoded character to UI via TextDecoder({ stream: true })
 *    - Updates real-time telemetry: TPS, Ring RTT, Host compute latency
 *    - Loops until maxTokens or null byte (0x00 EOS)
 */

const MAX_SEQ_LEN = 512;
const EOS_TOKEN   = 0x00; // Null byte used as end-of-sequence sentinel

class HostOrchestrator {
  /**
   * @param {DynamicP2PRingManager} ringManager - Active P2P ring transport manager
   * @param {object} nodeEngine - Transformer engine for Node 0 (implements embed, forwardLayer, reset)
   * @param {object} [callbacks] - Event callbacks
   * @param {function(number, string, object): void} [callbacks.onToken] - (tokenByte, decodedChar, stats)
   * @param {function(object): void} [callbacks.onComplete] - (finalStats)
   * @param {function(Error): void} [callbacks.onError] - (error)
   */
  constructor(ringManager, nodeEngine, callbacks = {}) {
    this._ring       = ringManager;
    this._engine     = nodeEngine;
    this._onToken    = callbacks.onToken    || (() => {});
    this._onComplete = callbacks.onComplete || (() => {});
    this._onError    = callbacks.onError    || (() => {});

    // Autoregressive context buffer
    this._context = [];
    this._running = false;

    // Streaming UTF-8 decoder handles multi-byte characters (emojis, unicode) cleanly
    this._decoder = new TextDecoder('utf-8', { stream: true, fatal: false });

    // Pending token promise resolver
    this._tokenResolver = null;
    this._tokenRejector = null;

    // Hook up token listener from Sampler shortcut channel
    this._ring.onTokenFromSampler((tokenByte) => {
      if (this._tokenResolver) {
        const resolve = this._tokenResolver;
        this._tokenResolver = null;
        this._tokenRejector = null;
        resolve(tokenByte);
      }
    });

    // Telemetry stats
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

  // ── Public Control API ─────────────────────────────────────────────────────

  /**
   * Start generating text from a prompt.
   * @param {string|Uint8Array} prompt - Input prompt text or byte array
   * @param {number} [maxNewTokens=200] - Maximum tokens to generate
   * @param {number} [temperature=0.0] - Sampling temperature (0.0 = greedy argmax)
   * @returns {Promise<object>} Resolves with final generation statistics
   */
  async startGeneration(prompt, maxNewTokens = 200, temperature = 0.0) {
    if (this._running) {
      console.warn('[HostOrchestrator] Generation already in progress.');
      return this._stats;
    }

    if (!this._ring.isRingSealed) {
      const err = new Error('Cannot start generation: P2P Ring is not fully sealed.');
      this._onError(err);
      throw err;
    }

    this._running = true;
    this._decoder = new TextDecoder('utf-8', { stream: true, fatal: false });

    // Convert prompt to byte array
    let promptBytes;
    if (typeof prompt === 'string') {
      promptBytes = prompt.length > 0
        ? new TextEncoder().encode(prompt)
        : new Uint8Array([EOS_TOKEN]); // Empty prompt -> feed null byte
    } else if (prompt instanceof Uint8Array) {
      promptBytes = prompt;
    } else {
      promptBytes = new Uint8Array([EOS_TOKEN]);
    }

    // Truncate to maximum context length
    this._context = Array.from(promptBytes).slice(0, MAX_SEQ_LEN - maxNewTokens);
    const promptLen = this._context.length;

    // Reset transformer engine & KV-cache on Node 0
    if (typeof this._engine.reset === 'function') {
      this._engine.reset();
    }

    // Reset stats
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

    console.log(`[HostOrchestrator] Starting generation. Prompt length: ${promptLen} bytes, Max new tokens: ${maxNewTokens}`);

    try {
      // ──────────────────────────────────────────────────────────────────────
      // 1. PREFILL PHASE: Process full prompt through Node 0's assigned layers
      // ──────────────────────────────────────────────────────────────────────
      const prefillStart = performance.now();

      // Embed all prompt tokens at startPos = 0
      let h0;
      if (typeof this._engine.embed === 'function') {
        h0 = this._engine.embed(this._context, /* startPos= */ 0);
        // Forward through each assigned layer on Node 0
        const assignedLayers = this._ring.assignedLayers || [0];
        for (const layerIdx of assignedLayers) {
          h0 = this._engine.forwardLayer(h0, layerIdx, /* startPos= */ 0);
        }
      } else if (typeof this._engine.forward === 'function') {
        // Fallback unified forward interface
        h0 = this._engine.forward(this._context, /* startPos= */ 0);
      } else {
        throw new Error('Node engine does not implement embed/forward.');
      }

      const hostPrefillComputeMs = performance.now() - prefillStart;
      this._stats.lastHostComputeMs = hostPrefillComputeMs;

      console.log(`[HostOrchestrator] Prefill computed in ${hostPrefillComputeMs.toFixed(1)}ms. Transmitting h₀ [${promptLen}, 384] downstream...`);

      // Transmit prefill activation matrix downstream and await token 1 from Sampler
      const rttStart = performance.now();
      const firstToken = await this._sendAndAwaitToken(h0);
      this._stats.lastRingRttMs = performance.now() - rttStart;

      if (firstToken === null || !this._running) {
        return this._finish();
      }

      this._recordToken(firstToken, performance.now() - prefillStart);

      // ──────────────────────────────────────────────────────────────────────
      // 2. DECODE LOOP: Generate 1 token per step autoregressively
      // ──────────────────────────────────────────────────────────────────────
      while (this._running && this._stats.tokensGenerated < maxNewTokens) {
        // Check context window boundary
        if (this._context.length >= MAX_SEQ_LEN) {
          console.log('[HostOrchestrator] Context window limit (512 tokens) reached.');
          break;
        }

        const stepStart = performance.now();
        const startPos = this._context.length - 1;
        const lastToken = this._context[this._context.length - 1];

        // Check EOS null byte sentinel
        if (lastToken === EOS_TOKEN && this._stats.tokensGenerated > 1) {
          console.log('[HostOrchestrator] End-of-sequence token (0x00) generated. Stopping.');
          break;
        }

        // Compute Node 0 forward for single new token
        const computeStart = performance.now();
        let hStep;

        if (typeof this._engine.embed === 'function') {
          hStep = this._engine.embed([lastToken], startPos);
          const assignedLayers = this._ring.assignedLayers || [0];
          for (const layerIdx of assignedLayers) {
            hStep = this._engine.forwardLayer(hStep, layerIdx, startPos);
          }
        } else {
          hStep = this._engine.forward([lastToken], startPos);
        }

        const hostStepComputeMs = performance.now() - computeStart;
        this._stats.lastHostComputeMs = hostStepComputeMs;

        // Transmit strictly 1,536 bytes (Float32Array[1, 384]) downstream
        const hopStart = performance.now();
        const nextToken = await this._sendAndAwaitToken(hStep);
        this._stats.lastRingRttMs = performance.now() - hopStart;

        if (nextToken === null || !this._running) {
          break;
        }

        const stepTotalMs = performance.now() - stepStart;
        this._recordToken(nextToken, stepTotalMs);
      }
    } catch (err) {
      console.error('[HostOrchestrator] Error during generation loop:', err);
      this._onError(err);
      throw err;
    } finally {
      return this._finish();
    }
  }

  /**
   * Stop generation loop immediately.
   */
  stop() {
    if (!this._running) return;
    console.log('[HostOrchestrator] Generation stopped by user.');
    this._running = false;

    if (this._tokenRejector) {
      const reject = this._tokenRejector;
      this._tokenResolver = null;
      this._tokenRejector = null;
      reject(new Error('Generation stopped by user'));
    }
  }

  // ── Internal Helpers ───────────────────────────────────────────────────────

  /**
   * Sends activation tensor downstream and returns a Promise that resolves
   * when the sampled token byte arrives from the Sampler node.
   * @param {Float32Array} activationTensor
   * @returns {Promise<number|null>}
   */
  _sendAndAwaitToken(activationTensor) {
    return new Promise((resolve, reject) => {
      if (!this._running) return resolve(null);

      // Backpressure safety timeout (10 seconds per hop)
      const timeout = setTimeout(() => {
        this._tokenResolver = null;
        this._tokenRejector = null;
        reject(new Error('DataChannel step timeout: Token not received within 10s.'));
      }, 10000);

      this._tokenResolver = (tokenByte) => {
        clearTimeout(timeout);
        resolve(tokenByte);
      };

      this._tokenRejector = (err) => {
        clearTimeout(timeout);
        resolve(null);
      };

      // Send 0x01 activation packet downstream
      try {
        this._ring.sendActivation(activationTensor);
      } catch (e) {
        clearTimeout(timeout);
        reject(e);
      }
    });
  }

  /**
   * Append token byte to context, update stats, and notify caller.
   */
  _recordToken(tokenByte, stepDurationMs) {
    this._context.push(tokenByte);
    this._stats.tokensGenerated++;

    const now = performance.now();
    const totalElapsedMs = now - this._stats.startTime;
    this._stats.totalElapsedMs = totalElapsedMs;
    this._stats.lastStepMs = stepDurationMs;
    this._stats.avgStepMs = totalElapsedMs / this._stats.tokensGenerated;
    this._stats.currentTps = (this._stats.tokensGenerated / (totalElapsedMs / 1000));

    // Decode byte chunk streamingly
    const char = this._decoder.decode(new Uint8Array([tokenByte]), { stream: true });

    // Notify UI
    this._onToken(tokenByte, char, { ...this._stats });
  }

  _finish() {
    this._running = false;
    // Flush any remaining buffered characters in decoder
    const remaining = this._decoder.decode();
    if (remaining) {
      this._onToken(0, remaining, { ...this._stats });
    }

    console.log(`[HostOrchestrator] Generation finished. Total tokens: ${this._stats.tokensGenerated}, Avg TPS: ${this._stats.currentTps.toFixed(1)}`);
    this._onComplete({ ...this._stats });
    return this._stats;
  }

  get isRunning() {
    return this._running;
  }

  get stats() {
    return { ...this._stats };
  }

  get context() {
    return [...this._context];
  }
}

export { HostOrchestrator };

