/** Host prefill/decode driver. A single step is in flight; stop drains it safely. */
export class HostOrchestrator {
  constructor(ringManager, nodeEngine, callbacks = {}, session = null) {
    this._ring = ringManager; this._engine = nodeEngine; this._callbacks = callbacks; this._session = session;
    this._running = false; this._context = []; this._stats = {}; this._pending = null;
    this._ring.onTokenFromSampler(token => {
      if (!this._pending) return;
      const { resolve, reject, timer } = this._pending;
      clearTimeout(timer); this._pending = null;
      if (!Number.isInteger(token) || token < 0 || token > 255) reject(new Error('Invalid token received'));
      else resolve(token);
    });
  }
  async startGeneration(prompt, maxNewTokens = 200, temperature = 0) {
    if (this._running) throw new Error('Generation already in progress');
    if (!this._ring.isRingSealed) throw new Error('Ring is not ready');
    if (!Number.isInteger(maxNewTokens) || maxNewTokens < 1 || maxNewTokens > 512) throw new RangeError('maxNewTokens must be 1..512');
    if (!Number.isFinite(temperature) || temperature < 0) throw new RangeError('Invalid temperature');
    const bytes = typeof prompt === 'string' ? new TextEncoder().encode(prompt) : prompt;
    if (!(bytes instanceof Uint8Array) || !bytes.length) throw new Error('Enter a nonempty UTF-8 prompt');
    // Last sampled byte need not be embedded, so P + generated - 1 must fit.
    if (bytes.length + maxNewTokens - 1 > 512) throw new RangeError('Prompt bytes + max tokens - 1 must fit the 512-position context');
    this._running = true; this._stopRequested = false; this._failure = null;
    this._context = Array.from(bytes); this._decoder = new TextDecoder('utf-8');
    this._stats = { tokensGenerated: 0, totalElapsedMs: 0, currentTps: 0, avgStepMs: 0, lastStepMs: 0, lastHostComputeMs: 0, lastRingRttMs: 0 };
    try {
      if (this._session) await this._session.prepare(temperature);
      else this._engine.reset();
      this._stats.startTime = performance.now();
      let input = bytes, position = 0;
      while (!this._stopRequested && this._stats.tokensGenerated < maxNewTokens) {
        if (this._failure) throw this._failure;
        const start = performance.now();
        let activation;
        if (this._engine.processStep) {
          const result = await this._engine.processStep(input, position, temperature);
          if (result.type !== 'activation') throw new Error('Host must produce an activation');
          activation = result.activation;
        } else {
          activation = this._engine.embed(input, position);
          for (const layer of this._ring.assignedLayers) activation = this._engine.forwardLayer(activation, layer, position);
        }
        this._stats.lastHostComputeMs = performance.now() - start;
        const transitStart = performance.now();
        const token = await this._sendAndAwaitToken(activation);
        this._stats.lastRingRttMs = performance.now() - transitStart; // Includes remote compute, not pure network RTT.
        position += input.length;
        this._context.push(token); this._stats.tokensGenerated++;
        this._stats.totalElapsedMs = performance.now() - this._stats.startTime;
        this._stats.lastStepMs = performance.now() - start;
        this._stats.avgStepMs = this._stats.totalElapsedMs / this._stats.tokensGenerated;
        this._stats.currentTps = this._stats.tokensGenerated * 1000 / Math.max(this._stats.totalElapsedMs, 0.001);
        this._session?.mirror(token);
        this._callbacks.onToken?.(token, this._decoder.decode(Uint8Array.of(token), { stream: true }), this.stats);
        input = Uint8Array.of(token);
        // All 256 values are byte tokens; the plan does not reserve NUL as EOS.
      }
      if (this._failure) throw this._failure;
      if (this._session) await this._session.finish();
      const tail = this._decoder.decode();
      if (tail) this._callbacks.onText?.(tail); // Flush text without inventing a token byte.
      this._callbacks.onComplete?.(this.stats);
      return this.stats;
    } catch (error) {
      this._session?.fail(error); this._callbacks.onError?.(error); throw error;
    } finally { this._running = false; }
  }
  _sendAndAwaitToken(activation) {
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => { this._pending = null; reject(new Error('Inference step timed out (120s)')); }, 120000);
      this._pending = { resolve, reject, timer };
      try { this._ring.sendActivation(activation); }
      catch (error) { clearTimeout(timer); this._pending = null; reject(error); }
    });
  }
  stop() { this._stopRequested = true; }
  abort(error) {
    this._failure = error;
    if (this._pending) { clearTimeout(this._pending.timer); this._pending.reject(error); this._pending = null; }
  }
  get isRunning() { return this._running; }
  get stats() { return { ...this._stats }; }
  get context() { return [...this._context]; }
}
