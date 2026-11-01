import { UniversalNodeWorker } from './node-worker.js';

/** Readiness/reset barriers travel once around the P2P ring, without a server. */
export class PipelineSession {
  constructor(ring, config, { ui, onError = () => {}, createWorker = UniversalNodeWorker.create } = {}) {
    this.ring = ring; this.config = config; this.ui = ui; this.onError = onError;
    this.pending = new Map(); this.nextId = 0; this.active = false; this.failed = null;
    this.queue = Promise.resolve();
    ring.onControl(message => this.enqueue(() => this.control(message)));
    ring.onActivation(input => this.enqueue(() => this.activation(input)));
    ring.onError(error => this.fail(error));
    // Install all receivers before starting downloads or WebRTC negotiation.
    this.ready = Promise.all([
      createWorker(config, p => ui?.setLoadingProgress(p)).then(worker => { this.worker = worker; }),
      ring.initiateConnections(),
    ]).then(() => { ui?.setReady(); });
    this.ready.catch(error => this.fail(error));
  }
  enqueue(action) {
    this.queue = this.queue.then(async () => {
      await this.ready;
      if (!this.failed) await action();
    }).catch(error => this.fail(error));
  }
  async control(message) {
    if (!message || !['ready', 'prepare', 'mirror', 'finish'].includes(message.kind)) throw new Error('Invalid session control');
    if (this.config.isHost) {
      const waiter = this.pending.get(message.id);
      if (waiter) { this.pending.delete(message.id); clearTimeout(waiter.timer); waiter.resolve(); }
      return;
    }
    if (message.kind === 'prepare') {
      if (!Number.isFinite(message.temperature) || message.temperature < 0) throw new Error('Invalid session temperature');
      this.worker.reset(); this.ui?.reset(); this.temperature = message.temperature; this.active = true;
    } else if (message.kind === 'mirror') {
      this.ui?.appendToken(message.token);
    } else if (message.kind === 'finish') {
      this.active = false; this.ui?.finishText(); this.ui?.setStatus('Generation complete — ready for another prompt');
    }
    this.ring.sendControl(message);
  }
  async activation(input) {
    if (this.config.isHost || !this.active) throw new Error('Activation received outside an active worker session');
    this.ui?.beginStep();
    const result = await this.worker.processStep(input, this.worker.sequenceLength, this.temperature);
    this.ui?.endStep(result);
    if (result.type === 'token') this.ring.sendTokenToHost(result.token);
    else this.ring.sendActivation(result.activation);
  }
  async barrier(kind, fields = {}) {
    await this.ready;
    if (this.failed) throw this.failed;
    if (!this.config.isHost) throw new Error('Only host starts session controls');
    const id = ++this.nextId;
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => this.fail(new Error(`${kind} barrier timed out (120s)`)), 120000);
      this.pending.set(id, { resolve, reject, timer });
      try { this.ring.sendControl({ ...fields, kind, id }); }
      catch (error) { this.fail(error); }
    });
  }
  async prepare(temperature) {
    await this.barrier('prepare', { temperature });
    this.worker.reset(); this.active = true;
  }
  mirror(token) { this.ring.sendControl({ kind: 'mirror', token }); }
  async finish() { await this.barrier('finish'); this.active = false; }
  fail(error) {
    if (this.failed) return;
    this.failed = error instanceof Error ? error : new Error(String(error)); this.active = false;
    for (const { reject, timer } of this.pending.values()) { clearTimeout(timer); reject(this.failed); }
    this.pending.clear(); this.ui?.setError(this.failed); this.ring.close(); this.onError(this.failed);
  }
}
