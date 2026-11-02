import { validateRingConfig } from '../core/node-worker.js';

/** Mount into Prathamesh Patil's (Member 1) page. Networking invokes these explicit lifecycle hooks. */
export class WorkerUI {
  constructor(root, config) {
    validateRingConfig(config);
    if (!root?.ownerDocument) throw new TypeError('WorkerUI requires a DOM element');
    this.root = root;
    this.document = root.ownerDocument;
    this.decoder = new TextDecoder('utf-8');
    this.text = '';
    this.processed = 0;
    this.timer = null;
    this.panel = this.document.createElement('section');
    this.panel.setAttribute('aria-label', 'Pipeline worker');
    this.panel.style.cssText = 'padding:1rem;border:1px solid #64748b;border-radius:12px;font-family:system-ui;overflow-wrap:anywhere';
    const add = tag => this.panel.appendChild(this.document.createElement(tag));
    this.badge = add('h2');
    this.badge.textContent = `Node ${config.nodeIndex} of ${config.totalNodes} — Computing Layer${config.assignedLayers.length > 1 ? 's' : ''} ${config.assignedLayers.join(', ')}${config.isSampler ? ' + LM Head' : ''}`;
    this.status = add('p');
    this.status.setAttribute('role', 'status');
    this.progress = add('progress');
    this.progress.max = 100;
    this.progress.value = 0;
    this.progress.setAttribute('aria-label', 'Weight download');
    this.metrics = add('p');
    this.output = add('pre');
    this.output.style.cssText = 'white-space:pre-wrap;overflow-wrap:anywhere';
    this.output.setAttribute('aria-label', 'Generated text mirror');
    this.root.appendChild(this.panel);
    this.setStatus('Waiting for weights');
  }
  setStatus(message) { this.status.textContent = String(message); }
  setLoadingProgress(pct) {
    if (!Number.isFinite(pct)) throw new TypeError('Invalid progress');
    this.progress.value = Math.max(0, Math.min(100, pct));
    this.setStatus(`Loading weights: ${Math.round(this.progress.value)}%`);
  }
  setReady() { this.progress.value = 100; this.setStatus('Ready — waiting for activations'); }
  beginStep() {
    this.started = performance.now();
    this.setStatus('Computing assigned layers');
    this.panel.style.outline = '2px solid #22c55e';
    clearTimeout(this.timer);
    this.timer = setTimeout(() => { this.panel.style.outline = ''; }, 250);
  }
  endStep(result) {
    this.processed++;
    this.metrics.textContent = `${this.processed} steps · ${(performance.now() - (this.started ?? performance.now())).toFixed(1)} ms last compute`;
    this.setStatus(result.type === 'token' ? `Sampled byte ${result.token} — returning to host` : 'Activation ready — forwarding downstream');
  }
  // Call once per broadcast token. Do not also append inside endStep (avoids duplicates).
  appendToken(token) {
    if (!Number.isInteger(token) || token < 0 || token > 255) throw new RangeError('Invalid token byte');
    this.text += this.decoder.decode(Uint8Array.of(token), { stream: true });
    this.output.textContent = this.text;
  }
  finishText() { this.text += this.decoder.decode(); this.output.textContent = this.text; }
  setError(error) { this.setStatus(`Error: ${error instanceof Error ? error.message : String(error)}`); this.panel.style.outline = ''; }
  reset() {
    this.decoder = new TextDecoder('utf-8'); this.text = ''; this.processed = 0;
    this.output.textContent = ''; this.metrics.textContent = '';
    clearTimeout(this.timer); this.panel.style.outline = ''; this.setReady();
  }
  destroy() { clearTimeout(this.timer); this.panel.remove(); }
}
