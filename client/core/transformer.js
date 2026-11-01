import { MODEL, LAYER_LAYOUT, validateLayers } from './weights.js';
const { width: D, heads: HEADS, context: MAX, vocab: VOCAB } = MODEL;
const HD = D / HEADS;

// PyTorch Linear convention: W[out, in], contiguous row-major FP32.
// Output-row and inner-dimension tiling keeps each weight row sequential.
function linear(x, w, bias, rows, input, output) {
  const y = new Float32Array(rows * output);
  for (let r = 0; r < rows; r++) {
    for (let tile = 0; tile < output; tile += 32) {
      for (let o = tile; o < Math.min(tile + 32, output); o++) {
        let sum = bias ? bias[o] : 0;
        for (let block = 0; block < input; block += 64) {
          for (let i = block; i < Math.min(block + 64, input); i++) sum += x[r * input + i] * w[o * input + i];
        }
        y[r * output + o] = sum;
      }
    }
  }
  return y;
}
function norm(x, w, b) {
  const y = new Float32Array(x.length);
  for (let base = 0; base < x.length; base += D) {
    let mean = 0, variance = 0;
    for (let j = 0; j < D; j++) mean += x[base + j];
    mean /= D;
    for (let j = 0; j < D; j++) variance += (x[base + j] - mean) ** 2;
    const scale = 1 / Math.sqrt(variance / D + 1e-5);
    for (let j = 0; j < D; j++) y[base + j] = (x[base + j] - mean) * scale * w[j] + b[j];
  }
  return y;
}
function tensor(value, length, name) {
  if (!(value instanceof Float32Array) || value.length !== length) throw new TypeError(`Invalid tensor ${name}: expected Float32Array[${length}]`);
}
function position(start, rows) {
  if (!Number.isInteger(start) || start < 0 || rows < 1 || start + rows > MAX) throw new RangeError('Sequence must fit positions 0..511');
}

export class TransformerDynamicBlock {
  constructor(weights, assignedLayers) {
    validateLayers(assignedLayers);
    if (!(weights.layers instanceof Map)) throw new TypeError('weights.layers must be a Map');
    this.weights = weights;
    this.assignedLayers = [...assignedLayers];
    this.caches = new Map();
    for (const layer of assignedLayers) {
      const w = weights.layers.get(layer);
      if (!w) throw new Error(`Missing layer ${layer}`);
      for (const [name, count] of LAYER_LAYOUT) tensor(w[name], count, `${layer}.${name}`);
      this.caches.set(layer, { k: new Float32Array(MAX * D), v: new Float32Array(MAX * D), length: 0 });
    }
  }
  embed(tokenIds, startPos) {
    if (!(tokenIds instanceof Uint8Array) && !Array.isArray(tokenIds)) throw new TypeError('Expected byte token array');
    position(startPos, tokenIds.length);
    tensor(this.weights.tokenEmb, VOCAB * D, 'tokenEmb');
    tensor(this.weights.posEmb, MAX * D, 'posEmb');
    if (Array.from(tokenIds).some(t => !Number.isInteger(t) || t < 0 || t >= VOCAB)) throw new RangeError('Tokens must be integers in 0..255');
    const out = new Float32Array(tokenIds.length * D);
    for (let r = 0; r < tokenIds.length; r++) {
      for (let j = 0; j < D; j++) out[r * D + j] = this.weights.tokenEmb[tokenIds[r] * D + j] + this.weights.posEmb[(startPos + r) * D + j];
    }
    return out;
  }
  forwardLayer(x, layerIdx, startPos) {
    if (!(x instanceof Float32Array) || x.length % D || !x.every(Number.isFinite)) throw new TypeError('Activation must contain finite rows of 384 floats');
    const rows = x.length / D;
    position(startPos, rows);
    const cache = this.caches.get(layerIdx);
    if (!cache) throw new RangeError(`Layer ${layerIdx} is not assigned`);
    if (cache.length !== startPos) throw new RangeError(`Layer ${layerIdx} expects position ${cache.length}; reset before a new prompt`);
    const w = this.weights.layers.get(layerIdx);
    const qkv = linear(norm(x, w.ln1w, w.ln1b), w.qkvW, w.qkvB, rows, D, 3 * D);
    for (let r = 0; r < rows; r++) {
      cache.k.set(qkv.subarray(r * 3 * D + D, r * 3 * D + 2 * D), (startPos + r) * D);
      cache.v.set(qkv.subarray(r * 3 * D + 2 * D, (r + 1) * 3 * D), (startPos + r) * D);
    }
    const attention = new Float32Array(rows * D);
    const scores = new Float64Array(startPos + rows);
    for (let r = 0; r < rows; r++) {
      const end = startPos + r + 1; // Causal: no access to later prefill positions.
      for (let head = 0; head < HEADS; head++) {
        const h = head * HD;
        let max = -Infinity;
        for (let t = 0; t < end; t++) {
          let score = 0;
          for (let j = 0; j < HD; j++) score += qkv[r * 3 * D + h + j] * cache.k[t * D + h + j];
          scores[t] = score / Math.sqrt(HD);
          max = Math.max(max, scores[t]);
        }
        let total = 0;
        for (let t = 0; t < end; t++) total += (scores[t] = Math.exp(scores[t] - max));
        for (let j = 0; j < HD; j++) {
          let value = 0;
          for (let t = 0; t < end; t++) value += scores[t] / total * cache.v[t * D + h + j];
          attention[r * D + h + j] = value;
        }
      }
    }
    const residual = linear(attention, w.oW, w.oB, rows, D, D);
    for (let i = 0; i < x.length; i++) residual[i] += x[i];
    const ff = linear(norm(residual, w.ln2w, w.ln2b), w.ffW1, w.ffB1, rows, D, 4 * D);
    for (let i = 0; i < ff.length; i++) {
      const v = ff[i];
      ff[i] = 0.5 * v * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (v + 0.044715 * v ** 3)));
    }
    const output = linear(ff, w.ffW2, w.ffB2, rows, 4 * D, D);
    for (let i = 0; i < output.length; i++) output[i] += residual[i];
    if (!output.every(Number.isFinite)) throw new Error('Non-finite transformer output; reset the session');
    cache.length = startPos + rows;
    return output;
  }
  sampleNextToken(h, temperature = 0) {
    if (!(h instanceof Float32Array) || !h.length || h.length % D || !h.every(Number.isFinite)) throw new TypeError('Invalid sampler activation');
    if (!Number.isFinite(temperature) || temperature < 0) throw new RangeError('Temperature must be finite and nonnegative');
    const w = this.weights;
    tensor(w.finalLnW, D, 'finalLnW'); tensor(w.finalLnB, D, 'finalLnB'); tensor(w.lmHeadW, VOCAB * D, 'lmHeadW');
    const logits = linear(norm(h.subarray(h.length - D), w.finalLnW, w.finalLnB), w.lmHeadW, null, 1, D, VOCAB);
    if (!logits.every(Number.isFinite)) throw new Error('Non-finite logits');
    let token = 0;
    for (let i = 1; i < VOCAB; i++) if (logits[i] > logits[token]) token = i;
    if (temperature > 0) {
      const max = logits[token];
      const probs = Float64Array.from(logits, v => Math.exp((v - max) / temperature));
      let draw = Math.random() * probs.reduce((a, b) => a + b, 0);
      for (let i = 0; i < VOCAB; i++) { draw -= probs[i]; if (draw < 0) { token = i; break; } }
    }
    return { token, logits };
  }
  reset() {
    for (const cache of this.caches.values()) { cache.k.fill(0); cache.v.fill(0); cache.length = 0; }
  }
  get sequenceLength() { return Math.min(...Array.from(this.caches.values(), cache => cache.length)); }
}
