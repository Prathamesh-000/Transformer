import { TransformerDynamicBlock } from './transformer.js';
import { loadDynamicNodeWeights } from './weights.js';

export function validateRingConfig(config) {
  const { nodeIndex: i, totalNodes: n, assignedLayers: layers, isHost, isSampler } = config;
  if (!Number.isInteger(n) || n < 2 || n > 6 || !Number.isInteger(i) || i < 0 || i >= n) throw new RangeError('Invalid ring node index/count');
  const base = Math.floor(6 / n), remainder = 6 % n;
  const first = i * base + Math.min(i, remainder);
  const count = base + (i < remainder ? 1 : 0);
  if (!Array.isArray(layers) || layers.length !== count || layers.some((v, j) => v !== first + j)) throw new RangeError('Layer assignment does not match the planned partition');
  if (isHost !== (i === 0) || isSampler !== (i === n - 1)) throw new Error('Role flags do not match ring position');
}

export class UniversalNodeWorker {
  constructor(weights, config) {
    validateRingConfig(config);
    this.assignedLayers = [...config.assignedLayers];
    this.isHost = config.isHost;
    this.isSampler = config.isSampler;
    this.transformer = new TransformerDynamicBlock(weights, this.assignedLayers);
    this.failed = false;
  }
  static async create(config, onProgress) {
    validateRingConfig(config);
    const weights = await loadDynamicNodeWeights(config.assignedLayers, config.isHost, config.isSampler, onProgress);
    return new UniversalNodeWorker(weights, config);
  }
  async processStep(input, startPos, temperature = 0) {
    if (this.failed) throw new Error('Reset required after a failed step');
    if (!Number.isFinite(temperature) || temperature < 0) throw new RangeError('Invalid temperature');
    if (startPos !== this.sequenceLength) throw new RangeError(`Expected startPos ${this.sequenceLength}`);
    try {
      let x = this.isHost ? this.transformer.embed(input, startPos) : input;
      for (const layer of this.assignedLayers) x = this.transformer.forwardLayer(x, layer, startPos);
      if (this.isSampler) return { type: 'token', ...this.transformer.sampleNextToken(x, temperature) };
      return { type: 'activation', activation: x };
    } catch (error) {
      // Some layers may already have advanced. Never resume a partially updated stage.
      this.failed = true;
      throw error;
    }
  }
  reset() { this.transformer.reset(); this.failed = false; }
  get sequenceLength() { return this.transformer.sequenceLength; }
}
