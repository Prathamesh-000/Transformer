/** Headerless FP32 layout shared with tools/generate_weights.py. */
export const MODEL = Object.freeze({ width: 384, heads: 6, layers: 6, context: 512, vocab: 256 });
export const PARAMETER_COUNT = 11040768;
export const WEIGHT_BYTES = PARAMETER_COUNT * 4;
const D = MODEL.width;
export const LAYER_LAYOUT = Object.freeze([
  ['ln1w', D], ['ln1b', D], ['qkvW', 3 * D * D], ['qkvB', 3 * D],
  ['oW', D * D], ['oB', D], ['ln2w', D], ['ln2b', D],
  ['ffW1', 4 * D * D], ['ffB1', 4 * D], ['ffW2', 4 * D * D], ['ffB2', D],
]);
export function validateLayers(layers) {
  if (!Array.isArray(layers) || !layers.length || layers.some((v, i) =>
    !Number.isInteger(v) || v < 0 || v >= 6 || (i > 0 && v !== layers[i - 1] + 1))) {
    throw new RangeError('assignedLayers must be nonempty, consecutive, increasing layer IDs in 0..5');
  }
}

function selectedLayout(assignedLayers, isHost, isSampler) {
  validateLayers(assignedLayers);
  if (typeof isHost !== 'boolean' || typeof isSampler !== 'boolean') throw new TypeError('Role flags must be boolean');
  const entries = [];
  let offset = 0;
  function add(name, count, selected, layer) {
    if (selected) entries.push({ name, count, offset, layer });
    offset += count * 4;
  }
  add('tokenEmb', 256 * D, isHost);
  add('posEmb', 512 * D, isHost);
  for (let layer = 0; layer < 6; layer++) {
    for (const [name, count] of LAYER_LAYOUT) add(name, count, assignedLayers.includes(layer), layer);
  }
  add('finalLnW', D, isSampler);
  add('finalLnB', D, isSampler);
  add('lmHeadW', 256 * D, isSampler);
  return entries;
}

/** Stream the existing server's full response, retaining only this node's tensors.
 * No Float32Array view retains the complete 44 MB model in browser RAM.
 */
export async function loadDynamicNodeWeights(assignedLayers, isHost, isSampler, onProgress) {
  const entries = selectedLayout(assignedLayers, isHost, isSampler);
  const response = await fetch('/weights.bin');
  if (!response.ok) throw new Error(`Weight download failed: HTTP ${response.status}`);
  const length = response.headers.get('content-length');
  if (length !== null && Number(length) !== WEIGHT_BYTES) {
    await response.body?.cancel();
    throw new Error(`Invalid weight size: expected ${WEIGHT_BYTES} bytes`);
  }
  for (const entry of entries) entry.bytes = new Uint8Array(entry.count * 4);
  let received = 0;
  onProgress?.(0);
  function consume(chunk) {
    if (received + chunk.byteLength > WEIGHT_BYTES) throw new Error('Weight file is too large');
    for (const entry of entries) {
      const begin = Math.max(received, entry.offset);
      const end = Math.min(received + chunk.byteLength, entry.offset + entry.bytes.length);
      if (begin < end) entry.bytes.set(chunk.subarray(begin - received, end - received), begin - entry.offset);
    }
    received += chunk.byteLength;
    // Reserve 100% for a fully validated file.
    onProgress?.(Math.min(99, received / WEIGHT_BYTES * 100));
  }
  if (response.body?.getReader) {
    const reader = response.body.getReader();
    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        consume(value);
      }
    } catch (error) {
      await reader.cancel().catch(() => {});
      throw error;
    } finally { reader.releaseLock(); }
  } else {
    // Compatibility fallback temporarily holds the download, but returned tensors are copies.
    consume(new Uint8Array(await response.arrayBuffer()));
  }
  if (received !== WEIGHT_BYTES) throw new Error('Truncated weight file');
  const result = { layers: new Map() };
  const littleEndian = new Uint8Array(new Uint32Array([1]).buffer)[0] === 1;
  for (const entry of entries) {
    let values = new Float32Array(entry.bytes.buffer);
    if (!littleEndian) {
      const view = new DataView(entry.bytes.buffer);
      values = Float32Array.from({ length: entry.count }, (_, i) => view.getFloat32(i * 4, true));
    }
    if (!values.every(Number.isFinite)) throw new Error(`Non-finite weight: ${entry.name}`);
    if (entry.layer === undefined) result[entry.name] = values;
    else {
      if (!result.layers.has(entry.layer)) result.layers.set(entry.layer, {});
      result.layers.get(entry.layer)[entry.name] = values;
    }
  }
  onProgress?.(100);
  return result;
}
