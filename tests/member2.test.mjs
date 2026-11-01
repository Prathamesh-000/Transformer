import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { loadDynamicNodeWeights, WEIGHT_BYTES } from '../client/core/weights.js';
import { TransformerDynamicBlock } from '../client/core/transformer.js';
import { UniversalNodeWorker } from '../client/core/node-worker.js';
import { WorkerUI } from '../client/ui/worker-ui.js';

const binary = new Uint8Array(await readFile(new URL('../weights.bin', import.meta.url)));
const oracle = JSON.parse(await readFile(new URL('./reference.json', import.meta.url), 'utf8'));
const originalFetch = globalThis.fetch;
function serve(bytes = binary, { status = 200, length = bytes.length } = {}) {
  globalThis.fetch = async url => {
    assert.equal(url, '/weights.bin');
    let cursor = 0;
    return new Response(new ReadableStream({
      pull(controller) {
        if (cursor === bytes.length) return controller.close();
        // Deliberately not float-aligned to exercise streaming tensor boundaries.
        const end = Math.min(cursor + 65539, bytes.length);
        controller.enqueue(bytes.subarray(cursor, end)); cursor = end;
      },
    }), { status, headers: length === null ? {} : { 'content-length': String(length) } });
  };
}
serve();
function config(n, i) {
  const base = Math.floor(6 / n), rem = 6 % n;
  const first = i * base + Math.min(i, rem);
  return { totalNodes: n, nodeIndex: i, assignedLayers: Array.from({ length: base + (i < rem ? 1 : 0) }, (_, j) => first + j), isHost: i === 0, isSampler: i === n - 1, downstreamPeerId: `peer${(i + 1) % n}`, hostPeerId: 'peer0' };
}
function close(actual, expected, tolerance = 3e-5) {
  assert.equal(actual.length, expected.length);
  let max = 0;
  for (let i = 0; i < actual.length; i++) max = Math.max(max, Math.abs(actual[i] - expected[i]));
  assert.ok(max < tolerance, `Max numerical error ${max} exceeds ${tolerance}`);
  return max;
}
const all = await loadDynamicNodeWeights([0, 1, 2, 3, 4, 5], true, true);

await test('binary and selective streaming loader match exact layout and ownership', async () => {
  assert.equal(binary.length, WEIGHT_BYTES);
  for (let n = 2; n <= 6; n++) {
    for (let i = 0; i < n; i++) {
      const c = config(n, i), progress = [];
      const w = await loadDynamicNodeWeights(c.assignedLayers, c.isHost, c.isSampler, p => progress.push(p));
      assert.deepEqual([...w.layers.keys()], c.assignedLayers);
      assert.equal(!!w.tokenEmb, c.isHost); assert.equal(!!w.lmHeadW, c.isSampler);
      assert.equal(progress[0], 0); assert.equal(progress.at(-1), 100);
      assert.ok(progress.every((p, j) => j === 0 || p >= progress[j - 1]));
      for (const layer of c.assignedLayers) for (const [name, value] of Object.entries(w.layers.get(layer))) {
        assert.deepEqual(value, all.layers.get(layer)[name]);
        assert.equal(value.buffer.byteLength, value.byteLength);
      }
    }
  }
});

await test('all 2–6 stage pipelines match independent PyTorch prefill and decode', async t => {
  let baseline;
  for (let n = 2; n <= 6; n++) {
    const workers = Array.from({ length: n }, (_, i) => new UniversalNodeWorker(all, config(n, i)));
    let input = Uint8Array.from(oracle.prompt), position = 0;
    const outputs = [];
    let maxError = 0;
    for (const expected of oracle.steps) {
      const rows = input.length;
      let result;
      for (const worker of workers) {
        result = await worker.processStep(input, position);
        if (result.type === 'activation') {
          assert.equal(result.activation.length, rows * 384);
          if (rows === 1) assert.equal(result.activation.byteLength, 1536);
          input = result.activation;
        }
      }
      assert.equal(result.type, 'token'); assert.equal(result.token, expected.token);
      maxError = Math.max(maxError, close(result.logits, expected.logits));
      outputs.push(Array.from(result.logits));
      position += rows;
      assert.ok(workers.every(w => w.sequenceLength === position));
      input = Uint8Array.of(result.token);
    }
    if (!baseline) baseline = outputs;
    else assert.deepEqual(outputs, baseline); // Partition boundaries must be exactly neutral.
    workers.forEach(w => w.reset());
    assert.ok(workers.every(w => w.sequenceLength === 0));
    let inputAgain = Uint8Array.from(oracle.prompt), result;
    for (const worker of workers) { result = await worker.processStep(inputAgain, 0); inputAgain = result.activation; }
    assert.deepEqual(Array.from(result.logits), outputs[0]);
    t.diagnostic(`N=${n}: exact cross-partition agreement; max PyTorch error ${maxError}`);
  }
});

await test('causal prefill equals incremental and chunked execution', () => {
  function run(chunks) {
    const block = new TransformerDynamicBlock(all, [0, 1, 2, 3, 4, 5]);
    let position = 0, result;
    const activations = [];
    for (const chunk of chunks) {
      let x = block.embed(chunk, position);
      for (let layer = 0; layer < 6; layer++) x = block.forwardLayer(x, layer, position);
      activations.push(...x);
      result = block.sampleNextToken(x); position += chunk.length;
    }
    return { result, activations };
  }
  const full = run([oracle.prompt]);
  const single = run(oracle.prompt.map(v => [v]));
  const chunked = run([oracle.prompt.slice(0, 2), oracle.prompt.slice(2)]);
  close(full.activations, single.activations, 1e-6);
  close(full.activations, chunked.activations, 1e-6);
});

await test('shape, role, cache order, context and temperature guards', async () => {
  const block = new TransformerDynamicBlock(all, [0]);
  assert.throws(() => block.embed([], 0));
  assert.throws(() => block.embed([256], 0));
  assert.equal(block.embed([255], 511).length, 384);
  assert.throws(() => block.embed([1, 2], 511));
  assert.throws(() => block.forwardLayer(new Float32Array(384), 1, 0));
  assert.throws(() => block.forwardLayer(new Float32Array(384), 0, 1));
  assert.throws(() => block.forwardLayer(new Float32Array(383), 0, 0));
  const x = block.forwardLayer(block.embed([0], 0), 0, 0);
  assert.throws(() => block.forwardLayer(x, 0, 0));
  assert.throws(() => block.sampleNextToken(x, -1));
  assert.ok(block.sampleNextToken(x, 0.8).token < 256);
  assert.throws(() => new UniversalNodeWorker(all, { ...config(3, 1), isHost: true }));
  const worker = new UniversalNodeWorker(all, config(6, 1));
  await assert.rejects(worker.processStep(new Float32Array(10), 0));
  await assert.rejects(worker.processStep(new Float32Array(384), 0), /Reset required/);
  worker.reset();
  assert.equal((await worker.processStep(new Float32Array(384), 0)).type, 'activation');
});

await test('loader rejects HTTP errors, truncation, oversize, nonfinite data and bad assignments', async () => {
  try {
    serve(binary, { status: 404 }); await assert.rejects(loadDynamicNodeWeights([0], true, false), /HTTP 404/);
    serve(binary.subarray(0, 10)); await assert.rejects(loadDynamicNodeWeights([0], true, false), /size/);
    serve(binary.subarray(0, 10), { length: null }); await assert.rejects(loadDynamicNodeWeights([0], true, false), /Truncated/);
    const corrupt = binary.slice(); new DataView(corrupt.buffer).setFloat32(0, NaN, true);
    serve(corrupt); await assert.rejects(loadDynamicNodeWeights([0], true, false), /Non-finite/);
    serve(new Uint8Array(WEIGHT_BYTES + 1), { length: null }); await assert.rejects(loadDynamicNodeWeights([0], true, false), /too large/);
    await assert.rejects(loadDynamicNodeWeights([2, 1], false, false), /assignedLayers/);
  } finally { serve(); }
});

await test('worker UI streams UTF-8 bytes safely and resets lifecycle state', () => {
  // Minimal DOM double verifies lifecycle/text behavior, not browser layout.
  const document = { createElement: () => ({ ownerDocument: document, style: {}, children: [], setAttribute() {}, appendChild(child) { this.children.push(child); return child; }, remove() { this.removed = true; } }) };
  const root = document.createElement('div');
  const ui = new WorkerUI(root, config(4, 3));
  assert.match(ui.badge.textContent, /Layer 5 \+ LM Head/);
  ui.setLoadingProgress(55); assert.equal(ui.progress.value, 55);
  ui.setReady(); ui.beginStep(); ui.endStep({ type: 'token', token: 65 });
  for (const byte of new TextEncoder().encode('🌍<script>')) ui.appendToken(byte);
  ui.finishText(); assert.equal(ui.output.textContent, '🌍<script>');
  assert.equal(ui.processed, 1);
  ui.setError(new Error('lost peer')); assert.match(ui.status.textContent, /lost peer/);
  ui.reset(); assert.equal(ui.output.textContent, ''); assert.equal(ui.processed, 0);
  ui.destroy(); assert.equal(ui.panel.removed, true);
});
globalThis.fetch = originalFetch;
