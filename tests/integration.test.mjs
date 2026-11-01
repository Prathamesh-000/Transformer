import test from 'node:test';
import assert from 'node:assert/strict';
import { DynamicP2PRingManager } from '../client/network/p2p-ring.js';
import { HostOrchestrator } from '../client/core/host-orchestrator.js';
import { PipelineSession } from '../client/core/pipeline-session.js';

const config = (i, n = 2) => ({ nodeIndex: i, totalNodes: n, isHost: i === 0, isSampler: i === n - 1, assignedLayers: i === 0 ? [0, 1, 2] : [3, 4, 5], downstreamPeerId: `p${(i + 1) % n}`, upstreamPeerId: `p${(i - 1 + n) % n}`, hostPeerId: 'p0', samplerPeerId: `p${n - 1}` });
const sig = { on() {}, off() {}, sendSignal() {} };

test('real transport preserves full-context prefill, subarray boundaries and decode framing', () => {
  const sender = new DynamicP2PRingManager(sig, config(0));
  const receiver = new DynamicP2PRingManager(sig, config(1));
  const packets = [];
  sender._downstreamChannel = { readyState: 'open', bufferedAmount: 0, send(buffer) { packets.push(buffer); receiver._receive(buffer, 'activation'); } };
  let received;
  receiver.onActivation(x => { received = x; });
  for (const rows of [1, 16, 512]) {
    const backing = Float32Array.from({ length: rows * 384 + 10 }, (_, i) => Math.sin(i));
    const input = backing.subarray(5, 5 + rows * 384);
    packets.length = 0; sender.sendActivation(input);
    assert.deepEqual(received, input);
    assert.ok(packets.every(p => p.byteLength <= 16384));
    if (rows === 1) assert.equal(packets[0].byteLength, 1537);
  }
  assert.throws(() => receiver._receive(Uint8Array.of(1, 0).buffer, 'activation'));
  assert.throws(() => sender.sendActivation(new Float32Array(1)));
  assert.throws(() => receiver._receive(Uint8Array.of(2).buffer, 'token_return'));
});

test('ICE is queued and routed by exact peer, direction and channel', async () => {
  const ring = new DynamicP2PRingManager(sig, config(0));
  const candidate = { candidate: 'candidate:test' };
  await ring._handleSignal({ fromPeerId: 'p1', signalType: 'ice', data: { channelType: 'token_return', side: 'out', candidate } });
  assert.deepEqual(ring._ice.get('in:token_return:p1'), [candidate]);
  const seen = [];
  ring._pcs.set('out:activation:p1', { remoteDescription: {}, addIceCandidate: c => seen.push(c) });
  await ring._handleSignal({ fromPeerId: 'p1', signalType: 'ice', data: { channelType: 'activation', side: 'in', candidate } });
  assert.equal(seen.length, 1); assert.equal(ring._ice.size, 1);
  await ring._handleSignal({ fromPeerId: 'p10', signalType: 'ice', data: { channelType: 'activation', side: 'in', candidate } });
  assert.equal(seen.length, 1);
});

function engine() {
  return { reset() { this.position = 0; }, async processStep(input, position) { assert.equal(position, this.position); this.position += input.length; return { type: 'activation', activation: new Float32Array(input.length * 384) }; } };
}
function tokenRing(tokens, delay = 0) {
  return { isRingSealed: true, onTokenFromSampler(cb) { this.cb = cb; }, sendActivation() { const value = tokens.shift() ?? 65; setTimeout(() => this.cb(value), delay); } };
}

test('orchestrator counts NUL as a byte, validates context without truncating, and flushes UTF-8 separately', async () => {
  const ring = tokenRing([0, 0xf0, 0x9f]); const seen = [], text = [];
  let completed = 0;
  const host = new HostOrchestrator(ring, engine(), { onToken: (t, s) => { seen.push(t); text.push(s); }, onText: s => text.push(s), onComplete: () => completed++ });
  await host.startGeneration('x', 3);
  assert.deepEqual(seen, [0, 0xf0, 0x9f]); assert.equal(text.join(''), '\0�'); assert.equal(completed, 1);
  await assert.rejects(host.startGeneration('x'.repeat(512), 2), /512/);
  await assert.rejects(host.startGeneration('', 1), /nonempty/);
  await assert.rejects(host.startGeneration('x', 0), /maxNewTokens/);
  await assert.rejects(host.startGeneration('x', 1, NaN), /temperature/);
  await host.startGeneration('x'.repeat(512), 1);
  assert.equal(host.context.length, 513);
});

test('stop drains in-flight token before restart; errors reject rather than being swallowed', async () => {
  const ring = tokenRing([65, 66], 15), host = new HostOrchestrator(ring, engine());
  const first = host.startGeneration('x', 20);
  await new Promise(resolve => setTimeout(resolve, 2)); host.stop();
  await assert.rejects(host.startGeneration('y', 1), /in progress/);
  assert.equal((await first).tokensGenerated, 1);
  await host.startGeneration('z', 1); assert.equal(host.context.at(-1), 66);
  ring.sendActivation = () => { throw new Error('transport failed'); };
  await assert.rejects(host.startGeneration('z', 1), /transport failed/);
  assert.equal(host.isRunning, false); assert.equal(host._pending, null);
});

test('P2P session waits for all workers, resets remote caches and propagates temperature/mirrors', async () => {
  const rings = [0, 1].map(i => ({ ...config(i), onControl(cb) { this.control = cb; }, onActivation(cb) { this.activation = cb; }, onError(cb) { this.error = cb; }, async initiateConnections() { this.isRingSealed = true; }, sendControl(message) { queueMicrotask(() => rings[1 - i].control(message)); }, close() {} }));
  let release, resets = 0, mirrored = [], finished = 0;
  const delayed = new Promise(resolve => { release = resolve; });
  const sessions = rings.map((ring, i) => new PipelineSession(ring, config(i), {
    createWorker: async () => { if (i === 1) await delayed; return { reset() { resets++; } }; },
    ui: { setReady() {}, setLoadingProgress() {}, reset() {}, appendToken(t) { mirrored.push(t); }, finishText() { finished++; }, setStatus() {} },
  }));
  let ready = false;
  const barrier = sessions[0].barrier('ready').then(() => { ready = true; });
  await new Promise(resolve => setTimeout(resolve, 5)); assert.equal(ready, false);
  release(); await barrier; await sessions[0].prepare(0.7);
  assert.equal(sessions[1].temperature, 0.7); assert.equal(resets, 2);
  sessions[0].mirror(240); await sessions[0].finish();
  assert.deepEqual(mirrored, [240]); assert.equal(finished, 1);
  await sessions[0].prepare(0); assert.equal(resets, 4);
});
