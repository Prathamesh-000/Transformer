/** Reliable ordered activation ring plus sampler-to-host token shortcut.
 * Public activation/token interfaces follow IMPLEMENTATION_PLAN.md.
 * Controls share the ordered ring; large activations use bounded fragments.
 */
const RTC_CONFIG = { iceServers: [{ urls: 'stun:stun.l.google.com:19302' }] };
const MAX_BYTES = 512 * 384 * 4;
const CHUNK_BYTES = 16 * 1024 - 9;
export class DynamicP2PRingManager {
  constructor(signaling, config) {
    this._sig = signaling;
    Object.assign(this, config);
    this._pcs = new Map();
    this._ice = new Map();
    this._channels = new Set();
    this._isSealed = false;
    this._closed = false;
    this._signalQueue = Promise.resolve();
    this._signalHandler = msg => {
      this._signalQueue = this._signalQueue.then(() => this._handleSignal(msg)).catch(error => this._fail(error));
    };
    this._sig.on('signal', this._signalHandler);
  }
  async initiateConnections() {
    this._connectionPromise = new Promise((resolve, reject) => {
      this._connected = resolve; this._connectionRejected = reject;
      this._connectionTimeout = setTimeout(() => this._fail(new Error('WebRTC connection timeout (30s)')), 30000);
    });
    this._offer(this.downstreamPeerId, 'activation').catch(error => this._fail(error));
    if (this.isSampler) this._offer(this.hostPeerId, 'token_return').catch(error => this._fail(error));
    this._checkReady();
    return this._connectionPromise;
  }
  _pc(peer, type, side) {
    const key = `${side}:${type}:${peer}`;
    if (this._pcs.has(key)) return this._pcs.get(key);
    const pc = new RTCPeerConnection(RTC_CONFIG);
    this._pcs.set(key, pc);
    pc.onicecandidate = event => {
      if (event.candidate && !this._closed) this._sig.sendSignal(peer, 'ice', { channelType: type, side, candidate: event.candidate });
    };
    pc.onconnectionstatechange = () => {
      if (pc.connectionState === 'failed' || pc.connectionState === 'closed') this._fail(new Error(`Peer ${peer} connection ${pc.connectionState}`));
    };
    pc.ondatachannel = event => this._bindChannel(event.channel, type, side);
    return pc;
  }
  async _offer(peer, type) {
    const pc = this._pc(peer, type, 'out');
    this._bindChannel(pc.createDataChannel(type, { ordered: true }), type, 'out');
    await pc.setLocalDescription(await pc.createOffer());
    if (!this._closed) this._sig.sendSignal(peer, 'offer', { channelType: type, sdp: pc.localDescription });
  }
  async _handleSignal({ fromPeerId: peer, signalType, data }) {
    if (this._closed || !['offer', 'answer', 'ice'].includes(signalType)) return;
    const type = data?.channelType;
    if (!['activation', 'token_return'].includes(type)) return;
    const side = signalType === 'offer' ? 'in' : signalType === 'answer' ? 'out' : data.side === 'out' ? 'in' : 'out';
    const expected = type === 'activation' ? (side === 'in' ? this.upstreamPeerId : this.downstreamPeerId)
      : (side === 'in' && this.isHost ? this.samplerPeerId : side === 'out' && this.isSampler ? this.hostPeerId : null);
    if (peer !== expected) return;
    const key = `${side}:${type}:${peer}`;
    if (signalType === 'ice') {
      if (!data.candidate) return;
      const pc = this._pcs.get(key);
      if (pc?.remoteDescription) await pc.addIceCandidate(data.candidate);
      else { if (!this._ice.has(key)) this._ice.set(key, []); this._ice.get(key).push(data.candidate); }
      return;
    }
    const pc = this._pc(peer, type, side);
    await pc.setRemoteDescription(data.sdp);
    for (const candidate of this._ice.get(key) || []) await pc.addIceCandidate(candidate);
    this._ice.delete(key);
    if (signalType === 'offer') {
      await pc.setLocalDescription(await pc.createAnswer());
      this._sig.sendSignal(peer, 'answer', { channelType: type, sdp: pc.localDescription });
    }
  }
  _bindChannel(dc, type, side) {
    this._channels.add(dc);
    dc.binaryType = 'arraybuffer';
    if (type === 'token_return') this._tokenChannel = dc;
    else if (side === 'out') this._downstreamChannel = dc;
    else this._upstreamChannel = dc;
    dc.onopen = () => this._checkReady();
    dc.onclose = () => this._fail(new Error('A ring DataChannel closed; reconnect the mesh'));
    dc.onerror = () => this._fail(new Error('DataChannel transport error'));
    dc.onmessage = event => {
      try { this._receive(event.data, type); } catch (error) { this._fail(error); }
    };
    this._checkReady();
  }
  _checkReady() {
    const open = channel => channel?.readyState === 'open';
    if (!this._closed && open(this._downstreamChannel) && open(this._upstreamChannel)
        && (!(this.isHost || this.isSampler) || open(this._tokenChannel))) {
      this._isSealed = true;
      clearTimeout(this._connectionTimeout);
      this._connected?.(); this._connected = null;
      this._onRingSealedCallback?.();
    }
  }
  _receive(buffer, channelType) {
    if (!(buffer instanceof ArrayBuffer) || !buffer.byteLength) throw new Error('Invalid DataChannel packet');
    const bytes = new Uint8Array(buffer), type = bytes[0];
    if (channelType === 'token_return') {
      if (!this.isHost || type !== 2 || bytes.length !== 2) throw new Error('Invalid token packet');
      this._onTokenCallback?.(bytes[1]); return;
    }
    if (type === 3) {
      if (bytes.length > 4096) throw new Error('Control packet exceeds limit');
      this._onControlCallback?.(JSON.parse(new TextDecoder().decode(bytes.subarray(1)))); return;
    }
    if (type === 1) {
      if (this._partial) throw new Error('Interleaved activation packet');
      this._deliverActivation(buffer.slice(1)); return;
    }
    if (type !== 4 || bytes.length <= 9) throw new Error('Invalid activation fragment');
    const header = new DataView(buffer), total = header.getUint32(1, true), offset = header.getUint32(5, true);
    if (!total || total > MAX_BYTES || total % 1536) throw new Error('Invalid activation size');
    if (!this._partial && offset === 0) this._partial = { bytes: new Uint8Array(total), offset: 0 };
    const partial = this._partial;
    if (!partial || partial.bytes.length !== total || offset !== partial.offset || offset + bytes.length - 9 > total) throw new Error('Out-of-order activation fragment');
    partial.bytes.set(bytes.subarray(9), offset); partial.offset += bytes.length - 9;
    if (partial.offset === total) { this._partial = null; this._deliverActivation(partial.bytes.buffer); }
  }
  _deliverActivation(buffer) {
    if (!buffer.byteLength || buffer.byteLength > MAX_BYTES || buffer.byteLength % 1536) throw new Error('Invalid activation dimensions');
    this._onActivationCallback?.(new Float32Array(buffer));
  }
  _send(channel, packet) {
    if (this._closed || channel?.readyState !== 'open') throw new Error('DataChannel is not open');
    if (channel.bufferedAmount > 2 * MAX_BYTES) throw new Error('DataChannel backpressure limit exceeded');
    channel.send(packet.buffer);
  }
  sendActivation(tensor) {
    if (!(tensor instanceof Float32Array) || !tensor.length || tensor.length % 384 || tensor.byteLength > MAX_BYTES) throw new Error('Invalid activation dimensions');
    const bytes = new Uint8Array(tensor.buffer, tensor.byteOffset, tensor.byteLength);
    if (bytes.length <= CHUNK_BYTES) {
      const packet = new Uint8Array(1 + bytes.length); packet[0] = 1; packet.set(bytes, 1);
      this._send(this._downstreamChannel, packet); return;
    }
    for (let offset = 0; offset < bytes.length; offset += CHUNK_BYTES) {
      const chunk = bytes.subarray(offset, offset + CHUNK_BYTES), packet = new Uint8Array(9 + chunk.length);
      packet[0] = 4;
      const header = new DataView(packet.buffer); header.setUint32(1, bytes.length, true); header.setUint32(5, offset, true);
      packet.set(chunk, 9); this._send(this._downstreamChannel, packet);
    }
  }
  sendTokenToHost(token) {
    if (!this.isSampler || !Number.isInteger(token) || token < 0 || token > 255) throw new Error('Invalid sampled token');
    this._send(this._tokenChannel, Uint8Array.of(2, token));
  }
  sendControl(message) {
    const bytes = new TextEncoder().encode(JSON.stringify(message));
    if (bytes.length > 4095) throw new Error('Control packet exceeds limit');
    const packet = new Uint8Array(bytes.length + 1); packet[0] = 3; packet.set(bytes, 1);
    this._send(this._downstreamChannel, packet);
  }
  onActivation(cb) { this._onActivationCallback = cb; }
  onTokenFromSampler(cb) { this._onTokenCallback = cb; }
  onControl(cb) { this._onControlCallback = cb; }
  onError(cb) { this._onErrorCallback = cb; }
  onRingSealed(cb) { this._onRingSealedCallback = cb; if (this._isSealed) cb(); }
  get isRingSealed() { return this._isSealed && !this._closed; }
  _fail(error) {
    if (this._closed) return;
    this._connectionRejected?.(error);
    this.close();
    this._onErrorCallback?.(error);
  }
  close() {
    if (this._closed) return;
    this._closed = true; this._isSealed = false;
    clearTimeout(this._connectionTimeout);
    this._connectionRejected?.(new Error('Ring closed'));
    this._sig.off('signal', this._signalHandler);
    for (const dc of this._channels) dc.close();
    for (const pc of this._pcs.values()) pc.close();
    this._pcs.clear(); this._ice.clear(); this._channels.clear(); this._partial = null;
  }
}
