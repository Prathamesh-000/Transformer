/**
 * p2p-ring.js — Circular WebRTC DataChannel Manager for Dynamic N-Phone Mesh
 *
 * Member 1 Deliverable (Phase 2):
 * Manages the WebRTC peer connections and binary data channels to form a closed
 * directional pipeline ring across any N phones (N = 2 to 6):
 *
 * Topology:
 *   - Activation channel: Node i -> Node (i + 1) % N (Label: 'activation')
 *   - Token shortcut return: Node N - 1 -> Node 0 (Label: 'token_return')
 *
 * Message Framing over DataChannels:
 *   - 0x01: Activation Tensor (Float32Array buffer)
 *   - 0x02: Sampled Token (Single uint8 byte [0..255])
 */

// Public STUN servers (STUN only — zero centralized TURN relays)
const RTC_CONFIG = {
  iceServers: [
    { urls: 'stun:stun.l.google.com:19302' },
    { urls: 'stun:stun1.l.google.com:19302' }
  ]
};

const CHANNEL_CONFIG = {
  ordered: true,        // Token stream must be strictly in-order
  maxRetransmits: 30    // TCP-like reliability on UDP
};

// Protocol framing headers
const MSG_ACTIVATION = 0x01; // Activation tensor
const MSG_TOKEN      = 0x02; // Sampled byte token

class DynamicP2PRingManager {
  /**
   * @param {SignalingClient} signaling - Active signaling client
   * @param {object} config - Dynamic ring configuration received from server
   */
  constructor(signaling, config) {
    this._sig    = signaling;
    this._config = config;

    this.nodeIndex        = config.nodeIndex;
    this.totalNodes       = config.totalNodes;
    this.assignedLayers   = config.assignedLayers || [];
    this.isHost           = config.isHost;
    this.isSampler        = config.isSampler;

    this.downstreamPeerId = config.downstreamPeerId;
    this.upstreamPeerId   = config.upstreamPeerId;
    this.hostPeerId       = config.hostPeerId;
    this.samplerPeerId    = config.samplerPeerId;

    // WebRTC Peer Connections: Map<peerId_purpose, RTCPeerConnection>
    this._pcs = new Map();

    // DataChannels
    this._downstreamChannel = null; // Output: sends activation to (nodeIndex + 1) % N
    this._upstreamChannel   = null; // Input: receives activation from (nodeIndex - 1 + N) % N
    this._tokenChannel      = null; // Token feedback: Sampler sends to Host, Host receives

    // Callbacks
    this._onActivationCallback = null;
    this._onTokenCallback      = null;
    this._onRingSealedCallback = null;

    // State
    this._isSealed = false;
    this._signalHandler = this._handleSignal.bind(this);
    this._sig.on('signal', this._signalHandler);
  }

  // ── Connection Initialization ──────────────────────────────────────────────

  /**
   * Initiates peer connection negotiations according to the node's position in the ring.
   * @returns {Promise<void>} Resolves when all required DataChannels for this node are OPEN.
   */
  async initiateConnections() {
    console.log(`[P2P:Init] Node ${this.nodeIndex} (${this.isHost ? 'Host' : this.isSampler ? 'Sampler' : 'Worker'}) initializing WebRTC links...`);

    const promises = [];

    // ────────────────────────────────────────────────────────────────────────
    // 1. Downstream Activation Link (Node i -> Node i+1)
    // ────────────────────────────────────────────────────────────────────────
    // Node i is the Offerer to Node i+1 for the 'activation' channel
    promises.push(this._setupDownstreamActivation());

    // ────────────────────────────────────────────────────────────────────────
    // 2. Token Return Shortcut Link (Node N-1 -> Node 0)
    // ────────────────────────────────────────────────────────────────────────
    if (this.isSampler) {
      // The final Sampler node is the Offerer to the Host for the 'token_return' channel
      promises.push(this._setupSamplerTokenReturn());
    } else if (this.isHost) {
      // Host waits for Sampler's 'token_return' channel to open
      promises.push(this._waitForTokenChannelOpen());
    }

    // Intermediate nodes wait for upstream channel to open
    if (!this.isHost) {
      promises.push(this._waitForUpstreamActivationChannelOpen());
    }

    // Wait until all required channels for this node are open
    await Promise.all(promises);

    this._isSealed = true;
    console.log(`[P2P:Sealed] 🟢 Node ${this.nodeIndex} ring connections sealed and ready!`);

    if (typeof this._onRingSealedCallback === 'function') {
      this._onRingSealedCallback();
    }
  }

  // ── Downstream Activation Link Setup ───────────────────────────────────────

  _setupDownstreamActivation() {
    return new Promise((resolve, reject) => {
      const pcKey = `downstream_${this.downstreamPeerId}`;
      const pc = new RTCPeerConnection(RTC_CONFIG);
      this._pcs.set(pcKey, pc);

      this._bindIceHandling(pc, this.downstreamPeerId, 'activation_downstream');

      // Create 'activation' DataChannel
      const dc = pc.createDataChannel('activation', CHANNEL_CONFIG);
      dc.binaryType = 'arraybuffer';
      this._downstreamChannel = dc;

      dc.onopen = () => {
        console.log(`[P2P:DataChannel] ✓ Downstream activation channel OPEN (Node ${this.nodeIndex} -> Node ${(this.nodeIndex + 1) % this.totalNodes})`);
        resolve();
      };

      dc.onerror = (e) => console.error('[P2P:DataChannel] Downstream channel error:', e);
      dc.onclose = () => console.log('[P2P:DataChannel] Downstream channel closed.');

      // Create and send SDP Offer
      pc.createOffer()
        .then(offer => pc.setLocalDescription(offer))
        .then(() => {
          this._sig.sendSignal(this.downstreamPeerId, 'offer', {
            channelType: 'activation',
            sdp: pc.localDescription
          });
        })
        .catch(reject);
    });
  }

  _waitForUpstreamActivationChannelOpen() {
    return new Promise((resolve) => {
      const check = setInterval(() => {
        if (this._upstreamChannel && this._upstreamChannel.readyState === 'open') {
          clearInterval(check);
          resolve();
        }
      }, 50);
    });
  }

  // ── Token Return Shortcut Link Setup ───────────────────────────────────────

  _setupSamplerTokenReturn() {
    return new Promise((resolve, reject) => {
      const pcKey = `token_return_${this.hostPeerId}`;
      const pc = new RTCPeerConnection(RTC_CONFIG);
      this._pcs.set(pcKey, pc);

      this._bindIceHandling(pc, this.hostPeerId, 'token_return');

      // Sampler creates 'token_return' DataChannel to Host
      const dc = pc.createDataChannel('token_return', CHANNEL_CONFIG);
      dc.binaryType = 'arraybuffer';
      this._tokenChannel = dc;

      dc.onopen = () => {
        console.log(`[P2P:DataChannel] ✓ Token shortcut channel OPEN (Sampler Node ${this.nodeIndex} -> Host Node 0)`);
        resolve();
      };

      dc.onerror = (e) => console.error('[P2P:DataChannel] Token channel error:', e);

      pc.createOffer()
        .then(offer => pc.setLocalDescription(offer))
        .then(() => {
          this._sig.sendSignal(this.hostPeerId, 'offer', {
            channelType: 'token_return',
            sdp: pc.localDescription
          });
        })
        .catch(reject);
    });
  }

  _waitForTokenChannelOpen() {
    return new Promise((resolve) => {
      const check = setInterval(() => {
        if (this._tokenChannel && this._tokenChannel.readyState === 'open') {
          clearInterval(check);
          resolve();
        }
      }, 50);
    });
  }

  // ── ICE & Signal Relaying Handshake ────────────────────────────────────────

  _bindIceHandling(pc, targetPeerId, contextTag) {
    pc.onicecandidate = (event) => {
      if (event.candidate) {
        this._sig.sendSignal(targetPeerId, 'ice', {
          context: contextTag,
          candidate: event.candidate
        });
      }
    };

    pc.onconnectionstatechange = () => {
      console.log(`[P2P:State] [${contextTag}] Connection state: ${pc.connectionState}`);
    };
  }

  async _handleSignal(msg) {
    const { fromPeerId, signalType, data } = msg;

    // ────────────────────────────────────────────────────────────────────────
    // Handle Incoming SDP Offer (Answering role)
    // ────────────────────────────────────────────────────────────────────────
    if (signalType === 'offer') {
      const { channelType, sdp } = data;
      const pcKey = `inbound_${channelType}_${fromPeerId}`;
      const pc = new RTCPeerConnection(RTC_CONFIG);
      this._pcs.set(pcKey, pc);

      this._bindIceHandling(pc, fromPeerId, `${channelType}_answer`);

      pc.ondatachannel = (event) => {
        const dc = event.channel;
        dc.binaryType = 'arraybuffer';

        if (dc.label === 'activation') {
          console.log(`[P2P:DataChannel] ✓ Upstream activation channel OPEN (From Node ${(this.nodeIndex - 1 + this.totalNodes) % this.totalNodes})`);
          this._upstreamChannel = dc;
          this._bindChannelReceiver(dc);
        } else if (dc.label === 'token_return') {
          console.log(`[P2P:DataChannel] ✓ Token shortcut channel OPEN on Host (From Sampler Node ${this.totalNodes - 1})`);
          this._tokenChannel = dc;
          this._bindChannelReceiver(dc);
        }
      };

      await pc.setRemoteDescription(new RTCSessionDescription(sdp));
      const answer = await pc.createAnswer();
      await pc.setLocalDescription(answer);

      this._sig.sendSignal(fromPeerId, 'answer', {
        channelType: channelType,
        sdp: pc.localDescription
      });
    }

    // ────────────────────────────────────────────────────────────────────────
    // Handle Incoming SDP Answer
    // ────────────────────────────────────────────────────────────────────────
    else if (signalType === 'answer') {
      const { channelType, sdp } = data;
      // Match the peer connection that created this offer
      let pc = null;
      if (channelType === 'activation') {
        pc = this._pcs.get(`downstream_${fromPeerId}`);
      } else if (channelType === 'token_return') {
        pc = this._pcs.get(`token_return_${fromPeerId}`);
      }

      if (pc) {
        await pc.setRemoteDescription(new RTCSessionDescription(sdp));
      }
    }

    // ────────────────────────────────────────────────────────────────────────
    // Handle Incoming ICE Candidate
    // ────────────────────────────────────────────────────────────────────────
    else if (signalType === 'ice') {
      const { candidate } = data;
      if (!candidate) return;

      // Add to any active peer connection matching this peer
      for (const [key, pc] of this._pcs.entries()) {
        if (key.includes(fromPeerId)) {
          try {
            await pc.addIceCandidate(new RTCIceCandidate(candidate));
          } catch (e) {
            console.warn(`[P2P:ICE] Could not add candidate to ${key}:`, e.message);
          }
        }
      }
    }
  }

  // ── Binary Framing & Channel Messaging ─────────────────────────────────────

  /**
   * Bind message parser to incoming DataChannel.
   * Dispatches 0x01 to activation callback and 0x02 to token callback.
   */
  _bindChannelReceiver(dc) {
    dc.onmessage = (event) => {
      const buffer = event.data;
      if (!(buffer instanceof ArrayBuffer)) {
        console.warn('[P2P:Msg] Received non-ArrayBuffer payload');
        return;
      }

      const view = new Uint8Array(buffer);
      const msgType = view[0];

      // 0x01: Activation Tensor
      if (msgType === MSG_ACTIVATION) {
        // Zero-copy view over the Float32 bytes (skip 1-byte header, offset = 1)
        // Must align to 4 bytes: buffer was constructed with byteOffset = 4 or slice
        const floatBytes = buffer.slice(1);
        const tensor = new Float32Array(floatBytes);

        if (typeof this._onActivationCallback === 'function') {
          this._onActivationCallback(tensor);
        }
      }
      // 0x02: Sampled Byte Token
      else if (msgType === MSG_TOKEN) {
        const token = view[1];
        if (typeof this._onTokenCallback === 'function') {
          this._onTokenCallback(token);
        }
      }
    };
  }

  // ── Public Data Transport Methods ──────────────────────────────────────────

  /**
   * Transmit intermediate activation tensor to downstream peer.
   * Uses 0x01 framing: [0x01][Float32Array bytes].
   * @param {Float32Array} tensor - Activation vector [1, 384] or matrix [P, 384]
   */
  sendActivation(tensor) {
    if (!this._downstreamChannel || this._downstreamChannel.readyState !== 'open') {
      throw new Error(`[P2P] Downstream channel is not open (Node ${this.nodeIndex})`);
    }

    const payload = new Uint8Array(1 + tensor.byteLength);
    payload[0] = MSG_ACTIVATION;
    payload.set(new Uint8Array(tensor.buffer, tensor.byteOffset, tensor.byteLength), 1);

    this._downstreamChannel.send(payload.buffer);
  }

  /**
   * Register listener for incoming activation from upstream peer.
   * @param {function(Float32Array): void} callback
   */
  onActivation(callback) {
    this._onActivationCallback = callback;
  }

  /**
   * Sampler Node transmits the sampled byte token directly to Host.
   * Uses 0x02 framing: [0x02][uint8 token].
   * @param {number} tokenByte - Byte value [0..255]
   */
  sendTokenToHost(tokenByte) {
    if (!this._tokenChannel || this._tokenChannel.readyState !== 'open') {
      throw new Error('[P2P] Token shortcut channel is not open');
    }

    const payload = new Uint8Array([MSG_TOKEN, tokenByte & 0xFF]);
    this._tokenChannel.send(payload.buffer);
  }

  /**
   * Host Node registers listener for incoming tokens from Sampler.
   * @param {function(number): void} callback
   */
  onTokenFromSampler(callback) {
    this._onTokenCallback = callback;
  }

  onRingSealed(callback) {
    this._onRingSealedCallback = callback;
    if (this._isSealed) callback();
  }

  get isRingSealed() {
    return this._isSealed;
  }

  // ── Teardown ───────────────────────────────────────────────────────────────

  close() {
    this._sig.off('signal', this._signalHandler);
    if (this._downstreamChannel) {
      try { this._downstreamChannel.close(); } catch (_) {}
      this._downstreamChannel = null;
    }
    if (this._upstreamChannel) {
      try { this._upstreamChannel.close(); } catch (_) {}
      this._upstreamChannel = null;
    }
    if (this._tokenChannel) {
      try { this._tokenChannel.close(); } catch (_) {}
      this._tokenChannel = null;
    }
    for (const pc of this._pcs.values()) {
      try { pc.close(); } catch (_) {}
    }
    this._pcs.clear();
    this._isSealed = false;
  }
}

export { DynamicP2PRingManager };

