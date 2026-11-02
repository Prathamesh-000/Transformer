/**
 * signaling.js — Dynamic Multi-Peer WebSocket Signaling Client
 *
 * Prathamesh Patil (Member 1) Deliverable (Phase 2):
 * Manages the WebSocket connection to the signaling server for any phone role.
 * Works uniformly on mobile Safari and Android Chrome.
 *
 * Supported Flow:
 *  - Host calls: createRoom(roomId, deviceName) -> waits for peers -> lockRing()
 *  - Workers call: joinRoom(roomId, deviceName) -> receives ring_config
 *  - Both exchange targeted WebRTC signals: sendSignal(targetPeerId, signalType, data)
 */

class SignalingClient {
  /**
   * @param {string} [serverUrl] - WebSocket URL (e.g. 'ws://localhost:3000/signal').
   *                               Defaults to current origin's ws/wss endpoint.
   */
  constructor(serverUrl) {
    if (!serverUrl && typeof window !== 'undefined') {
      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      this.serverUrl = `${protocol}//${window.location.host}/signal`;
    } else {
      this.serverUrl = serverUrl || 'ws://localhost:3000/signal';
    }

    this._ws = null;
    this._myPeerId = null;
    this._myRoomId = null;
    this._isHost = false;

    // Event listeners: Map<eventType, Set<Function>>
    this._events = new Map();

    // Reconnection parameters
    this._reconnectAttempts = 0;
    this._maxReconnectAttempts = 5;
    this._reconnectBaseDelayMs = 1000;
    this._manualClose = false;
    this._pingInterval = null;
  }

  // ── Connection Lifecycle ───────────────────────────────────────────────────

  /**
   * Connect to the signaling WebSocket server.
   * @returns {Promise<void>} Resolves when socket is open.
   */
  connect() {
    return new Promise((resolve, reject) => {
      this._manualClose = false;
      this._createSocket(resolve, reject);
    });
  }

  _createSocket(initialResolve, initialReject) {
    let resolved = false;

    try {
      this._ws = new WebSocket(this.serverUrl);
    } catch (e) {
      if (initialReject) initialReject(e);
      return;
    }

    this._ws.onopen = () => {
      console.log(`[Signaling] Connected to ${this.serverUrl}`);
      this._reconnectAttempts = 0;
      this._startHeartbeat();

      if (initialResolve && !resolved) {
        resolved = true;
        initialResolve();
      }
      this._emit('open', { url: this.serverUrl });
    };

    this._ws.onmessage = (event) => {
      let msg;
      try {
        msg = JSON.parse(event.data);
      } catch (err) {
        console.error('[Signaling] Failed to parse message:', event.data);
        return;
      }

      this._handleMessage(msg);
    };

    this._ws.onerror = (err) => {
      console.warn('[Signaling] Socket error:', err);
      if (initialReject && !resolved) {
        resolved = true;
        initialReject(err);
      }
      this._emit('error', err);
    };

    this._ws.onclose = (event) => {
      this._stopHeartbeat();
      console.log(`[Signaling] Socket closed (code: ${event.code}, reason: ${event.reason})`);
      this._emit('close', event);

      if (!this._manualClose && this._reconnectAttempts < this._maxReconnectAttempts) {
        const delay = this._reconnectBaseDelayMs * Math.pow(1.5, this._reconnectAttempts);
        this._reconnectAttempts++;
        console.log(`[Signaling] Attempting reconnect in ${delay.toFixed(0)}ms (Attempt ${this._reconnectAttempts}/${this._maxReconnectAttempts})...`);
        setTimeout(() => this._createSocket(), delay);
      }
    };
  }

  _startHeartbeat() {
    this._stopHeartbeat();
    // Mobile browsers may idle sockets after 30s. Send ping every 20s.
    this._pingInterval = setInterval(() => {
      if (this.isConnected) {
        this.send({ type: 'ping', timestamp: Date.now() });
      }
    }, 20000);
  }

  _stopHeartbeat() {
    if (this._pingInterval) {
      clearInterval(this._pingInterval);
      this._pingInterval = null;
    }
  }

  // ── Public Room Operations ─────────────────────────────────────────────────

  /**
   * Host creates a room lobby.
   * @param {string} [requestedRoomId] - Optional room code (e.g. 'RING42')
   * @param {string} [deviceName] - Friendly device name (e.g. 'iPhone 15 Pro')
   * @returns {Promise<{room: string, peerId: string, totalNodes: number}>}
   */
  createRoom(requestedRoomId, deviceName) {
    return new Promise((resolve, reject) => {
      if (!this.isConnected) return reject(new Error('Signaling socket is not connected.'));

      const onRoomCreated = (msg) => {
        this._myPeerId = msg.peerId;
        this._myRoomId = msg.room;
        this._isHost = true;
        this.off('room_created', onRoomCreated);
        this.off('error', onError);
        resolve(msg);
      };

      const onError = (msg) => {
        this.off('room_created', onRoomCreated);
        this.off('error', onError);
        reject(new Error(msg.message || 'Failed to create room.'));
      };

      this.on('room_created', onRoomCreated);
      this.on('error', onError);

      this.send({
        type: 'create_room',
        room: requestedRoomId,
        deviceName: deviceName || this._getDeviceDefaultName('Host Phone')
      });
    });
  }

  /**
   * Worker scans Master QR and joins a room lobby.
   * @param {string} roomId - 6-character room code from QR
   * @param {string} [deviceName] - Friendly device name
   * @returns {Promise<{room: string, peerId: string, totalNodes: number, peers: Array}>}
   */
  joinRoom(roomId, deviceName) {
    return new Promise((resolve, reject) => {
      if (!this.isConnected) return reject(new Error('Signaling socket is not connected.'));

      const onRoomJoined = (msg) => {
        this._myPeerId = msg.peerId;
        this._myRoomId = msg.room;
        this._isHost = false;
        this.off('room_joined', onRoomJoined);
        this.off('error', onError);
        resolve(msg);
      };

      const onError = (msg) => {
        this.off('room_joined', onRoomJoined);
        this.off('error', onError);
        reject(new Error(msg.message || 'Failed to join room.'));
      };

      this.on('room_joined', onRoomJoined);
      this.on('error', onError);

      this.send({
        type: 'join_room',
        room: roomId,
        deviceName: deviceName || this._getDeviceDefaultName('Worker Phone')
      });
    });
  }

  /**
   * Host locks the ring topology with whatever N phones are currently connected.
   */
  lockRing() {
    if (!this.isConnected) throw new Error('Signaling socket is not connected.');
    if (!this._isHost) throw new Error('Only the Host can lock the ring topology.');

    this.send({ type: 'lock_ring' });
  }

  /**
   * Relay WebRTC signaling payload (SDP offer/answer or ICE candidate) to a peer.
   * @param {string} targetPeerId - Recipient peerId
   * @param {'offer'|'answer'|'ice'} signalType - Signal type
   * @param {object} data - SDP or candidate payload
   */
  sendSignal(targetPeerId, signalType, data) {
    if (!this.isConnected) {
      console.warn('[Signaling] Cannot send signal: socket not connected.');
      return;
    }

    this.send({
      type: 'signal',
      targetPeerId: targetPeerId,
      signalType: signalType,
      data: data
    });
  }

  // ── Message Routing ────────────────────────────────────────────────────────

  _handleMessage(msg) {
    // Specific message type event
    if (msg.type) {
      this._emit(msg.type, msg);
    }

    // Catch-all message listener
    this._emit('*', msg);
  }

  send(obj) {
    if (this._ws && this._ws.readyState === WebSocket.OPEN) {
      this._ws.send(JSON.stringify(obj));
    }
  }

  // ── Event Emitter ──────────────────────────────────────────────────────────

  on(eventType, callback) {
    if (!this._events.has(eventType)) {
      this._events.set(eventType, new Set());
    }
    this._events.get(eventType).add(callback);
    return this;
  }

  off(eventType, callback) {
    if (this._events.has(eventType)) {
      this._events.get(eventType).delete(callback);
    }
    return this;
  }

  _emit(eventType, data) {
    if (this._events.has(eventType)) {
      for (const cb of this._events.get(eventType)) {
        try {
          cb(data);
        } catch (err) {
          console.error(`[Signaling:EventError] in "${eventType}":`, err);
        }
      }
    }
  }

  // ── Utilities ──────────────────────────────────────────────────────────────

  _getDeviceDefaultName(fallback) {
    if (typeof navigator !== 'undefined') {
      const ua = navigator.userAgent;
      if (/iPhone/i.test(ua)) return 'Apple iPhone';
      if (/iPad/i.test(ua)) return 'Apple iPad';
      if (/Android/i.test(ua)) {
        const match = ua.match(/Android[^;]+;\s*([^)]+)\)/);
        return match && match[1] ? match[1].split('Build')[0].trim() : 'Android Phone';
      }
    }
    return fallback;
  }

  get isConnected() {
    return this._ws !== null && this._ws.readyState === WebSocket.OPEN;
  }

  get myPeerId() {
    return this._myPeerId;
  }

  get myRoomId() {
    return this._myRoomId;
  }

  get isHost() {
    return this._isHost;
  }

  close() {
    this._manualClose = true;
    this._stopHeartbeat();
    if (this._ws) {
      try {
        this._ws.close();
      } catch (_) {}
      this._ws = null;
    }
  }
}

export { SignalingClient };

