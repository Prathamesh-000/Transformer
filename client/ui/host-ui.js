/**
 * host-ui.js — Host Dashboard, Dynamic Ring Visualizer & Master QR Display
 *
 * Member 1 Deliverable (Phase 4):
 * Manages the Phone 1 (Host) interface:
 *  1. Displays the Single Master QR Code for quick phone pairing
 *  2. Renders an interactive Canvas Ring Visualizer showing connected phones (N = 2 to 6)
 *  3. Controls: "Lock Mesh & Start", Prompt textarea, Temperature, Max tokens, Generate, Stop
 *  4. Real-time streaming terminal for generated characters
 *  5. Telemetry HUD: Tokens/sec (TPS), Ring Cycle RTT, Host Compute Latency
 */

class HostUI {
  constructor() {
    this._container = null;
    this._canvas = null;
    this._ctx = null;

    // Callbacks
    this._onLockRingCb = null;
    this._onGenerateCb = null;
    this._onStopCb     = null;

    // State
    this._peers = [];
    this._totalNodes = 1;
    this._isLocked = false;
    this._isGenerating = false;
    this._animFrameId = null;
    this._pulseNodeIndex = -1;
  }

  /**
   * Mount the Host UI into the given container element.
   * @param {HTMLElement} container
   */
  init(container) {
    this._container = container;
    this._renderDOM();
    this._bindEvents();
    this._initCanvas();
  }

  // ── DOM Construction ───────────────────────────────────────────────────────

  _renderDOM() {
    this._container.innerHTML = `
      <div class="host-app">
        <!-- Header -->
        <header class="app-header">
          <div class="header-title">
            <span class="badge-role">📱 Phone 1 — Host</span>
            <h2>P2P Mesh Transformer</h2>
          </div>
          <div id="connection-status-pill" class="status-pill status-connecting">
            <span class="status-indicator-dot"></span>
            <span id="connection-status-text">Creating Mesh...</span>
          </div>
        </header>

        <!-- Section 1: Dynamic Lobby & Master QR -->
        <section id="lobby-section" class="mesh-card">
          <div class="card-header">
            <h3>Pair Mesh Phones</h3>
            <span class="badge-accent" id="node-count-badge">1 / 6 Phones</span>
          </div>
          <p class="section-desc">
            Point other phones' camera apps at this QR code to join the mesh ring ($N = 2$ to $6$).
          </p>

          <div class="qr-container">
            <div id="qr-display" class="qr-box">
              <div class="qr-placeholder-spinner"></div>
            </div>
            <div class="room-details">
              <span class="room-label">ROOM CODE</span>
              <span id="room-code-display" class="room-code">------</span>
              <button id="copy-link-btn" class="btn-subtle" title="Copy join link">📋 Copy Link</button>
            </div>
          </div>

          <!-- Dynamic Visual Ring Canvas -->
          <div class="ring-canvas-wrapper">
            <canvas id="mesh-ring-canvas" width="320" height="260"></canvas>
          </div>

          <!-- Lobby Action -->
          <div class="lobby-actions">
            <button id="lock-ring-btn" class="btn btn-primary" disabled>
              🔒 Lock Mesh & Form Ring (Need ≥ 2 Phones)
            </button>
          </div>
        </section>

        <!-- Section 2: Inference & Prompt Control (Revealed after Ring is locked) -->
        <section id="inference-section" class="mesh-card hidden">
          <div class="card-header">
            <h3>Distributed Generation</h3>
            <span class="badge-ring-status" id="ring-status-badge">Ring Sealed ✓</span>
          </div>

          <!-- Prompt Form -->
          <div class="prompt-group">
            <label for="prompt-input" class="field-label">Input Prompt</label>
            <textarea
              id="prompt-input"
              class="prompt-textarea"
              rows="3"
              placeholder="Type your prompt here... (e.g. 'The future of mobile computing is')"
            >The future of mobile computing is</textarea>
          </div>

          <!-- Generation Parameters -->
          <div class="params-row">
            <div class="param-item">
              <div class="param-header">
                <label for="temp-slider">Temperature</label>
                <span id="temp-val" class="param-val">0.0</span>
              </div>
              <input type="range" id="temp-slider" min="0.0" max="1.0" step="0.1" value="0.0" />
            </div>

            <div class="param-item">
              <div class="param-header">
                <label for="max-tokens-input">Max Tokens</label>
                <span id="max-tokens-val" class="param-val">100</span>
              </div>
              <input type="number" id="max-tokens-input" min="10" max="400" value="100" />
            </div>
          </div>

          <!-- Action Buttons -->
          <div class="controls-row">
            <button id="generate-btn" class="btn btn-primary">
              ⚡ Generate Across Mesh
            </button>
            <button id="stop-btn" class="btn btn-danger hidden">
              ⏹ Stop
            </button>
          </div>

          <!-- Real-time Telemetry HUD -->
          <div class="telemetry-grid">
            <div class="telemetry-card">
              <span class="telemetry-label">Throughput</span>
              <span id="stat-tps" class="telemetry-value">-- <small>TPS</small></span>
            </div>
            <div class="telemetry-card">
              <span class="telemetry-label">Ring RTT</span>
              <span id="stat-rtt" class="telemetry-value">-- <small>ms</small></span>
            </div>
            <div class="telemetry-card">
              <span class="telemetry-label">Host Compute</span>
              <span id="stat-host-ms" class="telemetry-value">-- <small>ms</small></span>
            </div>
            <div class="telemetry-card">
              <span class="telemetry-label">Step Total</span>
              <span id="stat-step-ms" class="telemetry-value">-- <small>ms</small></span>
            </div>
          </div>

          <!-- Streaming Monospace Output -->
          <div class="output-wrapper">
            <div class="output-header">
              <span class="output-title">Streaming Terminal Output</span>
              <button id="clear-output-btn" class="btn-subtle" title="Clear terminal">Clear</button>
            </div>
            <div id="output-stream" class="output-stream" role="log" aria-live="polite">
              <span class="output-placeholder">Generated text will stream here in real time...</span>
            </div>
          </div>
        </section>

        <!-- Error Notification Toast -->
        <div id="toast-banner" class="toast-banner hidden">
          <span id="toast-message">Error message</span>
          <button id="toast-dismiss-btn" class="toast-dismiss">✕</button>
        </div>
      </div>
    `;
  }

  _bindEvents() {
    const q = id => document.getElementById(id);

    // Temperature slider display sync
    q('temp-slider').addEventListener('input', (e) => {
      q('temp-val').textContent = parseFloat(e.target.value).toFixed(1);
    });

    // Max tokens input sync
    q('max-tokens-input').addEventListener('input', (e) => {
      q('max-tokens-val').textContent = e.target.value;
    });

    // Copy link button
    q('copy-link-btn').addEventListener('click', () => {
      const room = q('room-code-display').textContent;
      const url = `${window.location.origin}/?room=${room}`;
      navigator.clipboard.writeText(url).then(() => {
        const btn = q('copy-link-btn');
        btn.textContent = '✓ Copied!';
        setTimeout(() => btn.textContent = '📋 Copy Link', 2000);
      });
    });

    // Lock Ring button
    q('lock-ring-btn').addEventListener('click', () => {
      if (this._onLockRingCb) {
        this._onLockRingCb();
      }
    });

    // Generate button
    q('generate-btn').addEventListener('click', () => {
      if (this._onGenerateCb) {
        const prompt = q('prompt-input').value;
        const maxTokens = parseInt(q('max-tokens-input').value, 10) || 100;
        const temperature = parseFloat(q('temp-slider').value) || 0.0;
        this._onGenerateCb({ prompt, maxTokens, temperature });
      }
    });

    // Stop button
    q('stop-btn').addEventListener('click', () => {
      if (this._onStopCb) {
        this._onStopCb();
      }
    });

    // Clear output
    q('clear-output-btn').addEventListener('click', () => {
      q('output-stream').innerHTML = '';
    });

    // Toast dismiss
    q('toast-dismiss-btn').addEventListener('click', () => {
      q('toast-banner').classList.add('hidden');
    });
  }

  // ── Master QR Code Generation ──────────────────────────────────────────────

  /**
   * Render Master QR code using available QR engine or fallback.
   * @param {string} roomId
   */
  renderMasterQR(roomId) {
    const q = id => document.getElementById(id);
    const roomUrl = `${window.location.origin}/?room=${roomId}`;

    q('room-code-display').textContent = roomId;
    const qrDisplay = q('qr-display');
    qrDisplay.innerHTML = '';

    // Check if qrcode-generator is loaded on window
    if (typeof window.qrcode === 'function') {
      try {
        const qr = window.qrcode(0, 'M');
        qr.addData(roomUrl);
        qr.make();
        qrDisplay.innerHTML = qr.createSvgTag(5, 4);
        return;
      } catch (e) {
        console.warn('[HostUI] window.qrcode failed, using canvas fallback:', e);
      }
    }

    // Canvas / Img Fallback using lightweight QR generator endpoint or SVG block
    const img = document.createElement('img');
    img.src = `https://api.qrserver.com/v1/create-qr-code/?size=180x180&data=${encodeURIComponent(roomUrl)}`;
    img.alt = `QR Code for room ${roomId}`;
    img.className = 'qr-img-element';
    img.onerror = () => {
      qrDisplay.innerHTML = `
        <div class="qr-offline-card">
          <p>Scan with other phone or open:</p>
          <a href="${roomUrl}" target="_blank" class="qr-url-link">${roomUrl}</a>
        </div>
      `;
    };
    qrDisplay.appendChild(img);
  }

  // ── Dynamic Canvas Ring Visualizer ─────────────────────────────────────────

  _initCanvas() {
    this._canvas = document.getElementById('mesh-ring-canvas');
    if (!this._canvas) return;
    this._ctx = this._canvas.getContext('2d');
    this._drawRing();
  }

  /**
   * Update the roster of connected devices in the lobby.
   * @param {Array<{peerId: string, deviceName: string}>} peers
   */
  updateLobbyPeers(peers) {
    this._peers = peers;
    this._totalNodes = peers.length;

    const q = id => document.getElementById(id);
    q('node-count-badge').textContent = `${this._totalNodes} / 6 Phones`;

    // Enable Lock button if we have at least 2 phones
    const lockBtn = q('lock-ring-btn');
    if (this._totalNodes >= 2) {
      lockBtn.disabled = false;
      lockBtn.innerHTML = `🔒 Lock Mesh & Form Ring (${this._totalNodes} Phones)`;
    } else {
      lockBtn.disabled = true;
      lockBtn.innerHTML = `🔒 Lock Mesh & Form Ring (Need ≥ 2 Phones)`;
    }

    this._drawRing();
  }

  /**
   * Draw circular dynamic ring with directional arrows and glowing nodes.
   */
  _drawRing() {
    if (!this._ctx || !this._canvas) return;
    const ctx = this._ctx;
    const w = this._canvas.width;
    const h = this._canvas.height;
    const cx = w / 2;
    const cy = h / 2;
    const radius = Math.min(w, h) * 0.36;

    ctx.clearRect(0, 0, w, h);

    const N = Math.max(this._totalNodes, 2);

    // 1. Draw circular ring track
    ctx.beginPath();
    ctx.arc(cx, cy, radius, 0, 2 * Math.PI);
    ctx.strokeStyle = this._isLocked ? 'rgba(0, 255, 136, 0.3)' : 'rgba(255, 255, 255, 0.1)';
    ctx.lineWidth = 2;
    ctx.setLineDash(this._isLocked ? [] : [4, 4]);
    ctx.stroke();
    ctx.setLineDash([]);

    // 2. Draw nodes around circle
    for (let i = 0; i < N; i++) {
      const angle = (2 * Math.PI * i) / N - Math.PI / 2;
      const nx = cx + radius * Math.cos(angle);
      const ny = cy + radius * Math.sin(angle);

      const isConnected = i < this._peers.length;
      const isHost = (i === 0);
      const isSampler = (i === N - 1);
      const isPulsing = (i === this._pulseNodeIndex);

      // Pulse ring glow
      if (isPulsing) {
        ctx.beginPath();
        ctx.arc(nx, ny, 22, 0, 2 * Math.PI);
        ctx.fillStyle = 'rgba(0, 255, 136, 0.35)';
        ctx.fill();
      }

      // Outer node circle
      ctx.beginPath();
      ctx.arc(nx, ny, 16, 0, 2 * Math.PI);

      if (isConnected) {
        ctx.fillStyle = isHost ? '#00ff88' : isSampler ? '#00b4d8' : '#2ecc71';
      } else {
        ctx.fillStyle = '#333333';
      }
      ctx.fill();
      ctx.strokeStyle = isConnected ? '#ffffff' : '#555555';
      ctx.lineWidth = 2;
      ctx.stroke();

      // Node number text
      ctx.fillStyle = isConnected ? '#111111' : '#888888';
      ctx.font = 'bold 11px sans-serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(`P${i + 1}`, nx, ny);

      // Node role label outside circle
      const labelDist = radius + 24;
      const lx = cx + labelDist * Math.cos(angle);
      const ly = cy + labelDist * Math.sin(angle);

      ctx.font = '10px sans-serif';
      ctx.fillStyle = isConnected ? '#e0e0e0' : '#666666';
      const roleText = isHost ? 'Host' : isSampler && this._isLocked ? 'Sampler' : `Node ${i}`;
      ctx.fillText(roleText, lx, ly);
    }

    // Center topology label
    ctx.fillStyle = this._isLocked ? '#00ff88' : '#aaaaaa';
    ctx.font = 'bold 12px monospace';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(this._isLocked ? `SEALED (${N} NODES)` : `LOBBY (${this._totalNodes}/6)`, cx, cy);
  }

  // ── Ring Transition to Inference ───────────────────────────────────────────

  /**
   * Transition view from pairing lobby to active distributed inference panel.
   * @param {object} ringConfig
   */
  showInferencePanel(ringConfig) {
    this._isLocked = true;
    const q = id => document.getElementById(id);

    // Hide lobby, reveal inference panel
    q('lobby-section').classList.add('hidden');
    q('inference-section').classList.remove('hidden');

    // Update header status pill
    const pill = q('connection-status-pill');
    pill.className = 'status-pill status-connected';
    q('connection-status-text').textContent = `Mesh Sealed (${ringConfig.totalNodes} Phones Active)`;

    // Redraw sealed ring
    this._drawRing();
  }

  // ── Streaming Text & Telemetry Updates ─────────────────────────────────────

  /**
   * Stream a decoded character to the terminal output.
   * @param {string} char
   */
  appendTokenChar(char) {
    const q = id => document.getElementById(id);
    const stream = q('output-stream');

    // Remove placeholder on first character
    const placeholder = stream.querySelector('.output-placeholder');
    if (placeholder) placeholder.remove();

    if (char === '\n') {
      stream.appendChild(document.createElement('br'));
    } else {
      const span = document.createElement('span');
      span.className = 'token-char';
      span.textContent = char;
      stream.appendChild(span);
    }

    // Auto-scroll to bottom
    stream.scrollTop = stream.scrollHeight;
  }

  /**
   * Update Telemetry HUD metrics.
   * @param {object} stats
   */
  updateTelemetry(stats) {
    const q = id => document.getElementById(id);

    if (stats.currentTps !== undefined) {
      q('stat-tps').innerHTML = `${stats.currentTps.toFixed(1)} <small>TPS</small>`;
    }
    if (stats.lastRingRttMs !== undefined) {
      q('stat-rtt').innerHTML = `${stats.lastRingRttMs.toFixed(0)} <small>ms</small>`;
    }
    if (stats.lastHostComputeMs !== undefined) {
      q('stat-host-ms').innerHTML = `${stats.lastHostComputeMs.toFixed(1)} <small>ms</small>`;
    }
    if (stats.lastStepMs !== undefined) {
      q('stat-step-ms').innerHTML = `${stats.lastStepMs.toFixed(0)} <small>ms</small>`;
    }
  }

  /**
   * Set generation active state (toggle buttons & spinners).
   * @param {boolean} isGenerating
   */
  setGenerating(isGenerating) {
    this._isGenerating = isGenerating;
    const q = id => document.getElementById(id);

    if (isGenerating) {
      q('generate-btn').classList.add('hidden');
      q('stop-btn').classList.remove('hidden');
      q('prompt-input').disabled = true;
    } else {
      q('generate-btn').classList.remove('hidden');
      q('stop-btn').classList.add('hidden');
      q('prompt-input').disabled = false;
    }
  }

  showError(msg) {
    const q = id => document.getElementById(id);
    q('toast-message').textContent = msg;
    q('toast-banner').classList.remove('hidden');
  }

  // ── Callback Registration ──────────────────────────────────────────────────

  onLockRing(cb) {
    this._onLockRingCb = cb;
  }

  onGenerate(cb) {
    this._onGenerateCb = cb;
  }

  onStop(cb) {
    this._onStopCb = cb;
  }
}

export { HostUI };

