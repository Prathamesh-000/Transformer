# P2P Mobile Mesh Inference: Complete Architectural Blueprint & Implementation Plan

> **Project:** Distributed Byte-Level Micro-Transformer across Two Stock Mobile Phones  
> **Environment:** Stock Mobile Browsers (iOS Safari / Android Chrome) on Web CPU (JavaScript)  
> **Target Devices:** Any two consumer smartphones connected over local peer-to-peer Wi-Fi  

---

## 1. Executive Summary & Objective

The objective of this project is to build a web application capable of running distributed text generation for a **Byte-Level Micro-Transformer** across **two physical mobile phones** connected over a local peer-to-peer network (WebRTC DataChannel).

The forward inference pass must be split across both phones in real time so that both devices actively participate in generating every single byte:
* **Phone 1 (Host / Initiator):** Computes Token Embeddings + Positional Embeddings + Transformer Layers 0, 1, and 2.
* **Phone 2 (Guest / Receiver):** Computes Transformer Layers 3, 4, and 5 + Final LayerNorm + LM Head + Greedy Token Sampler.

Neither phone ever executes the full model alone.

```
┌────────────────────────────────────────────────────────────────────────┐
│                        SPLIT INFERENCE TOPOLOGY                        │
└────────────────────────────────────────────────────────────────────────┘

  [Phone 1: Host]                                [Phone 2: Guest]
┌─────────────────────────┐                    ┌─────────────────────────┐
│ • Token Embedding       │                    │ • Transformer Layer 3   │
│ • Positional Embedding  │                    │ • Transformer Layer 4   │
│ • Transformer Layer 0   │                    │ • Transformer Layer 5   │
│ • Transformer Layer 1   │──h₂ [1, 384] (1.5KB)─▶ • Final LayerNorm     │
│ • Transformer Layer 2   │◀──Sampled Byte (1B)─│ • LM Head + Argmax      │
└─────────────────────────┘                    └─────────────────────────┘
             ▲                                              ▲
             └──────── WebRTC DataChannel (Local Wi-Fi) ────┘
```

---

## 2. Hard Requirements & Prohibitions

| Requirement | Strict Compliance Rules |
|---|---|
| **P2P Split Execution** | Layers 0–2 run on Phone 1; Layers 3–5 run on Phone 2. Split occurs at the activation tensor boundary $h_2$. |
| **Local Transport** | All activation tensors and generated tokens travel over WebRTC DataChannel directly between phones over local Wi-Fi. |
| **Standalone Runtime** | Zero inference offloading. Once loaded, generation runs 100% offline across the two devices. |
| **Zero App Installations** | Must run in stock mobile Safari or Chrome. Strictly NO native apps, NO APK sideloading, NO TestFlight, NO custom browser flags. |
| **Compute Target** | Web CPU only (pure JavaScript / Float32Array). Strictly NO WebGPU, NO WebGL shaders, NO WASM SIMD experimental toggles. |
| **Zero Centralized Relays** | Signaling server is strictly used for the initial SDP/ICE handshake. Zero activation or inference bytes may pass through any central server or TURN relay. |
| **Deterministic Seed 42** | Weights generated with PyTorch `torch.manual_seed(42)`. At `temperature = 0.0`, all implementations must output the exact same deterministic byte sequence. |

---

## 3. Exact Model Architecture & Parameter Verification

### 3.1 Architectural Hyperparameters
* **Model Class:** Vanilla Causal Decoder Transformer (GPT-2 style, Pre-LN)
* **Vocab Size ($V$):** `256` (Direct ASCII / UTF-8 bytes `[0..255]`; no BPE tokenizer needed)
* **Context Length ($T_{max}$):** `512`
* **Layers ($L$):** `6` (Phone 1: Layers 0–2, Phone 2: Layers 3–5)
* **Hidden Dimension ($d_{model}$):** `384`
* **Attention Heads ($n_{heads}$):** `6`
* **Head Dimension ($d_{head}$):** $384 / 6 = \mathbf{64}$
* **Feed-Forward Dimension ($d_{ff}$):** $4 \times 384 = \mathbf{1536}$

### 3.2 Exact Parameter Verification (11,040,768 Parameters)
Every linear layer includes additive bias; each Transformer block employs Pre-LayerNorm (with learned gain $\gamma$ and bias $\beta$):

| Component | Dimension / Formula | Weights | Biases | Total Parameters |
|---|---|---|---|---|
| **Token Embedding** | $256 \times 384$ | $98,304$ | — | $98,304$ |
| **Positional Embedding** | $512 \times 384$ | $196,608$ | — | $196,608$ |
| **Per Layer ($L_i$): LN 1** | $384$ | $384$ | $384$ | $768$ |
| **Per Layer ($L_i$): QKV Proj** | $384 \to 3 \times 384$ ($1152$) | $442,368$ | $1,152$ | $443,520$ |
| **Per Layer ($L_i$): Out Proj** | $384 \to 384$ | $147,456$ | $384$ | $147,840$ |
| **Per Layer ($L_i$): LN 2** | $384$ | $384$ | $384$ | $768$ |
| **Per Layer ($L_i$): FFN Up** | $384 \to 1536$ | $589,824$ | $1,536$ | $591,360$ |
| **Per Layer ($L_i$): FFN Down** | $1536 \to 384$ | $589,824$ | $384$ | $590,208$ |
| **Single Transformer Block** | Sum of above | $1,770,240$ | $4,224$ | **$1,774,464$** |
| **All 6 Blocks Total** | $6 \times 1,774,464$ | $10,621,440$ | $25,344$ | **$10,646,784$** |
| **Final LayerNorm** | $384$ | $384$ | $384$ | $768$ |
| **LM Head** (tied to Token Emb) | $384 \to 256$ | $98,304$ | — | $98,304$ |
| **GRAND TOTAL** | | **$11,014,656$** | **$26,112$** | **$\mathbf{11,040,768}$** |

$$\text{Binary Size in FP32} = 11,040,768 \times 4\text{ bytes} = \mathbf{44,163,072\text{ bytes}} \approx \mathbf{42.12\text{ MB}}$$

---

## 4. Team Work Breakdown: Member 1 vs Member 2

The project is split down the network boundary so both members can work, test, and debug in parallel.

```
┌────────────────────────────────────────────────────────────────────────┐
│                        MEMBER 1: NETWORK & HOST                        │
│                 (Phone 1 Pipeline & Web Infrastructure)                │
├────────────────────────────────┬───────────────────────────────────────┤
│ Responsibilities:              │ Deliverables:                         │
│ • WebRTC P2P DataChannel       │ 1. server/server.js                   │
│ • WebSocket Signaling Relay    │ 2. client/network/signaling.js        │
│ • Phone 1 Autoregressive Loop  │ 3. client/network/p2p.js              │
│ • Host UI & QR Code Generator  │ 4. client/core/phone1.js              │
│ • Real-time Telemetry (TPS/RTT)│ 5. client/ui/host-ui.js               │
│                                │ 6. client/index.html & style.css      │
└────────────────────────────────┴───────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────────┐
│                        MEMBER 2: ML & INFERENCE                        │
│                (Transformer Math Engine & Phone 2 Pipeline)            │
├────────────────────────────────┬───────────────────────────────────────┤
│ Responsibilities:              │ Deliverables:                         │
│ • PyTorch Seed-42 Weight Gen   │ 1. tools/generate_weights.py          │
│ • Zero-Copy Weight Slicer (JS) │ 2. tools/verify_weights.py            │
│ • Pure CPU Math Engine (JS)    │ 3. client/core/weights.js             │
│ • KV-Cache Partitioning System │ 4. client/core/transformer.js         │
│ • Phone 2 Autoregressive Loop  │ 5. client/core/phone2.js              │
│ • Guest UI & Live Token Mirror │ 6. client/ui/guest-ui.js              │
└────────────────────────────────┴───────────────────────────────────────┘
```

---

## 5. Technical Architecture & Protocols

### 5.1 Independent Partitioned KV-Cache Mechanics

To achieve high throughput on mobile Wi-Fi, the system separates **Prefill** from **Decode**:

```
PREFILL PHASE (Initial Prompt: length P tokens):
  1. Phone 1 receives prompt string, converts to bytes [b₀, b₁, ... b_{P-1}].
  2. Phone 1 embeds all P tokens and runs Layers 0–2 forward.
  3. Phone 1 stores K, V for Layers 0–2 at indices 0..P-1 in its internal cache.
  4. Phone 1 sends activation tensor h₂: shape [P, 384] to Phone 2.
  5. Phone 2 forwards [P, 384] through Layers 3–5.
  6. Phone 2 stores K, V for Layers 3–5 at indices 0..P-1 in its internal cache.
  7. Phone 2 takes h₂[P-1], applies Final LN + LM Head, computes Logits[256].
  8. Phone 2 samples token byte t_{P} via argmax (temp=0.0).
  9. Phone 2 sends 1 byte back to Phone 1.

DECODE PHASE (Autoregressive per-token generation: length 1 token):
  1. Phone 1 receives t_{P}.
  2. Phone 1 embeds ONLY token t_{P} at position startPos.
  3. Phone 1 runs Layers 0–2: new Query attends over cached K, V (indices 0..startPos).
  4. Phone 1 sends activation tensor h₂: shape [1, 384] (1,536 bytes) to Phone 2.
  5. Phone 2 receives 1,536 bytes.
  6. Phone 2 runs Layers 3–5: new Query attends over cached K, V (indices 0..startPos).
  7. Phone 2 applies Final LN + LM Head to produce Logits[256].
  8. Phone 2 samples next token byte via argmax.
  9. Phone 2 sends 1 byte back to Phone 1.
  10. Loop repeats until max_new_tokens or token == 0 (null byte sentinel).
```

**Network Bandwidth Advantage:**  
During decode, Phone 1 transmits only $1 \times 384 \times 4\text{ bytes} = \mathbf{1,536\text{ bytes}}$, and Phone 2 replies with **1 byte**. This ensures latency remains dominated by CPU compute rather than Wi-Fi packet transfer.

---

### 5.2 WebRTC DataChannel & Binary Wire Framing

To bypass JSON serialization overhead, all DataChannel traffic uses a binary framing protocol:

```
┌───────────────┬────────────────────────────────────────────────────────┐
│ Prefix (Byte) │ Data Payload                                           │
├───────────────┼────────────────────────────────────────────────────────┤
│ 0x01          │ Activation Tensor: Float32Array little-endian bytes    │
│               │ Length: (seq_len × 384 × 4) bytes                      │
├───────────────┼────────────────────────────────────────────────────────┤
│ 0x02          │ Token Byte: Uint8 value [0..255]                       │
│               │ Length: 1 byte                                         │
└───────────────┴────────────────────────────────────────────────────────┘
```

#### WebRTC Configuration Parameters:
```javascript
const RTC_CONFIG = {
  iceServers: [
    { urls: 'stun:stun.l.google.com:19302' },
    { urls: 'stun:stun1.l.google.com:19302' }
  ]
};

const DATA_CHANNEL_CONFIG = {
  ordered: true,        // Token stream must be strictly in-order
  maxRetransmits: 30    // Reliable transport over UDP
};
```

---

### 5.3 Pure Web CPU Math Engine

Since WebGPU and WebGL are prohibited, all operations are written in vanilla JavaScript using `Float32Array`:

1. **32×32 Tiled Matrix Multiplication (`matmul`):**
   Standard nested loops cause continuous L1/L2 cache evictions on mobile processors. Tiling matrix multiplication into $32 \times 32$ blocks ensures memory tiles remain in L1 cache during inner loop execution:
   ```javascript
   function matmulTiled(A, B, out, M, K, N) {
     const TILE = 32;
     out.fill(0);
     for (let i0 = 0; i0 < M; i0 += TILE) {
       for (let j0 = 0; j0 < N; j0 += TILE) {
         for (let k0 = 0; k0 < K; k0 += TILE) {
           const iMax = Math.min(i0 + TILE, M);
           const jMax = Math.min(j0 + TILE, N);
           const kMax = Math.min(k0 + TILE, K);
           for (let i = i0; i < iMax; i++) {
             for (let k = k0; k < kMax; k++) {
               const aik = A[i * K + k];
               for (let j = j0; j < jMax; j++) {
                 out[i * N + j] += aik * B[k * N + j];
               }
             }
           }
         }
       }
     }
   }
   ```

2. **Standard LayerNorm ($\epsilon = 10^{-5}$):**
   $$\mu = \frac{1}{d} \sum_{i=1}^d x_i, \quad \sigma^2 = \frac{1}{d} \sum_{i=1}^d (x_i - \mu)^2, \quad y_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}} \cdot \gamma_i + \beta_i$$

3. **GELU Activation (Abramowitz & Stegun Approximation 7.1.26):**
   $$\text{GELU}(x) = 0.5 \cdot x \cdot \left(1 + \text{erf}\left(\frac{x}{\sqrt{2}}\right)\right)$$
   Evaluated with Horner's polynomial method for maximum numerical stability and zero external dependencies.

4. **Greedy Argmax Sampling (`temp = 0.0`):**
   Scans `logits[0..255]` and selects the index with the highest activation value to ensure deterministic text generation.

---

## 6. Interface Contracts Between Member 1 & Member 2

To enable parallel development without integration friction, both members must adhere to these exact interfaces:

### Interface 1: Weight Slicer (`client/core/weights.js`)
*Produced by Member 2 $\to$ Consumed by Member 1's `index.html`*
```typescript
interface LayerWeights {
  ln1w: Float32Array;  // [384]
  ln1b: Float32Array;  // [384]
  qkvW: Float32Array;  // [1152, 384]
  qkvB: Float32Array;  // [1152]
  oW:   Float32Array;  // [384, 384]
  oB:   Float32Array;  // [384]
  ln2w: Float32Array;  // [384]
  ln2b: Float32Array;  // [384]
  ffW1: Float32Array;  // [1536, 384]
  ffB1: Float32Array;  // [1536]
  ffW2: Float32Array;  // [384, 1536]
  ffB2: Float32Array;  // [384]
}

interface Phone1Weights {
  tokenEmb: Float32Array; // [256, 384]
  posEmb:   Float32Array; // [512, 384]
  layers:   LayerWeights[]; // Exactly 3 items (Layers 0, 1, 2)
}

interface Phone2Weights {
  layers:   LayerWeights[]; // Exactly 3 items (Layers 3, 4, 5)
  finalLnW: Float32Array; // [384]
  finalLnB: Float32Array; // [384]
  lmHeadW:  Float32Array; // [256, 384]
}

function loadWeights(onProgress?: (pct: number) => void): Promise<{
  phone1Weights: Phone1Weights;
  phone2Weights: Phone2Weights;
}>;
```

### Interface 2: Transformer Math Engine (`client/core/transformer.js`)
*Produced by Member 2 $\to$ Consumed by `phone1.js` & `phone2.js`*
```typescript
class TransformerPhone1 {
  constructor(weights: Phone1Weights);
  forward(tokenIds: Uint8Array | number[], startPos: number): Float32Array;
  reset(): void;
  get sequenceLength(): number;
}

class TransformerPhone2 {
  constructor(weights: Phone2Weights);
  forward(h: Float32Array, startPos: number, temperature?: number): {
    token: number;
    logits: Float32Array;
  };
  reset(): void;
  get sequenceLength(): number;
}
```

### Interface 3: WebRTC P2P Channel (`client/network/p2p.js`)
*Produced by Member 1 $\to$ Consumed by `phone1.js` & `phone2.js`*
```typescript
class P2PChannel {
  constructor(signalingClient: SignalingClient, isHost: boolean);
  initiate(): Promise<void>;
  waitForOpen(): Promise<void>;
  
  sendActivation(data: Float32Array): void;
  onActivation(callback: (data: Float32Array) => void): void;
  offActivation(callback: Function): void;
  
  sendToken(token: number): void;
  onToken(callback: (token: number) => void): void;
  
  get isConnected(): boolean;
  close(): void;
}
```

---

## 7. Step-by-Step Implementation Roadmap

```
PHASE 1: Foundation & Model Weights Export
  ├─ Member 2: Generate & verify weights.bin with PyTorch seed 42
  └─ Member 1: Build Node.js WebSocket signaling server & room manager

PHASE 2: WebRTC Transport & Core Math
  ├─ Member 2: Build weights.js slicer & pure CPU transformer.js
  └─ Member 1: Build p2p.js DataChannel manager & binary framing

PHASE 3: Split Orchestration Runtimes
  ├─ Member 2: Build phone2.js (Layers 3-5 + argmax sampler)
  └─ Member 1: Build phone1.js (Autoregressive loop & prefill/decode)

PHASE 4: Mobile UI & QR Pairing
  ├─ Member 2: Build guest-ui.js (Mirror stream & compute latency stats)
  └─ Member 1: Build host-ui.js (QR generator, prompt input, TPS counter)

PHASE 5: Integration, Benchmarking & Polish
  └─ Both: End-to-end testing across 2 physical phones over local Wi-Fi
```

### Phase 1: Foundation & Model Weights Export
* **Member 2 Tasks:**
  1. Write `tools/generate_weights.py` with `torch.manual_seed(42)`.
  2. Implement sequential export of all 19 parameter groups as FP32 little-endian bytes.
  3. Tie `lm_head.weight` directly to `token_emb.weight`.
  4. Write `tools/verify_weights.py` to assert that the generated file is exactly `44,163,072` bytes and verify the sanity of the QKV projection output.
* **Member 1 Tasks:**
  1. Create `server/package.json` with dependencies `express` and `ws`.
  2. Write `server/server.js`:
     - Serves static assets from `client/`.
     - Serves `/weights.bin` with `Content-Type: application/octet-stream` and `Cache-Control` headers.
     - Implements `/signal` WebSocket endpoint to manage rooms (`create`, `join`) and relay SDP/ICE messages.

### Phase 2: WebRTC Transport & Core Math
* **Member 2 Tasks:**
  1. Build `client/core/weights.js`: Fetches `/weights.bin` via `XMLHttpRequest` with progress feedback, returns zero-copy typed views (`phone1Weights`, `phone2Weights`).
  2. Build `client/core/transformer.js`:
     - Implement 32×32 tiled `matmul` and `linearForward`.
     - Implement `layerNorm` and Horner's polynomial `gelu`.
     - Implement multi-head self-attention with causal masking.
     - Allocate KV-cache buffers of shape `[512, 384]` per layer.
* **Member 1 Tasks:**
  1. Build `client/network/signaling.js`: Client WebSocket wrapper supporting automatic reconnection and room events.
  2. Build `client/network/p2p.js`:
     - Implements `RTCPeerConnection` with STUN discovery.
     - Sets up `RTCDataChannel("inference", { ordered: true, maxRetransmits: 30 })`.
     - Handles binary message dispatching (`0x01` activation vs `0x02` token).

### Phase 3: Split Orchestration Runtimes
* **Member 1 Tasks (`phone1.js`):**
  1. Create `Phone1Runtime` class wrapping `TransformerPhone1`.
  2. Implement `generate(promptBytes, maxTokens, temperature)`:
     - **Prefill:** Pass prompt to `transformer.forward(prompt, 0)`. Send returned tensor via `p2p.sendActivation()`. Await first token.
     - **Decode Loop:** Iteratively forward single new token at `startPos`, transmit `[1, 384]` (1,536 bytes) to Phone 2, await next byte token.
     - Record timestamps: $T_{compute}$, $T_{RTT}$, $T_{step}$.
* **Member 2 Tasks (`phone2.js`):**
  1. Create `Phone2Runtime` class wrapping `TransformerPhone2`.
  2. Register `p2p.onActivation(async (float32Array) => { ... })`:
     - Infer sequence length from `float32Array.length / 384`.
     - Execute `transformer.forward(float32Array, startPos, temperature)`.
     - Select argmax byte token.
     - Send 1 byte back via `p2p.sendToken(token)`.

### Phase 4: Mobile UI & QR Pairing
* **Member 1 Tasks (`host-ui.js` & `index.html`):**
  1. Parse URL query params (`?role=host` vs `?role=guest`).
  2. On host: Connect to signaling $\to$ create room $\to$ render QR code encoding `https://<domain>/?role=guest&room=<ROOM_ID>`.
  3. Render prompt textarea, temperature slider, max tokens, and "Generate" button.
  4. Stream decoded UTF-8 characters as tokens arrive.
  5. Render real-time performance dashboard: **TPS**, **Phone 1 Latency**, **DataChannel RTT**.
* **Member 2 Tasks (`guest-ui.js`):**
  1. On guest: Connect to signaling $\to$ join room $\to$ establish P2P.
  2. Render connection status badge: "📱 Phone 2 — Layers 3–5 + LM Head".
  3. Render mirrored output box displaying decoded characters in real time.
  4. Render Phone 2 compute metrics tile.

### Phase 5: Integration, Benchmarking & Polish
* Both members test the end-to-end system on two physical mobile phones (iOS Safari + Android Chrome).
* Verify determinism: Run identical prompts at `temperature = 0.0` and assert 100% identical byte outputs.

---

## 8. Expected Directory Layout

```
Transformer/
├── IMPLEMENTATION_PLAN.md    # This master document
├── README.md                 # Project quickstart & team overview
├── weights.bin               # Seed-42 FP32 binary weights (44,163,072 bytes)
│
├── server/
│   ├── package.json          # Express + ws dependencies
│   └── server.js             # Signaling relay & static file host
│
├── client/
│   ├── index.html            # Main entry (detects ?role=host / ?role=guest)
│   ├── style.css             # Mobile-first dark theme CSS
│   │
│   ├── core/
│   │   ├── weights.js        # Zero-copy Float32Array weight loader
│   │   ├── transformer.js    # Pure CPU math engine (tiled matmul, attention, FFN)
│   │   ├── phone1.js         # Phone 1 orchestrator (Layers 0-2 loop)
│   │   └── phone2.js         # Phone 2 orchestrator (Layers 3-5 + sampling)
│   │
│   ├── network/
│   │   ├── signaling.js      # WebSocket signaling client
│   │   └── p2p.js            # WebRTC DataChannel manager
│   │
│   └── ui/
│       ├── host-ui.js        # Phone 1 UI (QR code, prompt, output, stats)
│       └── guest-ui.js       # Phone 2 UI (Status badge, mirror stream, stats)
│
└── tools/
    ├── generate_weights.py   # PyTorch seed-42 weights generator
    └── verify_weights.py     # Structural & mathematical sanity validator
```

---

## 9. Execution & Verification Routine

### Step 1: Generate & Validate Weights
```bash
cd tools
python generate_weights.py
python verify_weights.py
```
*Verification criteria:* File size must equal `44,163,072` bytes; total parameters must equal `11,040,768`.

### Step 2: Start Local Server
```bash
cd ../server
npm install
node server.js
```

### Step 3: Expose via HTTPS Tunnel (Required for WebRTC)
```bash
npx ngrok http 3000
```
Mobile Safari and Chrome require an HTTPS context to grant WebRTC permissions.

### Step 4: Pair Phones via QR Code
1. Open `https://<ngrok-subdomain>.ngrok-free.app/?role=host` on **Phone 1**.
2. Scan the displayed QR code with **Phone 2**.
3. Both phones negotiate direct P2P connection over the local Wi-Fi.

### Step 5: Benchmark Text Generation
1. Input prompt on Phone 1.
2. Set `temperature = 0.0`.
3. Tap **Generate Text**.
4. Observe real-time token streaming across both devices and record benchmark metrics.

---

## 10. Benchmark Metrics to Report to Judges

| Metric | Target | Measurement Method |
|---|---|---|
| **P2P Split Verification** | Compliant | Phone 1 logs show Layers 0–2 only; Phone 2 logs show Layers 3–5 only. |
| **Throughput (Tokens/sec)** | `> 15–30 TPS` | $\text{Generated Tokens} / \text{Elapsed Seconds}$ |
| **Phone 1 Compute Time** | `~12–25 ms` | `performance.now()` delta around `TransformerPhone1.forward()` |
| **Phone 2 Compute Time** | `~15–30 ms` | `performance.now()` delta around `TransformerPhone2.forward()` |
| **DataChannel RTT** | `< 4–10 ms` | Time delta from activation transmission to token receipt over local Wi-Fi |
| **Zero Cloud Compute** | Offline verified | Disable WAN connection on router after page load; inference continues uninterrupted. |
| **Determinism** | 100% Identical | Same prompt at `temp = 0.0` yields identical byte output on any pair of phones. |

---

## 11. Key Edge Cases & Mitigations

1. **Mobile Browser Sleep Throttling:**
   - *Issue:* iOS Safari throttles timers when inactive.
   - *Mitigation:* Activate `navigator.wakeLock.request('screen')` on the first touch event to prevent screen dimming during generation.
2. **TextDecoder UTF-8 Multi-byte Streaming:**
   - *Issue:* Multi-byte characters (emojis, unicode symbols) will break if decoded byte-by-byte.
   - *Mitigation:* Use `new TextDecoder('utf-8', { stream: true })` to buffer incomplete byte sequences until all constituent bytes arrive.
3. **DataChannel Backpressure:**
   - *Issue:* Unbounded `.send()` calls will overwhelm the network buffer.
   - *Mitigation:* Strictly sequential promise-locking: Phone 1 never transmits activation $t+1$ until receiving token $t$.

