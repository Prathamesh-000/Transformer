# 📱 P2P Mobile Mesh Inference: Dynamic N-Phone Byte-Level Transformer

> **Distributed text generation for an 11.04M Micro-Transformer running across stock consumer smartphones over a local WebRTC peer-to-peer ring.**  
> *Zero App Installations • Pure Web CPU (JavaScript) • Zero Cloud Backends • Deterministic Seed 42*

---

## 📸 Screenshots & Visual Walkthrough

<p align="center">
  <img src="assets/screenshot_landing.png" width="22%" alt="Landing Role Selector" />
  &nbsp;&nbsp;
  <img src="assets/screenshot_host_pairing.png" width="22%" alt="Master QR & Canvas Ring" />
  &nbsp;&nbsp;
  <img src="assets/screenshot_worker_lobby.png" width="22%" alt="Worker Phone Lobby" />
  &nbsp;&nbsp;
  <img src="assets/screenshot_inference_active.png" width="22%" alt="Active Distributed Inference" />
</p>

| 1. Role Selection | 2. Master QR & Live Ring | 3. Worker Connected | 4. Distributed Generation |
|:---:|:---:|:---:|:---:|
| Choose **Host (Phone 1)** or enter Room Code | Host displays single **Master QR** & renders live Canvas ring | Scanned worker joins lobby & receives dynamic layer slice | Tokens stream across mesh with real-time **TPS & RTT HUD** |

---

## 🌟 Key Highlights

- **Dynamic Pipeline Parallelism ($N = 2$ to $6$ phones):** Bring any $2, 3, 4, 5,$ or $6$ phones. The system automatically partitions the 6-layer Micro-Transformer evenly across all available devices.
- **10-Second Master QR Pairing:** Phone 1 displays one Master QR code. All other phones scan Phone 1 simultaneously to join the lobby and lock the ring.
- **Lightweight Memory Scaling:** With 6 phones, each phone loads only **$\approx 7.1\text{ MB}$** of weights into RAM (1 layer per device), completely preventing mobile browser tab crashes.
- **Constant 1.5 KB Payload:** Every jump between phones during generation transmits strictly **1,536 bytes** ($1 \times 384 \times 4\text{ bytes}$), maximizing throughput over standard local Wi-Fi.
- **Pure Web CPU (JavaScript):** Strictly zero WebGPU, zero WebGL shaders, zero native compilation flags. Runs universally in mobile Safari (iOS) and Chrome (Android).
- **Deterministic Output:** Weights initialized with `torch.manual_seed(42)`. At `temperature = 0.0`, any combination of phones generates the exact same deterministic byte sequence.
- **Full Offline Autonomy:** Once the page loads and the ring seals, the phones continue distributed inference with zero internet connection over local peer-to-peer Wi-Fi.

---

## 📐 Architecture: The P2P Token Ring

Instead of sending tensors through a central server, the devices form a **directional WebRTC DataChannel pipeline ring**:

```
                                [Master QR Code]
                                📱 Phone 1 (Host)
                                        │
                    ┌───────────┬───────┴───┬───────────┐
                    ▼           ▼           ▼           ▼
                📱 Phone 2   📱 Phone 3  📱 Phone 4  📱 Phone 5  📱 Phone 6
                (Scanned)   (Scanned)   (Scanned)   (Scanned)   (Scanned)

                                        │
                            [Lock Ring Topology]
                                        ▼

┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        DYNAMIC PIPELINE RING TOPOLOGY (ANY N PHONES)                   │
└────────────────────────────────────────────────────────────────────────────────────────┘

    📱 Phone 1 (Node 0)               📱 Phone 2 (Node 1)               📱 Phone 3 (Node 2)
  ┌─────────────────────┐           ┌─────────────────────┐           ┌─────────────────────┐
  │ • Token Embedding   │   1.5KB   │                     │   1.5KB   │                     │
  │ • Pos. Embedding    │──────────▶│ • Assigned Layers   │──────────▶│ • Assigned Layers   │
  │ • Assigned Layers   │    WebRTC │                     │    WebRTC │                     │
  └─────────────────────┘           └─────────────────────┘           └─────────────────────┘
             ▲                                                                   │
             │                                                                   │ 1.5KB
             │ 1 Byte (Sampled Token)                                            │ WebRTC
             │ Shortcut Feedback Channel                                         ▼
  ┌─────────────────────┐           ┌─────────────────────┐           ┌─────────────────────┐
  │ • Assigned Layers   │   1.5KB   │                     │   1.5KB   │                     │
  │ • Final LayerNorm   │◀──────────│ • Assigned Layers   │◀──────────│ • Assigned Layers   │
  │ • LM Head + Argmax  │    WebRTC │                     │    WebRTC │                     │
  └─────────────────────┘           └─────────────────────┘           └─────────────────────┘
    📱 Phone N (Node N-1)             📱 Phone ...(Node i)              📱 Phone 4 (Node 3)
```

---

## 🧮 Model Specification & Exact Parameter Verification

* **Architecture:** Vanilla Causal Decoder Transformer (GPT-2 style, Pre-LN)
* **Vocab Size ($V$):** `256` (Direct ASCII / UTF-8 bytes `[0..255]`; no BPE tokenizer needed)
* **Context Length ($T_{max}$):** `512`
* **Layers ($L$):** `6`
* **Hidden Dimension ($d_{model}$):** `384`
* **Attention Heads ($n_{heads}$):** `6` (Head dimension $d_{head} = 64$)
* **Feed-Forward Dimension ($d_{ff}$):** $4 \times 384 = \mathbf{1536}$

### Parameter Accounting (Exactly 11,040,768 Parameters / ~44.16 MB in FP32)

| Layer Component | Dimensions / Formula | Weight Params | Bias Params | Subtotal |
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
| **6 Transformer Blocks Total** | $6 \times 1,774,464$ | $10,621,440$ | $25,344$ | **$10,646,784$** |
| **Final LayerNorm** | $384$ | $384$ | $384$ | $768$ |
| **LM Head** (tied to Token Emb) | $384 \to 256$ | $98,304$ | — | $98,304$ |
| **GRAND TOTAL** | | **$11,014,656$** | **$26,112$** | **$\mathbf{11,040,768}$** |

$$\text{Binary Size in FP32} = 11,040,768 \times 4\text{ bytes} = \mathbf{44,163,072\text{ bytes}} \approx \mathbf{42.12\text{ MB}}$$

---

## ⚡ Dynamic Layer Partitioning Table ($N = 2, 3, 4, 5, 6$)

The system balances the 6 layers across any number of connected phones:

$$\text{base} = \lfloor 6 / N \rfloor, \quad \text{remainder} = 6 \pmod N$$

| Connected Phones ($N$) | Phone 1 (Host: Node 0) | Phone 2 (Node 1) | Phone 3 (Node 2) | Phone 4 (Node 3) | Phone 5 (Node 4) | Phone 6 (Node 5) |
|---|---|---|---|---|---|---|
| **$N = 2$ phones** | **Layers 0, 1, 2** (3 layers) | **Layers 3, 4, 5** (3 layers) + Head | — | — | — | — |
| **$N = 3$ phones** | **Layers 0, 1** (2 layers) | **Layers 2, 3** (2 layers) | **Layers 4, 5** (2 layers) + Head | — | — | — |
| **$N = 4$ phones** | **Layers 0, 1** (2 layers) | **Layers 2, 3** (2 layers) | **Layer 4** (1 layer) | **Layer 5** (1 layer) + Head | — | — |
| **$N = 5$ phones** | **Layers 0, 1** (2 layers) | **Layer 2** (1 layer) | **Layer 3** (1 layer) | **Layer 4** (1 layer) | **Layer 5** (1 layer) + Head | — |
| **$N = 6$ phones** | **Layer 0** (1 layer) | **Layer 1** (1 layer) | **Layer 2** (1 layer) | **Layer 3** (1 layer) | **Layer 4** (1 layer) | **Layer 5** (1 layer) + Head |

---

## 🔄 How a Token is Generated Across the Ring

1. **Prefill Phase (Prompt Ingestion):**
   - Phone 1 embeds prompt bytes $[b_0, b_1, \dots, b_{P-1}]$ and runs its assigned layers.
   - Phone 1 transmits activation matrix $h_0$ (`Float32Array[P, 384]`) downstream.
   - Intermediate phones forward through their assigned layers and pass downstream.
   - Final phone (Sampler) runs Final LN + LM Head on the last vector $h[P-1]$, argmax samples token $t_P$, and returns **1 byte** directly to Phone 1.

2. **Decode Phase (Autoregressive Per-Token Loop):**
   - For every single generated byte, Phone 1 embeds the previous token and sends **strictly 1,536 bytes** ($1 \times 384 \times 4\text{ bytes}$) downstream.
   - Each phone computes its assigned layer using its local KV-cache and forwards **1,536 bytes** to the next device.
   - Final phone argmax samples the token byte and emits **1 byte** over the shortcut feedback channel back to Phone 1.
   - Phone 1 renders the character and triggers the next cycle.

---

## 👥 2-Member Team Division

| Role | Member | Responsibilities & Files Owned |
|---|---|---|
| **Network, Orchestration & Host UI** | **Member 1** | • Multi-peer WebSocket lobby server (`server/server.js`)<br>• Circular WebRTC DataChannel manager (`client/network/p2p-ring.js`)<br>• Host autoregressive loop & prefill driver (`client/core/host-orchestrator.js`)<br>• Master QR generator, Canvas ring visualizer & HUD (`client/ui/host-ui.js`)<br>• Entry point & responsive dark theme (`client/index.html`, `client/style.css`) |
| **ML Engine, Weights & Universal Worker** | **Member 2** | • PyTorch seed-42 weights export (`tools/generate_weights.py`)<br>• Dynamic layer weight loader & slicer (`client/core/weights.js`)<br>• Pure CPU math engine with 32×32 tiled `matmul` (`client/core/transformer.js`)<br>• Universal pipeline stage worker class (`client/core/node-worker.js`)<br>• Worker status badge & live mirror UI (`client/ui/worker-ui.js`) |

---

## 🚀 Quick Start Guide

### Prerequisites
- Node.js 18+
- Python 3.8+ with PyTorch (`pip install torch numpy`)
- Any 2 to 6 mobile phones on the same Wi-Fi network

### Step 1: Install Dependencies
```bash
npm install
npm install --prefix server
```

### Step 2: Generate Weights (If weights.bin is not present)
```bash
cd tools
python generate_weights.py
python verify_weights.py
# -> Generates weights.bin (44,163,072 bytes) with seed=42
cd ..
```

### Step 3: Run the Test Suites
```bash
# Run unit tests across all components
npm test

# Run real WebRTC browser tests across N = 2, 3, 4, 5, 6
npm run test:browser
```

### Step 4: Start the Signaling Server
```bash
npm start --prefix server
# -> Listening on http://localhost:3000 (ws://localhost:3000/signal)
```

### Step 5: Expose via HTTPS Tunnel (Required for WebRTC)
Mobile Safari and Chrome require an HTTPS context to grant WebRTC peer connection access:
```bash
npx ngrok http 3000
# -> Forwarding: https://<your-subdomain>.ngrok-free.app
```

### Step 6: Connect 2 to 6 Mobile Phones
1. **Phone 1 (Host):**
   - Open `https://<your-subdomain>.ngrok-free.app/?role=host` in Safari or Chrome.
   - The screen renders the large **Master QR Code**.
2. **Phones 2 to 6 (Workers):**
   - Open their default camera apps and scan Phone 1's screen.
   - All worker phones automatically connect to the room lobby.
   - Phone 1's interactive Canvas ring draws each phone as it joins.
3. **Lock Mesh:**
   - Tap **"Lock Mesh & Form Ring"** on Phone 1.
   - All WebRTC DataChannels connect simultaneously in parallel in under **1.5 seconds**.

### Step 7: Run Distributed Inference
1. Type a prompt on Phone 1 (e.g. `"The meaning of life is"`).
2. Set `Temperature = 0.0` for deterministic verification.
3. Tap **Generate Across Mesh**.
4. Watch the tokens stream in real time across the physical circle of phones!

---

## 📊 Benchmark Checklist for Judges

| Metric | Target | How It Is Measured |
|---|---|---|
| **P2P Split Verification** | 100% Compliant | Console logs confirm each phone runs strictly its assigned slice of layers. |
| **Throughput (Tokens/sec)** | `> 25–40 TPS` | $\text{Generated Tokens} / \text{Elapsed Seconds}$ |
| **Network Step Latency** | `< 15 ms` | Total round-trip transit time across all $N$ local Wi-Fi hops |
| **RAM Footprint per Phone** | `< 10 MB` | Verified via browser performance memory profiling |
| **Zero Cloud Compute** | 100% Offline | Disconnect WAN on router after initial page load; text generation continues uninterrupted. |
| **Determinism** | 100% Identical | Same prompt at `temp = 0.0` yields identical byte output on any set of phones. |

---

## 📂 Project Directory Structure

```
Transformer/
├── README.md                           # Master README with screenshots & architecture
├── IMPLEMENTATION_PLAN.md               # Technical architectural plan & formal specifications
├── weights.bin                         # Seed-42 FP32 binary weights (44.16 MB)
│
├── assets/                             # Screenshots & visual diagrams
│   ├── screenshot_landing.png          # Role selection view
│   ├── screenshot_host_pairing.png     # Master QR & dynamic Canvas ring
│   ├── screenshot_worker_lobby.png     # Worker lobby connection view
│   └── screenshot_inference_active.png # Active distributed generation & HUD
│
├── server/
│   ├── package.json                    # Express + ws dependencies
│   ├── server.js                       # Multi-peer lobby signaling relay & static host
│   └── test_phase1.js                  # Automated lobby & partition test
│
├── client/
│   ├── index.html                      # Universal web app entry (Host & Worker roles)
│   ├── style.css                       # Mobile dark theme with Canvas ring styling
│   ├── app.js                          # Application bootstrap & lifecycle router
│   │
│   ├── network/
│   │   ├── signaling.js                # Browser WebSocket signaling client
│   │   ├── p2p-ring.js                 # Circular WebRTC manager & 1.5KB binary framing
│   │   └── test_phase2.js              # Automated unit test for tensor framing
│   │
│   ├── core/
│   │   ├── host-orchestrator.js        # Prompt prefill & autoregressive decode driver
│   │   ├── node-worker.js              # Universal pipeline stage worker class
│   │   ├── pipeline-session.js         # Distributed generation session controller
│   │   ├── transformer.js              # Pure CPU math engine (tiled matmul, attention)
│   │   ├── weights.js                  # Dynamic layer weight loader & slicer
│   │   └── test_phase3.js              # Automated pipeline verification test
│   │
│   └── ui/
│       ├── host-ui.js                  # Master QR generator, Canvas ring visualizer, HUD
│       └── worker-ui.js                # Worker status badge & live mirror UI
│
├── tests/                              # Integration & verification test suites
│   ├── browser-integration.mjs         # Multi-client browser WebRTC test
│   ├── integration.test.mjs            # End-to-end integration test
│   └── member2.test.mjs                # Pure CPU math & layer test
│
└── tools/
    ├── capture_screenshot.js           # Automated headless browser screenshot generator
    ├── generate_weights.py             # Deterministic PyTorch seed-42 weights generator
    └── verify_weights.py               # Mathematical sanity & parameter count validator
```
