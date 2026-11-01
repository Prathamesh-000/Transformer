# P2P Mobile Mesh Inference: Fully Dynamic N-Phone Pipeline Ring (N = 2 to 6)

> **Architecture:** Dynamic Heterogeneous Pipeline Parallelism over an Arbitrary P2P WebRTC Ring  
> **Supported Topology:** Any $N$ Phones where $N \in [2, 6]$ (2, 3, 4, 5, or 6 smartphones)  
> **Pairing Method:** Method 1 — Single Master QR Code Lobby (Simultaneous Dynamic Onboarding)  
> **Target Devices:** Stock Consumer Smartphones (iOS Safari / Android Chrome) on Web CPU (JavaScript)  
> **Model:** 6-Layer Byte-Level Micro-Transformer (11,040,768 parameters, PyTorch seed 42)  

---

## 1. Executive Summary: The Universal N-Phone Mesh

This implementation plan provides a **fully dynamic, adaptive pipeline ring architecture** that operates with **any arbitrary number of phones from $N = 2$ up to $N = 6$**.

You do not need to hardcode the system for a specific device count. Whether the judges hand you **2 phones, 3 phones, 4 phones, 5 phones, or 6 phones**, the system behaves identically:
1. **Phone 1 (Host)** opens the web app and displays a **Single Master QR Code**.
2. **Any number of phones ($N-1$ worker devices)** scan Phone 1's screen simultaneously to join the lobby.
3. The Host screen shows the dynamic roster: `[Connected: 4 Phones (iPhone 14, Pixel 7, Galaxy S23, OnePlus 11)]`.
4. The Host taps **"Form Ring & Lock Topology"**.
5. The **Dynamic Layer Partitioner** automatically calculates the optimal layer split across all $N$ connected devices, instructs each phone to allocate only its assigned weights, and establishes the circular WebRTC DataChannel ring in parallel.

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        DYNAMIC PIPELINE RING TOPOLOGY (ANY N PHONES)                   │
└────────────────────────────────────────────────────────────────────────────────────────┘

          Phone 1 (Node 0)                     Phone 2 (Node 1)
       ┌────────────────────┐               ┌────────────────────┐
       │ • Token Embedding  │     1.5 KB    │                    │
       │ • Pos. Embedding   │──────────────▶│ • Assigned Layers  │
       │ • Assigned Layers  │    WebRTC     │                    │
       └────────────────────┘               └────────────────────┘
                  ▲                                    │
                  │                                    │ 1.5 KB
                  │ 1 Byte (Sampled Token)             │ WebRTC
                  │ Direct Shortcut Return             ▼
       ┌────────────────────┐               ┌────────────────────┐
       │ • Assigned Layers  │     1.5 KB    │                    │
       │ • Final LayerNorm  │◀──────────────│ • Phone ...(Node i)│
       │ • LM Head + Argmax │    WebRTC     │                    │
       └────────────────────┘               └────────────────────┘
          Phone N (Node N-1)                   Phone 3 (Node 2)
```

---

## 2. Dynamic Layer Partitioning Algorithm (N = 2, 3, 4, 5, 6)

The Micro-Transformer has **6 identical Transformer blocks** (Layers 0 through 5). The partitioner dynamically distributes these 6 layers across any $N$ connected phones using a balanced modulo-chunking formula:

$$\text{base} = \lfloor 6 / N \rfloor, \quad \text{remainder} = 6 \pmod N$$

The first $\text{remainder}$ devices receive $(\text{base} + 1)$ layers, and the remaining devices receive $\text{base}$ layers.

```javascript
/**
 * Computes layer assignment for any N phones (N between 2 and 6)
 * @param {number} totalNodes - Number of connected phones (e.g. 2, 3, 4, 5, 6)
 * @returns {Array<{nodeIndex: number, assignedLayers: number[], isHost: boolean, isSampler: boolean}>}
 */
function computeDynamicPartition(totalNodes) {
  const TOTAL_LAYERS = 6;
  const base = Math.floor(TOTAL_LAYERS / totalNodes);
  const remainder = TOTAL_LAYERS % totalNodes;
  
  const partitions = [];
  let currentLayer = 0;
  
  for (let i = 0; i < totalNodes; i++) {
    const layerCount = i < remainder ? base + 1 : base;
    const assignedLayers = [];
    for (let l = 0; l < layerCount; l++) {
      assignedLayers.push(currentLayer++);
    }
    
    partitions.push({
      nodeIndex: i,
      assignedLayers: assignedLayers,
      isHost: (i === 0),
      isSampler: (i === totalNodes - 1)
    });
  }
  
  return partitions;
}
```

### Complete Layer Allocation Table for Every $N$:

| Connected Phones ($N$) | Node 0 (Phone 1: Host) | Node 1 (Phone 2) | Node 2 (Phone 3) | Node 3 (Phone 4) | Node 4 (Phone 5) | Node 5 (Phone 6) |
|---|---|---|---|---|---|---|
| **$N = 2$ phones** | **Layers 0, 1, 2** (3 layers) | **Layers 3, 4, 5** (3 layers) + LM Head | — | — | — | — |
| **$N = 3$ phones** | **Layers 0, 1** (2 layers) | **Layers 2, 3** (2 layers) | **Layers 4, 5** (2 layers) + LM Head | — | — | — |
| **$N = 4$ phones** | **Layers 0, 1** (2 layers) | **Layers 2, 3** (2 layers) | **Layer 4** (1 layer) | **Layer 5** (1 layer) + LM Head | — | — |
| **$N = 5$ phones** | **Layers 0, 1** (2 layers) | **Layer 2** (1 layer) | **Layer 3** (1 layer) | **Layer 4** (1 layer) | **Layer 5** (1 layer) + LM Head | — |
| **$N = 6$ phones** | **Layer 0** (1 layer) | **Layer 1** (1 layer) | **Layer 2** (1 layer) | **Layer 3** (1 layer) | **Layer 4** (1 layer) | **Layer 5** (1 layer) + LM Head |

---

## 3. Memory Footprint Scaling Per Phone

Because weights are loaded dynamically based on each phone's `assignedLayers`, the memory allocated in mobile browser RAM scales down dramatically as more devices are added:

| Connected Phones ($N$) | Max Layers on Any Single Phone | Active Weight Parameters per Phone | RAM Footprint in Browser |
|---|---|---|---|
| **$N = 2$ phones** | 3 layers | $\sim 5,520,000$ params | **$\approx 22.1\text{ MB}$** |
| **$N = 3$ phones** | 2 layers | $\sim 3,745,000$ params | **$\approx 15.0\text{ MB}$** |
| **$N = 4$ phones** | 2 layers (Nodes 0,1) / 1 layer (Nodes 2,3) | $\sim 1,774,000\text{ to }3,745,000$ params | **$\approx 7.1\text{ to }15.0\text{ MB}$** |
| **$N = 5$ phones** | 2 layers (Node 0) / 1 layer (Nodes 1–4) | $\sim 1,774,000\text{ to }3,745,000$ params | **$\approx 7.1\text{ to }15.0\text{ MB}$** |
| **$N = 6$ phones** | 1 layer per phone | $\sim 1,774,000$ params | **$\approx 7.1\text{ MB}$ (Lowest RAM)** |

---

## 4. Method 1 Dynamic Pairing Flow: Single Master QR

```
                              📱 Phone 1 (Host)
                            [Displays Master QR]
                                      │
               ┌──────────────┬───────┴───────┬──────────────┐
               ▼              ▼               ▼              ▼
           📱 Phone 2     📱 Phone 3      📱 Phone 4     📱 Phone 5 ...
           (Scanned)      (Scanned)       (Scanned)      (Scanned)
               │              │               │              │
               └──────────────┼───────────────┴──────────────┘
                              ▼
                      Signaling Lobby
                              │
               Host hits: "LOCK MESH (4 PHONES)"
                              │
               Partitioner: Assigns Dynamic Slices
                              ▼
        P1 [L0-1] ──▶ P2 [L2-3] ──▶ P3 [L4] ──▶ P4 [L5 + Head]
          ▲                                           │
          └───────────── 1-Byte Return ───────────────┘
```

### The 4-Step User Journey:
1. **Host Launches Lobby:**
   - Phone 1 navigates to `https://app.url/?role=host`.
   - Phone 1 registers room `MESH88` on the local WebSocket server and displays a large **Master QR Code**.
2. **Workers Join Dynamically:**
   - Any number of phones ($N-1$ phones, from 1 to 5 devices) open their native camera apps and scan Phone 1's screen.
   - Each worker connects to room `MESH88`.
   - Phone 1's display updates in real time with an animated visual ring showing connected devices:
     - *"2 Devices Connected: Host + Pixel 7"*
     - *"3 Devices Connected: Host + Pixel 7 + iPhone 14"*
     - *"4 Devices Connected: Host + Pixel 7 + iPhone 14 + Galaxy S22"*
3. **Lock Ring & Distribute Partitions:**
   - Whenever the user is ready, the Host taps **"Lock Mesh & Start"** (works for any count $N \ge 2$).
   - The signaling server runs `computeDynamicPartition(N)` and delivers each phone its unique role:
     - `nodeIndex`: Its position in the circle ($0 \dots N-1$).
     - `assignedLayers`: Array of layer IDs it will execute (e.g. `[0, 1]` or `[4]`).
     - `nextPeer`: Target peer to establish downstream DataChannel.
     - `isHost` / `isSampler`: Role flags.
4. **Parallel WebRTC Connect:**
   - All $N$ DataChannels connect in parallel in under 1.5 seconds.
   - Each phone's UI instantly displays its dynamic role badge (e.g., *"Node 2 of 4 — Computing Layer 4"*).

---

## 5. Generic Node Pipeline Worker (Runs on Any Phone)

To support arbitrary layer counts without code changes, the worker pipeline is completely generic:

```javascript
class UniversalNodeWorker {
  constructor(weights, config) {
    this.assignedLayers = config.assignedLayers; // e.g. [0, 1] or [2] or [4, 5]
    this.isHost         = config.isHost;
    this.isSampler      = config.isSampler;
    this.transformer    = new TransformerDynamicBlock(weights, this.assignedLayers);
  }

  /**
   * Universal forward pass: iterates sequentially through ALL assigned layers
   * @param {Float32Array|Uint8Array} input - Prompt/token (Host) or activation (Worker)
   * @param {number} startPos - Sequence position for KV-cache
   * @param {number} temperature - Sampling temperature
   */
  async processStep(input, startPos, temperature = 0.0) {
    let x;
    
    // 1. If Host: embed token(s) first
    if (this.isHost) {
      x = this.transformer.embed(input, startPos);
    } else {
      x = input; // Raw Float32Array activation from upstream peer
    }

    // 2. Sequentially forward through ALL layers assigned to this phone
    for (const layerIdx of this.assignedLayers) {
      x = this.transformer.forwardLayer(x, layerIdx, startPos);
    }

    // 3. If Final Sampler Node: run Final LN + LM Head + Argmax
    if (this.isSampler) {
      const { token, logits } = this.transformer.sampleNextToken(x, temperature);
      return { type: 'token', token, logits };
    }

    // 4. Intermediate Node: return activation tensor for downstream peer
    return { type: 'activation', activation: x };
  }
}
```

### Why this design is so robust:
- **Zero code branching based on $N$:** The worker doesn't care whether it is running 1 layer, 2 layers, or 3 layers. It simply loops through `this.assignedLayers`.
- **Constant Network Payload:** The activation tensor $x$ exiting any layer has shape `[1, 384]` (384 floats = 1,536 bytes during decode). Whether a phone computes 1 layer or 3 layers, the data it sends to the next phone is **always exactly 1,536 bytes**.

---

## 6. Forward Pass Execution Flow (Prefill vs. Decode)

```
========================================================================================
PREFILL PHASE (Initial Prompt: length P tokens)
========================================================================================
Node 0 (Host):
  1. Embeds prompt tokens P at indices 0..P-1.
  2. Forwards through its assigned layers (e.g. Layers 0, 1). Caches K, V.
  3. Sends activation h_out [P, 384] to Node 1.

Node 1 .. Node N-2 (Workers):
  1. Receives [P, 384] from upstream neighbor.
  2. Forwards through assigned layers. Caches K, V.
  3. Sends activation h_out [P, 384] to downstream neighbor.

Node N-1 (Final Sampler Node):
  1. Receives [P, 384] from upstream neighbor.
  2. Forwards through assigned layers. Caches K, V.
  3. Takes vector at final position [P-1], applies Final LN + LM Head -> logits[256].
  4. Argmax samples first token byte t_P.
  5. Sends t_P (1 byte) directly back to Node 0 across shortcut return channel.

========================================================================================
DECODE PHASE (Autoregressive Per-Token Loop: length 1 token)
========================================================================================
For each step t:
  Node 0:
    1. Embeds previous token byte at index startPos.
    2. Runs assigned layers (Query attends over cached K, V).
    3. Sends h_out: Float32Array[1, 384] (1,536 bytes) to Node 1.

  Node 1 .. Node N-2:
    1. Receives 1,536 bytes.
    2. Runs assigned layers (Query attends over cached K, V).
    3. Sends h_out: Float32Array[1, 384] (1,536 bytes) to downstream neighbor.

  Node N-1:
    1. Receives 1,536 bytes.
    2. Runs assigned layers (Query attends over cached K, V).
    3. Final LN + LM Head -> logits[256].
    4. Argmax samples next token byte.
    5. Transmits 1 byte directly back to Node 0.

  Node 0:
    1. Receives token byte -> renders character to UI.
    2. Appends to context -> immediately initiates step t+1.
```

---

## 7. Team Division: 2 Members Implementing the Dynamic N-Phone Mesh

The division of labor remains clean, structured, and modular:

```
┌────────────────────────────────────────────────────────────────────────┐
│               MEMBER 1: NETWORK, ORCHESTRATION & HOST UI               │
├────────────────────────────────┬───────────────────────────────────────┤
│ Responsibilities:              │ Deliverables:                         │
│ • Dynamic WebSocket Lobby      │ 1. server/server.js                   │
│ • N-Peer WebRTC Ring Manager   │ 2. client/network/signaling.js        │
│ • Host Pipeline Orchestrator   │ 3. client/network/p2p-ring.js         │
│ • Dynamic Master QR UI         │ 4. client/core/host-orchestrator.js   │
│ • Live Visual Mesh Ring Canvas │ 5. client/ui/host-ui.js               │
│ • Benchmark Telemetry HUD      │ 6. client/index.html & style.css      │
└────────────────────────────────┴───────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────────┐
│             MEMBER 2: ML, INFERENCE ENGINE & PIPELINE WORKER           │
├────────────────────────────────┬───────────────────────────────────────┤
│ Responsibilities:              │ Deliverables:                         │
│ • PyTorch Seed-42 Weight Gen   │ 1. tools/generate_weights.py          │
│ • Dynamic Sliced Weight Loader │ 2. tools/verify_weights.py            │
│ • Pure CPU Math Engine         │ 3. client/core/weights.js             │
│ • Multi-Layer KV-Cache System  │ 4. client/core/transformer.js         │
│ • Universal Node Worker Class  │ 5. client/core/node-worker.js         │
│ • Worker UI & Live Status      │ 6. client/ui/worker-ui.js             │
└────────────────────────────────┴───────────────────────────────────────┘
```

---

## 8. Exact Interface Contracts Between Member 1 & Member 2

### Interface 1: Dynamic Ring Configuration (`client/network/p2p-ring.js`)
*Produced by Member 1 $\to$ Delivered to Member 2's `node-worker.js`*
```typescript
interface DynamicRingConfig {
  nodeIndex: number;         // 0 to N-1
  totalNodes: number;        // e.g. 2, 3, 4, 5, or 6
  assignedLayers: number[];  // Dynamic list of layer indices, e.g. [0, 1] or [3]
  isHost: boolean;           // true if nodeIndex === 0
  isSampler: boolean;        // true if nodeIndex === totalNodes - 1
  downstreamPeerId: string;  // Target peer for intermediate activations
  hostPeerId: string;        // Target peer for token feedback
}

class DynamicP2PRingManager {
  constructor(signaling: SignalingClient, config: DynamicRingConfig);
  initiateConnections(): Promise<void>; // Sets up downstream and shortcut channels
  
  // Transmit intermediate activation to downstream peer:
  sendActivation(data: Float32Array): void;
  onActivation(callback: (data: Float32Array) => void): void;
  
  // Transmit sampled token from sampler to host:
  sendTokenToHost(token: number): void;
  onTokenFromSampler(callback: (token: number) => void): void;
  
  get isRingSealed(): boolean;
}
```

### Interface 2: Dynamic Sliced Weight Loader (`client/core/weights.js`)
*Produced by Member 2 $\to$ Consumed by `node-worker.js`*
```typescript
interface SingleLayerWeights {
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

interface DynamicNodeWeights {
  tokenEmb?: Float32Array;            // [256, 384] (Only on Node 0)
  posEmb?:   Float32Array;            // [512, 384] (Only on Node 0)
  layers:    Map<number, SingleLayerWeights>; // Layer ID -> weights
  finalLnW?: Float32Array;            // [384] (Only on Node N-1)
  finalLnB?: Float32Array;            // [384] (Only on Node N-1)
  lmHeadW?:  Float32Array;            // [256, 384] (Only on Node N-1)
}

// Loads weights.bin and dynamically extracts only the requested layer indices:
function loadDynamicNodeWeights(
  assignedLayers: number[], 
  isHost: boolean, 
  isSampler: boolean,
  onProgress?: (pct: number) => void
): Promise<DynamicNodeWeights>;
```

### Interface 3: Pure CPU Transformer Engine (`client/core/transformer.js`)
*Produced by Member 2 $\to$ Consumed by `node-worker.js`*
```typescript
class TransformerDynamicBlock {
  constructor(weights: DynamicNodeWeights, assignedLayers: number[]);
  
  // Embeds tokens (Host only):
  embed(tokenIds: Uint8Array | number[], startPos: number): Float32Array;
  
  // Computes a single transformer layer forward pass with its dedicated KV-cache:
  forwardLayer(x: Float32Array, layerIdx: number, startPos: number): Float32Array;
  
  // Computes Final LN + LM Head + argmax (Sampler only):
  sampleNextToken(h: Float32Array, temperature?: number): {
    token: number;
    logits: Float32Array;
  };

  reset(): void; // Clears KV-cache for all assigned layers
  get sequenceLength(): number;
}
```

---

## 9. 5-Phase Implementation Roadmap

```
PHASE 1: Foundation, Weight Generation & Dynamic Lobby
  ├─ Member 2: Generate seed-42 weights.bin & verify parameter layout
  └─ Member 1: Build Node.js WebSocket signaling server with dynamic room lobby

PHASE 2: Ring P2P Transport & Pure CPU Math Engine
  ├─ Member 2: Build weights.js (dynamic slicing) & transformer.js (tiled matmul)
  └─ Member 1: Build p2p-ring.js (arbitrary N-peer circular WebRTC setup)

PHASE 3: Universal Pipeline Worker & Host Orchestrator
  ├─ Member 2: Build node-worker.js (processes arbitrary assignedLayers array)
  └─ Member 1: Build host-orchestrator.js (prompt prefill & token decode loop)

PHASE 4: Master QR UI & Dynamic Animated Ring Visualizer
  ├─ Member 2: Build worker-ui.js (dynamic layer badge, live token mirror)
  └─ Member 1: Build host-ui.js (Master QR generator, dynamic ring canvas)

PHASE 5: Multi-Device Benchmarking & Validation Routine
  └─ Both: Test across 2, 3, 4, 5, and 6 phones, verify deterministic outputs
```

---

## 10. Expected Project Directory Structure

```
Transformer/
├── IMPLEMENTATION_PLAN.md      # This master document
├── README.md                   # Quickstart & architecture overview
├── weights.bin                 # Seed-42 FP32 binary weights (44,163,072 bytes)
│
├── server/
│   ├── package.json            # Express + ws dependencies
│   └── server.js               # Multi-peer lobby signaling relay & static host
│
├── client/
│   ├── index.html              # Universal entry (handles Host lobby vs Worker node)
│   ├── style.css               # Responsive dark theme with ring canvas animation
│   │
│   ├── core/
│   │   ├── weights.js          # Dynamic Float32Array weight slicer per node
│   │   ├── transformer.js      # Pure CPU math engine (tiled matmul, attention, KV-cache)
│   │   ├── host-orchestrator.js# Phone 1 autoregressive pipeline driver
│   │   └── node-worker.js      # Universal pipeline stage worker (Nodes 1 to N-1)
│   │
│   ├── network/
│   │   ├── signaling.js        # Multi-client WebSocket signaling client
│   │   └── p2p-ring.js         # Circular WebRTC DataChannel ring manager (any N)
│   │
│   └── ui/
│       ├── host-ui.js          # Master QR generator, dynamic ring canvas, prompt input
│       └── worker-ui.js        # Worker status badge, ring pulse animator, token mirror
│
└── tools/
    ├── generate_weights.py     # Deterministic PyTorch seed-42 weights generator
    └── verify_weights.py       # Mathematical sanity & parameter count validator
```

---

## 11. Testing & Demonstration Guide for Judges

### Step 1: Pre-generate & Validate Weights
```bash
cd tools
python generate_weights.py
python verify_weights.py
```
*Verification criteria:* File size must equal `44,163,072` bytes; total parameters must equal `11,040,768`.

### Step 2: Start the Signaling Server & Tunnel
```bash
cd ../server
npm install
node server.js
```
In a second terminal window:
```bash
npx ngrok http 3000
```
Copy the secure HTTPS URL: `https://<subdomain>.ngrok-free.app`.

### Step 3: Run the Dynamic Demonstration (Any $N$ Phones)
1. **On Phone 1 (Host):**
   - Open `https://<subdomain>.ngrok-free.app/?role=host`.
   - The screen renders the large **Master QR Code**.
2. **On Worker Phones (Any number: 1, 2, 3, 4, or 5 other phones):**
   - Open native camera apps and scan Phone 1's screen.
   - Watch Phone 1's dynamic ring canvas update in real time as each phone pops into the circle.
3. **Lock the Mesh:**
   - Tap **"Lock Mesh & Start"** on Phone 1.
   - The partitioner instantly distributes the 6 layers across the connected devices:
     - If 2 phones joined: 3 layers each.
     - If 3 phones joined: 2 layers each.
     - If 4 phones joined: 2 layers on first two phones, 1 layer on the last two.
     - If 5 phones joined: 2 layers on host, 1 layer on each of the 4 workers.
     - If 6 phones joined: 1 layer on every phone.
4. **Generate & Benchmark:**
   - Input prompt on Phone 1 (e.g., `"The nature of intelligence is"`).
   - Set `temperature = 0.0`.
   - Tap **Generate Text**.
   - Watch the generation pulse synchronously through all screens in the circle over local Wi-Fi.

---

## 12. Benchmark Metrics Checklist for Judges

| Metric | Target | How It Is Measured |
|---|---|---|
| **Dynamic N-Phone Support** | Validated | Works out of the box with 2, 3, 4, 5, or 6 phones without configuration changes. |
| **Throughput (Tokens/sec)** | `> 20–40 TPS` | $\text{Generated Tokens} / \text{Elapsed Seconds}$ |
| **Network Step Latency** | `< 15 ms` | Total round-trip transit time across all $N$ hops over local Wi-Fi |
| **Zero Cloud Compute** | 100% Offline | Disconnect WAN on router after initial load; inference continues seamlessly across local Wi-Fi. |
| **Determinism** | 100% Identical | At `temp = 0.0`, any combination of phones generates the exact same deterministic byte sequence. |
