# Vikrant Saini (Member 2): Implementation Walkthrough and Handoff

**Integration update:** Prathamesh Patil (Member 1) and Vikrant Saini (Member 2) are now connected by `client/app.js` and `client/core/pipeline-session.js`. The handoff discussion below describes the original module boundaries; those hooks are now wired. See [integration verification](INTEGRATION_VERIFICATION.md) for current behavior and tests.

This work implements Vikrant Saini's (Member 2) responsibilities in `IMPLEMENTATION_PLAN.md` in three phases. The original Member 2 phase left Prathamesh Patil's (Member 1) server and networking files unchanged; the later integration fixes are documented separately. Browser files are native ES modules; `client/package.json` also lets Node import them for tests.

## Phase 1 — A reproducible model

**Files:** `tools/generate_weights.py`, `tools/verify_weights.py`, `tools/requirements.txt`, root `weights.bin`, and `tests/reference.json`.

The generator uses a private CPU PyTorch generator seeded with 42. Matrix and embedding weights are normal samples with standard deviation 0.02. Biases are zero and LayerNorm scales are one. The LM head is independent of token embeddings. These choices are explicit because the plan specifies shapes and a seed but does not define initialization or weight tying.

The binary has no header: little-endian FP32 values in row-major order. Its order is token embedding, positional embedding, layers 0–5 (each in the exact `SingleLayerWeights` contract order), final LayerNorm scale/bias, then LM head. Each block has 1,774,464 parameters. Together with embeddings and the head, this gives exactly 11,040,768 parameters / 44,163,072 bytes.

**Why:** every phone must agree on what each byte means. A wrong offset can produce finite but meaningless activations, so checking the file size alone is insufficient. The verifier checks every tensor against a fresh seed-42 generation and prints a SHA256 digest. Its optional oracle computes logits with PyTorch operations, independently of the JavaScript implementation.

**Alternative:** a versioned header/manifest with tensor names and checksums would make future model changes safer. Per-layer files would simplify selective downloading. Both change the deployment format; this implementation keeps the planned single raw binary. These random weights are an inference demonstration, not a trained language model.

## Phase 2 — Selective loading and CPU inference

**Files:** `client/core/weights.js`, `client/core/transformer.js`, `client/package.json`.

The loader allocates only the requested tensors. It reads the full response as a stream and copies overlapping byte ranges, handling chunks that split FP32 values. It does not return views backed by a complete-model buffer. The existing server sends the full file despite advertising range support, so every phone currently downloads all 44 MB. Browsers without streaming temporarily require the full response in the compatibility fallback. Progress uses percentages from 0 to 100; 100 means size and selected tensors have been validated. The browser checks shape/size and finite values; seed identity is checked by the Python verifier, not cryptographically enforced by the browser.

For each layer:

1. LayerNorm normalizes each token's 384 values (epsilon 1e-5).
2. A linear projection produces queries, keys and values.
3. Six attention heads, each width 64, attend only to current and earlier positions. Scores are scaled by sqrt(64); subtracting the maximum makes softmax stable.
4. The output projection is added to the incoming residual.
5. A second LayerNorm and 384 → 1536 → 384 feed-forward network with tanh-approximate GELU produce the second residual update.

Six heads, pre-normalization, GELU approximation and epsilon are documented choices where the plan is silent. The Python oracle uses exactly these choices. Matrix storage follows PyTorch `Linear`: `[output, input]`. The CPU loops tile output rows and reduction dimensions; no WebGPU, cloud inference or runtime ML library is involved.

Each assigned layer owns separate K and V arrays for 512 positions. Prefill writes multiple positions with a causal mask. Decode appends one position and reads prior keys/values without recomputing earlier tokens. Cache length must match `startPos`; reset before every new prompt. The engine refuses empty input, non-finite activations, invalid shapes and positions beyond 511. `sequenceLength` is the minimum committed cache length across the stage's layers. Each layer's cache uses 1,572,864 bytes, in addition to weights and temporary activations.

The sampler reads the final input position, applies final LayerNorm and the 256-way LM head. At temperature zero it selects the first maximum deterministically. Positive temperatures use a stable probability distribution and `Math.random()`; this mode is intentionally not deterministic.

**Why:** caching removes redundant work during decode; explicit cache order prevents accidental reuse of another prompt. The same layer arithmetic is executed regardless of the node partition, so moving a layer between phones does not change the computation.

**Alternatives:** real byte-range responses or separate layer assets reduce download traffic. WebAssembly SIMD could accelerate CPU math; a dedicated browser Worker could keep the UI responsive. Neither is assumed here. `processStep()` is async for the contract but runs synchronous CPU arithmetic on its calling thread. Cross-browser floating-point identity and the plan's 20–40 TPS target require actual device measurements; local tests cannot establish them.

## Phase 3 — Worker, UI and integration validation

**Files:** `client/core/node-worker.js`, `client/ui/worker-ui.js`, `tests/member2.test.mjs`.

`UniversalNodeWorker` validates the node index, roles and exact balanced partition for 2–6 phones. `create(config, onProgress)` loads weights; the constructor also accepts already loaded weights, exactly as in the plan. `processStep(input, startPos, temperature)` embeds only on the host, runs all assigned layers in order, and returns either `{type: 'activation', activation}` or `{type: 'token', token, logits}`. A failure partway through a stage requires reset, because some layer caches may already have advanced.

`WorkerUI` mounts into a supplied DOM element without requiring changes to Member 1's HTML/CSS. It shows the node/layers, loading progress, processing pulse, last compute duration, step count, errors and token mirror. Text is assigned with `textContent`; token bytes are decoded with a streaming UTF-8 decoder so multi-byte characters are not split. Invalid byte sequences display standard replacement characters. `endStep()` does not append text: call `appendToken()` once per broadcast byte to avoid duplicate sampler output. `finishText()` flushes the decoder at generation end.

**Why:** the inference stage needs no WebRTC assumptions and can be tested independently. Explicit UI lifecycle hooks let Member 1 connect the same implementation to its orchestrator. Live mirroring on all phones requires a token broadcast/control path in Member 1's transport.

**Alternative:** running the engine in a dedicated Web Worker with transferred activation buffers would prevent long prefill calculations from blocking rendering. This should be added alongside the orchestrator, with explicit request IDs, reset acknowledgments and failure propagation.

### Member 1 wiring example

```js
import { UniversalNodeWorker } from './core/node-worker.js';
import { WorkerUI } from './ui/worker-ui.js';

const ui = new WorkerUI(document.querySelector('#worker'), config);
const worker = await UniversalNodeWorker.create(config, p => ui.setLoadingProgress(p));
ui.setReady();

// Register on non-host nodes once the ring is configured.
ring.onActivation(async activation => {
  try {
    ui.beginStep();
    const result = await worker.processStep(activation, worker.sequenceLength, sessionTemperature);
    ui.endStep(result);
    if (result.type === 'token') ring.sendTokenToHost(result.token);
    else ring.sendActivation(result.activation);
  } catch (error) {
    ui.setError(error);
    // Stop the generation on every peer through the session control path.
  }
});
```

The host calls `processStep(new TextEncoder().encode(prompt), 0, temperature)` for prefill and then passes `Uint8Array.of(lastToken)` at the next position for decode. For prompts split into chunks, prefill chunks advance positions; the host must disregard intermediate sampled tokens until the last prompt chunk. Each activation has `rows * 384` FP32 values, including exactly 1,536 bytes per decode activation. Only the sampled byte is transmitted on the return channel; logits stay local.

Before generation, synchronize temperature and reset every node; wait for every node's weights and channels to be ready. Allow only one pipeline step in flight. On a new prompt, peer failure or partial step error, stop and reset the whole session. Broadcast each returned token if all worker screens should mirror text. The plan's ring methods have no control-message API for these actions; Member 1 must supply it. Prompt plus processed generated tokens must fit the 512-position context. A browser's `async` callback alone is not a session queue.

### Reproduce verification

From the repository root (Python 3.12 was used):

```sh
python3 -m venv /tmp/transformer-member2-venv
/tmp/transformer-member2-venv/bin/pip install -r tools/requirements.txt
/tmp/transformer-member2-venv/bin/python tools/generate_weights.py
/tmp/transformer-member2-venv/bin/python tools/verify_weights.py --reference tests/reference.json
node --test tests/member2.test.mjs
```

The generator and verifier also work from `tools/`, as the plan requests: default paths are relative to the script's repository root, not the shell directory.

Tests check selective tensor ownership for every node of every topology, Python-logit agreement, exact local cross-partition agreement, prefill versus tokenwise/chunked inference, reset reproducibility, invalid configurations and cache positions, malformed downloads, and UTF-8 UI lifecycle behavior. The UI test uses a minimal DOM double and does not verify browser rendering. No real WebRTC or physical-phone performance is claimed.

### Physical-device acceptance (joint work with Member 1)

For each N = 2, 3, 4, 5, 6: join/lock the real lobby, confirm layer badges, wait for all downloads/channels, submit identical UTF-8 prompt bytes at temperature 0, compare generated byte IDs (not rendered glyphs), repeat after reset, and test a peer disconnect. Record prompt bytes, generated-byte count, prefill time, decode TPS, measured network round trip and device/browser versions. Test a multi-byte prompt, long prefill, context exhaustion, and WAN disconnection after initial setup. Member 1's client is now integrated and desktop WebRTC tests cover all N; this acceptance checklist still requires physical phones.
