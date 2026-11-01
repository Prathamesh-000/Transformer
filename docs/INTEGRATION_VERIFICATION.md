# Member 1 + Member 2 integration

## Phase 1: connect the planned contracts

The previous page used a mock host engine and pass-through workers. `client/app.js` now creates `UniversalNodeWorker` instances from each server-provided ring configuration. The host orchestrator calls `processStep`; workers call the same method on incoming activations. Only the sampler sends a byte on the token-return channel. `WorkerUI` is mounted and receives real loading, processing, completion, error, and token-mirror events. The host's ring canvas remains visible during generation.

**Reason:** forwarding activations alone does not execute a transformer and never invokes the LM head. Using the same worker implementation on all nodes preserves the plan's topology-independent loop over assigned layers.

**Possible improvement:** run inference inside a dedicated browser Worker so prefill does not block the page's main thread.

## Phase 2: coordinate the distributed state

`PipelineSession` adds control packets on the existing circular activation channels. A readiness packet passes each node only after local weights and channels are ready. Before each prompt, a prepare packet resets every stage's KV cache and sets temperature; it must return to the host before the host starts prefill. Token mirrors and a completion barrier travel over the same ordered ring. A worker's event queue preserves control/activation order.

The host allows one pipeline step in flight. Stop finishes that step and the completion barrier, avoiding stale tokens in the next prompt. Errors reject the generation promise instead of being hidden by a return in `finally`. Invalid context sizes are reported; prompts are not silently truncated. Byte zero is not treated as EOS because the plan does not reserve an EOS token. UTF-8 decoder flushing updates text without fabricating a token.

**Reason:** each phone has independent KV state. Resetting only the host can mix two prompts, and the sampler must actually receive the selected temperature.

**Possible improvement:** request IDs on every activation would support more advanced cancellation or pipelining. The current implementation deliberately serializes generation, matching the plan's autoregressive loop.

## Phase 3: transport and end-to-end checks

The ring uses reliable ordered channels, exact ICE routing (peer + purpose + direction), queued early candidates, connection timeouts, and close/error propagation. Long prompts are fragmented into packets of at most 16 KiB and reassembled with size/offset checks. Decode/token public interfaces are unchanged. A failed transport invalidates the session instead of continuing with partial caches.

The app survives signaling loss after the mesh is sealed, with all reset, temperature, completion and inference traffic remaining on P2P channels. Actual peer loss is detected through DataChannel/connection failure. The QR generator is bundled at `client/vendor/qrcode.js` from `qrcode-generator@1.4.4`, retaining its license header; pairing does not require a CDN or external QR endpoint.

**Possible improvement:** implement server byte ranges or per-layer assets to reduce the current full-file download per phone. Measure actual phone performance before changing the pure-JavaScript math backend.

## Verification coverage

- Python verifier: all tensors match seed 42; 11,040,768 FP32 parameters; 44,163,072 bytes. SHA256: `d209095fe23a0d5153af276de7faeaf514c2e2302834b3f67705e25fb192c251`.
- `npm test`: Member 1's three scripts and 11 Node test groups. The old orchestrator test now imports the real class instead of embedding a copy. Partition assertions cover all N; additional checks exercise real transport fragmentation, ICE routing, context limits, stop/restart, errors, session barriers and temperature propagation.
- Member 2 numerical checks: exact agreement across all local partitions; maximum logit difference from the independent PyTorch oracle is about `1.371e-6`.
- `npm run test:browser`: real Chrome WebRTC, WebSocket signaling and HTTP weight downloads for all five topologies. Checks oracle byte sequences, repeat/reset, worker mirrors, 64-byte prefill fragmentation, and inference with signaling closed.
- The same browser suite exercises the actual app bootstrap and QR/lock/generate controls, repeated generation after signaling loss, mobile-width layout, and peer-disconnect handling.

These checks do not establish physical-phone performance or Safari compatibility. The plan's final device benchmark remains a hands-on acceptance test. The desktop browser TPS values include prefill and are not phone benchmark claims.
