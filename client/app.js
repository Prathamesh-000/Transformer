import { SignalingClient } from './network/signaling.js';
import { DynamicP2PRingManager } from './network/p2p-ring.js';
import { HostUI } from './ui/host-ui.js';
import { WorkerUI } from './ui/worker-ui.js';
import { HostOrchestrator } from './core/host-orchestrator.js';
import { PipelineSession } from './core/pipeline-session.js';

const q = id => document.getElementById(id);
const params = new URLSearchParams(location.search);
const host = params.get('role') === 'host';
const room = (params.get('room') || '').trim().toUpperCase();
q('join-room-btn').onclick = () => {
  const code = q('manual-room-input').value.trim().toUpperCase();
  if (code) location.href = `?room=${encodeURIComponent(code)}`;
};
let wakeLock;
async function keepAwake() {
  if (document.visibilityState === 'visible' && 'wakeLock' in navigator) {
    try { wakeLock = await navigator.wakeLock.request('screen'); } catch { /* Optional browser feature. */ }
  }
}
window.addEventListener('click', keepAwake, { once: true });
document.addEventListener('visibilitychange', () => { if (wakeLock?.released) keepAwake(); });

if (host || room || params.get('role') === 'worker') {
  q('landing-view').classList.add('hidden');
  q(host ? 'host-view' : 'worker-view').classList.remove('hidden');
  const sig = new SignalingClient();
  let hostUi, workerUi, session, orchestrator, peers = [], disposed = false;
  if (host) { hostUi = new HostUI(); hostUi.init(q('host-view')); }
  else q('worker-room-badge').textContent = `Room: ${room}`;
  function fail(error) {
    error = error instanceof Error ? error : new Error(error.message || String(error));
    if (disposed) return;
    disposed = true;
    orchestrator?.abort(error); session?.fail(error); sig.close();
    if (host) { hostUi.setGenerating(false); hostUi.setReady(false, 'Mesh unavailable — reload to reconnect'); hostUi.showError(error.message); }
    else { workerUi?.setError(error); q('worker-status-text').textContent = error.message; }
  }
  sig.on('error', fail);
  sig.on('room_dissolved', error => { if (!session?.ring.isRingSealed) fail(error); });
  sig.on('close', () => {
    // Already-sealed P2P sessions survive loss of signaling/WAN.
    if (!session?.ring.isRingSealed) fail(new Error('Signaling disconnected; reload to reconnect'));
    else sig.close(); // The established mesh no longer needs a signaling connection.
  });
  sig.on('peer_joined', msg => { peers = msg.peers; hostUi?.updateLobbyPeers(peers); });
  sig.on('peer_left', msg => {
    if (session?.ring.isRingSealed) return; // DataChannel close/failure detects real peer loss.
    if (session) return fail(new Error('A phone left while the mesh was connecting'));
    peers = peers.filter(peer => peer.peerId !== msg.leftPeerId); hostUi?.updateLobbyPeers(peers);
  });
  sig.on('ring_config', config => {
    if (session || disposed) return;
    if (host) { hostUi.showInferencePanel(config); hostUi.setReady(false, 'Loading weights and connecting all phones…'); }
    else {
      q('worker-status-text').textContent = 'Loading and connecting…';
      q('worker-desc').textContent = `${config.totalNodes} phones — downstream: ${config.downstreamDeviceName}`;
      q('worker-stream').parentElement.classList.add('hidden');
      workerUi = new WorkerUI(q('worker-view'), config);
    }
    const ring = new DynamicP2PRingManager(sig, config);
    session = new PipelineSession(ring, config, { ui: workerUi, onError: fail });
    (async () => {
      await session.ready;
      if (host) {
        await session.barrier('ready'); // Every phone has loaded weights AND sealed its links.
        orchestrator = new HostOrchestrator(ring, session.worker, {
          onToken: (token, text, stats) => { hostUi.appendTokenChar(text); hostUi.updateTelemetry(stats); hostUi.pulseHost(); },
          onText: text => hostUi.appendTokenChar(text),
          onComplete: stats => { hostUi.setGenerating(false); hostUi.updateTelemetry(stats); },
          onError: error => hostUi.showError(error.message),
        }, session);
        hostUi.setReady(true, `Ready — ${config.totalNodes} phones connected`);
      } else q('worker-status-text').textContent = 'Ready';
    })().catch(fail);
  });
  if (host) {
    hostUi.onLockRing(() => {
      q('lock-ring-btn').disabled = true;
      try { sig.lockRing(); } catch (error) { fail(error); }
    });
    hostUi.onGenerate(async ({ prompt, maxTokens, temperature }) => {
      if (!orchestrator || disposed) return;
      q('output-stream').textContent = ''; hostUi.setGenerating(true);
      try { await orchestrator.startGeneration(prompt, maxTokens, temperature); }
      catch (error) { hostUi.showError(error.message); }
      finally { hostUi.setGenerating(false); }
    });
    hostUi.onStop(() => {
      orchestrator?.stop();
      // Keep Generate disabled until the in-flight token and finish barrier drain.
      q('stop-btn').textContent = 'Stopping after current step…';
    });
  }
  try {
    await sig.connect();
    if (disposed) throw new Error('Connection closed');
    const result = host ? await sig.createRoom() : await sig.joinRoom(room);
    peers = result.peers;
    if (host) { hostUi.renderMasterQR(result.room); hostUi.updateLobbyPeers(peers); }
    else q('worker-status-text').textContent = 'In lobby — waiting for host';
  } catch (error) { fail(error); }
  window.addEventListener('pagehide', () => { disposed = true; session?.ring.close(); sig.close(); workerUi?.destroy(); });
}
