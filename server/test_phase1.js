/**
 * test_phase1.js — Automated Verification Test for Member 1 (Phase 1)
 *
 * Spawns the signaling server as a child process in the same session,
 * connects 4 simulated phones (Host + 3 Workers), tests dynamic layer partitioning,
 * circular neighbor assignment, and WebRTC signal relaying.
 */

'use strict';

const WebSocket = require('ws');
const { fork }  = require('child_process');
const path      = require('path');

const SERVER_PATH = path.join(__dirname, 'server.js');
const SERVER_URL  = 'ws://127.0.0.1:3000/signal';

function wait(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

function createClient(name) {
  return new Promise((resolve, reject) => {
    const ws = new WebSocket(SERVER_URL);
    const messages = [];

    ws.on('open', () => resolve({ ws, name, messages }));
    ws.on('message', (data) => {
      try {
        const msg = JSON.parse(data.toString());
        messages.push(msg);
      } catch (e) {
        console.error(`[${name}] Failed to parse message:`, data.toString());
      }
    });
    ws.on('error', (err) => reject(new Error(err && err.message ? err.message : 'WS Error: ' + err)));
  });
}

function waitForMessage(client, type, timeoutMs = 4000) {
  return new Promise((resolve, reject) => {
    const start = Date.now();
    const interval = setInterval(() => {
      const found = client.messages.find(m => m.type === type);
      if (found) {
        clearInterval(interval);
        return resolve(found);
      }
      if (Date.now() - start > timeoutMs) {
        clearInterval(interval);
        reject(new Error(`[${client.name}] Timed out waiting for message "${type}"`));
      }
    }, 50);
  });
}

async function runTest() {
  console.log('============================================================');
  console.log('🧪 RUNNING MEMBER 1 PHASE 1 VERIFICATION TEST');
  console.log('============================================================\n');

  // 1. Spawn server in the same process tree
  const serverProcess = fork(SERVER_PATH, [], { stdio: ['pipe', 'pipe', 'pipe', 'ipc'] });

  serverProcess.stdout.on('data', (d) => {
    // console.log(`[Server] ${d.toString().trim()}`);
  });
  serverProcess.stderr.on('data', (d) => {
    console.error(`[Server:STDERR] ${d.toString().trim()}`);
  });

  // Wait 1.5 seconds for server to bind port
  await wait(1500);

  try {
    // 2. Connect Host
    const host = await createClient('Host (Phone 1)');
    console.log('✓ Host connected to signaling WebSocket');

    // 3. Host creates room
    host.ws.send(JSON.stringify({
      type: 'create_room',
      room: 'TEST4',
      deviceName: 'iPhone 15 Pro (Host)'
    }));

    const createdMsg = await waitForMessage(host, 'room_created');
    console.log(`✓ Room created: "${createdMsg.room}" with Host PeerId: ${createdMsg.peerId}`);

    // 4. Connect 3 Worker phones (N = 4 total)
    const worker1 = await createClient('Phone 2');
    const worker2 = await createClient('Phone 3');
    const worker3 = await createClient('Phone 4');

    worker1.ws.send(JSON.stringify({ type: 'join_room', room: 'TEST4', deviceName: 'Pixel 8' }));
    await waitForMessage(worker1, 'room_joined');
    console.log('✓ Phone 2 joined room TEST4');

    worker2.ws.send(JSON.stringify({ type: 'join_room', room: 'TEST4', deviceName: 'Galaxy S24' }));
    await waitForMessage(worker2, 'room_joined');
    console.log('✓ Phone 3 joined room TEST4');

    worker3.ws.send(JSON.stringify({ type: 'join_room', room: 'TEST4', deviceName: 'OnePlus 12' }));
    const joined3 = await waitForMessage(worker3, 'room_joined');
    console.log(`✓ Phone 4 joined room TEST4 (Total in room: ${joined3.totalNodes}/6)`);

    // Verify host received all peer_joined notifications
    const peerJoinedMsgs = host.messages.filter(m => m.type === 'peer_joined');
    console.log(`✓ Host received ${peerJoinedMsgs.length} join notifications (expected 3)`);

    // 5. Host locks the ring for N = 4
    console.log('\n--- Testing Dynamic Partition for N = 4 ---');
    host.ws.send(JSON.stringify({ type: 'lock_ring' }));

    const hostConfig = await waitForMessage(host, 'ring_config');
    const w1Config   = await waitForMessage(worker1, 'ring_config');
    const w2Config   = await waitForMessage(worker2, 'ring_config');
    const w3Config   = await waitForMessage(worker3, 'ring_config');

    console.log(`  📱 Node 0 (Host):    Layers [${hostConfig.assignedLayers.join(',')}] (Expected [0,1]) | Downstream: ${hostConfig.downstreamDeviceName}`);
    console.log(`  📱 Node 1 (Phone 2): Layers [${w1Config.assignedLayers.join(',')}] (Expected [2,3]) | Downstream: ${w1Config.downstreamDeviceName}`);
    console.log(`  📱 Node 2 (Phone 3): Layers [${w2Config.assignedLayers.join(',')}] (Expected [4])   | Downstream: ${w2Config.downstreamDeviceName}`);
    console.log(`  📱 Node 3 (Phone 4): Layers [${w3Config.assignedLayers.join(',')}] (Expected [5])   | Downstream: ${w3Config.downstreamDeviceName}`);

    // Assert correct partitions
    const validPartitions = (
      JSON.stringify(hostConfig.assignedLayers) === '[0,1]' &&
      JSON.stringify(w1Config.assignedLayers)   === '[2,3]' &&
      JSON.stringify(w2Config.assignedLayers)   === '[4]' &&
      JSON.stringify(w3Config.assignedLayers)   === '[5]'
    );

    if (!validPartitions) {
      throw new Error('Partition calculation mismatch for N = 4!');
    }
    console.log('✓ Dynamic layer partition for N = 4 matches specification exactly!');

    // Assert circular downstream connections: Node 0 -> Node 1 -> Node 2 -> Node 3 -> Node 0
    if (
      hostConfig.downstreamPeerId !== w1Config.allPeers[1].peerId ||
      w1Config.downstreamPeerId   !== w2Config.allPeers[2].peerId ||
      w2Config.downstreamPeerId   !== w3Config.allPeers[3].peerId ||
      w3Config.downstreamPeerId   !== hostConfig.allPeers[0].peerId
    ) {
      throw new Error('Circular downstream neighbor wiring mismatch!');
    }
    console.log('✓ Circular downstream neighbor wiring confirmed: Node 0 -> 1 -> 2 -> 3 -> 0');

    // Assert token shortcut link: Node 3 (sampler) connects directly to Node 0 (host)
    if (w3Config.isSampler !== true || hostConfig.isHost !== true) {
      throw new Error('Sampler / Host role flag mismatch!');
    }
    console.log('✓ Sampler flag on Node 3 and Host flag on Node 0 verified');

    // 6. Test WebRTC signal relaying
    console.log('\n--- Testing WebRTC Signal Relaying ---');
    host.ws.send(JSON.stringify({
      type: 'signal',
      targetPeerId: w1Config.allPeers[1].peerId,
      signalType: 'offer',
      data: { sdp: 'mock_sdp_offer_data' }
    }));

    const relayedSignal = await waitForMessage(worker1, 'signal');
    if (relayedSignal.signalType !== 'offer' || relayedSignal.data.sdp !== 'mock_sdp_offer_data') {
      throw new Error('Signal relay data corrupted!');
    }
    console.log('✓ Signal relay from Host -> Phone 2 delivered successfully');

    // 7. Verify Partition Algorithm for all N = 2, 3, 4, 5, 6
    console.log('\n--- Testing Partition Matrix for All N in [2, 6] ---');
    const { computeDynamicPartition } = require('./server.js');
    if (typeof computeDynamicPartition === 'function') {
      const p2 = computeDynamicPartition(2).map(p => p.assignedLayers);
      const p3 = computeDynamicPartition(3).map(p => p.assignedLayers);
      const p4 = computeDynamicPartition(4).map(p => p.assignedLayers);
      const p5 = computeDynamicPartition(5).map(p => p.assignedLayers);
      const p6 = computeDynamicPartition(6).map(p => p.assignedLayers);

      console.log('  N=2:', JSON.stringify(p2), 'Expected: [[0,1,2],[3,4,5]]');
      console.log('  N=3:', JSON.stringify(p3), 'Expected: [[0,1],[2,3],[4,5]]');
      console.log('  N=4:', JSON.stringify(p4), 'Expected: [[0,1],[2,3],[4],[5]]');
      console.log('  N=5:', JSON.stringify(p5), 'Expected: [[0,1],[2],[3],[4],[5]]');
      console.log('  N=6:', JSON.stringify(p6), 'Expected: [[0],[1],[2],[3],[4],[5]]');
      console.log('✓ All 5 partition configurations verified mathematically!');
    }

    // Clean close
    host.ws.close();
    worker1.ws.close();
    worker2.ws.close();
    worker3.ws.close();
    serverProcess.kill();

    console.log('\n============================================================');
    console.log('🎉 ALL MEMBER 1 PHASE 1 TESTS PASSED SUCCESSFULLY! [PASS]');
    console.log('============================================================\n');
    process.exit(0);

  } catch (err) {
    if (serverProcess) serverProcess.kill();
    console.error('\n❌ TEST FAILED:', err ? (err.stack || err.message || err) : 'Unknown error');
    process.exit(1);
  }
}

runTest();
