/**
 * server.js — Dynamic N-Phone Mesh Signaling Server & Static Host
 *
 * Responsibilities for Prathamesh Patil (Member 1) (Phase 1):
 *  1. Serve client/ static assets to mobile browsers
 *  2. Serve weights.bin with proper binary MIME & range-caching headers
 *  3. WebSocket signaling endpoint at /signal:
 *     - Multi-client room lobby (any N between 2 and 6 phones)
 *     - Dynamic layer partitioning & ring neighbor configuration
 *     - WebRTC SDP Offer / Answer and ICE candidate relaying
 *     - Peer lifecycle and disconnect management
 */

'use strict';

const express   = require('express');
const http      = require('http');
const WebSocket = require('ws');
const path      = require('path');
const fs        = require('fs');

// ── Configuration ────────────────────────────────────────────────────────────
const PORT         = process.env.PORT || 3000;
const CLIENT_DIR   = path.join(__dirname, '..', 'client');
const WEIGHTS_PATH = path.join(__dirname, '..', 'weights.bin');

const app = express();

// ── Static Routes ────────────────────────────────────────────────────────────

// Serve weights.bin directly from root with range and streaming support
app.get('/weights.bin', (req, res) => {
  fs.stat(WEIGHTS_PATH, (err, stat) => {
    if (err) {
      console.warn('[Server] weights.bin not found on disk yet (Vikrant Saini deliverable).');
      return res.status(404).send('weights.bin not found on server.');
    }
    res.setHeader('Content-Type', 'application/octet-stream');
    res.setHeader('Content-Length', stat.size);
    res.setHeader('Cache-Control', 'public, max-age=86400');
    res.setHeader('Accept-Ranges', 'bytes');
    console.log(`[Server:Weights] Streaming ${(stat.size / (1024 * 1024)).toFixed(2)} MB -> ${req.ip}`);
    fs.createReadStream(WEIGHTS_PATH).pipe(res);
  });
});

// Serve client assets if client/ directory exists
if (fs.existsSync(CLIENT_DIR)) {
  app.use(express.static(CLIENT_DIR));
}

// Fallback route for SPA parameter preservation (?role=host, ?room=...)
app.get('*', (req, res) => {
  const indexPath = path.join(CLIENT_DIR, 'index.html');
  if (fs.existsSync(indexPath)) {
    res.sendFile(indexPath);
  } else {
    res.send(`
      <!DOCTYPE html>
      <html>
      <head><title>P2P Mesh Signaling Server</title></head>
      <body style="font-family: sans-serif; background: #1a1a1a; color: #eee; padding: 20px;">
        <h2>🌐 P2P Mobile Mesh Signaling Server is Active</h2>
        <p>Listening on port: <strong>${PORT}</strong></p>
        <p>WebSocket Signaling Endpoint: <code>ws://${req.headers.host}/signal</code></p>
        <p>Status: Prathamesh Patil (Member 1) Phase 1 Signaling Layer Ready ✅</p>
      </body>
      </html>
    `);
  }
});

// ── HTTP & WebSocket Server Setup ────────────────────────────────────────────
const httpServer = http.createServer(app);
const wss = new WebSocket.Server({ server: httpServer, path: '/signal' });

// ── Dynamic Ring Partitioning Algorithm (N in [2, 6]) ─────────────────────────
/**
 * Partition 6 transformer layers across N connected phones
 * @param {number} totalNodes - Number of phones in the ring (2 to 6)
 * @returns {Array<{nodeIndex: number, assignedLayers: number[], isHost: boolean, isSampler: boolean}>}
 */
function computeDynamicPartition(totalNodes) {
  if (!Number.isInteger(totalNodes) || totalNodes < 2 || totalNodes > 6) throw new RangeError('Expected 2 to 6 nodes');
  const TOTAL_LAYERS = 6;
  const base = Math.floor(TOTAL_LAYERS / totalNodes);
  const remainder = TOTAL_LAYERS % totalNodes;

  const partitions = [];
  let currentLayer = 0;

  for (let i = 0; i < totalNodes; i++) {
    // Distribute remainder layers among the first 'remainder' devices
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

// ── Room State Management ────────────────────────────────────────────────────
// Map<roomId, RoomState>
// RoomState: {
//   roomId: string,
//   hostPeerId: string,
//   isLocked: boolean,
//   peers: Array<{ peerId: string, deviceName: string, ws: WebSocket, joinedAt: number }>
// }
const rooms = new Map();

function generateRoomId() {
  const chars = 'ABCDEFGHJKLMNPQRSTUVWXYZ23456789';
  let id = '';
  for (let i = 0; i < 6; i++) {
    id += chars.charAt(Math.floor(Math.random() * chars.length));
  }
  return id;
}

function sendJson(ws, obj) {
  if (ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify(obj));
  }
}

// ── WebSocket Connection & Message Protocol ──────────────────────────────────
wss.on('connection', (ws, req) => {
  let myPeerId = null;
  let myRoomId = null;

  console.log(`[WS:Connect] Client connected from ${req.socket.remoteAddress}`);

  ws.on('message', (raw) => {
    let msg;
    try {
      msg = JSON.parse(raw);
    } catch (e) {
      console.error('[WS:Error] Invalid JSON:', raw.toString());
      return sendJson(ws, { type: 'error', message: 'Malformed JSON payload' });
    }

    switch (msg.type) {
      // ──────────────────────────────────────────────────────────────────────
      // 1. Host creates a new room lobby
      // ──────────────────────────────────────────────────────────────────────
      case 'create_room': {
        if (myRoomId) return sendJson(ws, { type: 'error', message: 'Already in a room.' });
        const roomId = (msg.room && msg.room.trim().toUpperCase()) || generateRoomId();
        myPeerId = 'peer_' + Math.random().toString(36).substring(2, 9);
        myRoomId = roomId;

        if (rooms.has(roomId)) {
          return sendJson(ws, { type: 'error', message: `Room "${roomId}" already exists.` });
        }

        const roomState = {
          roomId: roomId,
          hostPeerId: myPeerId,
          isLocked: false,
          peers: [
            {
              peerId: myPeerId,
              deviceName: msg.deviceName || 'Host Phone (Phone 1)',
              ws: ws,
              joinedAt: Date.now()
            }
          ]
        };

        rooms.set(roomId, roomState);
        console.log(`[Room:Created] Room "${roomId}" created by Host [${myPeerId}]`);

        sendJson(ws, {
          type: 'room_created',
          room: roomId,
          peerId: myPeerId,
          isHost: true,
          totalNodes: 1,
          peers: [{ peerId: myPeerId, deviceName: msg.deviceName || 'Host Phone (Phone 1)' }]
        });
        break;
      }

      // ──────────────────────────────────────────────────────────────────────
      // 2. Worker phone scans Master QR and joins the room lobby
      // ──────────────────────────────────────────────────────────────────────
      case 'join_room': {
        if (myRoomId) return sendJson(ws, { type: 'error', message: 'Already in a room.' });
        const roomId = (msg.room || '').trim().toUpperCase();
        if (!roomId || !rooms.has(roomId)) {
          return sendJson(ws, { type: 'error', message: `Room "${roomId}" does not exist.` });
        }

        const room = rooms.get(roomId);

        if (room.isLocked) {
          return sendJson(ws, { type: 'error', message: `Room "${roomId}" is already locked in an active ring.` });
        }

        if (room.peers.length >= 6) {
          return sendJson(ws, { type: 'error', message: `Room "${roomId}" is at maximum capacity (6 phones).` });
        }

        myPeerId = 'peer_' + Math.random().toString(36).substring(2, 9);
        myRoomId = roomId;

        const peerRecord = {
          peerId: myPeerId,
          deviceName: msg.deviceName || `Worker Phone ${room.peers.length + 1}`,
          ws: ws,
          joinedAt: Date.now()
        };

        room.peers.push(peerRecord);
        console.log(`[Room:Join] Peer [${myPeerId}] (${peerRecord.deviceName}) joined room "${roomId}" (Total: ${room.peers.length}/6)`);

        const peerSummary = room.peers.map(p => ({ peerId: p.peerId, deviceName: p.deviceName }));

        // Send confirmation to the joining worker
        sendJson(ws, {
          type: 'room_joined',
          room: roomId,
          peerId: myPeerId,
          isHost: false,
          totalNodes: room.peers.length,
          peers: peerSummary
        });

        // Notify all other members in the room (especially the host) of the new peer
        room.peers.forEach(p => {
          if (p.peerId !== myPeerId) {
            sendJson(p.ws, {
              type: 'peer_joined',
              room: roomId,
              newPeer: { peerId: myPeerId, deviceName: peerRecord.deviceName },
              totalNodes: room.peers.length,
              peers: peerSummary
            });
          }
        });
        break;
      }

      // ──────────────────────────────────────────────────────────────────────
      // 3. Host locks the ring and initiates dynamic partition
      // ──────────────────────────────────────────────────────────────────────
      case 'lock_ring': {
        const roomId = myRoomId;
        if (!roomId || !rooms.has(roomId)) {
          return sendJson(ws, { type: 'error', message: 'Room not found.' });
        }

        const room = rooms.get(roomId);
        if (room.hostPeerId !== myPeerId) {
          return sendJson(ws, { type: 'error', message: 'Only the Host can lock the ring topology.' });
        }

        if (room.isLocked) return sendJson(ws, { type: 'error', message: 'Ring is already locked.' });
        const N = room.peers.length;
        if (N < 2) {
          return sendJson(ws, { type: 'error', message: 'At least 2 phones are required to form a mesh.' });
        }

        room.isLocked = true;
        console.log(`\n============================================================`);
        console.log(`  🔒 SEALING RING: ${N} PHONES CONNECTED IN ROOM "${roomId}"`);
        console.log(`============================================================`);

        // Compute layer split for N phones
        const partitions = computeDynamicPartition(N);

        // Host is Node 0, sampler is Node N - 1
        const hostPeer = room.peers[0];
        const samplerPeer = room.peers[N - 1];

        // Send tailored ring_config to each peer
        for (let i = 0; i < N; i++) {
          const peer = room.peers[i];
          const partition = partitions[i];

          // Circular downstream peer: (i + 1) % N
          const nextPeerIndex = (i + 1) % N;
          const nextPeer = room.peers[nextPeerIndex];

          // Upstream peer: (i - 1 + N) % N
          const prevPeerIndex = (i - 1 + N) % N;
          const prevPeer = room.peers[prevPeerIndex];

          const config = {
            type: 'ring_config',
            room: roomId,
            nodeIndex: i,
            totalNodes: N,
            assignedLayers: partition.assignedLayers,
            isHost: partition.isHost,
            isSampler: partition.isSampler,
            // Downstream target (Node i connects to Node i+1)
            downstreamPeerId: nextPeer.peerId,
            downstreamDeviceName: nextPeer.deviceName,
            // Upstream neighbor
            upstreamPeerId: prevPeer.peerId,
            // Token shortcut connection endpoints
            hostPeerId: hostPeer.peerId,
            samplerPeerId: samplerPeer.peerId,
            // Full roster for verification
            allPeers: room.peers.map((p, idx) => ({
              nodeIndex: idx,
              peerId: p.peerId,
              deviceName: p.deviceName,
              assignedLayers: partitions[idx].assignedLayers
            }))
          };

          console.log(`  📱 Node ${i} (${peer.deviceName}): Layers [${partition.assignedLayers.join(', ')}] -> Downstream: Node ${nextPeerIndex} (${nextPeer.deviceName})`);
          sendJson(peer.ws, config);
        }
        console.log(`============================================================\n`);
        break;
      }

      // ──────────────────────────────────────────────────────────────────────
      // 4. Targeted WebRTC Signaling Relay (SDP Offer / Answer / ICE)
      // ──────────────────────────────────────────────────────────────────────
      case 'signal': {
        const { targetPeerId, signalType, data } = msg;
        if (!myRoomId || !rooms.has(myRoomId)) return;

        const room = rooms.get(myRoomId);
        const target = room.peers.find(p => p.peerId === targetPeerId);

        if (target && target.ws.readyState === WebSocket.OPEN) {
          sendJson(target.ws, {
            type: 'signal',
            fromPeerId: myPeerId,
            signalType: signalType, // 'offer' | 'answer' | 'ice'
            data: data
          });
        } else {
          console.warn(`[WS:Signal] Target peer [${targetPeerId}] not found or unreachable.`);
        }
        break;
      }

      // ──────────────────────────────────────────────────────────────────────
      // 5. Keepalive Heartbeat
      // ──────────────────────────────────────────────────────────────────────
      case 'ping': {
        sendJson(ws, { type: 'pong', timestamp: Date.now() });
        break;
      }

      default:
        console.warn('[WS:Warn] Unrecognized message type:', msg.type);
    }
  });

  // ── Connection Teardown & Room Cleanup ────────────────────────────────────
  ws.on('close', () => {
    if (!myRoomId || !rooms.has(myRoomId)) return;

    const room = rooms.get(myRoomId);
    const peerIndex = room.peers.findIndex(p => p.peerId === myPeerId);

    if (peerIndex !== -1) {
      const removed = room.peers.splice(peerIndex, 1)[0];
      console.log(`[Room:Leave] Peer [${myPeerId}] (${removed.deviceName}) disconnected from room "${myRoomId}".`);

      // If the host left or room is now empty, dissolve room
      if (removed.peerId === room.hostPeerId || room.peers.length === 0) {
        console.log(`[Room:Dissolved] Host disconnected or room empty. Closing room "${myRoomId}".`);
        room.peers.forEach(p => {
          sendJson(p.ws, { type: 'room_dissolved', message: 'Host disconnected. Mesh session ended.' });
          try { p.ws.close(); } catch (_) {}
        });
        rooms.delete(myRoomId);
      } else {
        // Notify remaining peers of node dropout
        room.peers.forEach(p => {
          sendJson(p.ws, {
            type: 'peer_left',
            leftPeerId: myPeerId,
            remainingNodes: room.peers.length
          });
        });
      }
    }
  });

  ws.on('error', (err) => {
    console.error(`[WS:Error] Socket error on peer [${myPeerId}]:`, err.message);
  });
});

// ── Server Start ─────────────────────────────────────────────────────────────
if (require.main === module) {
  httpServer.listen(PORT, () => {
    console.log(`
============================================================
  P2P Mobile Mesh Inference — Dynamic Signaling Server
============================================================
  Port:                 ${PORT}
  HTTP endpoint:        http://localhost:${PORT}
  WebSocket signaling:  ws://localhost:${PORT}/signal
  Weights status:       ${fs.existsSync(WEIGHTS_PATH) ? 'Found (44.16 MB) ✓' : 'Not generated yet (Pending Vikrant Saini)'}
  Supported Nodes:      Dynamic (2 to 6 phones)
============================================================
  Ready for Phase 1 testing!
`);
  });
}

module.exports = { computeDynamicPartition, app, httpServer, wss };
