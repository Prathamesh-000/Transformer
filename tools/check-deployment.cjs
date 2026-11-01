'use strict';

// Fail the hosting build if the checkout is missing the local integration/assets.
const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const root = path.resolve(__dirname, '..');
for (const file of [
  'client/index.html', 'client/app.js', 'client/style.css',
  'client/core/weights.js', 'client/core/transformer.js',
  'client/core/node-worker.js', 'client/core/pipeline-session.js',
  'client/core/host-orchestrator.js', 'client/network/signaling.js',
  'client/network/p2p-ring.js', 'client/ui/host-ui.js',
  'client/ui/worker-ui.js', 'client/vendor/qrcode.js',
  'server/server.js', 'server/package.json', 'server/package-lock.json',
]) {
  if (!fs.statSync(path.join(root, file)).isFile()) throw new Error(`Missing deployment file: ${file}`);
}
const weights = fs.readFileSync(path.join(root, 'weights.bin'));
if (weights.length !== 44163072) throw new Error('weights.bin must contain 44,163,072 bytes');
const hash = crypto.createHash('sha256').update(weights).digest('hex');
// This is our verified seed-42 checkpoint, not an organizer-provided checksum.
if (hash !== 'd209095fe23a0d5153af276de7faeaf514c2e2302834b3f67705e25fb192c251') {
  throw new Error('weights.bin differs from the locally verified checkpoint');
}
console.log('Deployment files and verified seed-42 weights are present.');
