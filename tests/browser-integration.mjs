/** Real Chromium WebRTC + HTTP weight loading + PyTorch-oracle end-to-end checks. */
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { createRequire } from 'node:module';
import { chromium } from 'playwright';
const require = createRequire(import.meta.url);
const { httpServer, wss } = require('../server/server.js');
const oracle = JSON.parse(await readFile(new URL('./reference.json', import.meta.url), 'utf8'));
await new Promise((resolve, reject) => { httpServer.once('error', reject); httpServer.listen(0, '127.0.0.1', resolve); });
const origin = `http://127.0.0.1:${httpServer.address().port}`;
let browser;
try {
  browser = await chromium.launch({ channel: process.env.BROWSER_CHANNEL || 'chrome', headless: true });
  for (let n = 2; n <= 6; n++) {
    const context = await browser.newContext();
    // The inference transport itself has no external HTTP dependency.
    await context.route('https://cdn.jsdelivr.net/**', route => route.abort());
    const pages = [], errors = [];
    try {
      for (let i = 0; i < n; i++) {
        const page = await context.newPage(); pages.push(page);
        page.on('pageerror', error => errors.push(error.message));
        await page.goto(origin);
        await page.evaluate(async ({ i, n }) => {
          const { SignalingClient } = await import('/network/signaling.js');
          const { DynamicP2PRingManager } = await import('/network/p2p-ring.js');
          const { PipelineSession } = await import('/core/pipeline-session.js');
          const { HostOrchestrator } = await import('/core/host-orchestrator.js');
          const { WorkerUI } = await import('/ui/worker-ui.js');
          const sig = new SignalingClient();
          const mesh = window.mesh = { sig, tokens: [], errors: [], states: [] };
          mesh.ready = new Promise((resolve, reject) => {
            sig.on('ring_config', config => {
              mesh.config = config;
              const ring = mesh.ring = new DynamicP2PRingManager(sig, config);
              const root = document.body.appendChild(document.createElement('div'));
              const ui = mesh.ui = config.isHost ? null : new WorkerUI(root, config);
              const session = mesh.session = new PipelineSession(ring, config, { ui, onError(error) { mesh.errors.push(error.message); mesh.host?.abort(error); reject(error); } });
              session.ready.then(async () => {
                if (config.isHost) {
                  await session.barrier('ready');
                  mesh.host = new HostOrchestrator(ring, session.worker, { onToken(token) { mesh.tokens.push(token); } }, session);
                }
                resolve(true);
              }).catch(reject);
            });
          });
          mesh.ready.catch(error => mesh.errors.push(error.message));
          await sig.connect();
          if (i === 0) await sig.createRoom(`TEST${n}`); else await sig.joinRoom(`TEST${n}`);
        }, { i, n });
      }
      await pages[0].evaluate(() => mesh.sig.lockRing());
      await Promise.all(pages.map(page => page.evaluate(() => mesh.ready)));
      const generate = async prompt => pages[0].evaluate(async prompt => {
        mesh.tokens = [];
        const stats = await mesh.host.startGeneration(Uint8Array.from(prompt), 4, 0);
        return { tokens: mesh.tokens, stats };
      }, prompt);
      const first = await generate(oracle.prompt);
      assert.deepEqual(first.tokens, oracle.steps.map(step => step.token));
      const repeat = await generate(oracle.prompt); assert.deepEqual(repeat.tokens, first.tokens);
      for (const page of pages.slice(1)) {
        const state = await page.evaluate(() => ({ length: mesh.session.worker.sequenceLength, mirror: mesh.ui.output.textContent, errors: mesh.errors }));
        assert.equal(state.length, oracle.prompt.length + 3);
        assert.equal(state.mirror, new TextDecoder().decode(Uint8Array.from(first.tokens)));
        assert.deepEqual(state.errors, []);
      }
      // Prefill larger than a DataChannel's conservative message limit.
      await generate(Array.from({ length: 64 }, (_, i) => 65 + i % 26));
      // All session controls and inference must still run with signaling unavailable.
      await Promise.all(pages.map(page => page.evaluate(() => mesh.sig.close())));
      const offline = await generate(oracle.prompt); assert.deepEqual(offline.tokens, first.tokens);
      assert.deepEqual(errors, []);
      console.log(`PASS real browser N=${n}: PyTorch tokens, repeat/reset, worker mirrors, 64-byte prefill, signaling offline (${first.stats.currentTps.toFixed(1)} local TPS)`);
    } finally { await context.close(); }
  }
  // Exercise the actual app bootstrap and buttons, not just imported modules.
  const context = await browser.newContext({ viewport: { width: 390, height: 844 } });
  try {
    const host = await context.newPage(), worker = await context.newPage();
    const errors = [];
    for (const page of [host, worker]) page.on('pageerror', error => errors.push(error.message));
    await host.goto(`${origin}/?role=host`);
    await host.waitForFunction(() => document.querySelector('#room-code-display')?.textContent !== '------');
    await host.locator('#qr-display svg').waitFor();
    const room = await host.locator('#room-code-display').textContent();
    await worker.goto(`${origin}/?room=${room}`);
    await host.waitForFunction(() => !document.querySelector('#lock-ring-btn').disabled);
    await host.locator('#lock-ring-btn').click();
    await host.waitForFunction(() => document.querySelector('#ring-status-badge')?.textContent.startsWith('Ready'), null, { timeout: 120000 });
    await host.locator('#prompt-input').fill(new TextDecoder().decode(Uint8Array.from(oracle.prompt)));
    await host.locator('#max-tokens-input').fill('4');
    await host.locator('#generate-btn').click();
    await worker.waitForFunction(() => [...document.querySelectorAll('[role="status"]')].some(el => el.textContent.startsWith('Generation complete')), null, { timeout: 120000 });
    await host.waitForFunction(() => !document.querySelector('#generate-btn').classList.contains('hidden'));
    assert.equal(await host.locator('#output-stream').textContent(), new TextDecoder().decode(Uint8Array.from(oracle.steps.map(step => step.token))));
    assert.equal(await worker.locator('pre[aria-label="Generated text mirror"]').textContent(), await host.locator('#output-stream').textContent());
    for (const page of [host, worker]) assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), 'Mobile layout must not overflow horizontally');
    await host.screenshot({ path: '/private/tmp/transformer-integration-host.png', fullPage: true });
    await worker.screenshot({ path: '/private/tmp/transformer-integration-worker.png', fullPage: true });
    // Drop the actual app's signaling sockets while retaining local WebRTC.
    for (const client of wss.clients) client.terminate();
    await host.waitForTimeout(300);
    await host.locator('#generate-btn').click();
    await host.waitForFunction(() => !document.querySelector('#generate-btn').classList.contains('hidden'));
    assert.equal(await host.locator('#output-stream').textContent(), new TextDecoder().decode(Uint8Array.from(oracle.steps.map(step => step.token))));
    await worker.close();
    await host.waitForFunction(() => document.querySelector('#generate-btn').disabled);
    assert.deepEqual(errors, []);
    console.log('PASS actual app: QR lobby, lock, readiness, generation, repeat, worker mirror, peer-disconnect handling');
  } finally { await context.close(); }
} finally {
  await browser?.close();
  for (const client of wss.clients) client.terminate();
  await new Promise(resolve => wss.close(resolve));
  await new Promise(resolve => httpServer.close(resolve));
}
