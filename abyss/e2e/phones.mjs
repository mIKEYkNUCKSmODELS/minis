// End-to-end proof of the Phase 2 slice: three phones join one hunt by code and play it to the end
// through the real UI against the real server. The Abyss AI plays the Revenant and the other Hunter seats.
//
//   npm run build && node e2e/phones.mjs            (screenshots go to e2e/shots/)
import { spawn } from 'node:child_process';
import { mkdirSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const shots = join(root, 'e2e/shots');
mkdirSync(shots, { recursive: true });
const PORT = 2599;
const base = `http://localhost:${PORT}`;
const exe = process.env.CHROME ?? '/opt/pw-browsers/chromium-1194/chrome-linux/chrome';

const server = spawn(process.execPath, [join(root, 'node_modules/tsx/dist/cli.mjs'), 'packages/server/src/main.ts'], { cwd: root, env: { ...process.env, PORT: String(PORT), ABYSS_BOT_DELAY_MS: process.env.ABYSS_BOT_DELAY_MS ?? '250' }, stdio: ['ignore', 'pipe', 'inherit'] });
await new Promise((res, rej) => {
  server.stdout.on('data', d => { if (String(d).includes('listening')) res(); });
  server.on('exit', c => rej(new Error(`server exited ${c}`)));
});

const browser = await chromium.launch({ executablePath: exe });
const phone = () => browser.newContext({ viewport: { width: 390, height: 844 }, deviceScaleFactor: 2, isMobile: true, hasTouch: true });
const t0 = Date.now();
let failures = 0;
const fail = m => { failures++; console.error('FAIL', m); };

try {
  const [ca, cb, cc] = await Promise.all([phone(), phone(), phone()]);
  const [A, B, C] = await Promise.all([ca.newPage(), cb.newPage(), cc.newPage()]);
  for (const [p, n] of [[A, 'A'], [B, 'B'], [C, 'C']]) p.on('pageerror', e => fail(`${n} page error: ${e.message}`));

  await A.goto(base);
  await A.screenshot({ path: join(shots, '01_home.png') });
  await A.fill('#name', 'Mike');
  await A.click('#create');
  await A.waitForSelector('.code');
  const code = (await A.textContent('.code')).trim();
  console.log('hunt code', code);

  for (const [p, name] of [[B, 'Rook'], [C, 'Vee']]) {
    await p.goto(`${base}/?code=${code}`);
    await p.fill('#name', name);
    await p.click('#join-link');
    await p.waitForSelector('.code');
  }
  const sit = async (p, seat, cls) => {
    await p.click(`[data-seat="${seat}"]`);
    await p.waitForSelector('[data-cls]');
    await p.click(`[data-cls="${cls}"]`);
    await p.click('[data-vote="fight"]');
  };
  await sit(A, 1, 'arcblade');
  await sit(B, 2, 'lifebinder');
  await sit(C, 3, 'chronoStalker');
  await A.waitForTimeout(300);
  await A.screenshot({ path: join(shots, '02_lobby.png'), fullPage: true });
  await A.click('#start');
  await Promise.all([A, B, C].map(p => p.waitForSelector('#board canvas')));
  const tStart = Date.now();

  // One step of a simple human-ish player, through the UI only.
  const step = async p => {
    const s = await p.evaluate(() => {
      const w = window.__abyss;
      if (!w) return null;
      const { packet, mode } = w;
      const v = packet.view, me = v.hunters.find(h => h.seat === packet.you);
      const myTurn = v.phase === 'hunters' && me && me.actionsLeft > 0;
      return { over: v.phase === 'over', myTurn, me, monster: v.monster.pos, signals: v.signals, N: v.N, cells: mode?.cells?.map(c => c.pos) ?? [] };
    });
    if (!s || s.over || !s.myTurn) return s;
    const button = name => p.locator('#actions button', { hasText: name }).first();
    for (const name of ['Revive', 'Cut free', 'Heal', 'Attack', 'Scan']) {
      const b = button(name);
      if (await b.count() && await b.isEnabled()) { await b.click(); return s; }
    }
    // Move toward the Monster (or the nearest signal) by tapping a highlighted square on the canvas.
    const goal = s.monster ?? s.signals.reduce((b, q) => (!b || Math.max(Math.abs(q.x - s.me.pos.x), Math.abs(q.y - s.me.pos.y)) < Math.max(Math.abs(b.x - s.me.pos.x), Math.abs(b.y - s.me.pos.y)) ? q : b), null);
    if (goal && s.cells.length) {
      const best = s.cells.reduce((b, q) => (Math.max(Math.abs(q.x - goal.x), Math.abs(q.y - goal.y)) < Math.max(Math.abs(b.x - goal.x), Math.abs(b.y - goal.y)) ? q : b));
      const box = await p.locator('#board canvas').boundingBox();
      const cell = Math.min(box.width, box.height) / s.N;
      await p.touchscreen.tap(box.x + (best.x + 0.5) * cell, box.y + (best.y + 0.5) * cell);
      return s;
    }
    await button('End turn').click();
    return s;
  };

  let reloaded = false, midShot = false, steps = 0, lastSig = '', sameFor = 0;
  for (;;) {
    if (Date.now() - tStart > 8 * 60_000) { fail('hunt did not finish in 8 minutes'); break; }
    const states = [];
    for (const p of [A, B, C]) states.push(await step(p).catch(e => (fail(e.message), null)));
    steps++;
    if (states.every(s => s?.over)) break;
    const sig = await A.evaluate(() => { const v = window.__abyss?.packet.view; return v ? JSON.stringify([v.round, v.phase, v.hunters.map(h => [h.seat, h.actionsLeft, h.pos]), v.monster.actionsLeft, v.monster.hp]) : ''; });
    sameFor = sig === lastSig ? sameFor + 1 : 0; lastSig = sig;
    if (sameFor === 40) {
      for (const [p, n] of [[A, 'A'], [B, 'B'], [C, 'C']]) {
        const d = await p.evaluate(() => { const w = window.__abyss; const v = w?.packet.view; return v && { you: w.packet.you, phase: v.phase, round: v.round, hunters: v.hunters.map(h => ({ s: h.seat, a: h.actionsLeft, down: h.down, c: h.cocoon, pos: h.pos })), m: v.monster, mode: w.mode, banner: document.getElementById('banner')?.textContent, toast: document.querySelector('.toast')?.textContent, buttons: [...document.querySelectorAll('#actions button')].map(b => b.textContent + (b.disabled ? '(x)' : '')) }; });
        console.log('STALL', n, JSON.stringify(d));
      }
      fail('stalled'); break;
    }
    const anyRevealed = states.some(s => s && s.monster);
    if (!midShot && anyRevealed) { midShot = true; await A.screenshot({ path: join(shots, '03_game_revealed.png') }); }
    if (!midShot && steps === 25) await A.screenshot({ path: join(shots, '03_game_hunting.png') });
    // Prove a dropped phone gets its seat back: reload Rook's page mid-hunt.
    if (!reloaded && steps === 15) {
      reloaded = true;
      await B.reload();
      await B.waitForSelector('#board canvas', { timeout: 15000 });
      const you = await B.evaluate(() => window.__abyss?.packet.you);
      if (you !== 2) fail(`reload did not restore the seat (you=${you})`); else console.log('reload: seat 2 restored');
    }
    await A.waitForTimeout(150);
  }
  const end = await A.evaluate(() => { const v = window.__abyss.packet.view; return { result: v.result, round: v.round, reveal: !!window.__abyss.packet.reveal }; });
  await A.waitForTimeout(600);
  await A.screenshot({ path: join(shots, '04_over.png'), fullPage: true });
  await C.screenshot({ path: join(shots, '05_over_other_phone.png'), fullPage: true });
  console.log(`hunt over: ${end.result} after ${end.round} rounds; reveal=${end.reveal}; ${((Date.now() - tStart) / 1000).toFixed(0)}s of play`);
  if (!end.reveal) fail('no end-of-hunt reveal');
} catch (e) {
  fail(e.stack ?? e.message);
} finally {
  await browser.close();
  server.kill();
}
console.log(failures ? `E2E FAILED (${failures})` : `E2E PASSED in ${((Date.now() - t0) / 1000).toFixed(0)}s`);
process.exit(failures ? 1 : 0);
