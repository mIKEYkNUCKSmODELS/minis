import './style.css';
import {
  CLASSES, CLASS_IDS, MONSTERS, dist,
  type Action, type ClassId, type GameEvent, type HunterView, type Objective, type Pos, type SeatView,
} from '@abyss/engine';
import { BoardView } from './board';
import { createHunt, joinHunt, resumeHunt, type GamePacket, type Handlers, type LobbySnapshot, type Net } from './net';
import { esc, narrate } from './narrate';
import { CLASS_COLOR, CLASS_GLYPH, css } from './theme';

const app = document.getElementById('app')!;
const NAME_KEY = 'abyss.name';

// ------------------------------------------------------------------ state
let net: Net | null = null;
let lobby: LobbySnapshot | null = null;
let packet: GamePacket | null = null;
let board: BoardView | null = null;
let boardEl: HTMLElement | null = null;
let log: string[] = [];
let mode: Mode = { kind: 'idle' };
let wasMyTurn = false;
let status: 'online' | 'reconnecting' | 'gone' = 'online';

type Target = { pos: Pos; action: Action };
type Mode =
  | { kind: 'idle' }
  | { kind: 'cells'; label: string; cells: Target[] }   // move-like: tap a highlighted square
  | { kind: 'targets'; label: string; cells: Target[] }; // pick a target

const h = (html: string) => { const t = document.createElement('template'); t.innerHTML = html.trim(); return t.content.firstElementChild as HTMLElement; };
const myName = () => { try { return localStorage.getItem(NAME_KEY) ?? ''; } catch { return ''; } };
const saveName = (n: string) => { try { localStorage.setItem(NAME_KEY, n); } catch { /* */ } };

function toast(msg: string) {
  document.querySelectorAll('.toast').forEach(t => t.remove());
  const t = h(`<div class="toast" role="alert">${esc(msg)}</div>`);
  document.body.appendChild(t);
  setTimeout(() => t.remove(), 2600);
}

const handlers: Handlers = {
  lobby(l) { lobby = l; if (l.phase === 'lobby') render(); },
  game(p) { onPacket(p); },
  error(m) { toast(m); },
  status(s) { status = s; renderNet(); if (s === 'gone') { net = null; packet = null; lobby = null; render(); } },
};

// ------------------------------------------------------------------ screens
function render() {
  if (!net || !lobby) return renderHome();
  if (lobby.phase === 'lobby') return renderLobby();
  renderGame();
}

function renderHome() {
  board = null; boardEl = null;
  const code = new URLSearchParams(location.search).get('code')?.toUpperCase() ?? '';
  app.replaceChildren(h(`
    <main class="home">
      <div class="title"><h1>ABYSS</h1><p>Hunt the monster together. Or be the monster.</p></div>
      <div class="card col">
        <label class="small muted" for="name">Your name</label>
        <input id="name" maxlength="16" autocomplete="nickname" placeholder="Hunter" value="${esc(myName())}" />
      </div>
      ${code ? `<button class="primary" id="join-link">Join hunt ${esc(code)}</button><div class="divider">or</div>` : ''}
      <button class="${code ? '' : 'primary'}" id="create">Start a new hunt</button>
      <div class="divider">have a code?</div>
      <div class="row"><input id="code" class="grow" maxlength="5" autocapitalize="characters" placeholder="ABCDE" value="${esc(code)}" />
        <button id="join">Join</button></div>
      <p class="small muted" style="text-align:center">Empty seats are played by the Abyss AI. 1 to 6 people.</p>
    </main>`));
  const name = () => { const n = (document.getElementById('name') as HTMLInputElement).value.trim() || 'Hunter'; saveName(n); return n; };
  const go = async (fn: () => Promise<Net>) => {
    app.querySelectorAll('button').forEach(b => (b.disabled = true));
    try { net = await fn(); status = 'online'; }
    catch (e) { toast(`Could not connect: ${(e as Error).message || 'hunt not found'}`); renderHome(); }
  };
  document.getElementById('create')!.onclick = () => go(() => createHunt(name(), handlers));
  const join = (c: string) => go(() => joinHunt(c, name(), handlers));
  document.getElementById('join')!.onclick = () => join((document.getElementById('code') as HTMLInputElement).value);
  document.getElementById('join-link')?.addEventListener('click', () => join(code));
}

function renderLobby() {
  const l = lobby!;
  const me = net!.room.sessionId;
  const mySeat = l.seats.find(s => s.sessionId === me);
  const host = l.hostSessionId === me;
  const M = MONSTERS[l.monster];
  const link = `${location.origin}/?code=${l.code}`;
  const seatRow = (i: number) => {
    const s = l.seats[i];
    const mine = s.sessionId === me;
    const isM = i === 0;
    const badge = isM ? `<span class="badge" style="background:#b3253a;color:#fff">M</span>`
      : `<span class="badge" style="background:${s.cls ? css(CLASS_COLOR[s.cls]) : '#2c2340'}">${s.cls ? CLASS_GLYPH[s.cls] : i}</span>`;
    const who = s.kind === 'human' ? esc(s.name) + (s.connected ? '' : ' <span class="pill">offline</span>') : '<span class="muted">Open</span>';
    const sub = isM ? M.name : s.cls ? CLASSES[s.cls].name : 'Hunter';
    const tag = s.kind === 'human' ? (mine ? '<span class="pill">you</span>' : '') : '<span class="pill ai">Abyss AI</span>';
    return `<button class="seat ${mine ? 'me' : ''} ${isM ? 'monster' : ''}" data-seat="${i}" ${s.kind === 'human' && !mine ? 'disabled' : ''}>
      ${badge}<span class="grow"><div class="who">${who}</div><div class="small muted">${sub}</div></span>${tag}</button>`;
  };
  const classPicker = mySeat && mySeat.seat > 0 ? `
    <div class="card col">
      <div class="small muted">Your Hunter</div>
      <div class="classes">${CLASS_IDS.map(c => `<button data-cls="${c}" class="${mySeat.cls === c ? 'on' : ''}">
        <span class="badge" style="width:30px;height:30px;background:${css(CLASS_COLOR[c])}">${CLASS_GLYPH[c]}</span>${CLASSES[c].name.split(' ')[0]}</button>`).join('')}</div>
      <div class="small">${mySeat.cls ? esc(CLASSES[mySeat.cls].role) : ''}</div>
      <div class="row small"><span class="muted">Timeline</span>
        <div class="seg grow">${[0, 1, 2].map(t => `<button data-tl="${t}" class="${mySeat.look.timeline === t ? 'on' : ''}">${['I', 'II', 'III'][t]}</button>`).join('')}</div>
        <div class="seg">${(['M', 'F'] as const).map(x => `<button data-sex="${x}" class="${mySeat.look.sex === x ? 'on' : ''}">${x}</button>`).join('')}</div></div>
      <div class="small muted">Squad vote</div>
      <div class="seg">
        <button data-vote="fight" class="${mySeat.vote === 'fight' ? 'on' : ''}"><b>Fight</b><br><span class="small muted">Kill it. It hits harder.</span></button>
        <button data-vote="flight" class="${mySeat.vote === 'flight' ? 'on' : ''}"><b>Flight</b><br><span class="small muted">Cleanse 3 Hearts or survive 10 rounds.</span></button>
      </div>
    </div>` : mySeat ? `<div class="card small"><b>You are the ${M.name}.</b> ${esc(M.ability)} Hunt them before the Abyss swallows the board.</div>` : '';
  app.replaceChildren(h(`
    <main class="col" style="padding-bottom:24px">
      <div class="top"><div><div class="small muted">Hunt code</div><div class="code">${l.code}</div></div>
        <button id="share">Invite</button></div>
      <div class="card small"><b>${M.name}</b> · ${esc(M.ability)}</div>
      ${host ? `<div class="row small"><span class="muted grow">Hunters at the table</span><div class="seg">${[3, 4, 5].map(n => `<button data-squad="${n}" class="${l.squad === n ? 'on' : ''}">${n}</button>`).join('')}</div></div>` : ''}
      <div class="seats">${Array.from({ length: l.squad + 1 }, (_, i) => seatRow(i)).join('')}</div>
      ${classPicker}
      ${host ? `<button class="primary" id="start" ${mySeat ? '' : 'disabled'}>${mySeat ? 'Begin the hunt' : 'Take a seat to begin'}</button>`
        : `<p class="muted small" style="text-align:center">Waiting for the host to begin…</p>`}
      ${mySeat ? `<button class="ghost small" id="stand">Leave seat</button>` : `<p class="muted small" style="text-align:center">Tap a seat to sit. Tap the red seat to play the Monster.</p>`}
    </main>`));
  const send = net!.send;
  app.querySelectorAll<HTMLButtonElement>('[data-seat]').forEach(b => (b.onclick = () => send('sit', { seat: Number(b.dataset.seat) })));
  app.querySelectorAll<HTMLButtonElement>('[data-cls]').forEach(b => (b.onclick = () => send('class', { cls: b.dataset.cls })));
  app.querySelectorAll<HTMLButtonElement>('[data-tl]').forEach(b => (b.onclick = () => send('class', { look: { timeline: Number(b.dataset.tl) } })));
  app.querySelectorAll<HTMLButtonElement>('[data-sex]').forEach(b => (b.onclick = () => send('class', { look: { sex: b.dataset.sex } })));
  app.querySelectorAll<HTMLButtonElement>('[data-vote]').forEach(b => (b.onclick = () => send('vote', { objective: b.dataset.vote as Objective })));
  app.querySelectorAll<HTMLButtonElement>('[data-squad]').forEach(b => (b.onclick = () => send('squad', { squad: Number(b.dataset.squad) })));
  document.getElementById('start')?.addEventListener('click', () => send('start'));
  document.getElementById('stand')?.addEventListener('click', () => send('stand'));
  document.getElementById('share')!.onclick = async () => {
    try {
      if (navigator.share) await navigator.share({ title: 'ABYSS', text: `Join my ABYSS hunt: ${l.code}`, url: link });
      else { await navigator.clipboard.writeText(link); toast('Invite link copied.'); }
    } catch { /* cancelled */ }
  };
}

// ------------------------------------------------------------------ game
function names() {
  return new Map(packet!.seats.map(s => [s.seat, s.kind === 'bot' ? s.name : s.name || `Seat ${s.seat}`]));
}

function onPacket(p: GamePacket) {
  const prev = packet;
  packet = p;
  // Read-only hook for the end-to-end test: this is only this seat's own view.
  (window as unknown as { __abyss?: unknown }).__abyss = { packet: p, mode };
  const nm = names();
  for (const e of p.events) {
    const line = narrate(e, p.view, nm);
    if (line) log.unshift(line);
  }
  log = log.slice(0, 40);
  if (!boardEl || !document.body.contains(boardEl)) { renderGame(); }
  else { updateGame(); }
  animate(p.events, p.view, prev?.view ?? null);
}

function renderGame() {
  app.replaceChildren(h(`
    <main class="col" style="gap:0;padding-bottom:16px">
      <div class="hud"><span class="round" id="round"></span><span class="pill" id="obj"></span>
        <span class="grow"></span><span class="small" id="mhp"></span></div>
      <div class="hp" style="margin-bottom:10px"><i id="mbar"></i></div>
      <div class="banner" id="banner"></div>
      <div id="board"></div>
      <div class="hint" id="hint"></div>
      <div class="actions" id="actions"></div>
      <div class="log" id="log"></div>
    </main>`));
  boardEl = document.getElementById('board');
  board = null;
  BoardView.create(boardEl!).then(b => {
    board = b;
    b.onTap = onTap;
    updateGame();
  });
}

function myView(): { v: SeatView; seat: number | null } {
  return { v: packet!.view, seat: packet!.you };
}

function updateGame() {
  if (!packet) return;
  const { v, seat } = myView();
  const shown = packet.reveal ?? v;
  const m = v.monster;
  document.getElementById('round')!.textContent = `Round ${v.round}`;
  document.getElementById('obj')!.textContent = v.objective === 'fight' ? 'Fight' : `Flight · ${v.hearts.length} Heart${v.hearts.length === 1 ? '' : 's'} left`;
  document.getElementById('mhp')!.textContent = `${m.name} ${Math.max(0, m.hp)}/${m.maxHp}`;
  (document.getElementById('mbar') as HTMLElement).style.width = `${Math.max(0, Math.min(100, (m.hp / m.maxHp) * 100))}%`;

  const myTurn = isMyTurn(v, seat);
  if (myTurn && !wasMyTurn) { try { navigator.vibrate?.(60); } catch { /* */ } mode = { kind: 'idle' }; }
  wasMyTurn = myTurn;
  const banner = document.getElementById('banner')!;
  if (v.phase === 'over') { banner.className = 'banner monster'; banner.textContent = 'The hunt is over'; }
  else if (myTurn) {
    const left = seat === 0 ? m.actionsLeft : v.hunters.find(x => x.seat === seat)!.actionsLeft;
    banner.className = 'banner you'; banner.textContent = `Your move · ${left} action${left === 1 ? '' : 's'} left`;
  } else if (v.phase === 'monster') { banner.className = 'banner monster'; banner.textContent = `${m.name} is moving…`; }
  else { banner.className = 'banner wait'; banner.textContent = seat === 0 ? 'The Hunters are moving…' : 'Waiting for the other Hunters…'; }

  document.getElementById('log')!.innerHTML = log.slice(0, 6).map(l => `<div>${l}</div>`).join('');
  if (board) board.setView(shown, seat);
  renderActions();
  if (v.phase === 'over') showOver();
}

function isMyTurn(v: SeatView, seat: number | null) {
  if (seat === null || v.phase === 'over') return false;
  if (seat === 0) return v.phase === 'monster';
  return v.phase === 'hunters' && (v.hunters.find(x => x.seat === seat)?.actionsLeft ?? 0) > 0;
}

const cellsWithin = (v: SeatView, from: Pos, r: number) => {
  const out: Pos[] = [];
  for (let y = v.lo; y <= v.hi; y++) for (let x = v.lo; x <= v.hi; x++) if (dist(from, { x, y }) <= r) out.push({ x, y });
  return out;
};

type Btn = { label: string; sub?: string; enabled: boolean; run: () => void; cls?: string };

function hunterButtons(v: SeatView, me: HunterView): Btn[] {
  const c = CLASSES[me.cls];
  const pos = me.pos!;
  const m = v.monster;
  const send = (a: Action) => act(a);
  const allies = v.hunters.filter(o => o.pos);
  const heal = me.cls === 'lifebinder' ? allies.filter(o => !o.down && !o.cocoon && o.hp < o.maxHp && dist(o.pos!, pos) <= 2) : [];
  const rescue = allies.filter(o => o.cocoon > 0 && !o.down && dist(o.pos!, pos) <= 1);
  const revive = v.rules.revive ? allies.filter(o => o.down && !o.revived && !o.cocoonDead && dist(o.pos!, pos) <= 1) : [];
  const hearts = v.objective === 'flight' ? v.hearts.map((hh, i) => ({ hh, i })).filter(x => dist(x.hh.pos, pos) <= 1) : [];
  const onCover = v.cover.includes(pos.y * v.N + pos.x);
  const pickOr = (label: string, ts: Target[]) => (ts.length === 1 ? send(ts[0].action) : setMode({ kind: 'targets', label, cells: ts }));
  const btns: Btn[] = [
    { label: 'Move', sub: `${c.move} sq`, enabled: true, run: () => setMode(moveMode(v, me)) },
    { label: 'Attack', sub: m.pos ? `${me.attackDice ?? '?'} dice` : 'hidden', enabled: !!m.pos && dist(pos, m.pos) <= c.range, run: () => send({ seat: me.seat, kind: 'attack' }) },
    { label: 'Scan', sub: `radius ${c.scan}`, enabled: !m.pos, run: () => send({ seat: me.seat, kind: 'scan' }) },
    { label: 'Hide', sub: onCover ? 'in cover' : 'need cover', enabled: onCover && !me.hidden, run: () => send({ seat: me.seat, kind: 'hide' }) },
  ];
  if (me.cls === 'lifebinder') btns.push({ label: 'Heal', sub: '+1 HP', enabled: heal.length > 0, run: () => pickOr('Tap an ally to heal', heal.map(o => ({ pos: o.pos!, action: { seat: me.seat, kind: 'heal', target: o.seat } }))) });
  if (me.cls === 'riftweaver') btns.push({ label: 'Slow', sub: me.slowCd ? `ready in ${me.slowCd}` : '-1 Monster action', enabled: !!m.pos && dist(pos, m.pos) <= 4 && me.slowCd === 0, run: () => send({ seat: me.seat, kind: 'slow' }) });
  if (v.objective === 'flight') btns.push({ label: 'Cleanse', sub: 'a Heart', enabled: hearts.length > 0, run: () => pickOr('Tap a Heart', hearts.map(x => ({ pos: x.hh.pos, action: { seat: me.seat, kind: 'cleanse', heart: x.i } }))) });
  if (rescue.length) btns.push({ label: 'Cut free', sub: 'cocoon', enabled: true, run: () => pickOr('Tap the cocoon', rescue.map(o => ({ pos: o.pos!, action: { seat: me.seat, kind: 'rescue', target: o.seat } }))) });
  if (v.rules.revive) btns.push({ label: 'Revive', sub: 'once each', enabled: revive.length > 0, run: () => pickOr('Tap a fallen ally', revive.map(o => ({ pos: o.pos!, action: { seat: me.seat, kind: 'revive', target: o.seat } }))) });
  btns.push({ label: 'End turn', enabled: true, run: () => send({ seat: me.seat, kind: 'endTurn' }), cls: 'end' });
  return btns;
}

function moveMode(v: SeatView, me: HunterView): Mode {
  const cells = cellsWithin(v, me.pos!, CLASSES[me.cls].move).filter(p => dist(p, me.pos!) > 0);
  return { kind: 'cells', label: 'Tap a gold square to move', cells: cells.map(p => ({ pos: p, action: { seat: me.seat, kind: 'move', to: p } as Action })) };
}

function monsterButtons(v: SeatView): Btn[] {
  const m = v.monster, pos = m.pos!;
  const send = (a: Action) => act(a);
  const visible = v.hunters.filter(o => o.pos && !o.down && !o.cocoon);
  const inRange = visible.filter(o => dist(o.pos!, pos) <= m.range);
  const has = m.actionsLeft > 0;
  const btns: Btn[] = [];
  if (m.id === 'revenant') btns.push({ label: 'Phantom Step', sub: m.phantomStepUsed ? 'used' : 'free · 3 sq', enabled: !m.phantomStepUsed, run: () => setMode(phantomMode(v)) });
  btns.push({ label: 'Move', sub: `${m.move} sq`, enabled: has, run: () => setMode(monsterMoveMode(v)) });
  btns.push({ label: 'Attack', sub: `${m.attackDice ?? '?'} dice`, enabled: has && inRange.length > 0, run: () => setMode({ kind: 'targets', label: 'Tap a Hunter to strike', cells: inRange.map(o => ({ pos: o.pos!, action: { seat: 0, kind: 'attack', target: o.seat } })) }) });
  if (m.id === 'greedMaw') { const t = inRange.filter(o => o.hp <= 2 && dist(o.pos!, pos) <= 1); btns.push({ label: 'Cocoon', enabled: has && m.cocoonCd === 0 && t.length > 0, run: () => setMode({ kind: 'targets', label: 'Tap a weakened Hunter', cells: t.map(o => ({ pos: o.pos!, action: { seat: 0, kind: 'cocoon', target: o.seat } })) }) }); }
  if (m.id === 'chronophage') { const t = visible.filter(o => dist(o.pos!, pos) <= 4); btns.push({ label: 'Time Snare', enabled: has && m.snareCd === 0 && t.length > 0, run: () => setMode({ kind: 'targets', label: 'Tap a Hunter to snare', cells: t.map(o => ({ pos: o.pos!, action: { seat: 0, kind: 'snare', target: o.seat } })) }) }); }
  if (m.id === 'hiveQueen') btns.push({ label: 'Brood', enabled: has && !m.summonUsed && v.round % 3 === 1, run: () => send({ seat: 0, kind: 'summon' }) });
  btns.push({ label: 'Corrupt', sub: '+2 rot', enabled: has, run: () => send({ seat: 0, kind: 'corrupt' }) });
  btns.push({ label: 'End turn', enabled: true, run: () => send({ seat: 0, kind: 'endTurn' }), cls: 'end' });
  return btns;
}

function phantomMode(v: SeatView): Mode {
  const pos = v.monster.pos!;
  return { kind: 'cells', label: 'Phantom Step: tap a square (stays hidden)', cells: cellsWithin(v, pos, 3).map(p => ({ pos: p, action: { seat: 0, kind: 'phantomStep', to: p } as Action })) };
}
function monsterMoveMode(v: SeatView): Mode {
  const pos = v.monster.pos!;
  return { kind: 'cells', label: 'Tap a gold square to move', cells: cellsWithin(v, pos, v.monster.move).filter(p => dist(p, pos) > 0).map(p => ({ pos: p, action: { seat: 0, kind: 'move', to: p } as Action })) };
}

function defaultMode(v: SeatView, seat: number): Mode {
  if (seat === 0) {
    if (v.monster.id === 'revenant' && !v.monster.phantomStepUsed) return phantomMode(v);
    return v.monster.actionsLeft > 0 ? monsterMoveMode(v) : { kind: 'idle' };
  }
  const me = v.hunters.find(x => x.seat === seat)!;
  return moveMode(v, me);
}

function setMode(m: Mode) { mode = m; renderActions(); }
function exposeMode() { const w = window as unknown as { __abyss?: { mode: Mode } }; if (w.__abyss) w.__abyss.mode = mode; }

function renderActions() {
  const el = document.getElementById('actions');
  const hint = document.getElementById('hint');
  if (!el || !hint || !packet) return;
  const { v, seat } = myView();
  if (!isMyTurn(v, seat)) {
    el.replaceChildren();
    hint.textContent = seat === null ? 'Watching. Take a seat in the next hunt.' : '';
    board?.setHighlights([], 'move');
    return;
  }
  if (mode.kind === 'idle') mode = defaultMode(v, seat!);
  const btns = seat === 0 ? monsterButtons(v) : hunterButtons(v, v.hunters.find(x => x.seat === seat)!);
  el.replaceChildren(...btns.map(b => {
    const n = h(`<button class="${b.cls ?? ''}" ${b.enabled ? '' : 'disabled'}>${b.label}${b.sub ? `<small>${b.sub}</small>` : ''}</button>`) as HTMLButtonElement;
    n.onclick = b.run;
    return n;
  }));
  hint.textContent = mode.kind === 'idle' ? '' : mode.label;
  board?.setHighlights(mode.kind === 'idle' ? [] : mode.cells.map(c => c.pos), mode.kind === 'targets' ? 'target' : 'move');
  exposeMode();
}

function onTap(p: Pos) {
  if (!packet) return;
  const { v, seat } = myView();
  if (!isMyTurn(v, seat)) return;
  if (mode.kind !== 'idle') {
    const hit = mode.cells.find(c => c.pos.x === p.x && c.pos.y === p.y);
    if (hit) return act(hit.action);
  }
  // Shortcut: tapping the Monster attacks it when you can.
  if (seat !== 0 && v.monster.pos && v.monster.pos.x === p.x && v.monster.pos.y === p.y) {
    const me = v.hunters.find(x => x.seat === seat)!;
    if (dist(me.pos!, p) <= CLASSES[me.cls].range) act({ seat: seat!, kind: 'attack' });
  }
}

function act(a: Action) {
  mode = { kind: 'idle' };
  board?.setHighlights([], 'move');
  net?.send('act', { action: a });
}

// ------------------------------------------------------------------ feedback
function animate(events: GameEvent[], v: SeatView, prev: SeatView | null) {
  const where = (seat: number): Pos | null => seat === 0 ? (v.monster.pos ?? prev?.monster.pos ?? null) : (v.hunters.find(x => x.seat === seat)?.pos ?? null);
  for (const e of events) {
    if (e.type === 'attack') {
      showDice(e.dice, e.need, e.seat === 0 ? v.monster.name : CLASSES[v.hunters.find(x => x.seat === e.seat)!.cls].name);
      const p = where(e.target);
      if (p && board) board.pop(p, e.hits ? `-${e.hits}` : 'miss', e.hits ? '#ff5168' : '#9a8fb0', e.hits > 0 && e.target === packet?.you);
    } else if (e.type === 'revealed' && board) board.pop(e.pos, 'REVEALED', '#e6c74c', true);
    else if (e.type === 'down') { const p = where(e.seat); if (p && board) board.pop(p, 'DOWN', '#ff5168', true); }
    else if (e.type === 'heal' || e.type === 'revive' || e.type === 'rescue') { const p = where(e.target); if (p && board) board.pop(p, '+1', '#57d18a'); }
    else if (e.type === 'enrage' && v.monster.pos && board) board.pop(v.monster.pos, 'ENRAGED', '#ff2a45', true);
  }
}

function showDice(dice: number[], need: number, who: string) {
  document.querySelectorAll('.dice').forEach(d => d.remove());
  const d = h(`<div class="dice" aria-live="polite"><span class="lbl">${esc(who)}</span>${dice.map(x => `<span class="${x >= need ? 'hit' : ''}">${x}</span>`).join('')}</div>`);
  document.body.appendChild(d);
  setTimeout(() => d.remove(), 1500);
}

function showOver() {
  if (document.querySelector('.over')) return;
  const v = packet!.view;
  const you = packet!.you;
  const won = (v.result === 'hunters' && you !== null && you > 0) || (v.result === 'monster' && you === 0);
  const title = v.result === 'hunters' ? 'The Hunters prevail' : v.result === 'monster' ? `${v.monster.name} feeds` : 'Stalemate';
  const card = h(`<div class="card over col">
    <h2 style="color:${won ? 'var(--gold)' : 'var(--blood)'}">${won ? 'Victory' : you === null ? 'Hunt over' : 'Defeat'}</h2>
    <div>${title} in ${v.round} round${v.round === 1 ? '' : 's'}.</div>
    <div class="small muted">The board now shows where everything really was.</div>
    <button class="primary" id="again">Back to the table</button></div>`);
  document.getElementById('actions')!.replaceWith(card);
  document.getElementById('again')!.onclick = async () => { await net?.room.leave(); net = null; lobby = null; packet = null; log = []; render(); };
}

function renderNet() {
  document.querySelectorAll('.net').forEach(n => n.remove());
  if (status === 'reconnecting') document.body.appendChild(h('<div class="net">Signal lost. Reconnecting…</div>'));
  if (status === 'gone' && packet) toast('Disconnected from the hunt.');
}

// ------------------------------------------------------------------ boot
if ('serviceWorker' in navigator && location.protocol === 'https:') navigator.serviceWorker.register('/sw.js').catch(() => {});
(async () => {
  const resumed = await resumeHunt(handlers);
  if (resumed) { net = resumed; return; }
  renderHome();
})();
