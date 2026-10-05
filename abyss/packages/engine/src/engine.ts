// ABYSS rules engine. Pure and deterministic:
//   createGame(setup)            -> GameState
//   applyAction(state, action)   -> { state, events }   (input state is never mutated)
//   validate(state, action)      -> null | reason
// Every random draw comes from state.rng, so a seed plus an action list replays a game exactly.

import { CLASSES, LOCKED_RULES, MONSTERS, MONSTER_SEAT, type Rules } from './rules';
import { nextFloat, pick, randInt, shuffle } from './rng';
import type {
  Action, ApplyResult, GameEvent, GameState, HunterState, MonsterAction, HunterAction, Pos, Setup,
} from './types';

// ------------------------------------------------------------------ geometry
export const dist = (a: Pos, b: Pos) => Math.max(Math.abs(a.x - b.x), Math.abs(a.y - b.y));
export const samePos = (a: Pos, b: Pos) => a.x === b.x && a.y === b.y;
export const idx = (s: { N: number }, p: Pos) => p.y * s.N + p.x;
export const posOf = (s: { N: number }, i: number): Pos => ({ x: i % s.N, y: Math.floor(i / s.N) });
export const inBounds = (s: { lo: number; hi: number }, p: Pos) => s.lo <= p.x && p.x <= s.hi && s.lo <= p.y && p.y <= s.hi;

/** One king-step per square toward t (the movement the bots and Bloodlust use). */
export function stepToward(p: Pos, t: Pos, steps: number): Pos {
  let { x, y } = p;
  for (let i = 0; i < steps; i++) {
    if (x === t.x && y === t.y) break;
    x += Math.sign(t.x - x);
    y += Math.sign(t.y - y);
  }
  return { x, y };
}

const clampIn = (s: GameState, p: Pos): Pos => ({
  x: Math.min(s.hi, Math.max(s.lo, p.x)),
  y: Math.min(s.hi, Math.max(s.lo, p.y)),
});

// ------------------------------------------------------------------ queries
export const isActive = (h: HunterState) => !h.down && h.cocoon === 0;
export const activeHunters = (s: GameState) => s.hunters.filter(isActive);
export const hunterBySeat = (s: GameState, seat: number) => s.hunters.find(h => h.seat === seat);
export const isHigh = (s: GameState, p: Pos) => s.high.includes(idx(s, p));
export const isCover = (s: GameState, p: Pos) => s.cover.includes(idx(s, p));
export const isCorrupt = (s: GameState, p: Pos) => s.corrupt[idx(s, p)] === 1;

export function corruptionFraction(s: GameState): number {
  let n = 0;
  for (let y = s.lo; y <= s.hi; y++) for (let x = s.lo; x <= s.hi; x++) n += s.corrupt[y * s.N + x];
  return n / (s.hi - s.lo + 1) ** 2;
}

export function hunterAttackDice(s: GameState, h: HunterState): number {
  let n = CLASSES[h.cls].dice + s.rules.hunterDiceBonus;
  if (isHigh(s, h.pos)) n += 1;
  if (isCorrupt(s, h.pos)) n -= 1;
  if (s.monster.id === 'chronophage' && dist(h.pos, s.monster.pos) <= 2) n -= 1; // Chrono Pulse
  if (s.monster.id === 'harbinger' && !isHigh(s, h.pos)) n -= 1;                 // Wings of Terror
  return Math.max(0, n);
}

export function monsterAttackDice(s: GameState): number {
  const m = s.monster;
  let n = m.dice + (m.firstKill ? s.rules.firstKillDice : 0);
  if (s.objective === 'fight') n += s.rules.fightFuryDice;
  if (isCorrupt(s, m.pos)) n += 1;
  if (isHigh(s, m.pos)) n += 1;
  return Math.max(0, n);
}

/** The Monster may only target a hidden Hunter from 2 squares or closer. */
export const monsterCanSee = (s: GameState, h: HunterState) => isActive(h) && !(h.hidden && dist(h.pos, s.monster.pos) > 2);

export const monsterActionsFor = (s: GameState) =>
  s.rules.monsterActions[s.hunters.length] ?? Math.max(1, s.hunters.length - 1);

// ------------------------------------------------------------------ setup
export function createGame(setup: Setup): { state: GameState; events: GameEvent[] } {
  const rules: Rules = { ...LOCKED_RULES, ...setup.rules };
  const k = setup.hunters.length;
  if (k < rules.minHunters || k > rules.maxHunters) throw new Error(`squad size ${k} outside ${rules.minHunters}-${rules.maxHunters}`);
  const seats = new Set(setup.hunters.map(h => h.seat));
  if (seats.size !== k || seats.has(MONSTER_SEAT)) throw new Error('hunter seats must be unique and non-zero');

  const N = rules.board;
  const r = { rng: setup.seed >>> 0 };
  const cells = shuffle(r, Array.from({ length: N * N }, (_, i) => i));
  const nh = Math.floor(N * N * rules.highGroundPct);
  const nc = Math.floor(N * N * rules.coverPct);
  const high = cells.slice(0, nh).sort((a, b) => a - b);
  const cover = cells.slice(nh, nh + nc).sort((a, b) => a - b);

  // Hearts in the Monster's (north) half.
  const hearts: GameState['hearts'] = [];
  while (hearts.length < rules.hearts) {
    const p = { x: randInt(r, 1, N - 2), y: randInt(r, 0, N / 2 - 1) };
    if (!hearts.some(h => samePos(h.pos, p))) hearts.push({ pos: p, hp: rules.heartHp });
  }
  const corrupt = new Array(N * N).fill(0);
  for (const h of hearts) corrupt[h.pos.y * N + h.pos.x] = 1;

  const M = MONSTERS[setup.monster];
  const hp = rules.monsterHp[setup.monster] ?? M.hp;
  const monster: GameState['monster'] = {
    id: setup.monster, hp, maxHp: hp, dice: rules.monsterDice[setup.monster] ?? M.dice,
    range: M.range, move: M.move, stealthy: M.stealthy,
    pos: { x: randInt(r, 2, N - 3), y: randInt(r, 0, 2) }, hidden: true,
    slowed: 0, enraged: false, firstKill: false, snareCd: 0, cocoonCd: 0,
    actionsLeft: 0, phantomStepUsed: false, summonUsed: false,
  };

  // Hunters enter from the south edge on distinct columns.
  const xs = shuffle(r, Array.from({ length: N - 2 }, (_, i) => i + 1)).slice(0, k);
  const hunters: HunterState[] = setup.hunters.map((h, i) => {
    const c = CLASSES[h.cls];
    return {
      seat: h.seat, cls: h.cls, look: h.look ?? { timeline: 0, sex: 'M' },
      hp: c.hp, maxHp: c.hp, pos: { x: xs[i], y: N - 1 }, hidden: false, down: false,
      cocoon: 0, cocoonDead: false, revived: false, lostAction: 0, guardedRound: 0, slowCd: 0, actionsLeft: 0,
    };
  }).sort((a, b) => a.seat - b.seat);

  const s: GameState = {
    v: 1, rules, seed: setup.seed >>> 0, rng: r.rng, N, lo: 0, hi: N - 1,
    high, cover, corrupt, hearts, phantoms: [], minions: [], nextMinionId: 1,
    objective: setup.objective, round: 0, quiet: 0, quietHit: false,
    phase: 'hunters', result: null, hunters, monster, stats: {},
  };
  s.phantoms = Array.from({ length: rules.phantoms }, () => near(s, s.monster.pos, 4));
  const events: GameEvent[] = [];
  startRound(s, events);
  return { state: s, events };
}

function near(s: GameState, p: Pos, rad: number): Pos {
  return clampIn(s, { x: p.x + randInt(s, -rad, rad), y: p.y + randInt(s, -rad, rad) });
}

const bump = (s: GameState, key: string, n = 1) => { s.stats[key] = (s.stats[key] ?? 0) + n; };

function rollDice(s: GameState, n: number, need: number) {
  const dice = Array.from({ length: Math.max(0, n) }, () => randInt(s, 1, 6));
  return { dice, hits: dice.filter(d => d >= need).length };
}

/** Grow corruption by one square next to existing corruption (king-adjacent). Returns the cell or -1. */
function spreadOne(s: GameState, horizontalOnly = false): number {
  const cand: number[] = [];
  for (let i = 0; i < s.corrupt.length; i++) {
    if (!s.corrupt[i]) continue;
    const c = posOf(s, i);
    for (let dx = -1; dx <= 1; dx++) for (let dy = -1; dy <= 1; dy++) {
      if (horizontalOnly ? (dy !== 0 || dx === 0) : (dx === 0 && dy === 0)) continue;
      const p = { x: c.x + dx, y: c.y + dy };
      if (inBounds(s, p) && !s.corrupt[idx(s, p)]) cand.push(idx(s, p));
    }
  }
  if (!cand.length) return -1;
  const cell = pick(s, cand);
  s.corrupt[cell] = 1;
  return cell;
}

// ------------------------------------------------------------------ round flow
function startRound(s: GameState, ev: GameEvent[]) {
  s.round += 1;
  ev.push({ vis: 'all', type: 'round', round: s.round });
  abyssPhase(s, ev);
  if (checkEnd(s, ev)) return;
  startHunterPhase(s, ev);
}

function abyssPhase(s: GameState, ev: GameEvent[]) {
  const spread: number[] = [];
  for (let i = 0; i < s.hearts.length; i++)
    for (let j = 0; j < s.rules.spreadPerHeart; j++) { const c = spreadOne(s); if (c >= 0) spread.push(c); }

  const card = randInt(s, 1, 6);
  if (card <= 2) {
    for (let j = 0; j < 2; j++) { const c = spreadOne(s, true); if (c >= 0) spread.push(c); }
    ev.push({ vis: 'all', type: 'abyssCard', card: 'creepingRot' });
  } else if (card === 4) {
    const hurt = activeHunters(s).filter(h => h.hp < h.maxHp);
    if (hurt.length) {
      const h = pick(s, hurt); h.hp += 1;
      ev.push({ vis: 'all', type: 'abyssCard', card: 'temporalReversal', target: h.seat });
    } else ev.push({ vis: 'all', type: 'abyssCard', card: 'calm' });
  } else if (card === 5) {
    const a = activeHunters(s);
    if (a.length) {
      const h = pick(s, a); h.lostAction = 1;
      ev.push({ vis: 'all', type: 'abyssCard', card: 'lostMoment', target: h.seat });
    } else ev.push({ vis: 'all', type: 'abyssCard', card: 'calm' });
  } else ev.push({ vis: 'all', type: 'abyssCard', card: 'calm' });
  if (spread.length) ev.push({ vis: 'all', type: 'corruptionSpread', cells: spread });

  // Specters: each downed Hunter scrubs the nearest non-Heart corruption.
  for (const h of s.hunters) {
    if (!h.down) continue;
    let best = -1, bestD = Infinity;
    for (let i = 0; i < s.corrupt.length; i++) {
      if (!s.corrupt[i]) continue;
      const p = posOf(s, i);
      if (!inBounds(s, p) || s.hearts.some(hh => samePos(hh.pos, p))) continue;
      const d = dist(p, h.pos);
      if (d < bestD) { bestD = d; best = i; }
    }
    if (best >= 0) { s.corrupt[best] = 0; ev.push({ vis: 'all', type: 'specterScrub', seat: h.seat, cell: best }); }
  }
}

function startHunterPhase(s: GameState, ev: GameEvent[]) {
  s.phase = 'hunters';
  for (const h of s.hunters) {
    h.actionsLeft = isActive(h) ? Math.max(0, s.rules.hunterActions - h.lostAction) : 0;
    if (isActive(h)) h.lostAction = 0;
  }
  ev.push({ vis: 'all', type: 'phase', phase: 'hunters' });
  if (s.hunters.every(h => h.actionsLeft === 0)) startMonsterPhase(s, ev);
}

function startMonsterPhase(s: GameState, ev: GameEvent[]) {
  s.phase = 'monster';
  const m = s.monster;
  m.actionsLeft = Math.max(0, monsterActionsFor(s) - m.slowed);
  m.slowed = 0;
  m.phantomStepUsed = false;
  m.summonUsed = false;
  ev.push({ vis: 'all', type: 'phase', phase: 'monster' });
  if (m.actionsLeft === 0 && !canPhantomStep(s)) endMonsterPhase(s, ev);
}

const canPhantomStep = (s: GameState) => s.monster.id === 'revenant' && !s.monster.phantomStepUsed;

function endMonsterPhase(s: GameState, ev: GameEvent[]) {
  const m = s.monster;
  m.actionsLeft = 0;
  // Minions act: strike an adjacent Hunter or close 2 squares.
  for (const mi of s.minions) {
    const a = activeHunters(s);
    if (!a.length) break;
    const t = a.reduce((b, h) => (dist(h.pos, mi.pos) < dist(b.pos, mi.pos) ? h : b));
    if (dist(t.pos, mi.pos) <= 1) {
      const r = rollDice(s, 2, s.rules.monsterHit);
      ev.push({ vis: 'all', type: 'minionAttack', minion: mi.id, target: t.seat, dice: r.dice, hits: r.hits });
      damageHunter(s, t, r.hits, ev);
    } else {
      mi.pos = stepToward(mi.pos, t.pos, 2);
    }
  }
  if (checkEnd(s, ev)) return;
  // Hunters swat adjacent minions (free, 50% each).
  for (const h of activeHunters(s)) {
    s.minions = s.minions.filter(mi => {
      if (dist(mi.pos, h.pos) <= 1 && nextFloat(s) < 0.5) {
        ev.push({ vis: 'all', type: 'minionSwatted', minion: mi.id, seat: h.seat });
        return false;
      }
      return true;
    });
  }
  // Re-hide when no Hunter is within 3 (stealthy monsters anywhere, others only in cover).
  if (!m.hidden && activeHunters(s).every(h => dist(h.pos, m.pos) > 3) && (m.stealthy || isCover(s, m.pos))) {
    m.hidden = true;
    s.phantoms = Array.from({ length: s.rules.phantoms }, () => near(s, m.pos, 4));
    ev.push({ vis: 'all', type: 'vanished' });
  }
  endRound(s, ev);
}

function endRound(s: GameState, ev: GameEvent[]) {
  const m = s.monster;
  m.snareCd = Math.max(0, m.snareCd - 1);
  m.cocoonCd = Math.max(0, m.cocoonCd - 1);
  for (const h of s.hunters) h.slowCd = Math.max(0, h.slowCd - 1);

  for (const h of s.hunters) {
    if (h.cocoon > 0 && !h.down) {
      h.cocoon -= 1;
      if (h.cocoon === 0) {
        h.down = true; h.cocoonDead = true; h.hp = 0;
        bump(s, 'cocoonDeaths');
        ev.push({ vis: 'all', type: 'cocoonDeath', seat: h.seat });
      }
    }
  }

  // Bloodlust: two quiet rounds drag the Monster 3 squares toward the nearest Hunter.
  s.quiet = s.quietHit ? 0 : s.quiet + 1;
  s.quietHit = false;
  const alive = activeHunters(s);
  if (s.quiet >= s.rules.bloodlustQuietRounds && alive.length) {
    const t = alive.reduce((b, h) => (dist(h.pos, m.pos) < dist(b.pos, m.pos) ? h : b));
    m.pos = stepToward(m.pos, t.pos, 3);
    s.quiet = 0;
    bump(s, 'bloodlust');
    ev.push({ vis: m.hidden ? 'monster' : 'all', type: 'bloodlust', to: { ...m.pos } });
  }

  // Sudden Death (Fight): the arena shrinks one ring per round.
  if (s.objective === 'fight' && s.round >= s.rules.suddenDeathFight && s.hi - s.lo > 3) {
    s.lo += 1; s.hi -= 1;
    bump(s, 'suddenDeathRounds');
    ev.push({ vis: 'all', type: 'suddenDeath', lo: s.lo, hi: s.hi });
    for (const u of activeHunters(s)) {
      if (!inBounds(s, u.pos)) { u.pos = clampIn(s, u.pos); damageHunter(s, u, 1, ev); }
    }
    if (!inBounds(s, m.pos)) m.pos = clampIn(s, m.pos);
    s.minions.forEach(mi => { mi.pos = clampIn(s, mi.pos); });
    s.phantoms = s.phantoms.map(p => clampIn(s, p));
  }
  if (checkEnd(s, ev)) return;
  startRound(s, ev);
}

function checkEnd(s: GameState, ev: GameEvent[]): boolean {
  if (s.result) return true;
  if (s.monster.hp <= 0) s.result = 'hunters';
  else if (!activeHunters(s).length) s.result = 'monster';
  else if (corruptionFraction(s) >= s.rules.corruptionWin) { s.result = 'monster'; bump(s, 'corruptionWins'); }
  else if (s.objective === 'flight' && !s.hearts.length) s.result = 'hunters';
  else if (s.objective === 'flight' && s.round >= s.rules.flightSurviveRounds) { s.result = 'hunters'; bump(s, 'surviveWins'); }
  else if (s.round >= s.rules.roundCap) s.result = 'draw';
  if (!s.result) return false;
  s.phase = 'over';
  for (const h of s.hunters) h.actionsLeft = 0;
  s.monster.actionsLeft = 0;
  ev.push({ vis: 'all', type: 'gameOver', result: s.result });
  return true;
}

function damageHunter(s: GameState, target: HunterState, dmg: number, ev: GameEvent[]) {
  let h = target;
  // Vanguard Guard: an adjacent healthy Vanguard takes the hit (once a round).
  if (h.cls !== 'vanguard') {
    for (const g of activeHunters(s)) {
      if (g.cls !== 'vanguard' || g === h || dist(g.pos, h.pos) > 1 || g.hp <= 1) continue;
      if (s.rules.guardOncePerRound) {
        if (g.guardedRound === s.round) continue;
        g.guardedRound = s.round;
      }
      ev.push({ vis: 'all', type: 'guard', seat: g.seat, for: h.seat });
      h = g;
      break;
    }
  }
  if (dmg <= 0) return;
  const amount = Math.max(1, dmg - CLASSES[h.cls].armor);
  h.hp = Math.max(0, h.hp - amount);
  ev.push({ vis: 'all', type: 'damage', seat: h.seat, amount, hp: h.hp });
  if (h.hp === 0) {
    h.down = true; h.actionsLeft = 0;
    bump(s, 'hunterDowns');
    ev.push({ vis: 'all', type: 'down', seat: h.seat });
    if (!s.monster.firstKill) s.monster.firstKill = true; // Empowerment: +1 die
  }
}

// ------------------------------------------------------------------ validation
export function validate(s: GameState, a: Action): string | null {
  if (s.phase === 'over') return 'game over';
  if (a.seat === MONSTER_SEAT) return validateMonster(s, a as MonsterAction);
  return validateHunter(s, a as HunterAction);
}

function validateHunter(s: GameState, a: HunterAction): string | null {
  if (s.phase !== 'hunters') return 'not the Hunter phase';
  const h = hunterBySeat(s, a.seat);
  if (!h) return 'no such Hunter';
  if (h.actionsLeft <= 0) return 'no actions left';
  const m = s.monster;
  const c = CLASSES[h.cls];
  switch (a.kind) {
    case 'endTurn': return null;
    case 'move':
      if (!a.to || !inBounds(s, a.to)) return 'out of bounds';
      return dist(h.pos, a.to) <= c.move ? null : 'too far';
    case 'attack':
      if (m.hidden) return 'the Monster is hidden';
      return dist(h.pos, m.pos) <= c.range ? null : 'out of range';
    case 'scan':
      return m.hidden ? null : 'the Monster is already revealed';
    case 'hide':
      if (h.hidden) return 'already hidden';
      return isCover(s, h.pos) ? null : 'no cover here';
    case 'heal': {
      if (h.cls !== 'lifebinder') return 'only a Lifebinder can heal';
      const t = hunterBySeat(s, a.target);
      if (!t || !isActive(t)) return 'invalid target';
      if (t.hp >= t.maxHp) return 'target is unhurt';
      return dist(t.pos, h.pos) <= 2 ? null : 'out of range';
    }
    case 'slow':
      if (h.cls !== 'riftweaver') return 'only a Riftweaver can slow';
      if (h.slowCd > 0) return 'Slow is recharging';
      if (m.hidden) return 'the Monster is hidden';
      return dist(h.pos, m.pos) <= 4 ? null : 'out of range';
    case 'rescue': {
      const t = hunterBySeat(s, a.target);
      if (!t || t.cocoon === 0 || t.down) return 'nobody to rescue';
      return dist(t.pos, h.pos) <= 1 ? null : 'not adjacent';
    }
    case 'revive': {
      if (!s.rules.revive) return 'revive is off';
      const t = hunterBySeat(s, a.target);
      if (!t || !t.down || t.revived || t.cocoonDead) return 'cannot revive';
      return dist(t.pos, h.pos) <= 1 ? null : 'not adjacent';
    }
    case 'cleanse': {
      if (s.objective !== 'flight') return 'only in Flight';
      const heart = s.hearts[a.heart];
      if (!heart) return 'no such Heart';
      return dist(heart.pos, h.pos) <= 1 ? null : 'not adjacent';
    }
  }
  return 'unknown action';
}

function validateMonster(s: GameState, a: MonsterAction): string | null {
  if (s.phase !== 'monster') return 'not the Monster phase';
  const m = s.monster;
  if (a.kind === 'endTurn') return null;
  if (a.kind === 'phantomStep') {
    if (!canPhantomStep(s)) return 'no Phantom Step';
    if (!a.to || !inBounds(s, a.to)) return 'out of bounds';
    return dist(m.pos, a.to) <= 3 ? null : 'too far';
  }
  if (m.actionsLeft <= 0) return 'no actions left';
  switch (a.kind) {
    case 'move':
      if (!a.to || !inBounds(s, a.to)) return 'out of bounds';
      return dist(m.pos, a.to) <= m.move ? null : 'too far';
    case 'attack': case 'cocoon': case 'snare': {
      const t = hunterBySeat(s, a.target);
      if (!t || !monsterCanSee(s, t)) return 'no visible target';
      const d = dist(t.pos, m.pos);
      if (a.kind === 'attack') return d <= m.range ? null : 'out of range';
      if (a.kind === 'cocoon') {
        if (m.id !== 'greedMaw') return 'only the Greed Maw cocoons';
        if (m.cocoonCd > 0) return 'Cocoon is recharging';
        if (t.hp > 2) return 'target too healthy';
        return d <= 1 ? null : 'not adjacent';
      }
      if (m.id !== 'chronophage') return 'only the Chronophage snares';
      if (m.snareCd > 0) return 'Time Snare is recharging';
      return d <= 4 ? null : 'out of range';
    }
    case 'summon':
      if (m.id !== 'hiveQueen') return 'only the Hive Queen summons';
      if (m.summonUsed || s.round % 3 !== 1) return 'Brood is not ready';
      return null;
    case 'corrupt':
      return null;
  }
  return 'unknown action';
}

// ------------------------------------------------------------------ apply
export function applyAction(state: GameState, a: Action): ApplyResult {
  const s = structuredClone(state);
  const events = applyActionMut(s, a);
  return { state: s, events };
}

/** Mutating variant for the simulator's hot loop. Throws on an illegal action. */
export function applyActionMut(s: GameState, a: Action): GameEvent[] {
  const err = validate(s, a);
  if (err) throw new Error(`illegal ${a.kind} by seat ${a.seat}: ${err}`);
  const ev: GameEvent[] = [];
  if (a.seat === MONSTER_SEAT) doMonster(s, a as MonsterAction, ev);
  else doHunter(s, a as HunterAction, ev);
  return ev;
}

function doHunter(s: GameState, a: HunterAction, ev: GameEvent[]) {
  const h = hunterBySeat(s, a.seat)!;
  const m = s.monster;
  if (a.kind === 'endTurn') {
    h.actionsLeft = 0;
    ev.push({ vis: 'all', type: 'endTurn', seat: h.seat });
  } else {
    h.actionsLeft -= 1;
    switch (a.kind) {
      case 'move': {
        const from = { ...h.pos };
        h.pos = { ...a.to }; h.hidden = false;
        ev.push({ vis: 'all', type: 'move', seat: h.seat, from, to: { ...h.pos } });
        break;
      }
      case 'attack': {
        const r = rollDice(s, hunterAttackDice(s, h), s.rules.hunterHit);
        m.hp -= r.hits; h.hidden = false; s.quietHit = true;
        bump(s, 'hunterHits', r.hits);
        ev.push({ vis: 'all', type: 'attack', seat: h.seat, target: MONSTER_SEAT, dice: r.dice, need: s.rules.hunterHit, hits: r.hits });
        if (!m.enraged && m.hp <= Math.floor(m.maxHp / 2) && m.hp > 0) {
          m.enraged = true; m.hp += s.rules.enrageHeal;
          ev.push({ vis: 'all', type: 'enrage', hp: m.hp });
        }
        break;
      }
      case 'scan': {
        const range = CLASSES[h.cls].scan;
        bump(s, 'scans');
        if (dist(m.pos, h.pos) <= range) {
          m.hidden = false; s.phantoms = [];
          bump(s, 'revealsByScan');
          ev.push({ vis: 'all', type: 'scan', seat: h.seat, found: true, cleared: 0 });
          ev.push({ vis: 'all', type: 'revealed', by: 'scan', pos: { ...m.pos } });
        } else {
          const before = s.phantoms.length;
          s.phantoms = s.phantoms.filter(p => dist(p, h.pos) > range);
          ev.push({ vis: 'all', type: 'scan', seat: h.seat, found: false, cleared: before - s.phantoms.length });
        }
        break;
      }
      case 'hide':
        h.hidden = true;
        ev.push({ vis: 'all', type: 'hide', seat: h.seat });
        break;
      case 'heal': {
        const t = hunterBySeat(s, a.target)!;
        t.hp += 1;
        ev.push({ vis: 'all', type: 'heal', seat: h.seat, target: t.seat });
        break;
      }
      case 'slow':
        m.slowed = 1; h.slowCd = 3;
        ev.push({ vis: 'all', type: 'slow', seat: h.seat });
        break;
      case 'rescue': {
        const t = hunterBySeat(s, a.target)!;
        t.cocoon = 0; t.hp = 1; t.actionsLeft = 0;
        bump(s, 'rescues');
        ev.push({ vis: 'all', type: 'rescue', seat: h.seat, target: t.seat });
        break;
      }
      case 'revive': {
        const t = hunterBySeat(s, a.target)!;
        t.down = false; t.revived = true; t.hp = 1; t.actionsLeft = 0;
        bump(s, 'revives');
        ev.push({ vis: 'all', type: 'revive', seat: h.seat, target: t.seat });
        break;
      }
      case 'cleanse': {
        const heart = s.hearts[a.heart];
        heart.hp -= 1;
        ev.push({ vis: 'all', type: 'cleanse', seat: h.seat, heart: a.heart, hp: heart.hp });
        if (heart.hp <= 0) { s.hearts.splice(a.heart, 1); bump(s, 'heartsCleansed'); }
        break;
      }
    }
  }
  if (checkEnd(s, ev)) return;
  if (s.hunters.every(x => x.actionsLeft === 0)) startMonsterPhase(s, ev);
}

function doMonster(s: GameState, a: MonsterAction, ev: GameEvent[]) {
  const m = s.monster;
  if (a.kind === 'endTurn') {
    ev.push({ vis: 'all', type: 'endTurn', seat: MONSTER_SEAT });
    endMonsterPhase(s, ev);
    return;
  }
  if (a.kind === 'phantomStep') {
    m.phantomStepUsed = true;
    const from = { ...m.pos };
    m.pos = { ...a.to };
    ev.push({ vis: m.hidden ? 'monster' : 'all', type: 'move', seat: MONSTER_SEAT, from, to: { ...m.pos } });
  } else {
    m.actionsLeft -= 1;
    switch (a.kind) {
      case 'move': {
        const from = { ...m.pos };
        m.pos = { ...a.to };
        ev.push({ vis: m.hidden ? 'monster' : 'all', type: 'move', seat: MONSTER_SEAT, from, to: { ...m.pos } });
        break;
      }
      case 'attack': {
        const t = hunterBySeat(s, a.target)!;
        if (m.hidden) ev.push({ vis: 'all', type: 'revealed', by: 'attack', pos: { ...m.pos } });
        m.hidden = false; s.phantoms = [];
        const r = rollDice(s, monsterAttackDice(s), s.rules.monsterHit);
        ev.push({ vis: 'all', type: 'attack', seat: MONSTER_SEAT, target: t.seat, dice: r.dice, need: s.rules.monsterHit, hits: r.hits });
        damageHunter(s, t, r.hits, ev);
        s.quietHit = true;
        break;
      }
      case 'cocoon': {
        const t = hunterBySeat(s, a.target)!;
        t.cocoon = 3; t.hidden = false; t.actionsLeft = 0;
        m.hp = Math.min(m.maxHp, m.hp + 2); m.cocoonCd = 2;
        bump(s, 'cocoons');
        ev.push({ vis: 'all', type: 'cocoon', target: t.seat });
        break;
      }
      case 'snare': {
        const t = hunterBySeat(s, a.target)!;
        t.lostAction = 1; m.snareCd = 2;
        ev.push({ vis: 'all', type: 'snare', target: t.seat });
        break;
      }
      case 'summon': {
        m.summonUsed = true;
        const spots: number[] = [];
        for (let i = 0; i < s.corrupt.length; i++) if (s.corrupt[i] && inBounds(s, posOf(s, i))) spots.push(i);
        const made = [];
        for (let j = 0; j < 2 && spots.length; j++) {
          const mi = { id: s.nextMinionId++, pos: posOf(s, pick(s, spots)) };
          s.minions.push(mi); made.push({ ...mi });
        }
        ev.push({ vis: 'all', type: 'summon', minions: made });
        break;
      }
      case 'corrupt': {
        const cells = [spreadOne(s), spreadOne(s)].filter(c => c >= 0);
        ev.push({ vis: 'all', type: 'corruptionSpread', cells });
        break;
      }
    }
  }
  if (checkEnd(s, ev)) return;
  if (m.actionsLeft === 0 && !canPhantomStep(s)) endMonsterPhase(s, ev);
}

/** Whose move is it? Seats that can act right now. */
export function seatsToAct(s: GameState): number[] {
  if (s.phase === 'hunters') return s.hunters.filter(h => h.actionsLeft > 0).map(h => h.seat);
  if (s.phase === 'monster') return [MONSTER_SEAT];
  return [];
}
