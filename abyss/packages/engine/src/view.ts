// Per-seat views. A view is everything one seat is allowed to know; bots and clients only ever get views.
// Hidden information:
//   - Hunters never see the hidden Monster's square. They see "signals": the true square shuffled in with
//     phantom tokens, sorted so the order leaks nothing.
//   - The Monster never sees a hidden Hunter more than 2 squares away.

import { CLASSES, MONSTER_SEAT, MONSTERS, type ClassId, type MonsterId, type Objective, type Rules } from './rules';
import { dist, hunterAttackDice, monsterActionsFor, monsterAttackDice, monsterCanSee } from './engine';
import type { GameEvent, GameState, Heart, Look, Minion, Phase, Pos, Result } from './types';

export interface HunterView {
  seat: number;
  cls: ClassId;
  look: Look;
  hp: number;
  maxHp: number;
  pos: Pos | null; // null = hidden from this seat
  hidden: boolean;
  down: boolean;
  cocoon: number;
  cocoonDead: boolean;
  revived: boolean;
  actionsLeft: number;
  lostAction: number;
  slowCd: number;
  /** null while the Monster is hidden from this seat (Chrono Pulse would leak its distance). */
  attackDice: number | null;
}

export interface MonsterView {
  id: MonsterId;
  name: string;
  hp: number;
  maxHp: number;
  pos: Pos | null;
  hidden: boolean;
  range: number;
  move: number;
  enraged: boolean;
  firstKill: boolean;
  actionsLeft: number;
  actionsPerTurn: number;
  attackDice: number | null;
  snareCd: number;
  cocoonCd: number;
  phantomStepUsed: boolean;
  summonUsed: boolean;
}

export interface SeatView {
  seat: number;
  role: 'monster' | 'hunter' | 'spectator';
  rules: Rules;
  N: number;
  lo: number;
  hi: number;
  round: number;
  phase: Phase;
  result: Result;
  objective: Objective;
  high: number[];
  cover: number[];
  corrupt: number[];
  hearts: Heart[];
  minions: Minion[];
  hunters: HunterView[];
  monster: MonsterView;
  /** Hunters only, while the Monster is hidden: possible Monster squares (one is real). */
  signals: Pos[];
  quiet: number;
}

const SPECTATOR = -1;

export function viewFor(s: GameState, seat: number): SeatView {
  const role = seat === MONSTER_SEAT ? 'monster' : s.hunters.some(h => h.seat === seat) ? 'hunter' : 'spectator';
  const m = s.monster;
  const omniscient = seat === SPECTATOR;
  const seeMonster = omniscient || role === 'monster' || !m.hidden;

  const hunters: HunterView[] = s.hunters.map(h => {
    const visible = omniscient || role !== 'monster' || h.down || h.cocoon > 0 || monsterCanSee(s, h);
    return {
      seat: h.seat, cls: h.cls, look: h.look, hp: h.hp, maxHp: h.maxHp,
      pos: visible ? { ...h.pos } : null, hidden: h.hidden, down: h.down, cocoon: h.cocoon,
      cocoonDead: h.cocoonDead, revived: h.revived, actionsLeft: h.actionsLeft, lostAction: h.lostAction,
      slowCd: h.slowCd, attackDice: seeMonster ? hunterAttackDice(s, h) : null,
    };
  });

  let signals: Pos[] = [];
  if (!seeMonster) {
    signals = [m.pos, ...s.phantoms].map(p => ({ ...p }));
    signals.sort((a, b) => a.y - b.y || a.x - b.x);
    // Two tokens on the same square would reveal nothing extra; collapse duplicates.
    signals = signals.filter((p, i) => i === 0 || p.x !== signals[i - 1].x || p.y !== signals[i - 1].y);
  }

  return {
    seat, role, rules: s.rules, N: s.N, lo: s.lo, hi: s.hi, round: s.round, phase: s.phase, result: s.result,
    objective: s.objective, high: s.high, cover: s.cover, corrupt: s.corrupt.slice(),
    hearts: s.hearts.map(h => ({ pos: { ...h.pos }, hp: h.hp })), minions: s.minions.map(mi => ({ id: mi.id, pos: { ...mi.pos } })),
    hunters,
    monster: {
      id: m.id, name: MONSTERS[m.id].name, hp: m.hp, maxHp: m.maxHp, pos: seeMonster ? { ...m.pos } : null,
      hidden: m.hidden, range: m.range, move: m.move, enraged: m.enraged, firstKill: m.firstKill,
      actionsLeft: m.actionsLeft, actionsPerTurn: monsterActionsFor(s),
      attackDice: seeMonster ? monsterAttackDice(s) : null,
      snareCd: m.snareCd, cocoonCd: m.cocoonCd, phantomStepUsed: m.phantomStepUsed, summonUsed: m.summonUsed,
    },
    signals,
    quiet: s.quiet,
  };
}

/** Omniscient view for replays and the end-of-game reveal. */
export const spectatorView = (s: GameState) => viewFor(s, SPECTATOR);

/** Drop events this seat must not see. */
export function eventsFor(events: GameEvent[], seat: number): GameEvent[] {
  const isMonster = seat === MONSTER_SEAT;
  return events.filter(e => e.vis === 'all' || (e.vis === 'monster' ? isMonster : !isMonster));
}

export const viewDist = dist;
export const classStats = (c: ClassId) => CLASSES[c];
