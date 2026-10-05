// The Abyss AI. Bots read only their own seat's view (never GameState), so they cannot cheat.
// Behaviour is ported from abyss_sim.py (20260925) so the balance numbers stay comparable.
// Difficulty: "steady". Simple, consistent, no bluffing.

import { CLASSES, CLASS_IDS, MONSTER_SEAT, type ClassId, type Objective } from './rules';
import { Rng } from './rng';
import { dist, samePos, stepToward } from './engine';
import type { Action, HunterAction, MonsterAction, Pos } from './types';
import type { HunterView, SeatView } from './view';

const activeView = (h: HunterView) => !h.down && h.cocoon === 0;

export interface Bot { seat: number; act(view: SeatView): Action }

export class HunterBot implements Bot {
  private guess: Pos | null = null;
  constructor(public seat: number, private rng: Rng) {}

  act(v: SeatView): HunterAction {
    const seat = this.seat;
    const me = v.hunters.find(h => h.seat === seat)!;
    const pos = me.pos!;
    const c = CLASSES[me.cls];
    const m = v.monster;
    const mPos = m.pos; // null while hidden

    // 1. Cut a cocooned ally free.
    for (const o of v.hunters)
      if (o.cocoon > 0 && !o.down && o.pos && dist(o.pos, pos) <= 1) return { seat, kind: 'rescue', target: o.seat };
    // 1b. Revive a downed ally, or walk to one.
    if (v.rules.revive) {
      const revivable = v.hunters.filter(o => o.down && !o.revived && !o.cocoonDead && o.pos);
      for (const o of revivable) if (dist(o.pos!, pos) <= 1) return { seat, kind: 'revive', target: o.seat };
      const dn = revivable.filter(o => dist(o.pos!, pos) <= 3);
      if (dn.length && (!mPos || dist(mPos, pos) > m.range)) return { seat, kind: 'move', to: stepToward(pos, dn[0].pos!, c.move) };
    }
    // 2. Lifebinder heals the weakest nearby ally.
    if (me.cls === 'lifebinder') {
      const hurt = v.hunters.filter(o => activeView(o) && o.pos && o.hp <= o.maxHp - 1 && o.hp <= 2 && dist(o.pos, pos) <= 2);
      if (hurt.length) return { seat, kind: 'heal', target: hurt.reduce((b, o) => (o.hp < b.hp ? o : b)).seat };
    }
    // 3. Hide when fragile and threatened.
    if (me.hp === 1 && v.cover.includes(pos.y * v.N + pos.x) && !me.hidden && mPos && dist(pos, mPos) <= 3)
      return { seat, kind: 'hide' };
    // 4. Riftweaver slows the Monster.
    if (me.cls === 'riftweaver' && mPos && dist(pos, mPos) <= 4 && me.slowCd <= 0) return { seat, kind: 'slow' };
    // Flight: cleanse Hearts first.
    if (v.objective === 'flight' && v.hearts.length) {
      let ti = 0;
      v.hearts.forEach((h, i) => { if (dist(h.pos, pos) < dist(v.hearts[ti].pos, pos)) ti = i; });
      const target = v.hearts[ti].pos;
      const threat = !!mPos && dist(mPos, pos) <= m.range;
      if (dist(target, pos) <= 1 && !(threat && c.range === 1 && this.rng.float() < 0.5)) return { seat, kind: 'cleanse', heart: ti };
      if (!threat) return { seat, kind: 'move', to: stepToward(pos, target, c.move) };
    }
    // Monster visible: fight.
    if (mPos) {
      const d = dist(pos, mPos);
      if (d <= c.range) return { seat, kind: 'attack' };
      // (abyss_sim.py let this step exceed the Hunter's move stat; the engine enforces it, so we cap.)
      if (c.range >= 3 && d <= m.range + m.move) return { seat, kind: 'move', to: stepToward(pos, mPos, Math.min(c.move, Math.max(0, d - c.range))) };
      return { seat, kind: 'move', to: stepToward(pos, mPos, Math.min(c.move, d - c.range)) };
    }
    // Monster hidden: scan a nearby signal, or walk to the one we're chasing.
    const cands = v.signals;
    if (!cands.length) return { seat, kind: 'endTurn' };
    if (cands.some(p => dist(p, pos) <= c.scan)) return { seat, kind: 'scan' };
    if (!this.guess || !cands.some(p => samePos(p, this.guess!))) this.guess = this.rng.pick(cands);
    return { seat, kind: 'move', to: stepToward(pos, this.guess, c.move) };
  }
}

export class MonsterBot implements Bot {
  seat = MONSTER_SEAT;
  constructor(private rng: Rng) {}

  private pickTarget(v: SeatView): HunterView | null {
    const mPos = v.monster.pos!;
    let cand = v.hunters.filter(h => activeView(h) && h.pos);
    if (v.objective === 'flight') {
      const nearHeart = cand.filter(h => v.hearts.some(hh => dist(h.pos!, hh.pos) <= 1));
      if (nearHeart.length) cand = nearHeart;
    }
    if (!cand.length) return null;
    const score = (h: HunterView) => dist(h.pos!, mPos) + h.hp * 1.5;
    return cand.reduce((b, h) => (score(h) < score(b) ? h : b));
  }

  act(v: SeatView): MonsterAction {
    const m = v.monster;
    const mPos = m.pos!;
    if (m.id === 'revenant' && !m.phantomStepUsed) {
      const t = this.pickTarget(v);
      return { seat: 0, kind: 'phantomStep', to: t ? stepToward(mPos, t.pos!, 3) : { ...mPos } };
    }
    if (m.actionsLeft <= 0) return { seat: 0, kind: 'endTurn' };
    if (m.id === 'hiveQueen' && v.round % 3 === 1 && !m.summonUsed) return { seat: 0, kind: 'summon' };
    const t = this.pickTarget(v);
    if (m.id === 'chronophage' && t && m.snareCd <= 0 && dist(t.pos!, mPos) <= 4) return { seat: 0, kind: 'snare', target: t.seat };
    if (!t) return { seat: 0, kind: 'corrupt' };
    const d = dist(t.pos!, mPos);
    if (d <= m.range) {
      if (m.id === 'greedMaw' && t.hp <= 2 && d <= 1 && m.cocoonCd <= 0) return { seat: 0, kind: 'cocoon', target: t.seat };
      return { seat: 0, kind: 'attack', target: t.seat };
    }
    return { seat: 0, kind: 'move', to: stepToward(mPos, t.pos!, Math.min(m.move, d - m.range)) };
  }
}

export function makeBot(seat: number, seed: number): Bot {
  const rng = new Rng(seed ^ Math.imul(seat + 1, 0x9e3779b1));
  return seat === MONSTER_SEAT ? new MonsterBot(rng) : new HunterBot(seat, rng);
}

// ------------------------------------------------------------------ pre-game choices (AI seats)
/** Draw 2, keep 1 per seat (prefer a class the squad lacks); the Monster then forces a redraw of the last pick. */
export function draftClasses(k: number, rng: Rng): ClassId[] {
  const team: ClassId[] = [];
  for (let i = 0; i < k; i++) {
    const a = rng.pick(CLASS_IDS), b = rng.pick(CLASS_IDS);
    team.push(team.includes(a) ? b : a);
  }
  if (k) team[k - 1] = rng.pick(CLASS_IDS);
  return team;
}

/** Bot squad vote: Fight if the squad has enough firepower, else Flight. */
export function voteObjective(team: ClassId[]): Objective {
  const dice = team.reduce((n, c) => n + CLASSES[c].dice, 0);
  return dice >= 3.2 * team.length ? 'fight' : 'flight';
}
