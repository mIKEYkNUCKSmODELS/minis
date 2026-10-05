import { describe, expect, it } from 'vitest';
import {
  CLASSES, MONSTER_IDS, applyAction, createGame, eventsFor, makeBot, playBotGame, seatsToAct,
  validate, viewFor, type Action, type GameState, type Setup,
} from '../src/index';

const squad = (n: number) =>
  (['vanguard', 'arcblade', 'riftweaver', 'lifebinder', 'chronoStalker'] as const)
    .slice(0, n).map((cls, i) => ({ seat: i + 1, cls }));

const setup = (over: Partial<Setup> = {}): Setup => ({ monster: 'revenant', objective: 'fight', seed: 7, hunters: squad(4), ...over });

/** Replay an action list from scratch through the pure API. */
function replay(s: Setup, actions: Action[]): GameState {
  let { state } = createGame(s);
  for (const a of actions) state = applyAction(state, a).state;
  return state;
}

describe('determinism', () => {
  it('same seed + same actions = identical game', () => {
    for (const monster of MONSTER_IDS) {
      const s = setup({ monster, seed: 1234 });
      const a = playBotGame(s, { record: true });
      const b = playBotGame(s, { record: true });
      expect(b.actions).toEqual(a.actions);
      expect(b.state).toEqual(a.state);
      expect(replay(s, a.actions)).toEqual(a.state);
    }
  });

  it('state survives a JSON round-trip mid-game', () => {
    const s = setup({ seed: 99, objective: 'flight' });
    const { actions, state: final } = playBotGame(s, { record: true });
    const half = Math.floor(actions.length / 2);
    const mid = JSON.parse(JSON.stringify(replay(s, actions.slice(0, half))));
    let state = mid as GameState;
    for (const a of actions.slice(half)) state = applyAction(state, a).state;
    expect(state).toEqual(final);
  });

  it('different seeds give different games', () => {
    const a = playBotGame(setup({ seed: 1 }), { record: true });
    const b = playBotGame(setup({ seed: 2 }), { record: true });
    expect(a.actions).not.toEqual(b.actions);
  });
});

describe('purity', () => {
  it('applyAction never mutates its input', () => {
    const { state } = createGame(setup());
    const before = JSON.stringify(state);
    const seat = seatsToAct(state)[0];
    applyAction(state, makeBot(seat, 1).act(viewFor(state, seat)));
    expect(JSON.stringify(state)).toBe(before);
  });
});

describe('hidden information', () => {
  it('Hunters never see a hidden Monster square; the real square is among the signals', () => {
    for (let seed = 1; seed <= 40; seed++) {
      const s = setup({ seed });
      let { state } = createGame(s);
      const bots = new Map<number, ReturnType<typeof makeBot>>();
      while (state.phase !== 'over') {
        for (const h of state.hunters) {
          const v = viewFor(state, h.seat);
          if (state.monster.hidden) {
            expect(v.monster.pos).toBeNull();
            expect(v.monster.attackDice).toBeNull();
            expect(v.hunters.every(x => x.attackDice === null)).toBe(true);
            expect(v.signals.some(p => p.x === state.monster.pos.x && p.y === state.monster.pos.y)).toBe(true);
            // The real square must not be identifiable by order.
            const sorted = [...v.signals].sort((a, b) => a.y - b.y || a.x - b.x);
            expect(v.signals).toEqual(sorted);
          } else {
            expect(v.monster.pos).toEqual(state.monster.pos);
            expect(v.signals).toEqual([]);
          }
        }
        const seat = seatsToAct(state)[0];
        if (!bots.has(seat)) bots.set(seat, makeBot(seat, seed));
        const r = applyAction(state, bots.get(seat)!.act(viewFor(state, seat)));
        // Hidden Monster moves never reach Hunter seats.
        if (state.monster.hidden) for (const e of eventsFor(r.events, 1)) {
          if (e.type === 'move') expect(e.seat).not.toBe(0);
          if (e.type === 'bloodlust') throw new Error('hidden bloodlust leaked to a Hunter');
        }
        state = r.state;
      }
    }
  });

  it('the Monster cannot see a hidden Hunter more than 2 squares away', () => {
    const { state } = createGame(setup());
    const h = state.hunters[0];
    h.hidden = true;
    h.pos = { x: state.monster.pos.x, y: Math.min(state.N - 1, state.monster.pos.y + 3) };
    expect(viewFor(state, 0).hunters[0].pos).toBeNull();
    h.pos = { x: state.monster.pos.x, y: state.monster.pos.y + 2 };
    expect(viewFor(state, 0).hunters[0].pos).toEqual(h.pos);
    h.hidden = false;
    h.pos = { x: state.monster.pos.x, y: Math.min(state.N - 1, state.monster.pos.y + 6) };
    expect(viewFor(state, 0).hunters[0].pos).toEqual(h.pos);
  });

  it('views never contain the RNG state', () => {
    const { state } = createGame(setup());
    for (const seat of [0, 1, 2, 3, 4]) expect(JSON.stringify(viewFor(state, seat))).not.toContain('"rng"');
  });
});

describe('locked rules', () => {
  it('Monster actions = Hunters - 1 (change 2)', () => {
    for (const k of [3, 4, 5]) {
      let { state } = createGame(setup({ hunters: squad(k), monster: 'greedMaw' }));
      for (const h of state.hunters) state = applyAction(state, { seat: h.seat, kind: 'endTurn' }).state;
      expect(state.phase).toBe('monster');
      expect(state.monster.actionsLeft).toBe(k - 1);
    }
  });

  it('squads must be 3-5 Hunters (change 8)', () => {
    expect(() => createGame(setup({ hunters: squad(2) }))).toThrow();
    expect(() => createGame(setup({ hunters: [...squad(5), { seat: 6, cls: 'arcblade' }] }))).toThrow();
  });

  it('Monster needs 5+ and Hunters roll +1 die (changes 1 and 4)', () => {
    let { state } = createGame(setup({ monster: 'greedMaw' }));
    state.monster.hidden = false;
    const arc = state.hunters.find(h => h.cls === 'arcblade')!;
    arc.pos = { x: state.monster.pos.x, y: state.monster.pos.y + 1 };
    state.high = []; state.corrupt = state.corrupt.map(() => 0);
    const r = applyAction(state, { seat: arc.seat, kind: 'attack' });
    const atk = r.events.find(e => e.type === 'attack')!;
    expect(atk.type === 'attack' && atk.dice.length).toBe(CLASSES.arcblade.dice + 1);
    expect(atk.type === 'attack' && atk.need).toBe(4);
    expect(state.rules.monsterHit).toBe(5);
  });

  it('Vanguard guards only once a round (change 7)', () => {
    let { state } = createGame(setup({ monster: 'greedMaw', hunters: squad(3) }));
    const [van, arc] = state.hunters;
    state.monster.hidden = false;
    state.monster.pos = { x: 5, y: 5 };
    arc.pos = { x: 5, y: 6 }; van.pos = { x: 6, y: 6 };
    for (const h of state.hunters) state = applyAction(state, { seat: h.seat, kind: 'endTurn' }).state;
    // Force hits: roll until the Monster lands damage, counting guards in this round.
    let guards = 0;
    for (let i = 0; i < state.monster.actionsLeft + 1 && state.phase === 'monster'; i++) {
      const target = state.hunters.find(h => h.cls === 'arcblade')!;
      if (target.down) break;
      const r = applyAction(state, { seat: 0, kind: 'attack', target: target.seat });
      guards += r.events.filter(e => e.type === 'guard').length;
      state = r.state;
    }
    expect(guards).toBeLessThanOrEqual(1);
  });

  it('a downed Hunter can be revived once, then not again (change 5)', () => {
    let { state } = createGame(setup({ hunters: squad(3) }));
    const [a, b] = state.hunters;
    b.down = true; b.hp = 0; b.pos = { x: a.pos.x, y: a.pos.y - 1 };
    expect(validate(state, { seat: a.seat, kind: 'revive', target: b.seat })).toBeNull();
    state = applyAction(state, { seat: a.seat, kind: 'revive', target: b.seat }).state;
    const b2 = state.hunters.find(h => h.seat === b.seat)!;
    expect(b2.down).toBe(false);
    expect(b2.hp).toBe(1);
    b2.down = true; b2.hp = 0;
    expect(validate(state, { seat: a.seat, kind: 'revive', target: b.seat })).toBe('cannot revive');
  });

  it('Fury Mode: +1 Monster die when the Hunters chose Fight (change 6)', () => {
    const f = createGame(setup({ objective: 'fight' })).state;
    const g = createGame(setup({ objective: 'flight' })).state;
    for (const s of [f, g]) { s.high = []; s.corrupt = s.corrupt.map(() => 0); }
    expect(viewFor(f, 0).monster.attackDice! - viewFor(g, 0).monster.attackDice!).toBe(1);
  });
});

describe('legality', () => {
  it('rejects moves beyond the move stat and actions out of turn', () => {
    const { state } = createGame(setup());
    const h = state.hunters[0];
    const far = { x: h.pos.x, y: h.pos.y - (CLASSES[h.cls].move + 1) };
    expect(validate(state, { seat: h.seat, kind: 'move', to: far })).toBe('too far');
    expect(validate(state, { seat: 0, kind: 'endTurn' })).toBe('not the Monster phase');
    expect(validate(state, { seat: h.seat, kind: 'attack' })).toBe('the Monster is hidden');
  });

  it('bots only ever choose legal actions and every game ends (2,000 games)', () => {
    let n = 0;
    for (let seed = 1; seed <= 400; seed++) for (const monster of MONSTER_IDS) {
      const { state } = playBotGame({ monster, seed, objective: seed % 2 ? 'fight' : 'flight', hunters: squad(3 + (seed % 3)) });
      expect(state.phase).toBe('over');
      expect(state.result).not.toBeNull();
      n++;
    }
    expect(n).toBe(2000);
  });

  it('downed and cocooned Hunters cannot act; squads move inside the live board', () => {
    for (let seed = 1; seed <= 60; seed++) {
      let { state } = createGame(setup({ seed, monster: 'greedMaw' }));
      const bots = new Map<number, ReturnType<typeof makeBot>>();
      while (state.phase !== 'over') {
        for (const h of state.hunters) if (h.down || h.cocoon) expect(h.actionsLeft).toBe(0);
        const seat = seatsToAct(state)[0];
        if (!bots.has(seat)) bots.set(seat, makeBot(seat, seed));
        state = applyAction(state, bots.get(seat)!.act(viewFor(state, seat))).state;
        for (const h of state.hunters) {
          expect(h.pos.x).toBeGreaterThanOrEqual(state.lo); expect(h.pos.x).toBeLessThanOrEqual(state.hi);
        }
      }
    }
  });
});
