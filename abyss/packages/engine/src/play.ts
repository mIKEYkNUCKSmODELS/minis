// Drives a whole game with bots in every seat. Used by the simulator and the tests.
import { applyActionMut, createGame, seatsToAct } from './engine';
import { makeBot, type Bot } from './bots';
import { viewFor } from './view';
import type { Action, GameState, Setup } from './types';

export interface PlayedGame { state: GameState; actions: Action[] }

export function playBotGame(setup: Setup, opts: { record?: boolean; maxActions?: number } = {}): PlayedGame {
  const { state } = createGame(setup);
  const bots = new Map<number, Bot>();
  const botFor = (seat: number) => {
    let b = bots.get(seat);
    if (!b) bots.set(seat, (b = makeBot(seat, setup.seed)));
    return b;
  };
  const actions: Action[] = [];
  const max = opts.maxActions ?? 5000;
  while (state.phase !== 'over') {
    if (actions.length >= max) throw new Error(`game did not finish in ${max} actions (seed ${setup.seed})`);
    // Simultaneous Hunter turns: offline we resolve seats in seat order, one action at a time per seat,
    // finishing a Hunter's actions before the next (matches the 20260925 simulator).
    const seat = seatsToAct(state)[0];
    const a = botFor(seat).act(viewFor(state, seat));
    applyActionMut(state, a);
    actions.push(a);
  }
  return { state, actions: opts.record ? actions : [] };
}
