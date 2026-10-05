// Turns engine events into short lines for the log. Kept terse: it scrolls on a phone.
import { CLASSES, type GameEvent, type HunterView, type SeatView } from '@abyss/engine';

export function narrate(e: GameEvent, v: SeatView, names: Map<number, string>): string | null {
  const who = (seat: number) => {
    if (seat === 0) return `<b>${v.monster.name}</b>`;
    const h = v.hunters.find((x: HunterView) => x.seat === seat);
    return `<b>${esc(names.get(seat) ?? `Hunter ${seat}`)}</b>${h ? ` (${CLASSES[h.cls].name})` : ''}`;
  };
  switch (e.type) {
    case 'round': return `— Round ${e.round} —`;
    case 'abyssCard':
      if (e.card === 'creepingRot') return 'The Abyss stirs: <b>Creeping Rot</b> spreads.';
      if (e.card === 'temporalReversal') return `Temporal Reversal mends ${who(e.target!)}.`;
      if (e.card === 'lostMoment') return `Lost Moment: ${who(e.target!)} loses an action.`;
      return null;
    case 'attack': return `${who(e.seat)} strikes ${e.target === 0 ? who(0) : who(e.target)}: <b>${e.hits}</b> hit${e.hits === 1 ? '' : 's'}.`;
    case 'guard': return `${who(e.seat)} guards ${who(e.for)}.`;
    case 'down': return `${who(e.seat)} is <b>down</b>.`;
    case 'enrage': return `${who(0)} is <b>enraged</b> and heals.`;
    case 'revealed': return `${who(0)} is <b>revealed</b>!`;
    case 'vanished': return `${who(0)} melts back into the dark.`;
    case 'scan': return e.found ? null : `${who(e.seat)} scans: nothing${e.cleared ? `, ${e.cleared} false trail${e.cleared === 1 ? '' : 's'} cleared` : ''}.`;
    case 'hide': return `${who(e.seat)} hides.`;
    case 'heal': return `${who(e.seat)} heals ${who(e.target)}.`;
    case 'slow': return `${who(e.seat)} slows ${who(0)}.`;
    case 'rescue': return `${who(e.seat)} cuts ${who(e.target)} free!`;
    case 'revive': return `${who(e.seat)} revives ${who(e.target)}!`;
    case 'cleanse': return e.hp <= 0 ? `${who(e.seat)} <b>cleanses a Heart</b>!` : `${who(e.seat)} wounds a Heart.`;
    case 'cocoon': return `${who(0)} cocoons ${who(e.target)}!`;
    case 'cocoonDeath': return `${who(e.seat)} is lost in the cocoon.`;
    case 'snare': return `${who(0)} snares ${who(e.target)} in time.`;
    case 'summon': return `${who(0)} births ${e.minions.length} minions.`;
    case 'minionAttack': return e.hits ? `A minion bites ${who(e.target)}.` : null;
    case 'bloodlust': return `${who(0)} is drawn by <b>Bloodlust</b>.`;
    case 'suddenDeath': return '<b>Sudden Death</b>: the arena collapses inward.';
    case 'gameOver': return e.result === 'hunters' ? '<b>The Hunters win.</b>' : e.result === 'monster' ? `<b>${v.monster.name} wins.</b>` : '<b>Draw.</b>';
    default: return null;
  }
}

export const esc = (s: string) => s.replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]!));
