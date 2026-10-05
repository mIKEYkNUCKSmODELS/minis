import { describe, expect, it } from 'vitest';
import { Table, type GamePacket, type LobbySnapshot } from '../src/table';

/** A fake clock: scheduled callbacks run when we say so. */
function harness(seed = 0.42) {
  const queue: { fn: () => void; dead: boolean }[] = [];
  const lobbies = new Map<string, LobbySnapshot>();
  const games = new Map<string, GamePacket[]>();
  const errors: string[] = [];
  const table = new Table({
    sendLobby: (sid, l) => lobbies.set(sid, l),
    sendGame: (sid, p) => { if (!games.has(sid)) games.set(sid, []); games.get(sid)!.push(p); },
    sendError: (_sid, m) => errors.push(m),
    schedule: (fn) => { const t = { fn, dead: false }; queue.push(t); return { clear: () => { t.dead = true; } }; },
    random: () => seed,
  }, { code: 'TEST1', botDelayMs: { hunter: 1, monster: 1 } });
  const tick = () => { const t = queue.shift(); if (t && !t.dead) t.fn(); return !!t; };
  const last = (sid: string) => games.get(sid)!.at(-1)!;
  return { table, queue, lobbies, games, errors, tick, last };
}

describe('hunt table', () => {
  it('one human Hunter + the Abyss AI everywhere else plays a full hunt', () => {
    const h = harness();
    h.table.join('mike', 'Mike');
    h.table.sit('mike', 1);
    h.table.pickClass('mike', 'chronoStalker');
    h.table.vote('mike', 'fight');
    h.table.start('mike');
    expect(h.table.phase).toBe('playing');
    expect(h.table.seats.slice(0, 6).filter(s => s.kind === 'bot')).toHaveLength(5);

    let guard = 0;
    while (h.table.phase === 'playing' && guard++ < 10000) {
      const p = h.last('mike');
      const me = p.view.hunters.find(x => x.seat === 1)!;
      // The human never sees the hidden Revenant.
      if (p.view.monster.hidden) expect(p.view.monster.pos).toBeNull();
      if (p.view.phase === 'hunters' && me.actionsLeft > 0) h.table.act('mike', { kind: 'endTurn' });
      else if (!h.tick()) break;
    }
    expect(h.table.phase).toBe('over');
    expect(h.last('mike').reveal?.monster.pos).not.toBeNull();
    expect(h.table.log.length).toBeGreaterThan(5);
  });

  it('rejects acting for someone else or out of turn', () => {
    const h = harness();
    h.table.join('a', 'A'); h.table.join('b', 'B');
    h.table.sit('a', 1); h.table.sit('b', 0);
    h.table.start('b');
    expect(h.errors).toContain('Only the host can start.');
    h.table.start('a');
    // Seat is forced from the session, so 'b' (the Monster) cannot move Hunter 1.
    h.table.act('b', { kind: 'endTurn', seat: 1 });
    expect(h.errors).toContain('not the Monster phase');
  });

  it('a player who leaves mid-hunt is replaced by the Abyss AI and the hunt finishes', () => {
    const h = harness(0.77);
    h.table.join('a', 'A');
    h.table.sit('a', 2);
    h.table.start('a');
    h.table.leave('a');
    expect(h.table.seats[2].kind).toBe('bot');
    let n = 0;
    while (h.table.phase === 'playing' && n++ < 10000) if (!h.tick()) break;
    expect(h.table.phase).toBe('over');
  });

  it('lobby rules: squad size, seat limits, one seat per player, names sanitised', () => {
    const h = harness();
    h.table.join('a', '<script>Ana</script>');
    expect(h.lobbies.get('a')!.players[0].name).toBe('scriptAnascript');
    h.table.setSquad('a', 3);
    h.table.sit('a', 5);
    expect(h.errors).toContain('That seat is not at this table.');
    h.table.sit('a', 1); h.table.sit('a', 2);
    expect(h.table.seats.filter(s => s.sessionId === 'a')).toHaveLength(1);
    h.table.start('a');
    expect(h.table.game!.hunters).toHaveLength(3);
    expect(h.table.game!.monster.id).toBe('revenant');
  });

  it('the Monster player sees the Revenant; unseated watchers get only Hunter knowledge', () => {
    const h = harness();
    h.table.join('m', 'M'); h.table.join('w', 'Watcher');
    h.table.sit('m', 0);
    h.table.start('m');
    expect(h.last('m').view.monster.pos).not.toBeNull();
    expect(h.last('w').view.monster.pos).toBeNull();
    expect(h.last('w').you).toBeNull();
  });
});
