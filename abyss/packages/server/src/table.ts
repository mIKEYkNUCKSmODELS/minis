// A hunt table: lobby, seats, AI seats and the authoritative game. No networking in here;
// the Colyseus room wires it to sockets, and the tests drive it directly.

import {
  CLASS_IDS, MAX_SEATS, MONSTER_SEAT, Rng, applyAction, createGame, draftClasses, eventsFor, makeBot,
  seatsToAct, validate, viewFor, voteObjective,
  type Action, type Bot, type ClassId, type GameEvent, type GameState, type Look, type MonsterId, type Objective, type SeatView,
} from '@abyss/engine';

export type SeatKind = 'empty' | 'human' | 'bot';

export interface Seat {
  seat: number;
  kind: SeatKind;
  name: string;
  sessionId: string | null;
  connected: boolean;
  cls: ClassId | null;
  look: Look;
  vote: Objective | null;
}

export interface LobbySnapshot {
  code: string;
  phase: 'lobby' | 'playing' | 'over';
  hostSessionId: string | null;
  monster: MonsterId;
  squad: number;
  seats: Seat[];
  players: { sessionId: string; name: string; seat: number | null; connected: boolean }[];
}

export interface GamePacket {
  you: number | null;
  view: SeatView;
  events: GameEvent[];
  seats: { seat: number; name: string; kind: SeatKind; connected: boolean }[];
  /** Full board revealed to everyone once the hunt is over. */
  reveal: SeatView | null;
}

export interface TableIO {
  sendLobby(sessionId: string, lobby: LobbySnapshot): void;
  sendGame(sessionId: string, packet: GamePacket): void;
  sendError(sessionId: string, message: string): void;
  schedule(fn: () => void, ms: number): { clear(): void };
  random(): number; // seed source
}

export interface TableOptions {
  code: string;
  monster?: MonsterId;
  allowedMonsters?: MonsterId[];
  /** Delays that make AI turns readable. */
  botDelayMs?: { hunter: number; monster: number };
}

const NAME_MAX = 16;
export const cleanName = (n: unknown) =>
  (typeof n === 'string' ? n : '').replace(/[^\p{L}\p{N} _.'-]/gu, '').trim().slice(0, NAME_MAX) || 'Hunter';

export class Table {
  readonly code: string;
  phase: 'lobby' | 'playing' | 'over' = 'lobby';
  monster: MonsterId;
  squad = 5; // 6 seats by default: the Monster + 5 Hunters
  seats: Seat[];
  hostSessionId: string | null = null;
  players = new Map<string, { name: string; connected: boolean }>();
  game: GameState | null = null;
  private bots = new Map<number, Bot>();
  private pending = new Map<number, { clear(): void }>();
  private allowed: MonsterId[];
  private delay: { hunter: number; monster: number };
  /** Actions so far, so a finished hunt can be replayed or audited. */
  readonly log: Action[] = [];

  constructor(private io: TableIO, opts: TableOptions) {
    this.code = opts.code;
    this.allowed = opts.allowedMonsters ?? ['revenant'];
    this.monster = opts.monster && this.allowed.includes(opts.monster) ? opts.monster : this.allowed[0];
    this.delay = opts.botDelayMs ?? { hunter: 650, monster: 900 };
    this.seats = Array.from({ length: MAX_SEATS }, (_, seat) => ({
      seat, kind: 'empty', name: '', sessionId: null, connected: false, cls: null,
      look: { timeline: 0, sex: seat % 2 ? 'F' : 'M' }, vote: null,
    }));
  }

  // ---------------------------------------------------------------- players
  join(sessionId: string, name: unknown) {
    this.players.set(sessionId, { name: cleanName(name), connected: true });
    if (!this.hostSessionId) this.hostSessionId = sessionId;
    this.broadcast();
  }

  seatOf(sessionId: string): Seat | undefined {
    return this.seats.find(s => s.sessionId === sessionId);
  }

  setConnected(sessionId: string, connected: boolean) {
    const p = this.players.get(sessionId);
    if (p) p.connected = connected;
    const s = this.seatOf(sessionId);
    if (s) s.connected = connected;
    this.broadcast();
  }

  /** A player is gone for good. In the lobby their seat frees up; mid-hunt the Abyss AI takes it over. */
  leave(sessionId: string) {
    const s = this.seatOf(sessionId);
    if (s) {
      if (this.phase === 'lobby') this.clearSeat(s);
      else { s.kind = 'bot'; s.sessionId = null; s.connected = false; s.name = `${s.name} (AI)`; this.wakeBots(); }
    }
    this.players.delete(sessionId);
    if (this.hostSessionId === sessionId) this.hostSessionId = [...this.players.keys()][0] ?? null;
    this.broadcast();
  }

  private clearSeat(s: Seat) {
    s.kind = 'empty'; s.name = ''; s.sessionId = null; s.connected = false; s.cls = null; s.vote = null;
  }

  // ---------------------------------------------------------------- lobby
  sit(sessionId: string, seat: number) {
    if (this.phase !== 'lobby') return this.fail(sessionId, 'The hunt has started.');
    const target = this.seats[seat];
    if (!target || seat > this.squad) return this.fail(sessionId, 'That seat is not at this table.');
    if (target.kind === 'human' && target.sessionId !== sessionId) return this.fail(sessionId, 'Seat taken.');
    const mine = this.seatOf(sessionId);
    if (mine && mine !== target) this.clearSeat(mine);
    const p = this.players.get(sessionId)!;
    target.kind = 'human'; target.sessionId = sessionId; target.name = p.name; target.connected = p.connected;
    if (seat !== MONSTER_SEAT && !target.cls) target.cls = this.freeClass();
    this.broadcast();
  }

  stand(sessionId: string) {
    if (this.phase !== 'lobby') return;
    const s = this.seatOf(sessionId);
    if (s) this.clearSeat(s);
    this.broadcast();
  }

  pickClass(sessionId: string, cls: unknown, look?: Partial<Look>) {
    const s = this.seatOf(sessionId);
    if (this.phase !== 'lobby' || !s || s.seat === MONSTER_SEAT) return;
    if (typeof cls === 'string' && (CLASS_IDS as string[]).includes(cls)) s.cls = cls as ClassId;
    if (look) {
      if (look.timeline === 0 || look.timeline === 1 || look.timeline === 2) s.look = { ...s.look, timeline: look.timeline };
      if (look.sex === 'M' || look.sex === 'F') s.look = { ...s.look, sex: look.sex };
    }
    this.broadcast();
  }

  vote(sessionId: string, objective: unknown) {
    const s = this.seatOf(sessionId);
    if (this.phase !== 'lobby' || !s || s.seat === MONSTER_SEAT) return;
    if (objective === 'fight' || objective === 'flight') s.vote = objective;
    this.broadcast();
  }

  setSquad(sessionId: string, n: unknown) {
    if (sessionId !== this.hostSessionId || this.phase !== 'lobby') return;
    if (typeof n !== 'number' || n < 3 || n > 5) return;
    this.squad = n;
    for (const s of this.seats) if (s.seat > n && s.kind !== 'empty') this.clearSeat(s);
    this.broadcast();
  }

  setMonster(sessionId: string, m: unknown) {
    if (sessionId !== this.hostSessionId || this.phase !== 'lobby') return;
    if (typeof m === 'string' && (this.allowed as string[]).includes(m)) this.monster = m as MonsterId;
    this.broadcast();
  }

  private freeClass(): ClassId {
    const taken = new Set(this.seats.map(s => s.cls));
    return CLASS_IDS.find(c => !taken.has(c)) ?? CLASS_IDS[0];
  }

  start(sessionId: string) {
    if (sessionId !== this.hostSessionId) return this.fail(sessionId, 'Only the host can start.');
    if (this.phase !== 'lobby') return;
    if (!this.seats.some(s => s.kind === 'human')) return this.fail(sessionId, 'Take a seat first.');
    const seed = Math.floor(this.io.random() * 2 ** 32) >>> 0;
    const rng = new Rng(seed ^ 0x5bd1e995);

    // The Abyss AI fills every empty seat.
    const used = this.seats.slice(0, this.squad + 1);
    const drafted = draftClasses(this.squad, rng);
    for (const s of used) {
      if (s.kind !== 'empty') continue;
      s.kind = 'bot'; s.name = s.seat === MONSTER_SEAT ? 'Abyss AI' : `Abyss AI ${s.seat}`;
      if (s.seat !== MONSTER_SEAT) {
        const taken = new Set(used.map(x => x.cls));
        const want = drafted[s.seat - 1];
        s.cls = taken.has(want) ? (CLASS_IDS.find(c => !taken.has(c)) ?? want) : want;
        s.look = { timeline: rng.int(0, 2) as Look['timeline'], sex: rng.float() < 0.5 ? 'M' : 'F' };
      }
    }
    const hunters = used.filter(s => s.seat !== MONSTER_SEAT);
    const team = hunters.map(s => s.cls!);
    // Hunters vote Fight or Flight; AI Hunters vote by squad firepower; a tie goes to Fight.
    const aiVote = voteObjective(team);
    let fight = 0, flight = 0;
    for (const s of hunters) ((s.kind === 'human' ? s.vote ?? aiVote : aiVote) === 'fight' ? fight++ : flight++);
    const objective: Objective = fight >= flight ? 'fight' : 'flight';

    this.game = createGame({
      monster: this.monster, objective, seed,
      hunters: hunters.map(s => ({ seat: s.seat, cls: s.cls!, look: s.look })),
    }).state;
    this.phase = 'playing';
    for (const s of used) if (s.kind === 'bot') this.bots.set(s.seat, makeBot(s.seat, seed));
    this.broadcast([]);
    this.wakeBots();
  }

  // ---------------------------------------------------------------- play
  act(sessionId: string, action: unknown) {
    const s = this.seatOf(sessionId);
    if (!this.game || this.phase !== 'playing' || !s || s.kind !== 'human') return this.fail(sessionId, 'Not your turn.');
    if (!action || typeof action !== 'object') return this.fail(sessionId, 'Bad action.');
    const a = { ...(action as Action), seat: s.seat } as Action;
    const err = validate(this.game, a);
    if (err) return this.fail(sessionId, err);
    this.apply(a);
  }

  private apply(a: Action) {
    const r = applyAction(this.game!, a);
    this.game = r.state;
    this.log.push(a);
    if (this.game.phase === 'over') {
      this.phase = 'over';
      for (const p of this.pending.values()) p.clear();
      this.pending.clear();
    }
    this.broadcast(r.events);
    this.wakeBots();
  }

  /** Schedule one action for every AI seat that can act now. Hunter AIs act side by side (simultaneous turns). */
  private wakeBots() {
    if (!this.game || this.phase !== 'playing') return;
    for (const seat of seatsToAct(this.game)) {
      const s = this.seats[seat];
      if (s.kind !== 'bot' || this.pending.has(seat)) continue;
      if (!this.bots.has(seat)) this.bots.set(seat, makeBot(seat, this.game.seed));
      const ms = seat === MONSTER_SEAT ? this.delay.monster : this.delay.hunter;
      this.pending.set(seat, this.io.schedule(() => {
        this.pending.delete(seat);
        if (!this.game || this.phase !== 'playing' || !seatsToAct(this.game).includes(seat)) return this.wakeBots();
        this.apply(this.bots.get(seat)!.act(viewFor(this.game, seat)));
      }, ms));
    }
  }

  // ---------------------------------------------------------------- output
  lobby(): LobbySnapshot {
    return {
      code: this.code, phase: this.phase, hostSessionId: this.hostSessionId, monster: this.monster, squad: this.squad,
      seats: this.seats.map(s => ({ ...s, look: { ...s.look } })),
      players: [...this.players].map(([sessionId, p]) => ({ sessionId, name: p.name, seat: this.seatOf(sessionId)?.seat ?? null, connected: p.connected })),
    };
  }

  packetFor(sessionId: string, events: GameEvent[] = []): GamePacket | null {
    if (!this.game) return null;
    const s = this.seatOf(sessionId);
    const seat = s ? s.seat : 99; // unseated watchers get the Hunters' knowledge, never the Monster's
    return {
      you: s ? s.seat : null,
      view: viewFor(this.game, seat),
      events: eventsFor(events, seat),
      seats: this.seats.slice(0, this.squad + 1).map(x => ({ seat: x.seat, name: x.name, kind: x.kind, connected: x.connected })),
      reveal: this.game.phase === 'over' ? viewFor(this.game, -1) : null,
    };
  }

  private broadcast(events: GameEvent[] = []) {
    const lobby = this.lobby();
    for (const sid of this.players.keys()) {
      this.io.sendLobby(sid, lobby);
      const p = this.packetFor(sid, events);
      if (p) this.io.sendGame(sid, p);
    }
  }

  /** Re-send everything to one client (after a reconnect). */
  resync(sessionId: string) {
    this.io.sendLobby(sessionId, this.lobby());
    const p = this.packetFor(sessionId);
    if (p) this.io.sendGame(sessionId, p);
  }

  private fail(sessionId: string, message: string) {
    this.io.sendError(sessionId, message);
  }

  dispose() {
    for (const p of this.pending.values()) p.clear();
    this.pending.clear();
  }
}
