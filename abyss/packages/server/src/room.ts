import { Room, type Client } from '@colyseus/core';
import { Table } from './table';

const CODE_ALPHABET = 'ABCDEFGHJKMNPQRSTUVWXYZ23456789'; // no 0/O, 1/I/L
export const RECONNECT_SECONDS = 120;
const live = new Set<string>();

export function makeCode(rand = Math.random): string {
  for (;;) {
    let c = '';
    for (let i = 0; i < 5; i++) c += CODE_ALPHABET[Math.floor(rand() * CODE_ALPHABET.length)];
    if (!live.has(c)) return c;
  }
}

const botDelay = () => {
  const ms = Number(process.env.ABYSS_BOT_DELAY_MS);
  return Number.isFinite(ms) && ms > 0 ? { hunter: ms, monster: ms } : undefined;
};

/** One hunt. The room id is the 5-letter invite code. */
export class HuntRoom extends Room {
  override maxClients = 12; // 6 seats plus friends watching
  table!: Table;

  override onCreate(options: { monster?: string } = {}) {
    this.roomId = makeCode();
    live.add(this.roomId);
    this.table = new Table({
      sendLobby: (sid, lobby) => this.clients.get(sid)?.send('lobby', lobby),
      sendGame: (sid, packet) => this.clients.get(sid)?.send('game', packet),
      sendError: (sid, message) => this.clients.get(sid)?.send('error', { message }),
      schedule: (fn, ms) => this.clock.setTimeout(fn, ms),
      random: Math.random,
    }, { code: this.roomId, monster: options.monster as never, botDelayMs: botDelay() });

    const on = (type: string, fn: (sid: string, msg: any) => void) =>
      this.onMessage(type, (client: Client, msg: any) => fn(client.sessionId, msg ?? {}));
    on('sit', (sid, m) => this.table.sit(sid, Number(m.seat)));
    on('stand', sid => this.table.stand(sid));
    on('class', (sid, m) => this.table.pickClass(sid, m.cls, m.look));
    on('vote', (sid, m) => this.table.vote(sid, m.objective));
    on('squad', (sid, m) => this.table.setSquad(sid, Number(m.squad)));
    on('monster', (sid, m) => this.table.setMonster(sid, m.monster));
    on('start', sid => { this.table.start(sid); if (this.table.phase !== 'lobby') this.lock(); });
    on('act', (sid, m) => this.table.act(sid, m.action));
    on('sync', sid => this.table.resync(sid));
  }

  override onJoin(client: Client, options: { name?: string } = {}) {
    this.table.join(client.sessionId, options.name);
  }

  override async onDrop(client: Client) {
    this.table.setConnected(client.sessionId, false);
    await this.allowReconnection(client, RECONNECT_SECONDS);
  }

  override onReconnect(client: Client) {
    this.table.setConnected(client.sessionId, true);
    this.table.resync(client.sessionId);
  }

  override onLeave(client: Client) {
    this.table.leave(client.sessionId);
  }

  override onDispose() {
    this.table.dispose();
    live.delete(this.roomId);
  }
}
