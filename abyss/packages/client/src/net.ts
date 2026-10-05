import { Client, type Room } from '@colyseus/sdk';
import type { GamePacket, LobbySnapshot } from '../../server/src/table';
export type { GamePacket, LobbySnapshot };

const endpoint = (import.meta.env.VITE_SERVER as string | undefined)
  ?? `${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}`;
const TOKEN_KEY = 'abyss.reconnect';

export interface Net {
  room: Room;
  send(type: string, msg?: unknown): void;
}

export interface Handlers {
  lobby(l: LobbySnapshot): void;
  game(p: GamePacket): void;
  error(message: string): void;
  status(s: 'online' | 'reconnecting' | 'gone'): void;
}

function wire(room: Room, h: Handlers): Net {
  try { sessionStorage.setItem(TOKEN_KEY, room.reconnectionToken); } catch { /* private mode */ }
  room.onMessage('lobby', h.lobby);
  room.onMessage('game', h.game);
  room.onMessage('error', (m: { message: string }) => h.error(m.message));
  room.onDrop(() => h.status('reconnecting'));
  room.onReconnect(() => { h.status('online'); try { sessionStorage.setItem(TOKEN_KEY, room.reconnectionToken); } catch { /* */ } });
  room.onLeave(() => { h.status('gone'); try { sessionStorage.removeItem(TOKEN_KEY); } catch { /* */ } });
  return { room, send: (t, m) => room.send(t, m ?? {}) };
}

const client = () => new Client(endpoint);

export async function createHunt(name: string, h: Handlers) {
  return wire(await client().create('hunt', { name }), h);
}

export async function joinHunt(code: string, name: string, h: Handlers) {
  return wire(await client().joinById(code.trim().toUpperCase(), { name }), h);
}

/** After a page refresh, try to take the same seat back. */
export async function resumeHunt(h: Handlers): Promise<Net | null> {
  let token: string | null = null;
  try { token = sessionStorage.getItem(TOKEN_KEY); } catch { /* */ }
  if (!token) return null;
  try {
    const net = wire(await client().reconnect(token), h);
    net.send('sync');
    return net;
  } catch {
    try { sessionStorage.removeItem(TOKEN_KEY); } catch { /* */ }
    return null;
  }
}
