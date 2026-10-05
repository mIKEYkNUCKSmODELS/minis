// ABYSS Online server: Colyseus rooms over WebSockets + the built phone client over HTTP.
//   PORT=2567 npm start      then open http://localhost:2567
import { existsSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { Server } from '@colyseus/core';
import { WebSocketTransport } from '@colyseus/ws-transport';
import express from 'express';
import { HuntRoom } from './room';

const here = dirname(fileURLToPath(import.meta.url));
const clientDist = join(here, '../../client/dist');
const port = Number(process.env.PORT ?? 2567);

const transport = new WebSocketTransport();
const app = transport.getExpressApp();
app.get('/healthz', (_req, res) => { res.json({ ok: true }); });
if (existsSync(clientDist)) {
  app.use(express.static(clientDist, { index: 'index.html', maxAge: '1h' }));
} else {
  app.get('/', (_req, res) => { res.type('text').send('Client not built. Run: npm run build'); });
}

const server = new Server({ transport, greet: false });
server.define('hunt', HuntRoom);
await server.listen(port);
console.log(`ABYSS Online listening on http://localhost:${port}`);
