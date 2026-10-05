// PixiJS board. Everything is drawn from code (no third-party art, canon rule 11).
import { Application, Container, Graphics, Text } from 'pixi.js';
import type { Pos, SeatView, HunterView } from '@abyss/engine';
import { CLASS_COLOR, CLASS_GLYPH } from './theme';

type Tok = { node: Container; tx: number; ty: number; seen: boolean };

export class BoardView {
  onTap: (p: Pos) => void = () => {};
  private app!: Application;
  private tiles = new Graphics();
  private rot = new Graphics();
  private marks = new Graphics();
  private hl = new Graphics();
  private tokens = new Container();
  private fx = new Container();
  private toks = new Map<string, Tok>();
  private view: SeatView | null = null;
  private you: number | null = null;
  private cell = 40;
  private highlights: { cells: Pos[]; kind: 'move' | 'target' } = { cells: [], kind: 'move' };
  private t = 0;
  private shake = 0;

  static async create(el: HTMLElement): Promise<BoardView> {
    const b = new BoardView();
    b.app = new Application();
    await b.app.init({ resizeTo: el, background: '#07060b', antialias: true, autoDensity: true, resolution: Math.min(window.devicePixelRatio || 1, 2) });
    el.appendChild(b.app.canvas);
    const world = new Container();
    world.addChild(b.tiles, b.rot, b.marks, b.hl, b.tokens, b.fx);
    b.app.stage.addChild(world);
    b.app.stage.eventMode = 'static';
    b.app.stage.hitArea = b.app.screen;
    b.app.stage.on('pointertap', e => {
      if (!b.view) return;
      const x = Math.floor(e.global.x / b.cell), y = Math.floor(e.global.y / b.cell);
      if (x >= 0 && y >= 0 && x < b.view.N && y < b.view.N) b.onTap({ x, y });
    });
    b.app.ticker.add(tk => {
      b.t += tk.deltaMS / 1000;
      const k = 1 - Math.pow(0.0005, tk.deltaMS / 1000); // smooth glide
      for (const tok of b.toks.values()) {
        tok.node.x += (tok.tx - tok.node.x) * k;
        tok.node.y += (tok.ty - tok.node.y) * k;
      }
      for (const [id, tok] of b.toks) if (id.startsWith('sig') || id.startsWith('heart')) tok.node.alpha = 0.55 + 0.35 * Math.sin(b.t * 3 + tok.tx);
      b.hl.alpha = 0.65 + 0.25 * Math.sin(b.t * 5);
      if (b.shake > 0) { b.shake = Math.max(0, b.shake - tk.deltaMS / 400); world.x = (Math.random() - .5) * 8 * b.shake; world.y = (Math.random() - .5) * 8 * b.shake; }
      else { world.x = 0; world.y = 0; }
      for (const f of [...b.fx.children]) { f.alpha -= tk.deltaMS / 900; f.y -= tk.deltaMS / 40; if (f.alpha <= 0) f.destroy(); }
    });
    new ResizeObserver(() => { if (b.view) b.render(); }).observe(el);
    return b;
  }

  setView(v: SeatView, you: number | null) {
    this.view = v; this.you = you;
    this.render();
  }

  setHighlights(cells: Pos[], kind: 'move' | 'target') {
    this.highlights = { cells, kind };
    this.drawHighlights();
  }

  /** Floating text over a square, plus an optional screen shake. */
  pop(p: Pos, text: string, color: string, shake = false) {
    const c = this.cell;
    const t = new Text({ text, style: { fontFamily: 'system-ui', fontSize: Math.max(14, c * 0.42), fontWeight: '800', fill: color, stroke: { color: '#000', width: 4 } } });
    t.anchor.set(0.5);
    t.x = (p.x + 0.5) * c; t.y = (p.y + 0.2) * c;
    this.fx.addChild(t);
    if (shake) this.shake = 1;
  }

  private render() {
    const v = this.view!;
    this.cell = Math.floor(Math.min(this.app.screen.width, this.app.screen.height) / v.N);
    this.drawTiles();
    this.drawHighlights();
    this.drawTokens();
  }

  private drawTiles() {
    const v = this.view!, c = this.cell, N = v.N;
    const g = this.tiles.clear();
    const high = new Set(v.high), cover = new Set(v.cover);
    for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
      const i = y * N + x;
      const base = (x + y) % 2 ? 0x15111f : 0x120e1a;
      g.rect(x * c, y * c, c, c).fill(high.has(i) ? 0x2a2440 : cover.has(i) ? 0x10201c : base);
    }
    // grid lines
    for (let k = 0; k <= N; k++) {
      g.moveTo(k * c, 0).lineTo(k * c, N * c).moveTo(0, k * c).lineTo(N * c, k * c);
    }
    g.stroke({ width: 1, color: 0x221a31, alpha: 0.9 });

    const m = this.marks.clear();
    for (const i of v.high) { // high ground: a ridge chevron
      const x = (i % N) * c, y = Math.floor(i / N) * c;
      m.moveTo(x + c * .22, y + c * .72).lineTo(x + c * .5, y + c * .36).lineTo(x + c * .78, y + c * .72).stroke({ width: Math.max(1.5, c * .06), color: 0x8f86b8, alpha: .55 });
    }
    for (const i of v.cover) { // cover: fog wisps
      const x = (i % N) * c, y = Math.floor(i / N) * c;
      for (const [dx, dy, r] of [[.3, .4, .13], [.55, .55, .17], [.72, .38, .1]] as const) m.circle(x + c * dx, y + c * dy, c * r).fill({ color: 0x6fb5a0, alpha: .16 });
    }

    const r = this.rot.clear();
    v.corrupt.forEach((on, i) => {
      if (!on) return;
      const x = (i % N) * c, y = Math.floor(i / N) * c;
      r.rect(x + 1, y + 1, c - 2, c - 2).fill({ color: 0x7a2bd1, alpha: .32 });
      r.moveTo(x + c * .15, y + c * .8).bezierCurveTo(x + c * .4, y + c * .3, x + c * .6, y + c * .9, x + c * .85, y + c * .2)
        .stroke({ width: Math.max(1, c * .05), color: 0xb46bff, alpha: .55 });
    });
    // Sudden Death: everything outside the live ring is gone.
    for (let y = 0; y < N; y++) for (let x = 0; x < N; x++)
      if (x < v.lo || x > v.hi || y < v.lo || y > v.hi) r.rect(x * c, y * c, c, c).fill({ color: 0x000000, alpha: .78 });
  }

  private drawHighlights() {
    if (!this.view) return;
    const c = this.cell, g = this.hl.clear();
    const color = this.highlights.kind === 'move' ? 0xe6c74c : 0xff5168;
    for (const p of this.highlights.cells) {
      g.roundRect(p.x * c + 3, p.y * c + 3, c - 6, c - 6, c * .18).stroke({ width: Math.max(2, c * .07), color });
      if (this.highlights.kind === 'move') g.roundRect(p.x * c + 3, p.y * c + 3, c - 6, c - 6, c * .18).fill({ color, alpha: .1 });
    }
  }

  private tok(id: string, make: () => Container, p: Pos): Tok {
    const c = this.cell;
    let tok = this.toks.get(id);
    const tx = (p.x + .5) * c, ty = (p.y + .5) * c;
    if (!tok) {
      const node = make();
      node.x = tx; node.y = ty;
      this.tokens.addChild(node);
      tok = { node, tx, ty, seen: true };
      this.toks.set(id, tok);
    }
    tok.tx = tx; tok.ty = ty; tok.seen = true;
    return tok;
  }

  private drawTokens() {
    const v = this.view!, c = this.cell;
    for (const t of this.toks.values()) t.seen = false;
    // Tokens are rebuilt each render (cheap at this size) but keep their animated position.
    const rebuild = (id: string, p: Pos, draw: (node: Container) => void) => {
      const tok = this.tok(id, () => new Container(), p);
      tok.node.removeChildren().forEach(ch => ch.destroy());
      draw(tok.node);
    };

    v.hearts.forEach((h, i) => rebuild(`heart${h.pos.x},${h.pos.y}`, h.pos, n => {
      const g = new Graphics();
      const s = c * .34;
      g.poly([0, -s, s, 0, 0, s, -s, 0]).fill(0xe2384d).stroke({ width: 2, color: 0xffb3bd });
      g.circle(0, 0, s * 1.3).stroke({ width: 2, color: 0xe2384d, alpha: .35 });
      for (let k = 0; k < h.hp; k++) g.circle((k - (h.hp - 1) / 2) * c * .16, s + c * .1, c * .05).fill(0xffd3d9);
      n.addChild(g);
      void i;
    }));

    v.minions.forEach(mi => rebuild(`minion${mi.id}`, mi.pos, n => {
      const g = new Graphics();
      const s = c * .2;
      g.poly([0, -s, s, s * .8, -s, s * .8]).fill(0x9b4dff).stroke({ width: 1.5, color: 0x2a0f45 });
      n.addChild(g);
    }));

    v.signals.forEach((p, i) => rebuild(`sig${i}`, p, n => {
      const g = new Graphics();
      g.circle(0, 0, c * .34).fill({ color: 0x7a2bd1, alpha: .28 }).stroke({ width: 2, color: 0xc7a6ff, alpha: .8 });
      const q = new Text({ text: '?', style: { fontFamily: 'Georgia, serif', fontSize: c * .44, fontWeight: '700', fill: '#e8dcff' } });
      q.anchor.set(.5);
      n.addChild(g, q);
    }));

    const m = v.monster;
    if (m.pos) rebuild('monster', m.pos, n => {
      const g = new Graphics();
      const r = c * .44;
      if (m.hidden) g.circle(0, 0, r * 1.15).stroke({ width: 2, color: 0xc7a6ff, alpha: .6 }); // the Monster sees itself as hidden
      g.circle(0, 0, r).fill(m.enraged ? 0xff2a45 : 0xb3253a).stroke({ width: 3, color: 0x2a0710 });
      g.ellipse(0, 0, r * .62, r * .34).fill(0xfff1c9);
      g.ellipse(0, 0, r * .12, r * .3).fill(0x1a0006);
      for (let k = 0; k < 6; k++) { // spines
        const a = (k / 6) * Math.PI * 2 + .3;
        g.moveTo(Math.cos(a) * r, Math.sin(a) * r).lineTo(Math.cos(a) * r * 1.28, Math.sin(a) * r * 1.28).stroke({ width: 2.5, color: 0xe2384d });
      }
      n.addChild(g);
      n.alpha = m.hidden ? .75 : 1;
    });

    for (const h of v.hunters) if (h.pos) rebuild(`h${h.seat}`, h.pos, n => this.drawHunter(n, h));

    for (const [id, t] of [...this.toks]) if (!t.seen) { t.node.destroy({ children: true }); this.toks.delete(id); }
    // Keep the Monster and the player's own token on top.
    const top = (id: string) => { const t = this.toks.get(id); if (t) this.tokens.addChild(t.node); };
    top('monster'); if (this.you !== null) top(`h${this.you}`);
  }

  private drawHunter(n: Container, h: HunterView) {
    const c = this.cell, r = c * .36;
    const g = new Graphics();
    const color = h.down ? 0x4a4458 : CLASS_COLOR[h.cls];
    if (h.seat === this.you) g.circle(0, 0, r + c * .08).stroke({ width: 3, color: 0xe6c74c });
    g.circle(0, 0, r).fill(color).stroke({ width: 2.5, color: 0x0b0912 });
    if (h.cocoon > 0) for (let k = -2; k <= 2; k++) g.moveTo(-r, k * r * .35).lineTo(r, -k * r * .35).stroke({ width: 1.5, color: 0xf2e9d0, alpha: .8 });
    n.addChild(g);
    const t = new Text({ text: h.down ? '✕' : CLASS_GLYPH[h.cls], style: { fontFamily: 'system-ui', fontSize: c * .36, fontWeight: '800', fill: h.down ? '#bbb' : '#0b0912' } });
    t.anchor.set(.5);
    n.addChild(t);
    if (!h.down) {
      const pips = new Graphics();
      for (let k = 0; k < h.maxHp; k++)
        pips.roundRect((k - h.maxHp / 2) * c * .14 + 1, r + c * .02, c * .12, c * .07, 2).fill(k < h.hp ? 0x57d18a : 0x2c2340);
      n.addChild(pips);
    }
    n.alpha = h.hidden ? .5 : 1;
  }
}
