// Seeded PRNG (mulberry32). The whole generator state is one uint32, so it lives
// inside GameState and survives JSON round-trips: same seed + same actions = same game.

export function nextFloat(state: { rng: number }): number {
  let a = (state.rng = (state.rng + 0x6d2b79f5) | 0);
  let t = Math.imul(a ^ (a >>> 15), 1 | a);
  t ^= t + Math.imul(t ^ (t >>> 7), 61 | t);
  return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
}

/** Integer in [lo, hi] inclusive. */
export function randInt(state: { rng: number }, lo: number, hi: number): number {
  return lo + Math.floor(nextFloat(state) * (hi - lo + 1));
}

export function pick<T>(state: { rng: number }, items: readonly T[]): T {
  return items[Math.floor(nextFloat(state) * items.length)];
}

export function shuffle<T>(state: { rng: number }, items: T[]): T[] {
  for (let i = items.length - 1; i > 0; i--) {
    const j = Math.floor(nextFloat(state) * (i + 1));
    [items[i], items[j]] = [items[j], items[i]];
  }
  return items;
}

/** Turns any string or number into a uint32 seed. */
export function seedFrom(input: string | number): number {
  const s = String(input);
  let h = 2166136261;
  for (let i = 0; i < s.length; i++) h = Math.imul(h ^ s.charCodeAt(i), 16777619);
  return h >>> 0;
}

/** A standalone RNG for bots and tools (never touches game state). */
export class Rng {
  rng: number;
  constructor(seed: number) { this.rng = seed >>> 0; }
  float() { return nextFloat(this); }
  int(lo: number, hi: number) { return randInt(this, lo, hi); }
  pick<T>(items: readonly T[]) { return pick(this, items); }
}
