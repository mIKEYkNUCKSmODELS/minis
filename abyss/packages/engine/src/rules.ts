// ABYSS locked rules.
// Source: abyss_sim.py build 20260925-161000-0500 (TUNED preset) + ABYSS_BALANCE_REPORT_20260925-161000-0500,
// locked by Mike in ABYSS_TASK_20260927-065800-0500 ("the 8 balance changes from the 20260925 simulator").
// Numbers marked ASSUMED were left open by the design doc; they are placeholders until human playtests.

export type ClassId = 'vanguard' | 'arcblade' | 'riftweaver' | 'lifebinder' | 'chronoStalker';
export type MonsterId = 'revenant' | 'hiveQueen' | 'chronophage' | 'greedMaw' | 'harbinger';
export type Objective = 'fight' | 'flight';

export interface ClassStats {
  name: string;
  role: string;
  hp: number;
  dice: number;  // base attack pool (before the +1 Hunter die)
  range: number; // attack range, Chebyshev squares
  move: number;
  armor: number;
  scan: number;  // scan radius
}

export const CLASSES: Record<ClassId, ClassStats> = {
  vanguard:      { name: 'Vanguard',       role: 'Tank. Guards an adjacent ally once a round.', hp: 4, dice: 3, range: 1, move: 2, armor: 1, scan: 3 },
  arcblade:      { name: 'Arcblade',       role: 'Melee striker.',                              hp: 3, dice: 4, range: 1, move: 3, armor: 0, scan: 3 },
  riftweaver:    { name: 'Riftweaver',     role: 'Support. Slows the Monster.',                 hp: 3, dice: 2, range: 3, move: 2, armor: 0, scan: 3 },
  lifebinder:    { name: 'Lifebinder',     role: 'Healer.',                                     hp: 3, dice: 2, range: 2, move: 2, armor: 0, scan: 3 },
  chronoStalker: { name: 'Chrono Stalker', role: 'Ranged tracker. Long scan.',                  hp: 3, dice: 3, range: 4, move: 2, armor: 0, scan: 5 },
};
export const CLASS_IDS = Object.keys(CLASSES) as ClassId[];

export interface MonsterStats {
  name: string;
  hp: number;
  dice: number;
  range: number;
  move: number;
  stealthy: boolean; // re-hides anywhere when no Hunter is within 3
  ability: string;
}

// Per-monster tuning (attack dice / HP) from the 20260925 report.
export const MONSTERS: Record<MonsterId, MonsterStats> = {
  revenant:    { name: 'Revenant',          hp: 18, dice: 4, range: 1, move: 3, stealthy: true,  ability: 'Phantom Step: a free 3-square move each turn that keeps it hidden.' },
  hiveQueen:   { name: 'Hive Queen',        hp: 22, dice: 5, range: 1, move: 3, stealthy: false, ability: 'Brood: every third round, spend an action to spawn 2 minions on corruption.' },
  chronophage: { name: 'Chronophage',       hp: 20, dice: 4, range: 2, move: 3, stealthy: false, ability: 'Time Snare (a Hunter within 4 loses an action) and Chrono Pulse (-1 die to Hunters within 2).' },
  greedMaw:    { name: 'Greed Maw',         hp: 20, dice: 4, range: 1, move: 3, stealthy: false, ability: 'Cocoon: swallow an adjacent Hunter at 2 HP or less; allies have 3 rounds to cut them free.' },
  harbinger:   { name: 'Abyssal Harbinger', hp: 20, dice: 3, range: 2, move: 5, stealthy: false, ability: 'Wings of Terror: Hunters off high ground roll 1 fewer die.' },
};
export const MONSTER_IDS = Object.keys(MONSTERS) as MonsterId[];

export interface Rules {
  board: number;
  highGroundPct: number;
  coverPct: number;
  hearts: number;
  heartHp: number;
  spreadPerHeart: number;
  hunterActions: number;
  /** Change 2: Monster actions = Hunters - 1. Keyed by squad size. */
  monsterActions: Record<number, number>;
  corruptionWin: number;
  flightSurviveRounds: number;
  suddenDeathFight: number;
  roundCap: number;
  bloodlustQuietRounds: number;
  phantoms: number;
  monsterHit: number;      // change 1
  hunterHit: number;
  hunterDiceBonus: number; // change 4
  revive: boolean;         // change 5
  fightFuryDice: number;   // change 6
  guardOncePerRound: boolean; // change 7
  minHunters: number;      // change 8
  maxHunters: number;
  enrageHeal: number;
  firstKillDice: number;
  /** Per-monster HP override (change 3: flat HP). */
  monsterHp: Partial<Record<MonsterId, number>>;
  monsterDice: Partial<Record<MonsterId, number>>;
}

export const LOCKED_RULES: Rules = {
  board: 12,                 // ASSUMED
  highGroundPct: 0.12,       // ASSUMED
  coverPct: 0.15,            // ASSUMED
  hearts: 3,                 // ASSUMED
  heartHp: 2,                // ASSUMED
  spreadPerHeart: 1,         // ASSUMED
  hunterActions: 2,
  monsterActions: { 3: 2, 4: 3, 5: 4 },
  corruptionWin: 0.8,
  flightSurviveRounds: 10,
  suddenDeathFight: 8,
  roundCap: 16,              // ASSUMED
  bloodlustQuietRounds: 2,
  phantoms: 2,
  monsterHit: 5,
  hunterHit: 4,
  hunterDiceBonus: 1,
  revive: true,
  fightFuryDice: 1,
  guardOncePerRound: true,
  minHunters: 3,
  maxHunters: 5,
  enrageHeal: 3,
  firstKillDice: 1,
  monsterHp: {},
  monsterDice: {},
};

/** The 6-seat table: seat 0 is the Monster, seats 1-5 are Hunters. */
export const MONSTER_SEAT = 0;
export const MAX_SEATS = 6;
