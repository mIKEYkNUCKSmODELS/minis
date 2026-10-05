import type { ClassId, MonsterId, Objective, Rules } from './rules';

export interface Pos { x: number; y: number }

export type Timeline = 0 | 1 | 2;
export type Sex = 'M' | 'F';
/** Cosmetic only: 5 classes x 3 timelines x M/F. */
export interface Look { timeline: Timeline; sex: Sex }

export interface HunterState {
  seat: number;
  cls: ClassId;
  look: Look;
  hp: number;
  maxHp: number;
  pos: Pos;
  hidden: boolean;
  down: boolean;
  cocoon: number;       // rounds left inside a Greed Maw cocoon (0 = free)
  cocoonDead: boolean;  // died in a cocoon: cannot be revived
  revived: boolean;     // already used their one revive
  lostAction: number;   // actions lost next Hunter phase
  guardedRound: number; // last round this Vanguard guarded
  slowCd: number;       // Riftweaver Slow cooldown
  actionsLeft: number;
}

export interface MonsterState {
  id: MonsterId;
  hp: number;
  maxHp: number;
  dice: number;
  range: number;
  move: number;
  stealthy: boolean;
  pos: Pos;
  hidden: boolean;
  slowed: number;
  enraged: boolean;
  firstKill: boolean;
  snareCd: number;
  cocoonCd: number;
  actionsLeft: number;
  phantomStepUsed: boolean;
  summonUsed: boolean;
}

export interface Heart { pos: Pos; hp: number }
export interface Minion { id: number; pos: Pos }

export type Phase = 'hunters' | 'monster' | 'over';
export type Result = 'hunters' | 'monster' | 'draw' | null;

export interface GameState {
  v: 1;
  rules: Rules;
  seed: number;
  rng: number;
  N: number;
  lo: number;  // live board bounds (Sudden Death shrinks them)
  hi: number;
  high: number[];    // cell indices (y * N + x)
  cover: number[];
  corrupt: number[]; // 0/1 per cell
  hearts: Heart[];
  phantoms: Pos[];
  minions: Minion[];
  nextMinionId: number;
  objective: Objective;
  round: number;
  quiet: number;
  quietHit: boolean;
  phase: Phase;
  result: Result;
  hunters: HunterState[];
  monster: MonsterState;
  stats: Record<string, number>;
}

export interface SetupHunter { seat: number; cls: ClassId; look?: Look }
export interface Setup {
  monster: MonsterId;
  hunters: SetupHunter[];
  objective: Objective;
  seed: number;
  rules?: Partial<Rules>;
}

// ------------------------------------------------------------------ actions
export type HunterAction =
  | { seat: number; kind: 'move'; to: Pos }
  | { seat: number; kind: 'attack' }
  | { seat: number; kind: 'scan' }
  | { seat: number; kind: 'hide' }
  | { seat: number; kind: 'heal'; target: number }
  | { seat: number; kind: 'slow' }
  | { seat: number; kind: 'rescue'; target: number }
  | { seat: number; kind: 'revive'; target: number }
  | { seat: number; kind: 'cleanse'; heart: number }
  | { seat: number; kind: 'endTurn' };

export type MonsterAction =
  | { seat: 0; kind: 'move'; to: Pos }
  | { seat: 0; kind: 'phantomStep'; to: Pos }
  | { seat: 0; kind: 'attack'; target: number }
  | { seat: 0; kind: 'cocoon'; target: number }
  | { seat: 0; kind: 'snare'; target: number }
  | { seat: 0; kind: 'summon' }
  | { seat: 0; kind: 'corrupt' }
  | { seat: 0; kind: 'endTurn' };

export type Action = HunterAction | MonsterAction;

// ------------------------------------------------------------------ events
/** Who may see an event: everyone, only the Monster seat, or only Hunter seats. */
export type Visibility = 'all' | 'monster' | 'hunters';

export type GameEvent = { vis: Visibility } & (
  | { type: 'round'; round: number }
  | { type: 'phase'; phase: Phase }
  | { type: 'abyssCard'; card: 'creepingRot' | 'calm' | 'temporalReversal' | 'lostMoment'; target?: number }
  | { type: 'corruptionSpread'; cells: number[] }
  | { type: 'specterScrub'; seat: number; cell: number }
  | { type: 'move'; seat: number; from: Pos; to: Pos }
  | { type: 'attack'; seat: number; target: number; dice: number[]; need: number; hits: number }
  | { type: 'guard'; seat: number; for: number }
  | { type: 'damage'; seat: number; amount: number; hp: number }
  | { type: 'down'; seat: number }
  | { type: 'enrage'; hp: number }
  | { type: 'revealed'; by: 'scan' | 'attack'; pos: Pos }
  | { type: 'vanished' }
  | { type: 'scan'; seat: number; found: boolean; cleared: number }
  | { type: 'hide'; seat: number }
  | { type: 'heal'; seat: number; target: number }
  | { type: 'slow'; seat: number }
  | { type: 'rescue'; seat: number; target: number }
  | { type: 'revive'; seat: number; target: number }
  | { type: 'cleanse'; seat: number; heart: number; hp: number }
  | { type: 'cocoon'; target: number }
  | { type: 'cocoonDeath'; seat: number }
  | { type: 'snare'; target: number }
  | { type: 'summon'; minions: Minion[] }
  | { type: 'minionAttack'; minion: number; target: number; dice: number[]; hits: number }
  | { type: 'minionSwatted'; minion: number; seat: number }
  | { type: 'bloodlust'; to: Pos }
  | { type: 'suddenDeath'; lo: number; hi: number }
  | { type: 'endTurn'; seat: number }
  | { type: 'gameOver'; result: Exclude<Result, null> }
);

export interface ApplyResult {
  state: GameState;
  events: GameEvent[];
}
