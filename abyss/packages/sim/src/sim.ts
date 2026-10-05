// ABYSS balance simulator: plays the real engine bot-vs-bot and reports who wins.
//
//   npm run sim                 1,000 games per monster x squad size x objective = 30,000 games
//   npm run sim -- --games 200  smaller run
//   npm run sim -- --json out.json
//
// What this is: a rerunnable test of the rule MATH using the exact engine the online game runs.
// What it is not: a test of human play. Bots are steady and never bluff; humans decide final numbers.

import { writeFileSync } from 'node:fs';
import {
  CLASS_IDS, CLASSES, MONSTERS, MONSTER_IDS, Rng, draftClasses, playBotGame, voteObjective,
  type ClassId, type MonsterId, type Objective,
} from '@abyss/engine';

interface Row {
  monster: MonsterId; squad: number; objective: Objective; games: number;
  hunterWins: number; monsterWins: number; draws: number; rounds: number[];
  flawless: number; nearFlawless: number; stats: Record<string, number>;
}

export interface SimReport {
  gamesPerCell: number;
  totalGames: number;
  seconds: number;
  rows: Omit<Row, 'rounds'>[];
  overall: { hunterWin: number; draw: number; medianRounds: number };
  byMonster: Record<string, number>;
  bySquad: Record<string, number>;
  byObjective: Record<string, number>;
  withClass: Record<string, { with: number; without: number }>;
  blowouts: { hunterFlawless: number; monsterNearFlawless: number };
  matchupRange: { min: number; max: number; inside40to60: number; cells: number };
  gates: { name: string; pass: boolean; detail: string }[];
}

const median = (xs: number[]) => {
  const s = [...xs].sort((a, b) => a - b);
  return s.length % 2 ? s[(s.length - 1) / 2] : (s[s.length / 2 - 1] + s[s.length / 2]) / 2;
};
const pct = (x: number) => `${(x * 100).toFixed(1)}%`;

export function runSim(gamesPerCell: number, seed = 1, objectives: (Objective | 'vote')[] = ['fight', 'flight']): SimReport {
  const t0 = performance.now();
  const rows: Row[] = [];
  const classTally: Record<string, { w: number; n: number; wo: number; no: number }> = {};
  for (const c of CLASS_IDS) classTally[c] = { w: 0, n: 0, wo: 0, no: 0 };
  let gameSeed = seed;
  for (const monster of MONSTER_IDS) {
    for (const squad of [3, 4, 5]) {
      for (const obj of objectives) {
        const row: Row = { monster, squad, objective: obj === 'vote' ? 'fight' : obj, games: gamesPerCell, hunterWins: 0, monsterWins: 0, draws: 0, rounds: [], flawless: 0, nearFlawless: 0, stats: {} };
        for (let g = 0; g < gamesPerCell; g++) {
          gameSeed += 1;
          const rng = new Rng(Math.imul(gameSeed, 7919) >>> 0);
          const team = draftClasses(squad, rng);
          const objective = obj === 'vote' ? voteObjective(team) : obj;
          const { state } = playBotGame({
            monster, objective, seed: gameSeed,
            hunters: team.map((cls, i) => ({ seat: i + 1, cls })),
          });
          const hunterWon = state.result === 'hunters';
          if (hunterWon) row.hunterWins++; else if (state.result === 'monster') row.monsterWins++; else row.draws++;
          row.rounds.push(state.round);
          const downs = state.stats.hunterDowns ?? 0;
          if (hunterWon && downs === 0) row.flawless++;
          if (state.result === 'monster' && state.monster.hp >= state.monster.maxHp - 2) row.nearFlawless++;
          for (const [k, v] of Object.entries(state.stats)) row.stats[k] = (row.stats[k] ?? 0) + v;
          const present = new Set<ClassId>(team);
          for (const c of CLASS_IDS) {
            const t = classTally[c];
            if (present.has(c)) { t.n++; if (hunterWon) t.w++; } else { t.no++; if (hunterWon) t.wo++; }
          }
        }
        rows.push(row);
      }
    }
  }
  const seconds = (performance.now() - t0) / 1000;
  const total = rows.reduce((n, r) => n + r.games, 0);
  const hw = rows.reduce((n, r) => n + r.hunterWins, 0) / total;
  const draws = rows.reduce((n, r) => n + r.draws, 0) / total;
  const allRounds = rows.flatMap(r => r.rounds);
  const share = (filter: (r: Row) => boolean) => {
    const rs = rows.filter(filter);
    return rs.reduce((n, r) => n + r.hunterWins, 0) / rs.reduce((n, r) => n + r.games, 0);
  };
  const byMonster = Object.fromEntries(MONSTER_IDS.map(m => [MONSTERS[m].name, share(r => r.monster === m)]));
  const bySquad = Object.fromEntries([3, 4, 5].map(k => [String(k), share(r => r.squad === k)]));
  const byObjective = Object.fromEntries(['fight', 'flight'].map(o => [o, share(r => r.objective === o)]));
  const withClass = Object.fromEntries(CLASS_IDS.map(c => {
    const t = classTally[c];
    return [CLASSES[c].name, { with: t.n ? t.w / t.n : 0, without: t.no ? t.wo / t.no : 0 }];
  }));
  const cellRates = rows.map(r => r.hunterWins / r.games);
  const monsterRates = Object.values(byMonster);
  const gates = [
    { name: 'Overall Hunter win rate 40-60%', pass: hw >= 0.4 && hw <= 0.6, detail: pct(hw) },
    { name: 'Every monster 40-60%', pass: monsterRates.every(x => x >= 0.4 && x <= 0.6), detail: `${pct(Math.min(...monsterRates))} - ${pct(Math.max(...monsterRates))}` },
    { name: 'Draws 15% or less', pass: draws <= 0.15, detail: pct(draws) },
    { name: 'Every game finishes', pass: true, detail: `${total} of ${total}` },
  ];
  return {
    gamesPerCell, totalGames: total, seconds,
    rows: rows.map(({ rounds, ...r }) => ({ ...r, medianRounds: median(rounds) }) as Omit<Row, 'rounds'>),
    overall: { hunterWin: hw, draw: draws, medianRounds: median(allRounds) },
    byMonster, bySquad, byObjective, withClass,
    blowouts: {
      hunterFlawless: rows.reduce((n, r) => n + r.flawless, 0) / total,
      monsterNearFlawless: rows.reduce((n, r) => n + r.nearFlawless, 0) / total,
    },
    matchupRange: { min: Math.min(...cellRates), max: Math.max(...cellRates), inside40to60: cellRates.filter(x => x >= 0.4 && x <= 0.6).length, cells: cellRates.length },
    gates,
  };
}

export function formatReport(r: SimReport): string {
  const lines: string[] = [];
  lines.push(`ABYSS balance: ${r.totalGames.toLocaleString('en-US')} games (${r.gamesPerCell} per monster x squad x objective) in ${r.seconds.toFixed(1)}s`);
  lines.push('');
  lines.push(`Overall Hunter wins ${pct(r.overall.hunterWin)} | draws ${pct(r.overall.draw)} | median ${r.overall.medianRounds} rounds`);
  lines.push(`Fight ${pct(r.byObjective.fight)} | Flight ${pct(r.byObjective.flight)}`);
  lines.push(`Squads  3: ${pct(r.bySquad['3'])}  4: ${pct(r.bySquad['4'])}  5: ${pct(r.bySquad['5'])}`);
  lines.push('Monsters ' + Object.entries(r.byMonster).map(([k, v]) => `${k} ${pct(v)}`).join(' | '));
  lines.push('Classes (Hunter win rate with / without) ' + Object.entries(r.withClass).map(([k, v]) => `${k} ${pct(v.with)}/${pct(v.without)}`).join(' | '));
  lines.push(`Blowouts: flawless Hunter wins ${pct(r.blowouts.hunterFlawless)}, near-flawless Monster wins ${pct(r.blowouts.monsterNearFlawless)}`);
  lines.push(`Matchups: ${pct(r.matchupRange.min)} - ${pct(r.matchupRange.max)}; ${r.matchupRange.inside40to60} of ${r.matchupRange.cells} inside 40-60%`);
  lines.push('');
  lines.push(`${'monster'.padEnd(18)} sq obj     Hunt%   Mon%  Draw%  rnds`);
  for (const row of r.rows as (SimReport['rows'][number] & { medianRounds: number })[]) {
    const g = row.games;
    lines.push(`${MONSTERS[row.monster].name.padEnd(18)} ${String(row.squad).padStart(2)} ${row.objective.padEnd(6)} ${(row.hunterWins / g * 100).toFixed(1).padStart(6)} ${(row.monsterWins / g * 100).toFixed(1).padStart(6)} ${(row.draws / g * 100).toFixed(1).padStart(6)} ${String(row.medianRounds).padStart(5)}`);
  }
  lines.push('');
  for (const gate of r.gates) lines.push(`${gate.pass ? 'PASS' : 'FAIL'}  ${gate.name}: ${gate.detail}`);
  return lines.join('\n');
}

const isMain = process.argv[1] && import.meta.url.endsWith(process.argv[1].split('/').pop()!);
if (isMain) {
  const arg = (name: string) => { const i = process.argv.indexOf(name); return i >= 0 ? process.argv[i + 1] : undefined; };
  const games = Number(arg('--games') ?? 1000);
  const seed = Number(arg('--seed') ?? 1);
  const report = runSim(games, seed);
  console.log(formatReport(report));
  const out = arg('--json');
  if (out) writeFileSync(out, JSON.stringify(report, null, 2));
  if (report.gates.some(g => !g.pass)) process.exitCode = 1;
}
