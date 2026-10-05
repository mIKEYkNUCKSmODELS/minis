# Status: parked (20261005-001833-0500)

Read `ABYSS_TASK_20261005-001833-0500 (DIRECTION LOCK)` in the ABYSS_ONLINE Drive folder first.

ABYSS is **a tabletop RPG played online between real home tables with physical minis**, built on WRAITH's ideas.
It is **not** a digital board game. The minis (trading, painting, 3D printing) are the focus.

- `packages/engine`, `packages/sim`: rules engine, AI, simulator. Reusable as the referee under the home-table design.
- `packages/server`, `packages/client`, `e2e`: parked. A browser board with flat tokens, built against an unlocked
  proposal. Keep only the reusable parts (lobby, reconnect, per-seat hidden info).

No further build work until the home-table design is agreed (EXPLAIN, then GO).
