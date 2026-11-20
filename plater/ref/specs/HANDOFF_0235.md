# HANDOFF CLAUDE 2026-11-20 02:35 CT - plater: 32-bed run, heat maps, cockpit

Supersedes nothing; adds to HANDOFF CLAUDE 20261119-1705 and its addenda. Chat 42aaf7d0 ("Swarm agents for chat handoffs") never wrote the handoff Mike asked for at its end; this is it, written by the next chat.

## Mike's words this stretch (CT)
- 11/19 23:53: "I think that filled plate creates suctions cups because it is too full."
- 11/20 ~02:00: "run some bundles through it now and show me the plates do 32"
- 11/20 ~02:10: "I want a place that will actually print and succeed... I do want to see the heat zone that UV Tools actually creates from each plate."
- 11/20 02:13: "You are driving and should have a bundle of 32 plates running soon. take that over."
- 11/20 02:15: "make sure the plater cockpit is ready and shows the heat maps for hotspots on plates regarding peel load and the peel front per plate"

## What exists now
1. 32-bed plater run (started by chat 42aaf7d0 at 02:15:56, finished in 72 s): C:\MKM_TMP\full_plates_20261120\ (run_full.cmd, run.log, out\P*\recipe.json + PNG + RING_MANIFEST.json).
   - Settings: MKM_FULLNESS_MM2=3260, GB_LEGACY_PLACEHOLDERS=1 (3 mm gap, 3 mm edge), GB_NPOOL=32, GB_NOEXPORT=1, GB_VERDICT=defer.
   - Result: 26 plates written, 6 beds left empty (4, 12, 18, 20, 21, 25), 0 failed, 1 piece carried (TEST5_HAND_RING_1110, too big for the allotment).
   - Planner change behind it (gated, off unless the env var is set): gpu_scan\mkm_gpupool10.py, 2 lines; backup mkm_gpupool10.py.backup_20261120-0220_before_fullness_allotment; undo C:\MKM_TMP\full_plates_20261120\UNDO_fullness.cmd.
2. Heat maps (this chat): C:\MKM_TMP\plate_heat_20261120\plate_heat.py + run_heat.cmd; output full32\HEAT_P*.png + heat_summary.json + heat.log. Read-only on the plater: slices each plate from its recipe.json with the plater's own GPU slicer (mkm_gpuport.GSlicer + gfill), 0.25 mm grid, 0.5 mm steps, about 10 s a plate.
   - Colour = layers cured at each spot (full height). Blue = the peel ring, mm in from the bed edge. Bar chart = cured area the ring meets per 5 mm of travel. Curve = cured area per layer up the print against the 3463 line.
   - Usage: run_heat.cmd <planner out folder> <new output folder> 3463
3. Cockpit: F:\MKM_STAGE\PLATER_COCKPIT\build_cockpit.py has a new section "PLATE HEAT MAPS" (shows the newest heat_summary.json under C:\MKM_TMP\plate_heat_20261120\). Backup build_cockpit.py.backup_20261120-0225_before_heatmaps; undo UNDO_heatmaps.cmd.

## Results
- All 26 plates are under the fullness line (3463 mm2 cured per layer over 1.8-30 mm). Highest: P7 3292, P14 3248, P3 3215, P1 3173.
- NEW: the fullness allotment only budgets 1.8-30 mm. Higher up, plates get fuller. Fullest 28 mm stretch per plate: P7 6095 (35-63 mm), P6 5165 (38-66 mm), P14 4371, P9 4271, P1 4152, P16 4027.
- Same measure on the sliced printed plates (C:\MKM_TMP\plate_heat_20261120\known_worst.py, from uvtools_bridge_20261120\gpu_*.json): clean prints reached 4832 (FE2_02), 4585 (FE2_03), 4117 (FE2_01); failed plates run 3140 to 6485. So this measure does NOT separate clean from failed (overlap 3873-4832); it is only a "has anything this full ever printed clean" check. P6 and P7 are fuller there than any clean print.
- The 3463 line is fitted to the 19 printed plates (18 of 19 held out). Observed, not a law. Nothing here is sliced, scanned or sent.

## Not done
- No plate sliced in Chitubox, no UVtools scan, nothing sent. Mike inspects every plate before anything goes out.
- UVtools' own detectors (suction cup, resin trap, island: gpu_scan\mkm_uvtools_gpu.py) need a sliced file; the heat maps here are from the layout, not from a slice.
- Fullness allotment does not yet cover the height above 30 mm (P6, P7 show why). Proposal, not applied: budget the fullest 28 mm stretch too.
- Sparse plates: P3 has 3 pieces at 26% footprint because its cured-area allotment is used up. Conflicts with Mike's no-sparse-plates rule; his call which wins.
- Jupiter 2 / Saturn 3, 4, 4 Ultra slice readers: not started.
- Laptop free RAM was 1.7-3.5 GB during this (Chrome about 4.4 GB). Plater loop feeder has been paused since 01:39 for that reason.
- Still pending from 11/19: thermal printer admin prompt (JD-268BT queue still on COM5), Y41BT driver, leveller pick, which pieces failed on the Juggernaut plate.
