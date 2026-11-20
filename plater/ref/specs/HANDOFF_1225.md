# HANDOFF CLAUDE 20261120-1225 (read first) - start a FRESH chat from this, to save usage

Why this exists: Mike's order 12:20 "Bottom line I just want to save usage period." The 11/20 morning chat grew very long; every turn re-sends all of it. A new chat that reads only this file is far cheaper. Do NOT re-read the old chat unless a fact here is missing.

## What Mike wants (his words, today)
- Plates that print. Gap 4 mm, LOCKED (10:53: "a .5 mm gap would fail. Do a 4 mm gap and lock that in"; 0.5 mm "would make one giant suction cup").
- Layout is the only variable; film, resin, printer are constants, never revisited ("stop worrying about that").
- Go/no-go on every layer 0 to 100 percent, no fitted or borrowed numbers, no rule so strict he gets no plates. Then fix vents/resin traps in meshes, then swap the model, then flag that ONE plate; never stop the batch.
- The shrinking band (edge to centre) findings belong in go/no-go.
- 11:54: evaluate ONLY the near-miss plate (clean print, a few small failed pieces; the Ark of Grief spear plate) using ONLY geometry + the band + tap-video film tension (1056 N/m on a 352.38 x 210.12 mm span) and get a consensus on what happened.
- Maps: "dont make soft maps"; masters at 4x show size (4320x7680), downscaled, never upscaled. WRAITH must never depend on Unreal running; Unreal/Blender only make assets ahead of time. Pipeline: 3D build, then Grok paints over it. Using his terrain STLs as the room kit: he found that "interesting".
- Claude drives; agents do the work, including the checking. Save usage.

## State of the work
PLATES: C:\MKM_TMP\plate_heat_20261120\gap4\N20\out = 20 plates, 4 mm gap, all 369 pieces placed, 54-75 percent of bed. 19 plates leaves 20 pieces off, so 20 stands. Plate 1 holds one large piece (breaks his "no plate with just one model" rule; flagged to him, not resolved). Planner used: gpu_scan\mkm_gpupool10_fill.py (agent copy; patched line 513 print bug, backup .backup_20261120-1058_before_split_print_fix). The morning "fill_plater.py" driver is WRONG (do not use it).
EVERY-LAYER CHECK: code at C:\MKM_TMP\gonogo_20261120\gonogo.py; partial results in result_gap4 (P1-P7; P6 and P7 flagged overlap, UNPROVEN - may be the check counting one figure's body and base as two models, or a real overhang at height). Mike STOPPED this agent at about 12:15. Do not restart unless he says.
NEAR-MISS PLATE (membrane analysis): agent was started 11:58 with the full method (film as tensioned membrane, Laplace solve, march the peel, force per raft area per mm of lift); Mike STOPPED it at about 12:15. Work folder C:\MKM_TMP\membrane_nearmiss_20261120 may be partial. The consensus question is posted: "MIKE 1154 - CONSENSUS WANTED - what happened on the near-miss plate..." (+ TASK GROK-BUILD copy). Check for ANSWER files. Claude's prediction is in that file (small raft left holding the film tent at the ring end takes about 4x the stress of a large one).
CONSENSUS DONE: "MERGED RULE CLAUDE - plate go-nogo every layer + shrinking band (ChatGPT + Grok consensus) - 20261120-1135.md" in this hub. Band is binding on the layout (three yes/no events), not a veto.
ROOMS: F:\MKM_STAGE\EP1_ROOM_PROOF_20261120 (Blender cube room + Grok paint-over; paint-over came back small and soft - rejected). F:\MKM_STAGE\TERRAIN_KIT_20261120 (17 terrain STL pieces as a Blender kit, KIT_INDEX.csv of 429 files; corrected preview and 4x master NOT rendered yet; commands in REPORT.txt). F:\MKM_STAGE\UNREAL_LANE_20261120 (scripts ready, nothing rendered; ask the gate for 5 GB not 8). Licence terms of the terrain sets for stream use are unchecked.
WRAITH: closed (server killed 10:28 to free RAM). Locked look intact. Built and on: shadows, light match, fog over minis, encounter zoom. Built but off and untested: depth of field, parallax, depth fog, torch, grade, atmosphere, moment effects. Not built: lot card, shake, room wipe. Mike must move the mini table, then a new clean plate. Needs a 15 minute live test before any show. Report: REPORT CLAUDE-AGENT - WRAITH effects status + safe wiring - 20261120.md.
LAPTOP: heavy jobs go through F:\MKM_STAGE\MKM_GATE\mkm_gate.cmd. Chrome and the ChatGPT app were closed at Mike's request. Grok runs from the scheduled task MKM_GROK_DRIVER and only picks up files named TASK GROK-BUILD...; ChatGPT answers MIKE ... files from the cloud.
MODS: Delegate and Continuity installed today (Continuity's storage was not connected in the old chat). Desktop Statusline not installed.

## How to work cheaply (binding)
- One agent with several related jobs, not many small agents (each agent costs 130k-240k tokens here because every connected tool rides along).
- No AI polling: write a small script on the laptop that waits and writes one result file, then read that file once.
- Short replies, results only. Heavy thinking goes to ChatGPT and Grok through this hub.