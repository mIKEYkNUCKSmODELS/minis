# MKM PLATER - code package and how it is supposed to work (2026-11-20 05:35 CT)

Code: `PLATER_CODE_PACKAGE_20261120-0535.zip` in this same folder. This note explains the intent, what the code does today, and what is wrong with it. Written by Claude at Mike's request ("give the code to the design extractor and explain how it is supposed to work").

## 1. What the plater is for (Mike's words)
- "This plater is supposed to design plates that will work. Thats it thats all." Not a tool that says "no this wont work" but one that says "yep this will work".
- "The laws of physics is what run the plater, not guesswork." "No more arbitrary rules. Laws of physics not arbitrary rules."
- "the single variable is going to be the layout. Everything else is constant." Level plate, good film, his normal settings (50 s bottom, 2.45 s normal, 0.03 mm layers, M7 Max, ACF film) are taken as perfect.
- "The plater should decide how many plates are needed and how much space between models there should be to minimize failures" (11/20 02:38).
- "I think that filled plate creates suctions cups because it is too full." (11/19 23:53)
- No big empty gap in the middle of a plate (11/20 05:27: "why do I still have a big ass gap in the middle. you know I don't want that"). No sparse plates. Never a plate with one model.
- Every model's first layer sits flat on the build plate. Hollow versions of parts, not solid.
- A model may be split over up to 3 plates, 4 only when necessary.
- His allotment method: a plate has a budget; add complete model sets until it is used up; do not test-fit piece by piece.
- Peel picture: one ring of peel closing in from the bed edge to the centre. A small model beside a much larger one is at risk unless it is shielded by neighbours when its own peel starts; the larger neighbours should be hit first. He said this is "not absolute law".
- GPU for everything that can use it, in batches, fast. Output plates like an assembly line; one bad model never stops the batch.
- Nothing goes to a printer without his go. He inspects every plate.

## 2. What the code does today
Machine: bed 298 x 164 mm, grid 0.5 mm. Python: `C:\Users\confi\AI\ComfyUI_windows_portable\python_embeded\python.exe -I`. Live tree: `C:\MKM_TMP\plater_one_build_20261117\tree`.

1. **Piece measurement** (`gpu_scan\mkm_gpupool10.py` compute_piece / get_piece, `gpu_scan\mkm_gpuport.py`): each STL is loaded, its footprint mask made for the 4 quarter-turn rotations, and its cured area measured every 0.5 mm of height (GPU slicer `GSlicer` + `gfill`). Cached.
2. **Allotment** (env `MKM_AUTO=1`, added 11/20): each piece "spends" its 28 mm running mean of cured area at every height. A plate may not exceed the fullest plate that has printed clean: 3364 mm2 per layer over 1.8-30 mm, 4832 for any other 28 mm stretch (both lowered 4% to cover the gap between the 0.5 mm piece measurement and a 0.25 mm slice). These two numbers come from 19 printed plates. They are observed limits, not physics, and they are Claude's reading, not Mike's decision.
3. **Plate count** (same file, `_need`): beds = what the cured-area allotment and the bed area need. `auto\auto_plater.py` adds a bed only while that places more pieces.
4. **Placement** (same file, `find` and the rounds loop, about lines 340-500): pieces go on in three rounds, large then medium then small, each piece once, greedy. A piece goes to the legal spot with the lowest "film cost": a membrane model of the film (tension stand-in 7041 N/m) where a load in the middle of the film costs more than a load near the clamped edge.
5. **Gap** (`auto\auto_plater.py`): one gap for the whole bundle, the largest at which every piece still fits the chosen plate count. Bed-edge clearance is set equal to the gap. Last result: 22 plates, 3.5 mm, 368 of 369 pieces.
6. **Check and pictures** (`plate_heat.py`): re-slices every finished plate at 0.25 mm from its `recipe.json`, measures cured area per layer up the whole height, and draws the heat map (layers cured per spot, green to red) with the peel ring. `F:\MKM_STAGE\PLATER_COCKPIT\build_cockpit.py` shows them.
7. **Per-plate spread** (`auto\spread.py`, started 05:30, unproven): re-places each plate's own pieces alone with a wider and wider gap, keeping the widest that fits, to turn the empty centre into even spacing.

## 3. What is wrong
- **Empty centre.** Step 4's film cost makes edges cheap and the centre expensive, so heavy pieces go to the rim. Plates are limited by cured area before they run out of room (real footprint 32-67% of the bed), so the spare room is left as one hole in the middle. Step 7 is a patch around this, not a fix of the objective.
- **Placement is one greedy pass.** No piece is moved after it lands. No swapping between plates to even them out.
- **One gap for the whole bundle** in step 5; plates with room to spare do not use it.
- **The limits are fitted, not derived.** Cured area per layer called 18 of 19 printed plates correctly. Gap width, sealed pockets, flow resistance and the peel-ring / shielding measures did not separate clean prints from failed ones on those 19 plates, so none of them is in the pass/fail.
- **Nothing here is sliced or scanned.** Suction cups, resin traps and islands need a sliced file (`gpu_scan\mkm_uvtools_gpu.py` reads M7 Max .pm7m files on the GPU).
- **Thin plates.** A plate whose allowance is used by one heavy piece ends up with few pieces (latest P4: 7 pieces, 32% footprint).
- Sets of several figures get spread over 4 plates in places.

## 4. What a better placement step has to do
Keep steps 1-3 and 6. Replace the step 4 objective so that, with the plate's pieces already chosen:
- the whole bed is used, centre included, with the spare room shared out as spacing between models;
- no model is closer to a neighbour or to the bed edge than it has to be, and the closest pair is as far apart as the bed allows;
- (Mike's peel idea, to be tested, not assumed) small models sit where larger neighbours meet the closing ring first;
- every model stays flat on the plate, quarter-turn rotations only;
- it runs on the GPU for a whole bundle at once and finishes in about a minute.

## 5. Files in the zip
`gpu_scan\mkm_gpupool10.py` (planner), `mkm_gpuport.py` (GPU slicer and masks), `mkm_slice_reader.py`, `mkm_uvtools_gpu.py`, `mkm_ring_layout_hook*.py` (11/19 ring layout experiments), `physics\mkm_planner_params.py` and the parameter json, `physics\mkm_planner_geom.py`, `plate_packer\mkm_suction.py` (predictor), `plate_heat_20261120\plate_heat.py`, `auto\auto_plater.py`, `auto\run_all.py`, `auto\spread.py`, the three patch scripts with their UNDO files, the latest run log and `AUTO_RESULT.json`, and the 4 plate sheets + 2 heat sheets.
