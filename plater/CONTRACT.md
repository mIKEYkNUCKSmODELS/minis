# MKM PLATER — finish wiring (cloud build, 2026-11-20)

Code is written here in the cloud, then copied onto MKM (Windows laptop) into
`C:\MKM_TMP\gonogo_20261120\plater_v3\`. Nothing here can be run against the real
plates from the cloud (no GPU, no UVtools, no STLs). **Coding pass only: no tests run yet.**
Every module must at least pass `python3 -m py_compile`.

Reference code (read-only, do not edit): `plater/ref/` — laptop package of 11/20 05:35
(`gpu_scan/`, `physics/`, `plate_heat_20261120/`, `plate_packer/`) plus `ref/specs/` (handoffs,
merged rule, UVtools route report) and `ref/drive_src/`. Package README: `ref/specs/PACKAGE_README.md`.

## Mike's rules (binding, from the 11/20 15:27 handoff)
- One run: PLATE -> SLICE ONCE at printer pixels on the GPU -> slices saved -> UVtools print file
  (M7 Max .pm7m) -> UVtools detect + repair + heat map -> rating -> STOP and wait for Mike.
- NEVER print or send to a printer. Making the file is allowed. No code path may copy to a printer,
  USB/printer share, or network printer.
- Gap 4 mm, locked (min model-to-model and model-to-bed-edge). Fewest plates holding every piece.
- Layout is the only variable. Film/resin/printer values are constants. No fitted or borrowed cut-off numbers.
- Grid = printer pixels 6480 x 3600 over 298.08 x 165.6 mm (pitch 0.046 mm), mirror x. Layer 0.03 mm, EVERY layer 0..top.
- GPU in batches of many layers; no per-layer CPU work in layer loops (no cv2 / .cpu() / .numpy() inside a per-layer loop;
  move data to CPU once per batch).
- Slice each plate once; keep the slices run-length in the store; never re-slice or re-scan a whole plate.
- Use UVtools 7.1.2 `"C:\Program Files\UVtools\UVtoolsCmd.exe"` for detection parity, repair and the print file.
- Trap rule: skip deliberate hollows that drain; fill small accidental voids; vent real sealed cavities; unclear = vent and mark.
  Judge by shape, not a size cut-off. (Custom vent/fill mesh surgery is DROPPED in favour of UVtools repair.)
- Originals are read-only. Never write next to or into Mike's model library; work on copies.
- Rating per plate: LIKELY / LIKELY WITH NOTES / RISKY / WILL NOT PRINT, with reasons. One bad plate never stops the batch.
- No big empty middle; no sparse plates; never a one-model plate.
- Machine: 16 GB RAM, RTX 5070 Ti 12 GB. Heavy jobs go through
  `F:\MKM_STAGE\MKM_GATE\mkm_gate.cmd run --name <LABEL> --ram-gb N --vram-gb M --class heavy -- <command>`.
  Never start a heavy job while OBS is running (check process `obs64.exe`; refuse and say so).
  Launch subprocesses hidden (`creationflags=0x08000000` on Windows). Python: `C:\Users\confi\AI\ComfyUI_windows_portable\python_embeded\python.exe -I -u`.
- Never delete anything of Mike's. Write only into the run's own output folder.

## Shared layout of one run (`RUN` = output folder given on the command line)
```
RUN/
  layout/P{n}/recipe.json            # planner schema=1 (see ref/gpu_scan/mkm_gpupool10.py ~l.560) — input or step (d) output
  layout/P{n}/RING_MANIFEST.json     # copied from planner
  store/P{n}/meta.json               # slice store, written by mkm_slice_store
  store/P{n}/layers.bin + index.npy  # run-length layers
  print/P{n}/P{n}.sl1  P{n}.pm7m     # step (a)
  uv/P{n}/issues_raw.json issues_repaired.json P{n}_repaired.pm7m uv_log.txt   # step (b)
  uv/P{n}/gpu_issues.json            # mkm_uvtools_gpu on the repaired file (only for the parity plate, or all if --gpu-all)
  uv/PARITY.json                     # UVtools vs GPU counts on one plate
  rate/P{n}/rating.json  P{n}_heat.png                                         # step (c)
  SUMMARY.md  SUMMARY.png  RUN_STATE.json                                      # step (c)/(e)
```

## Module interfaces (each owner writes exactly these files in `plater/v3/`)

### `mkm_slice_store.py` + `mkm_plate_slice.py` — owner: builder A (step a, part 1)
- `mkm_slice_store.SliceStore(dir, mode="r"|"w")`
  - `.write_meta(res=(6480,3600), pixel_mm=(0.046,0.046), layer_mm=0.03, n_layers, mirror_x=True, origin_mm, recipe_sha256)`
  - `.append_batch(masks)` — masks: torch bool or numpy bool, shape (B, 3600, 6480) row=y col=x, ALREADY mirrored.
    Encodes run-length per row-major layer (start value 0, alternating run lengths, uint32), one CPU transfer per batch.
  - `.n_layers`, `.meta`, `.read_layer(i) -> np.bool_ (3600,6480)`, `.read_batch(i0,i1, device=None) -> torch bool (B,H,W)`,
    `.cured_px(i) -> int` (from runs, no image), `.close()`.
- `mkm_plate_slice.slice_plate(recipe_path, store_dir, batch=64, device="cuda")` — loads each source STL
  (path from recipe `sources[]`, verify sha256, apply the 4x4 `matrix` once), slices at layer midpoints
  z=(i+0.5)*0.03 for i in 0..ceil(top/0.03)-1, rasterizes at printer pixels with pixel-centre sampling
  over the 298.08 x 165.6 mm screen centred on the bed origin, mirrors x, writes via SliceStore.
  Reuse `ref/gpu_scan/mkm_gpuport.py` GSlicer/gfill (import it from the laptop path; do not copy it).
  Batch: compute segments per z on GPU, fill B layers, one transfer per batch. Skips plates whose store meta
  recipe_sha256 matches (slice once). CLI: `python mkm_plate_slice.py <recipe.json> <store_dir>`.

### `mkm_pm7m_build.py` — owner: builder A (step a, part 2)
- `build_print_file(store_dir, out_dir, name, ref_pm7m=None, uvtools=UVTOOLS) -> path.pm7m`
  - Writes `name.sl1` (zip): `job00000.png ...` 8-bit grayscale, numbered from 0, flat; `config.ini`
    (jobDir=job, layerHeight=0.03, expTime=2.45, expTimeFirst=50, numFast=<n_layers>, numSlow=0, printerModel=SL1);
    `prusaslicer.ini` (display_pixels_x=6480, display_pixels_y=3600, display_width=298.08, display_height=165.6,
    layer_height=0.03, printer_notes containing FILEFORMAT_PM7M). Mirror already in the store: do NOT mirror again
    (add display_mirror_x=0 and say so in a comment). PNGs encoded in a worker pool from the store (CPU is fine here:
    this is file writing, not a scan).
  - `UVtoolsCmd convert name.sl1 pm7m name.pm7m`; if ref_pm7m given, `UVtoolsCmd copy-parameters <ref_pm7m> name.pm7m`
    (bottom layer count, lifts) — exposure stays Mike's 50 s / 2.45 s.
  - `print-properties name.pm7m` captured; assert ResolutionX 6480, ResolutionY 3600, LayerHeight 0.03, LayerCount == store.n_layers,
    else raise with the printed values.
  - Never writes outside out_dir.

### `mkm_uv_check.py` — owner: builder B (step b)
- `uv_check(pm7m, out_dir, uvtools=UVTOOLS) -> dict` — runs `print-issues <pm7m> -i -o -r -s` (raw), saves stdout,
  parses counts per type (Island, ResinTrap, SuctionCup, Overhang, TouchingBounds, EmptyLayer, others) and per-layer
  detail if printed; runs `run <pm7m> OperationRepairLayers -o <repaired>` (defaults), plus a second profile
  `-p RepairIslands -p RepairResinTraps` only if the CLI accepts it (detect via `--help`), re-runs print-issues on the repaired file.
  Writes issues_raw.json, issues_repaired.json, uv_log.txt. Raw counts are NOT a limit (merged rule) — just recorded.
- `parity(pm7m, out_dir)` — runs `ref/gpu_scan/mkm_uvtools_gpu.plate_issues_gpu` on the same file, writes gpu_issues.json and a
  comparison (layers affected, first/last layer, per-type) to PARITY.json. Stated plainly as unproven until Mike's run.
- Each UVtools call goes through the gate wrapper when `--gate` is set; timeouts logged, failure of one plate recorded and returned, never raised past the plate.

### `mkm_rate.py` — owner: builder C (step c)
- `rate_plate(plate_dir_layout, store_dir, uv_dir, out_dir) -> dict` — reads recipe, RING_MANIFEST, store, uv results.
  Hard gate (merged rule `ref/specs/MERGED_RULE_go-nogo_band_20261120-1135.md`, quote it in comments): cured cells outside bed,
  two models in one cured cell, component never connected to plate, sealed cavity/cup with no open path, sealed pocket between parts.
  Each UVtools/GPU hit is verified against connected geometry from the store (GPU batches), not taken raw.
  Band events per model per layer (START one-sided nested, LANDING open edge, END shared last loop), shared peel front from bed edge
  inward one grid step at a time, identical repeated layers counted once, layer interval reported.
  Rating mapping (PROPOSAL until Mike says yes — write that in rating.json `rating_basis`):
  WILL NOT PRINT = a verified hard-gate defect survives repair; RISKY = a hard-gate hit that could not be verified either way
  (unclear) or the plate has a shared last loop; LIKELY WITH NOTES = hard gate clean, leftover band events written on it;
  LIKELY = hard gate clean, no events. Reasons list: model, layer interval, event/defect.
- `heat_png(store_dir, out_png, ...)` — heat map at printer pixels from the store (layers cured per spot, green->red, downsampled for
  display), peel ring overlay, model labels, rating banner. Reuse the look of `ref/plate_heat_20261120/plate_heat.py`.
- `summarize(run_dir)` — SUMMARY.md (table: plate, pieces, height, rating, top reasons, UVtools before/after counts) and SUMMARY.png contact sheet.

### `mkm_layout_fill.py` — owner: builder D (step d)
- `refill(layout_in, layout_out, gap_mm=4.0)` — per plate, keep the plate's chosen pieces (no moving pieces between plates
  unless a band event or hard defect needs a swap), re-place them with quarter-turn rotations only, flat, so the whole bed is used
  (centre included) and the spare room is shared as spacing; maximise the smallest model-model / model-edge distance with 4 mm as the floor.
  Then apply band events using piece height profiles: move/rotate/swap to clear a worse event without creating a hard defect or a
  same-class event (comparison order from the merged rule). Writes new recipe.json + RING_MANIFEST.json per plate in planner schema=1,
  plus LAYOUT_NOTES.json (moves made, leftover events). Footprints come from the planner's piece cache
  (`ref/gpu_scan/mkm_gpupool10.py` compute_piece/get_piece) — import, don't copy. GPU, whole bundle at once.

### `plate_all.py` — owner: orchestrator (step e)
One command: `plate_all.py <planner_out_dir> <RUN> [--refill] [--ref-pm7m X] [--gate] [--only P3,P7]`.
Steps per plate, resumable via RUN_STATE.json, one bad plate never stops the batch, then summary, then
prints "STOPPED - waiting for Mike. Nothing was sent to any printer."

Paths/constants live in `plater/v3/mkm_v3_config.py` (owner: orchestrator), importable by all:
`UVTOOLS, PY, GATE, REF_TREE (gpu_scan path), RES=(6480,3600), SCREEN_MM=(298.08,165.6), LAYER_MM=0.03, GAP_MM=4.0,
BOTTOM_EXP_S=50.0, NORMAL_EXP_S=2.45, NO_WINDOW=0x08000000, obs_running(), gate_cmd(label, ram, vram, argv)`.
