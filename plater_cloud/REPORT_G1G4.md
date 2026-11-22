# REPORT CLAUDE-CLOUD - plater blocker 13 (Grok R18 G1-G4) - 20261122

Built against the CODE SNAPSHOT R18 20261122 (Drive). Patch for the driver's R21 3-way merge: `G1G4.patch` (6 files under v3/, plus a new `v3/mkm_accounting_gate.py`). R19/R20 untouched.

| Item | Change | Gate NEW / OLD(R18) |
|---|---|---|
| G1 copies counted with Counter | mkm_accounting.py:15,49-71 | exit 0 / exit 1 "one of two copies of a.stl vanished and was not flagged" |
| G2 NOT CHECKED = refusal (MKM_ALLOW_UNACCOUNTED=1 skips only the no-pieces.json case, logged UNPROVEN) | accounting_gate; hooks plate_all.py:368-373, plate_batch.py:105-109 | exit 0 / exit 1 "went on toward slicing" |
| G3 edge, gap, fill minimum enforced before slicing | settings_problems; same hooks | exit 0 / exit 1 (all 4 violating plans went on) |
| G4 plan reuse keyed on plan_signature (pieces path+sha256, fill min, edge, gap, layer) | mkm_plan.py:145-163 | exit 0 / exit 1 (0.5 plan reused at 0.68; reused after pieces.json changed) |
| Patch applies cleanly to R18 | git apply --check | clean, applied tree identical |

Gates: `python gates/gate_gN_*.py <tree>`; re-run independently by the cloud driver.
Checked: R18 mkm_gpupool10_fill.py writes gap_need_mm, edge_need_mm and footprint.fill_min_pct / fp_pct into hard_checks.json, so G3 reads real fields.

UNPROVEN / OPEN
- No real run here (no GPU, no STLs); planner output is synthetic, mkm_plan._one faked in the G4 gate.
- QUESTION FOR MIKE: G3 refuses any plate whose real footprint is under its fill minimum. The planner marks some plates `at_geometry_limit` or `remainder_plate` (the last leftover plate). If such a plate is allowed under the minimum, G3 will refuse the real run. The patch does not exempt them; whether it should is Mike's call. Check on the N20 delta sweep before the one test.
- Case-on path not exercised (gates use MKM_CASE_MODE=off). plate_batch.main hook not run by a gate (same edit as plate_all.main).
- Grok's G5/G6 (fill label, mkm_case.py) not applied, as they are outside blocker 13.
- Old gate fixtures without gap_need_mm/edge_need_mm in hard_checks.json will now be refused; add the fields to those fixtures (don't weaken G3).
