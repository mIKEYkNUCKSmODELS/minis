# R21 G1-G4 carry-over

Base r18_orig, ours r18_g1g4, theirs r21_orig (R21 Drive folder, subfolder v3). Merge via `git merge-file` (ours, base, theirs).

Files (all 6 present in R21, copied byte-exact; sizes match Drive): mkm_accounting, mkm_plan, plate_all, plate_batch, mkm_v3_config, mkm_archive_io.
R21 had no mkm_accounting_gate / accounting_gate / settings_problems / plan_signature (grep and title search), so no prior partial work.
Merge result: 0 conflicts in all 6 files. mkm_accounting_gate.py copied unchanged from r18_g1g4.
R21_G1G4.patch touches: mkm_accounting, mkm_accounting_gate (new), mkm_plan, plate_all, plate_batch. mkm_v3_config and mkm_archive_io merge to identical-to-R21 content.

Gates:
- G1-G4 on r21_g1g4: all four exit 0 (PASS).
- G1-G4 OLD on r21_orig: all four exit 1, on behaviour (G1 vanished copy not flagged, G2/G3 plan went on toward slicing, G4 stale plan reused). No ImportError of the code under test. No `_fx.py` change was needed.
- G5: git apply --check on a fresh copy of r21_orig is clean; applying it reproduces r21_g1g4 exactly (diff -r identical).

UNPROVEN:
- Clean merge is textual only; no semantic conflicts were detected beyond what G1-G4 cover. R21's own regression gates (reg_gate_*, gate_r10_plan) were not run.
- G2/G3 OLD failure text includes ModuleNotFoundError('mkm_plate_slice') (module is not in the tree). Same on r18_orig, so it is the fixture's "went on toward slicing" signal, not an R21 change.
- Nothing was run against the live code on MKM.
