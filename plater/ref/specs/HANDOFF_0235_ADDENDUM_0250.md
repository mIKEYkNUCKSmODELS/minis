# ADDENDUM 02:50 CT 2026-11-20 - plater decides plate count and gap

Mike 02:35: "was that a mockup or just poorly filled plates?" (real layouts, poorly filled: 32 beds hand-set, 3 mm gap hand-set).
Mike 02:38: "The plater should decide how many plates are needed and how much space between models there should be to minimize failures"

## Built (gated, exact undo)
- Planner gpu_scan\mkm_gpupool10.py, 3 lines, active only with env MKM_AUTO=1. Backup mkm_gpupool10.py.backup_20261120-0245_before_auto_count_gap; undo C:\MKM_TMP\plate_heat_20261120\auto\UNDO_auto.cmd.
  - Allotment = no plate fuller, at any height, than the fullest plate that has printed clean: 3364 mm2 cured per layer over 1.8-30 mm (FE2_03), 4832 for any other 28 mm stretch (FE2_02). Env MKM_AUTO_LOW / MKM_AUTO_HIGH. These two numbers are Claude's reading of the printed plates, NOT Mike's decision.
  - Plate count = what that allotment and the bed area need (was bed area only).
- Driver C:\MKM_TMP\plate_heat_20261120\auto\auto_plater.py [pieces.json]: lets the planner set the count, adds a bed only while that places more pieces, then searches for the largest gap at which that count still holds every piece, then heat maps + cockpit refresh. About 9 minutes, no AI needed. Log auto.log, result AUTO_RESULT.json.

## Result (pieces.json of 11/17, 369 pieces)
- 21 plates, gap 4.0 mm (5.0 mm drops 10 pieces, 8.0 mm drops 44). 368 placed, 1 carried (TEST5_HAND_RING_1110).
- Layouts: auto\try_N21_g4.0\out\P1..P21. Heat maps: C:\MKM_TMP\plate_heat_20261120\heat_auto\. Cockpit shows them.
- Real footprint 33-71% of bed (most 45-64%); allotment used 73-100%.
- Independent check (finer slice, 0.25 mm): 19 of 21 inside. P10 reads 3485 (over the 3463 line by 22); P7 reads 4836 higher up (4 over 4832). The planner's own piece profiles (0.5 mm grid) read about 1-4% lower than the fine slice, so plates it fills to 100% land slightly over. Not corrected yet.
- Edge margin is still the parameter file's 0.5 mm; not searched.
- Sets spread over 4 plates: 05_Untethered Souls x7 (P2/P3/P5/P8), 10_The Fingers x5 (P6/P14/P15/P19). Mike's rule is 3 plates, 4 only when necessary.
- The gap search is one gap for the whole bundle (largest that fits), not a per-plate gap. The printed-plate data did not show gap width separating clean from failed; cured area per layer did.
- Deleted as superseded scratch: plate_heat_20261120\full32, run22, heat22, and the losing gap tries. The 02:15 run in C:\MKM_TMP\full_plates_20261120 is untouched.
- Nothing sliced, scanned or sent.
