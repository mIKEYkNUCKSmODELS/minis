"""G2/G3/G4 helpers (verifier blocker 13): the accounting decision, enforced settings, and the plan-reuse signature.  Pure Python, no slicer / GPU imports.
Kept out of mkm_accounting.py / plate_all.py main() so the call sites stay one small hook each."""
import hashlib, os

import mkm_accounting as MA


def accounting_gate(pieces_json, planner_out):
    """G2: one decision for both entry points -> (ok, line, problems).  'NOT CHECKED' is a refusal; MKM_ALLOW_UNACCOUNTED=1 skips only that case for one run
    and the line says so.  It never skips a real accounting failure when a pieces.json exists."""
    if not pieces_json:
        if os.environ.get("MKM_ALLOW_UNACCOUNTED") == "1":
            return True, "model accounting: NOT CHECKED (no pieces.json recorded) - ALLOWED by MKM_ALLOW_UNACCOUNTED=1 for this run only - UNPROVEN", []
        return False, "model accounting: NOT CHECKED (no pieces.json recorded for this planner output): REFUSED; re-plan from pieces.json or set MKM_ALLOW_UNACCOUNTED=1 for this run", ["no pieces.json recorded for this planner output"]
    probs = MA.model_accounting(pieces_json, planner_out)
    if probs:
        return False, "model accounting FAILED (%d problems)" % len(probs), probs
    return True, "model accounting: every model in %s is placed or carried, none both, no failed plate" % os.path.basename(pieces_json), []


def settings_problems(planner_out, edge_mm, gap_mm, fill_min):
    """G3: the settings the planner REALLY used (its hard_checks.json) must not be weaker than the configured ones, and every plate must meet its own fill minimum.
    fill_min is a fraction (0.68); 0 skips only the fill part.  -> list of problems ([] = ok)."""
    out = []
    eff = MA.effective_settings(planner_out)["plates"]
    if not eff:
        return ["no P*/hard_checks.json under the planner output: the settings it used are unknown"]
    for p, v in sorted(eff.items()):
        if v.get("edge_need_mm") is None or float(v["edge_need_mm"]) + 1e-9 < float(edge_mm):
            out.append("%s: planner edge margin %s mm < configured %.1f mm" % (p, v.get("edge_need_mm"), float(edge_mm)))
        r = MA._jload(os.path.join(os.path.abspath(planner_out), p, "hard_checks.json"), {}) or {}
        if r.get("gap_need_mm") is None or abs(float(r["gap_need_mm"]) - float(gap_mm)) > 1e-6:
            out.append("%s: planner gap %s mm != configured %.1f mm" % (p, r.get("gap_need_mm"), float(gap_mm)))
        if float(fill_min) > 0:
            if v.get("fill_min_pct") is None or float(v["fill_min_pct"]) + 1e-6 < 100.0 * float(fill_min):
                out.append("%s: planner fill minimum %s%% < configured %.1f%% (stale or pre-R14 plan?)" % (p, v.get("fill_min_pct"), 100.0 * float(fill_min)))
            elif v.get("fp_pct") is None or float(v["fp_pct"]) + 1e-6 < float(v["fill_min_pct"]):
                out.append("%s: real footprint %s%% is under its own minimum %s%% (the fill veto did not hold)" % (p, v.get("fp_pct"), v["fill_min_pct"]))
    return out


def plan_signature(pieces, fill_min, edge_mm, gap_mm, layer_mm):
    """G4: what a saved PLAN_RESULT must match to be reused: the pieces file (path + sha256 of its bytes) and the plan settings."""
    with open(pieces, "rb") as fh:
        sha = hashlib.sha256(fh.read()).hexdigest()
    return dict(pieces=os.path.normcase(os.path.abspath(pieces)), pieces_sha256=sha, fill_min=float(fill_min), edge_mm=float(edge_mm), gap_mm=float(gap_mm), layer_mm=float(layer_mm))
