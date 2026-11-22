"""Run independent plate_all workers concurrently, with a bounded shared GPU budget.

Usage: python plate_batch.py <planner_out> <RUN> [workers=3]
No printer I/O. Workers retain their own states, and only the coordinator
merges verified worker results into RUN_STATE.json.
"""
import copy
import glob
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import plate_all as PA
import mkm_v3_config as C

VRAM_TOTAL_FRACTION = 0.90
MAX_GPU_HOLD_SECONDS = 30 * 60      # the DEFAULT run limit (Mike 14:03); unchanged
def run_limit_s():
    """R14 (Mike 21:59): the ONE full 20-plate test may lift the cap for that run only: MKM_MAX_RUN_S (seconds) in the command; unset = 30 minutes, the default.  Recorded in RUN_STATE and SUMMARY."""
    return float(os.environ.get("MKM_MAX_RUN_S", MAX_GPU_HOLD_SECONDS))
WRAP = ("import sys, runpy; sys.path.insert(0, sys.argv[1]); "
        "sys.argv = sys.argv[2:]; runpy.run_path(sys.argv[0], run_name='__main__')")


def kill_tree(proc):
    """R9 (verifier R6 #1): proc.kill() only ends the `cmd /c` gate wrapper; the python worker below it keeps the GPU. Kill the WHOLE tree by pid (taskkill /T /F)."""
    try:
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)], capture_output=True, timeout=30, creationflags=C.NO_WINDOW)
    except Exception:
        pass
    try:
        proc.kill()
    except Exception:
        pass


def _atomic_json(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=1)
    os.replace(tmp, path)


def _verified(st, run, plate):
    """Don't count only a cached 'done' state if its printable output is missing."""
    ps = st.get("plates", {}).get(plate, {})
    expected = os.path.join(run, "print", plate, plate + ".pm7m")
    output = ps.get("pm7m_path") or expected
    return (ps.get("rate") == "done" and ps.get("pm7m") == "done"
            and os.path.isfile(output) and os.path.getsize(output) > 0
            and ps.get("batch_status") != "FAILED")


def _failed(ps, reason):
    ps["batch_status"] = "FAILED"
    ps["batch_error"] = reason
    # Never allow stale 'done' entries to satisfy the summary after a failed worker.
    ps["rate"] = "failed"
    ps["pm7m"] = "failed"
    ps["rating"] = "FAILED"


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    use_gate = "--no-gate" not in args                       # R6 (verifier R5 #4 / F3): every worker goes through the MKM gate unless a unit test says otherwise
    args = [a for a in args if a != "--no-gate"]
    if not 2 <= len(args) <= 3:
        raise SystemExit("usage: plate_batch.py <planner_out> <RUN> [workers=3] [--no-gate]")
    layout_src, run = args[:2]
    run = os.path.abspath(run)
    n = int(args[2]) if len(args) > 2 else 3
    os.makedirs(run, exist_ok=True)
    if C.obs_running():                                      # a show is on: no heavy job starts (rule), and the caller must see it
        PA.log(run, "batch: OBS is running (a show). Nothing started.")
        return 2
    t_run0 = time.monotonic()      # R10 BLOCKER 3: the shared run limit starts before the planner, not after it
    import mkm_plan
    pieces_src = os.path.abspath(layout_src) if str(layout_src).lower().endswith(".json") else None
    if str(layout_src).lower().endswith(".json"):      # ONE ENTRY POINT: a pieces.json is planned here (new planner: cross-section, 2 mm margin, per-band collision, layout score, peel term, plate count from the math), then the plates run concurrently
        wall = time.time() + run_limit_s()
        try:
            PA.log(run, "batch step 0a: footprint + height-band caches (separate CPU job) for %s" % layout_src)
            mkm_plan.precompute(layout_src, os.path.join(run, "planner"), deadline=wall, gate=use_gate)
            PA.log(run, "batch step 0b: planner (plate count from the math)")
            pr_ = mkm_plan.plan(layout_src, os.path.join(run, "planner"), gate=use_gate, deadline=wall)
        except TimeoutError as ex_:
            PA.log(run, "batch: STOPPED CLEANLY by the run limit during planning: %s" % ex_); return 4
        except RuntimeError as ex_:
            _hold = r"F:\MKM_STAGE\MKM_GATE\HOLD_plater_tests.txt"
            if os.path.exists(_hold):      # R12 (verifier R9): the gate refuses plater-lane jobs while the HOLD file exists; say that, not a hard-check failure
                PA.log(run, "batch: NOT RUN, the gate HOLD is on (%s exists, Mike 11/21 17:06): no plan was attempted. Detail: %s" % (_hold, ex_)); return 5
            PA.log(run, "batch: REFUSED, no plan without a hard-check failure: %s" % ex_); return 3
        layout_src = pr_["out"]
        PA.log(run, "batch: planned %d plates (math said %s), tries %s" % (pr_["n"], pr_.get("n_math"), pr_.get("tries")))
    _bad = mkm_plan.hard_fails_in(layout_src)      # R10 BLOCKER 2: a plan with a hard-check failure never reaches a worker
    if _bad:
        PA.log(run, "batch: REFUSED, the plan carries a hard-check failure: " + "; ".join("%s %s" % (p_, "; ".join(r_)[:100]) for p_, r_ in _bad[:4])); return 3
    import mkm_accounting      # R14 (Mike 21:59): model accounting before any worker starts
    _pj = pieces_src if pieces_src else mkm_accounting.pieces_file_for(layout_src)
    import mkm_accounting_gate as _G      # G2 (NOT CHECKED is a refusal) + G3 (edge / gap / fill minimum enforced, not only logged)
    _ok, _acc_line, _acc = _G.accounting_gate(_pj, layout_src)
    _acc = list(_acc) + _G.settings_problems(layout_src, C.EDGE_MARGIN_MM, C.GAP_MM, float(os.environ.get("MKM_FILL_MIN", C.FILL_MIN)))
    if _acc:
        PA.log(run, "batch: REFUSED, %s / settings check (%d problems): %s" % (_acc_line, len(_acc), "; ".join(_acc[:3])[:400])); return 3
    PA.log(run, "batch: " + _acc_line)
    PA.log(run, "batch: " + mkm_accounting.case_mode()["label"])      # R17 (verifier R16 blocker 4): the case mode is in run.log, RUN_STATE settings and SUMMARY
    _hold = r"F:\MKM_STAGE\MKM_GATE\HOLD_plater_tests.txt"
    if use_gate and os.path.exists(_hold):      # R13 (verifier R10 small #2): the resume path (an existing planner folder) says HOLD too, not "Pn FAILED (worker exit 3)"
        PA.log(run, "batch: NOT RUN, the gate HOLD is on (%s exists, Mike 11/21 17:06): no worker was started." % _hold); return 5
    if n < 1:
        raise ValueError("workers must be >= 1")
    os.makedirs(run, exist_ok=True)
    st = PA.load_state(run)
    st.setdefault("plates", {})
    plates = PA.plate_names(os.path.join(run, "layout")) or PA.plate_names(layout_src)
    if not plates:
        PA.log(run, "batch: no plates found")
        return 1

    todo = [p for p in plates if not _verified(st, run, p)]
    count = min(n, len(todo))
    groups = [todo[k::count] for k in range(count)] if count else []
    # Clear stale states BEFORE worker launch, including states from fewer/more workers.
    for stale in glob.glob(os.path.join(run, "RUN_STATE_w*.json")):
        os.remove(stale)
    for stale in glob.glob(os.path.join(run, "RUN_STATE_w*.json.tmp")):
        os.remove(stale)

    # Count actual workers, not requested workers; sum of process caps cannot exceed 90%.
    per_worker_fraction = VRAM_TOTAL_FRACTION / count if count else 0
    env = dict(os.environ, MKM_MAX_BUILDS="1", MKM_PNG_WORKERS="6",
               PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    if count:
        env["MKM_VRAM_FRAC"] = format(per_worker_fraction, ".12g")      # ONE name: plate_all and mkm_gpuport both read MKM_VRAM_FRAC
        # R9 (verifier R6 #2): gpuport's fit check wanted 3500+512 MiB free no matter the share; the need is now this worker's own share (the cap it really runs under)
        env["MKM_VRAM_NEED_MIB"] = str(int(12.0 * 1024 * per_worker_fraction - 512))
    # R9 (verifier R6 #6): the coordinator copies the layout ONCE, before any worker starts (workers used to race on it)
    layout_dst = os.path.join(run, "layout")
    if not PA.plate_names(layout_dst):
        PA.copy_layout(layout_src, layout_dst)
    procs = []
    exit_codes = {}
    start = t_run0
    deadline = start + run_limit_s()
    try:
        for k, group in enumerate(groups):
            tag = "w%d" % k
            # Required: PA.load_state reads the worker-specific file after --worker.
            seeded = copy.deepcopy(st)
            for plate in group:
                seeded["plates"].setdefault(plate, {}).pop("batch_status", None)
            _atomic_json(os.path.join(run, "RUN_STATE_%s.json" % tag), seeded)
            cmd = [sys.executable, "-I", "-u", "-c", WRAP, HERE,
                   os.path.join(HERE, "plate_all.py"), layout_src, run,
                   "--only", ",".join(group), "--worker", tag]
            if use_gate:                                      # vram ask = this worker's share of the card (12 GB x 0.90 / workers), ram 3 GB (rule: 3 GB or less)
                cmd = C.gate_cmd("PLATEALL_%s" % tag, 3, round(12.0 * per_worker_fraction, 1), cmd)
            log_file = open(os.path.join(run, "worker_%s.log" % tag), "w", encoding="utf-8")
            try:
                proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT,
                                        env=env, creationflags=C.NO_WINDOW)
            except Exception:
                log_file.close()
                exit_codes[tag] = "launch failed"
                continue
            procs.append((tag, group, proc, log_file))
            PA.log(run, "batch: %s started on %s (VRAM fraction %.4f)" %
                   (tag, ", ".join(group), per_worker_fraction))

        for tag, group, proc, log_file in procs:
            try:
                # Shared absolute deadline, not 30 min times number of workers.
                remaining = max(0.01, deadline - time.monotonic())
                exit_codes[tag] = proc.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                kill_tree(proc)
                try:
                    proc.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    pass
                exit_codes[tag] = "TIMEOUT (%ds shared GPU ceiling)" % run_limit_s()
            finally:
                log_file.close()

        merged = PA.load_state(run)
        merged.setdefault("plates", {})
        for k, group in enumerate(groups):
            tag = "w%d" % k
            sf = os.path.join(run, "RUN_STATE_%s.json" % tag)
            code = exit_codes.get(tag, "missing worker")
            worker = None
            if os.path.isfile(sf):
                try:
                    with open(sf, encoding="utf-8") as handle:
                        worker = json.load(handle)
                except (OSError, ValueError) as exc:
                    code = "invalid state: %s" % exc
            for plate in group:
                ps = (worker or {}).get("plates", {}).get(plate, {})
                if code == 0:
                    merged["plates"][plate] = ps
                    if _verified(merged, run, plate):
                        merged["plates"][plate].pop("batch_error", None)
                        merged["plates"][plate]["batch_status"] = "DONE"
                    else:
                        _failed(merged["plates"][plate], "worker %s exited 0 but plate incomplete" % tag)
                else:
                    # Even when a crashed worker wrote 'done', NEVER count it as done.
                    merged["plates"][plate] = ps
                    _failed(merged["plates"][plate], "worker %s exit %s" % (tag, code))
                if merged["plates"][plate].get("batch_status") == "FAILED":
                    PA.log(run, "batch: %s FAILED (%s)" % (plate, merged["plates"][plate]["batch_error"]))
        _eff = mkm_accounting.effective_settings(layout_src); _ovr = "MKM_MAX_RUN_S" in os.environ and run_limit_s() != MAX_GPU_HOLD_SECONDS
        merged["settings"] = dict(planner_out=os.path.abspath(layout_src), edge_margin_mm_configured=C.EDGE_MARGIN_MM, gap_mm=C.GAP_MM, fill_min_configured=float(os.environ.get("MKM_FILL_MIN", C.FILL_MIN)), plates_effective=_eff["plates"], carried=_eff["carried"], accounting=_acc_line, case=mkm_accounting.case_mode(),      # R17: CASE=OFF/ON is part of the effective settings
                                  run_limit_s=run_limit_s(), run_limit_default_s=MAX_GPU_HOLD_SECONDS, run_limit_override=_ovr)
        old_name = PA.STATE_NAME
        try:
            PA.STATE_NAME = "RUN_STATE.json"
            PA.save_state(run, merged)
        finally:
            PA.STATE_NAME = old_name
        import mkm_rate
        mkm_rate.summarize(run)
        ok = sum(bool(_verified(merged, run, plate)) for plate in plates)
        _lc = lone_carried_names(layout_src)      # R13 (verifier R10 small #1): a figure left out by the lone-plate rule is named here and in SUMMARY.md
        PA.log(run, "batch: %d of %d plates fully complete (rated AND print-file exists). Summary: %s%s" %
               (ok, len(plates), os.path.join(run, "SUMMARY.md"),
                ("  | LEFT OUT (lone model carried, no plate could take it): " + "; ".join(_lc)) if _lc else ""))
        if _lc:
            with open(os.path.join(run, "SUMMARY.md"), "a", encoding="utf-8") as sm_:
                sm_.write("\n\n## Left out of every plate (lone-plate rule)\nNo plate may hold one model, and no other plate could take these, so they are NOT in any print file: " + "; ".join(_lc) + "\n")
        PA.log(run, "STOPPED - waiting for Mike. Nothing was sent to any printer.")
        return 0 if ok == len(plates) else 1
    finally:
        # A state file is only kept by the coordinator after merge. Crash leftovers
        # are removed on the NEXT invocation before any new worker starts.
        for tag, group, proc, handle in procs:
            if not handle.closed:
                handle.close()
        if 'merged' in locals():
            for stale in glob.glob(os.path.join(run, "RUN_STATE_w*.json")):
                os.remove(stale)


def lone_carried_names(layout_src):
    """the lone models the planner carried (LONE_CARRY.json in its out folder), as 'set (file)' names; [] when none."""
    fp = os.path.join(layout_src, "LONE_CARRY.json")
    try:
        return sorted({"%s (%s)" % (x["set"], x["file"]) for x in json.load(open(fp, encoding="utf-8-sig"))})
    except (OSError, ValueError, KeyError, TypeError):
        return []


if __name__ == "__main__":
    sys.exit(main())
