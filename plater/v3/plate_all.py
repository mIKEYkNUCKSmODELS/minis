"""MKM plater v3 - one command for the whole run.

    python plate_all.py <planner_out_dir> <RUN> [--refill] [--ref-pm7m X] [--gate] [--only P3,P7] [--parity P1] [--gpu-all] [--overlap]

Per plate: (d) refill layout (optional) -> (a) slice once into the store -> (a) build .pm7m via UVtools
-> (b) UVtools detect + repair -> (c) rating + heat map. Then the summary. Resumable: RUN/RUN_STATE.json
records each finished step per plate, so a re-run skips what is done. One bad plate never stops the batch.

It NEVER prints and never copies anything to a printer. Printing needs Mike's explicit go, outside this tool.
"""
import argparse, glob, json, os, shutil, sys, threading, time, traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mkm_v3_config as C

STEPS = ("slice", "pm7m", "uv", "rate")


_LOG_LOCK = threading.Lock()


def log(run, msg):
    line = time.strftime("%Y-%m-%d %H:%M:%S ") + msg
    with _LOG_LOCK:
        print(line, flush=True)
        with open(os.path.join(run, "plate_all.log"), "a", encoding="utf-8") as f:
            f.write(line + "\n")


def load_state(run):
    p = os.path.join(run, "RUN_STATE.json")
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else {"plates": {}}


def save_state(run, st):
    p = os.path.join(run, "RUN_STATE.json")
    tmp = p + ".tmp"
    json.dump(st, open(tmp, "w", encoding="utf-8"), indent=1)
    os.replace(tmp, p)


def plate_names(layout):
    names = [os.path.basename(os.path.dirname(p)) for p in glob.glob(os.path.join(layout, "P*", "recipe.json"))]
    return sorted(names, key=lambda s: int("".join(ch for ch in s if ch.isdigit()) or 0))


def copy_layout(src, dst):
    """Copy only recipe.json + RING_MANIFEST.json per plate (planner output stays untouched)."""
    for p in plate_names(src):
        os.makedirs(os.path.join(dst, p), exist_ok=True)
        for fn in ("recipe.json", "RING_MANIFEST.json"):
            s = os.path.join(src, p, fn)
            if os.path.exists(s):
                shutil.copy2(s, os.path.join(dst, p, fn))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("planner_out")
    ap.add_argument("run")
    ap.add_argument("--refill", action="store_true", help="step (d): re-place pieces to use the whole bed")
    ap.add_argument("--ref-pm7m", default=None, help="a known-good M7 Max file to copy lift/bottom-layer parameters from")
    ap.add_argument("--gate", action="store_true", help="send heavy steps through the MKM gate")
    ap.add_argument("--only", default=None, help="comma list of plates, e.g. P3,P7")
    ap.add_argument("--parity", default=None, help="plate to compare UVtools vs GPU detector counts on (default: first)")
    ap.add_argument("--gpu-all", action="store_true", help="run the GPU detectors on every plate, not only the parity plate")
    ap.add_argument("--overlap", action="store_true", help="keep the GPU working (slice next plate, rate finished ones) while UVtools runs")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args(argv)

    run = os.path.abspath(a.run)
    os.makedirs(run, exist_ok=True)
    if C.obs_running():
        log(run, "OBS is running (a show). Heavy jobs are not started. Re-run when the show is over.")
        return 2

    import mkm_plate_slice, mkm_pm7m_build, mkm_uv_check, mkm_rate

    layout = os.path.join(run, "layout")
    st = load_state(run)
    if not plate_names(layout):
        if a.refill:
            import mkm_layout_fill
            log(run, "step d: refill layout -> %s" % layout)
            mkm_layout_fill.refill(a.planner_out, layout, gap_mm=C.GAP_MM)
        else:
            copy_layout(a.planner_out, layout)
    plates = plate_names(layout)
    if a.only:
        want = set(x.strip() for x in a.only.split(","))
        plates = [p for p in plates if p in want]
    if not plates:
        log(run, "no plates found in %s" % layout)
        return 1
    parity_plate = a.parity or plates[0]
    log(run, "%d plates: %s" % (len(plates), ", ".join(plates)))

    lock = threading.Lock()                 # RUN_STATE.json is written from the GPU thread and (with --overlap) one CPU worker

    def mark(p, **kv):
        with lock:
            st["plates"].setdefault(p, {}).update(kv)
            save_state(run, st)

    def paths(p):
        return (os.path.join(run, "store", p), os.path.join(run, "print", p), os.path.join(run, "uv", p), os.path.join(run, "rate", p))

    def guarded(p, what, fn):
        """Run one stage of one plate. A failure is written on that plate and the batch continues."""
        if st["plates"].get(p, {}).get("error"):
            return False
        try:
            fn(p)
            return True
        except Exception as e:
            mark(p, error="%s: %s: %s" % (what, type(e).__name__, e), traceback=traceback.format_exc())
            log(run, "%s: FAILED at %s (%s: %s) - batch continues" % (p, what, type(e).__name__, e))
            return False

    def gpu_slice(p):                       # GPU
        if st["plates"][p].get("slice") != "done":
            log(run, "%s: slice once at printer pixels" % p)
            mkm_plate_slice.slice_plate(os.path.join(layout, p, "recipe.json"), paths(p)[0], device=a.device)
            mark(p, slice="done")

    def cpu_stage(p):                       # CPU only: PNG/zip writing and UVtoolsCmd
        store, prn, uv, _ = paths(p)
        if st["plates"][p].get("pm7m") != "done":
            log(run, "%s: build M7 Max file (file only, not sent)" % p)
            os.makedirs(prn, exist_ok=True)
            mark(p, pm7m="done", pm7m_path=mkm_pm7m_build.build_print_file(store, prn, p, ref_pm7m=a.ref_pm7m))
        if st["plates"][p].get("uv") != "done" and not st["plates"][p].get("uv_checked"):
            log(run, "%s: UVtools detect + repair" % p)
            os.makedirs(uv, exist_ok=True)
            r = mkm_uv_check.uv_check(st["plates"][p]["pm7m_path"], uv, gate=a.gate)
            if r.get("status") == "WAITING_OBS":
                raise RuntimeError("OBS started; UVtools step not run - re-run when the show is over")
            mark(p, uv_status=r.get("status"), uv_checked=True)

    def gpu_finish(p):                      # GPU: parity (one plate) + rating
        store, prn, uv, rate = paths(p)
        ps = st["plates"][p]
        if ps.get("uv") != "done":
            if p == parity_plate or a.gpu_all:
                log(run, "%s: GPU detectors vs UVtools (parity, unproven)" % p)
                mkm_uv_check.parity(ps["pm7m_path"], uv)
            mark(p, uv="done")
        if ps.get("rate") != "done":
            log(run, "%s: rating + heat map (rate_plate writes the picture)" % p)
            os.makedirs(rate, exist_ok=True)
            r = mkm_rate.rate_plate(os.path.join(layout, p), store, uv, rate, device=a.device)
            mark(p, rating=r.get("rating"), rate="done")
        log(run, "%s: %s" % (p, ps.get("rating")))

    for p in plates:                        # a re-run retries plates that failed last time
        with lock:
            ps = st["plates"].setdefault(p, {})
            ps.pop("error", None); ps.pop("traceback", None)
    stopped = {"obs": False}

    def show_started():
        if not stopped["obs"] and C.obs_running():
            stopped["obs"] = True
            log(run, "OBS is running (a show). No further heavy work is started. Re-run when the show is over; finished steps are kept.")
        return stopped["obs"]

    if not a.overlap:
        for p in plates:
            if show_started():
                break
            if guarded(p, "slice", gpu_slice) and guarded(p, "print file / UVtools", cpu_stage) and not show_started():
                guarded(p, "rating", gpu_finish)
    else:
        # --overlap: the GPU slices the next plate (and rates finished ones) while ONE CPU worker runs the print-file build and
        # UVtools of an earlier plate. Never two UVtools at once (RAM). Same steps, same files, same RUN_STATE.json.
        from concurrent.futures import ThreadPoolExecutor
        log(run, "overlap on: GPU work continues while UVtools runs (one UVtools at a time)")
        waiting = []                        # (plate, future) in plate order

        def finish_ready(block):
            while waiting and (block or waiting[0][1].done()):
                q, fut = waiting.pop(0)
                if fut.result() and not show_started():
                    guarded(q, "rating", gpu_finish)

        with ThreadPoolExecutor(max_workers=1) as pool:
            for p in plates:
                if show_started():
                    break
                if guarded(p, "slice", gpu_slice):
                    waiting.append((p, pool.submit(guarded, p, "print file / UVtools", cpu_stage)))
                finish_ready(block=False)
            finish_ready(block=True)

    mkm_rate.summarize(run)
    ok = sum(1 for p in plates if st["plates"][p].get("rate") == "done")
    log(run, "%d of %d plates rated. Summary: %s" % (ok, len(plates), os.path.join(run, "SUMMARY.md")))
    log(run, "STOPPED - waiting for Mike. Nothing was sent to any printer.")
    if stopped["obs"]:
        return 2
    return 0 if ok == len(plates) else 1


if __name__ == "__main__":
    sys.exit(main())
