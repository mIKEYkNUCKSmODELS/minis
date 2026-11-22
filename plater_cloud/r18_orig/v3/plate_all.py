"""MKM plater v3 - one command for the whole run.

    python plate_all.py <planner_out_dir | --pieces pieces.json> <RUN> [--refill] [--ref-pm7m X] [--gate] [--only P3,P7] [--parity P1] [--gpu-all]

Step 0 (only with --pieces): mkm_plan places the pieces at the 4 mm gap with the fewest plates.
Per plate: (d) refill layout (optional) -> (a) slice once into the store -> (a) build .pm7m via UVtools
-> GPU detect -> GPU repair (UVtools Repair layers) -> GPU scan of the repaired plate -> print file from the repaired slices -> (c) rating + heat map. Then the summary. Resumable: RUN/RUN_STATE.json
records each finished step per plate, so a re-run skips what is done. One bad plate never stops the batch.

It NEVER prints and never copies anything to a printer. Printing needs Mike's explicit go, outside this tool.
"""
import argparse, glob, json, os, shutil, sys, time, traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mkm_v3_config as C

STEPS = ("slice", "pm7m", "uv", "rate")


def log(run, msg):
    line = time.strftime("%Y-%m-%d %H:%M:%S ") + msg
    print(line, flush=True)
    with open(os.path.join(run, "plate_all.log"), "a", encoding="utf-8") as f:
        f.write(line + "\n")


STATE_NAME = "RUN_STATE.json"


def invalidate_if_stale_store(ps, store, version):
    """R7 (verifier R2 finding 10): a RUN_STATE that says 'slice done' must not keep a store made by older slicer code (or an unfinished one).
    If the store's meta.json is missing, incomplete or carries another slicer_version, every step of this plate is reset so it is sliced again. Returns True if it reset."""
    if ps.get("slice") != "done":
        return False
    try:
        m = json.load(open(os.path.join(store, "meta.json"), encoding="utf-8"))
    except (OSError, ValueError):
        m = {}
    if m.get("complete") and m.get("slicer_version") == version:
        return False
    for k in ("slice", "repair", "uv", "uv_status", "uv_parity", "rate", "pm7m", "detect_mode", "rating"):
        ps.pop(k, None)
    return True


def load_state(run):
    p = os.path.join(run, STATE_NAME)
    if not os.path.exists(p):
        p = os.path.join(run, "RUN_STATE.json")          # a concurrent worker starts from the shared state, then keeps its own
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else {"plates": {}}


def save_state(run, st):
    p = os.path.join(run, STATE_NAME)
    tmp = p + ".tmp"
    json.dump(st, open(tmp, "w", encoding="utf-8"), indent=1)
    os.replace(tmp, p)


class GpuSampler:
    """Mike 15:11: GPU-busy % over the run. nvidia-smi every 2 s in a thread. It reads the WHOLE card (other processes included; the gate lane
    keeps WRAITH off during a plater window), so it is reported as card utilisation, never as this process alone."""

    def __init__(self, every=2.0):
        import threading
        self.s, self.every, self._stop = [], every, threading.Event()
        self.t = threading.Thread(target=self._run, daemon=True); self.t.start()

    def _run(self):
        import subprocess
        while not self._stop.is_set():
            try:
                o = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"], capture_output=True,
                                   text=True, timeout=10, creationflags=0x08000000).stdout.split(",")
                self.s.append((time.time(), float(o[0]), float(o[1])))
            except Exception:
                pass
            self._stop.wait(self.every)

    def window(self, t0, t1):
        v = [x for x in self.s if t0 <= x[0] <= t1]
        if not v:
            return dict(samples=0)
        return dict(samples=len(v), avg_util_pct=round(sum(x[1] for x in v) / len(v), 1), samples_ge_50pct=round(100.0 * sum(1 for x in v if x[1] >= 50) / len(v), 1),
                    samples_0pct=round(100.0 * sum(1 for x in v if x[1] < 1) / len(v), 1), max_mem_mb=int(max(x[2] for x in v)), every_s=self.every)

    def stop(self):
        self._stop.set()


class AsyncFeed:
    """Hands every finished slice batch to the GPU detector on its OWN CUDA stream in a worker thread, so slicing the next batch overlaps the
    detector's host-sync gaps (profile 11/21: slicer 8-25 ms/layer, detector 20-75 ms/layer). Queue depth 1 keeps VRAM to ~2 batches in flight.
    A detector exception is re-raised in the slicing thread at the next hand-over / at close()."""

    def __init__(self, det, depth=1):
        import queue, threading, torch
        self.det, self.err, self.torch = det, None, torch
        self.q = queue.Queue(depth); self.s2 = torch.cuda.Stream()
        self.s2.wait_stream(torch.cuda.current_stream())          # the detector's state tensors were built on the default stream
        self.t = threading.Thread(target=self._run, daemon=True); self.t.start()

    def _run(self):
        torch = self.torch
        while True:
            it = self.q.get()
            if it is None:
                return
            masks, ev = it
            try:
                if self.err is None:
                    with torch.cuda.stream(self.s2):
                        self.s2.wait_event(ev)
                        self.det.feed(masks)
            except Exception as e_:
                self.err = e_
            del masks, it

    def __call__(self, masks):
        if self.err is not None:
            raise self.err
        masks.record_stream(self.s2)                              # the slicer frees/reuses its tensor on the default stream: keep it alive for s2
        ev = self.torch.cuda.Event(); ev.record()
        self.q.put((masks, ev))

    def close(self):
        self.q.put(None); self.t.join()
        self.torch.cuda.current_stream().wait_stream(self.s2); self.torch.cuda.synchronize()
        if self.err is not None:
            raise self.err


def pick_batch(det_on, floor=8, ceil=96):
    """Slice batch from the VRAM that is free NOW (Mike 15:14), not a fixed 32: the (B+2,H,W) bool masks plus ~2 working copies must fit in
    35% of what this process may still use (free VRAM, and MKM_VRAM_FRACTION / the gpuport cap when set)."""
    try:
        import torch
        free, total = torch.cuda.mem_get_info()
        cap = float(os.environ.get("MKM_VRAM_FRAC", "0.45")) * total
        usable = min(free, max(0.0, cap - torch.cuda.memory_allocated()))
        W, H = C.RES
        return int(max(floor, min(ceil, 0.35 * usable / (3.0 * H * W) / (2.0 if det_on else 1.0))))
    except Exception:
        return 32 if det_on else 64


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


CUT_PAD = 8     # px searched around a model's own window: repair may add pixels just outside the exact fill


def cut_owned(x, own, win, cc_cache, dev):
    """R6: clear ONE model's cured pixels from x [B,H,W] bool (in place) by PER-PIXEL OWNERSHIP; returns the number of pixels cleared.
    own [B,h,w] = that model's exact pixels (mkm_plate_slice.owned_window_masks), win = (c0,r0,c1,r1) its half-open window in the same pixel grid as x.
    Every 4-connected cured component of x that holds at least one of the model's own pixels is cleared (that also removes what repair added to the
    model, a hollow's wall, a ring around a neighbour). Models never touch (4 mm gap), so a neighbour's component never holds an own pixel and is never
    touched, whatever the boxes look like (no box test anywhere)."""
    import torch, mkm_gpu_detect as D
    c0, r0, c1, r1 = win
    H, W = x.shape[1:]
    R0, R1, C0, C1 = max(0, r0 - CUT_PAD), min(H, r1 + CUT_PAD), max(0, c0 - CUT_PAD), min(W, c1 + CUT_PAD)
    h, w = R1 - R0, C1 - C0
    cc = cc_cache.get((h, w)) or cc_cache.setdefault((h, w), D._CC(h, w, dev))
    cleared = 0
    for b in range(x.shape[0]):
        crop = x[b, R0:R1, C0:C1]
        own_f = torch.zeros((h, w), dtype=torch.bool, device=x.device)
        own_f[r0 - R0:r1 - R0, c0 - C0:c1 - C0] = own[b]
        hit = crop & own_f
        if not bool(hit.any()):
            continue
        idx = torch.nonzero(crop.reshape(-1)).squeeze(1)
        lab, k = cc.labels(idx)
        comp = torch.zeros(k, dtype=torch.bool, device=x.device)
        comp[lab[hit.reshape(-1)[idx]]] = True
        kill = torch.zeros(h * w, dtype=torch.bool, device=x.device); kill[idx[comp[lab]]] = True
        cleared += int(kill.sum()); crop &= ~kill.reshape(h, w)
    return cleared


def _mesh_dev_label():
    """RUN_STATE device of the source-mesh soundness check (Mike 0140): CUDA unless MKM_MESH_DEV=cpu / no CUDA; the STL read itself is file I/O on the CPU"""
    import mkm_mesh_check as _MC
    return "cuda (torch.unique on the weld grid + edge keys; STL read = CPU file I/O)" if _MC._gpu_dev() is not None else "cpu (numpy fallback: no CUDA or MKM_MESH_DEV=cpu)"


def count_bodies(store_dir, device, layer=0):
    """R6 (verifier R5 #5, 'count bodies, not files'): number of separate 4-connected cured bodies on `layer` of a store (the first layer is the footprint on the plate;
    a pre-built ring STL of 33 pieces is ONE source file but many bodies)."""
    import torch, mkm_gpu_detect as D
    from mkm_slice_store import SliceStore
    st = SliceStore(store_dir); x = st.read_batch(layer, layer + 1, device=device).bool()[0]
    idx = torch.nonzero(x.reshape(-1)).squeeze(1)
    if not idx.numel():
        return 0
    return int(D._CC(x.shape[0], x.shape[1], x.device).labels(idx)[1])


def models_on_layer0(store_dir, tris, skip, origin, r, W, H, mirror, device, units=None):
    """number of MODELS (not bodies, not files) whose own pixels stand on layer 0 of a store, ignoring model numbers in `skip`.  B4 counts models: one source with two feet is one model, and
    (R12, verifier R9 H1) sources that share a planner unit (a figure's body + its own base files) are ONE model: `units` = the unit of each source, in source order (None = every source its own model)."""
    import mkm_plate_slice as PS, torch
    from mkm_slice_store import SliceStore
    dev = torch.device(device)
    x0 = SliceStore(store_dir).read_batch(0, 1, device=device).bool()[0]
    left = set()
    for m in range(1, len(tris) + 1):
        if m in skip:
            continue
        win = PS.model_window(tris[m - 1], origin, r, W, H, pad=4)
        if win is None:
            continue
        win_s, own = PS.owned_window_masks(tris[m - 1], win, 0, 1, origin, r, W, dev, mirror=mirror)
        c0, r0, c1, r1 = win_s
        if bool((x0[r0:r1, c0:c1] & own[0]).any()):
            left.add(units[m - 1] if (units and units[m - 1]) else ("_src", m))
    return len(left)


def prepare_print_store(run, p, rstore, rj, device, recipe_path):
    """H2/R6: -> (store to build the print file from | None, dropped model numbers, note). No rejected model: the repaired store itself. Else a copy of it with each
    rejected model's own pixels cleared on the GPU by per-pixel ownership (the model's exact fill from its own triangles; never a box, never a whole-plate re-slice).
    Blocked (None) when fewer than 2 bodies remain on the first layer (B4: no single-model plate)."""
    rej = sorted(int(x["model"]) for x in (rj.get("rejected_models") or []))
    import mkm_plate_slice as PS, torch
    from mkm_slice_store import SliceStore
    st = SliceStore(rstore); N = st.n_layers
    W, H = st.meta["res"]; origin = tuple(st.meta["origin_mm"]); r = C.PIXEL_MM[0]; mirror = bool(st.meta.get("mirror_x", True))
    recipe = json.load(open(recipe_path, encoding="utf-8-sig"))
    tris, _top = PS.load_plate(recipe)                           # sha256-verified, matrix applied once, as the slice did
    units = [s.get("unit") for s in recipe["sources"]]           # R12: the planner's model id of each source (a figure's body + base share one)
    if not rej:
        # R11 (verifier R8 HIGH-3): B4 is checked even when nothing was rejected (a planned single-model remainder plate must not get a print file)
        left0 = models_on_layer0(rstore, tris, [], origin, r, W, H, mirror, device, units)
        if left0 < 2:
            return None, [], "plate stands %d model(s) on the first layer: print file NOT built (a plate needs at least 2 models, B4)" % left0
        return rstore, [], None
    wins = {m: PS.model_window(tris[m - 1], origin, r, W, H, pad=4) for m in rej}
    out = os.path.join(run, "store_print", p)
    o = SliceStore(out, "w")
    o.write_meta(**{k: st.meta[k] for k in ("res", "pixel_mm", "layer_mm", "n_layers", "mirror_x", "origin_mm") if k in st.meta}, recipe_sha256=st.meta.get("recipe_sha256"))
    cleared = 0; cc_cache = {}; dev = torch.device(device)
    for i0 in range(0, N, 16):
        i1 = min(N, i0 + 16)
        x = st.read_batch(i0, i1, device=device).bool()
        for m in rej:
            if wins[m] is None:
                continue
            win_s, own = PS.owned_window_masks(tris[m - 1], wins[m], i0, i1, origin, r, W, dev, mirror=mirror)
            cleared += cut_owned(x, own, win_s, cc_cache, dev)
        o.append_batch(x)
    o.meta["rejected_models_removed"] = rej; o.meta["model_checks"] = st.meta.get("model_checks"); o.meta["repair_stats"] = st.meta.get("repair_stats"); o.close()      # repair_stats: the rating maps repaired layers back to raw ones by it
    bodies = count_bodies(out, device)
    # R9 (verifier R6 #4): B4 counts MODELS, not bodies. One source with two disconnected feet is two bodies but ONE model.
    left = models_on_layer0(out, tris, rej, origin, r, W, H, mirror, device, units)
    if left < 2:
        return None, rej, "after removing rejected models %s only %d model(s) (%d bod(ies)) still stand on the first layer: print file NOT built (a plate needs at least 2 models, B4)" % (rej, left, bodies)
    return out, rej, "print file built WITHOUT rejected model(s) %s (%d cured px cleared by per-pixel ownership, %d models / %d bodies stand on layer 0); the plate needs a substitute" % (rej, cleared, left, bodies)


def rate_print(run, p, layout_p, pstore, dropped, gpu, device, mesh=None):
    """R9 (verifier R6 #3): the PRINT store is rated again (pixel pass, heat map, issues vs the raw scan), so rating.json / heat map / SUMMARY describe what is printed, not the uncut plate.
    The print scan (issues_print) is handed to the rater under the name it reads for "the plate after repair"."""
    import mkm_rate_px
    gp = os.path.join(run, "gpu_print", p); os.makedirs(gp, exist_ok=True)
    for src, dst in (("issues_raw.json", "issues_raw.json"), ("stdout_print_issues_raw.txt", "stdout_print_issues_raw.txt"),
                     ("issues_print.json", "issues_repaired.json"), ("stdout_print_issues_print.txt", "stdout_print_issues_repaired.txt")):
        s = os.path.join(gpu, src)
        if os.path.isfile(s):
            shutil.copy2(s, os.path.join(gp, dst))
    return mkm_rate_px.rate_plate(layout_p, pstore, gp, os.path.join(run, "rate_print", p), device=device, mesh=mesh, dropped=dropped,
                                  basis="print store: repaired store minus rejected models %s" % list(dropped))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("planner_out", nargs="?", default=None, help="finished planner output (P*/recipe.json); omit when --pieces is given")
    ap.add_argument("--pieces", default=None, help="pieces.json: run the planner first (4 mm gap, fewest plates), then everything below")
    ap.add_argument("run")
    ap.add_argument("--refill", action="store_true", help="step (d): re-place pieces to use the whole bed")
    ap.add_argument("--ref-pm7m", default=None, help="a known-good M7 Max file to copy lift/bottom-layer parameters from")
    ap.add_argument("--gate", action="store_true", help="send heavy steps through the MKM gate")
    ap.add_argument("--only", default=None, help="comma list of plates, e.g. P3,P7")
    ap.add_argument("--parity", default=None, help="OPT-IN: plate on which real UVtools detection also runs (CPU, ~6 min) as the yardstick for the GPU detectors; compare with parity_report.py")
    ap.add_argument("--gpu-all", action="store_true", help="run the GPU detectors on every plate, not only the parity plate")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--worker", default=None, help="concurrent worker tag: own state file, no summary (plate_batch.py merges)")
    ap.add_argument("--uv-repair", action="store_true", help="old route: UVtools repair on the CPU instead of the GPU repair")
    a = ap.parse_args(argv)

    run = os.path.abspath(a.run)
    import time as _t0m
    RUN_T0 = _t0m.time(); LIMIT_S = float(os.environ.get("MKM_MAX_RUN_S", 30 * 60))      # R10 BLOCKER 3: the 30-minute run limit starts HERE, so the planning step is inside it (it used to start after the planner)
    if os.environ.get("MKM_VRAM_FRAC") and a.device == "cuda":   # ONE env name, read here AND by mkm_gpuport (R6): a concurrent worker keeps to its share of the card
        import torch
        torch.cuda.set_per_process_memory_fraction(float(os.environ["MKM_VRAM_FRAC"]))
    global STATE_NAME
    if a.worker:
        STATE_NAME = "RUN_STATE_%s.json" % a.worker
    os.makedirs(run, exist_ok=True)
    if C.obs_running():
        log(run, "OBS is running (a show). Heavy jobs are not started. Re-run when the show is over.")
        return 2

    if a.ref_pm7m is None and os.path.isfile(C.REF_PM7M):
        a.ref_pm7m = C.REF_PM7M                      # real M7 Max file: bottom layers, lifts (exposure stays Mike's 50 s / 2.45 s)
    import mkm_plan
    if a.pieces and not a.planner_out:
        # R10 BLOCKER 3: ONE entry point.  0a: separate CPU job fills the footprint + band caches; 0b: the NEW planner (cross-section, 2 mm margin, per-band collision, layout score + tolerance + peel term,
        # plate count from the math, hard checks that drop); both inside the run limit.
        log(run, "step 0a: footprint + height-band caches (separate job, GPU unless MKM_PRECOMPUTE=cpu) for %s" % a.pieces)
        try:
            pre = mkm_plan.precompute(a.pieces, os.path.join(run, "planner"), deadline=RUN_T0 + LIMIT_S, gate=a.gate)
            log(run, "step 0a done: %s" % pre["line"])
            log(run, "step 0b: planner on %s (gap %s mm, plate count from the math)" % (a.pieces, C.GAP_MM))
            pr_ = mkm_plan.plan(a.pieces, os.path.join(run, "planner"), gate=a.gate, deadline=RUN_T0 + LIMIT_S)
        except TimeoutError as e_:
            log(run, "STOPPED CLEANLY by the run limit during planning: %s (limit %.1f min)" % (e_, LIMIT_S / 60.0)); return 4
        except RuntimeError as e_:
            log(run, "REFUSED: the planner could not produce a plan without a hard-check failure: %s" % e_); return 3
        a.planner_out = pr_["out"]
        log(run, "planned: %d plates (math said %s), tries %s, hard-check drops %d" % (pr_["n"], pr_.get("n_math"), pr_.get("tries"), len(pr_.get("hard_dropped", []))))
    if not a.planner_out:
        ap.error("give a planner output folder or --pieces")
    # MIKE 2202 (ChatGPT R14 patch, merged): even a resumed/external planner folder must not bypass the
    # required case-and-raft source gate. Explicit MKM_CASE_MODE=off is A/B ONLY.
    if os.environ.get("MKM_CASE_MODE", "on").lower() == "on":
        import mkm_accounting as _case_accounting
        _case_pieces = ((pr_.get("pieces") if "pr_" in locals() else None)
                        or _case_accounting.pieces_file_for(a.planner_out))
        try:
            _case_rows = json.load(open(_case_pieces, encoding="utf-8-sig")) if _case_pieces else []
            if not _case_rows or not all(r.get("case_raft", {}).get("verified") is True for r in _case_rows):
                raise ValueError("plan has naked/unverified models or no certified composite inventory")
        except (OSError, ValueError, TypeError) as ex_:
            log(run, "REFUSED CASE+RAFT: %s. No slicing or printer output." % ex_)
            return 3
    # R10 BLOCKER 2: a plan that carries a hard_fail of any kind is refused, whoever made it (plan result or any plate's hard_checks.json)
    _hf = [("%s" % p_, r_) for p_, r_ in mkm_plan.hard_fails_in(a.planner_out)]
    for _pr in (os.path.join(os.path.dirname(os.path.abspath(a.planner_out)), "PLAN_RESULT.json"), os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(a.planner_out))), "PLAN_RESULT.json")):
        if os.path.exists(_pr):
            try:
                _hf += [("plan", [str(x)[:160]]) for x in json.load(open(_pr, encoding="utf-8")).get("hard_fail", [])]
            except Exception:
                pass
    if _hf:
        os.makedirs(run, exist_ok=True); json.dump(_hf, open(os.path.join(run, "REFUSED_HARD_FAIL.json"), "w"), indent=1)
        log(run, "REFUSED: the plan carries a hard-check failure (not anchored / footprints or height bands closer than the 4 mm gap / box inside the 2 mm margin): %s. Nothing was sliced." % "; ".join("%s %s" % (p_, "; ".join(r_)[:120]) for p_, r_ in _hf[:4]))
        return 3
    # R14 (Mike 21:59): MODEL ACCOUNTING before anything is sliced.  Every pieces.json model is placed or carried, never both; no failed plate; plate folders = plates written.
    import mkm_accounting
    _pj = ((pr_.get("pieces") if "pr_" in locals() else None)
           or mkm_accounting.pieces_file_for(a.planner_out) or a.pieces)      # case-on: the planned inventory (composites), not the naked paths
    if _pj:
        _acc = mkm_accounting.model_accounting(_pj, a.planner_out)
        if _acc:
            os.makedirs(run, exist_ok=True); json.dump(_acc, open(os.path.join(run, "REFUSED_ACCOUNTING.json"), "w"), indent=1)
            log(run, "REFUSED: model accounting failed (%d problems): %s. Nothing was sliced." % (len(_acc), "; ".join(_acc[:3])[:400])); return 3
        _acc_line = "model accounting: every model in %s is placed or carried, none both, no failed plate" % os.path.basename(_pj)
    else:
        _acc_line = "model accounting: NOT CHECKED (no pieces.json recorded for this planner output) - UNPROVEN"
    log(run, _acc_line)
    import mkm_plate_slice, mkm_pm7m_build, mkm_uv_check, mkm_uv_repair, mkm_gpu_detect, mkm_gpu_repair, mkm_rate, mkm_rate_px
    layout = os.path.join(run, "layout")
    st = load_state(run)
    _eff = mkm_accounting.effective_settings(a.planner_out)      # R14: what the planner REALLY used, from its own hard_checks.json (not from a label)
    st["settings"] = dict(planner_out=os.path.abspath(a.planner_out), edge_margin_mm_configured=C.EDGE_MARGIN_MM, gap_mm=C.GAP_MM, fill_min_configured=float(os.environ.get("MKM_FILL_MIN", C.FILL_MIN)), plates_effective=_eff["plates"], carried=_eff["carried"], accounting=_acc_line, case=mkm_accounting.case_mode(),      # R17: CASE=OFF/ON is part of the effective settings
                          run_limit_s=LIMIT_S, run_limit_default_s=30 * 60, run_limit_override=("MKM_MAX_RUN_S" in os.environ and LIMIT_S != 30 * 60))
    _fm = sorted({v.get("fill_min_pct") for v in _eff["plates"].values() if v.get("fill_min_pct") is not None}); _en = sorted({v.get("edge_need_mm") for v in _eff["plates"].values() if v.get("edge_need_mm") is not None})
    log(run, "SETTINGS: edge margin configured %.1f mm, planner used %s mm | fill minimum configured %.0f%%, planner used %s%% | gap %.0f mm | %s | run limit %.1f min%s" % (
        C.EDGE_MARGIN_MM, _en or "UNKNOWN (planner output has no record)", 100 * st["settings"]["fill_min_configured"], _fm or "UNKNOWN (planner output has no record)", C.GAP_MM, st["settings"]["case"]["label"], LIMIT_S / 60.0,
        "  *** OVERRIDE for this test only (MKM_MAX_RUN_S); the default stays 30 min ***" if st["settings"]["run_limit_override"] else " (default)"))
    save_state(run, st)
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
    log(run, "%d plates: %s" % (len(plates), ", ".join(plates)))

    import time as _t, subprocess
    pend = {}
    MAX_BUILDS = int(os.environ.get("MKM_MAX_BUILDS", "2"))

    def reap(wait):
        """D8: record every finished print-file build the moment we look (and all of them at the end when wait=True). pm7m_seconds is the
        build's own duration: start -> the output file's last write (never the time the batch got around to collecting it)."""
        for p_ in list(pend):
            pr, lf, t0 = pend[p_]
            if wait:
                try:
                    pr.wait(timeout=float(os.environ.get("MKM_BUILD_WAIT_S", 900)))        # R9 (verifier R6 #8): no unlimited wait; a stuck build is killed by tree
                except subprocess.TimeoutExpired:
                    subprocess.run(["taskkill", "/T", "/F", "/PID", str(pr.pid)], capture_output=True, creationflags=0x08000000)
                    pr.wait()
            elif pr.poll() is None:
                continue
            lf.close(); del pend[p_]
            ps_ = st["plates"][p_]; f = os.path.join(run, "print", p_, p_ + ".pm7m")
            if pr.returncode == 0 and os.path.isfile(f):
                ps_["pm7m"] = "done"; ps_["pm7m_path"] = f; ps_["pm7m_seconds"] = round(max(0.0, os.path.getmtime(f) - t0), 1)
                ps_.pop("pm7m_error", None)
                if os.environ.get("MKM_SSD_CACHE", "0") == "1":      # ChatGPT SSD patch: a finished-plate checkpoint is written only after the print file exists and the canonical store is complete
                    try:
                        import mkm_ssd_cache as SSD
                        SSD.PlateCheckpoints(run).record(p_, f, (os.path.join(run, "store_print", p_) if os.path.isdir(os.path.join(run, "store_print", p_)) else os.path.join(run, "store_repaired", p_)), {"case_mode": os.environ.get("MKM_CASE_MODE", "on").lower()})
                    except Exception as _ex:
                        ps_["ssd_checkpoint"] = "skipped: %s" % str(_ex)[:120]
                log(run, "%s: print file ready %s (%.0f s build)" % (p_, f, ps_["pm7m_seconds"]))
            else:
                ps_["pm7m_error"] = "build exit %s, see %s" % (pr.returncode, os.path.join(run, "print", p_, "build.log"))
                log(run, "%s: print file FAILED (%s)" % (p_, ps_["pm7m_error"]))
            save_state(run, st)
    import threading, mkm_mesh_check

    def mesh_for(recipe_path, out):
        """D7: soundness of every source mesh (read-only, numpy, cached by the file's sha256 in RUN/mesh). Runs on the CPU in a thread beside the GPU slice."""
        rec = json.load(open(recipe_path, encoding="utf-8-sig")); res = []
        os.makedirs(os.path.join(run, "mesh"), exist_ok=True)
        for s in rec["sources"]:
            cp = os.path.join(run, "mesh", (s.get("sha256") or "nosha") + ".json")
            if s.get("sha256") and os.path.isfile(cp):
                rep = json.load(open(cp, encoding="utf-8"))
            else:
                try:
                    rep = mkm_mesh_check.check_file(s["path"])
                except Exception as e_:
                    rep = dict(error=repr(e_), sound=None)
                if s.get("sha256") and "error" not in rep:
                    json.dump(rep, open(cp, "w", encoding="utf-8"))
            rep = dict(rep); rep["file"] = os.path.basename(s["path"]); res.append(rep)
        out.extend(res)

    sampler = GpuSampler()
    T0 = RUN_T0      # Mike 11/21 14:03: no run holds RAM/VRAM over 30 minutes (R10: the clock started before the planner, LIMIT_S is set at the top of main)

    class Watchdog(Exception):
        pass

    def chk():
        """R6 watchdog, inside plate_all (the gate lanes are not a job limit): between steps a run past LIMIT_S stops cleanly; finished plates and their files stay."""
        if _t.time() - T0 > LIMIT_S:
            raise Watchdog("run limit reached (%.1f min; limit %.1f min)" % ((_t.time() - T0) / 60.0, LIMIT_S / 60.0))
    wd_hit = None
    for p in plates:
        if wd_hit:
            st["plates"].setdefault(p, {})["not_run"] = "watchdog: " + wd_hit
            continue
        ps = st["plates"].setdefault(p, {})
        mesh_out = []; mesh_t = threading.Thread(target=mesh_for, args=(os.path.join(layout, p, "recipe.json"), mesh_out), daemon=True); mesh_t0 = _t.time(); mesh_t.start()
        store = os.path.join(run, "store", p)
        prn = os.path.join(run, "print", p)
        uv = os.path.join(run, "uv", p)          # UVtools: repair only (+ the yardstick scan on the --parity plate)
        gpu = os.path.join(run, "gpu", p)        # GPU detectors: issues_raw / issues_repaired
        rate = os.path.join(run, "rate", p)
        pm7m = os.path.join(prn, p + ".pm7m")
        tp = _t.time()
        sliced_now = False; did_repair = False; did_rate = False; ps.pop("not_run", None)      # R9: a plate that runs again is no longer "not run"
        try:
            chk()
            if invalidate_if_stale_store(ps, store, mkm_plate_slice.SLICER_VERSION):
                log(run, "%s: the stored slices are from an older slicer (or unfinished): every step of this plate is redone" % p); save_state(run, st)
            need_raw = not os.path.isfile(os.path.join(gpu, "issues_raw.json"))
            if ps.get("slice") != "done" or need_raw:
                import torch
                W, H = C.RES
                det = mkm_gpu_detect.Detector(H, W, a.device) if ps.get("slice") != "done" else None
                log(run, "%s: slice once at printer pixels%s" % (p, " + GPU detectors in the same pass" if det else ""))
                feed = AsyncFeed(det) if (det is not None and a.device == "cuda" and os.environ.get("MKM_ASYNC_FEED") == "1") else None
                bsz = pick_batch(det is not None) if a.device == "cuda" else 32
                ps["slice_batch"] = bsz; ps["feed"] = "async (own CUDA stream, opt-in MKM_ASYNC_FEED=1: measured no gain)" if feed else "inline"
                try:
                    info = mkm_plate_slice.slice_plate(os.path.join(layout, p, "recipe.json"), store, batch=bsz, device=a.device, on_batch=(feed if feed else (det.feed if det else None)))
                finally:
                    if feed:
                        feed.close()
                ps["slice"] = "done"; ps["slice_seconds"] = info.get("seconds"); sliced_now = not info.get("skipped"); save_state(run, st); chk()
                if os.environ.get("MKM_SSD_CACHE", "0") == "1":      # ChatGPT SSD patch (Mike 0152): optional packed sidecar of the FINISHED canonical store; the canonical RLE stays authoritative, any failure just skips it
                    try:
                        import mkm_ssd_cache as SSD
                        from mkm_slice_store import SliceStore as _SS; _sc = SSD.build_packed_sidecar(store, _SS, {"case_mode": os.environ.get("MKM_CASE_MODE", "on").lower()}); ps["ssd_sidecar"] = str(_sc) if _sc else "none (not enough free space)"
                    except Exception as _ex:
                        ps["ssd_sidecar"] = "skipped: %s" % str(_ex)[:120]
                if det is not None and not info.get("skipped"):
                    j = mkm_gpu_detect.finish_and_write(det, gpu, "raw", dict(grid="printer pixels %dx%d, every layer, same pass as the slice" % (W, H), device=a.device, store=store))
                    ps["detect_mode"] = "same pass as the slice"
                else:
                    del det
                    log(run, "%s: GPU detectors on the stored slices" % p)
                    j = mkm_gpu_detect.detect_store(store, gpu, "raw", device=a.device, log=lambda s: log(run, "%s: %s" % (p, s)))
                    ps["detect_mode"] = "stored slices"
                ps["gpu_raw_seconds"] = j.get("seconds"); ps["gpu_raw_counts"] = j.get("counts"); save_state(run, st)
                torch.cuda.empty_cache()
            rstore = os.path.join(run, "store_repaired", p)
            if ps.get("repair") != "done":
                chk()
                import torch
                log(run, "%s: repair on the GPU (UVtools Repair layers: islands, resin traps, gap closing)" % p)
                # R9: repair, re-scan, and sweep the sealed pockets the first pass's own gap closing created (repair_until_clean); the final re-scan is issues_repaired.*
                r, j = mkm_gpu_repair.repair_until_clean(store, gpu, rstore, device=a.device, log=lambda s: log(run, "%s: %s" % (p, s)))
                ps["repair_stats"] = r; ps["repair_seconds"] = r.get("total_seconds", r.get("seconds")); save_state(run, st)
                ps["gpu_repaired_seconds"] = j.get("seconds"); ps["gpu_repaired_counts"] = j.get("counts")
                ps["traps_after_pass1"] = r.get("traps_after_pass1"); ps["traps_after_pass2"] = r.get("traps_after_pass2")
                ps["repair"] = "done"; ps["uv"] = "done"; ps["uv_status"] = "GPU repair"; did_repair = True; save_state(run, st)
                torch.cuda.empty_cache() if a.device == "cuda" else None
                chk()
            if a.uv_repair and ps.get("uv_parity") != "done":
                # R3 #5: --uv-repair is a PARITY route only (real UVtools repair on the RAW slices, scanned as "uvtools_repaired" beside the GPU repair).
                # Its file lives in RUN/uv_input and is never a print file: the print file always comes from the rating step below (accepted models only).
                log(run, "%s: parity: raw slices -> UVtools file -> UVtools repair on the CPU (--uv-repair; not a print file)" % p)
                uvin = os.path.join(run, "uv_input", p); os.makedirs(uvin, exist_ok=True); os.makedirs(uv, exist_ok=True)
                raw_pm7m = mkm_pm7m_build.build_print_file(store, uvin, p, ref_pm7m=a.ref_pm7m)
                r = mkm_uv_repair.repair_only(raw_pm7m, uv, gate=a.gate)
                ps["uv_parity_status"] = r.get("status"); ps["uv_parity_repair_seconds"] = r.get("repair_seconds")
                if r.get("repaired_pm7m"):
                    fl, iou, lay = mkm_gpu_detect.find_flip(store, r["repaired_pm7m"], a.device)
                    j = mkm_gpu_detect.detect_pm7m(r["repaired_pm7m"], gpu, "uvtools_repaired", device=a.device, flip=fl, log=lambda s: log(run, "%s: %s" % (p, s)))
                    ps["uv_parity_counts"] = j.get("counts")
                ps["uv_parity"] = "done"; save_state(run, st)
            if ps.get("rate") != "done":
                log(run, "%s: rating + heat map (rate_plate writes the picture)" % p)
                os.makedirs(rate, exist_ok=True)
                # D8: rate what the print file is built from - the REPAIRED store (the raw store only when repair did not run, --uv-repair)
                rated = rstore
                ps["rated_store"] = "repaired"
                mesh_t.join(); ps["mesh_seconds"] = round(_t.time() - mesh_t0, 1)
                r = mkm_rate_px.rate_plate(os.path.join(layout, p), rated, gpu, rate, device=a.device, mesh=(list(mesh_out) if mesh_out else None))   # L1: a mesh thread that produced nothing is "not checked", never a silent pass
                ps["rating"] = r.get("rating")
                ps["rate"] = "done"; did_rate = True; save_state(run, st)
            if ps.get("pm7m") != "done" and ps.get("rate") == "done":
                # H2 (verifier R2): the print file is built AFTER the rating, from the repaired store with every REJECTED model removed (A4/A5: a failed model is
                # never pushed through to the printer). Removal = those pixels cleared in a copy; nothing is re-sliced.
                rj = json.load(open(os.path.join(rate, "rating.json"), encoding="utf-8"))
                pstore, dropped, note = prepare_print_store(run, p, rstore, rj, a.device, os.path.join(layout, p, "recipe.json"))
                ps["print_store"] = "repaired" if pstore == rstore else ("repaired minus rejected models %s" % dropped if pstore else "none"); ps["dropped_models"] = dropped
                if note:
                    ps["print_file_note"] = note; log(run, "%s: %s" % (p, note))
                if pstore and pstore != rstore:
                    # R6 (verifier R5 #5): the print store is a NEW store (rejected models cut out): it is scanned again; what it holds is what gets printed.
                    j = mkm_gpu_detect.detect_store(pstore, gpu, "print", device=a.device, log=lambda s: log(run, "%s: %s" % (p, s)))
                    ps["gpu_print_counts"] = j.get("counts"); ps["gpu_print_seconds"] = j.get("seconds")
                    ps["print_bodies_layer0"] = count_bodies(pstore, a.device)
                    rc = ps.get("gpu_repaired_counts") or {}; pc = j.get("counts") or {}
                    more = {k: (pc.get(k, 0) - rc.get(k, 0)) for k in set(pc) | set(rc) if pc.get(k, 0) > rc.get(k, 0)}
                    ps["print_scan_vs_repaired"] = ("print store has MORE issues than the repaired plate: %s (cutting created them)" % more) if more else "no new issues from the cut"
                    rp_ = rate_print(run, p, os.path.join(layout, p), pstore, dropped, gpu, a.device, (list(mesh_out) if mesh_out else None))
                    ps["print_rating"] = rp_.get("rating"); ps["print_rating_top"] = rp_.get("top_reason"); ps["print_rate_seconds"] = rp_.get("seconds")
                    if more and rp_.get("rating") in ("LIKELY", "LIKELY WITH NOTES"):
                        ps["print_rating"] = "RISKY"; ps["print_rating_top"] = ps["print_scan_vs_repaired"]
                if pstore:
                    while len([q for q in pend.values() if q[0].poll() is None]) >= MAX_BUILDS:
                        _t.sleep(2)
                    os.makedirs(prn, exist_ok=True)
                    cmd = [sys.executable, "-I", "-u", "-c", "import sys, runpy; sys.path.insert(0, sys.argv[1]); sys.argv = sys.argv[2:]; runpy.run_path(sys.argv[0], run_name=\"__main__\")", HERE, os.path.join(HERE, "mkm_pm7m_build.py"), pstore, prn, p] + (["--ref-pm7m", a.ref_pm7m] if a.ref_pm7m else [])
                    lf = open(os.path.join(prn, "build.log"), "w")
                    pend[p] = (subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, creationflags=0x08000000), lf, _t.time())
                    log(run, "%s: print file building in the background on the CPU (file only, not sent); GPU goes on" % p)
                else:
                    ps["pm7m_error"] = "not built: " + (note or "no print store")
            ps["minutes"] = round((_t.time() - tp) / 60.0, 1)
            ps["gpu_busy"] = dict(sampler.window(tp, _t.time()), scope="WHOLE CARD (nvidia-smi, every process on it) over this plate's wall time in this process; not this process alone")
            dv = a.device; ps["devices"] = dict(slice=("%s (GSlicer + gfill, torch)" % dv) if sliced_now else "store reused from an earlier run: not re-sliced, device of that run not recorded here",
                                 detect_raw=("%s (same pass as the slice)" % dv) if ps.get("detect_mode") == "same pass as the slice" else ("%s (on the stored slices)" % dv), repair=(dv if did_repair else "ran in an earlier session (device not recorded here)"), detect_repaired=(dv if did_repair else "ran in an earlier session (device not recorded here)"), rating=(dv if did_rate else "ran in an earlier session (device not recorded here)"),
                                 mesh_check=_mesh_dev_label(), print_file="cpu (UVtools convert / .pm7m encoder; file I/O bound, no GPU path)")
            ps.pop("error", None); ps.pop("traceback", None)        # D8: a plate that got through every step carries no old error
            save_state(run, st)
            reap(False)
            log(run, "%s: %s  (%.1f min this session)" % (p, ps.get("rating"), (_t.time() - tp) / 60.0))
        except Watchdog as e_:
            wd_hit = str(e_); ps["not_run"] = "watchdog: " + wd_hit; save_state(run, st); log(run, "%s: STOPPED CLEANLY - %s; finished plates are kept, the rest were not run" % (p, wd_hit))
        except Exception as e:
            ps["error"] = "%s: %s" % (type(e).__name__, e)
            ps["traceback"] = traceback.format_exc()
            save_state(run, st)
            log(run, "%s: FAILED (%s) - batch continues" % (p, ps["error"]))
    reap(True)
    save_state(run, st)
    sampler.stop()
    if a.worker:
        log(run, "worker %s done (%s)" % (a.worker, ", ".join(plates)))
        return 0
    mkm_rate.summarize(run)
    ok = sum(1 for p in plates if st["plates"][p].get("rate") == "done")
    full = [p for p in plates if st["plates"][p].get("rate") == "done" and st["plates"][p].get("pm7m") == "done"]     # D8: rated AND print file built
    nofile = [p for p in plates if st["plates"][p].get("rate") == "done" and st["plates"][p].get("pm7m") != "done"]
    log(run, "%d of %d plates rated, %d with a print file. Summary: %s" % (ok, len(plates), len(full), os.path.join(run, "SUMMARY.md")))
    if nofile:
        log(run, "NO PRINT FILE for: %s (rated, but the file build failed or never ran - see RUN_STATE pm7m_error)" % ", ".join(nofile))
    log(run, "STOPPED - waiting for Mike. Nothing was sent to any printer.")
    return 0 if len(full) == len(plates) else 1


if __name__ == "__main__":
    sys.exit(main())







