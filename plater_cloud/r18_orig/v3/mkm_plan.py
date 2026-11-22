"""MKM plater v3 - step 0: place the pieces (4 mm gap, fewest plates). Wraps the existing planner, does not reimplement it.
    precompute(pieces_json, out_dir, deadline=None)       -> separate job: opens every STL once, fills the largest-cross-section + per-height-band caches (GPU by default, MKM_PRECOMPUTE=cpu for the old CPU path; no placement)
    plan(pieces_json, out_dir, n_start=None, n_max=None, gate=False, deadline=None) -> dict(done, n, carried, out, hard_fail, n_math, fill)   (out/ holds P*/recipe.json)
R10 changes (verifier FINAL, BLOCKERS 2 and 3, item 4):
  * the plate count comes from the MATH: the first planner run has no plate count, the planner takes it from the total footprint / one plate's usable area ("job needs N").  Only if pieces
    are left over is the count raised, one at a time (no n_start=20).  Plates that came out under the fill target are reported per plate (PLAN FILL lines); the one plate the math leaves
    short is the remainder plate.
  * a plan is accepted only when nothing is carried AND no plate carries a hard_fail (hard_checks.json of every P*).  The planner itself drops a model that fails a hard check before writing
    (mkm_layout_hard.enforce); that model counts as carried, so the plate count goes up and it is placed again.  If no count works the plan RAISES; it never returns a plan with a hard_fail.
  * deadline (epoch seconds, set by plate_all from the 30-minute run limit): the planner subprocess is killed (whole tree) when it passes, and the plan raises TimeoutError.
Resumable: out_dir/PLAN_RESULT.json is kept; a finished plan (done, nothing carried, no hard_fail) is not re-run."""
import glob, json, os, re, subprocess, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import mkm_v3_config as C

PLANNER = os.path.join(C.REF_TREE, "mkm_gpupool10_fill.py")
PLANNER_CWD = r"C:\Users\confi\MKM Tools\Plate Packer"
PLAN_VRAM_GB = round((float(os.environ.get("MKM_VRAM_NEED_MIB", 3500)) + 512.0) / 1024.0 + 0.05, 1)      # mkm_gpuport's own fit check: need MKM_VRAM_NEED_MIB (3500) + 512 MiB margin must be free
XSEC_CACHE = os.environ.get("MKM_XSEC_CACHE", r"F:\MKM_STAGE\PLATER_XSEC_CACHE")      # footprint + band cache lives on F: (scratch), shared by precompute and plan


def _case_input(pieces, out_dir):
    """MIKE 2202 (ChatGPT R14 patch, merged): real composite case STLs must be certified before *any* layout.
    The inventory is used by BOTH precompute and plan, so xsec/bands, 4-mm placement, raft, slice/scan/rating and accounting all reference one mesh."""
    import mkm_case_raft
    return mkm_case_raft.prepare_pieces(
        pieces, out_dir, mode=os.environ.get("MKM_CASE_MODE", "on").lower(),
        manifest=os.environ.get("MKM_CASE_MANIFEST"))


def _env(n):
    env = dict(os.environ, GB_NOEXPORT="1", GB_VERDICT="defer", GB_FILL_RAFT="1", GB_MIN_GAP_MM=str(int(C.GAP_MM)),
               PYTHONIOENCODING="utf-8", MKM_RAM_FLOOR_GIB="2.5", MKM_EDGE_MM=str(C.EDGE_MARGIN_MM), MKM_FILL_MIN=os.environ.get("MKM_FILL_MIN", str(C.FILL_MIN)), MKM_LAYOUT_SCORE="1", MKM_XSEC="1", MKM_LAYER_MM=str(C.LAYER_MM), MKM_XSEC_CACHE=XSEC_CACHE)   # R7: scored items 4-7 arrange the plates.  R6: the 2 mm margin (Mike 11/21 14:18) is planned here.  R10: per-height-band collision is on inside the planner (MKM_XSEC=1)
    if n:
        env["GB_NPOOL"] = str(n)
    else:
        env.pop("GB_NPOOL", None)      # no plate count: the planner takes it from the total footprint (the math)
    for k in ("MKM_AUTO", "MKM_FULLNESS_MM2", "GB_LEGACY_PLACEHOLDERS", "MKM_RING_LAYOUT", "MKM_BANDS", "MKM_BAND_PLACE", "MKM_REMAINDER_PLATE"):
        env.pop(k, None)
    return env


def _run(argv, env, log, deadline):
    """run hidden, write the log, enforce the deadline by killing the whole tree.  -> returncode"""
    os.makedirs(os.path.dirname(log), exist_ok=True)
    with open(log, "w", encoding="utf-8") as f:
        p = subprocess.Popen(argv, cwd=PLANNER_CWD, env=env, stdout=f, stderr=subprocess.STDOUT, creationflags=C.NO_WINDOW)
        try:
            p.wait(timeout=None if deadline is None else max(1.0, deadline - time.time()))
        except subprocess.TimeoutExpired:
            subprocess.run(["taskkill", "/PID", str(p.pid), "/T", "/F"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=C.NO_WINDOW)
            p.wait()
            raise TimeoutError("planner stopped at the run limit (%s)" % log)
    return p.returncode


def precompute(pieces, out_dir, deadline=None, gate=False):
    """separate CPU job (Mike 14:00: nothing slow inside the planning step): every model's largest cross-section and height-band silhouettes are measured once and cached on F:."""
    pieces = _case_input(pieces, out_dir)
    out = os.path.join(os.path.abspath(out_dir), "precompute"); os.makedirs(out, exist_ok=True)
    gpu_pre = os.environ.get("MKM_PRECOMPUTE", "cuda").lower() != "cpu"      # Mike 11/22 01:39: step 0a runs on the GPU (mkm_xsec_footprint._measure_gpu); MKM_PRECOMPUTE=cpu keeps the old CPU path
    env = _env(None); env.update(MKM_XSEC_PRECOMPUTE="1", MKM_DEV=("cuda" if gpu_pre else "cpu"), MKM_PRECOMPUTE=("cuda" if gpu_pre else "cpu"), GB_NPOOL="1")
    argv = [C.PY, "-I", "-u", PLANNER, pieces, os.path.join(out, "out"), "FILL"]
    if gate:
        argv = C.gate_cmd("PLAN_PRECOMPUTE", 3, (4 if gpu_pre else 0), argv, lane="plater")    # CPU job, but 369 STLs are read: it takes its turn in the gate like every heavy job
    rc = _run(argv, env, os.path.join(out, "run.log"), deadline)
    s = open(os.path.join(out, "run.log"), encoding="utf-8", errors="replace").read()
    m = re.search(r"PRECOMPUTE DONE: (\d+) models.*", s)
    if not m:
        raise RuntimeError("precompute did not finish (rc=%s); see %s" % (rc, os.path.join(out, "run.log")))
    return dict(done=True, models=int(m.group(1)), line=m.group(0), cache=XSEC_CACHE)


def hard_fails_in(planner_out):
    """every plate's hard_checks.json under a planner output folder (P*/hard_checks.json) that is not ok -> [(plate, reasons)].  Also used by plate_all."""
    bad = []
    plates = sorted(set(os.path.dirname(f) for f in glob.glob(os.path.join(planner_out, "P*", "hard_checks.json")) + glob.glob(os.path.join(planner_out, "P*", "recipe.json"))))
    for d in plates:
        f = os.path.join(d, "hard_checks.json"); p = os.path.basename(d); why = []
        if not os.path.exists(f):      # R11 (verifier R8 MED-A): a plate with no hard-check record has not been checked, so it is not clean
            bad.append((p, ["hard_checks.json is missing: the plate was never hard-checked"])); continue
        try:
            r = json.load(open(f, encoding="utf-8"))
        except Exception as ex:
            bad.append((p, ["hard_checks.json unreadable: %r" % ex])); continue
        if not r.get("ok", False):
            why += r.get("reasons") or ["ok is false"]
        rp = os.path.join(d, "recipe.json")      # R11 (HIGH-3): B4, at least 2 models on EVERY plate, also when no check flagged it; `sources: []` is not clean
        try:
            srcs = json.load(open(rp, encoding="utf-8-sig")).get("sources") or []
            ns = len({s["unit"] if s.get("unit") else ("_src", i) for i, s in enumerate(srcs)})      # R12 (verifier R9 H1): B4 counts MODELS: sources sharing a unit (a figure's body + its own base) are ONE model
        except Exception:
            ns = None
        if ns is None:
            why.append("recipe.json missing or unreadable")
        elif ns < 2:
            why.append("plate holds %d model(s): every plate needs at least 2 (B4; a figure's body + base files are one model)" % ns)
        if why:
            bad.append((p, why))
    return bad


def _one(pieces, out_dir, n, gate, deadline=None):
    out = os.path.join(out_dir, "N%s" % (n if n else "math")); os.makedirs(out, exist_ok=True)
    argv = [C.PY, "-I", "-u", PLANNER, pieces, os.path.join(out, "out"), "FILL"]
    if gate:
        argv = C.gate_cmd("PLAN_N%s" % (n or "math"), 3, PLAN_VRAM_GB, argv, lane="plater")     # R11: the planner's GPU slicer-free layout still loads mkm_gpuport: ask its real need (MKM_VRAM_NEED_MIB + 512 MiB margin), not 0
    _run(argv, _env(n), os.path.join(out, "run.log"), deadline)
    s = open(os.path.join(out, "run.log"), encoding="utf-8", errors="replace").read()
    c = re.search(r"carried to the next bundle: (\d+) pieces", s)
    done = "BUNDLEDONE" in s
    lc = re.findall(r"LONE CARRY: (\d+) pieces", s)      # R12 (verifier R9 M1): a lone model no plate could take is carried and reported, it never makes a one-model plate and never refuses the job
    hf = re.findall(r"LAYOUT HARD CHECKS bed \d+: FAIL.*", s) + ["%s: %s" % (p, "; ".join(r)) for p, r in hard_fails_in(os.path.join(out, "out"))]
    fc = re.findall(r"FILL CARRY: bed \d+ .*? its (\d+) pieces", s)      # R14 (Mike 22:00): pieces of a plate under the 68% real-footprint minimum are carried by name, a rule outcome like the lone carry, not a failed plan
    fill2 = re.findall(r"PLAN FILL2 bed (\d+): real footprint ([\d.]+)% .*?minimum ([\d.]+)%: (.*)", s)
    cr = []      # R17 (driver 0045 item 1, verifier R16 blocker 1): the planner names every carried piece with its reason; the plan is judged on those NAMES, never on two counts that happen to match
    for ln in re.findall(r"^CARRY REASON (\{.*\})\s*$", s, re.M):
        try: cr.append(json.loads(ln))
        except Exception: pass
    jm = re.search(r"NPOOL=(\d+) beds: job needs (\d+)", s)
    fill = re.findall(r"PLAN FILL bed (\d+): use ([\d.]+) target ([\d.]+) (\S+)", s)
    drops = re.findall(r"HARD DROP bed .*", s)
    return dict(pieces=os.path.abspath(pieces), n=int(jm.group(1)) if jm else n, n_math=int(jm.group(2)) if (jm and not n) else None, done=done, carried=int(c.group(1)) if c else (0 if done else -1), lone_carried=sum(int(x) for x in lc), fill_carried=sum(int(x) for x in fc), fill2=[dict(bed=int(a), fp_pct=float(b), min_pct=float(c), why=d.strip()) for a, b, c, d in fill2], out=os.path.join(out, "out"), hard_fail=hf, carry_reasons=cr,
                hard_dropped=drops, fill=[dict(bed=int(a), use=float(u), target=float(t), status=st) for a, u, t, st in fill])


def accepted(r):
    """a plan is accepted when the planner finished, no plate carries a hard_fail, and the only pieces left unplaced are lone models carried by the lone-plate rule (R12)."""
    if not r["done"] or r["hard_fail"]:
        return False
    cr = r.get("carry_reasons")
    if r["carried"] == 0 and not cr:      # nothing carried: there is no name to account for
        return True
    if cr is None:      # carried pieces but no names (an old PLAN_RESULT.json from before R17): not trusted, it is re-planned (the old count rule is the thing that rejected correct plans)
        return False
    # by NAME: every carried piece has a reason line, the reasons are all rule outcomes (lone model / fill minimum), and the named set is as large as the planner's own carried count
    return len(cr) == r["carried"] and all(x.get("r") in ("lone", "fill") and x.get("file") for x in cr)


def plan(pieces, out_dir, n_start=None, n_max=None, gate=False, deadline=None):
    pieces = _case_input(pieces, out_dir)
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    rp = os.path.join(out_dir, "PLAN_RESULT.json")
    if os.path.exists(rp):
        r = json.load(open(rp, encoding="utf-8"))
        if (accepted(r) and r.get("out") and not hard_fails_in(r["out"])
                and os.path.normcase(os.path.abspath(r.get("pieces", ""))) == os.path.normcase(os.path.abspath(pieces))):
            return r
    r = _one(pieces, out_dir, n_start, gate, deadline)      # n_start None = the plate count is taken from the math by the planner
    n_math = r.get("n_math") or r["n"] or 1
    tries = [r]
    json.dump(r, open(rp, "w", encoding="utf-8"), indent=1)
    n = max(n_math, n_start or 0); n_cap = n_max or (n + max(4, -(-n // 4)))      # a bounded search upward from the math, never a fixed 20
    while not accepted(r):
        n += 1
        if n > n_cap:
            raise RuntimeError("planner could not place every piece without a hard-check failure up to %d plates (math said %d); last run: carried %s, hard_fail %s; see %s" % (
                n_cap, n_math, r["carried"], r["hard_fail"][:2], out_dir))
        r = _one(pieces, out_dir, n, gate, deadline); r["n_math"] = n_math; tries.append(r)
        json.dump(r, open(rp, "w", encoding="utf-8"), indent=1)
    r["n_math"] = n_math; r["tries"] = [(t["n"], t["carried"], len(t["hard_fail"])) for t in tries]
    json.dump(r, open(rp, "w", encoding="utf-8"), indent=1)
    return r


if __name__ == "__main__":
    if "--precompute" in sys.argv:
        print(json.dumps(precompute(sys.argv[1], sys.argv[2]), indent=1))
    else:
        print(json.dumps(plan(sys.argv[1], sys.argv[2], gate="--gate" in sys.argv), indent=1))
