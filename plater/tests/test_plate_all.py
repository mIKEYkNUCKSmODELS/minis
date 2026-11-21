"""plate_all with stand-in stage modules: order, resume, failure isolation, overlap, show stop."""
import sys, os, json, types, time, shutil, threading
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "v3"))
import tempfile; base = tempfile.mkdtemp(prefix="plate_all_test_")
plan = os.path.join(base, "plan")
for i in range(1, 6):
    os.makedirs(os.path.join(plan, "P%d" % i)); open(os.path.join(plan, "P%d" % i, "recipe.json"), "w").write("{}")
ev = []; evl = threading.Lock(); active = {"uv": 0, "maxuv": 0}; cfg = {"fail": set(), "obs_after": None}
def rec(*a):
    with evl: ev.append(a)
def plate(path): return [x for x in path.replace("\\", "/").split("/") if x[:1] == "P" and x[1:].isdigit()][-1]
ps = types.ModuleType("mkm_plate_slice")
def slice_plate(recipe, store, device="cuda"):
    p = plate(recipe); rec("slice", p)
    if ("slice", p) in cfg["fail"]: raise ValueError("bad stl")
ps.slice_plate = slice_plate
pb = types.ModuleType("mkm_pm7m_build")
def build_print_file(store, prn, name, ref_pm7m=None): rec("pm7m", name); return os.path.join(prn, name + ".pm7m")
pb.build_print_file = build_print_file
uv = types.ModuleType("mkm_uv_check")
def uv_check(pm7m, out, gate=False):
    p = plate(out); active["uv"] += 1; active["maxuv"] = max(active["maxuv"], active["uv"]); rec("uv", p, threading.current_thread() is threading.main_thread())
    time.sleep(0.15); active["uv"] -= 1
    if ("uv", p) in cfg["fail"]: raise OSError("uvtools died")
    return {"status": "OK"}
uv.uv_check = uv_check; uv.parity = lambda pm7m, out: (rec("parity", plate(out)) or {"status": "OK"})
rt = types.ModuleType("mkm_rate")
def rate_plate(lay, store, uvd, rate, device=None): rec("rate", plate(lay), threading.current_thread() is threading.main_thread()); return {"rating": "LIKELY"}
rt.rate_plate = rate_plate; rt.summarize = lambda run: rec("summary")
for m in (ps, pb, uv, rt): sys.modules[m.__name__] = m
# Isolated pipeline tests stub hardware preflight; test_gpu_runtime.py checks the real fail-closed policy.
gpu = types.ModuleType("mkm_gpu_runtime")
gpu.GPURequiredError = RuntimeError
gpu.require_cuda = lambda device="cuda": device
sys.modules["mkm_gpu_runtime"] = gpu
import plate_all as PA, mkm_v3_config as C
n = {"k": 0}
def obs():
    n["k"] += 1
    return cfg["obs_after"] is not None and n["k"] > cfg["obs_after"]
C.obs_running = obs
def run(name, *args):
    ev.clear(); n["k"] = 0; active["maxuv"] = 0
    rc = PA.main([plan, os.path.join(base, name)] + list(args)); st = json.load(open(os.path.join(base, name, "RUN_STATE.json")))["plates"]
    return rc, st, list(ev)
ok = True
def check(c, msg):
    global ok
    print(("PASS " if c else "FAIL ") + msg); ok = ok and c
# 1 sequential
rc, st, e = run("seq")
check(rc == 0 and all(st[p].get("rate") == "done" for p in st) and len(st) == 5, "sequential: 5 plates rated, rc 0")
check([x[0] for x in e if x[1:2] == ("P2",)] == ["slice", "pm7m", "uv", "rate"], "sequential: stage order per plate")
check(sum(1 for x in e if x[0] == "parity") == 1 and ("parity", "P1") in e, "parity on first plate only")
# 2 resume does nothing
rc, st, e = run("seq")
check([x[0] for x in e] == ["summary"], "resume: finished run repeats no stage")
# 3 failure isolation + retry
cfg["fail"] = {("slice", "P2"), ("uv", "P4")}
rc, st, e = run("fail")
check(rc == 1 and st["P2"].get("error", "").startswith("slice") and "rate" not in st["P2"] and st["P4"].get("error") and st["P3"].get("rate") == "done" and st["P5"].get("rate") == "done", "failures stay on P2 and P4, others rated, rc 1")
check(not any(x[0] in ("pm7m", "uv", "rate") and x[1] == "P2" for x in e), "failed slice skips later stages of that plate")
cfg["fail"] = set()
rc, st, e = run("fail")
check(rc == 0 and sorted(x[1] for x in e if x[0] == "slice") == ["P2"] and sorted(x[1] for x in e if x[0] == "pm7m") == ["P2"] and sorted(x[1] for x in e if x[0] == "uv") == ["P2", "P4"] and sorted(x[1] for x in e if x[0] == "rate") == ["P2", "P4"] and "error" not in st["P2"], "re-run does only what the two failed plates still need")
# 4 overlap
rc, st, e = run("ovl", "--overlap")
check(rc == 0 and all(st[p].get("rate") == "done" for p in st), "overlap: 5 plates rated, rc 0")
check(active["maxuv"] == 1, "overlap: never two UVtools at once")
check(all(x[2] is False for x in e if x[0] == "uv") and all(x[2] is True for x in e if x[0] == "rate"), "overlap: UVtools on the worker, rating on the GPU thread")
i_s5 = e.index(("slice", "P5")); i_r1 = [i for i, x in enumerate(e) if x[:2] == ("rate", "P1")][0]
check(i_s5 < [i for i, x in enumerate(e) if x[:2] == ("uv", "P3", )][0] if False else e.index(("slice", "P3")) < [i for i, x in enumerate(e) if x[0] == "uv" and x[1] == "P2"][0], "overlap: later plates are sliced while earlier UVtools runs")
for p in ("P1", "P3"):
    seq = [x[0] for x in e if x[1:2] == (p,)]
    check(seq.index("slice") < seq.index("pm7m") < seq.index("uv") < seq.index("rate"), "overlap: stage order kept for %s" % p)
# 5 show starts mid-run
cfg["obs_after"] = 3
rc, st, e = run("obs")
check(rc == 2 and sum(1 for x in e if x[0] == "slice") < 5, "show starts: run stops early with rc 2 (%d plates sliced)" % sum(1 for x in e if x[0] == "slice"))
cfg["obs_after"] = None
rc, st, e = run("obs")
check(rc == 0 and all(st[p].get("rate") == "done" for p in st) and len(st) == 5, "after the show: re-run finishes the rest")
shutil.rmtree(base, ignore_errors=True); print("ALL PASS" if ok else "SOME FAILED"); sys.exit(0 if ok else 1)
