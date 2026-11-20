#!/usr/bin/env python3
"""End-to-end CPU smoke run of plater/v3/plate_all.py on a synthetic fixture with a fake UVtools."""
import json
import math
import os
import shutil
import subprocess
import sys
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
WORK = os.path.join(HERE, "_work")
FIX = os.path.join(WORK, "fixture")
RUN = os.path.join(WORK, "run1")
LOG1 = os.path.join(WORK, "run1.log")
LOG2 = os.path.join(WORK, "run2.log")
PLATES = ["P1", "P2", "P3"]
LABELS = ("LIKELY", "LIKELY WITH NOTES", "RISKY", "WILL NOT PRINT")
STEPS = ("slice", "pm7m", "uv", "rate")
STOP = "STOPPED - waiting for Mike. Nothing was sent to any printer."

results = []


def check(name, ok, detail=""):
    results.append((name, bool(ok), detail))
    print("%s  %s%s" % ("PASS" if ok else "FAIL", name, ("  [%s]" % detail) if (detail and not ok) else ""))
    return ok


def jload(p):
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def env():
    e = dict(os.environ)
    e["MKM_UVTOOLS"] = os.path.join(HERE, "fake_uvtools.py")
    e["MKM_GPU_SCAN"] = os.path.join(ROOT, "plater", "ref", "gpu_scan")
    e["MKM_PHYS_DIR"] = os.path.join(ROOT, "plater", "ref", "physics")
    # ref/physics looks for profiles_filled/, which is not in the tree; the param file sits beside the module
    e["GB_PARAMS"] = os.path.join(ROOT, "plater", "ref", "physics", "planner_params_m7max_acf_sunlu.json")
    stubs = os.path.join(HERE, "stubs")
    e["PYTHONPATH"] = os.pathsep.join([stubs] + ([e["PYTHONPATH"]] if e.get("PYTHONPATH") else []))
    return e


def run_plate_all(log_path):
    cmd = [sys.executable, os.path.join(ROOT, "plater", "v3", "plate_all.py"), FIX, RUN, "--device", "cpu"]
    with open(log_path, "w") as lf:
        try:
            rc = subprocess.run(cmd, cwd=ROOT, env=env(), stdout=lf, stderr=subprocess.STDOUT, timeout=40 * 60).returncode
        except subprocess.TimeoutExpired:
            lf.write("\nTIMEOUT after 40 min\n")
            rc = -1
    return rc


def tail(path, n):
    with open(path, errors="replace") as f:
        return f.read().splitlines()[-n:]


def main():
    os.makedirs(WORK, exist_ok=True)
    os.makedirs(os.path.join(HERE, "stubs"), exist_ok=True)
    for d in (FIX, RUN):
        shutil.rmtree(d, ignore_errors=True)
    subprocess.run([sys.executable, os.path.join(HERE, "make_fixture.py"), FIX], check=True, cwd=ROOT)

    rc = run_plate_all(LOG1)
    print("--- last 30 lines of run1.log (exit %s) ---" % rc)
    print("\n".join(tail(LOG1, 30)))
    print("---")

    st = jload(os.path.join(RUN, "RUN_STATE.json")) or {}
    plates = st.get("plates", {})
    for p in PLATES:
        ps = plates.get(p) or {}
        check("%s: steps done" % p, all(ps.get(s) == "done" for s in STEPS), json.dumps({s: ps.get(s) for s in STEPS}))
        check("%s: no error" % p, "error" not in ps, ps.get("error", ""))
        if "error" in ps:
            print("TRACEBACK %s:\n%s" % (p, ps.get("traceback", "(none)")))
        man = jload(os.path.join(FIX, p, "RING_MANIFEST.json")) or [{}]
        h = man[0].get("height", 0)
        n_exp = math.ceil(round(h / 0.03, 6))
        meta = jload(os.path.join(RUN, "store", p, "meta.json")) or {}
        check("%s: store n_layers == %d" % (p, n_exp), meta.get("n_layers") == n_exp, "got %s" % meta.get("n_layers"))
        pm = os.path.join(RUN, "print", p, "%s.pm7m" % p)
        nz = -1
        if os.path.exists(pm):
            try:
                with zipfile.ZipFile(pm) as z:
                    nz = sum(1 for n in z.namelist() if n.endswith(".pw0Img"))
            except Exception as e:
                nz = -2
        check("%s: pm7m exists with %d pw0Img" % (p, n_exp), nz == n_exp, "found %s" % nz)
        for f in ("issues_raw.json", "issues_repaired.json"):
            check("%s: uv/%s exists" % (p, f), os.path.exists(os.path.join(RUN, "uv", p, f)))
        rt = jload(os.path.join(RUN, "rate", p, "rating.json")) or {}
        check("%s: rating label valid" % p, rt.get("rating") in LABELS, "got %r" % rt.get("rating"))
        if p == "P2":
            check("P2: rating != LIKELY (floating cube caught)", rt.get("rating") != "LIKELY", "got %r" % rt.get("rating"))
        check("%s: heat png exists" % p, os.path.exists(os.path.join(RUN, "rate", p, "%s_heat.png" % p)))
    check("SUMMARY.md exists", os.path.exists(os.path.join(RUN, "SUMMARY.md")))
    check("SUMMARY.png exists", os.path.exists(os.path.join(RUN, "SUMMARY.png")))
    lines = [l for l in open(LOG1, errors="replace").read().splitlines() if l.strip()]
    check("log ends with STOPPED line", bool(lines) and STOP in lines[-1], lines[-1] if lines else "empty log")

    # second invocation: resume must skip finished work
    rc2 = run_plate_all(LOG2)
    log2 = open(LOG2, errors="replace").read()
    check("run2: no 'slice once' in log", "slice once" not in log2, "exit %s" % rc2)
    after = (jload(os.path.join(RUN, "RUN_STATE.json")) or {}).get("plates", {})
    same = all((plates.get(p) or {}).get(s) == (after.get(p) or {}).get(s) for p in PLATES for s in STEPS)
    check("run2: RUN_STATE done steps unchanged", same)

    bad = [n for n, ok, _ in results if not ok]
    if bad:
        print("SMOKE FAIL")
        for n in bad:
            print("  failed:", n)
        return 1
    print("SMOKE PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
