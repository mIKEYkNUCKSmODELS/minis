# Per-plate spread (Claude 11/20 05:30, Mike: "why do I still have a big ass gap in the middle"). Each plate keeps its own pieces; the gap between models is widened per plate
# to the largest value at which all of that plate's pieces still fit one bed, so spare room becomes even spacing instead of an empty centre. Read-only on the bundle result.
import os, sys, re, json, subprocess, time, shutil, glob
from concurrent.futures import ThreadPoolExecutor
D = r"C:\MKM_TMP\plate_heat_20261120\auto"; S = os.path.join(D, "spread"); PY = r"C:\Users\confi\AI\ComfyUI_windows_portable\python_embeded\python.exe"
PL = r"C:\MKM_TMP\plater_one_build_20261117\tree\gpu_scan\mkm_gpupool10.py"; NW = 0x08000000; LOG = os.path.join(S, "spread.log")
R = json.load(open(os.path.join(D, "AUTO_RESULT.json"))); BASE = float(R["gap_mm"]); rows = json.load(open(r"C:\MKM_TMP\realbatch_20261117\pieces.json", encoding="utf-8"))
os.makedirs(S, exist_ok=True)
def log(s):
    open(LOG, "a", encoding="utf-8").write(time.strftime("%H:%M:%S ") + s + "\n")
GAPS = [g for g in [BASE, 5, 6, 7, 8, 10, 12, 14, 16, 19, 22, 26, 30] if g >= BASE]
def run(pid, sub, gap):
    out = os.path.join(S, "try_%s_g%s" % (pid, gap)); shutil.rmtree(out, ignore_errors=True); os.makedirs(out); pj = os.path.join(out, "pieces.json"); json.dump(sub, open(pj, "w", encoding="utf-8"))
    env = dict(os.environ, MKM_AUTO="1", GB_NOEXPORT="1", GB_VERDICT="defer", GB_FILL_RAFT="1", GB_MIN_GAP_MM=str(gap), GB_EDGE_MM=str(BASE), GB_NPOOL="1", MKM_AUTO_LOW=str(float(R["low_cap"]) * 1.0005), MKM_AUTO_HIGH=str(float(R["high_cap"]) * 1.0005), PYTHONIOENCODING="utf-8")
    for k in ("MKM_FULLNESS_MM2", "GB_LEGACY_PLACEHOLDERS", "MKM_RING_LAYOUT"): env.pop(k, None)
    with open(os.path.join(out, "run.log"), "w", encoding="utf-8") as f:
        subprocess.run([PY, "-I", "-u", PL, pj, os.path.join(out, "out"), "AUTO"], cwd=r"C:\Users\confi\MKM Tools\Plate Packer", env=env, stdout=f, stderr=subprocess.STDOUT, creationflags=NW)
    s = open(os.path.join(out, "run.log"), encoding="utf-8", errors="replace").read()
    m = re.search(r"\(b\) PLACE: (\d+) pieces on", s); e = re.search(r"empty radius at centre ([\d.]+) mm", s); w = re.search(r"WRITE SUMMARY: beds written (\d+)", s)
    return dict(dir=out, placed=int(m.group(1)) if m else -1, empty=float(e.group(1)) if e else -1.0, ok=bool(m and int(m.group(1)) == len(sub) and w and int(w.group(1)) == 1))
def plate(pd):
    pid = os.path.basename(pd); man = json.load(open(os.path.join(pd, "RING_MANIFEST.json"), encoding="utf-8"))[0]; pool = list(rows); sub = []
    for p in man["pieces"]:
        i = next((k for k, r in enumerate(pool) if os.path.normcase(r["path"]) == os.path.normcase(p["path"])), None)
        if i is None: log("%s: piece not found in pieces.json: %s" % (pid, p["path"])); return
        sub.append(pool.pop(i))
    old = re.search(r"empty radius at centre ([\d.]+) mm", open(os.path.join(os.path.dirname(os.path.dirname(pd)), "run.log"), encoding="utf-8", errors="replace").read().split("\n" + pid + ":")[1]) if True else None
    lo, hi, good = 0, len(GAPS) - 1, {}
    r0 = run(pid, sub, GAPS[0])
    if not r0["ok"]: log("%s: does not re-place alone at %s mm (placed %d of %d); bundle layout kept" % (pid, GAPS[0], r0["placed"], len(sub))); shutil.copytree(pd, os.path.join(S, "out", pid), dirs_exist_ok=True); return
    good[0] = r0
    while lo < hi:
        mid = (lo + hi + 1) // 2; r = run(pid, sub, GAPS[mid])
        if r["ok"]: good[mid] = r; lo = mid
        else: hi = mid - 1
    fin = good[lo]; dst = os.path.join(S, "out", pid); shutil.rmtree(dst, ignore_errors=True); shutil.copytree(os.path.join(fin["dir"], "out", "P1"), dst)
    json.dump(dict(gap_mm=GAPS[lo], empty_centre_mm=fin["empty"], empty_centre_before_mm=float(old.group(1)) if old else None), open(os.path.join(dst, "SPREAD.json"), "w"))
    log("%s: gap %s mm, empty radius at centre %s -> %s mm, %d pieces" % (pid, GAPS[lo], old.group(1) if old else "?", fin["empty"], len(sub)))
    for t in glob.glob(os.path.join(S, "try_%s_g*" % pid)): shutil.rmtree(t, ignore_errors=True)
pds = sorted(glob.glob(os.path.join(R["out"], "P*")), key=lambda p: int(os.path.basename(p)[1:]))
def safe(pd):
    try: plate(pd)
    except Exception as ex:
        import traceback; log("%s ERROR %r %s" % (os.path.basename(pd), ex, traceback.format_exc()[-600:]))
with ThreadPoolExecutor(2) as ex: list(ex.map(safe, pds))
log("SPREAD DONE")
