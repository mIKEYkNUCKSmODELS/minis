# MKM auto plater driver (Claude 11/20, Mike 02:38: "The plater should decide how many plates are needed and how much space between models there should be to minimize failures").
# 1) plate count: the planner works it out from the cured-area allotment (no plate fuller, at any height, than the fullest plate that has printed clean) and the bed area; a bed is added only while that places more pieces.
# 2) gap: the largest gap between models at which that plate count still holds every piece (searched, not set).
# 3) heat maps for the result + cockpit refresh. Nothing is sliced or sent.
import os, sys, re, json, subprocess, time, shutil
D = r"C:\MKM_TMP\plate_heat_20261120\auto"; PY = r"C:\Users\confi\AI\ComfyUI_windows_portable\python_embeded\python.exe"
PL = r"C:\MKM_TMP\plater_one_build_20261117\tree\gpu_scan\mkm_gpupool10.py"; PIECES = sys.argv[1] if len(sys.argv) > 1 else r"C:\MKM_TMP\realbatch_20261117\pieces.json"
LOG = os.path.join(D, "auto.log"); NW = 0x08000000
def log(s):
    open(LOG, "a", encoding="utf-8").write(time.strftime("%H:%M:%S ") + s + "\n")
def run(n, gap):
    out = os.path.join(D, "try_N%s_g%s" % (n or "auto", gap)); shutil.rmtree(out, ignore_errors=True); os.makedirs(out)
    env = dict(os.environ, MKM_AUTO="1", GB_NOEXPORT="1", GB_VERDICT="defer", GB_FILL_RAFT="1", GB_MIN_GAP_MM=str(gap), GB_EDGE_MM=str(gap), MKM_AUTO_LOW=str(LOW), MKM_AUTO_HIGH=str(HIGH), PYTHONIOENCODING="utf-8")
    for k in ("GB_NPOOL", "MKM_FULLNESS_MM2", "GB_LEGACY_PLACEHOLDERS", "MKM_RING_LAYOUT"): env.pop(k, None)
    if n: env["GB_NPOOL"] = str(n)
    t = time.time()
    with open(os.path.join(out, "run.log"), "w", encoding="utf-8") as f:
        subprocess.run([PY, "-I", "-u", PL, PIECES, os.path.join(out, "out"), "AUTO"], cwd=r"C:\Users\confi\MKM Tools\Plate Packer", env=env, stdout=f, stderr=subprocess.STDOUT, creationflags=NW)
    s = open(os.path.join(out, "run.log"), encoding="utf-8", errors="replace").read()
    m = re.search(r"\(b\) PLACE: (\d+) pieces on (\d+) of (\d+) beds", s); c = re.search(r"carried to the next bundle: (\d+) pieces", s); nb = re.search(r"NPOOL=(\d+) beds: job needs (\d+)", s); g = re.search(r"gap ([\d.]+) mm, edge ([\d.]+) mm", s); w = re.search(r"WRITE SUMMARY: beds written (\d+)", s)
    r = dict(out=os.path.join(out, "out"), placed=int(m.group(1)) if m else -1, beds_used=int(m.group(2)) if m else -1, beds=int(nb.group(1)) if nb else -1, carried=int(c.group(1)) if c else (0 if m else -1), gap_log=g.group(1) if g else "?", edge_log=g.group(2) if g else "?", written=int(w.group(1)) if w else -1, secs=round(time.time() - t))
    log("run N=%s gap=%s -> %s" % (n or "auto", gap, json.dumps({k: v for k, v in r.items() if k != "out"}))); return r
GAPS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 8.0]; LOW0, HIGH0 = 3364.0, 4832.0; SC = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0; LOW, HIGH = LOW0 / SC, HIGH0 / SC; log("START caps %.0f / %.0f (measurement correction %.3f); bed-edge clearance = gap" % (LOW, HIGH, SC))
base = run(None, GAPS[0]); N = base["beds"]; best = base
if base["placed"] < 0: log("planner failed; see " + base["out"]); sys.exit(1)
while best["carried"] > 0:
    nxt = run(N + 1, GAPS[0])
    if nxt["placed"] > best["placed"]: N += 1; best = nxt
    else: break
log("PLATE COUNT decided: %d (places %d pieces, %d carried at the smallest gap)" % (N, best["placed"], best["carried"]))
lo, hi = 0, len(GAPS) - 1; ok = {0: best}
while lo < hi:
    mid = (lo + hi + 1) // 2; r = run(N, GAPS[mid])
    if r["placed"] >= best["placed"]: ok[mid] = r; lo = mid
    else: hi = mid - 1
fin = ok[lo]; log("GAP decided: %s mm (largest tried that still places all %d pieces on %d plates; next size up %s)" % (GAPS[lo], fin["placed"], N, ("%s mm does not fit" % GAPS[lo + 1]) if lo + 1 < len(GAPS) else "not tried"))
json.dump(dict(plates=N, gap_mm=GAPS[lo], placed=fin["placed"], carried=fin["carried"], out=fin["out"], low_cap=LOW, high_cap=HIGH, scale=SC), open(os.path.join(D, "AUTO_RESULT.json"), "w"), indent=1)
heat = os.path.join(r"C:\MKM_TMP\plate_heat_20261120", "heat_auto"); shutil.rmtree(heat, ignore_errors=True); os.makedirs(heat)
with open(os.path.join(heat, "stdout.txt"), "w") as f:
    subprocess.run([PY, "-I", "-u", r"C:\MKM_TMP\plate_heat_20261120\plate_heat.py", fin["out"], heat, "3463"], cwd=r"C:\Users\confi\MKM Tools\Plate Packer", stdout=f, stderr=subprocess.STDOUT, creationflags=NW)
subprocess.run([r"C:\Users\confi\AppData\Local\Programs\Python\Python312\python.exe", "-I", r"F:\MKM_STAGE\PLATER_COCKPIT\build_cockpit.py"], creationflags=NW)
hs = json.load(open(os.path.join(heat, "heat_summary.json"))); worst = max(max(v["fullness_mm2"] / LOW0, v["worst28_mm2"] / HIGH0) for v in hs.values())
log("CHECK fine slice: worst plate at %.3f of the clean-print limit; %d plates" % (worst, len(hs)))
json.dump(dict(worst_ratio=worst, scale_used=SC), open(os.path.join(D, "CHECK.json"), "w"))
log("AUTO DONE")
