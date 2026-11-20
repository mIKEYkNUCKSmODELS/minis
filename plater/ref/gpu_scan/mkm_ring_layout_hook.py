# mkm_ring_layout_hook.py  (new 2026-11-19 22:15, Claude coding agent, order of Mike 22:06 CT)
# Pure-geometry layout rule for the plater. Only runs when MKM_RING_LAYOUT=1 (called from mkm_gpupool10.py right after the "(b) PLACE" print).
# Rearranges footprints on each bed by legal moves only (planner's own legality test: 3 mm gap via max_pool dilation g, inside bed via K["inb"]),
# to minimise a score that uses ONLY the layout (convex hull, ring depth, area centroid, 8 depth bands, height tiers). No force/tension/kPa numbers.
import os, sys, time, json, math
import numpy as np
from scipy.spatial import ConvexHull

# ------------------------------------------------------------------ pure scoring (numpy/scipy only; also used by the renderer)
def pool_map(mask, px, py, s):
    sw, sh = mask.shape
    a = np.zeros((sw + px, sh + py), np.float32); a[px:, py:] = mask
    pw = (-a.shape[0]) % s; ph = (-a.shape[1]) % s
    if pw or ph: a = np.pad(a, ((0, pw), (0, ph)))
    return a.reshape(a.shape[0] // s, s, a.shape[1] // s, s).mean((1, 3))

def hull_depth(A, c):
    """A: footprint fraction map on a c-mm grid. Returns hull/depth geometry or None."""
    occ = A > 0
    rows_ = occ.any(1)
    if not rows_.any(): return None
    xs = np.nonzero(rows_)[0]
    ymin = np.argmax(occ[xs], axis=1); ymax = occ.shape[1] - 1 - np.argmax(occ[xs][:, ::-1], axis=1)
    pts = np.concatenate([np.stack([xs, ymin], 1), np.stack([xs + 1, ymin], 1), np.stack([xs, ymax + 1], 1), np.stack([xs + 1, ymax + 1], 1)]).astype(float) * c
    try: hu = ConvexHull(pts)
    except Exception: return None
    V = pts[hu.vertices]; eq = hu.equations
    x0 = int(np.floor(V[:, 0].min() / c)); x1 = int(np.ceil(V[:, 0].max() / c)); y0 = int(np.floor(V[:, 1].min() / c)); y1 = int(np.ceil(V[:, 1].max() / c))
    x0 = max(x0, 0); y0 = max(y0, 0); x1 = min(x1, A.shape[0]); y1 = min(y1, A.shape[1])
    X, Y = np.meshgrid((np.arange(x0, x1) + 0.5) * c, (np.arange(y0, y1) + 0.5) * c, indexing="ij")
    d = -(eq[:, 0, None, None] * X[None] + eq[:, 1, None, None] * Y[None] + eq[:, 2, None, None]).max(0)   # distance to the outside of the hull (inside positive)
    inside = d >= 0
    # polygon centroid
    x = V[:, 0]; y = V[:, 1]; x2 = np.roll(x, -1); y2 = np.roll(y, -1); cr = x * y2 - x2 * y; ar = cr.sum() / 2
    hc = ((x + x2) * cr).sum() / (6 * ar), ((y + y2) * cr).sum() / (6 * ar)
    half_diag = 0.5 * math.hypot(np.ptp(V[:, 0]), np.ptp(V[:, 1]))
    D = float(d[inside].max()) if inside.any() else 0.0
    me = inside & (d >= D - 0.5 * c)
    end = (float(X[me].mean()), float(Y[me].mean()))
    return dict(V=V, hc=hc, half_diag=half_diag, X=X, Y=Y, d=d, inside=inside, D=D, end=end, sl=(slice(x0, x1), slice(y0, y1)))

def _a(M, Hh):
    s = M.sum()
    if s <= 0: return 0.0
    cx = (M * Hh["X"]).sum() / s; cy = (M * Hh["Y"]).sum() / s
    return float(math.hypot(cx - Hh["hc"][0], cy - Hh["hc"][1]) / Hh["half_diag"])

def _b(M, Hh):
    D = Hh["D"]
    if D <= 0: return 0.0
    ins = Hh["inside"]; band = np.minimum((Hh["d"][ins] / D * 8).astype(int), 7)
    cnt = np.bincount(band, minlength=8).astype(float); fl = np.bincount(band, weights=M[ins], minlength=8) / np.maximum(cnt, 1)
    m = fl.mean()
    return float(fl.std() / m) if m > 0 else 0.0

def score_maps(A, A1, A2, c, gap_mm):
    """A = all footprints; A1/A2 = only pieces taller than 1/3 / 2/3 of the tallest. Lexicographic (c, sum of a+b over 3 tiers)."""
    Hh = hull_depth(A, c)
    if Hh is None: return None
    sl = Hh["sl"]; out = {}
    tot = 0.0
    for nm, M in (("0", A), ("1", A1), ("2", A2)):
        Mb = M[sl]; a = _a(Mb, Hh); b = _b(Mb, Hh); out["a" + nm] = a; out["b" + nm] = b; tot += a + b
    # open end: distance from the end point to the nearest footprint cell square
    xs, ys = np.nonzero(A > 0)
    cxm = (xs + 0.5) * c; cym = (ys + 0.5) * c
    dx = np.maximum(np.abs(cxm - Hh["end"][0]) - c / 2, 0); dy = np.maximum(np.abs(cym - Hh["end"][1]) - c / 2, 0)
    dist = float(np.sqrt(dx * dx + dy * dy).min())
    out.update(c=int(dist < gap_mm), end_dist_mm=dist, total=tot, D=Hh["D"], end=Hh["end"], key=(int(dist < gap_mm), tot))
    return out

def maps_from_lab(lab, hmap, s):
    """lab (W,H) int label raster (0 empty) -> A, A1, A2 on the s-pixel grid. hmap: dict label -> height mm."""
    hmax = max(hmap.values())
    def pool(m):
        a = m.astype(np.float32); pw = (-a.shape[0]) % s; ph = (-a.shape[1]) % s
        if pw or ph: a = np.pad(a, ((0, pw), (0, ph)))
        return a.reshape(a.shape[0] // s, s, a.shape[1] // s, s).mean((1, 3))
    A = pool(lab > 0)
    t1 = [l for l, h in hmap.items() if h > hmax / 3]; t2 = [l for l, h in hmap.items() if h > 2 * hmax / 3]
    return A, pool(np.isin(lab, t1)), pool(np.isin(lab, t2))

# ------------------------------------------------------------------ planner-side driver
def _free_ram_gb():
    import ctypes
    class MS(ctypes.Structure):
        _fields_ = [("l", ctypes.c_ulong), ("load", ctypes.c_ulong), ("tp", ctypes.c_ulonglong), ("ap", ctypes.c_ulonglong), ("tpf", ctypes.c_ulonglong), ("apf", ctypes.c_ulonglong), ("tv", ctypes.c_ulonglong), ("av", ctypes.c_ulonglong), ("ae", ctypes.c_ulonglong)]
    m = MS(); m.l = ctypes.sizeof(MS); ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)); return m.ap / 2 ** 30

def apply(G):
    if os.environ.get("MKM_RING_PROBE") == "1":
        pl = G["placements"]; use = G["use"]; n = sum(len(x) for x in pl)
        print(f"RINGPROBE: g={G['g']} R={G['R']} NPOOL={G['NPOOL']} placed {n}/{len(G['rows'])} failed_models {len(G['failed'])} CARRY {G['CARRY']} MAXSPLIT {G['MAXSPLIT']}", flush=True)
        print("RINGPROBE: use per bed " + " ".join(f"{b + 1}:{len(pl[b])}pcs/{use(b):.2f}" for b in range(G['NPOOL']) if pl[b]), flush=True); return
    if os.environ.get("MKM_RING_ROUND") == "2":
        import mkm_ring_layout_hook2 as H2; return H2.apply(G)
    if os.environ.get("MKM_RING_ROUND") == "3":
        import mkm_ring_layout_hook3 as H3; return H3.apply(G)
    return apply1(G)

def apply1(G):
    import torch, torch.nn.functional as F
    T0 = time.time()
    OUT = os.environ.get("MKM_RING_OUT", r"C:\MKM_TMP\ring_plates_20261119"); DUMP = os.path.join(OUT, "ring_dump"); os.makedirs(DUMP, exist_ok=True)
    T_CAP = float(os.environ.get("MKM_RING_SECS", 60)); MOVE_CAP = int(os.environ.get("MKM_RING_MOVES", 200))
    placements = G["placements"]; rows = G["rows"]; NPOOL = G["NPOOL"]; W = G["W"]; H = G["H"]; R = G["R"]; PX = G["PX"]; PY = G["PY"]
    g = int(G["g"]); dev = G["dev"]; get_K = G["get_K"]; commit = G["commit"]; uncommit = G["uncommit"]; circle = G["circle"]
    use = G["use"]; CARRY = G["CARRY"]; CS = G.get("CENTRE_SMALL", False); tier = G.get("tier")
    s = max(1, int(round(1.0 / R))); c_mm = s * R; GAP_MM = g * R
    ram0 = _free_ram_gb(); ram_min = ram0
    print(f"RING: start. free RAM {ram0:.1f} GB, grid {W}x{H} @ {R} mm, search/score grid {c_mm} mm, gap {GAP_MM} mm (g={g}), caps {T_CAP}s / {MOVE_CAP} moves per bed", flush=True)
    P = {}; PM = {}
    def prep(i):
        if i not in P:
            P[i] = [dict(f=K["f"].clone(), inb=K["inb"], nz=K["nz"], sw=int(K["sw"]), sh=int(K["sh"])) for K in get_K(i)]
        return P[i]
    def occ_of(state, skip):
        o = torch.zeros((W, H), device=dev)
        for (i, k, ox, oy) in state:
            if i in skip: continue
            nz = prep(i)[k]["nz"]; o[ox + nz[:, 0], oy + nz[:, 1]] = 1.0
        return o
    def put(o, i, k, ox, oy):
        nz = prep(i)[k]["nz"]; o = o.clone(); o[ox + nz[:, 0], oy + nz[:, 1]] = 1.0; return o
    def legal_maps(o, i):
        bl = F.max_pool2d(o[None, None], 2 * g + 1, 1, g)[0, 0] + (0.0 if (CS and tier is not None and tier[i] == 2) else circle)
        bf = torch.fft.rfft2(bl)
        return [(torch.fft.irfft2(bf * c["f"], s=(W, H)) < 0.5) & c["inb"] for c in prep(i)]
    def nearest(L, tx, ty):
        idx = torch.nonzero(L)
        if idx.shape[0] == 0: return None
        j = torch.argmin((idx[:, 0] - tx) ** 2 + (idx[:, 1] - ty) ** 2); return int(idx[j, 0]), int(idx[j, 1])
    def pooled(i, k, ox, oy):
        key = (i, k, ox % s, oy % s)
        if key not in PM: PM[key] = pool_map(rows[i]["D"]["rots"][k]["mask"], ox % s, oy % s, s)
        return PM[key]
    GW = -(-W // s); GH = -(-H // s)
    def maps(state):
        hs = [rows[i]["h"] for (i, k, ox, oy) in state]; hmax = max(hs)
        A = np.zeros((GW, GH), np.float32); A1 = A.copy(); A2 = A.copy()
        for (i, k, ox, oy), h in zip(state, hs):
            p = pooled(i, k, ox, oy); x0 = ox // s; y0 = oy // s; sl = (slice(x0, x0 + p.shape[0]), slice(y0, y0 + p.shape[1])); ps = p[:GW - x0, :GH - y0]
            A[x0:x0 + ps.shape[0], y0:y0 + ps.shape[1]] += ps
            if h > hmax / 3: A1[x0:x0 + ps.shape[0], y0:y0 + ps.shape[1]] += ps
            if h > 2 * hmax / 3: A2[x0:x0 + ps.shape[0], y0:y0 + ps.shape[1]] += ps
        return A, A1, A2
    def ev(state):
        sc = score_maps(*maps(state), c_mm, GAP_MM)
        return sc
    anchors = [(W * fx, H * fy) for fx in (1 / 8, 3 / 8, 5 / 8, 7 / 8) for fy in (1 / 4, 3 / 4)]
    def repl(state, j, new):
        st = list(state); st[j] = new; return st
    def candidates(state):
        out = []; n = len(state)
        for j, (i, k, ox, oy) in enumerate(state):
            cs = prep(i); L = legal_maps(occ_of(state, {i}), i); cxc = ox + cs[k]["sw"] / 2; cyc = oy + cs[k]["sh"] / 2
            for k2 in range(4):
                if k2 == k: continue
                r = nearest(L[k2], cxc - cs[k2]["sw"] / 2, cyc - cs[k2]["sh"] / 2)
                if r: out.append(repl(state, j, (i, k2, r[0], r[1])))
            for ax, ay in anchors:
                r = nearest(L[k], ax - cs[k]["sw"] / 2, ay - cs[k]["sh"] / 2)
                if r and r != (ox, oy): out.append(repl(state, j, (i, k, r[0], r[1])))
        for j in range(n):
            for l in range(j + 1, n):
                iA, kA, oxA, oyA = state[j]; iB, kB, oxB, oyB = state[l]
                csA = prep(iA); csB = prep(iB); o = occ_of(state, {iA, iB})
                cA = (oxA + csA[kA]["sw"] / 2, oyA + csA[kA]["sh"] / 2); cB = (oxB + csB[kB]["sw"] / 2, oyB + csB[kB]["sh"] / 2)
                LA = legal_maps(o, iA); newA = None
                for k2 in [kA] + [q for q in range(4) if q != kA]:
                    r = nearest(LA[k2], cB[0] - csA[k2]["sw"] / 2, cB[1] - csA[k2]["sh"] / 2)
                    if r: newA = (iA, k2, r[0], r[1]); break
                if newA is None: continue
                LB = legal_maps(put(o, *newA), iB); newB = None
                for k2 in [kB] + [q for q in range(4) if q != kB]:
                    r = nearest(LB[k2], cA[0] - csB[k2]["sw"] / 2, cA[1] - csB[k2]["sh"] / 2)
                    if r: newB = (iB, k2, r[0], r[1]); break
                if newB is None: continue
                st = list(state); st[j] = newA; st[l] = newB; out.append(st)
        return out
    def all_legal(state):
        for j, (i, k, ox, oy) in enumerate(state):
            L = legal_maps(occ_of(state, {i}), i)
            if not bool(L[k][ox, oy]): return False
        return True
    def lab_of(state):
        lab = np.zeros((W, H), np.int16)
        for n_, (i, k, ox, oy) in enumerate(state, 1):
            m = rows[i]["D"]["rots"][k]["mask"]; sw, sh = m.shape; v = lab[ox:ox + sw, oy:oy + sh]; v[m] = n_
        return lab
    # ---- which beds are plates
    summary = dict(plates=[], carried_beds=[], unplaced=[], oversize=[])
    plate_beds = [b for b in range(NPOOL) if len(placements[b]) >= 2 and use(b) >= CARRY]
    for b in range(NPOOL):
        if placements[b] and b not in plate_beds:
            summary["carried_beds"].append(dict(bed=b + 1, pieces=[dict(model=rows[r["i"]]["set"], file=rows[r["i"]]["file"]) for r in placements[b]], why="fewer than 2 pieces or use below the planner's minimum plate use (carried, not a plate)"))
    placed = {r["i"] for b in range(NPOOL) for r in placements[b]}
    for i in range(len(rows)):
        if i in placed: continue
        ov = not any(c["inb"].any().item() for c in prep(i))
        summary["unplaced"].append(dict(model=rows[i]["set"], file=rows[i]["file"], path=rows[i]["path"], oversize=bool(ov), rot_shapes=[list(r["mask"].shape) for r in rows[i]["D"]["rots"]]))
        if ov: summary["oversize"].append(dict(model=rows[i]["set"], file=rows[i]["file"]))
    print(f"RING: {len(plate_beds)} plate beds, {len(summary['carried_beds'])} carried beds, {len(summary['unplaced'])} unplaced pieces ({len(summary['oversize'])} OVERSIZE)", flush=True)
    num = 0
    for b in plate_beds:
        num += 1; pid = "RING-%03d" % num; tb = time.time()
        rr = sorted(placements[b], key=lambda r: r["seq"])
        state0 = [(r["i"], r["k"], r["ox"], r["oy"]) for r in rr]; old = {r["i"]: r for r in rr}
        sc0 = ev(state0)
        if sc0 is None: print(f"RING {pid}: score undefined, left as is", flush=True); continue
        cur = sc0; state = state0; acc = 0; why = "no improving move"
        while True:
            if acc >= MOVE_CAP: why = "move cap"; break
            if time.time() - tb > T_CAP: why = "time cap"; break
            if _free_ram_gb() < 3.0: why = "RAM below 3 GB"; break
            bestst = None; bests = cur
            for st in candidates(state):
                sc = ev(st)
                if sc is not None and sc["key"] < bests["key"]: bests = sc; bestst = st
                if time.time() - tb > T_CAP * 1.5: break
            if bestst is None: break
            state = bestst; cur = bests; acc += 1
            ram_min = min(ram_min, _free_ram_gb())
        ok = True; applied = False
        moved = [t for t in state if t not in state0]
        if moved:
            ok = all_legal(state) and len(state) == len(state0) and {t[0] for t in state} == {t[0] for t in state0}
            if ok:
                olds = {}
                try:
                    for (i, k, ox, oy) in moved: olds[i] = uncommit(i)
                    for (i, k, ox, oy) in moved:
                        K = get_K(i)[k]; pull = rows[i]["pull"]
                        commit(dict(b=b, i=i, ox=ox, oy=oy, k=k, nz=K["nz"].cpu(), rnz=K["rnz"].cpu(), lv=(K["SH"][:, ox, oy] * pull).clone(), cx=((ox + K["rcx"]) * R - PX / 2), cy=((oy + K["rcy"]) * R - PY / 2), nlegal=olds[i].get("nlegal", 0), seq=olds[i].get("seq", 0)))
                    applied = True
                except Exception as ex:
                    print(f"RING {pid}: apply failed ({str(ex)[:120]}), reverting", flush=True)
                    for i in [t[0] for t in moved]:
                        try: uncommit(i)
                        except Exception: pass
                    for i, rec in olds.items(): commit(rec)
                    ok = False; state = state0; cur = sc0
            else:
                print(f"RING {pid}: final state failed the legality re-check, bed left as the planner placed it", flush=True)
                state = state0; cur = sc0
        # independent check of what the planner now holds
        now = [(r["i"], r["k"], r["ox"], r["oy"]) for r in placements[b]]
        assert sorted(now) == sorted(state), "planner state differs from reported state"
        assert all_legal(now) and len(now) == len(state0)
        lab0 = lab_of(state0); lab1 = lab_of(state)
        names = [dict(n=n_, model=rows[i]["set"], file=rows[i]["file"], path=rows[i]["path"], h=float(rows[i]["h"]), k=k, ox=ox, oy=oy, area_mm2=float(rows[i]["D"]["rots"][k]["mask"].sum() * R * R)) for n_, (i, k, ox, oy) in enumerate(state, 1)]
        names0 = {i: (k, ox, oy) for (i, k, ox, oy) in state0}
        for d_, (i, k, ox, oy) in zip(names, state): d_["before"] = list(names0[i])
        # state0 and state share piece order? make label numbering consistent: lab0 uses state0 order -> remap to state order
        rem = {i: n_ for n_, (i, *_r) in enumerate(state, 1)}; lab0 = np.zeros((W, H), np.int16)
        for (i, k, ox, oy) in state0:
            m = rows[i]["D"]["rots"][k]["mask"]; sw, sh = m.shape; v = lab0[ox:ox + sw, oy:oy + sh]; v[m] = rem[i]
        np.savez_compressed(os.path.join(DUMP, pid + ".npz"), lab_before=lab0, lab_after=lab_of(state))
        dt = time.time() - tb
        ent = dict(plate=pid, bed=b + 1, pieces=names, score_before={k_: v for k_, v in sc0.items() if k_ != "key"}, score_after={k_: v for k_, v in cur.items() if k_ != "key"}, moves=acc, stop=why, seconds=round(dt, 1), applied=applied, R=R, PX=PX, PY=PY, grid=[W, H], c_mm=c_mm, gap_mm=GAP_MM, planner_use=float(use(b)))
        summary["plates"].append(ent)
        print(f"RING {pid} (bed {b + 1}): {len(state)} pcs | c {sc0['c']}->{cur['c']} | sum {sc0['total']:.4f}->{cur['total']:.4f} | moves {acc} ({why}) | {dt:.0f}s | free RAM min {ram_min:.1f} GB", flush=True)
    summary.update(seconds=round(time.time() - T0, 1), ram_free_start_gb=round(ram0, 2), ram_free_min_gb=round(ram_min, 2), cuda_max_alloc_gb=(round(torch.cuda.max_memory_allocated() / 2 ** 30, 2) if torch.cuda.is_available() else None), R=R, PX=PX, PY=PY, failed_models=[str(x) for x in G.get("failed", [])])
    json.dump(summary, open(os.path.join(DUMP, "summary.json"), "w"), indent=1, default=str)
    print(f"RING: done in {summary['seconds']}s, summary in {DUMP}", flush=True)
