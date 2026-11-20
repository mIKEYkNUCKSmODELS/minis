# mkm_ring_layout_hook3.py (round 3). Starts from round 2's final layout (ring2_dump\summary.json), same 17 beds / same piece-to-bed assignment, and only changes the objective + moves.
# Objective, lexicographic: (1) open-end clearance (mm from the ring end point to the nearest footprint edge), maximise;
# (2) spread: smallest gap between any two footprints and between a footprint and the bed edge, maximise; ties: mean of each piece's nearest-neighbour gap, maximise;
# (3) minimise a0+b0+aH+bH (round-2 balance + continuous evenness). No distance number anywhere except the grid cell (CELL = one 0.5 mm raster cell) used as the "within one cell" tolerance.
import os, sys, time, json, math
import numpy as np, cv2
import mkm_ring_layout_hook as H1
import mkm_ring_layout_hook2 as H2

def dt_info(lab, n):
    zero = (lab == 0).astype(np.uint8)
    dist, comp = cv2.distanceTransformWithLabels(zero, cv2.DIST_L2, cv2.DIST_MASK_PRECISE, labelType=cv2.DIST_LABEL_CCOMP)
    pc = np.zeros(int(comp.max()) + 1, np.int32); m = lab > 0; pc[comp[m]] = lab[m]
    M = np.full((n + 1, n + 1), 1e9)
    for c1, c2, d1, d2 in ((comp[:-1], comp[1:], dist[:-1], dist[1:]), (comp[:, :-1], comp[:, 1:], dist[:, :-1], dist[:, 1:])):
        df = c1 != c2
        if not df.any(): continue
        a = pc[c1[df]]; b = pc[c2[df]]; v = (d1[df] + d2[df]).astype(float); k = a != b
        np.minimum.at(M, (a[k], b[k]), v[k]); np.minimum.at(M, (b[k], a[k]), v[k])
    return dist, comp, pc, M            # M in cells: clear gap between pieces ~ (d1+d2) cells

def exact_metrics(lab, hmap, R, c, n):
    """accurate readout used for the reports: per-piece EDT gaps (mm), edge gap, clearance at the ring end point."""
    nn = []; ed = []
    H, W_ = lab.shape
    for p in range(1, n + 1):
        dt = cv2.distanceTransform((lab != p).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE); oth = (lab > 0) & (lab != p)
        nn.append(max(float(dt[oth].min()) - 1.0, 0.0) * R)
        xs, ys = np.nonzero(lab == p); ed.append(min(xs.min(), H - 1 - xs.max(), ys.min(), W_ - 1 - ys.max()) * R)
    A, AH = H2.maps_from_lab2(lab, hmap, max(1, int(round(1.0 / R)))); Hh = H1.hull_depth(A, c)
    tot = H1._a(A[Hh["sl"]], Hh) + H2.even(A[Hh["sl"]], Hh, c) + H1._a(AH[Hh["sl"]], Hh) + H2.even(AH[Hh["sl"]], Hh, c)
    dist = cv2.distanceTransform((lab == 0).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    ex = min(int(Hh["end"][0] / R), lab.shape[0] - 1); ey = min(int(Hh["end"][1] / R), lab.shape[1] - 1)
    clear = max(float(dist[ex, ey]) - 0.5, 0.0) * R
    return dict(clear=clear, gap_min=min(nn), gap_med=float(np.median(nn)), gap_mean=float(np.mean(nn)), edge_min=min(ed), total=tot, end=Hh["end"], D=Hh["D"], nn=nn)

def apply(G):
    import torch, torch.nn.functional as F
    T0 = time.time()
    OUT = os.environ.get("MKM_RING_OUT", r"C:\MKM_TMP\ring_plates_20261119"); DUMP = os.path.join(OUT, "ring3_dump"); os.makedirs(DUMP, exist_ok=True)
    T_BED = float(os.environ.get("MKM_RING_SECS", 240)); MOVE_CAP = int(os.environ.get("MKM_RING_MOVES", 3000))
    placements = G["placements"]; rows = G["rows"]; NPOOL = G["NPOOL"]; W = G["W"]; H = G["H"]; R = G["R"]; PX = G["PX"]; PY = G["PY"]
    g = int(G["g"]); dev = G["dev"]; get_K = G["get_K"]; commit = G["commit"]; uncommit = G["uncommit"]; circle = G["circle"]
    CS = G.get("CENTRE_SMALL", False); tier = G.get("tier")
    s = max(1, int(round(1.0 / R))); c_mm = s * R; CELL = R
    ram0 = H1._free_ram_gb(); ram_min = [ram0]
    def wait_ram():
        t_ = time.time()
        while H1._free_ram_gb() < 3.0:
            if time.time() - t_ > 1200: return False
            time.sleep(15)
        return True
    import threading
    def _samp():
        f = open(os.path.join(OUT, "ram3_samples.csv"), "w", buffering=1); f.write("t,free_gb\n")
        while True:
            f.write("%.0f,%.2f\n" % (time.time() - T0, H1._free_ram_gb())); time.sleep(30)
    threading.Thread(target=_samp, daemon=True).start()
    print(f"RING3: start. free RAM {ram0:.1f} GB, grid {W}x{H} @ {R} mm, score grid {c_mm} mm, gap {g * R} mm (g={g}), cap {T_BED}s per bed", flush=True)
    P = {}; PM = {}; CEN = {}
    def prep(i):
        if i not in P: P[i] = [dict(f=K["f"].clone(), inb=K["inb"], nz=K["nz"], sw=int(K["sw"]), sh=int(K["sh"])) for K in get_K(i)]
        return P[i]
    def cen(i, k):
        if (i, k) not in CEN: xs, ys = np.nonzero(rows[i]["D"]["rots"][k]["mask"]); CEN[(i, k)] = (xs.mean() + 0.5, ys.mean() + 0.5)
        return CEN[(i, k)]
    def occ_of(state, skip=()):
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
        if key not in PM: PM[key] = H1.pool_map(rows[i]["D"]["rots"][k]["mask"], ox % s, oy % s, s)
        return PM[key]
    GW = -(-W // s); GH = -(-H // s)
    def maps(state):
        A = np.zeros((GW, GH), np.float32); AH = A.copy()
        for (i, k, ox, oy) in state:
            p = pooled(i, k, ox, oy); x0 = ox // s; y0 = oy // s; ps = p[:GW - x0, :GH - y0]
            A[x0:x0 + ps.shape[0], y0:y0 + ps.shape[1]] += ps; AH[x0:x0 + ps.shape[0], y0:y0 + ps.shape[1]] += ps * float(rows[i]["h"])
        return A, AH
    def lab_of(state):
        lab = np.zeros((W, H), np.int16)
        for n_, (i, k, ox, oy) in enumerate(state, 1):
            m = rows[i]["D"]["rots"][k]["mask"]; sw, sh = m.shape; lab[ox:ox + sw, oy:oy + sh][m] = n_
        return lab
    EVC = {}
    def ev(state):
        key = tuple(state)
        if key in EVC: return EVC[key]
        A, AH = maps(state); Hh = H1.hull_depth(A, c_mm)
        if Hh is None: return None
        sl = Hh["sl"]; tot = H1._a(A[sl], Hh) + H2.even(A[sl], Hh, c_mm) + H1._a(AH[sl], Hh) + H2.even(AH[sl], Hh, c_mm)
        n = len(state); lab = lab_of(state); dist, comp, pc, M = dt_info(lab, n)
        ex = min(int(Hh["end"][0] / R), W - 1); ey = min(int(Hh["end"][1] / R), H - 1)
        t1 = max(float(dist[ex, ey]) - 0.5, 0.0) * R; endp = int(pc[comp[ex, ey]])
        nnc = M[1:, 1:].min(1)
        edge = np.array([min(ox, W - (ox + prep(i)[k]["sw"]), oy, H - (oy + prep(i)[k]["sh"])) for (i, k, ox, oy) in state], float)
        nnc = np.where(nnc > 1e8, edge, nnc)
        mg = min(float(nnc.min()), float(edge.min())) * R; mean = float(nnc.mean()) * R
        iu = np.dstack(np.unravel_index(np.argsort(np.where(np.triu(np.ones_like(M, bool), 1)[:, :] & (M < 1e8), M, 1e9), axis=None)[:3], M.shape))[0]
        pairs = [(int(a), int(b)) for a, b in iu if 1 <= a <= n and 1 <= b <= n and M[a, b] < 1e8]
        e = dict(t1=t1, mg=mg, mean=mean, tot=tot, end=Hh["end"], D=Hh["D"], pairs=pairs, endp=endp)
        if len(EVC) > 3000: EVC.clear()
        EVC[key] = e; return e
    lex = lambda e: (e["t1"], e["mg"], e["mean"], -e["tot"])
    def acceptable(nw, cur, anc):
        if nw is None: return False
        if nw["t1"] > cur["t1"]: return True
        if nw["t1"] >= anc["t1"] - CELL and (nw["mg"], nw["mean"]) > (cur["mg"], cur["mean"]): return True
        if nw["t1"] >= anc["t1"] - CELL and nw["mg"] >= anc["mg"] - CELL and nw["tot"] < cur["tot"]: return True
        return False
    anchors = [(W * fx, H * fy) for fx in (1 / 8, 3 / 8, 5 / 8, 7 / 8) for fy in (1 / 4, 3 / 4)]
    def all_legal(state):
        for (i, k, ox, oy) in state:
            if not bool(legal_maps(occ_of(state, {i}), i)[k][ox, oy]): return False
        return True
    def slide(state, j, u, L=None):
        """one-cell legal slides of piece j in directions with positive dot product to u; returns list of new tuples."""
        i, k, ox, oy = state[j]; out = []
        if L is None: L = legal_maps(occ_of(state, {i}), i)[k]
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if (dx or dy) and dx * u[0] + dy * u[1] > 1e-9:
                    nx, ny = ox + dx, oy + dy
                    if 0 <= nx < W and 0 <= ny < H and bool(L[nx, ny]): out.append((i, k, nx, ny))
        return out
    def cheap(state, cur):
        out = []
        pos = {}
        def c_of(j): i, k, ox, oy = state[j]; cx, cy = cen(i, k); return (ox + cx, oy + cy)
        for (a, b) in cur["pairs"]:
            ja, jb = a - 1, b - 1; ca, cb = c_of(ja), c_of(jb); u = (ca[0] - cb[0], ca[1] - cb[1]); nrm = math.hypot(*u) or 1.0; u = (u[0] / nrm, u[1] / nrm)
            na = slide(state, ja, u); nb = slide(state, jb, (-u[0], -u[1]))
            for t in na: st = list(state); st[ja] = t; out.append(st)
            for t in nb: st = list(state); st[jb] = t; out.append(st)
            if na and nb:
                best_a = max(na, key=lambda t: (t[2] - state[ja][2]) * u[0] + (t[3] - state[ja][3]) * u[1]); best_b = min(nb, key=lambda t: (t[2] - state[jb][2]) * u[0] + (t[3] - state[jb][3]) * u[1])
                st = list(state); st[ja] = best_a; st[jb] = best_b; out.append(st)
        je = cur["endp"] - 1; ce = c_of(je); ep = (cur["end"][0] / R, cur["end"][1] / R); u = (ce[0] - ep[0], ce[1] - ep[1]); nrm = math.hypot(*u) or 1.0
        for t in slide(state, je, (u[0] / nrm, u[1] / nrm)): st = list(state); st[je] = t; out.append(st)
        return out
    def full(state):
        out = []; n = len(state)
        for j, (i, k, ox, oy) in enumerate(state):
            cs = prep(i); L = legal_maps(occ_of(state, {i}), i); cxc = ox + cs[k]["sw"] / 2; cyc = oy + cs[k]["sh"] / 2
            for k2 in range(4):
                if k2 == k: continue
                r = nearest(L[k2], cxc - cs[k2]["sw"] / 2, cyc - cs[k2]["sh"] / 2)
                if r: st = list(state); st[j] = (i, k2, r[0], r[1]); out.append(st)
            for ax, ay in anchors:
                r = nearest(L[k], ax - cs[k]["sw"] / 2, ay - cs[k]["sh"] / 2)
                if r and r != (ox, oy): st = list(state); st[j] = (i, k, r[0], r[1]); out.append(st)
        for j in range(n):
            for l in range(j + 1, n):
                iA, kA, oxA, oyA = state[j]; iB, kB, oxB, oyB = state[l]; csA = prep(iA); csB = prep(iB); o = occ_of(state, {iA, iB})
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
    # ---------------- start layout = round 2 final
    S2 = json.load(open(os.path.join(OUT, "ring2_dump", "summary.json")))
    idx = {(r["path"], r["file"]): i for i, r in enumerate(rows)}
    BEDS = {}; plates2 = {}
    for pl in S2["plates"]:
        BEDS[pl["bed"] - 1] = [(idx[(p["path"], p["file"])], p["k"], p["ox"], p["oy"]) for p in pl["pieces"]]; plates2[pl["bed"] - 1] = pl
    allp = sorted(i for st in BEDS.values() for (i, *_r) in st)
    assert allp == list(range(len(rows))), "round-2 layout does not cover the pool"
    for b, st in BEDS.items(): assert all_legal(st), f"round-2 bed {b + 1} not legal in this run"
    START = {b: list(v) for b, v in BEDS.items()}
    print(f"RING3: round-2 layout loaded: {len(BEDS)} beds, {len(allp)} pieces, all legal", flush=True)
    log = {}
    for b in sorted(BEDS):
        EVC.clear(); PM.clear(); CEN.clear(); import gc; gc.collect()
        tb = time.time(); state = list(BEDS[b]); cur = ev(state); e0 = cur; best = (cur, state); anc = cur; acc = 0; nfull = 0; seen = {tuple(state)}; since = 0; why = "no acceptable move"
        def choose(cands, cur, anc):
            pick = None; pk = None
            for st in cands:
                if tuple(st) in seen: continue
                e = ev(st)
                if acceptable(e, cur, anc) and (pk is None or lex(e) > pk): pick = (e, st); pk = lex(e)
                if time.time() - tb > T_BED: break
            return pick
        while True:
            if time.time() - tb > T_BED: why = "time cap"; break
            if acc >= MOVE_CAP: why = "move cap"; break
            if not wait_ram(): why = "RAM stayed below 3 GB for 20 min"; break
            pick = choose(cheap(state, cur), cur, anc)
            if pick is None:
                nfull += 1; pick = choose(full(state), cur, anc)
                if pick is None: break
            cur, state = pick; acc += 1; seen.add(tuple(state)); since += 1
            if lex(cur) > lex(best[0]): best = (cur, state); anc = cur; since = 0
            if since >= 150: why = "no new best in 150 accepted moves"; break
            ram_min[0] = min(ram_min[0], H1._free_ram_gb())
        BEDS[b] = best[1]; log[b] = dict(moves=acc, full_passes=nfull, stop=why, seconds=round(time.time() - tb, 1))
        e1 = best[0]
        print(f"RING3 bed {b + 1}: {len(state)} pcs | clear {e0['t1']:.2f}->{e1['t1']:.2f} mm | min gap {e0['mg']:.2f}->{e1['mg']:.2f} | mean nn {e0['mean']:.2f}->{e1['mean']:.2f} | sum {e0['tot']:.4f}->{e1['tot']:.4f} | moves {acc} full {nfull} ({why}) {time.time() - tb:.0f}s", flush=True)
    for b in BEDS: assert all_legal(BEDS[b]), f"bed {b + 1} illegal at end"
    allq = sorted(i for st in BEDS.values() for (i, *_r) in st); assert allq == allp
    # ---------------- apply to the planner's structures
    newpos = {i: (b, k, ox, oy) for b, st in BEDS.items() for (i, k, ox, oy) in st}
    OLD = {r["i"]: r for b in range(NPOOL) for r in placements[b]}; old_pos = {i: (b, r["k"], r["ox"], r["oy"]) for b in range(NPOOL) for r in placements[b] for i in [r["i"]]}
    moved = [i for i in newpos if newpos[i] != old_pos.get(i)]; olds = {}; applied = True
    try:
        for i in moved:
            if i in old_pos: olds[i] = uncommit(i)
        for i in moved:
            b, k, ox, oy = newpos[i]; K = get_K(i)[k]; pull = rows[i]["pull"]
            commit(dict(b=b, i=i, ox=ox, oy=oy, k=k, nz=K["nz"].cpu(), rnz=K["rnz"].cpu(), lv=(K["SH"][:, ox, oy] * pull).clone(), cx=((ox + K["rcx"]) * R - PX / 2), cy=((oy + K["rcy"]) * R - PY / 2), nlegal=olds.get(i, {}).get("nlegal", 0), seq=olds.get(i, {}).get("seq", 0)))
    except Exception as ex:
        applied = False; print(f"RING3: apply to planner failed ({str(ex)[:150]}); reverting", flush=True)
        for i in moved:
            try: uncommit(i)
            except Exception: pass
        for i, rec in olds.items(): commit(rec)
    # ---------------- dump
    summary = dict(plates=[], R=R, PX=PX, PY=PY, gap_mm=g * R, c_mm=c_mm, applied_to_planner=applied)
    num = 0
    for b in sorted(BEDS):
        num += 1; pid = "RING3-%03d" % num; state = BEDS[b]; lab = lab_of(state); names = []
        for n_, (i, k, ox, oy) in enumerate(state, 1):
            m = rows[i]["D"]["rots"][k]["mask"]
            names.append(dict(n=n_, model=rows[i]["set"], file=rows[i]["file"], path=rows[i]["path"], h=float(rows[i]["h"]), k=k, ox=ox, oy=oy, area_mm2=float(m.sum() * R * R)))
        np.savez_compressed(os.path.join(DUMP, pid + ".npz"), lab_after=lab)
        e = ev(state); e0 = ev(START[b])
        summary["plates"].append(dict(plate=pid, bed=b + 1, pieces=names, round2_plate=plates2[b]["plate"], search=log[b], hook_start={k_: v for k_, v in e0.items() if k_ in ("t1", "mg", "mean", "tot")}, hook_after={k_: v for k_, v in e.items() if k_ in ("t1", "mg", "mean", "tot")}))
    summary.update(seconds=round(time.time() - T0, 1), ram_free_start_gb=round(ram0, 2), ram_free_min_gb=round(ram_min[0], 2), cuda_max_alloc_gb=(round(torch.cuda.max_memory_allocated() / 2 ** 30, 2) if torch.cuda.is_available() else None))
    json.dump(summary, open(os.path.join(DUMP, "summary.json"), "w"), indent=1, default=str)
    print(f"RING3: done in {summary['seconds']}s, {len(BEDS)} plates", flush=True)
