# mkm_ring_layout_hook2.py (round 2, 2026-11-19 ~23:00). Called from mkm_ring_layout_hook.apply when MKM_RING_ROUND=2 (planner untouched except the round-1 gated lines + MKM_RING_NOCARRY).
# Layout rule = geometry only. Score: (open end first) then a0 + b0 + aH + bH, where
#   a = distance of the (weighted) footprint centroid to the hull centroid / hull half-diagonal,
#   b = weighted std / weighted mean of fill(d) over EVERY depth step of one grid cell (weights = hull cells at that depth),
#   0 = unweighted footprint (the raft layers), H = footprint weighted by piece height rows[i]["h"] (everything above).
# No band count, no tier cut-offs, no fill target. Gap = the planner's own min gap for this run (g cells).
import os, sys, time, json, math
import numpy as np
import mkm_ring_layout_hook as H1

def even(M, Hh, c):
    ins = Hh["inside"]; st = np.floor(Hh["d"][ins] / c).astype(int)
    n = np.bincount(st).astype(float); s = np.bincount(st, weights=M[ins], minlength=len(n)); tot = n.sum()
    mean = s.sum() / tot
    if mean <= 0: return 0.0
    fill = s / np.maximum(n, 1); var = (n * (fill - mean) ** 2).sum() / tot
    return float(math.sqrt(var) / mean)

def score2(A, AH, c, gap_mm):
    Hh = H1.hull_depth(A, c)
    if Hh is None: return None
    sl = Hh["sl"]; o = {}
    a0 = H1._a(A[sl], Hh); b0 = even(A[sl], Hh, c); aH = H1._a(AH[sl], Hh); bH = even(AH[sl], Hh, c)
    xs, ys = np.nonzero(A > 0); cxm = (xs + 0.5) * c; cym = (ys + 0.5) * c
    dx = np.maximum(np.abs(cxm - Hh["end"][0]) - c / 2, 0); dy = np.maximum(np.abs(cym - Hh["end"][1]) - c / 2, 0)
    dist = float(np.sqrt(dx * dx + dy * dy).min()); cc = int(dist < gap_mm)
    tot = a0 + b0 + aH + bH
    o.update(c=cc, end_dist_mm=dist, a0=a0, b0=b0, aH=aH, bH=bH, total=tot, D=Hh["D"], end=Hh["end"], key=(cc, tot)); return o

def maps_from_lab2(lab, hmap, s):
    def pool(m):
        a = m.astype(np.float32); pw = (-a.shape[0]) % s; ph = (-a.shape[1]) % s
        if pw or ph: a = np.pad(a, ((0, pw), (0, ph)))
        return a.reshape(a.shape[0] // s, s, a.shape[1] // s, s).mean((1, 3))
    hl = np.zeros(int(lab.max()) + 1, np.float32)
    for l, h in hmap.items(): hl[l] = h
    return pool(lab > 0), pool(hl[lab])

def apply(G):
    import torch, torch.nn.functional as F
    T0 = time.time()
    OUT = os.environ.get("MKM_RING_OUT", r"C:\MKM_TMP\ring_plates_20261119"); DUMP = os.path.join(OUT, "ring2_dump"); os.makedirs(DUMP, exist_ok=True)
    T_BED = float(os.environ.get("MKM_RING_SECS", 180)); MOVE_CAP = int(os.environ.get("MKM_RING_MOVES", 400)); T_X = float(os.environ.get("MKM_RING_XSECS", 900)); T_POL = float(os.environ.get("MKM_RING_POLISH", 60))
    placements = G["placements"]; rows = G["rows"]; NPOOL = G["NPOOL"]; W = G["W"]; H = G["H"]; R = G["R"]; PX = G["PX"]; PY = G["PY"]
    g = int(G["g"]); dev = G["dev"]; get_K = G["get_K"]; commit = G["commit"]; uncommit = G["uncommit"]; circle = G["circle"]
    CS = G.get("CENTRE_SMALL", False); tier = G.get("tier"); BEDV = float(G["BED"]); MAXSPLIT = int(G["MAXSPLIT"])
    assert not G["ABUD"] and not G["FBUD"], "numeric allotments are active; cross-bed budget check not implemented"
    s = max(1, int(round(1.0 / R))); c_mm = s * R; GAP_MM = g * R
    ram0 = H1._free_ram_gb(); ram_min = [ram0]
    print(f"RING2: start. free RAM {ram0:.1f} GB, grid {W}x{H} @ {R} mm, score grid {c_mm} mm, gap {GAP_MM} mm (g={g}), caps {T_BED}s/{MOVE_CAP} moves per bed, cross {T_X}s", flush=True)
    P = {}; PM = {}; AREA = {}
    def prep(i):
        if i not in P: P[i] = [dict(f=K["f"].clone(), inb=K["inb"], nz=K["nz"], sw=int(K["sw"]), sh=int(K["sh"])) for K in get_K(i)]
        return P[i]
    def area(i):
        if i not in AREA: AREA[i] = float(rows[i]["D"]["rots"][0]["mask"].sum())
        return AREA[i]
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
    EVC = {}
    def ev(state):
        key = tuple(sorted(state))
        if key not in EVC:
            if len(EVC) > 20000: EVC.clear()
            EVC[key] = score2(*maps(state), c_mm, GAP_MM)
        return EVC[key]
    BIG = (9, 1e9)
    def kk(state):
        sc = ev(state); return sc["key"] if sc is not None else BIG
    anchors = [(W * fx, H * fy) for fx in (1 / 8, 3 / 8, 5 / 8, 7 / 8) for fy in (1 / 4, 3 / 4)]
    def repl(state, j, new):
        st = list(state); st[j] = new; return st
    def all_legal(state):
        for (i, k, ox, oy) in state:
            if not bool(legal_maps(occ_of(state, {i}), i)[k][ox, oy]): return False
        return True
    halo = lambda st: sum(float(rows[i]["halo"]) for (i, _k, _x, _y) in st)
    nmodels = lambda st: len({rows[i]["set"] for (i, _k, _x, _y) in st})
    # ---------------- state
    BEDS = {b: [(r["i"], r["k"], r["ox"], r["oy"]) for r in sorted(placements[b], key=lambda r: r["seq"])] for b in range(NPOOL) if placements[b]}
    ORIG = {b: list(v) for b, v in BEDS.items()}; OLDREC = {r["i"]: r for b in range(NPOOL) for r in placements[b]}
    placed = set(OLDREC)
    unplaced = [dict(model=rows[i]["set"], file=rows[i]["file"], path=rows[i]["path"]) for i in range(len(rows)) if i not in placed]
    summary = dict(plates=[], unplaced=unplaced, planner_beds_before=len(BEDS), planner_pieces_placed=len(placed), total_pieces=len(rows), merges=[], notes=[])
    print(f"RING2: planner placed {len(placed)}/{len(rows)} on {len(BEDS)} beds; {len(unplaced)} unplaced", flush=True)
    bedof = {i: b for b, st in BEDS.items() for (i, *_r) in st}
    def spans_ok(moves):
        """moves: {i: newbed}. unit bed-count must not increase, model bed-count <= max(MAXSPLIT, before)."""
        nb = dict(bedof); nb.update(moves); un = {rows[i]["unit"] for i in moves}; ms = {rows[i]["set"] for i in moves}
        for u in un:
            before = {bedof[j] for j in bedof if rows[j]["unit"] == u}; after = {nb[j] for j in nb if rows[j]["unit"] == u}
            if len(after) > len(before): return False
        for m in ms:
            before = {bedof[j] for j in bedof if rows[j]["set"] == m}; after = {nb[j] for j in nb if rows[j]["set"] == m}
            if len(after) > max(MAXSPLIT, len(before)): return False
        return True
    # ---------------- stage A: merge thin beds by geometry (no fill gate); Mike's one-model rule handled after
    def place_anywhere(state, i):
        L = legal_maps(occ_of(state), i); best = None
        for k in range(4):
            idx = torch.nonzero(L[k])
            if idx.shape[0] == 0: continue
            j = torch.argmin(idx[:, 1] * W + idx[:, 0] * 1.0); cand = (int(idx[j, 1]), int(idx[j, 0]), k, int(idx[j, 0]), int(idx[j, 1]))
            if best is None or cand[:2] < best[:2]: best = cand
        return None if best is None else (i, best[2], best[3], best[4])
    def try_empty(X, others):
        pcs = sorted(BEDS[X], key=lambda t: -area(t[0])); tent = {b: list(BEDS[b]) for b in others}; mv = {}
        for (i, k, ox, oy) in pcs:
            done = False
            for Y in sorted(others, key=lambda b: halo(tent[b])):
                if halo(tent[Y]) + float(rows[i]["halo"]) > BEDV: continue
                if not spans_ok({**mv, i: Y}): continue
                new = place_anywhere(tent[Y], i)
                if new is None: continue
                tent[Y].append(new); mv[i] = Y; done = True; break
            if not done: return None
        return tent, mv
    changed = True; tried = set()
    while changed:
        changed = False
        for X in sorted(BEDS, key=lambda b: sum(area(t[0]) for t in BEDS[b])):
            if X not in BEDS or (X, tuple(sorted(t[0] for t in BEDS[X]))) in tried: continue
            others = [b for b in BEDS if b != X]; r = try_empty(X, others)
            tried.add((X, tuple(sorted(t[0] for t in BEDS[X]))))
            if r is None: continue
            tent, mv = r
            if not all(all_legal(tent[b]) for b in tent): print(f"RING2: merge of bed {X + 1} failed re-check, skipped", flush=True); continue
            n = len(BEDS[X]); BEDS.update(tent); del BEDS[X]
            for i, Y in mv.items(): bedof[i] = Y
            summary["merges"].append(dict(emptied_bed=X + 1, pieces=n)); changed = True
            print(f"RING2: bed {X + 1} emptied onto other beds ({n} pieces); beds left {len(BEDS)}", flush=True); break
    # Mike's standing rule: no plate with just one model -> receive pieces from other beds (geometry-legal moves only)
    for X in [b for b in sorted(BEDS) if nmodels(BEDS[b]) < 2]:
        got = False
        for Z in sorted((b for b in BEDS if b != X), key=lambda b: -len(BEDS[b])):
            for (i, k, ox, oy) in sorted(BEDS[Z], key=lambda t: area(t[0])):
                if rows[i]["set"] in {rows[j]["set"] for (j, *_r) in BEDS[X]} or nmodels(BEDS[Z]) < 3 and len({t for t in BEDS[Z] if rows[t[0]]["set"] == rows[i]["set"]}) == len(BEDS[Z]): continue
                if halo(BEDS[X]) + float(rows[i]["halo"]) > BEDV or not spans_ok({i: X}): continue
                new = place_anywhere(BEDS[X], i)
                rest = [t for t in BEDS[Z] if t[0] != i]
                if new is None or len(rest) < 1 or nmodels(rest) < 2: continue
                BEDS[X] = BEDS[X] + [new]; BEDS[Z] = rest; bedof[i] = X; got = True; break
            if got: break
        if not got: summary["notes"].append(f"bed {X + 1} has one model and no legal piece could be added from another bed")
    for b in list(BEDS):
        if len(BEDS[b]) < 2 or nmodels(BEDS[b]) < 2: summary["notes"].append(f"bed {b + 1} still has {len(BEDS[b])} pieces / {nmodels(BEDS[b])} model(s) after merge/receive")
    A_STATE = {b: list(v) for b, v in BEDS.items()}
    print(f"RING2: after merge {len(BEDS)} beds, {sum(len(v) for v in BEDS.values())} pieces", flush=True)
    # ---------------- within-bed search
    def valid_bed(st): return nmodels(st) >= min(2, nmodels(A_STATE.get(0, st)) if False else 2) or True
    def within(b, tcap, mcap, tag):
        tb = time.time(); state = list(BEDS[b]); cur = kk(state); acc = 0; why = "no improving move"
        if ev(state) is None: return 0, "score undefined"
        def cands(state):
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
        while True:
            if acc >= mcap: why = "move cap"; break
            if time.time() - tb > tcap: why = "time cap"; break
            if H1._free_ram_gb() < 3.0: why = "RAM below 3 GB"; break
            best = None; bk = cur
            for st in cands(state):
                k_ = kk(st)
                if k_ < bk: bk = k_; best = st
                if time.time() - tb > tcap * 1.5: break
            if best is None: break
            state = best; cur = bk; acc += 1; ram_min[0] = min(ram_min[0], H1._free_ram_gb())
        BEDS[b] = state; return acc, why
    log_within = {}
    for b in sorted(BEDS):
        tb = time.time(); k0 = kk(BEDS[b]); acc, why = within(b, T_BED, MOVE_CAP, "A"); log_within[b] = (acc, why, round(time.time() - tb, 1))
        print(f"RING2 bed {b + 1}: {len(BEDS[b])} pcs | key {k0[0]},{k0[1]:.4f} -> {kk(BEDS[b])[0]},{kk(BEDS[b])[1]:.4f} | moves {acc} ({why}) | {time.time() - tb:.0f}s", flush=True)
    # ---------------- cross-bed moves and swaps (summed score of the two beds, open end first)
    tx0 = time.time(); xacc = 0; xlog = []
    def pair_best(X, Y):
        sx, sy = BEDS[X], BEDS[Y]; base = (kk(sx)[0] + kk(sy)[0], kk(sx)[1] + kk(sy)[1]); best = None; bk = base
        def consider(nx, ny, mv):
            nonlocal best, bk
            if len(nx) < 1 or len(ny) < 1 or nmodels(nx) < 2 or nmodels(ny) < 2: return
            if halo(nx) > BEDV or halo(ny) > BEDV or not spans_ok(mv): return
            a, b_ = kk(nx), kk(ny); k_ = (a[0] + b_[0], a[1] + b_[1])
            if k_ < bk: bk = k_; best = (nx, ny, mv)
        for (src, dst, S, D_) in ((X, Y, sx, sy), (Y, X, sy, sx)):
            for j, (i, k, ox, oy) in enumerate(S):
                if len(S) <= 2: break
                L = legal_maps(occ_of(D_), i); cs = prep(i); rest = [t for t in S if t[0] != i]
                for ax, ay in anchors:
                    r = nearest(L[k], ax - cs[k]["sw"] / 2, ay - cs[k]["sh"] / 2)
                    if r: consider(rest, D_ + [(i, k, r[0], r[1])], {i: dst}) if False else (consider(rest, D_ + [(i, k, r[0], r[1])], {i: dst}) if src == X else consider(D_ + [(i, k, r[0], r[1])], rest, {i: dst}))
        for (iA, kA, oxA, oyA) in sx:
            cand = sorted(sy, key=lambda t: abs(area(t[0]) - area(iA)))[:2]
            for (iB, kB, oxB, oyB) in cand:
                csA = prep(iA); csB = prep(iB)
                LA = legal_maps(occ_of(sy, {iB}), iA); LB = legal_maps(occ_of(sx, {iA}), iB)
                cB = (oxB + csB[kB]["sw"] / 2, oyB + csB[kB]["sh"] / 2); cA = (oxA + csA[kA]["sw"] / 2, oyA + csA[kA]["sh"] / 2); nA = nB = None
                for k2 in [kA] + [q for q in range(4) if q != kA]:
                    r = nearest(LA[k2], cB[0] - csA[k2]["sw"] / 2, cB[1] - csA[k2]["sh"] / 2)
                    if r: nA = (iA, k2, r[0], r[1]); break
                for k2 in [kB] + [q for q in range(4) if q != kB]:
                    r = nearest(LB[k2], cA[0] - csB[k2]["sw"] / 2, cA[1] - csB[k2]["sh"] / 2)
                    if r: nB = (iB, k2, r[0], r[1]); break
                if nA is None or nB is None: continue
                consider([t for t in sx if t[0] != iA] + [nB], [t for t in sy if t[0] != iB] + [nA], {iA: Y, iB: X})
        return best
    stop_x = False
    while not stop_x and xacc < MOVE_CAP:
        improved = False
        order = sorted(BEDS, key=lambda b: (-kk(BEDS[b])[0], -kk(BEDS[b])[1]))
        for X in order:
            for Y in sorted(BEDS, key=lambda b: (-kk(BEDS[b])[0], -kk(BEDS[b])[1])):
                if Y == X: continue
                if time.time() - tx0 > T_X: stop_x = True; break
                if H1._free_ram_gb() < 3.0: stop_x = True; break
                r = pair_best(X, Y)
                if r is None: continue
                nx, ny, mv = r
                if not (all_legal(nx) and all_legal(ny)): continue
                BEDS[X], BEDS[Y] = nx, ny
                for i, b in mv.items(): bedof[i] = b
                xacc += 1; improved = True; xlog.append((X + 1, Y + 1, list(mv)))
                print(f"RING2 cross {xacc}: beds {X + 1}/{Y + 1} pieces {[rows[i]['file'][:30] for i in mv]} | {time.time() - tx0:.0f}s", flush=True)
                if xacc >= MOVE_CAP: stop_x = True; break
            if stop_x: break
        if not improved: break
    touched = {b for b in BEDS if sorted(BEDS[b]) != sorted(A_STATE.get(b, []))}
    for b in sorted(touched):
        tb = time.time(); acc, why = within(b, T_POL, MOVE_CAP, "P"); print(f"RING2 polish bed {b + 1}: moves {acc} ({why}) {time.time() - tb:.0f}s", flush=True)
    # ---------------- final verification and apply to planner
    for b in BEDS:
        assert all_legal(BEDS[b]), f"bed {b + 1} illegal"
    allp = sorted(i for st in BEDS.values() for (i, *_r) in st)
    assert allp == sorted(placed) and len(allp) == len(set(allp)), "piece set changed"
    newpos = {i: (b, k, ox, oy) for b, st in BEDS.items() for (i, k, ox, oy) in st}
    old_pos = {i: (b, r["k"], r["ox"], r["oy"]) for i, r in OLDREC.items()}
    moved = [i for i in newpos if newpos[i] != old_pos[i]]
    olds = {}
    try:
        for i in moved: olds[i] = uncommit(i)
        for i in moved:
            b, k, ox, oy = newpos[i]; K = get_K(i)[k]; pull = rows[i]["pull"]
            commit(dict(b=b, i=i, ox=ox, oy=oy, k=k, nz=K["nz"].cpu(), rnz=K["rnz"].cpu(), lv=(K["SH"][:, ox, oy] * pull).clone(), cx=((ox + K["rcx"]) * R - PX / 2), cy=((oy + K["rcy"]) * R - PY / 2), nlegal=olds[i].get("nlegal", 0), seq=olds[i].get("seq", 0)))
    except Exception as ex:
        print(f"RING2: apply to planner failed ({str(ex)[:150]}); planner state reverted to its own placement", flush=True)
        for i in moved:
            try: uncommit(i)
            except Exception: pass
        for i, rec in olds.items(): commit(rec)
        summary["notes"].append("apply to planner failed and was reverted; layouts below are the hook's own and were not committed")
    # ---------------- dump
    num = 0
    for b in sorted(BEDS):
        num += 1; pid = "RING2-%03d" % num; state = BEDS[b]; sc1 = ev(state)
        lab = np.zeros((W, H), np.int16); names = []
        for n_, (i, k, ox, oy) in enumerate(state, 1):
            m = rows[i]["D"]["rots"][k]["mask"]; sw, sh = m.shape; v = lab[ox:ox + sw, oy:oy + sh]; v[m] = n_
            names.append(dict(n=n_, model=rows[i]["set"], file=rows[i]["file"], path=rows[i]["path"], h=float(rows[i]["h"]), k=k, ox=ox, oy=oy, area_mm2=float(m.sum() * R * R), was_bed=(old_pos[i][0] + 1)))
        np.savez_compressed(os.path.join(DUMP, pid + ".npz"), lab_after=lab)
        sc0 = ev(A_STATE[b]) if b in A_STATE else None
        summary["plates"].append(dict(plate=pid, bed=b + 1, pieces=names, score_start=({k_: v_ for k_, v_ in sc0.items() if k_ != "key"} if sc0 else None), score_after={k_: v_ for k_, v_ in sc1.items() if k_ != "key"},
                                      within=log_within.get(b), grid=[W, H], R=R, PX=PX, PY=PY, c_mm=c_mm, gap_mm=GAP_MM, start_pieces=len(A_STATE.get(b, []))))
    summary.update(seconds=round(time.time() - T0, 1), cross_moves=xacc, ram_free_start_gb=round(ram0, 2), ram_free_min_gb=round(ram_min[0], 2), cuda_max_alloc_gb=(round(torch.cuda.max_memory_allocated() / 2 ** 30, 2) if torch.cuda.is_available() else None), g=g, R=R, PX=PX, PY=PY, gap_mm=GAP_MM)
    json.dump(summary, open(os.path.join(DUMP, "summary.json"), "w"), indent=1, default=str)
    print(f"RING2: done in {summary['seconds']}s: {len(BEDS)} plates, {len(allp)} pieces, cross moves {xacc}", flush=True)
