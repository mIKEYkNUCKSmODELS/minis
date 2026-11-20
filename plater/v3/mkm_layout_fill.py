"""mkm_layout_fill.py - MKM plater v3 step (d): layout refill (no big empty middle) + band events at layout time.

Replaces the 05:30 patch ref/plate_heat_20261120/auto/spread.py (that one re-ran the old greedy film-cost planner at a wider
and wider gap; the objective that empties the centre stayed). Here, per plate, with the plate's pieces kept:
  1. footprints for the 4 quarter-turns from the planner piece cache (mkm_gpupool10 get_piece, loaded from REF_TREE);
  2. greedy maximin landing on an occupancy grid (each piece where its clearance is largest; compact fallback);
  3. spread phase: every piece, lowest clearance first, is lifted and re-placed (move and/or rotate) where its own clearance
     is largest; the smallest clearance never goes down (a moved piece is farther from everyone than before);
  4. band phase (merged rule 11:35 s.3): rotate / swap positions to clear a worse event, taken only without a hard defect
     and without a new event of the same class; then a guarded spread (no move that worsens the band);
  5. swaps between plates (1-for-1, plate count unchanged) only for a plate with a hard defect or leftover band events.
Writes layout_out/P{n}/recipe.json (schema 1, matrix updated, verdict untouched), RING_MANIFEST.json, LAYOUT_NOTES.json, P{n}_top.png,
and layout_out/LAYOUT_NOTES.json. The band is NOT a veto: leftover events are written down (model, height interval, event).

ASSUMPTIONS - verify on MKM
- A1 The planner cache entry (npz: mask0-3 at raster_pitch_mm, mn/c0/pad, area per FH.DZ height from FH.Z0) is what get_piece
  returns; mask[x, y] with x along the bed's 298 mm side; area is in mm2. Loaded by executing ONLY the cache functions of
  mkm_gpupool10.py (ast), because importing that file runs the whole planner.
- A2 Bed for placement = planner machine profile bed_mm (298 x 164), grid = raster_pitch_mm; world x = ox*R - PX/2 (write_bed).
- A3 Clearance is measured cell centre to cell centre minus one cell (= edge-to-edge on the grid, planner_geom convention);
  the bed edge is a ring of cells just outside the bed. Floor = ceil(gap_mm / R) cells. True geometry is within one cell.
- A4 Band events at layout time use a PROXY cross-section: at each height sample the piece's cured area is taken as the
  deepest cells of its silhouette (inside distance), as many as area / R^2. Real per-layer shapes are checked by mkm_rate.
- A5 Peel front = rectangular rings at distance t from the bed edge (Chebyshev), one grid step at a time.
  START: a nested piece (none of its first-met cells is visible from the bed edge along the direction it is met from) with
  fewer than 2 shielded sides (side = -x,+x,-y,+y; shielded = another piece lies between it and that edge).
  LANDING: a larger shielding neighbour's last cell lets go while the smaller piece is mid-peel (met, not finished) -> open edge.
  END: the last ring holds cells of 2+ pieces (shared last loop). Identical repeated height samples are counted once.
- A6 Cross-plate swaps ignore the planner's cured-area allotment (a fitted number); a swap is recorded so Mike can refuse it.
- A7 The *_RING.stl named in RING_MANIFEST is not rewritten; the slicer reads recipe.json sources + matrix.
- A8 Search budgets (passes, candidates, thresholds per batch) bound run time only; they are not pass/fail numbers.
"""
import os, sys, json, ast, math, time, argparse, hashlib
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import mkm_v3_config as CFG

GAP_MM = CFG.GAP_MM
try:
    import torch
except Exception:
    torch = None
try:
    from scipy import ndimage
except Exception:
    ndimage = None

N_THRESH = 24          # A8: clearance thresholds tested in one FFT batch
SPREAD_PASSES = 8      # A8
BAND_CANDIDATES = 60   # A8: band-phase candidate layouts per plate
SWAP_CANDIDATES = 4    # A8: cross-plate swap candidates per problem plate
DIRS = ("-x", "+x", "-y", "+y")


# ------------------------------------------------------------------ grid core (no planner needed)
def _dev():
    if torch is not None and os.environ.get("MKM_DEV", "cuda").lower() != "cpu" and torch.cuda.is_available():
        return "cuda"
    return None


def _edt(free):
    """Euclidean distance (cells) from each cell to the nearest False cell of `free`."""
    if ndimage is not None:
        return ndimage.distance_transform_edt(free)
    W, H = free.shape; big = W + H + 1
    g = np.full((W, H), big, np.float64)                      # pass 1: exact 1-D distance along axis 0
    idx = np.where(~free, np.arange(W)[:, None], -big)
    last = np.maximum.accumulate(idx, 0); g = np.minimum(g, np.arange(W)[:, None] - last)
    idx = np.where(~free, np.arange(W)[:, None], 2 * big + W)
    nxt = np.minimum.accumulate(idx[::-1], 0)[::-1]; g = np.minimum(g, nxt - np.arange(W)[:, None])
    g2 = g ** 2; out = np.empty((W, H)); yy = np.arange(H)
    for r0 in range(0, W, 64):                                # pass 2: brute min over axis 1 (fallback only)
        blk = g2[r0:r0 + 64]
        out[r0:r0 + 64] = (blk[:, None, :] + (yy[None, :, None] - yy[None, None, :]) ** 2).min(2)
    return np.sqrt(out)


def _dist_field(occ):
    """distance (cells, centre to centre) to the nearest occupied cell or to the ring of cells just outside the bed (A3)."""
    p = np.zeros((occ.shape[0] + 2, occ.shape[1] + 2), bool); p[0, :] = p[-1, :] = p[:, 0] = p[:, -1] = True; p[1:-1, 1:-1] = occ
    return _edt(~p)[1:-1, 1:-1]


def _free_starts(blocked, mask):
    """blocked (T,W,H) bool, mask (w,h) bool -> (T,W-w+1,H-h+1) bool: True where the mask with its corner there touches no blocked cell.
    FFT correlation, all T thresholds in one batch (GPU when torch+cuda, else CPU)."""
    T, W, H = blocked.shape; w, h = mask.shape
    if w > W or h > H:
        return np.zeros((T, 0, 0), bool)
    s = (W + w - 1, H + h - 1); km = np.ascontiguousarray(mask[::-1, ::-1], np.float32); dev = _dev()
    if dev or torch is not None:
        d = dev or "cpu"; b = torch.from_numpy(blocked.astype(np.float32)).to(d); k = torch.from_numpy(km).to(d)
        c = torch.fft.irfft2(torch.fft.rfft2(b, s=s) * torch.fft.rfft2(k, s=s), s=s)[:, w - 1:W, h - 1:H]
        return (c < 0.5).cpu().numpy()                         # one transfer per batch
    c = np.fft.irfft2(np.fft.rfft2(blocked.astype(np.float32), s=s) * np.fft.rfft2(km, s=s), s=s)[:, w - 1:W, h - 1:H]
    return c < 0.5


def _rots(m):
    if isinstance(m, (list, tuple)):
        return [np.asarray(r, bool) for r in m]
    m = np.asarray(m, bool); return [np.rot90(m, k) for k in range(4)]


def _occ(R, pl, shape, skip=None):
    o = np.zeros(shape, bool)
    for j, p in enumerate(pl):
        if p is None or j == skip:
            continue
        x, y, k = p; m = R[j][k]; o[x:x + m.shape[0], y:y + m.shape[1]] |= m
    return o


def _piece_clear(R, pl, shape, i, D=None):
    """clearance (cells, edge to edge) of piece i to the others and the bed edge."""
    if pl[i] is None:
        return -1.0
    x, y, k = pl[i]; m = R[i][k]; W, H = shape
    if x < 0 or y < 0 or x + m.shape[0] > W or y + m.shape[1] > H:
        return -1.0
    if D is None:
        D = _dist_field(_occ(R, pl, shape, skip=i))
    return float(D[x:x + m.shape[0], y:y + m.shape[1]][m].min()) - 1.0


def min_clearance(masks, placements, bed_shape_px):
    """smallest model-model / model-edge clearance in cells (edge to edge, A3). Unplaced or outside the bed -> -1."""
    R = [_rots(m) for m in masks]
    return min(_piece_clear(R, list(placements), tuple(bed_shape_px), i) for i in range(len(R))) if R else 0.0


def _best_spot(rots, occ, c_lo, target, mode="max", ks=(0, 1, 2, 3)):
    """-> (clearance, x, y, k) or None. mode max: largest clearance >= c_lo (tie: nearest target); nearest: clearance >= c_lo, nearest target."""
    D = _dist_field(occ); cmax = int(D.max()) - 1
    if cmax < c_lo:
        return None
    best = None
    for k in ks:
        m = rots[k]
        if mode == "max":
            th = np.unique(np.linspace(c_lo, cmax, min(N_THRESH, cmax - c_lo + 1)).round().astype(int))
        else:
            th = np.array([c_lo])
        z = _free_starts(D[None] < (th[:, None, None] + 1), m)
        if z.size == 0:
            continue
        feas = z.reshape(len(th), -1).any(1)
        if not feas[0]:
            continue
        t = int(np.nonzero(feas)[0].max()); c = int(th[t]); zz = z[t]
        if mode == "max" and t + 1 < len(th) and th[t + 1] - th[t] > 1:      # fine step between two coarse thresholds
            th2 = np.arange(th[t] + 1, th[t + 1]); z2 = _free_starts(D[None] < (th2[:, None, None] + 1), m)
            f2 = z2.reshape(len(th2), -1).any(1)
            if f2.any():
                t2 = int(np.nonzero(f2)[0].max()); c = int(th2[t2]); zz = z2[t2]
        xs, ys = np.nonzero(zz)
        d2 = (xs + m.shape[0] / 2 - target[0]) ** 2 + (ys + m.shape[1] / 2 - target[1]) ** 2; j = int(d2.argmin())
        cand = (c, int(xs[j]), int(ys[j]), k, float(d2[j]))
        if best is None or (mode == "max" and (cand[0], -cand[4]) > (best[0], -best[4])) or (mode != "max" and cand[4] < best[4]):
            best = cand
    return None if best is None else best[:4]


def _initial(R, shape, gap):
    W, H = shape; order = sorted(range(len(R)), key=lambda i: -int(R[i][0].sum())); centre = (W / 2, H / 2)
    for mode in ("max", "compact"):
        pl = [None] * len(R); ok = True
        for i in order:
            o = _occ(R, pl, shape)
            s = _best_spot(R[i], o, gap, centre, "max") if mode == "max" else _best_spot(R[i], o, gap, (0, 0), "nearest")
            if s is None:
                ok = False; continue
            pl[i] = (s[1], s[2], s[3])
        if ok:
            return pl
    return pl                                                  # hard defect: unplaced pieces stay None


def _centre(R, pl, i):
    x, y, k = pl[i]; m = R[i][k]; return (x + m.shape[0] / 2, y + m.shape[1] / 2)


def _spread(R, pl, shape, gap, guard=None):
    """maximin: lowest clearance first, each piece re-placed (move + quarter-turn) where its own clearance is largest."""
    pl = list(pl); moves = 0
    for _ in range(SPREAD_PASSES):
        changed = False
        cl = [_piece_clear(R, pl, shape, i) for i in range(len(R))]
        for i in sorted(range(len(R)), key=lambda i: cl[i]):
            if pl[i] is None:
                continue
            o = _occ(R, pl, shape, skip=i); cur = _piece_clear(R, pl, shape, i, _dist_field(o))
            s = _best_spot(R[i], o, max(gap, int(cur) + 1), _centre(R, pl, i), "max")
            if s is None:
                continue
            new = list(pl); new[i] = (s[1], s[2], s[3])
            if guard is not None and not guard(new):
                continue
            pl = new; changed = True; moves += 1
        if not changed:
            break
    return pl, moves


def place_masks(masks, bed_shape_px, gap_px):
    """numpy bool footprint masks (one 2-D mask, rotated by rot90, or a list of the 4 quarter-turn masks) -> list of (x, y, rot);
    None for a piece that does not fit at gap_px. x along axis 0, y along axis 1, top-left corner of the rotated mask."""
    R = [_rots(m) for m in masks]; shape = tuple(int(v) for v in bed_shape_px); g = int(math.ceil(gap_px))
    pl = _initial(R, shape, g)
    if all(p is not None for p in pl):
        pl, _ = _spread(R, pl, shape, g)
    return pl


# ------------------------------------------------------------------ band events (A4, A5)
def _depth_order(m):
    """cells of a mask, deepest (inside distance) first: the proxy cross-section of A4 is a prefix of this list."""
    p = np.zeros((m.shape[0] + 2, m.shape[1] + 2), bool); p[1:-1, 1:-1] = m
    d = _edt(p)[1:-1, 1:-1]; xs, ys = np.nonzero(m); o = np.argsort(-d[xs, ys], kind="stable")
    return xs[o].astype(np.int32), ys[o].astype(np.int32)


def _nearest_before(L, axis, rev):
    """label of the nearest occupied cell strictly between each cell and the bed edge along axis (towards index 0, or the end if rev)."""
    A = np.flip(L, axis) if rev else L; n = A.shape[axis]
    ar = np.arange(n)[:, None] if axis == 0 else np.arange(n)[None, :]
    last = np.maximum.accumulate(np.where(A >= 0, ar, -1), axis)
    prev = np.concatenate([np.full((1, A.shape[1]) if axis == 0 else (A.shape[0], 1), -1), last[:-1] if axis == 0 else last[:, :-1]], axis)
    pc = np.clip(prev, 0, None)
    F = np.where(prev >= 0, np.take_along_axis(A, pc, axis), -1)
    return np.flip(F, axis) if rev else F


def band_events(P, pl, shape):
    """P[i] = dict(order=[(xs,ys) per k], n=cells per height sample). -> dict(key counts, events, shield score)."""
    W, H = shape; S = max((len(p["n"]) for p in P), default=0); seen = {}; ev = []; shield = 0
    for s in range(S):
        act = tuple((i, int(P[i]["n"][s])) for i in range(len(P)) if pl[i] is not None and s < len(P[i]["n"]) and P[i]["n"][s] > 0)
        if not act:
            continue
        if act in seen:                                        # identical repeated layer: counted once, interval still reported
            seen[act][1].append(s); continue
        L = np.full((W, H), -1, np.int32); cells = {}
        for i, n in act:
            x0, y0, k = pl[i]; xs, ys = P[i]["order"][k]; xs = xs[:n] + x0; ys = ys[:n] + y0; L[xs, ys] = i; cells[i] = (xs, ys)
        F = {"-x": _nearest_before(L, 0, False), "+x": _nearest_before(L, 0, True), "-y": _nearest_before(L, 1, False), "+y": _nearest_before(L, 1, True)}
        info = {}; area = dict(act)
        for i, (xs, ys) in cells.items():
            de = np.minimum(np.minimum(xs, W - 1 - xs), np.minimum(ys, H - 1 - ys)); tmin, tmax = int(de.min()), int(de.max()); f = de == tmin
            met = {"-x": xs == tmin, "+x": W - 1 - xs == tmin, "-y": ys == tmin, "+y": H - 1 - ys == tmin}
            vis = any(((F[d][xs, ys] == -1) & met[d] & f).any() for d in DIRS)
            sh = {d: set(np.unique(F[d][xs, ys]).tolist()) - {-1, i} for d in DIRS}
            info[i] = (tmin, tmax, not vis, sh)
        out = []
        for i, (tmin, tmax, nested, sh) in info.items():
            if nested and sum(1 for d in DIRS if sh[d]) < 2:
                out.append(("START", i, -1))
            for j in sorted(set().union(*sh.values())):
                if area[j] > area[i]:
                    shield += 1
                    if tmin < info[j][1] < tmax:
                        out.append(("LANDING", i, j))
        T = max(v[1] for v in info.values()); own = sorted(i for i, v in info.items() if v[1] == T)
        if len(own) >= 2:
            out.append(("END", own[0], own[1]))
        seen[act] = (out, [s])
    for out, ss in seen.values():
        for e in out:
            ev.append((e[0], e[1], e[2], tuple(ss)))
    cnt = {c: sum(1 for e in ev if e[0] == c) for c in ("END", "LANDING", "START")}
    return dict(count=cnt, events=ev, shield=shield)


def _key(b, hard, prefer):
    k = (hard, b["count"]["END"], b["count"]["LANDING"], b["count"]["START"])
    return k + ((-b["shield"],) if prefer else ())


def _better(nb, nh, ob, oh, prefer):
    """merged rule s.3: compare hard defect, shared last loop, open-edge landings, one-sided starts; a change that clears a worse
    event is taken only without a hard defect and without a new event of the same class."""
    if nh:
        return False
    kn, ko = _key(nb, nh, prefer), _key(ob, oh, prefer)
    if kn >= ko:
        return False
    for ci, c in enumerate(("END", "LANDING", "START")):
        if kn[ci + 1] != ko[ci + 1]:
            ids = lambda b: {(e[0], e[1], e[2]) for e in b["events"] if e[0] == c}
            return ids(nb) <= ids(ob)
    return True                                                # equal band, better shield tie-break (prefer_shielded)


def _hard(R, pl, shape, gap):
    if any(p is None for p in pl):
        return 1
    return int(bool(R) and min(_piece_clear(R, pl, shape, i) for i in range(len(R))) < gap)


def _band_phase(R, P, pl, shape, gap, prefer):
    b = band_events(P, pl, shape); h = _hard(R, pl, shape, gap); moves = []; tried = 0
    while tried < BAND_CANDIDATES:
        cls = next((c for c in ("END", "LANDING", "START") if b["count"][c]), None)
        if cls is None and not prefer:
            break
        owners = sorted({e[1] for e in b["events"] if e[0] == cls} | {e[2] for e in b["events"] if e[0] == cls and e[2] >= 0}) if cls else list(range(len(R)))
        took = False
        for i in owners:
            cands = []
            for k in range(4):                                 # rotate in place
                if k != pl[i][2]:
                    s = _best_spot(R[i], _occ(R, pl, shape, skip=i), gap, _centre(R, pl, i), "nearest", ks=(k,))
                    if s:
                        c = list(pl); c[i] = (s[1], s[2], s[3]); cands.append((c, "rotate P%d to k=%d" % (i, k)))
            for j in range(len(R)):                            # swap positions with another piece
                if j == i:
                    continue
                ci, cj = _centre(R, pl, i), _centre(R, pl, j); c = list(pl); c[i] = c[j] = None
                si = _best_spot(R[i], _occ(R, c, shape), gap, cj, "nearest")
                if not si:
                    continue
                c[i] = (si[1], si[2], si[3]); sj = _best_spot(R[j], _occ(R, c, shape), gap, ci, "nearest")
                if sj:
                    c[j] = (sj[1], sj[2], sj[3]); cands.append((c, "swap places of pieces %d and %d" % (i, j)))
            for c, why in cands:
                tried += 1; nb = band_events(P, c, shape); nh = _hard(R, c, shape, gap)
                if _better(nb, nh, b, h, prefer):
                    pl, b, h = c, nb, nh; moves.append(why); took = True; break
                if tried >= BAND_CANDIDATES:
                    break
            if took or tried >= BAND_CANDIDATES:
                break
        if not took:
            break
    kb = _key(b, h, prefer)
    pl, n = _spread(R, pl, shape, gap, guard=lambda c: _hard(R, c, shape, gap) == 0 and _key(band_events(P, c, shape), 0, prefer) <= kb)
    if n:
        moves.append("guarded spread: %d moves" % n)
    return pl, band_events(P, pl, shape), moves


# ------------------------------------------------------------------ planner cache (A1) - executed from REF_TREE, not copied
_PL = [None]
_FUNCS = {"rot", "dense_pts", "load_raw", "file_identity", "_area_now", "compute_piece", "_rots_from_file", "fill_raft_band",
          "cache_file", "_has_rm", "legacy_cache_files", "resolve_cache", "get_piece"}
_VARS = {"DT", "CACHE_DIR", "CACHE_VER", "LEGACY_CACHE_DIR", "ALIAS_DIR", "FP_MEASURED", "FP_HOW", "LEGACY_DIRS", "FILL_DIR",
         "FP_FILLED", "FID_FILE", "_FID_DISK", "_FID_NEW", "_FILE_ID"}


def _planner():
    if _PL[0] is not None:
        return _PL[0]
    pk = os.environ.get("MKM_PACKER_DIR", os.path.join(os.path.dirname(CFG.REF_TREE), "plate_packer"))
    for d in (CFG.REF_TREE, CFG.PHYSICS_TREE, pk):
        if os.path.isdir(d) and d not in sys.path:
            sys.path.insert(0, d)
    import mkm_planner_params as PP, mkm_fullheight as FH, mkm_ringpack as RP, mkm_core_api as api, mkm_gpuport as GP
    pn = os.environ.get("GB_PROFILES", "anycubic_m7max,film_acf,resin_sunlu_abslike,settings_mkm_mini").split(",")
    ch = os.environ.get("GB_BRACKET", "central")
    PROF = api.load_profiles(*pn, directory=os.environ.get("GB_PROFILE_DIR", api.PROFILE_DIR), choice=ch)
    PRM = PP.load(choice=ch, printer=pn[0], film=pn[1], resin=pn[2])
    ns = dict(os=os, sys=sys, json=json, hashlib=hashlib, np=np, ndimage=ndimage, torch=torch, PP=PP, FH=FH, RP=RP, GP=GP, api=api,
              SYNTH=False, GPUSCAN_DIR=CFG.REF_TREE, PRM=PRM, PROF=PROF, RAFT_BAND=PRM.get("raft_band_mm"), R=float(PRM.get("raster_pitch_mm")))
    GP.RAFT_BAND_MM = ns["RAFT_BAND"]; ns["DEVSTR"] = _dev() or "cpu"
    ns["GPU_RASTER"] = ns["DEVSTR"] == "cuda" and GP.configure() is not None
    src = os.path.join(CFG.REF_TREE, "mkm_gpupool10.py"); tree = ast.parse(open(src, encoding="utf-8").read()); keep = []; got = set()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in _FUNCS and node.name not in got:
            keep.append(node); got.add(node.name)
        elif isinstance(node, ast.Assign) and all(isinstance(t, ast.Name) and t.id in _VARS for t in node.targets):
            keep.append(node)
    exec(compile(ast.Module(body=keep, type_ignores=[]), src, "exec"), ns)
    PX, PY = [float(v) for v in PROF.raw["machine"]["_known_facts_not_used_by_core"]["bed_mm"]["value"]]
    _PL[0] = dict(get_piece=ns["get_piece"], R=ns["R"], PX=PX, PY=PY, Z0=float(FH.Z0), DZ=float(FH.DZ), cache={})
    return _PL[0]


def _piece(PL, pc):
    k = os.path.normcase(os.path.abspath(pc["path"]))
    if k not in PL["cache"]:
        D, _ = PL["get_piece"](dict(path=pc["path"], set=pc.get("set", "")))
        R = [np.asarray(D["rots"][j]["mask"], bool) for j in range(4)]; R2 = PL["R"] ** 2
        n = np.minimum(np.round(np.asarray(D["area"], float) / R2), R[0].sum()).astype(int)
        PL["cache"][k] = dict(D=D, R=R, P=dict(order=[_depth_order(m) for m in R], n=np.maximum(n, 0)))
    return PL["cache"][k]


# ------------------------------------------------------------------ per plate
def _layout(pieces, shape, gap, prefer):
    R = [p["R"] for p in pieces]; P = [p["P"] for p in pieces]
    pl = _initial(R, shape, gap); moves = []
    if any(p is None for p in pl):
        return pl, None, ["did not fit at the gap floor: %d of %d pieces placed" % (sum(p is not None for p in pl), len(pl))], 1
    pl, n = _spread(R, pl, shape, gap); moves.append("spread: %d moves" % n)
    pl, b, mv = _band_phase(R, P, pl, shape, gap, prefer)
    return pl, b, moves + mv, _hard(R, pl, shape, gap)


def _events_out(b, names, Z0, DZ):
    out = []
    for cls, a, c, ss in (b or {}).get("events", []):
        ss = sorted(ss); runs = []; st = pr = ss[0]
        for s in ss[1:] + [None]:
            if s is not None and s == pr + 1:
                pr = s; continue
            runs.append([round(Z0 + DZ * st, 2), round(Z0 + DZ * (pr + 1), 2)])
            if s is not None:
                st = pr = s
        out.append(dict(event=cls, model=names[a], other=names[c] if c >= 0 else None, height_mm=runs))
    return out


def _png(path, R, pl, shape, title):
    try:
        from PIL import Image, ImageDraw
    except Exception:
        return False
    W, H = shape; img = np.full((H, W, 3), 245, np.uint8); rng = np.random.default_rng(7)
    for i, p in enumerate(pl):
        if p is None:
            continue
        x, y, k = p; m = R[i][k]; xs, ys = np.nonzero(m); img[H - 1 - (ys + y), xs + x] = rng.integers(40, 220, 3)
    im = Image.fromarray(img).resize((W * 2, H * 2)); dr = ImageDraw.Draw(im); dr.rectangle([0, 0, W * 2 - 1, H * 2 - 1], outline=(0, 0, 0))
    for i, p in enumerate(pl):
        if p is not None:
            cx, cy = _centre(R, pl, i); dr.text((cx * 2, (H - cy) * 2), str(i + 1), fill=(0, 0, 0))
    dr.text((6, 6), title, fill=(0, 0, 0)); im.save(path); return True


def _write_plate(src_dir, dst_dir, name, pieces, man, rec, pl, b, moves, hard, PL, shape, gap):
    os.makedirs(dst_dir, exist_ok=True); R, PX, PY = PL["R"], PL["PX"], PL["PY"]; Rm = [p["R"] for p in pieces]
    names = ["%d %s" % (i + 1, p["pc"].get("file", os.path.basename(p["pc"]["path"]))) for i, p in enumerate(pieces)]
    NM = ['back-left', 'back-centre', 'back-right', 'middle-left', 'middle-centre', 'middle-right', 'front-left', 'front-centre', 'front-right']
    un = _occ(Rm, pl, shape); fill = round(100.0 * un.sum() * R * R / (PX * PY), 1)
    clear_mm = round(min(_piece_clear(Rm, pl, shape, i) for i in range(len(Rm))) * R, 2) if not any(p is None for p in pl) else None
    if not hard:
        for i, p in enumerate(pieces):
            ox, oy, k = pl[i]; r = p["D"]["rots"][k]; c_, s_ = [(1, 0), (0, 1), (-1, 0), (0, -1)][k]
            dx = ox * R - PX / 2 - float(r["mn"][0]) - (int(r["c0"][0]) - int(r["pad"])) * R
            dy = oy * R - PY / 2 - float(r["mn"][1]) - (int(r["c0"][1]) - int(r["pad"])) * R
            src = p["src"]; dz = float(src["matrix"][2][3])
            src["matrix"] = [[float(c_), float(-s_), 0.0, float(np.float32(dx))], [float(s_), float(c_), 0.0, float(np.float32(dy))], [0.0, 0.0, 1.0, dz], [0.0, 0.0, 0.0, 1.0]]
            sw, sh = r["mask"].shape; cx = (ox + sw / 2) * R - PX / 2; cy = (oy + sh / 2) * R - PY / 2
            col = 0 if cx > PX / 6 else (2 if cx < -PX / 6 else 1); row = 0 if cy > PY / 6 else (2 if cy < -PY / 6 else 1)
            p["pc"].update(k=k, x=round(cx, 1), y=round(cy, 1), where=NM[row * 3 + col])
        rec["sources"] = [p["src"] for p in pieces]
        rec["placement"] = str(rec.get("placement", "")) + " | re-placed by mkm_layout_fill (maximin, gap floor %.2f mm)" % (gap * R)
        man[0]["pieces"] = [p["pc"] for p in pieces]; man[0]["fill"] = fill
        man[0].setdefault("notes", []).append("mkm_layout_fill: smallest clearance %s mm; *_RING.stl not rewritten (recipe is the layout)" % clear_mm)
    json.dump(rec, open(os.path.join(dst_dir, "recipe.json"), "w"), indent=1)          # verdict left as the planner left it
    json.dump(man, open(os.path.join(dst_dir, "RING_MANIFEST.json"), "w", encoding="utf-8"), indent=1)
    notes = dict(plate=name, pieces=len(pieces), fill_pct=fill, min_clearance_mm=clear_mm, gap_floor_mm=gap * R, hard_defect=bool(hard),
                 layout="kept from planner (refill failed)" if hard else "refilled", one_model_plate=len(pieces) == 1,
                 band_counts=(b or {}).get("count"), shield_pairs=(b or {}).get("shield"), moves=moves,
                 leftover_events=_events_out(b, names, PL["Z0"], PL["DZ"]), note="band events are not a veto (merged rule 11:35 s.3)")
    json.dump(notes, open(os.path.join(dst_dir, "LAYOUT_NOTES.json"), "w", encoding="utf-8"), indent=1)
    if not hard:
        _png(os.path.join(dst_dir, name + "_top.png"), Rm, pl, shape, "%s  %d pcs  fill %.1f%%  min clearance %s mm" % (name, len(pieces), fill, clear_mm))
    return notes


def refill(layout_in, layout_out, gap_mm=GAP_MM, prefer_shielded=False):
    if gap_mm < GAP_MM:
        raise ValueError("gap %.2f mm is below Mike's locked floor %.2f mm" % (gap_mm, GAP_MM))
    PL = _planner(); R = PL["R"]; shape = (int(round(PL["PX"] / R)), int(round(PL["PY"] / R))); gap = int(math.ceil(gap_mm / R - 1e-9))
    names = sorted((d for d in os.listdir(layout_in) if d[:1] == "P" and d[1:].isdigit()), key=lambda d: int(d[1:]))
    plates = {}
    for nm in names:
        d = os.path.join(layout_in, nm); rec = json.load(open(os.path.join(d, "recipe.json")))
        man = json.load(open(os.path.join(d, "RING_MANIFEST.json"), encoding="utf-8")); pcs = man[0]["pieces"]
        if len(pcs) != len(rec["sources"]) or any(os.path.normcase(os.path.abspath(a["path"])) != os.path.normcase(os.path.abspath(s["path"])) for a, s in zip(pcs, rec["sources"])):
            raise ValueError("%s: RING_MANIFEST pieces and recipe sources do not match one to one" % nm)
        pieces = [dict(_piece(PL, pc), pc=dict(pc), src=dict(s)) for pc, s in zip(pcs, rec["sources"])]
        pl, b, mv, hard = _layout(pieces, shape, gap, prefer_shielded)
        plates[nm] = dict(rec=rec, man=man, pieces=pieces, pl=pl, b=b, moves=mv, hard=hard)
    # swaps between plates: only for a hard defect or a leftover band event; 1-for-1, plate count never changes (A6)
    for nm, A in plates.items():
        if not (A["hard"] or (A["b"] and any(A["b"]["count"].values()))):
            continue
        if A["hard"]:
            owner = max(range(len(A["pieces"])), key=lambda i: A["pieces"][i]["R"][0].sum())
        else:
            e = sorted(A["b"]["events"], key=lambda e: ("END", "LANDING", "START").index(e[0]))[0]; owner = e[1]
        pa = A["pieces"][owner]; aa = int(pa["R"][0].sum()); sets = {p["pc"].get("set") for p in A["pieces"]}
        cands = [(on, j) for on, B in plates.items() if on != nm for j, q in enumerate(B["pieces"]) if not A["hard"] or q["R"][0].sum() < aa]
        cands.sort(key=lambda t: (plates[t[0]]["pieces"][t[1]]["pc"].get("set") not in sets, abs(int(plates[t[0]]["pieces"][t[1]]["R"][0].sum()) - aa)))
        for on, j in cands[:SWAP_CANDIDATES]:
            B = plates[on]; na = list(A["pieces"]); nb_ = list(B["pieces"]); na[owner], nb_[j] = B["pieces"][j], pa
            ra = _layout(na, shape, gap, prefer_shielded); rb = _layout(nb_, shape, gap, prefer_shielded)
            okA = (not ra[3]) and (A["hard"] or _better(ra[1], ra[3], A["b"], A["hard"], prefer_shielded))
            okB = (not rb[3]) and (B["hard"] or _key(rb[1], 0, prefer_shielded) <= _key(B["b"], 0, prefer_shielded))
            if okA and okB:
                why = "swap with %s: %s <-> %s" % (on, pa["pc"].get("file"), B["pieces"][j]["pc"].get("file"))
                A.update(pieces=na, pl=ra[0], b=ra[1], moves=ra[2] + [why], hard=0); B.update(pieces=nb_, pl=rb[0], b=rb[1], moves=rb[2] + [why], hard=0)
                break
    out = []
    for nm, A in plates.items():                               # a plate still hard keeps the planner's layout, written down as a hard defect
        out.append(_write_plate(os.path.join(layout_in, nm), os.path.join(layout_out, nm), nm, A["pieces"], A["man"], A["rec"], A["pl"], A["b"], A["moves"], A["hard"], PL, shape, gap))
    json.dump(dict(gap_mm=gap_mm, prefer_shielded=prefer_shielded, plates=out, assumptions="see ASSUMPTIONS in mkm_layout_fill.py"),
              open(os.path.join(layout_out, "LAYOUT_NOTES.json"), "w", encoding="utf-8"), indent=1)
    for n in out:
        print("%s: %d pcs, fill %.1f%%, min clearance %s mm, events %s%s" % (n["plate"], n["pieces"], n["fill_pct"], n["min_clearance_mm"], n["band_counts"],
              " HARD DEFECT (planner layout kept)" if n["hard_defect"] else ""), flush=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="MKM plater v3 step (d): layout refill + band events")
    ap.add_argument("layout_in"); ap.add_argument("layout_out")
    ap.add_argument("--gap", type=float, default=GAP_MM); ap.add_argument("--prefer-shielded", action="store_true")
    a = ap.parse_args(); t0 = time.time()
    refill(a.layout_in, a.layout_out, gap_mm=a.gap, prefer_shielded=a.prefer_shielded)
    print("LAYOUT FILL DONE %.0fs" % (time.time() - t0))
