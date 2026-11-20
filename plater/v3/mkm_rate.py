# MKM plater v3 - step (c): per-plate rating, heat map picture, run summary. Owner: builder C. Coding pass only: NOT run on MKM yet.
# Interfaces per plater/CONTRACT.md: rate_plate(), heat_png(), summarize(); rating_from(defects, events) is the pure rating mapping.
# Binding rule: ref/specs/MERGED_RULE_go-nogo_band_20261120-1135.md. Quoted where used below.
#
# ASSUMPTIONS - verify on MKM
#  A1 Store rows: row r -> bed y = origin_y + (r+0.5)*pitch (row 0 at y min, as plate_heat.py's gfill grid); cols are mirrored
#     (store col c = unmirrored col W-1-c). origin_mm from store meta, else recipe geometry, else -screen/2 (bed centred on 0,0).
#  A2 Printable bed = 298.0 x 164.0 mm centred on the screen (planner PX, PY in ref/gpu_scan/mkm_gpupool10.py). The screen is
#     298.08 x 165.6. "Outside the bed" = cured pixel centre outside that rectangle; "touching the edge" = cured pixel in the
#     bed's outermost pixel ring (warning); cured within GAP_MM of the bed edge = warning (layout gap, not hard gate).
#  A3 Connectivity checks (islands, merges, cavities, band) run on a working grid of DS=4 printer pixels (0.184 mm), the grid
#     mkm_uvtools_gpu uses. Cured-connectivity uses "any pixel cured" in a cell (never invents an island); cavity empties use
#     "half or more cured" exactly as mkm_uvtools_gpu (parity). Outside-bed / edge checks and the heat map use full printer pixels.
#  A4 Model identity (store holds the union; no re-slicing): each recipe source's STL bounding box (vertices read once with
#     numpy, matrix applied) -> pixel box, padded one working cell. A cured cell inside exactly ONE box belongs to that model;
#     labels then spread within each layer's connected cured cells (max and min label, GPU). A component carrying two labels
#     = "two models in one cured cell". LIMITS: a component lying only where boxes overlap (or outside every box) gets no
#     label (counted as unattributed, a warning); a component touching three models reports its lowest and highest model only.
#     recipe sources[i] <-> RING_MANIFEST pieces[i] (no = i+1), as the planner writes them.
#  A5 Island ("a component that never connects ... to something that reaches the build plate"): layer 0 is on the plate; a cell
#     is plate-connected when it is cured and joins, in-plane, a cell cured over (or beside-over, 4-neighbour) a plate-connected
#     cell of the layer before (mkm_uvtools_gpu's `cur & _dil4(prev_cured)` overlap, chained). Every other cured cell is an
#     island cell, also when the region joins plate-connected geometry later (it was printed hanging); the report says whether
#     the model still has island cells at its top layer.
#  A6 Cavities follow mkm_uvtools_gpu: open = empty connected in-plane to the screen edge; hollow = other empty; vented = hollow
#     reached through earlier layers from open/vented resin; suction = hollow and not vented. Suction that never opens toward
#     the film = sealed cavity (rule (i)) or sealed pocket between parts (iii); suction that opens on a later layer = cup sealed
#     against the film over its sealed interval (rule (ii)). Small deviation from mkm_uvtools_gpu pass 2: a pocket counts as
#     opening when the cell above it is open OR vented (not only open), and the plate's last layer counts as open (same as theirs).
#     One part vs between parts: the cured cells walling the pocket carry one model label or two.
#  A7 Band (merged rule s.3) on the same working grid: the shared front is the shrinking ring from the bed edge (ring depth in
#     cells = steps, as plate_heat.py's peel ring). A cell lets go at the step equal to its ring depth. Per layer, per model:
#     f = first step it is met, l = step its last cell lets go.
#     START: nested model (its first-met cells are not the first cured cells seen from their nearest bed edge) whose first-met
#       face is not shielded on both lateral sides; a side is shielded when the nearest cured cell across empty film along the
#       face row/column is still stuck at step f. Rim models are never flagged.
#     LANDING: for facing neighbours (nearest cured cells across empty film along a row or column) with the neighbour larger
#       (cured cells on this layer): at step t = neighbour's l, the smaller model is held (t < f), free (t >= its l), touched at
#       one front cell (1 cell let go) or mid-peel (event).
#     END: the layer's last front step that still meets cured cells holds cells of two or more models (event).
#     Identical repeated layers count once: consecutive layers with the same event (class, model, other model) are one event,
#     reported with its layer interval.
#  A8 UVtools results only point at layers. A hard-gate defect is what the store geometry shows. UVtools hits the geometry does
#     not confirm are listed (uvtools_unconfirmed) and do not change the rating. "Survives repair": repaired-file GPU scan
#     (uv/P{n}/gpu_issues.json per_layer) if present, else UVtools issues_repaired.json (count 0 -> no; hit in the interval -> yes;
#     count > 0 without layer detail or missing/failed file -> unknown). Merges, outside-bed and pockets between parts always
#     survive (UVtools does not move models; rule s.4(iii): never drill a model to fix a gap).
#  A9 issues_*.json 'layers' detail is read tolerantly: {type: [layer,...] | [[a,b],...] | [{"layer"|"start","end"}...]} or a
#     list of {"type","layer"} rows. Layer numbers are the same 0-based indices as the store.
#  A10 Rating mapping is a PROPOSAL (contract). Verified defect repaired away, edge/gap warnings and unattributed cells are
#     written as notes (LIKELY WITH NOTES), not as RISKY.
import os, sys, json, glob, re, math, time, hashlib

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import mkm_v3_config as C            # puts REF_TREE (gpu_scan) on sys.path; no torch here

BED_MM = (298.0, 164.0)              # A2
DS = 4                               # A3
BATCH = 32
RATINGS = ("LIKELY", "LIKELY WITH NOTES", "RISKY", "WILL NOT PRINT")
RATING_BASIS = "PROPOSAL - mapping not yet approved by Mike"
STOP_LINE = "STOPPED - waiting for Mike. Nothing was sent to any printer."
UV_TYPE = {"island": ("Island",), "sealed_cavity": ("ResinTrap",), "film_cup": ("SuctionCup",),
           "outside_bed": ("TouchingBounds",)}
ALWAYS_SURVIVES = ("merge", "outside_bed", "pocket_between_parts", "cup_between_parts")
ORDER = {"defect": 0, "END": 1, "LANDING": 2, "START": 3, "warning": 4}      # merged rule s.3 comparison order


# ------------------------------------------------------------------ rating mapping (pure, no torch)
def rating_from(defects, events):
    """Contract mapping (PROPOSAL). WILL NOT PRINT = a verified hard-gate defect survives repair; RISKY = a hard-gate hit
    that could not be verified either way, or a shared last loop (END); LIKELY WITH NOTES = hard gate clean, notes or band
    events left; LIKELY = nothing."""
    hard = [d for d in defects if not d.get("warning")]
    if any(d.get("verified") is True and d.get("survives_repair") is True for d in hard):
        return "WILL NOT PRINT"
    if any(d.get("verified") is not True or d.get("survives_repair") is None for d in hard):
        return "RISKY"
    if any(e.get("event") == "END" for e in events):
        return "RISKY"
    if defects or events:
        return "LIKELY WITH NOTES"
    return "LIKELY"


def _reason(x):
    who = "model %s" % x.get("model", "?") + (" (%s)" % x["set"] if x.get("set") else "")
    if x.get("other"):
        who += " + model %s" % x["other"]
    lay = x.get("layers") or [None, None]
    z = " z %.2f-%.2f mm" % tuple(x["z_mm"]) if x.get("z_mm") else ""
    what = x.get("event") or x.get("defect")
    tag = "" if "event" in x else (" [warning]" if x.get("warning") else " [verified=%s, survives repair=%s]" % (x.get("verified"), x.get("survives_repair")))
    return "%s, layers %s-%s%s: %s%s%s" % (who, lay[0], lay[1], z, what, tag, (" - " + x["detail"]) if x.get("detail") else "")


def _rank(x):
    if "event" in x:
        return (ORDER[x["event"]], x["layers"][0])
    if x.get("warning"):
        return (ORDER["warning"], x["layers"][0] if x.get("layers") else 0)
    return (ORDER["defect"] - (1 if x.get("survives_repair") is True and x.get("verified") is True else 0), x["layers"][0] if x.get("layers") else 0)


def reasons_from(defects, events):
    return [_reason(x) for x in sorted(list(defects) + list(events), key=_rank)]


# ------------------------------------------------------------------ small readers
def _load(p, default=None):
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def _uv_ok(d):
    return isinstance(d, dict) and str(d.get("status", "")).lower() in ("ok", "done", "success", "pass", "true")


def _uv_layers(d, typ):
    """A9: list of layer ints for one UVtools type, or None when the file has no layer detail for it."""
    L = (d or {}).get("layers")
    if L is None:
        return None
    out = []
    def add(v):
        if isinstance(v, (int, float)):
            out.append(int(v))
        elif isinstance(v, (list, tuple)) and len(v) == 2 and all(isinstance(a, (int, float)) for a in v):
            out.extend(range(int(v[0]), int(v[1]) + 1))
        elif isinstance(v, dict):
            if "layer" in v:
                out.append(int(v["layer"]))
            elif "start" in v:
                out.extend(range(int(v["start"]), int(v.get("end", v["start"])) + 1))
    if isinstance(L, dict):
        rows = L.get(typ)
        if rows is None:
            return None
        if isinstance(rows, dict):                          # mkm_uv_check.parse_issues shape: {layers_affected, first_layer, last_layer, layer_list}
            rows = rows.get("layer_list")
            if rows is None:
                return None
        for v in rows:
            add(v)
    elif isinstance(L, list):
        for v in L:
            if isinstance(v, dict) and str(v.get("type", "")).lower() == typ.lower():
                add(v)
    else:
        return None
    return sorted(set(out))


def _stl_bbox(path):
    """Binary STL vertex min/max (float32 widened), read once with numpy; ASCII fallback. No slicing, no raster."""
    import numpy as np
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        f.seek(80); n = int.from_bytes(f.read(4), "little")
    if 84 + 50 * n == size:
        dt = np.dtype([("n", "<f4", (3,)), ("v", "<f4", (3, 3)), ("a", "<u2")])
        v = np.memmap(path, dtype=dt, mode="r", offset=84, shape=(n,))["v"].reshape(-1, 3)
        v = v[np.isfinite(v).all(1)]
    else:
        v = np.array([[float(t) for t in ln.split()[1:4]] for ln in open(path, errors="ignore") if ln.strip().startswith("vertex")])
    return v.min(0).astype(float), v.max(0).astype(float)


def _model_boxes(recipe, cache_path):
    """Per source: bbox in bed mm after the recipe's 4x4 matrix (8 corners transformed; quarter turns keep it exact)."""
    import numpy as np
    cache = _load(cache_path, {}) or {}
    out = []
    for s in recipe["sources"]:
        key = s.get("sha256") or s["path"]
        if key not in cache:
            lo, hi = _stl_bbox(s["path"])
            cache[key] = [lo.tolist(), hi.tolist()]
        lo, hi = (np.array(a) for a in cache[key])
        M = np.array(s["matrix"], float)
        cs = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
        t = cs @ M[:3, :3].T + M[:3, 3]
        out.append((t.min(0), t.max(0)))
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    json.dump(cache, open(cache_path, "w"), indent=0)
    return out


def _intervals(flags):
    """Runs of True in a 1-D bool array -> [(a, b)] inclusive (identical repeated layers count once)."""
    import numpy as np
    f = np.concatenate([[False], np.asarray(flags, bool), [False]]).astype(np.int8)
    d = np.diff(f)
    return list(zip(np.nonzero(d == 1)[0].tolist(), (np.nonzero(d == -1)[0] - 1).tolist()))


# ------------------------------------------------------------------ GPU helpers (torch imported lazily)
def _T():
    import torch, torch.nn.functional as F
    return torch, F


def _UG():
    import mkm_uvtools_gpu as UG         # from mkm_v3_config.REF_TREE (gpu_scan); reused, not copied
    return UG


def _stack(x, S):
    """(B,H,W) bool -> (B*S, W) bool with >=1 empty spacer row per layer, so mkm_uvtools_gpu's 2-D _open/_flood/_dil4
    run on a whole batch at once. Spacer rows are empty and touch the side edges, i.e. open exactly like each layer's own
    top/bottom edge row, so no layer gains openness it lacks alone; cured/hollow floods never enter them."""
    torch, _ = _T()
    B, H, W = x.shape
    y = torch.zeros((B, S, W), dtype=torch.bool, device=x.device)
    y[:, :H] = x
    return y.reshape(B * S, W)


def _unstack(y, B, S, H):
    return y.reshape(B, S, -1)[:, :H]


def _prop(x, mode):
    """4-neighbour max (mode='max') or min (mode='min') of a (B,H,W) float tensor, itself included."""
    torch, F = _T()
    s = 1.0 if mode == "max" else -1.0
    xf = (x * s)[:, None]
    y = torch.maximum(F.max_pool2d(xf, (3, 1), 1, (1, 0)), F.max_pool2d(xf, (1, 3), 1, (0, 1)))[:, 0]
    return y * s


def _dil4b(x):
    torch, F = _T()
    xf = x[:, None].float()
    return (torch.maximum(F.max_pool2d(xf, (3, 1), 1, (1, 0)), F.max_pool2d(xf, (1, 3), 1, (0, 1)))[:, 0] > 0)


def _vflood(allowed, src_prev, extra=None, vertical_dil=False, down=False, cap=20000, check=8):
    """Volume flood over a (B,H,W) batch: a cell joins when allowed and (4-neighbour in-plane of the set, or over the set /
    the fixed source on the layer before [after if down]). src_prev = (H,W) bool of the layer before the batch.
    extra = (B,H,W) fixed per-layer source (e.g. 'open' of the layer before is passed as part of shift). Converges per batch;
    one scalar sync per `check` iterations of the batch, never per layer."""
    torch, _ = _T()
    A = torch.zeros_like(allowed)
    last = -1; it = 0
    while it < cap:
        for _ in range(check):
            prev = A if extra is None else (A | extra)
            if down:
                sh = torch.cat([prev[1:], src_prev[None]], 0)
            else:
                sh = torch.cat([src_prev[None], prev[:-1]], 0)
            if vertical_dil:
                sh = _dil4b(sh)
            A = allowed & (_dil4b(A) | sh)
        it += check
        c = int(A.sum())
        if c == last:
            break
        last = c
    return A


def _labels(cur, owner, cap=20000, check=8):
    """Spread model labels inside each layer's cured components: returns (lab_max, lab_min) int (B,H,W); 0 = unlabelled."""
    torch, _ = _T()
    big = float(int(owner.max()) + 1)
    seed = cur & (owner[None] > 0)
    mx = torch.where(seed, owner[None].float(), torch.zeros((), device=cur.device)).expand_as(cur).clone()
    mn = torch.where(seed, owner[None].float(), torch.full((), big, device=cur.device)).expand_as(cur).clone()
    it = 0
    zero = torch.zeros((), device=cur.device); bigt = torch.full((), big, device=cur.device)
    while it < cap:
        mx0, mn0 = mx, mn
        for _ in range(check):                                    # `check` real propagation steps per host sync
            mx = torch.where(cur, _prop(mx, "max"), zero)
            mn = torch.where(cur, _prop(mn, "min"), bigt)
        it += check
        if torch.equal(mx, mx0) and torch.equal(mn, mn0):
            break
    mn = torch.where(mx > 0, mn, torch.zeros((), device=cur.device))
    return mx.long(), mn.long()


def _first_label(lab, dim, rev):
    """Label of the first cured cell along `dim` (from the high end if rev); 0 where the line is empty."""
    torch, _ = _T()
    c = (lab > 0)
    if rev:
        c = c.flip(dim); lab = lab.flip(dim)
    i = torch.argmax(c.to(torch.uint8), dim=dim, keepdim=True)
    return torch.gather(lab, dim, i)


def _across(cur, s, dim, rev):
    """For every cell: ring step of the nearest cured cell across empty film, on the low side along `dim` (high side if rev),
    measured from the cell's own run of contiguous cured cells; -1 if none. Also that cell's label via the caller."""
    torch, _ = _T()
    x = cur.flip(dim) if rev else cur
    n = x.shape[dim]
    shape = [1] * x.dim(); shape[dim] = n
    pos = torch.arange(n, device=x.device).view(shape).expand_as(x)
    left = torch.cat([torch.zeros_like(x.narrow(dim, 0, 1)), x.narrow(dim, 0, n - 1)], dim)
    start = torch.where(x & ~left, pos, torch.full_like(pos, -1)).cummax(dim).values          # run start of each cell
    lastc = torch.where(x, pos, torch.full_like(pos, -1)).cummax(dim).values                  # last cured at or before
    before = torch.cat([torch.full_like(lastc.narrow(dim, 0, 1), -1), lastc.narrow(dim, 0, n - 1)], dim)
    nb = torch.gather(before, dim, start.clamp(min=0))                                           # cured cell before the run
    nb = torch.where(start >= 0, nb, torch.full_like(nb, -1))
    if rev:
        nb = nb.flip(dim); nb = torch.where(nb >= 0, n - 1 - nb, nb)
    return nb


def _band_batch(lab, s, sdir, M):
    """A7 events for a (B,H,W) label batch; returns dict of small tensors (B, M+1[, M+1])."""
    torch, _ = _T()
    B, H, W = lab.shape; K = M + 1; dev = lab.device
    cur = lab > 0
    sB = s[None].expand(B, H, W)
    bidx = torch.arange(B, device=dev).view(B, 1, 1).expand(B, H, W)
    key = (bidx * K + lab)[cur]; sv = sB[cur]
    big = int(s.max()) + 1
    f = torch.full((B * K,), big, dtype=torch.long, device=dev).scatter_reduce(0, key, sv.long(), "amin")
    l = torch.full((B * K,), -1, dtype=torch.long, device=dev).scatter_reduce(0, key, sv.long(), "amax")
    area = torch.bincount(key, minlength=B * K)
    present = area > 0
    fC = f.view(B, K).gather(1, lab.view(B, -1)).view(B, H, W)
    first = cur & (sB == fC)
    # rim: first-met cells that are the first cured cells seen from their nearest bed edge (dir 0 x-low,1 x-high,2 y-low,3 y-high)
    fl = [_first_label(lab, 2, False).expand(B, H, W), _first_label(lab, 2, True).expand(B, H, W),
          _first_label(lab, 1, False).expand(B, H, W), _first_label(lab, 1, True).expand(B, H, W)]
    seen = torch.zeros_like(cur)
    for d in range(4):
        seen |= (sdir[None] == d) & (lab == fl[d])
    rim = torch.zeros(B * K, dtype=torch.long, device=dev).scatter_reduce(0, (bidx * K + lab)[first], seen[first].long(), "amax")
    # lateral shields of the first-met face: along x for y-fronts, along y for x-fronts
    flat_s = sB
    def shield(dim, rev):
        nb = _across(cur, s, dim, rev)
        ok = nb >= 0
        g = torch.gather(flat_s, dim, nb.clamp(min=0))
        return ok & (g >= fC), nb
    sxa, nxa = shield(2, False); sxb, nxb = shield(2, True); sya, nya = shield(1, False); syb, nyb = shield(1, True)
    yfront = sdir[None] >= 2
    sa = torch.where(yfront, sxa, sya); sb = torch.where(yfront, sxb, syb)
    kf = (bidx * K + lab)[first]
    sha = torch.zeros(B * K, dtype=torch.long, device=dev).scatter_reduce(0, kf, sa[first].long(), "amax")
    shb = torch.zeros(B * K, dtype=torch.long, device=dev).scatter_reduce(0, kf, sb[first].long(), "amax")
    start = present & (rim == 0) & ~((sha > 0) & (shb > 0))
    start = start.view(B, K); start[:, 0] = False
    # facing neighbours (nearest cured cell across empty film along rows and columns)
    adj = torch.zeros(B * K * K, dtype=torch.bool, device=dev)
    for nb, dim in ((nxa, 2), (nya, 1)):
        other = torch.gather(lab, dim, nb.clamp(min=0))
        m = cur & (nb >= 0) & (other > 0) & (other != lab)
        adj[(bidx * K * K + lab * K + other)[m]] = True
        adj[(bidx * K * K + other * K + lab)[m]] = True
    adj = adj.view(B, K, K)
    # LANDING: at t = l(n) for a larger facing neighbour n, the smaller model m is mid-peel (more than one cell let go, not all)
    S1 = big + 1
    hist = torch.bincount((bidx * K + lab)[cur] * S1 + sv.long(), minlength=B * K * S1).view(B, K, S1).cumsum(2)
    ar = area.view(B, K); lv = l.view(B, K); fv = f.view(B, K)
    t = lv.clamp(min=0)[:, None, :].expand(B, K, K)                          # [b, m, n] = l(n)
    released = hist.gather(2, t)                                             # cells of m let go by step l(n)
    larger = ar[:, None, :] > ar[:, :, None]
    land = adj & larger & (released > 1) & (released < ar[:, :, None]) & (lv[:, None, :] >= 0)
    land[:, 0, :] = False; land[:, :, 0] = False
    # END: last front step that still meets cured cells, shared by two or more models
    dstar = torch.where(present.view(B, K), lv, torch.full_like(lv, -1))
    dstar[:, 0] = -1
    dmax = dstar.max(1, keepdim=True).values
    endm = (dstar == dmax) & (dmax >= 0)
    endm[:, 0] = False
    return dict(start=start, land=land, end=endm, fstep=fv.clamp(max=big), lstep=lv)


# ------------------------------------------------------------------ the plate pass
def _scan(store, recipe, boxes, device, batch, log):
    """One pass over the store in GPU batches. Returns host numpy per-layer arrays + heat (display grid)."""
    import numpy as np
    torch, F = _T(); UG = _UG()
    meta = store.meta or {}
    Wf, Hf = tuple(meta.get("res", C.RES)); pmm = tuple(meta.get("pixel_mm", C.PIXEL_MM))
    org = meta.get("origin_mm") or (recipe.get("geometry") or {}).get("origin_mm") or [-C.SCREEN_MM[0] / 2, -C.SCREEN_MM[1] / 2]
    N = store.n_layers; M = len(boxes); K = M + 1
    H, W = Hf // DS, Wf // DS; S = int(math.ceil((H + 1) / 16.0) * 16)
    dev = torch.device(device)
    # full-res coordinates (store col c is mirrored: bed x of col c uses unmirrored col Wf-1-c)  A1
    xs = org[0] + (Wf - 1 - torch.arange(Wf, device=dev, dtype=torch.float64) + 0.5) * pmm[0]
    ys = org[1] + (torch.arange(Hf, device=dev, dtype=torch.float64) + 0.5) * pmm[1]
    bx, by = BED_MM[0] / 2, BED_MM[1] / 2
    inx = (xs.abs() <= bx); iny = (ys.abs() <= by)
    bed = iny[:, None] & inx[None, :]
    edge = bed & ~(((xs.abs() <= bx - pmm[0])[None, :]) & ((ys.abs() <= by - pmm[1])[:, None]))
    gapb = bed & ~(((xs.abs() <= bx - C.GAP_MM)[None, :]) & ((ys.abs() <= by - C.GAP_MM)[:, None]))
    # owner maps (unique box -> model id 1..M), full res and working grid  A4
    own_f = torch.zeros((Hf, Wf), dtype=torch.int16, device=dev); cnt_f = torch.zeros((Hf, Wf), dtype=torch.int16, device=dev)
    xs_w = xs.view(W, DS).mean(1); ys_w = ys.view(H, DS).mean(1)
    own_w = torch.zeros((H, W), dtype=torch.long, device=dev); cnt_w = torch.zeros((H, W), dtype=torch.int16, device=dev)
    pad = DS * pmm[0]
    for i, (lo, hi) in enumerate(boxes):
        mf = ((xs >= lo[0] - pad) & (xs <= hi[0] + pad))[None, :] & ((ys >= lo[1] - pad) & (ys <= hi[1] + pad))[:, None]
        mw = ((xs_w >= lo[0] - pad) & (xs_w <= hi[0] + pad))[None, :] & ((ys_w >= lo[1] - pad) & (ys_w <= hi[1] + pad))[:, None]
        own_f[mf] = i + 1; cnt_f += mf.to(torch.int16); own_w[mw] = i + 1; cnt_w += mw.to(torch.int16)
    own_f = torch.where(cnt_f == 1, own_f, torch.zeros_like(own_f)).long(); own_w = torch.where(cnt_w == 1, own_w, torch.zeros_like(own_w))
    # ring depth (steps from the bed edge) and nearest-edge direction on the working grid  A7
    dx0 = ((xs_w + bx) / (DS * pmm[0])).clamp(min=0); dx1 = ((bx - xs_w) / (DS * pmm[0])).clamp(min=0)
    dy0 = ((ys_w + by) / (DS * pmm[1])).clamp(min=0); dy1 = ((by - ys_w) / (DS * pmm[1])).clamp(min=0)
    dd = torch.stack([dx0[None, :].expand(H, W), dx1[None, :].expand(H, W), dy0[:, None].expand(H, W), dy1[:, None].expand(H, W)])
    s = dd.min(0).values.floor().long(); sdir = dd.argmin(0)
    def _static(mask):                                            # flat pixel list of a fixed full-res region + its owners
        idx = mask.reshape(-1).nonzero(as_tuple=False)[:, 0]
        return idx, own_f.reshape(-1)[idx]
    st_out, st_edge, st_gap = _static(~bed), _static(edge), _static(gapb)
    del bed, edge, gapb, cnt_f
    heat = torch.zeros((Hf, Wf), dtype=torch.int32, device=dev)
    keys = ("cured", "float", "outside", "edge", "gap", "unattr", "merge", "start", "land", "end", "fstep", "lstep")
    acc = {k: [] for k in keys}
    sparse = []                                                   # suction cells: (flat index, escape, bmin, bmax)
    A_prev = torch.ones((H, W), dtype=torch.bool, device=dev)     # the build plate under layer 0
    VO_prev = torch.zeros((H, W), dtype=torch.bool, device=dev)   # open|vented of the layer before (none under layer 0)
    S_prev = None
    for i0 in range(0, N, batch):
        i1 = min(N, i0 + batch); B = i1 - i0
        full = store.read_batch(i0, i1, device=dev).bool()
        heat += full.sum(0, dtype=torch.int32)
        fullv = full.reshape(B, -1)
        def per_model(st):                                         # cured pixels per model inside one fixed region
            idx, own = st
            o = torch.zeros((B, K), dtype=torch.int32, device=dev)
            if idx.numel():
                o.index_add_(1, own, fullv[:, idx].to(torch.int32))
            return o
        outside = per_model(st_out); edg = per_model(st_edge); gp = per_model(st_gap)
        cnt = full.view(B, H, DS, W, DS).sum(dim=(2, 4), dtype=torch.uint8)   # cured printer pixels per working cell (no float copy)
        cur = cnt > 0                                              # A3 connectivity grid (= max-pool > 0)
        half = cnt.to(torch.int32) * 2 >= DS * DS                  # A3 cavity grid (= avg-pool >= 0.5, as mkm_uvtools_gpu)
        del cnt, fullv, full
        lab, labmin = _labels(cur, own_w)
        bid = torch.arange(B, device=dev).view(B, 1, 1).expand(B, H, W)
        cured = torch.bincount((bid * K + lab)[cur], minlength=B * K).view(B, K)
        mg = cur & (labmin > 0) & (labmin != lab)
        merge = torch.bincount((bid * K * K + labmin * K + lab)[mg], minlength=B * K * K).view(B, K, K)
        # islands  A5
        A = _vflood(cur, A_prev, vertical_dil=True)
        fl = cur & ~A
        flo = torch.bincount((bid * K + lab)[fl], minlength=B * K).view(B, K)
        A_prev = A[-1]
        # cavities  A6: open via mkm_uvtools_gpu._open on the stacked batch
        empty = ~half
        op = _unstack(UG._open(_stack(empty, S)), B, S, H)
        hol = empty & ~op
        V = _vflood(hol, VO_prev, extra=op)
        VO_prev = (op | V)[-1]
        Sx = hol & ~V
        if S_prev is not None:                                    # escape of the previous batch's last layer
            sp_i, sp_l, sp_bmin, sp_bmax = S_prev
            ok = (empty[0] & ~Sx[0]).view(-1)[sp_i]
            sparse.append((sp_l * H * W + sp_i, ok, sp_bmin, sp_bmax))
        esc = Sx[:-1] & empty[1:] & ~Sx[1:]
        if Sx.any():
            big = float(K)
            labf = torch.where(cur, lab.float(), torch.zeros((), device=dev)); lmf = torch.where(cur & (labmin > 0), labmin.float(), torch.full((), big, device=dev))
            bmx = torch.where(Sx, _prop(labf, "max"), torch.zeros((), device=dev)); bmn = torch.where(Sx, _prop(lmf, "min"), torch.full((), big, device=dev))
            for _ in range(20000 // 8):
                bmx2, bmn2 = bmx, bmn
                for _ in range(8):
                    bmx2 = torch.where(Sx, _prop(bmx2, "max"), bmx2); bmn2 = torch.where(Sx, _prop(bmn2, "min"), bmn2)
                if torch.equal(bmx2, bmx) and torch.equal(bmn2, bmn):
                    break
                bmx, bmn = bmx2, bmn2
            bmn = torch.where(bmn >= big, torch.zeros_like(bmn), bmn)
            nz = Sx[:-1].nonzero(as_tuple=False)
            if nz.numel():
                fi = nz[:, 1] * W + nz[:, 2]; li = nz[:, 0] + i0
                sparse.append((li * H * W + fi, esc[nz[:, 0], nz[:, 1], nz[:, 2]], bmn[nz[:, 0], nz[:, 1], nz[:, 2]].long(), bmx[nz[:, 0], nz[:, 1], nz[:, 2]].long()))
            last = Sx[-1].view(-1).nonzero(as_tuple=False)[:, 0]
            S_prev = (last, torch.tensor(i1 - 1, device=dev), bmn[-1].view(-1)[last].long(), bmx[-1].view(-1)[last].long()) if last.numel() else None
        else:
            S_prev = None
        band = _band_batch(lab, s, sdir, M)
        unat = (cur & (lab == 0)).sum((1, 2))
        res = dict(cured=cured, float=flo, outside=outside, edge=edg, gap=gp, unattr=unat, merge=merge, start=band["start"],
                   land=band["land"], end=band["end"], fstep=band["fstep"], lstep=band["lstep"])
        flat = torch.cat([res[k].reshape(-1).long() for k in keys])            # ONE host transfer per batch
        host = flat.cpu().numpy()
        o = 0
        for k in keys:
            n = res[k].numel(); acc[k].append(host[o:o + n].reshape(tuple(res[k].shape))); o += n
        if log and (i0 // batch) % 10 == 0:
            log("rate scan %d/%d" % (i1, N))
    if S_prev is not None:                                        # last layer of the plate counts as opening (as mkm_uvtools_gpu)
        sp_i, sp_l, sp_bmin, sp_bmax = S_prev
        sparse.append((sp_l * H * W + sp_i, torch.ones_like(sp_i, dtype=torch.bool), sp_bmin, sp_bmax))
    out = {k: np.concatenate(v, 0) for k, v in acc.items()}
    # reverse pass on the suction cells only (film -> plate): cup if it reaches an escape going toward later layers  A6
    pocket = np.zeros((N, 2, K, K), np.int64)                    # [layer, trap0/cup1, bmin, bmax]
    if sparse:
        gi = torch.cat([a[0] for a in sparse]); ge = torch.cat([a[1] for a in sparse]); gmn = torch.cat([a[2] for a in sparse]); gmx = torch.cat([a[3] for a in sparse])
        gl = gi // (H * W)
        nxt = torch.zeros((H, W), dtype=torch.bool, device=dev)
        for j1 in range(N, 0, -batch):
            j0 = max(0, j1 - batch); B = j1 - j0
            sel = (gl >= j0) & (gl < j1)
            if not bool(sel.any()):
                nxt = torch.zeros((H, W), dtype=torch.bool, device=dev); continue
            loc = gi[sel] - j0 * H * W
            Sd = torch.zeros(B * H * W, dtype=torch.bool, device=dev); Sd[loc] = True; Sd = Sd.view(B, H, W)
            Ed = torch.zeros(B * H * W, dtype=torch.bool, device=dev); Ed[loc] = ge[sel]; Ed = Ed.view(B, H, W)
            Acup = torch.zeros_like(Sd); last = -1
            for _ in range(20000 // 8):
                for _ in range(8):
                    sh = torch.cat([Acup[1:], nxt[None]], 0)
                    Acup = Sd & (_dil4b(Acup) | Ed | sh)
                c = int(Acup.sum())
                if c == last:
                    break
                last = c
            nxt = Acup[0]
            cls = Acup.view(-1)[loc].long()
            key = ((gl[sel] - j0) * 2 + cls) * K * K + gmn[sel] * K + gmx[sel]
            h = torch.bincount(key, minlength=B * 2 * K * K).view(B, 2, K, K).cpu().numpy()    # one host transfer per batch
            pocket[j0:j1] = h
    out["pocket"] = pocket
    heat_d = F.avg_pool2d(heat[None, None].float(), 6)[0, 0].cpu().numpy()                  # display only: once per plate
    return out, heat_d, dict(H=H, W=W, Hf=Hf, Wf=Wf, pitch=pmm, origin=org)


# ------------------------------------------------------------------ defects + events from the host arrays
def _z(a, b):
    return [round(a * C.LAYER_MM, 3), round((b + 1) * C.LAYER_MM, 3)]


def _defects_events(out, man_pieces, recipe):
    import numpy as np
    N, K = out["cured"].shape; M = K - 1
    def who(m):
        p = man_pieces[m - 1] if 0 < m <= len(man_pieces) else {}
        return dict(model=m, set=p.get("set", ""), file=p.get("file") or os.path.basename(recipe["sources"][m - 1]["path"]))
    defects, events, models = [], [], []
    present = out["cured"] > 0
    for m in range(1, K):
        lay = np.nonzero(present[:, m])[0]
        models.append(dict(who(m), first_layer=int(lay[0]) if len(lay) else None, last_layer=int(lay[-1]) if len(lay) else None,
                           cured_cells_total=int(out["cured"][:, m].sum())))
    def add_runs(arr, base, lst):
        for a, b in _intervals(arr):
            d = dict(base); d["layers"] = [a, b]; d["z_mm"] = _z(a, b); lst.append(d)
    for m in range(K):
        mm = who(m) if m else dict(model=0, set="unattributed")
        add_runs(out["outside"][:, m] > 0, dict(mm, defect="outside_bed", verified=True, survives_repair=True, detail="cured cells outside the 298 x 164 mm bed"), defects)
        add_runs((out["edge"][:, m] > 0) & ~(out["outside"][:, m] > 0), dict(mm, defect="edge_touch", warning=True, detail="cured cells touch the bed edge"), defects)
        add_runs((out["gap"][:, m] > 0) & ~(out["edge"][:, m] > 0) & ~(out["outside"][:, m] > 0), dict(mm, defect="inside_edge_gap", warning=True, detail="cured within the %.0f mm edge gap" % C.GAP_MM), defects)
        if m == 0:
            continue
        top = models[m - 1]["last_layer"]
        for a, b in _intervals(out["float"][:, m] > 0):
            defects.append(dict(mm, defect="island", verified=True, survives_repair=None, layers=[a, b], z_mm=_z(a, b),
                                detail="cured with no chain down to the build plate" + ("; still hanging at the model's top layer" if top is not None and b >= top else "; later joined plate-connected geometry")))
    for a, b in _intervals(out["float"][:, 0] > 0):
        defects.append(dict(model=0, set="unattributed", defect="island", verified=False, survives_repair=None, layers=[a, b], z_mm=_z(a, b), detail="island cells not attributable to one model box"))
    for a, b in _intervals(out["unattr"] > 0):
        defects.append(dict(model=0, set="unattributed", defect="unattributed_cells", warning=True, layers=[a, b], z_mm=_z(a, b), detail="cured cells only in overlapping model boxes or outside every box (A4 limit)"))
    mg = out["merge"]
    for m in range(1, K):
        for n in range(m + 1, K):
            for a, b in _intervals(mg[:, m, n] > 0):
                defects.append(dict(who(m), other=n, defect="merge", verified=True, survives_repair=True, layers=[a, b], z_mm=_z(a, b), detail="two models in one cured component"))
    pk = out["pocket"]
    for cup, name1, name2 in ((0, "sealed_cavity", "pocket_between_parts"), (1, "film_cup", "cup_between_parts")):
        for m in range(0, K):
            for n in range(m, K):
                arr = pk[:, cup, m, n] > 0
                if not arr.any():
                    continue
                for a, b in _intervals(arr):
                    if m == n and m > 0:
                        defects.append(dict(who(m), defect=name1, verified=True, survives_repair=None, layers=[a, b], z_mm=_z(a, b),
                                            detail="hollow walled in on its layer, not vented through earlier layers" + (", opens later toward the film" if cup else ", never opens")))
                    elif m > 0 and n > 0:
                        defects.append(dict(who(m), other=n, defect=name2, verified=True, survives_repair=True, layers=[a, b], z_mm=_z(a, b),
                                            detail="sealed pocket walled by two models (move/rotate/swap, never drill)"))
                    else:
                        defects.append(dict(who(n) if n else dict(model=0, set="unattributed"), defect=name1, verified=False, survives_repair=None, layers=[a, b], z_mm=_z(a, b),
                                            detail="sealed pocket whose walls could not be attributed to a model"))
    for m in range(1, K):
        for a, b in _intervals(out["start"][:, m] > 0):
            events.append(dict(who(m), event="START", layers=[a, b], z_mm=_z(a, b), detail="nested model met by the front on an open side (not shielded on two sides)"))
        for n in range(1, K):
            for a, b in _intervals(out["land"][:, m, n] > 0):
                events.append(dict(who(m), other=n, event="LANDING", layers=[a, b], z_mm=_z(a, b), detail="mid-peel on an open edge when larger neighbour %d lets go" % n))
    endm = out["end"] > 0
    shared = endm.sum(1) >= 2
    for a, b in _intervals(shared):
        ms = sorted(set(np.nonzero(endm[a:b + 1].any(0))[0].tolist()))
        events.append(dict(who(ms[0]), other=",".join(str(x) for x in ms[1:]), event="END", layers=[a, b], z_mm=_z(a, b), detail="last loop shared by models %s" % ms))
    return defects, events, models


def _verify_uv(defects, raw, rep, gpu_rep, n_layers):
    """A8: survives-repair per defect, and the UVtools hits the geometry does not confirm."""
    import numpy as np
    per = (gpu_rep or {}).get("per_layer") or {}
    gkey = {"island": "island_mm2", "sealed_cavity": "trap_mm2", "film_cup": "cup_mm2"}
    for d in defects:
        if d.get("warning") or d.get("verified") is not True:
            continue
        if d["defect"] in ALWAYS_SURVIVES:
            d["survives_repair"] = True; continue
        a, b = d["layers"]
        g = per.get(gkey.get(d["defect"], ""))
        if g:
            d["survives_repair"] = bool(np.any(np.asarray(g[a:b + 1]) > 0)); d["repair_evidence"] = "gpu_issues.json on the repaired file"; continue
        if not _uv_ok(rep):
            d["survives_repair"] = None; d["repair_evidence"] = "no usable issues_repaired.json"; continue
        types = UV_TYPE.get(d["defect"], ())
        cnt = sum(int((rep.get("counts") or {}).get(t, 0) or 0) for t in types)
        if cnt == 0:
            d["survives_repair"] = False; d["repair_evidence"] = "UVtools after repair: 0 %s" % "/".join(types); continue
        lays = [x for t in types for x in (_uv_layers(rep, t) or [])]
        detail = [t for t in types if _uv_layers(rep, t) is not None]
        d["survives_repair"] = (any(a <= x <= b for x in lays) if detail else None)
        d["repair_evidence"] = "UVtools after repair: %d %s%s" % (cnt, "/".join(types), "" if detail else " (no layer detail)")
    unconf = []
    if _uv_ok(raw):
        for cls, types in UV_TYPE.items():
            for t in types:
                n = int((raw.get("counts") or {}).get(t, 0) or 0)
                if not n:
                    continue
                lays = _uv_layers(raw, t)
                mine = [d for d in defects if d["defect"].startswith(cls.split("_")[0]) or d["defect"] == cls or
                        (cls == "sealed_cavity" and d["defect"] == "pocket_between_parts") or (cls == "film_cup" and d["defect"] == "cup_between_parts") or
                        (cls == "outside_bed" and d["defect"] in ("edge_touch",))]
                if lays is None:
                    if not mine:
                        unconf.append(dict(type=t, count=n, layers=None, note="no geometry hit of this class on any layer"))
                    continue
                miss = [x for x in lays if not any(d["layers"][0] <= x <= d["layers"][1] for d in mine)]
                if miss:
                    unconf.append(dict(type=t, count=n, layers=[[a, b] for a, b in _intervals(np.isin(np.arange(n_layers), miss))]))
    return unconf


# ------------------------------------------------------------------ public API
def rate_plate(plate_dir_layout, store_dir, uv_dir, out_dir, device=None, batch=BATCH, log=None):
    """Rate one plate; always writes out_dir/rating.json and P{n}_heat.png; returns the rating dict."""
    t0 = time.time()
    os.makedirs(out_dir, exist_ok=True)
    pid = os.path.basename(os.path.normpath(plate_dir_layout))
    recipe = _load(os.path.join(plate_dir_layout, "recipe.json"), {}) or {}
    man = _load(os.path.join(plate_dir_layout, "RING_MANIFEST.json"), [{}]) or [{}]
    man = man[0] if isinstance(man, list) and man else (man if isinstance(man, dict) else {})
    pieces = man.get("pieces", [])
    raw = _load(os.path.join(uv_dir, "issues_raw.json")); rep = _load(os.path.join(uv_dir, "issues_repaired.json"))
    gpu_rep = _load(os.path.join(uv_dir, "gpu_issues.json"))
    r = dict(plate=pid, rating_basis=RATING_BASIS, pieces=len(recipe.get("sources", [])), height_mm=man.get("height"),
             attribution_method="A4: recipe matrix x STL bounding box -> pixel boxes; unique-box cured cells seed model labels that "
                                "spread within each layer's connected cured cells on the GPU; two labels in one component = merge. "
                                "Limits: overlap-only components unattributed; 3-model components report lowest+highest.",
             grid=dict(working_ds=DS, working_mm=round(DS * C.PIXEL_MM[0], 4), bed_mm=BED_MM, band="ring depth from bed edge, one working cell per step"),
             uvtools=dict(raw=dict(status=(raw or {}).get("status"), counts=(raw or {}).get("counts")),
                          repaired=dict(status=(rep or {}).get("status"), counts=(rep or {}).get("counts")), gpu_issues=bool(gpu_rep)),
             errors=[])
    defects, events, heat, cured_mm2 = [], [], None, None
    try:
        import torch
        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        from mkm_slice_store import SliceStore
        store = SliceStore(store_dir)
        boxes = _model_boxes(recipe, os.path.join(out_dir, "model_bbox_cache.json"))
        out, heat, geo = _scan(store, recipe, boxes, dev, batch, log)
        defects, events, models = _defects_events(out, pieces, recipe)
        r["uvtools_unconfirmed"] = _verify_uv(defects, raw, rep, gpu_rep, store.n_layers)
        r["models"] = [dict(mo, box_mm=[[round(float(v), 2) for v in b[0][:2]], [round(float(v), 2) for v in b[1][:2]]]) for mo, b in zip(models, boxes)]
        r["n_layers"] = int(store.n_layers)
        r["height_mm"] = r["height_mm"] or round(store.n_layers * C.LAYER_MM, 2)
        cell_mm2 = (DS * C.PIXEL_MM[0]) * (DS * C.PIXEL_MM[1])
        cured_mm2 = (out["cured"].sum(1) * cell_mm2).round(1).tolist()
        r["band_counts"] = {k: sum(1 for e in events if e["event"] == k) for k in ("END", "LANDING", "START")}
        r["end_last_loop"] = "shared" if r["band_counts"]["END"] else "one model or empty bed on every layer"
    except Exception as ex:
        import traceback
        r["errors"].append("%r\n%s" % (ex, traceback.format_exc()[-1500:]))
        defects.append(dict(model=0, set="", defect="check_failed", verified=False, survives_repair=None, layers=[0, 0], detail="store geometry check did not run: %r" % ex))
        for t, n in ((raw or {}).get("counts") or {}).items():          # unverifiable UVtools hard-gate hits -> unclear
            if n and any(t in v for v in UV_TYPE.values()):
                defects.append(dict(model=0, set="", defect="uvtools_" + t, verified=False, survives_repair=None, layers=[0, 0], detail="UVtools %d %s, not verified" % (n, t)))
    r["rating"] = rating_from(defects, events)
    r["reasons"] = reasons_from(defects, events)
    r["top_reason"] = r["reasons"][0] if r["reasons"] else "hard gate clean, no band events"
    r["defects"] = [d for d in defects if not d.get("warning")]
    r["warnings"] = [d for d in defects if d.get("warning")]
    r["events"] = events
    r["seconds"] = round(time.time() - t0, 1)
    png = os.path.join(out_dir, "%s_heat.png" % pid)
    try:
        heat_png(store_dir, png, rating=r, layout_dir=plate_dir_layout, heat=heat, cured_mm2=cured_mm2, device=device)
        r["heat_png"] = png
    except Exception as ex:
        r["errors"].append("heat_png: %r" % ex)
    tmp = os.path.join(out_dir, "rating.json.tmp")
    json.dump(r, open(tmp, "w", encoding="utf-8"), indent=1, default=str); os.replace(tmp, os.path.join(out_dir, "rating.json"))
    return r


def _font(sz, b=False):
    from PIL import ImageFont
    try:
        return ImageFont.truetype(r"C:\Windows\Fonts\arialbd.ttf" if b else r"C:\Windows\Fonts\arial.ttf", sz)
    except Exception:
        return ImageFont.load_default()


RCOL = {"LIKELY": (90, 230, 120), "LIKELY WITH NOTES": (255, 200, 60), "RISKY": (255, 140, 50), "WILL NOT PRINT": (255, 80, 80)}


def heat_png(store_dir, out_png, rating=None, layout_dir=None, heat=None, cured_mm2=None, device=None, batch=BATCH):
    """Heat map in the look of ref plate_heat.py: layers cured per spot (sum over layers, 6x down for display), green->red,
    peel ring every 10 mm in from the bed edge, model labels from RING_MANIFEST, banner with rating + top reason.
    heat = precomputed (Hf/6, Wf/6) mean layers per pixel (rate_plate passes it); else one GPU pass over the store."""
    import numpy as np
    from PIL import Image, ImageDraw
    if heat is None:
        import torch, torch.nn.functional as F
        from mkm_slice_store import SliceStore
        st = SliceStore(store_dir); dev = device or ("cuda" if torch.cuda.is_available() else "cpu"); acc = None
        for i0 in range(0, st.n_layers, batch):
            x = st.read_batch(i0, min(st.n_layers, i0 + batch), device=dev).sum(0, dtype=torch.int32)
            acc = x if acc is None else acc + x
        heat = F.avg_pool2d(acc[None, None].float(), 6)[0, 0].cpu().numpy()      # display only: once per plate
    lay = np.asarray(heat, np.float32)[::-1, ::-1].copy()       # un-mirror x (A1) and put bed y up
    H6, W6 = lay.shape; px6 = 6 * C.PIXEL_MM[0]
    vmax = max(float(np.percentile(lay[lay > 0], 99.5)) if (lay > 0).any() else 1.0, 1.0)
    t_ = np.clip(lay / vmax, 0, 1)[..., None]
    g_, y_, r_ = np.array([0, 175, 70.]), np.array([255, 225, 0.]), np.array([225, 25, 25.])
    col = np.where(t_ < 0.5, g_ + (y_ - g_) * (t_ * 2), y_ + (r_ - y_) * (t_ * 2 - 1)).astype(np.uint8); col[lay <= 0] = (18, 18, 22)
    rating = rating or {}
    topm, RW = 110, 560
    img = Image.new("RGB", (W6 + RW, topm + H6 + 70), (20, 22, 26)); img.paste(Image.fromarray(col), (0, topm)); dr = ImageDraw.Draw(img)
    dr.rectangle([0, topm, W6 - 1, topm + H6 - 1], outline=(255, 255, 255))
    ox = (W6 - BED_MM[0] / px6) / 2; oy = (H6 - BED_MM[1] / px6) / 2
    for lv in range(0, int(BED_MM[1] / 2) + 1, 10):              # peel ring: 0 = bed edge, then every 10 mm inward
        a = lv / px6
        dr.rectangle([ox + a, topm + oy + a, W6 - ox - a, topm + H6 - oy - a], outline=(255, 255, 255) if lv == 0 else (90, 200, 255))
        if lv:
            dr.text((W6 / 2 + 4, topm + H6 - oy - a - 16), "%d mm" % lv, fill=(90, 200, 255), font=_font(13))
    man = _load(os.path.join(layout_dir, "RING_MANIFEST.json"), [{}]) if layout_dir else [{}]
    man = man[0] if isinstance(man, list) and man else {}
    boxes = {m["model"]: m.get("box_mm") for m in rating.get("models", [])}
    for p in man.get("pieces", []):
        b = boxes.get(p.get("no"))
        cx, cy = ((b[0][0] + b[1][0]) / 2, (b[0][1] + b[1][1]) / 2) if b else (p.get("x", 0.0), p.get("y", 0.0))
        tx = W6 / 2 + cx / px6; ty = topm + H6 / 2 - cy / px6
        dr.text((tx - 6, ty - 9), str(p.get("no", "?")), fill=(255, 255, 255), font=_font(17, True), stroke_width=2, stroke_fill=(0, 0, 0))
    rt = rating.get("rating", "NOT RATED")
    dr.text((10, 6), "%s   %s pieces   %s mm tall" % (rating.get("plate", os.path.basename(os.path.dirname(out_png))), rating.get("pieces", len(man.get("pieces", []))), rating.get("height_mm", "?")), fill=(255, 255, 255), font=_font(28, True))
    dr.text((10, 42), "%s  (%s)" % (rt, RATING_BASIS), fill=RCOL.get(rt, (200, 200, 200)), font=_font(22, True))
    dr.text((10, 70), ("top reason: " + rating.get("top_reason", ""))[:200], fill=(230, 230, 230), font=_font(14))
    dr.text((10, 90), "colour = layers cured at that spot: green = few, yellow = middle, red = most (red tops at %.0f layers) | blue = peel ring, mm in from bed edge | numbers = piece no." % vmax, fill=(190, 190, 190), font=_font(12))
    gx = W6 + 40; gw = RW - 70
    if cured_mm2:
        g2 = topm + 20; h2 = 220; am = max(max(cured_mm2), 1.0); n = len(cured_mm2)
        dr.text((gx, g2 - 22), "Cured area per layer up the print (mm2)", fill=(230, 230, 230), font=_font(14, True))
        dr.rectangle([gx, g2, gx + gw, g2 + h2], outline=(90, 90, 90))
        step = max(1, n // gw)
        pts = [(gx + gw * i / max(n - 1, 1), g2 + h2 - h2 * cured_mm2[i] / am) for i in range(0, n, step)]
        if len(pts) > 1:
            dr.line(pts, fill=(255, 190, 70), width=2)
        dr.text((gx, g2 + h2 + 4), "layer 0", fill=(180, 180, 180), font=_font(12)); dr.text((gx + gw - 70, g2 + h2 + 4), "layer %d" % (n - 1), fill=(180, 180, 180), font=_font(12))
    y = topm + 290
    dr.text((gx, y), "Reasons (hard gate, END, LANDING, START, warnings)", fill=(230, 230, 230), font=_font(14, True)); y += 20
    for line in (rating.get("reasons") or ["none"])[:24]:
        dr.text((gx, y), line[:88], fill=(200, 200, 200), font=_font(11)); y += 15
    names = "   ".join("%s  %s" % (p.get("no"), p.get("set", "")) for p in man.get("pieces", []))
    for k in range(3):
        dr.text((10, topm + H6 + 6 + 18 * k), names[230 * k:230 * (k + 1)], fill=(170, 170, 170), font=_font(12))
    os.makedirs(os.path.dirname(os.path.abspath(out_png)), exist_ok=True)
    img.save(out_png)
    return out_png


def _fmt_counts(c):
    return ", ".join("%s %s" % (k, v) for k, v in sorted((c or {}).items()) if v) or "0"


def summarize(run_dir):
    """SUMMARY.md table + SUMMARY.png contact sheet(s) (6 tiles per sheet, as ref auto/run_all.py)."""
    import textwrap
    from PIL import Image, ImageDraw
    rates = sorted(glob.glob(os.path.join(run_dir, "rate", "P*", "rating.json")), key=lambda p: int(re.sub(r"\D", "", os.path.basename(os.path.dirname(p))) or 0))
    rows = []; tiles = []
    for rp in rates:
        r = _load(rp, {}) or {}
        pid = r.get("plate") or os.path.basename(os.path.dirname(rp))
        uvb = _fmt_counts(((r.get("uvtools") or {}).get("raw") or {}).get("counts")); uva = _fmt_counts(((r.get("uvtools") or {}).get("repaired") or {}).get("counts"))
        top = " / ".join((r.get("reasons") or [r.get("top_reason", "")])[:3]).replace("|", "/")
        rows.append("| %s | %s | %s | %s | %s | %s | %s |" % (pid, r.get("pieces", "?"), r.get("height_mm", "?"), r.get("rating", "NOT RATED"), top or "-", uvb, uva))
        png = r.get("heat_png") or os.path.join(os.path.dirname(rp), "%s_heat.png" % pid)
        t = Image.new("RGB", (900, 700), (245, 245, 245)); dr = ImageDraw.Draw(t)
        rt = r.get("rating", "NOT RATED")
        dr.text((10, 6), "%s   %s pieces   %s mm tall   %s" % (pid, r.get("pieces", "?"), r.get("height_mm", "?"), rt), fill={"LIKELY": (0, 110, 40), "LIKELY WITH NOTES": (150, 110, 0), "RISKY": (190, 90, 0)}.get(rt, (190, 30, 20)), font=_font(24, True))
        y = 40
        for l in textwrap.wrap(r.get("top_reason", ""), 118)[:2]:
            dr.text((10, y), l, fill=(40, 40, 40), font=_font(13)); y += 16
        if os.path.exists(png):
            im = Image.open(png).convert("RGB"); w, h = im.size; im = im.resize((880, int(880 * h / w)), Image.LANCZOS)
            t = t.crop((0, 0, 900, y + 6 + im.size[1] + 10)); t.paste(im, (10, y + 6))
        tiles.append(t)
    sheets = []
    for old in glob.glob(os.path.join(run_dir, "SUMMARY*.png")):
        os.remove(old)                                          # our own previous sheets in the run folder only
    for n in range(0, len(tiles), 6):
        g = tiles[n:n + 6]; hh = [max(a.size[1] for a in g[i:i + 2]) for i in range(0, len(g), 2)]
        sh = Image.new("RGB", (1800, sum(hh)), (245, 245, 245)); y = 0
        for i, a in enumerate(g):
            sh.paste(a, ((i % 2) * 900, y))
            if i % 2 == 1:
                y += hh[i // 2]
        name = "SUMMARY.png" if n == 0 else "SUMMARY_%d.png" % (n // 6 + 1)
        sh.save(os.path.join(run_dir, name)); sheets.append(name)
    md = ["# MKM plater v3 - run summary", "", "Rating basis: %s." % RATING_BASIS,
          "UVtools counts are recorded, not a limit (merged rule s.2); each hit was checked against the sliced geometry.", "",
          "| plate | pieces | height mm | rating | top reasons | UVtools before | UVtools after repair |",
          "|---|---|---|---|---|---|---|"] + rows + ["", "Sheets: " + (", ".join(sheets) or "none"), "", STOP_LINE, ""]
    open(os.path.join(run_dir, "SUMMARY.md"), "w", encoding="utf-8").write("\n".join(md))
    return os.path.join(run_dir, "SUMMARY.md")


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "summarize":
        print(summarize(sys.argv[2]))
    elif len(sys.argv) >= 5:
        print(json.dumps({k: v for k, v in rate_plate(*sys.argv[1:5]).items() if k in ("plate", "rating", "top_reason", "errors")}, indent=1))
    else:
        print("usage: mkm_rate.py <layout/P{n}> <store/P{n}> <uv/P{n}> <rate/P{n}>  |  mkm_rate.py summarize <RUN>")
