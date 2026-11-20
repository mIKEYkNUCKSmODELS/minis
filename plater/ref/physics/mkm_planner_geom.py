"""mkm_planner_geom.py - the planner's non-hand-set helpers (20261116, planner fix). Pure functions, unit-tested in tests/test_planner_fix.py.

  finite_tri_count / header_tri_count   ntri = triangles a plate file will really contain (finite ones). The old code mixed two meanings of 'ntri'.
  min_gap_cells / gap_ok                gap tolerance derived from the placement raster, replacing the literal 'GAP - 0.6'.
  classify_big                          big-model report: FLAG, never hold (nothing is removed from the pool).
  place_unit_atomic                     body and its own base (one figure = one 'unit') are placed together on one plate; split only if no single plate can hold the unit.
  self_compliance_map                   film self-compliance of a footprint at every origin, solved on a lattice that is refined until it converges (replaces blob sigma 3 mm / stride 16).
  layer_hours                           printer-hours estimate from seconds/layer (a parameter, not a literal)."""
import os, re, math
import numpy as np

STL_DT = np.dtype([("n", "<f4", (3,)), ("v", "<f4", (3, 3)), ("a", "<u2")])


# ------------------------------------------------------------------------------------------------------------------------------- ntri
def header_tri_count(path):
    """count stored in the binary STL header (bytes 80..83); the file must be exactly 84 + 50 n bytes."""
    with open(path, "rb") as fh:
        head = fh.read(84)
    n = int(np.frombuffer(head[80:84], "<u4")[0])
    if 84 + 50 * n != os.path.getsize(path):
        raise ValueError("%s: header says %d triangles but the file size says %s" % (path, n, (os.path.getsize(path) - 84) / 50.0))
    return n


def finite_tri_count(path):
    """number of triangles whose 9 vertex coordinates are all finite = what mkm_packer.tri_array, mkm_planner.load_raw and the written plate keep.
    THIS is 'ntri' everywhere (footprint dicts, RING_MANIFEST 'n', plate triangle spans). The header count may be larger (non-finite triangles are dropped)."""
    n = header_tri_count(path)
    v = np.memmap(path, dtype=STL_DT, mode="r", offset=84, shape=(n,))["v"]
    cnt = 0
    for c in range(0, n, 1_000_000):
        cnt += int(np.isfinite(np.asarray(v[c:c + 1_000_000]).reshape(-1, 9)).all(1).sum())
    return cnt


# ------------------------------------------------------------------------------------------------------------------------------- gap
def min_gap_cells(gap_mm, pitch_mm):
    """cells kept clear around every piece by the legality test: g = round(gap/R) (the max-pool window of find() is 2g+1)."""
    return int(round(gap_mm / pitch_mm))


def achieved_gap_floor_mm(gap_mm, pitch_mm):
    """what write_bed measures for a legal placement: legality keeps >= g EMPTY cells between any two pieces (max-pool window 2g+1), so the nearest occupied cell of another piece is >= g+1 cells away,
    and the Euclidean distance transform between cell centres returns >= (g + 1) * R. (The edge-to-edge clearance is that minus one cell = g * R = the requested gap.) Derived from the raster,
    replaces the literal 'GAP - 0.6' which accepted 2.4 mm for a 3 mm gap, i.e. placements that the legality test itself would have refused."""
    return (min_gap_cells(gap_mm, pitch_mm) + 1) * pitch_mm


def gap_ok(measured_mm, gap_mm, pitch_mm):
    return measured_mm >= achieved_gap_floor_mm(gap_mm, pitch_mm) - 1e-6


# ------------------------------------------------------------------------------------------------------------------------------- big models
def classify_big(sets, tall_mm, allow_regex, bed_share, force_share=None):
    """sets = {name: [rows]} with row['h'] (mm). bed_share(rows) -> plates of bed allotment the model needs by itself; force_share(rows) -> plates of force allotment or None (budget pending).
    Returns a list of dicts sorted by tallest first; flag=True means 'tell Mike before plating (hard sell)'. NOTHING is removed from the pool: the caller keeps every row."""
    out = []
    for name, g in sets.items():
        tall = max(x["h"] for x in g); bs = bed_share(g); fs = force_share(g) if force_share else None
        why = [w for w, c in ((("needs %.1f plates of bed allotment by itself" % bs), bs > 1.0), (("needs %.1f plates of force allotment by itself" % fs) if fs is not None else "", fs is not None and fs > 1.0),
                              (("tallest part %.0f mm (%.1f in) > %.0f mm" % (tall, tall / 25.4, tall_mm)), tall > tall_mm)) if c]
        allowed = bool(allow_regex) and re.search(allow_regex, name, re.I) is not None
        out.append(dict(name=name, parts=len(g), tall_mm=tall, bed_plates=bs, force_plates=fs, reasons=why, allowed=allowed, flag=bool(why) and not allowed, held=False))
    return sorted(out, key=lambda d: -d["tall_mm"])


def flag_report_lines(rows, hours_fn):
    return ["%s %-45s %2d parts | tallest %5.0f mm (%4.1f in) | %4.2f plates | ~%4.1f h | %s" % (
        "FLAG" if d["flag"] else ("ok* " if d["allowed"] and d["reasons"] else "ok  "), d["name"].split(" (")[0], d["parts"], d["tall_mm"], d["tall_mm"] / 25.4, d["bed_plates"],
        hours_fn(d["tall_mm"]), "; ".join(d["reasons"]) + (" (allowed: big on purpose)" if d["allowed"] and d["reasons"] else "") + ("  -> PLATED, NOT HELD" if d["flag"] else "")) for d in rows]


# ------------------------------------------------------------------------------------------------------------------------------- units (body + its own base)
def unit_key(row):
    """a 'unit' is one figure: its body, base, arms ... (mkm_packer.model_unit). Rows without a unit fall back to their set."""
    u = row.get("unit")
    return tuple(u) if isinstance(u, (list, tuple)) else (row["set"], u if u is not None else "")


def place_unit_atomic(pieces, beds, try_place, commit, uncommit, max_span=3, order_key=None):
    """Place all pieces of ONE unit. Pass 1: for each bed in preference order, try to put EVERY piece on that single bed (commit as we go; uncommit all on the first failure).
    Pass 2 (no single bed holds the unit): split over as few beds as possible, at most max_span beds, by repeatedly taking the bed that fits the most remaining pieces.
    try_place(piece, beds) -> record|None (does not commit); commit(rec); uncommit(piece). Returns (list of (piece, rec), n_beds) or (None, 0) with nothing committed."""
    pieces = list(pieces)
    for b in (sorted(beds, key=order_key) if order_key else list(beds)):
        got = []
        for p in pieces:
            rec = try_place(p, [b])
            if rec is None:
                break
            commit(rec); got.append((p, rec))
        if len(got) == len(pieces):
            return got, 1
        for p, _ in got:
            uncommit(p)
    got, left, used = [], list(pieces), []
    while left and len(used) < max_span:
        best = None
        for b in beds:
            if b in used:
                continue
            tmp = []
            for p in left:
                rec = try_place(p, [b])
                if rec is None:
                    continue
                commit(rec); tmp.append((p, rec))
            for p, _ in tmp:
                uncommit(p)
            if best is None or len(tmp) > len(best[1]):
                best = (b, tmp)
        if best is None or not best[1]:
            break
        b, _ = best; used.append(b)
        for p in list(left):
            rec = try_place(p, [b])
            if rec is None:
                continue
            commit(rec); got.append((p, rec)); left.remove(p)
    if left:
        for p, _ in got:
            uncommit(p)
        return None, 0
    return got, len(used)


# ------------------------------------------------------------------------------------------------------------------------------- film self-compliance
def self_compliance_map(defl, W, H, dx, rmask, rel_tol, device="cpu", max_levels=6, chunk=16):
    """c(origin) [m/N] for a piece whose raft footprint `rmask` [sw, sh] has its corner cell at origin (ox, oy) on the bed grid [W, H] (pitch dx m): the mean film deflection over the
    footprint when 1 N is spread uniformly over it, computed with the SAME membrane solve (defl: q [..., W, H] N/m2 -> w [..., W, H] m) that scores the cross terms. U_self = 1/2 F^2 c.
    Solved exactly on a lattice of origins; the lattice spacing halves until the bilinear interpolation of the previous lattice reproduces the finer one to within rel_tol (of the map's peak).
    Returns (map [W, H] float64 numpy, info dict(levels, lattice, solves, rel_err, converged)). Not converged -> converged False and the caller must not use it (UNSOLVED)."""
    import torch
    from scipy.interpolate import RegularGridInterpolator
    sw, sh = rmask.shape; nx, ny = max(W - sw + 1, 1), max(H - sh + 1, 1)
    rm = torch.as_tensor(np.asarray(rmask, bool), dtype=torch.float64, device=device); n = float(rm.sum())
    if n <= 0:
        raise ValueError("empty footprint")
    qv = rm / (n * dx * dx)
    cache = {}

    def interp(ax, ay, C):
        """cubic when the lattice allows it (>= 4 points per axis): the map has a log-like boundary layer at the clamped frame, linear interpolation converges only O(h) there (measured: 0.21/0.10/0.047/0.028
        at strides 10/5/3/2 vs cubic 0.13/0.038/0.014/0.0056)."""
        return RegularGridInterpolator((ax, ay), C, method="cubic" if (len(ax) >= 4 and len(ay) >= 4) else "linear")

    def solve(pts):
        todo = [p for p in pts if p not in cache]
        for c0 in range(0, len(todo), chunk):
            part = todo[c0:c0 + chunk]
            q = torch.zeros((len(part), W, H), dtype=torch.float64, device=device)
            for k, (ox, oy) in enumerate(part):
                q[k, ox:ox + sw, oy:oy + sh] = qv
            w = defl(q)
            c = (w * q).sum((-2, -1)) * dx * dx
            for k, p in enumerate(part):
                cache[p] = float(c[k])

    def axis(stride, N):
        a = list(range(0, N, stride))
        if a[-1] != N - 1:
            a.append(N - 1)
        return np.array(a)

    stride = max(1, max(nx, ny) // 4); prev = None; err = None; levels = 0
    while True:
        ax, ay = axis(stride, nx), axis(stride, ny)
        solve([(int(i), int(j)) for i in ax for j in ay])
        C = np.array([[cache[(int(i), int(j))] for j in ay] for i in ax]); levels += 1
        if prev is not None:
            (pax, pay, PC) = prev
            f = interp(pax, pay, PC)
            gx, gy = np.meshgrid(ax, ay, indexing="ij"); est = f(np.stack([gx.ravel(), gy.ravel()], 1)).reshape(C.shape)
            err = float(np.abs(est - C).max() / max(float(np.abs(C).max()), 1e-300))
            if err <= rel_tol:
                break
        if stride == 1 or levels >= max_levels:
            break
        prev = (ax, ay, C); stride = max(1, stride // 2)
    exact = bool(stride == 1 and len(ax) == nx and len(ay) == ny)          # every origin solved: nothing interpolated, so nothing left to converge
    ok = (err is not None and err <= rel_tol) or exact
    f = interp(ax, ay, C)
    gx, gy = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    full = f(np.stack([gx.ravel(), gy.ravel()], 1)).reshape(nx, ny)
    full = np.pad(full, ((0, W - nx), (0, H - ny)), mode="edge")
    return full, dict(levels=levels, lattice=(len(ax), len(ay)), solves=len(cache), rel_err=err, converged=bool(ok), exact=exact)


# ------------------------------------------------------------------------------------------------------------------------------- hours
def layer_hours(height_mm, layer_h_mm, seconds_per_layer):
    return math.ceil(height_mm / layer_h_mm) * seconds_per_layer / 3600.0
