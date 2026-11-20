"""MKM GPU PORT (Claude 11/15, Mike 11/14 standing ruling: "everything that can done with the gpu needs to happen on the gpu utilizing its capacity safely").
PLUMBING ONLY: the same arithmetic as the CPU code, on torch/cuda, so masks and numbers are the same (equivalence proven by _gpu_port\\test_equiv.py).
Switch: env MKM_DEV=cpu forces the original CPU code; default = cuda when available.  VRAM: capped to MKM_VRAM_FRAC (0.45) and to what is left under 80% of the card;
no work is launched while the GPU is >= 87 C (waits).  Every function here mirrors a CPU function and is called from a dispatch in that CPU function's own file."""
import os, sys, time, math, subprocess, numpy as np
import torch
for _d in (os.environ.get('MKM_PHYS_DIR', os.path.abspath(os.path.join(os.path.dirname(__file__),'..','physics'))),):
    if _d not in sys.path: sys.path.insert(0, _d)
import mkm_planner_params as _PP
RAFT_BAND_MM = _PP.raft_band_mm()      # 20261116: raft band from the planner parameter file (was the literal 0.30 / 0.3)
R = 0.5
_STATE = dict(dev=None, init=False, note="", last_t=0.0, temp=0.0)
TEMP_STOP = min(87., float(os.environ.get("MKM_TEMP_STOP", 87)))
def _smi(q):
    try:
        o = subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout.strip().split(",")
        return [float(x) for x in o]
    except Exception: return None
def configure():
    """decide the device once: cuda unless MKM_DEV=cpu / no cuda / no VRAM left under the 80% ceiling."""
    if _STATE["init"]: return _STATE["dev"]
    _STATE["init"] = True
    if os.environ.get("MKM_DEV", "cuda").lower() == "cpu": _STATE["note"] = "MKM_DEV=cpu"; return None
    if not torch.cuda.is_available(): _STATE["note"] = "no cuda"; return None
    s = _smi("memory.used,memory.total")
    frac = min(0.45, float(os.environ.get("MKM_VRAM_FRAC", 0.45)))
    if not s: raise RuntimeError("UNSOLVED: cannot read GPU usage")
    if s:
        used, total = s
        # 20261118 Mike 11/17 15:16 (Gpu limit can be used by multiple jobs safely, not one job): the hand-set 5 GiB aggregate cap and the 80 pct ceiling are replaced by a real fit check. This job measured need (planner peak 3.4 GiB device-wide, 20261118) + 512 MiB margin must be FREE. Env MKM_VRAM_NEED_MIB overrides the need.
        _need = float(os.environ.get("MKM_VRAM_NEED_MIB", 3500)); _margin = 512.0
        if total - used < _need + _margin: raise RuntimeError("UNSOLVED: VRAM does not fit: %.0f MiB free, this job needs %.0f MiB + %.0f MiB margin" % (total - used, _need, _margin))
        room = total - used - _margin
        if room < 400: _STATE["note"] = f"only {room:.0f} MiB left under the 80% ceiling -> CPU"; return None
        frac = min(frac, room / total)
    torch.cuda.set_per_process_memory_fraction(max(0.02, min(0.95, frac)), 0)
    _STATE["dev"] = torch.device("cuda"); _STATE["note"] = f"cuda, VRAM fraction {frac:.3f}"; return _STATE["dev"]
def dev(): return configure()
def temp_gate():
    """stop launching work above TEMP_STOP C: wait until it cools."""
    from mkm_safety import checkpoint
    while True:
        checkpoint()
        usage = _smi("memory.used")
        if usage is None or usage[0] >= 11000: raise MemoryError("UNSOLVED: 11000 MiB VRAM guard or unavailable telemetry")
        t = time.time()
        if t - _STATE["last_t"] > 4:
            s = _smi("temperature.gpu"); _STATE["last_t"] = t; _STATE["temp"] = s[0] if s else 0.0
        if _STATE["temp"] < TEMP_STOP: return
        print(f"[gpuport] GPU {_STATE['temp']:.0f} C >= {TEMP_STOP:.0f}: holding new work", flush=True); time.sleep(10); _STATE["last_t"] = 0.0

# ---------------- binary morphology, scipy-exact (cross structure, border 0 / flood border 1) ----------------
def dil(x):
    y = x.clone(); y[1:] |= x[:-1]; y[:-1] |= x[1:]; y[:, 1:] |= x[:, :-1]; y[:, :-1] |= x[:, 1:]; return y
def ero(x):
    y = x.clone(); y[1:] &= x[:-1]; y[0] = False; y[:-1] &= x[1:]; y[-1] = False; y[:, 1:] &= x[:, :-1]; y[:, 0] = False; y[:, :-1] &= x[:, 1:]; y[:, -1] = False; return y
def closing(g, it):
    for _ in range(it): g = dil(g)
    for _ in range(it): g = ero(g)
    return g
def fill_holes(g):
    """background connected (4-conn) to the array border stays background; everything else is filled (scipy binary_fill_holes)."""
    bg = ~g; seed = torch.zeros_like(g); seed[0] = True; seed[-1] = True; seed[:, 0] = True; seed[:, -1] = True
    reach = seed & bg
    while True:
        prev = reach
        for _ in range(24): reach = dil(reach) & bg
        if bool((reach == prev).all()): break
    return ~reach

# ---------------- compute_piece rasterisation (mkm_gpupool10.compute_piece masks) ----------------
_W = {}
def _weights(lvl, d):
    if lvl not in _W:
        i, j = np.meshgrid(np.arange(lvl + 1), np.arange(lvl + 1), indexing="ij"); m = (i + j) <= lvl
        wa = (i[m] / lvl).astype(np.float32); wb = (j[m] / lvl).astype(np.float32); wc = 1 - wa - wb
        _W[lvl] = tuple(torch.from_numpy(w).to(d) for w in (wa, wb, wc))
    return _W[lvl]
def _dense_gen(raw_v32, d, tri_chunk=1000000):
    """yields float32 (m,3) point chunks = numpy dense_pts() split into pieces (order irrelevant: only a scatter/min/max follows)."""
    for c in range(0, len(raw_v32), tri_chunk):
        tv = torch.from_numpy(np.ascontiguousarray(raw_v32[c:c + tri_chunk])).to(d)
        yield tv.reshape(-1, 3)
        xy = tv[:, :, :2].double(); e = []
        for a_, b_ in ((1, 0), (2, 1), (0, 2)):
            df = xy[:, a_] - xy[:, b_]; e.append(torch.sqrt(df[:, 0] * df[:, 0] + df[:, 1] * df[:, 1]))
        e = torch.maximum(torch.maximum(e[0], e[1]), e[2]); n = torch.ceil(e / 0.25).long()
        for lvl in (2, 4, 8, 16, 32, 64, 128, 256, 512, 1024):
            sel = torch.nonzero((n > lvl // 2) & ((n <= lvl) | (lvl == 1024)))[:, 0]
            if not sel.numel(): continue
            wa, wb, wc = _weights(lvl, d); step = max(1, 4000000 // wa.numel())
            for c0 in range(0, sel.numel(), step):
                t = tv[sel[c0:c0 + step]]
                yield (wa[None, :, None] * t[:, 0][:, None, :] + wb[None, :, None] * t[:, 1][:, None, :] + wc[None, :, None] * t[:, 2][:, None, :]).reshape(-1, 3)
def _rot_xy(v, k):
    c, s = [(1, 0), (0, 1), (-1, 0), (0, -1)][k]
    return c * v[:, 0] - s * v[:, 1], s * v[:, 0] + c * v[:, 1]
def _rot_ok(v, k):
    x, y = _rot_xy(v, k); z = v[:, 2]
    ok = torch.isfinite(x) & torch.isfinite(y) & torch.isfinite(z) & (x.abs() < 1e4) & (y.abs() < 1e4) & (z.abs() < 1e4)
    return x, y, z, ok
def cp_masks(raw_v32, pad=4, *, layer_z=None, shape=None, pitch=None, origin=(0.,0.), device="cuda", session=None, layer_index=None):
    if layer_z is not None:
        if str(device).startswith("cuda"):
            if dev() is None: raise RuntimeError("UNSOLVED: requested CUDA unavailable")
            temp_gate()
        if session is not None:
            if layer_index is None or float(session.zs[layer_index]) != float(layer_z): raise ValueError('exact layer identity mismatch')
            return session.layer(layer_index)
        from mkm_slicer_bridge import raster
        return raster(raw_v32, layer_z, shape, pitch, origin, device)
    """raw_v32: numpy float32 (n,3,3), already finite-filtered and z-shifted exactly as load_raw does. Returns list of 4 dicts(mask, rmask, mn, c0, pad)."""
    d = dev(); temp_gate()
    big = float("inf"); mn = [[big, big] for _ in range(4)]; mx = [[-big, -big] for _ in range(4)]; zmn = [big] * 4
    mnt = [torch.full((2,), big, device=d) for _ in range(4)]; mxt = [torch.full((2,), -big, device=d) for _ in range(4)]; zt = [torch.tensor(big, device=d) for _ in range(4)]
    for P in _dense_gen(raw_v32, d):
        for k in range(4):
            x, y, z, ok = _rot_ok(P, k); x, y, z = x[ok], y[ok], z[ok]
            if not x.numel(): continue
            mnt[k] = torch.minimum(mnt[k], torch.stack([x.min(), y.min()])); mxt[k] = torch.maximum(mxt[k], torch.stack([x.max(), y.max()])); zt[k] = torch.minimum(zt[k], z.min())
    G, RM, THR = [], [], []
    for k in range(4):
        m_ = mnt[k]; ext = ((mxt[k] - m_) / R).long() + 2 * pad + 1
        G.append(torch.zeros((int(ext[0]), int(ext[1])), dtype=torch.bool, device=d)); RM.append(torch.zeros_like(G[-1]))
        THR.append(float(np.float32(zt[k].item()) + RAFT_BAND_MM))
    for P in _dense_gen(raw_v32, d):
        for k in range(4):
            x, y, z, ok = _rot_ok(P, k); x, y, z = x[ok], y[ok], z[ok]
            ix = ((x - mnt[k][0]) / R).long() + pad; iy = ((y - mnt[k][1]) / R).long() + pad
            G[k][ix, iy] = True; sl = z <= THR[k]; RM[k][ix[sl], iy[sl]] = True
    out = []
    for k in range(4):
        g = fill_holes(closing(G[k], 3)); xs = torch.nonzero(g.any(1))[:, 0]; ys = torch.nonzero(g.any(0))[:, 0]
        x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max()); gc = g[x0:x1 + 1, y0:y1 + 1]; rm = RM[k][x0:x1 + 1, y0:y1 + 1]
        if not bool(rm.any()): rm = gc.clone()
        out.append(dict(mask=gc.cpu().numpy(), rmask=rm.cpu().numpy(), mn=mnt[k].cpu().numpy().astype(np.float32), c0=(np.int64(x0), np.int64(y0)), pad=pad))
    del G, RM; return out

# ---------------- mkm_packer.footprint on the GPU ----------------
def footprint(path, res, P):
    """same output dict as P.footprint(path, res); falls back to it for ASCII STL."""
    arr = P.tri_array(path)
    if arr is None: return P.footprint(path, res)
    d = dev(); temp_gate(); n = len(arr); CH = 1500000
    def chunks():
        for c in range(0, n, CH):
            yield torch.from_numpy(np.ascontiguousarray(arr["v"][c:c + CH, 3:])).to(d).reshape(-1, 3, 3).double()
    mn = torch.full((3,), float("inf"), dtype=torch.float64, device=d); mx = -mn.clone()
    for v in chunks(): f = v.reshape(-1, 3); mn = torch.minimum(mn, f.min(0).values); mx = torch.maximum(mx, f.max(0).values)
    xmin, ymin, zmin = [float(a) for a in mn.cpu()]; xmax, ymax, zmax = [float(a) for a in mx.cpu()]
    W = max(1, math.ceil((xmax - xmin) / res - 1e-9)); H = max(1, math.ceil((ymax - ymin) / res - 1e-9))
    G = torch.zeros((H, W), dtype=torch.bool, device=d); org = torch.tensor([xmin, ymin], dtype=torch.float64, device=d); cap = torch.tensor([W - 1, H - 1], device=d)
    def cell(x, y): return torch.clamp(((x - xmin) / res).long(), max=W - 1), torch.clamp(((y - ymin) / res).long(), max=H - 1)
    for v in chunks():
        xy = v[:, :, :2]; cxy = torch.minimum(((xy - org) / res).long(), cap); lo = cxy.min(1).values; hi = cxy.max(1).values
        small = ((hi - lo + 1).prod(1) <= 4)
        for dx in (0, 1):
            for dy in (0, 1):
                c = torch.minimum(lo[small] + torch.tensor([dx, dy], device=d), hi[small]); G[c[:, 1], c[:, 0]] = True
        t = xy[~small]
        if not t.shape[0]: continue
        ax, ay, bx, by, cx_, cy_ = t[:, 0, 0], t[:, 0, 1], t[:, 1, 0], t[:, 1, 1], t[:, 2, 0], t[:, 2, 1]
        # --- pixel test inside each triangle's cell box (only |d| > 1e-12)
        x0, y0 = cell(torch.minimum(torch.minimum(ax, bx), cx_), torch.minimum(torch.minimum(ay, by), cy_)); x1, y1 = cell(torch.maximum(torch.maximum(ax, bx), cx_), torch.maximum(torch.maximum(ay, by), cy_))
        dd = (by - cy_) * (ax - cx_) + (cx_ - bx) * (ay - cy_); use = dd.abs() > 1e-12
        bw = (x1 - x0 + 1); cnt = bw * (y1 - y0 + 1) * use
        cs = torch.cumsum(cnt, 0); LIM = 6000000; a = 0; m_ = cnt.numel()
        while a < m_:
            base = int(cs[a - 1]) if a else 0; b = int(torch.searchsorted(cs, torch.tensor(base + LIM, device=d), right=True)); b = max(b, a + 1); b = min(b, m_)
            c_ = cnt[a:b]; tot = int(c_.sum())
            if tot:
                ti = torch.repeat_interleave(torch.arange(a, b, device=d), c_); loc = torch.arange(tot, device=d) - torch.repeat_interleave(torch.cumsum(c_, 0) - c_, c_)
                xx = x0[ti] + loc % bw[ti]; yy = y0[ti] + loc // bw[ti]
                px = xmin + (xx.double() + 0.5) * res; py = ymin + (yy.double() + 0.5) * res
                Bx, By, Cx, Cy, Ax, Ay, D = bx[ti], by[ti], cx_[ti], cy_[ti], ax[ti], ay[ti], dd[ti]
                l1 = ((By - Cy) * (px - Cx) + (Cx - Bx) * (py - Cy)) / D; l2 = ((Cy - Ay) * (px - Cx) + (Ax - Cx) * (py - Cy)) / D
                ok = (l1 >= -1e-9) & (l2 >= -1e-9) & (l1 + l2 <= 1 + 1e-9); G[yy[ok], xx[ok]] = True
            a = b
        # --- edge sampling marks (every triangle, whatever |d|)
        for (px_, py_, qx_, qy_) in ((ax, ay, bx, by), (bx, by, cx_, cy_), (cx_, cy_, ax, ay)):
            ex, ey = qx_ - px_, qy_ - py_; L = torch.hypot(ex, ey); tq = L / (res * 0.5); nn = torch.clamp(tq.long(), min=1)
            near = (tq - torch.round(tq)).abs() < 1e-7 * torch.clamp(tq, min=1.0)       # hypot could differ by an ulp only here -> use math.hypot
            if bool(near.any()):
                ii = torch.nonzero(near)[:, 0]; ex_, ey_ = ex[ii].tolist(), ey[ii].tolist()
                nn[ii] = torch.tensor([max(1, int(math.hypot(a_, b_) / (res * 0.5))) for a_, b_ in zip(ex_, ey_)], device=d)
            cn = nn + 1; cs2 = torch.cumsum(cn, 0); a = 0; m2 = cn.numel()
            while a < m2:
                base = int(cs2[a - 1]) if a else 0; b = int(torch.searchsorted(cs2, torch.tensor(base + 8000000, device=d), right=True)); b = max(b, a + 1); b = min(b, m2)
                c_ = cn[a:b]; tot = int(c_.sum()); ti = torch.repeat_interleave(torch.arange(a, b, device=d), c_); kk = torch.arange(tot, device=d) - torch.repeat_interleave(torch.cumsum(c_, 0) - c_, c_)
                f = kk.double() / nn[ti].double(); X = px_[ti] + ex[ti] * f; Y = py_[ti] + ey[ti] * f; cx, cy = cell(X, Y); G[cy, cx] = True; a = b
    g = fill_holes(G); gn = g.cpu().numpy()
    pk = np.packbits(gn, axis=1, bitorder="little"); rows = [format(int.from_bytes(pk[y].tobytes(), "little"), "x") for y in range(H)]
    return dict(xmin=xmin, ymin=ymin, zmin=zmin, xmax=xmax, ymax=ymax, zmax=zmax, W=W, H=H, rows=rows, ntri=n, res=res)


# ---------------- exact slicer (mkm_exactslice.Slicer.segs / fill) on the GPU ----------------
def load_T32(path):
    """binary STL -> float32 (n,3,3) with non-finite triangles dropped (same rows XS.load keeps; the float64 values are these widened exactly)."""
    DT = np.dtype([("n", "<f4", (3,)), ("v", "<f4", (3, 3)), ("a", "<u2")])
    v = np.fromfile(path, dtype=DT, offset=84)["v"]; return np.ascontiguousarray(v[np.isfinite(v.reshape(-1, 9)).all(1)])
class GSlicer:
    def __init__(self, T32, zshift=None, d=None):
        self.d = d or dev(); self.zs = zshift; T = torch.from_numpy(T32).to(self.d)
        zmin = T[:, :, 2].min(1).values.double(); zmax = T[:, :, 2].max(1).values.double()
        if zshift is not None: zmin = zmin - zshift; zmax = zmax - zshift
        o = torch.sort(zmin, stable=True)[1]; self.T = T[o]; self.zmin = zmin[o]; self.zmax = zmax[o]; del T, o
    def segs(self, z):
        d = self.d; hi = int(torch.searchsorted(self.zmin, torch.tensor([z], dtype=torch.float64, device=d), right=False)[0])
        idx = torch.nonzero(self.zmax[:hi] > z)[:, 0]
        if not idx.numel(): return None
        t = self.T[idx].double()
        if self.zs is not None: t[:, :, 2] = t[:, :, 2] - self.zs
        a = t[:, 1] - t[:, 0]; b = t[:, 2] - t[:, 0]
        n0 = a[:, 1] * b[:, 2] - a[:, 2] * b[:, 1]; n1 = a[:, 2] * b[:, 0] - a[:, 0] * b[:, 2]
        m = t.shape[0]; P0 = torch.full((m, 2), float("nan"), dtype=torch.float64, device=d); P1 = P0.clone(); cnt = torch.zeros(m, dtype=torch.long, device=d)
        for a_, b_ in ((0, 1), (1, 2), (2, 0)):
            A = t[:, a_]; B = t[:, b_]; za = A[:, 2]; zb = B[:, 2]; c = (za - z) * (zb - z) < 0
            u = ((z - za) / torch.where(c, zb - za, torch.ones_like(za)))[:, None]; p = (A + u * (B - A))[:, :2]
            m1 = c & (cnt == 1); m0 = c & (cnt == 0)
            P1 = torch.where(m1[:, None], p, P1); P0 = torch.where(m0[:, None], p, P0); cnt = cnt + (m0 | m1).long()
        ok = cnt == 2; P0 = P0[ok]; P1 = P1[ok]; n0 = n0[ok]; n1 = n1[ok]
        dd = P1 - P0; s = dd[:, 1] * n0 - dd[:, 0] * n1; f = (s < 0)[:, None]
        Q0 = torch.where(f, P1, P0); Q1 = torch.where(f, P0, P1)
        return Q0, Q1
def gfill(P, x0, y0, r, W, H, d, LIM=12000000):
    """XS.fill (non-zero winding union) on the GPU -> bool (H,W)."""
    if P is None or not P[0].shape[0]: return torch.zeros((H, W), dtype=torch.bool, device=d)
    Q0, Q1 = P; xa, ya, xb, yb = Q0[:, 0], Q0[:, 1], Q1[:, 0], Q1[:, 1]
    lo = torch.minimum(ya, yb); hi = torch.maximum(ya, yb)
    j0 = torch.clamp(torch.ceil((lo - y0) / r - 0.5).long(), 0, H); j1 = torch.clamp(torch.ceil((hi - y0) / r - 0.5).long(), 0, H)
    n = j1 - j0; ks = torch.nonzero(n > 0)[:, 0]
    if not ks.numel(): return torch.zeros((H, W), dtype=torch.bool, device=d)
    nk = n[ks]; cs = torch.cumsum(nk, 0); tog = torch.zeros(H * (W + 1), dtype=torch.float64, device=d); a = 0; m_ = ks.numel()
    while a < m_:
        base = int(cs[a - 1]) if a else 0; b = int(torch.searchsorted(cs, torch.tensor(base + LIM, device=d), right=True)); b = min(max(b, a + 1), m_)
        c_ = nk[a:b]; tot = int(c_.sum()); idx = ks[a:b][torch.repeat_interleave(torch.arange(b - a, device=d), c_)]
        loc = torch.arange(tot, device=d) - torch.repeat_interleave(torch.cumsum(c_, 0) - c_, c_); jj = j0[idx] + loc
        yc = y0 + (jj.double() + 0.5) * r; t = (yc - ya[idx]) / (yb[idx] - ya[idx]); xc = xa[idx] + t * (xb[idx] - xa[idx])
        ii = torch.clamp(torch.ceil((xc - x0) / r - 0.5).long(), 0, W); w = torch.where(yb[idx] < ya[idx], 1.0, -1.0).double()
        tog += torch.bincount(jj * (W + 1) + ii, weights=w, minlength=H * (W + 1)); a = b
    return (torch.cumsum(tog.reshape(H, W + 1), 1)[:, :W].abs() > 0.5)

def raft_area(path, fp, XS_):
    d = dev(); temp_gate(); T32 = load_T32(path); z0 = float(fp["zmin"])
    zmin = T32[:, :, 2].min(1); sel = zmin.astype(np.float64) < z0 + 1.0
    if not sel.any(): return 0.0
    S = GSlicer(np.ascontiguousarray(T32[sel]), None, d); x0, y0 = float(fp["xmin"]) - 1.0, float(fp["ymin"]) - 1.0
    W = int((float(fp["xmax"]) - x0) / 0.25) + 3; H = int((float(fp["ymax"]) - y0) / 0.25) + 3
    return float(int(gfill(S.segs(z0 + RAFT_BAND_MM), x0, y0, 0.25, W, H, d).sum())) * 0.0625
def floor_area(path, fp, XS_, FLOOR_SLAB):
    d = dev(); temp_gate(); T32 = load_T32(path); z0 = float(fp["zmin"])
    sel = T32[:, :, 2].max(1).astype(np.float64) <= z0 + FLOOR_SLAB
    if not sel.any(): return 0.0
    T = torch.from_numpy(np.ascontiguousarray(T32[sel])).to(d).double(); a = T[:, 1] - T[:, 0]; b = T[:, 2] - T[:, 0]
    c0 = a[:, 1] * b[:, 2] - a[:, 2] * b[:, 1]; c1 = a[:, 2] * b[:, 0] - a[:, 0] * b[:, 2]; c2 = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
    return float(0.5 * torch.sqrt((c0 * c0 + c1 * c1) + c2 * c2).sum())
def peel_profile(path, fp, XS_, ndimage, PEEL_EXP, PEEL_DZ):
    d = dev(); temp_gate(); T32 = load_T32(path); z0 = float(fp["zmin"]); top = float(fp["zmax"]) - z0; r = 0.5
    x0, y0 = float(fp["xmin"]) - 1.0, float(fp["ymin"]) - 1.0; W = int((float(fp["xmax"]) - x0) / r) + 3; H = int((float(fp["ymax"]) - y0) / r) + 3
    S = GSlicer(T32, z0, d); s135, s2, area = [], [], []
    for z in np.arange(RAFT_BAND_MM, top, PEEL_DZ):
        g = gfill(S.segs(float(z)), x0, y0, r, W, H, d)
        if not bool(g.any()): s135.append(0.0); s2.append(0.0); area.append(0.0); continue
        cl = closing(torch.nn.functional.pad(g, (4, 4, 4, 4)), 2)[4:-4, 4:-4] | g
        lab, n = ndimage.label(cl.cpu().numpy(), np.ones((3, 3))); gc = g.cpu().numpy()
        A = ndimage.sum(gc, lab, np.arange(1, n + 1)) * r * r
        s135.append(round(float((A ** PEEL_EXP).sum()) / 1000.0, 2)); s2.append(round(float((A ** 2).sum()) / 1e6, 4)); area.append(round(float(A.sum()), 1))
    return dict(s135=s135, s2=s2, area=area)

# ---- mkm_fullheight.piece: slicing + the two distance transforms ----
def dilate_disc(g, r2):
    """cells with a True cell of g within integer offsets dx^2+dy^2 <= r2 (exact, no float)."""
    H, W = g.shape; rr = int(math.isqrt(r2)); out = torch.zeros_like(g); gf = g.float()[None, None]
    for dy in range(-rr, rr + 1):
        hw = int(math.isqrt(r2 - dy * dy)); h = torch.nn.functional.max_pool2d(gf, (1, 2 * hw + 1), 1, (0, hw))[0, 0] > 0.5     # horizontal run of half-width hw
        if dy >= 0: out[dy:] |= h[:H - dy] if dy else h
        else: out[:dy] |= h[-dy:]
    return out
def edt_sq(free, chunk_bytes=200000000):
    """exact squared Euclidean distance (int32) of every cell to the nearest True cell of `free` (H,W)."""
    H, W = free.shape; d = free.device; BIG = 1000000000
    idx = torch.arange(H, device=d)[:, None].expand(H, W)
    left = torch.cummax(torch.where(free, idx, torch.full_like(idx, -BIG)), 0).values; right = torch.flip(torch.cummin(torch.flip(torch.where(free, idx, torch.full_like(idx, BIG)), [0]), 0).values, [0])
    dy = torch.minimum(idx - left, right - idx).clamp(max=30000); a = (dy * dy).int(); xs = torch.arange(W, device=d, dtype=torch.int32)
    best = torch.full((H, W), BIG, dtype=torch.int32, device=d); B = max(1, chunk_bytes // (4 * H * W))
    for c in range(0, W, B):
        xp = xs[c:c + B]; cand = (xs[None, :, None] - xp[None, None, :]) ** 2 + a[:, c:c + B][:, None, :]; best = torch.minimum(best, cand.min(2).values)
    return best
def fh_piece(path, XS_, Z0, DZ, R_, edt="gpu", area_dz=None, area_only=False):
    """per-layer slicing of one piece. Default: (area[], peel[]) both on the Z0 + DZ*i grid (unchanged). area_dz given: area[] is returned at EVERY area_dz step from Z0 (cheap: slice + fill + one sum per layer, no distance transform), peel[] stays on the DZ grid; area_only=True skips the peel loop (peel[] returned empty)."""
    from scipy import ndimage
    d = dev(); temp_gate(); T32 = load_T32(path); zmn = float(T32[:, :, 2].min()); h = float(T32[:, :, 2].max()) - zmn; zs = zmn
    x0, y0 = float(T32[:, :, 0].min()) - 8, float(T32[:, :, 1].min()) - 8
    W = int((float(T32[:, :, 0].max()) - x0 + 8) / R_) + 2; H = int((float(T32[:, :, 1].max()) - y0 + 8) / R_) + 2
    S = GSlicer(T32, zs, d); A, F = [], []
    A_every = None
    if area_dz is not None:
        tot = []
        for z in np.arange(Z0, h, area_dz):
            tot.append(gfill(S.segs(float(z)), x0, y0, R_, W, H, d).sum())
        A_every = np.array([round(int(v) * R_ * R_, 1) for v in (torch.stack(tot).cpu().tolist() if tot else [])], float)
        if area_only: return A_every, np.array([], float)
    for z in np.arange(Z0, h, DZ):
        g = gfill(S.segs(float(z)), x0, y0, R_, W, H, d)
        if not bool(g.any()): A.append(0.0); F.append(0.0); continue
        free = ~dilate_disc(g, 143)               # scipy: edt(~g)*R >= 6.0  <=>  d^2 >= 144
        if edt == "gpu":
            d2 = edt_sq(free); L = torch.sqrt(d2.double()) * R_; v = (L[g] ** 2)
            A.append(round(float(int(g.sum())) * R_ * R_, 1)); F.append(round(float(v.sum()) * R_ * R_ / 1e3, 2))
        else:
            L = ndimage.distance_transform_edt(~free.cpu().numpy()) * R_; gc = g.cpu().numpy()
            A.append(round(float(gc.sum()) * R_ * R_, 1)); F.append(round(float((L[gc] ** 2).sum()) * R_ * R_ / 1e3, 2))
    return (A_every if A_every is not None else np.array(A, float)), np.array(F, float)


def foot_halo(v32, R_, halo_iters):
    """the foot/halo cache entry of mkm_gpupool10 (vertex scatter -> closing 3 -> fill holes -> area, dilated area); v32 = raw['v'] (n,3,3) float32."""
    d = dev(); temp_gate(); v = torch.from_numpy(np.ascontiguousarray(v32.reshape(-1, 3))).to(d); v = v[torch.isfinite(v).all(1)]
    mn = v[:, :2].min(0).values; ix = ((v[:, 0] - mn[0]) / R_).long() + 8; iy = ((v[:, 1] - mn[1]) / R_).long() + 8
    g = torch.zeros((int(ix.max()) + 9, int(iy.max()) + 9), dtype=torch.bool, device=d); g[ix, iy] = True; g = fill_holes(closing(g, 3))
    h = g
    for _ in range(halo_iters): h = dil(h)
    return dict(foot=float(int(g.sum()) * R_ * R_), halo=float(int(h.sum()) * R_ * R_))
