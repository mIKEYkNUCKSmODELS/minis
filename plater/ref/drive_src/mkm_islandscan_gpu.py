"""MKM ISLAND SCAN - GPU PORT (Claude 11/14 02:40, Mike said "lets try that"). Same scan as mkm_islandscan.py, same numbers:
the layer slicing (raster) and the 'is anything cured here that has nothing under it' test run on the graphics card.
Only a layer that actually has a candidate island (or a floating chain still open) drops to the ORIGINAL cpu code for that layer,
so the island list is the original scan's by construction. Read-only. Writes <plate><suffix> (default _ISLANDS_GPU.txt while on trial).
usage: mkm_islandscan_gpu.py PLATE.stl [...]"""
import sys, os, struct, time
import numpy as np
import torch, torch.nn.functional as F
from scipy import ndimage
dev = torch.device("cuda")
dt = np.dtype([('n', '<f4', (3,)), ('v', '<f4', (3, 3)), ('a', '<u2')])
PX, PY = 297.5, 164.0; R = 0.1; LH = float(os.environ.get("MKM_ISL_LH", 0.05)); MINAREA = 0.04; TOL = 2
X0, Y0 = -PX / 2 - 1, -PY / 2 - 1; NX, NY = int((PX + 2) / R), int((PY + 2) / R)
ST = ndimage.generate_binary_structure(2, 2); SUF = os.environ.get("MKM_ISL_SUFFIX", "_ISLANDS_GPU.txt")
D = torch.float64

def raster(v, z):
    zs = v[:, :, 2]; idx_l, xy_l = [], []
    for a, b in ((0, 1), (1, 2), (2, 0)):
        za, zb = zs[:, a], zs[:, b]; m = (za - z) * (zb - z) < 0
        if not bool(m.any()): continue
        t = (z - za[m]) / (zb[m] - za[m])
        idx_l.append(torch.nonzero(m).squeeze(1)); xy_l.append(v[m, a, :2] + (v[m, b, :2] - v[m, a, :2]) * t[:, None])
    if not idx_l: return torch.zeros((NY, NX), dtype=torch.bool, device=dev)
    idx = torch.cat(idx_l); xy = torch.cat(xy_l); o = torch.argsort(idx, stable=True); idx, xy = idx[o], xy[o]
    Fz = torch.zeros(1, dtype=torch.bool, device=dev)
    keep = torch.cat([idx[1:] == idx[:-1], Fz]); p0, p1 = xy[keep], xy[torch.cat([Fz, keep[:-1]])]
    tv = v[idx[keep]]; fn = torch.linalg.cross(tv[:, 1] - tv[:, 0], tv[:, 2] - tv[:, 0])
    tdir = torch.stack([-fn[:, 1], fn[:, 0]], 1); flip = ((p1 - p0) * tdir).sum(1) < 0
    p0, p1 = torch.where(flip[:, None], p1, p0), torch.where(flip[:, None], p0, p1)
    fy0 = (p0[:, 1] - Y0) / R - 0.5; fy1 = (p1[:, 1] - Y0) / R - 0.5
    lo = torch.ceil(torch.minimum(fy0, fy1)).long(); hi = torch.ceil(torch.maximum(fy0, fy1)).long() - 1
    cnt = torch.clamp(hi - lo + 1, min=0); tot = int(cnt.sum())
    T = torch.zeros(NY * (NX + 1), dtype=torch.int32, device=dev)
    if tot:
        si = torch.repeat_interleave(torch.arange(len(p0), device=dev), cnt)
        rows = lo[si] + (torch.arange(tot, device=dev) - (torch.cumsum(cnt, 0) - cnt)[si])
        t = (rows.to(D) - fy0[si]) / (fy1[si] - fy0[si])
        fx = (p0[si, 0] + (p1[si, 0] - p0[si, 0]) * t - X0) / R - 0.5
        cols = torch.clamp(torch.ceil(fx).long(), 0, NX); ok = (rows >= 0) & (rows < NY)
        w = torch.where(fy1[si] > fy0[si], 1, -1).to(torch.int32)
        T.index_add_(0, rows[ok] * (NX + 1) + cols[ok], w[ok])
    S = torch.cumsum(T.view(NY, NX + 1), 1)[:, :NX] != 0
    L = torch.clamp((p1 - p0).abs().max(1).values / R, min=1).long(); k = torch.repeat_interleave(torch.arange(len(p0), device=dev), L + 1)
    f = (torch.arange(k.numel(), device=dev) - (torch.cumsum(L + 1, 0) - (L + 1))[k]).to(D) / L[k].to(D)
    pts = p0[k] + (p1[k] - p0[k]) * f[:, None]
    cx = ((pts[:, 0] - X0) / R).long(); cy = ((pts[:, 1] - Y0) / R).long()
    ok = (cx >= 0) & (cy >= 0) & (cx < NX) & (cy < NY); S[cy[ok], cx[ok]] = True
    return S

for path in sys.argv[1:]:
    t0 = time.time()
    mm = np.memmap(path, dtype=np.uint8, mode='r'); n = struct.unpack('<I', bytes(mm[80:84]))[0]
    V = np.ascontiguousarray(np.fromfile(path, dtype=dt, count=n, offset=84)['v']); del mm
    zmn = V[:, :, 2].min(1); zmx = V[:, :, 2].max(1)
    L0 = np.ceil(zmn / LH - 0.5).astype(np.int64); L1 = np.floor(zmx / LH - 0.5).astype(np.int64); span = np.clip(L1 - L0 + 1, 0, None)
    tri_of = np.repeat(np.arange(n), span); lay = L0[tri_of] + (np.arange(tri_of.size) - np.repeat(np.cumsum(span) - span, span))
    o = np.argsort(lay, kind='stable'); lay, tri_of = lay[o], tri_of[o]; nl = int(lay.max()) + 1 if lay.size else 0
    starts = np.searchsorted(lay, np.arange(nl + 1)); del lay, o
    Vg = torch.from_numpy(V).to(dev); del V
    prev = None; fid = None; chains = {}; nid = 0; fast = slow = 0; tload = time.time() - t0
    for k in range(nl):
        sel = tri_of[starts[k]:starts[k + 1]]; z = (k + 0.5) * LH + 1e-4
        S = raster(Vg[torch.from_numpy(sel).to(dev)].to(D), z) if sel.size else torch.zeros((NY, NX), dtype=torch.bool, device=dev)
        newfid = None
        if k % 100 == 0: print(f'  layer {k}/{nl}  {time.time()-t0:.0f}s  gpu-only {fast} cpu {slow}', flush=True)
        if k >= 3 and bool(S.any()):
            if fid is None:
                supd_g = F.max_pool2d(prev[None, None].half(), 2 * TOL + 1, 1, TOL)[0, 0] > 0
                if not bool((S & ~supd_g).any()): fast += 1; prev = S; continue
            slow += 1; Sc = S.cpu().numpy(); pc = prev.cpu().numpy(); fidc = fid if fid is not None else np.zeros((NY, NX), np.int32)
            newfid = np.zeros((NY, NX), np.int32)
            supd = ndimage.binary_dilation(pc & (fidc == 0), ST, iterations=TOL)
            fl_d = ndimage.grey_dilation(fidc, footprint=ST, size=None) if fidc.any() else fidc
            if fidc.any():
                for _ in range(TOL - 1): fl_d = ndimage.grey_dilation(fl_d, footprint=ST)
            lab, nc = ndimage.label(Sc, ST)
            if nc:
                idxs = np.arange(1, nc + 1); sup = ndimage.maximum(supd, lab, idxs)
                flo = ndimage.maximum(fl_d, lab, idxs) if fidc.any() else np.zeros(nc)
                cnts = np.bincount(lab.ravel(), minlength=nc + 1); objs = None
                for c in np.where(sup == 0)[0] + 1:
                    a = cnts[c] * R * R; cid = int(flo[c - 1])
                    if cid == 0:
                        if a < MINAREA: continue
                        if objs is None: objs = ndimage.find_objects(lab)
                        sl = objs[c - 1]; ys, xs = np.nonzero(lab[sl] == c); nid += 1; cid = nid
                        chains[cid] = dict(z0=z, a=a, x=X0 + (xs.mean() + sl[1].start + 0.5) * R, y=Y0 + (ys.mean() + sl[0].start + 0.5) * R, layers=1, merged=None)
                    else:
                        ch = chains[cid]; ch['layers'] += 1; ch['a'] = max(ch['a'], a)
                    if objs is None: objs = ndimage.find_objects(lab)
                    sl = objs[c - 1]; newfid[sl][lab[sl] == c] = cid
                for c in np.where((sup > 0) & (flo > 0))[0] + 1:
                    ch = chains.get(int(flo[c - 1]))
                    if ch and ch['merged'] is None: ch['merged'] = z
            if not newfid.any(): newfid = None
        prev, fid = S, newfid
    isl = sorted(chains.values(), key=lambda c: c['z0'])
    for c in isl: c['h'] = (c['merged'] - c['z0']) if c['merged'] is not None else c['layers'] * LH
    risky = [c for c in isl if c['h'] > 0.15 or c['a'] >= 1.0]
    with open(path[:-4] + SUF, "w", encoding="utf-8") as fh:
        fh.write(f"ISLAND SCAN {os.path.basename(path)}  layers {LH} mm, grid {R} mm, union fill, {nl} layers, {time.time()-t0:.0f}s\n")
        fh.write(f"ISLANDS: {len(isl)}   RISK (floats > 0.15 mm before joining supported material, or >= 1 mm2): {len(risky)}\n")
        for c in isl:
            tag = "RISK " if c in risky else "joins"
            fh.write(f"  {tag} Z {c['z0']:6.2f} mm  max {c['a']:5.2f} mm2  floats {c['h']:.2f} mm  "
                     f"{'joins at Z %.2f' % c['merged'] if c['merged'] is not None else 'NEVER JOINS'}  at plate X {c['x']:7.1f} Y {c['y']:7.1f}\n")
    print(f"{os.path.basename(path)}: {len(isl)} island(s), RISK {len(risky)}, max float {max([c['h'] for c in isl], default=0):.2f} mm, "
          f"max area {max([c['a'] for c in isl], default=0):.2f} mm2  [{time.time()-t0:.0f}s; load {tload:.0f}s; gpu-only layers {fast}, cpu layers {slow}]", flush=True)
    del Vg; torch.cuda.empty_cache()
