# MKM GPU issue detectors - suction cups, resin traps, islands, overhangs - run on the GPU straight from a slice.
# Behaviour modelled on the UVtools project's issue detection (github.com/sn4k3/UVtools); this is an independent torch
# re-implementation, not their code, and is NOT yet proven count-for-count against UVtools (see parity test).
# Definitions used here (layer 0 = on the build plate, last layer = at the film):
#   open     = empty cells connected in-plane to the edge of the screen (resin can flow in).
#   hollow   = empty cells walled in on their own layer.
#   vented   = hollow cells that connect, through earlier layers, to open resin (vent toward the plate side).
#   suction  = hollow and not vented: the film is pulled against a sealed pocket on that layer.
#   resin trap  = suction pocket that is also sealed going on toward the film (never opens).
#   suction cup = suction pocket that does open toward the film side later.
#   island   = cured region on a layer with nothing cured under any part of it.
import sys, os, json, time, zlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch, torch.nn.functional as F
import mkm_slice_reader as R

def _dil4(x):
    xf = x[None, None].float()
    return (F.max_pool2d(xf, (3, 1), 1, (1, 0)) + F.max_pool2d(xf, (1, 3), 1, (0, 1)))[0, 0] > 0

def _flood(seed, allowed, cap=6000, check=8):
    cur = seed & allowed; last = -1; it = 0
    while it < cap:
        for _ in range(check):
            cur = _dil4(cur) & allowed
        it += check
        c = int(cur.sum())
        if c == last: break
        last = c
    return cur

def _up(x, k, shape):
    y = x.repeat_interleave(k, 0).repeat_interleave(k, 1)
    out = torch.zeros(shape, dtype=torch.bool, device=x.device)
    h, w = min(shape[0], y.shape[0]), min(shape[1], y.shape[1]); out[:h, :w] = y[:h, :w]
    return out

def _open(empty):
    """empty cells connected in-plane to the screen edge. Coarse-to-fine so it converges fast and stays exact."""
    seed = torch.zeros_like(empty); seed[0, :] = True; seed[-1, :] = True; seed[:, 0] = True; seed[:, -1] = True
    cured = (~empty)[None, None].float()
    for k in (16, 4):
        ce = F.max_pool2d(cured, k)[0, 0] == 0            # coarse cell is empty only if every fine cell in it is empty
        cs = F.max_pool2d(seed[None, None].float(), k)[0, 0] > 0
        seed = seed | _up(_flood(cs & ce, ce), k, empty.shape)
    return _flood(seed, empty)

def _pack(m):
    return zlib.compress(np.packbits(m.cpu().numpy()).tobytes(), 1)
def _unpack(b, shape, dev):
    a = np.unpackbits(np.frombuffer(zlib.decompress(b), np.uint8))[:shape[0] * shape[1]].reshape(shape).astype(bool)
    return torch.from_numpy(a).to(dev)

def plate_issues_gpu(path, ds=4, device=None, progress=None):
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    S = R.open_slice(path); N = S.n_layers; w, h = S.res
    cell_mm2 = (S.pixel_um * ds / 1000.0) ** 2
    t0 = time.time()
    hollow_px = np.zeros(N, np.int64); suction_px = hollow_px.copy(); cured_px = hollow_px.copy(); island_px = hollow_px.copy(); overhang_px = hollow_px.copy()
    cup_px = hollow_px.copy(); trap_px = hollow_px.copy()
    P_open = [None] * N; P_hol = [None] * N; P_ven = [None] * N
    prev_open = prev_ven = prev_cured = None; shape = None
    for L in range(N):                                    # pass 1: plate -> film
        full = S.layer_gpu(L, dev)
        cur = F.avg_pool2d(full[None, None].float(), ds)[0, 0] >= 0.5
        shape = tuple(cur.shape); empty = ~cur
        cured_px[L] = int(cur.sum())
        if cured_px[L] == 0:
            op = empty; hol = torch.zeros_like(cur); ven = hol
        else:
            op = _open(empty); hol = empty & ~op
            if L == 0 or not bool(hol.any()):
                ven = torch.zeros_like(cur)
            else:
                ven = _flood(hol & (prev_open | prev_ven), hol)
            if prev_cured is not None:
                sup = cur & _dil4(prev_cured)             # cured cells with something cured under or beside-under them
                isl = cur & ~_flood(sup, cur)
                island_px[L] = int(isl.sum()); overhang_px[L] = int((cur & ~sup).sum())
        hollow_px[L] = int(hol.sum()); suction_px[L] = int((hol & ~ven).sum())
        P_open[L] = _pack(op); P_hol[L] = _pack(hol) if hollow_px[L] else None; P_ven[L] = _pack(ven) if hollow_px[L] else None
        prev_open, prev_ven, prev_cured = op, ven, cur
        if progress and L % 500 == 0: progress("pass1 %d/%d" % (L, N))
    nxt_open = nxt_A = None
    for L in range(N - 1, -1, -1):                        # pass 2: film -> plate
        op = _unpack(P_open[L], shape, dev)
        if P_hol[L] is None:
            A = torch.zeros(shape, dtype=torch.bool, device=dev)
        else:
            hol = _unpack(P_hol[L], shape, dev); ven = _unpack(P_ven[L], shape, dev)
            A = hol.clone() if L == N - 1 else _flood(hol & (nxt_open | nxt_A), hol)
            suc = hol & ~ven
            cup_px[L] = int((suc & A).sum()); trap_px[L] = int((suc & ~A).sum())
        nxt_open, nxt_A = op, A
    def summ(a):
        nz = np.nonzero(a)[0]
        return dict(layers_affected=int(len(nz)), first_layer=int(nz[0]) if len(nz) else None, last_layer=int(nz[-1]) if len(nz) else None,
                    total_mm2_layers=float(a.sum() * cell_mm2), worst_layer=int(a.argmax()) if len(nz) else None, worst_mm2=float(a.max() * cell_mm2))
    return dict(file=os.path.basename(path), properties=S.properties(), grid_mm=S.pixel_um * ds / 1000.0, seconds=round(time.time() - t0, 1), device=dev,
                hollow=summ(hollow_px), suction=summ(suction_px), suction_cup=summ(cup_px), resin_trap=summ(trap_px), island=summ(island_px), overhang=summ(overhang_px),
                per_layer=dict(cured_mm2=(cured_px * cell_mm2).round(2).tolist(), suction_mm2=(suction_px * cell_mm2).round(2).tolist(),
                               cup_mm2=(cup_px * cell_mm2).round(2).tolist(), trap_mm2=(trap_px * cell_mm2).round(2).tolist(), island_mm2=(island_px * cell_mm2).round(2).tolist()))

if __name__ == "__main__":
    r = plate_issues_gpu(sys.argv[1], progress=lambda s: print(s, flush=True))
    out = sys.argv[2] if len(sys.argv) > 2 else None
    if out: json.dump(r, open(out, "w"))
    print(json.dumps({k: v for k, v in r.items() if k != "per_layer"}, indent=1))
