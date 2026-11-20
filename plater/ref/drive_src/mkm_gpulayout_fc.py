"""MKM GPU LAYOUT SEARCH - trial (Claude 11/14 03:05; Mike: "couldn't the gpu also try multiple layouts quickly based off of all the equations").
Tries thousands of whole layouts on the graphics card from the real footprints and keeps the best by the same numbers we judge plates on:
nothing overlapping (>= GAP mm between pieces), EDGE mm from the bed edge, centre circle clear, raft pull even across the 8 outer zones,
centre zone lightest. Cured-area-per-layer and total pull do not depend on where pieces sit, so they are checked once up front.
usage: mkm_gpulayout.py JOB.json     JOB = {out, tag, lift, pieces:[{set,file,path}]}"""
import sys, os, json, struct, time, hashlib, numpy as np
from scipy import ndimage
_T = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))      # tree root (relative: no absolute import path)
sys.path.insert(0, os.path.join(_T, "plate_packer")); sys.path.insert(0, os.path.join(_T, "physics"))
import mkm_ringpack as RP, mkm_fullheight as FH
import mkm_plate_verdict as PV, mkm_planner_params as _PP      # physics dir is on sys.path through mkm_ringpack
DT = np.dtype([('n', '<f4', (3,)), ('v', '<f4', (3, 3)), ('a', '<u2')])
PX, PY, R, EDGE, GAP, CLEAR = 297.5, 164.0, 0.5, 3.0, 3.0, 0.0  # 32 mm centre floor removed - Mike 2026-11-17 10:09 (undo: C:\MKM_TMP\backup\centre_floor_20261117)
W, H = int(PX / R), int(PY / R); B = int(os.environ.get("GL_BATCH", 384)); ROUNDS = int(os.environ.get("GL_ROUNDS", 12)); AREA_LINE = 5296.0
NAMES = ['back-left', 'back-centre', 'back-right', 'middle-left', 'middle-centre', 'middle-right', 'front-left', 'front-centre', 'front-right']
def rot(a, k):      # rotate xyz array (...,3) by 90*k about Z
    c, s = [(1, 0), (0, 1), (-1, 0), (0, -1)][k]; o = a.copy(); o[..., 0] = c * a[..., 0] - s * a[..., 1]; o[..., 1] = s * a[..., 0] + c * a[..., 1]; return o
def dense_pts(tv):      # 11/14 04:35: footprints from FILLED triangles, not just corner points. A long flat face (staff, arm, raft) has corners
    # only at its ends, so a corner-point footprint left holes and the verify scan caught a real overlap (FOG_OCT26 P1, Unna's arm).
    xy = tv[:, :, :2].astype(np.float64); e = np.stack([np.linalg.norm(xy[:, 1] - xy[:, 0], axis=1), np.linalg.norm(xy[:, 2] - xy[:, 1], axis=1), np.linalg.norm(xy[:, 0] - xy[:, 2], axis=1)], 1).max(1)
    n = np.ceil(e / 0.25).astype(int); out = [tv.reshape(-1, 3)]
    for lvl in (2, 4, 8, 16, 32, 64, 128, 256, 512, 1024):
        sel = np.nonzero((n > lvl // 2) & ((n <= lvl) | (lvl == 1024)))[0]
        if not sel.size: continue
        i, j = np.meshgrid(np.arange(lvl + 1), np.arange(lvl + 1), indexing="ij"); m = (i + j) <= lvl
        wa = (i[m] / lvl).astype(np.float32); wb = (j[m] / lvl).astype(np.float32); wc = 1 - wa - wb; step = max(1, 4000000 // wa.size)
        for c0 in range(0, sel.size, step):
            t = tv[sel[c0:c0 + step]].astype(np.float32)
            out.append((wa[None, :, None] * t[:, 0][:, None, :] + wb[None, :, None] * t[:, 1][:, None, :] + wc[None, :, None] * t[:, 2][:, None, :]).reshape(-1, 3))
    return np.concatenate(out)

CACHE_DIR = os.path.join(_T, "gpu_scan", "fp_cache"); CACHE_VER = "fc1"
def load_raw(path):      # exactly the original read: NaN-triangle filter, then drop-to-plate if the file does not start at z=0
    raw = np.fromfile(path, dtype=DT, offset=84); raw = raw[np.isfinite(raw['v'].reshape(-1, 9)).all(1)]; zmn = float(raw['v'][:, :, 2].min())
    if abs(zmn + 0.005) > 0.02:
        v_ = raw['v'].copy(); v_[:, :, 2] -= np.float32(zmn + 0.005); raw['v'] = v_
    return raw, zmn
def cache_file(path):
    import mkm_contentkey as CK; key = f"{CACHE_VER}|{CK.content_sha(path)}"      # 20261117 content key
    return os.path.join(CACHE_DIR, os.path.splitext(os.path.basename(path))[0][:40].replace(" ", "_") + "_" + hashlib.sha1(key.encode("utf-8")).hexdigest()[:20] + ".npz")
def compute_piece(q):
    m = RP.model(q["path"], q["set"]); raw, zmn = load_raw(q["path"])
    a = np.asarray(FH.piece(q["path"])[0], float)
    rots = []; dp = dense_pts(raw["v"])
    for k in range(4):
        v = rot(dp, k); v = v[np.isfinite(v).all(1) & (np.abs(v) < 1e4).all(1)]; mn = v[:, :2].min(0); pad = 4
        ix = ((v[:, 0] - mn[0]) / R).astype(int) + pad; iy = ((v[:, 1] - mn[1]) / R).astype(int) + pad
        g = np.zeros((ix.max() + pad + 1, iy.max() + pad + 1), bool); g[ix, iy] = True; g = ndimage.binary_fill_holes(ndimage.binary_closing(g, iterations=3))
        xs, ys = np.nonzero(g); c0 = (xs.min(), ys.min()); g = g[xs.min():xs.max() + 1, ys.min():ys.max() + 1]
        rots.append(dict(mask=g, mn=mn, c0=c0, pad=pad))
    return dict(raft=float(m["raft"]), h=float(m["h"]), zmn=zmn, area=a, rots=rots)
def get_piece(q):
    cf = cache_file(q["path"])
    if os.path.exists(cf):
        try:
            z = np.load(cf, allow_pickle=False)
            d = dict(raft=float(z["raft"]), h=float(z["h"]), zmn=float(z["zmn"]), area=z["area"],
                     rots=[dict(mask=z[f"mask{k}"], mn=z[f"mn{k}"], c0=(z[f"c0{k}"][0], z[f"c0{k}"][1]), pad=int(z["pad"])) for k in range(4)])
            return d, True
        except Exception: pass
    d = compute_piece(q); os.makedirs(CACHE_DIR, exist_ok=True); arrs = dict(raft=d["raft"], h=d["h"], zmn=d["zmn"], area=d["area"], pad=d["rots"][0]["pad"])
    for k, r in enumerate(d["rots"]): arrs[f"mask{k}"] = r["mask"]; arrs[f"mn{k}"] = r["mn"]; arrs[f"c0{k}"] = np.array(r["c0"])
    tmp = cf[:-4] + f".tmp{os.getpid()}.npz"; np.savez_compressed(tmp, **arrs); os.replace(tmp, cf)
    return d, False
if "--warm" in sys.argv:
    for f in [x for x in sys.argv[1:] if x != "--warm" and not x.endswith(".py")]:
        t_ = time.time(); d, hit = get_piece(dict(path=f, set="warm", file=os.path.basename(f)))
        print(f"{'hit ' if hit else 'built'} {time.time()-t_:5.1f}s  {os.path.basename(f)}", flush=True)
    sys.exit(0)
import torch, torch.nn.functional as F
dev = torch.device("cuda")
job = json.load(open(sys.argv[1], encoding="utf-8")); KN = 1.0      # RETIRED legacy lift factor (0.012*1.025*lift/120 N/mm2, unsourced). Layout ranking is invariant to this scalar; "N" in the prints below now means mm2 of raft. Verdict = real solver (mkm_gpupool10 / mkm_plate_verdict).; t0 = time.time()
P = []; prof = np.zeros(600); dropped = []; nhit = nmiss = 0; t_load = time.time()
for q in job["pieces"]:
    d, hit = get_piece(q); nhit += hit; nmiss += (not hit)
    if abs(d["zmn"] + 0.005) > 0.02: dropped.append(f"{q['file']} was {d['zmn']:+.2f} mm")
    a = d["area"]; prof[:len(a)] += a
    P.append(dict(q=q, raft=d["raft"], h=d["h"], raw=None, rots=d["rots"], pull=d["raft"] * KN))
t_load = time.time() - t_load
n = len(P); tot_raft = sum(p["raft"] for p in P)
print(f"cache: {nhit} hit, {nmiss} miss, load {t_load:.1f}s", flush=True)
print(f"{n} pieces loaded {time.time()-t0:.0f}s | total raft {tot_raft:.0f} mm2 = {tot_raft * KN:.0f} N at lift {job.get('lift', 60)} | cured-area envelope peak {prof.max():.0f} mm2 at {10 + 0.5 * prof.argmax():.1f} mm (line {AREA_LINE:.0f})", flush=True)
xx = (torch.arange(W, device=dev) + 0.5) * R - PX / 2; yy = (torch.arange(H, device=dev) + 0.5) * R - PY / 2
circle = ((xx[:, None] ** 2 + yy[None, :] ** 2) <= (CLEAR / 2) ** 2).float() * float(CLEAR > 0)  # 32 mm centre floor removed - Mike 2026-11-17 10:09 (undo: C:\MKM_TMP\backup\centre_floor_20261117)
e = int(EDGE / R) + 1; g = int(round(GAP / R)); NB = 10
frame = torch.ones((W, H), device=dev); frame[e + 6:W - e - 6, e + 6:H - e - 6] = 0
KF = {}                                                    # kernel FFTs
for i, p in enumerate(P):
    for k, r in enumerate(p["rots"]):
        ker = torch.zeros((W, H), device=dev); mk = torch.from_numpy(r["mask"]).to(dev); sw, sh = mk.shape; ker[:sw, :sh] = mk.float()
        ox = torch.arange(W, device=dev); oy = torch.arange(H, device=dev)
        cx = (ox + sw / 2) * R - PX / 2; cy = (oy + sh / 2) * R - PY / 2
        col = torch.where(cx > PX / 6, 0, torch.where(cx < -PX / 6, 2, 1)); row = torch.where(cy > PY / 6, 0, torch.where(cy < -PY / 6, 2, 1))
        inb = ((ox >= e) & (ox <= W - sw - e))[:, None] & ((oy >= e) & (oy <= H - sh - e))[None, :]
        nz = torch.nonzero(mk)
        KF[(i, k)] = dict(f=torch.conj(torch.fft.rfft2(ker)), z=(row[None, :] * 3 + col[:, None]), inb=inb, nz=nz, sw=sw, sh=sh, cx=cx, cy=cy)
best = None; tried = valid = 0; gen = torch.Generator(device=dev); gen.manual_seed(int(os.environ.get("GL_SEED", 1114)))
rafts = np.array([p["raft"] for p in P]); rng = np.random.default_rng(1114)
for rd in range(ROUNDS):
    order = np.argsort(-(rafts * (1 + (0.35 * rng.standard_normal(n) if rd else 0))))
    occ = torch.zeros((B, W, H), device=dev); load = torch.zeros((B, 9), device=dev); ok = torch.ones(B, dtype=torch.bool, device=dev)
    pos = torch.zeros((B, n, 3), dtype=torch.long, device=dev); copx = torch.zeros(B, device=dev); copy = torch.zeros(B, device=dev)
    kk = torch.randint(0, 4, (B, n), device=dev, generator=gen); jit = 0.5 + 3.0 * torch.rand(B, device=dev, generator=gen); wc = float(os.environ.get('GL_PACK', 10)) * torch.rand(B, device=dev, generator=gen)
    for i in order:
        blocked = F.max_pool2d(occ[:, None], 2 * g + 1, 1, g)[:, 0] + circle[None]; bf = torch.fft.rfft2(blocked); nf = torch.fft.rfft2(torch.clamp(F.max_pool2d(blocked[:, None], 2 * NB + 1, 1, NB)[:, 0] + frame[None], max=1.0))
        for k in range(4):
            sel = torch.nonzero((kk[:, i] == k) & ok).squeeze(1)
            if not sel.numel(): continue
            K = KF[(int(i), k)]; corr = torch.fft.irfft2(bf[sel] * K["f"][None], s=(W, H)); legal = (corr < 0.5) & K["inb"][None]
            sc = -load[sel][:, K["z"].reshape(-1)].reshape(-1, W, H) - 1e3 * (K["z"] == 4)[None] + jit[sel, None, None] * torch.rand((sel.numel(), W, H), device=dev, generator=gen)
            sc = sc + wc[sel, None, None] * torch.fft.irfft2(nf[sel] * K["f"][None], s=(W, H)) / K["nz"].shape[0]      # pack: prefer spots that hug neighbours / the bed edge
            sc = torch.where(legal, sc, torch.full_like(sc, -1e9)); mx, am = sc.reshape(sel.numel(), -1).max(1); good = mx > -1e8
            ox = am // H; oy = am % H; ok[sel[~good]] = False; s2 = sel[good]; ox, oy = ox[good], oy[good]
            if not s2.numel(): continue
            nz = K["nz"]; bi = s2[:, None].expand(-1, nz.shape[0]); occ[bi, ox[:, None] + nz[None, :, 0], oy[:, None] + nz[None, :, 1]] = 1.0
            z = K["z"][ox, oy]; load[s2, z] += p_pull if False else P[int(i)]["pull"]
            pos[s2, int(i), 0] = ox; pos[s2, int(i), 1] = oy; pos[s2, int(i), 2] = k
            copx[s2] += K["cx"][ox] * P[int(i)]["raft"]; copy[s2] += K["cy"][oy] * P[int(i)]["raft"]
        del blocked, bf, nf
    outer = load[:, [0, 1, 2, 3, 5, 6, 7, 8]]; spread = outer.max(1).values - outer.min(1).values
    cop = torch.sqrt((copx / tot_raft) ** 2 + (copy / tot_raft) ** 2)
    score = spread + 3.0 * load[:, 4] + 0.1 * cop; score = torch.where(ok, score, torch.full_like(score, 1e9)); tried += B; valid += int(ok.sum())
    j = int(score.argmin())
    if bool(ok[j]) and (best is None or float(score[j]) < best["score"]):
        best = dict(score=float(score[j]), load=load[j].cpu().numpy(), pos=pos[j].cpu().numpy(), spread=float(spread[j]), cop=float(cop[j]), occ=occ[j].cpu().numpy() > 0)
    print(f"  round {rd + 1}/{ROUNDS}: {tried} layouts tried, {valid} legal, best outer-zone spread {best['spread'] if best else float('nan'):.1f} N  [{time.time()-t0:.0f}s]", flush=True)
assert best is not None, "no legal layout found"
# ---- write the winner: STL + manifest + picture, then an exact clearance check on the cpu ----
os.makedirs(job["out"], exist_ok=True); tag = job["tag"]; hmax = max(p["h"] for p in P); fname = f"{tag}_{hmax:.0f}mm_{n}pcs_RING.stl"; pcs = []; off = 0; masks = []
for p in P: p["raw"] = load_raw(p["q"]["path"])[0]
with open(os.path.join(job["out"], fname), "wb") as fo:
    fo.write(b"MKM PLATE GPU LAYOUT 20261114".ljust(80, b" ")); fo.write(struct.pack("<I", sum(len(p["raw"]) for p in P)))
    for i, p in enumerate(P):
        ox, oy, k = (int(q) for q in best["pos"][i]); r = p["rots"][k]; t = p["raw"].copy(); t["v"] = rot(p["raw"]["v"], k); t["n"] = rot(p["raw"]["n"], k)
        dx = ox * R - PX / 2 - r["mn"][0] - (r["c0"][0] - r["pad"]) * R; dy = oy * R - PY / 2 - r["mn"][1] - (r["c0"][1] - r["pad"]) * R
        v = t["v"]; v[:, :, 0] += np.float32(dx); v[:, :, 1] += np.float32(dy); t["v"] = v; fo.write(t.tobytes())
        sw, sh = r["mask"].shape; cx = (ox + sw / 2) * R - PX / 2; cy = (oy + sh / 2) * R - PY / 2
        col = 0 if cx > PX / 6 else (2 if cx < -PX / 6 else 1); row = 0 if cy > PY / 6 else (2 if cy < -PY / 6 else 1)
        full = np.zeros((W, H), bool); full[ox:ox + sw, oy:oy + sh] = r["mask"]; masks.append(full)
        pcs.append(dict(no=i + 1, set=p["q"]["set"], file=p["q"]["file"], path=p["q"]["path"], off=off, n=len(t), k=k, x=round(cx, 1), y=round(cy, 1), where=NAMES[row * 3 + col],
                        raft=round(p["raft"]), height=round(p["h"], 1), unit=p["q"]["set"], pulled=False, zmin=float(v[:, :, 2].min()))); off += len(t)
gaps = []
for i in range(n):
    o = np.zeros((W, H), bool)
    for j2 in range(n):
        if j2 != i: o |= masks[j2]
    gaps.append(float((ndimage.distance_transform_edt(~o) * R)[masks[i]].min()))
fp = sum(m.sum() for m in masks) * R * R; zn = best["load"]
note = (f"GPU layout search (Claude 11/14): best of {tried} layouts ({valid} legal); {n} pcs, {hmax:.1f} mm; raft pull {tot_raft * KN:.0f} N at lift {job.get('lift', 60)}; "
        f"outer-zone spread {best['spread']:.1f} N, centre zone {zn[4]:.1f} N, centre of pull {best['cop']:.1f} mm off middle; smallest gap {min(gaps):.1f} mm; "
        f"real footprint {100 * fp / (PX * PY):.0f}% of bed; cured-area envelope peak {prof.max():.0f} mm2 at {10 + 0.5 * prof.argmax():.1f} mm")
json.dump([dict(tag=tag, file=fname, height=round(hmax, 1), fill=round(100 * fp / (PX * PY), 1), safe_d=CLEAR, hold=None, notes=[note], leftover=[], pieces=pcs, peel=dict(weak_supports=[]))],
          open(os.path.join(job["out"], "RING_MANIFEST.json"), "w", encoding="utf-8"), indent=1)
open(os.path.join(job["out"], "PLATE_CHECK.txt"), "w", encoding="utf-8").write(PV.plate_check_text("MKM RING PACKER CHECK  (legacy GPU layout search; NO solver is wired into this planner version, 20261116)", fname, n, PV.unsolved_no_solver("legacy planner version without a physics-solver link: use mkm_gpupool10.py or mkm_suction core_forces for a verdict"), note, [f"{p['set']} > {p['file']}" for p in pcs], sorted({p['set'] for p in pcs})))
try:
    from PIL import Image, ImageDraw
    S = 4; im = Image.new("RGB", (W * S // 2, H * S // 2 + 22), (0, 0, 0)); d = ImageDraw.Draw(im); cr = np.random.default_rng(7)
    px = np.zeros((H, W, 3), np.uint8)
    for i, m in enumerate(masks): px[m.T[::-1, ::-1]] = cr.integers(70, 240, 3)       # +x = left, +y = back (top of picture)
    im.paste(Image.fromarray(px).resize((W * S // 2, H * S // 2), Image.NEAREST), (0, 22))
    for a in (1, 2): d.line([(a * W * S // 6, 22), (a * W * S // 6, H * S // 2 + 22)], fill=(110, 110, 110)); d.line([(0, 22 + a * H * S // 6), (W * S // 2, 22 + a * H * S // 6)], fill=(110, 110, 110))
    for z in range(9): d.text(((z % 3) * W * S // 6 + 4, 24 + (z // 3) * H * S // 6), f"{zn[z]:.0f} N", fill=(255, 255, 0))
    for p in pcs: d.text(((PX / 2 - p["x"]) / R * S / 2 - 4, 22 + (PY / 2 - p["y"]) / R * S / 2 - 5), str(p["no"]), fill=(255, 255, 255))
    rr = CLEAR / 2 / R * S / 2; d.ellipse([W * S / 4 - rr, 22 + H * S / 4 - rr, W * S / 4 + rr, 22 + H * S / 4 + rr], outline=(255, 255, 255))
    d.text((4, 4), f"{tag}  {n} pcs  {hmax:.0f} mm  footprint {100 * fp / (PX * PY):.0f}%  PULL {tot_raft * KN:.0f} N  spread {best['spread']:.1f} N  min gap {min(gaps):.1f} mm  first layer {sum(1 for p in pcs if abs(p['zmin']) < 0.05)}/{n}", fill=(255, 255, 255))
    im.save(os.path.join(job["out"], f"{tag}.png"))
except Exception as ex: print("picture failed:", ex)
print('dropped to plate:', dropped if dropped else 'none needed'); print(note); print("zones N: " + " | ".join(f"{NAMES[z]} {zn[z]:.1f}" for z in range(9)))
print("first layer on plate (z min within 0.05 mm of 0):", sum(1 for p in pcs if abs(p["zmin"]) < 0.05), "of", n, "| lowest/highest zmin", min(p["zmin"] for p in pcs), max(p["zmin"] for p in pcs))
print("file:", os.path.join(job["out"], fname), f"| total {time.time()-t0:.0f}s")
