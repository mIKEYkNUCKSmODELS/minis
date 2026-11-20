"""MKM GPU PLATE VIEW (Claude 11/14 04:15; Mike: "why are we getting blobs? can't we get the real footprint immediately").
Top-down picture of a laid-out plate from the real geometry, on the graphics card: every vertex of every piece is dropped onto a 0.2 mm grid
keeping the highest point (a height map), shaded so the models read as models, tinted per piece. What actually TOUCHES the plate (the first
0.3 mm - rafts/feet) is drawn in bright outline, so the real contact footprint is visible apart from the overhanging shadow.
usage: mkm_gpuview.py PLATE_DIR   (uses RING_MANIFEST.json; writes <tag>_VIEW.png)"""
import sys, os, json, numpy as np, torch, time
from PIL import Image, ImageDraw
dev = torch.device("cuda"); DT = np.dtype([('n', '<f4', (3,)), ('v', '<f4', (3, 3)), ('a', '<u2')]); PX, PY, R = 297.5, 164.0, 0.2
W, H = int(PX / R), int(PY / R); d = sys.argv[1]; t0 = time.time()
pm = json.load(open(os.path.join(d, "RING_MANIFEST.json"), encoding="utf-8"))[0]; raw = np.memmap(os.path.join(d, pm["file"]), dtype=DT, mode="r", offset=84)
hm = torch.zeros(H * W, device=dev); pid = torch.zeros(H * W, dtype=torch.int32, device=dev); touch = torch.zeros(H * W, dtype=torch.bool, device=dev); rng = np.random.default_rng(7)
cols = [(0, 0, 0)] + [tuple(int(c) for c in rng.integers(90, 235, 3)) for _ in pm["pieces"]]; contact = []
for i, p in enumerate(pm["pieces"], 1):
    v = torch.from_numpy(np.ascontiguousarray(raw[p["off"]:p["off"] + p["n"]]["v"]).reshape(-1, 3)).to(dev); v = v[torch.isfinite(v).all(1)]
    cx = ((PX / 2 - v[:, 0]) / R).long().clamp(0, W - 1); cy = ((PY / 2 - v[:, 1]) / R).long().clamp(0, H - 1); idx = cy * W + cx
    ph = torch.zeros(H * W, device=dev).scatter_reduce_(0, idx, v[:, 2].clamp(min=0.01), "amax"); m = ph > 0
    hm = torch.where(m & (ph >= hm), ph, hm); pid = torch.where(m, torch.full_like(pid, i), pid)
    lo = v[:, 2] < 0.3; t = torch.zeros(H * W, dtype=torch.bool, device=dev); t[idx[lo]] = True; touch |= t; contact.append(float(t.sum()) * R * R)
hm2 = hm.view(H, W); gy, gx = torch.gradient(hm2); shade = (0.55 + 0.45 * torch.tanh((-gx - gy) * 0.9)).clamp(0.15, 1.0); depth = 0.55 + 0.45 * (hm2 / hm2.max())
lut = torch.tensor(cols, dtype=torch.float32, device=dev); img = lut[pid.view(H, W).long()] * (shade * depth)[..., None]
tv = touch.view(H, W).float(); er = -torch.nn.functional.max_pool2d(-tv[None, None], 3, 1, 1)[0, 0]; edge = (tv > 0) & (er == 0)
img[edge] = torch.tensor([255.0, 255.0, 255.0], device=dev)
im = Image.fromarray(img.clamp(0, 255).byte().cpu().numpy()); S = 1; dr = ImageDraw.Draw(im)
for a in (1, 2): dr.line([(a * W // 3, 0), (a * W // 3, H)], fill=(90, 90, 90)); dr.line([(0, a * H // 3), (W, a * H // 3)], fill=(90, 90, 90))
rr = 16 / R; dr.ellipse([W / 2 - rr, H / 2 - rr, W / 2 + rr, H / 2 + rr], outline=(255, 255, 0))
for i, p in enumerate(pm["pieces"], 1): dr.text(((PX / 2 - p["x"]) / R - 6, (PY / 2 - p["y"]) / R - 6), str(i), fill=(255, 255, 0))
out = os.path.join(d, pm["tag"] + "_VIEW.png"); im.save(out)
print(f"{out} | {len(pm['pieces'])} pieces | real contact on the plate {sum(contact):.0f} mm2 ({100 * sum(contact) / (PX * PY):.0f}% of bed) | shadow {float((pid > 0).sum()) * R * R:.0f} mm2 ({100 * float((pid > 0).sum()) * R * R / (PX * PY):.0f}%) | {time.time()-t0:.1f}s")
