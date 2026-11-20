# MKM plate heat maps (Claude 11/20). READ-ONLY on the plater: reads each plate's recipe.json (exact source STLs + placement), slices the whole plate on the GPU
# with the plater's own slicer (gpu_scan\mkm_gpuport.GSlicer + gfill), and writes per plate: HEAT_<plate>.png + a row in heat_summary.json.
#   heat      = how many layers cure at each spot of the film (0.03 mm layers), full height  -> where the peel load sits
#   peel ring = distance in from the bed edge (the one ring closing from the edge to the centre), contour every 10 mm; bar chart = cured area per layer met by the ring in each 5 mm of travel
#   fullness  = mean cured area per layer over 1.8-30 mm (layers 60-1000), against the line fitted on the 19 printed plates (observed, not a law)
import sys, os, json, time, glob, re
sys.path.insert(0, r"C:\MKM_TMP\plater_one_build_20261117\tree\gpu_scan")
import numpy as np, torch, cv2
from PIL import Image, ImageDraw, ImageFont
import mkm_gpuport as GP
SRC = sys.argv[1]; OUT = sys.argv[2]; LINE = float(sys.argv[3]) if len(sys.argv) > 3 else 3463.0
os.makedirs(OUT, exist_ok=True); LOG = os.path.join(OUT, "heat.log"); SUMF = os.path.join(OUT, "heat_summary.json")
PX, PY, r, DZ, LH = 298.0, 164.0, 0.25, 0.5, 0.03
W = int(round(PX / r)); H = int(round(PY / r)); dev = GP.dev() or "cpu"
def log(s):
    open(LOG, "a", encoding="utf-8").write(s + "\n")
def font(sz, b=False):
    try: return ImageFont.truetype(r"C:\Windows\Fonts\arialbd.ttf" if b else r"C:\Windows\Fonts\arial.ttf", sz)
    except Exception: return ImageFont.load_default()
summ = json.load(open(SUMF)) if os.path.exists(SUMF) else {}
plates = sorted([p for p in glob.glob(os.path.join(SRC, "P*")) if os.path.exists(os.path.join(p, "recipe.json"))], key=lambda p: int(re.sub(r"\D", "", os.path.basename(p)) or 0))
yy, xx = torch.meshgrid(torch.arange(H, device=dev), torch.arange(W, device=dev), indexing="ij")
depth = torch.minimum(torch.minimum(xx + 0.5, W - xx - 0.5), torch.minimum(yy + 0.5, H - yy - 0.5)).float() * r      # mm in from the bed edge
for pd in plates:
    pid = os.path.basename(pd)
    if pid in summ and os.path.exists(os.path.join(OUT, "HEAT_%s.png" % pid)): continue
    t0 = time.time()
    try:
        rec = json.load(open(os.path.join(pd, "recipe.json"))); man = {}
        mf = os.path.join(pd, "RING_MANIFEST.json")
        if os.path.exists(mf): man = json.load(open(mf, encoding="utf-8"))[0]
        SL = []; cen = []; lo = np.array([1e9, 1e9, 1e9]); hi = -lo
        for s in rec["sources"]:
            T = GP.load_T32(s["path"]); M = np.array(s["matrix"], np.float64)
            T = (T.reshape(-1, 3).astype(np.float64) @ M[:3, :3].T + M[:3, 3]).astype(np.float32).reshape(-1, 3, 3)
            mn = T.reshape(-1, 3).min(0); mx = T.reshape(-1, 3).max(0); lo = np.minimum(lo, mn); hi = np.maximum(hi, mx); cen.append(((mn[0] + mx[0]) / 2, (mn[1] + mx[1]) / 2))
            SL.append(GP.GSlicer(np.ascontiguousarray(T), None, dev)); del T
        x0, y0 = (-PX / 2, -PY / 2) if lo[0] < -1.0 or lo[1] < -1.0 else (0.0, 0.0)
        inside = bool(lo[0] >= x0 - 0.01 and lo[1] >= y0 - 0.01 and hi[0] <= x0 + PX + 0.01 and hi[1] <= y0 + PY + 0.01)
        top = float(hi[2]); zs = np.arange(DZ / 2, top, DZ)
        cnt = torch.zeros((H, W), dtype=torch.float32, device=dev); cw = torch.zeros_like(cnt); area = []; nw = 0
        for z in zs:
            Q0 = []; Q1 = []
            for S_ in SL:
                sg = S_.segs(float(z))
                if sg is not None and sg[0].shape[0]: Q0.append(sg[0]); Q1.append(sg[1])
            if not Q0: area.append(0.0); continue
            g = GP.gfill((torch.cat(Q0), torch.cat(Q1)), x0, y0, r, W, H, dev).float()
            cnt += g; area.append(float(g.sum()) * r * r)
            if 1.8 <= z <= 30.0: cw += g; nw += 1
        area = np.array(area); win = (zs >= 1.8) & (zs <= 30.0)
        full = float(area[win].mean()) if win.any() else float(area.mean())
        i0 = int(np.searchsorted(zs, 1.8)); a2 = area[i0:]; nwin = 56                       # worst 28 mm stretch anywhere above the raft (same length as the 1.8-30 mm window the line was fitted on)
        if len(a2) >= nwin:
            cs_ = np.concatenate([[0.0], np.cumsum(a2)]); mv = (cs_[nwin:] - cs_[:-nwin]) / nwin; wi = int(mv.argmax()); worst = float(mv[wi]); wz = float(zs[i0 + wi])
        else: worst = full; wz = 1.8
        cnt = torch.where(cnt >= 3, cnt, torch.zeros_like(cnt))                             # picture only: drop stray 1-2 slice scanline specks from open support meshes
        lay = cnt * (DZ / LH)                                           # layers cured at each spot
        k = int(round(10.0 / r)) | 1
        loc = torch.nn.functional.avg_pool2d(cw[None, None], k, 1, k // 2)[0, 0] / max(nw, 1)   # share of a 10 mm square that cures per layer (1.8-30 mm)
        hy, hx = divmod(int(loc.argmax()), W); hot = float(loc.max())
        bands = []; edges = np.arange(0, PY / 2 + 5, 5.0)
        for a, b in zip(edges[:-1], edges[1:]):
            m = (depth >= a) & (depth < b); bands.append(float(cw[m].sum()) * r * r / max(nw, 1))
        # ---------- picture
        sc = 4; lay_np = lay.cpu().numpy(); vmax = max(float(np.percentile(lay_np[lay_np > 0], 99.5)) if (lay_np > 0).any() else 1.0, 1.0)
        small = cv2.resize(lay_np, (W // sc * 4, H // sc * 4), interpolation=cv2.INTER_AREA) if sc != 4 else lay_np
        t_ = np.clip(small / vmax, 0, 1)[..., None]; g_, y_, r_ = np.array([0, 175, 70.]), np.array([255, 225, 0.]), np.array([225, 25, 25.])      # Mike 11/20 05:16: green to red
        col = np.where(t_ < 0.5, g_ + (y_ - g_) * (t_ * 2), y_ + (r_ - y_) * (t_ * 2 - 1)).astype(np.uint8); col[small <= 0] = (18, 18, 22)
        col = col[::-1].copy()                                           # bed y up
        dnp = depth.cpu().numpy()[::-1]
        for lv in range(10, int(PY / 2) + 1, 10):
            cs, _ = cv2.findContours((dnp >= lv).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE); cv2.drawContours(col, cs, -1, (90, 200, 255), 1)
        BW, BH = col.shape[1], col.shape[0]; topm = 96; RW = 560
        img = Image.new("RGB", (BW + RW, topm + BH + 70), (20, 22, 26)); img.paste(Image.fromarray(col), (0, topm)); dr = ImageDraw.Draw(img)
        dr.rectangle([0, topm, BW - 1, topm + BH - 1], outline=(255, 255, 255))
        for lv in range(10, int(PY / 2) + 1, 10): dr.text((BW / 2 + 4, topm + BH - lv / r - 16), "%d mm" % lv, fill=(90, 200, 255), font=font(13))
        px, py = hx, topm + (H - 1 - hy); dr.ellipse([px - 20, py - 20, px + 20, py + 20], outline=(255, 255, 255), width=3)
        dr.text((px + 24, py - 10), "hottest 10 mm spot: %.0f%% cures per layer" % (100 * hot), fill=(255, 255, 255), font=font(15, True), stroke_width=2, stroke_fill=(0, 0, 0))
        for i, (cx, cy) in enumerate(cen):
            tx = (cx - x0) / r; ty = topm + BH - (cy - y0) / r; t = str(i + 1); dr.text((tx - 6, ty - 9), t, fill=(255, 255, 255), font=font(17, True), stroke_width=2, stroke_fill=(0, 0, 0))
        CLEANMAX = 4832.0                                                                    # fullest 28 mm stretch on any plate that printed clean (FE2_02), same measure, from the sliced files
        side = ("UNDER the line; fullest stretch %.0f at %.0f-%.0f mm (clean prints have reached %.0f)" % (worst, wz, wz + 28, CLEANMAX) if worst <= CLEANMAX else "under at 1.8-30 mm, but %.0f at %.0f-%.0f mm is fuller than any clean print (%.0f)" % (worst, wz, wz + 28, CLEANMAX)) if full <= LINE else "OVER the line"; sc_col = (90, 230, 120) if (full <= LINE and worst <= CLEANMAX) else ((255, 200, 60) if full <= LINE else (255, 90, 80))
        dr.text((10, 6), "%s   %d pieces   %.0f mm tall" % (pid, len(SL), top), fill=(255, 255, 255), font=font(28, True))
        dr.text((10, 42), "fullness %.0f mm2 cured per layer (1.8-30 mm)  vs line %.0f  ->  %s" % (full, LINE, side), fill=sc_col, font=font(21, True))
        dr.text((10, 70), "colour = layers cured at that spot: green = few, yellow = middle, red = most (red tops at %.0f layers) | blue = peel ring, mm in from bed edge | numbers = piece no. in the plate list" % vmax, fill=(190, 190, 190), font=font(13))
        if not inside: dr.text((BW - 420, 6), "WARNING: a piece reaches past the bed edge", fill=(255, 90, 80), font=font(18, True))
        gx = BW + 50; gw = RW - 80; gy = topm + 20; gh = 230; bm = max(max(bands), 1.0)
        dr.text((gx, topm - 8), "Peel ring: cured area it meets, per 5 mm of travel (edge -> centre)", fill=(230, 230, 230), font=font(14, True))
        bw_ = gw / len(bands)
        for i, v in enumerate(bands):
            hh = gh * v / bm; cbar = (255, 150, 60) if v == max(bands) else (110, 160, 230)
            dr.rectangle([gx + i * bw_ + 2, gy + gh - hh, gx + (i + 1) * bw_ - 2, gy + gh], fill=cbar)
            if i % 2 == 0: dr.text((gx + i * bw_, gy + gh + 4), "%d" % edges[i], fill=(180, 180, 180), font=font(12))
        dr.text((gx, gy + gh + 20), "mm in from edge      peak band %.0f mm2 per layer at %d-%d mm" % (bm, edges[int(np.argmax(bands))], edges[int(np.argmax(bands))] + 5), fill=(180, 180, 180), font=font(12))
        g2 = gy + gh + 70; h2 = 230; am = max(float(area.max()), LINE * 1.15)
        dr.text((gx, g2 - 22), "Cured area per layer up the print (mm2)", fill=(230, 230, 230), font=font(14, True))
        dr.rectangle([gx, g2, gx + gw, g2 + h2], outline=(90, 90, 90))
        xa = gx + gw * 1.8 / max(top, 30.0); xb = gx + gw * 30.0 / max(top, 30.0); dr.rectangle([xa, g2 + 1, min(xb, gx + gw), g2 + h2 - 1], fill=(34, 40, 52))
        pts = [(gx + gw * z / max(top, 30.0), g2 + h2 - h2 * a / am) for z, a in zip(zs, area)]
        if len(pts) > 1: dr.line(pts, fill=(255, 190, 70), width=2)
        ly = g2 + h2 - h2 * LINE / am; dr.line([gx, ly, gx + gw, ly], fill=(255, 90, 80), width=1); dr.text((gx + gw - 150, ly - 16), "line %.0f (observed)" % LINE, fill=(255, 90, 80), font=font(12))
        dr.text((gx, g2 + h2 + 4), "0 mm", fill=(180, 180, 180), font=font(12)); dr.text((gx + gw - 50, g2 + h2 + 4), "%.0f mm" % max(top, 30.0), fill=(180, 180, 180), font=font(12))
        dr.text((gx, g2 + h2 + 22), "shaded = 1.8-30 mm, where the fullness number is taken | peak %.0f mm2 at %.1f mm" % (float(area.max()), float(zs[int(area.argmax())])), fill=(180, 180, 180), font=font(12))
        names = [("%d  %s" % (i + 1, os.path.basename(s["path"])))[:62] for i, s in enumerate(rec["sources"])]
        y = topm + BH + 6; dr.text((10, y), "   ".join(names)[:230], fill=(170, 170, 170), font=font(12)); dr.text((10, y + 18), "   ".join(names)[230:460], fill=(170, 170, 170), font=font(12)); dr.text((10, y + 36), "   ".join(names)[460:690], fill=(170, 170, 170), font=font(12))
        img.save(os.path.join(OUT, "HEAT_%s.png" % pid))
        summ[pid] = dict(pieces=len(SL), height_mm=round(top, 1), fullness_mm2=round(full, 0), line=LINE, under=bool(full <= LINE), worst28_mm2=round(worst, 0), worst28_from_mm=round(wz, 1), under_all=bool(full <= LINE and worst <= CLEANMAX), clean_max=CLEANMAX, peak_area_mm2=round(float(area.max()), 0), peak_at_mm=round(float(zs[int(area.argmax())]), 1),
                         hot_share=round(hot, 3), hot_xy_mm=[round(x0 + hx * r, 1), round(y0 + hy * r, 1)], band_peak_mm2=round(bm, 0), band_peak_at_mm=float(edges[int(np.argmax(bands))]), bands=[round(v, 0) for v in bands],
                         inside_bed=inside, models=sorted(set(p.get("set", "") for p in man.get("pieces", []))), files=[os.path.basename(s["path"]) for s in rec["sources"]], seconds=round(time.time() - t0, 1), png=os.path.join(OUT, "HEAT_%s.png" % pid))
        json.dump(summ, open(SUMF + ".tmp", "w"), indent=1); os.replace(SUMF + ".tmp", SUMF)
        log("%s worst 28 mm stretch %.0f from %.0f mm" % (pid, worst, wz))
        log("%s | %d pcs | %.0f mm | fullness %.0f (%s) | peak %.0f at %.1f mm | hot %.0f%% | ring peak %.0f at %d mm | inside %s | %.0f s" % (pid, len(SL), top, full, "under" if full <= LINE else "OVER", area.max(), zs[int(area.argmax())], 100 * hot, bm, edges[int(np.argmax(bands))], inside, time.time() - t0))
        del SL, cnt, cw, lay, loc; torch.cuda.empty_cache()
    except Exception as ex:
        import traceback; log("%s ERROR %r\n%s" % (pid, ex, traceback.format_exc()[-900:]))
log("HEAT DONE")
