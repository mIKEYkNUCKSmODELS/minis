# runs the auto plater, and if the fine-slice check finds a plate over the clean-print limit, reruns once with the caps lowered by exactly that overshoot. Then builds the plate sheets.
import subprocess, json, os, sys, shutil, glob, re, textwrap
PY = r"C:\Users\confi\AI\ComfyUI_windows_portable\python_embeded\python.exe"; D = r"C:\MKM_TMP\plate_heat_20261120\auto"; NW = 0x08000000
sc = 1.04
for it in range(3):
    for t in glob.glob(D + r"\try_*"): shutil.rmtree(t, ignore_errors=True)
    subprocess.run([PY, "-I", "-u", D + r"\auto_plater.py", r"C:\MKM_TMP\realbatch_20261117\pieces.json", "%.4f" % sc], creationflags=NW)
    c = json.load(open(D + r"\CHECK.json"))
    if c["worst_ratio"] <= 1.0: break
    sc *= c["worst_ratio"] * 1.002
from PIL import Image, ImageDraw, ImageFont
def font(sz, b=False):
    try: return ImageFont.truetype(r"C:\Windows\Fonts\arialbd.ttf" if b else r"C:\Windows\Fonts\arial.ttf", sz)
    except Exception: return ImageFont.load_default()
R = json.load(open(D + r"\AUTO_RESULT.json")); H = json.load(open(r"C:\MKM_TMP\plate_heat_20261120\heat_auto\heat_summary.json")); ks = sorted(H, key=lambda k: int(k[1:])); tiles = []
for k in ks:
    pd = os.path.join(R["out"], k); man = json.load(open(os.path.join(pd, "RING_MANIFEST.json"), encoding="utf-8"))[0]; png = glob.glob(os.path.join(pd, "AUTO_P*.png"))[0]
    im = Image.open(png).convert("RGB"); w, h = im.size; im = im.crop((0, 20, w, h)); im = im.resize((880, int(880 * (h - 20) / w)), Image.LANCZOS)
    sets = {}
    for p in man["pieces"]: sets.setdefault(p["set"], []).append(str(p["no"]))
    lines = []
    for s_, nos in sets.items(): lines += textwrap.wrap("%s: %s" % (", ".join(nos), s_), 118)
    v = H[k]; ok = v["fullness_mm2"] <= 3364 and v["worst28_mm2"] <= 4832
    t = Image.new("RGB", (900, 64 + im.size[1] + 16 * len(lines) + 16), (245, 245, 245)); dr = ImageDraw.Draw(t)
    dr.text((10, 6), "%s   %d pieces   %.0f mm tall   %s" % (k, v["pieces"], v["height_mm"], "INSIDE what has printed clean" if ok else "OVER what has printed clean"), fill=(0, 110, 40) if ok else (190, 30, 20), font=font(24, True))
    dr.text((10, 38), "cured area per layer: %.0f in the first 30 mm (clean prints up to 3364), fullest stretch higher up %.0f (clean up to 4832)   gap and edge %s mm" % (v["fullness_mm2"], v["worst28_mm2"], R["gap_mm"]), fill=(40, 40, 40), font=font(14))
    t.paste(im, (10, 62)); y = 66 + im.size[1]
    for l in lines: dr.text((10, y), l, fill=(20, 20, 20), font=font(13)); y += 16
    tiles.append(t)
for old in glob.glob(D + r"\PLATES_*.png"): os.remove(old)
for n in range(0, len(tiles), 6):
    g = tiles[n:n + 6]; hh = [max(a.size[1] for a in g[i:i + 2]) for i in range(0, len(g), 2)]; sh = Image.new("RGB", (1800, sum(hh)), (245, 245, 245)); y = 0
    for i, a in enumerate(g):
        sh.paste(a, ((i % 2) * 900, y))
        if i % 2 == 1: y += hh[i // 2]
    sh.save(D + r"\PLATES_%d.png" % (n // 6 + 1))
open(D + r"\ALL_DONE.txt", "w").write("ok")
