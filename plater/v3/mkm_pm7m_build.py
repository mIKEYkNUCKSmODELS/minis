"""MKM plater v3 - build the UVtools print file for one plate from its slice store (step a, part 2).
Contract: plater/CONTRACT.md "mkm_pm7m_build.py".   CLI: python mkm_pm7m_build.py <store_dir> <out_dir> <name> [--ref-pm7m X]

Route (REPORT_UVTOOLS_ROUTE.txt, proven on a 60-layer test): PNG layers -> Prusa SL1 zip -> UVtoolsCmd convert -> pm7m.
  1. out_dir/name.sl1  zip: job00000.png.. flat, 8-bit grayscale (mode "L", 0/255), numbered from 0, + config.ini + prusaslicer.ini
  2. UVtoolsCmd convert name.sl1 pm7m name.pm7m
  3. optional UVtoolsCmd copy-parameters <ref_pm7m> name.pm7m   (bottom layer count, lifts; exposure stays 50 s / 2.45 s)
  4. UVtoolsCmd print-properties name.pm7m -> must show 6480 x 3600, 0.03 mm, LayerCount == store.n_layers, else raise.
Only writes inside out_dir. Making the file is the end of this module: it has no path to any printer.

ASSUMPTIONS - verify on MKM
  - The full config.ini / prusaslicer.ini key list of the working 60-layer test is not in the package (the report names it
    only in summary), so this writes the contract's keys plus the exposure lines the report mentions. If UVtools rejects the file,
    diff against the 60-layer test's ini files first.
  - display_mirror_x = 0 is what keeps the SL1 reader from mirroring again; whether UVtools 7.1.2 honours that key is unverified.
    The store is already mirrored once; a flipped pm7m means this key is ignored.
  - print-properties output is parsed as "<Key><space|:|=><number>" per line (ResolutionX, ResolutionY, LayerHeight, LayerCount).
  - PNG compress_level=1: cheap, the layers are long runs so they stay small.
"""
import io
import json
import multiprocessing as mp
import os
import re
import sys
import zipfile

import numpy as np

import mkm_v3_config as C

_ST = None


def _init(store_dir):
    global _ST
    from mkm_slice_store import SliceStore
    _ST = SliceStore(store_dir)


def _png(i):
    from PIL import Image
    arr = _ST.read_layer(i).view(np.uint8) * np.uint8(255)       # 0 / 255
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG", compress_level=1)
    return i, buf.getvalue()


def _config_ini(n_layers):
    # keys and sources:
    #  jobDir=job                   REPORT_UVTOOLS_ROUTE (SL1 reader looks for job%05d.png)
    #  layerHeight                  CONTRACT / mkm_v3_config.LAYER_MM
    #  expTime, expTimeFirst        Mike's constants NORMAL_EXP_S, BOTTOM_EXP_S
    #  numFast=n_layers, numSlow=0  CONTRACT (every layer is a "fast" layer; bottom layer count comes from copy-parameters)
    #  printerModel=SL1             CONTRACT
    return "\n".join(["jobDir = job", "layerHeight = %s" % C.LAYER_MM, "expTime = %s" % C.NORMAL_EXP_S,
                      "expTimeFirst = %s" % C.BOTTOM_EXP_S, "numFast = %d" % n_layers, "numSlow = 0", "printerModel = SL1", ""])


def _prusaslicer_ini(meta):
    # keys and sources:
    #  display_pixels_x/y, display_width/height   CONTRACT; values from the store meta (6480 x 3600, 298.08 x 165.6 mm)
    #  layer_height                               CONTRACT / store meta layer_mm
    #  printer_notes = FILEFORMAT_PM7M            CONTRACT + REPORT ("FILEFORMAT_PM7M in printer_notes")
    #  display_mirror_x = 0                       CONTRACT: the store is ALREADY mirrored (mkm_plate_slice flips columns once);
    #                                             0 tells the SL1 reader not to mirror again
    #  exposure_time / initial_exposure_time      REPORT ("layer_height/exposure lines"); Mike's 2.45 s / 50 s
    W, H = meta["res"]
    return "\n".join(["display_pixels_x = %d" % W, "display_pixels_y = %d" % H,
                      "display_width = %s" % round(W * meta["pixel_mm"][0], 4), "display_height = %s" % round(H * meta["pixel_mm"][1], 4),
                      "layer_height = %s" % meta["layer_mm"], "printer_notes = FILEFORMAT_PM7M", "display_mirror_x = 0",
                      "exposure_time = %s" % C.NORMAL_EXP_S, "initial_exposure_time = %s" % C.BOTTOM_EXP_S, ""])


def build_sl1(store_dir, sl1_path, workers=None):
    from mkm_slice_store import SliceStore
    st = SliceStore(store_dir)
    n = st.n_layers
    if not st.meta.get("complete"):
        raise RuntimeError("store %s is not complete (%d of %s layers)" % (store_dir, n, st.meta.get("n_layers")))
    workers = workers or max(1, min(6, (os.cpu_count() or 2) - 2))
    tmp = sl1_path + ".part"
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_STORED) as z:     # PNG is already compressed
        z.writestr("config.ini", _config_ini(n))
        z.writestr("prusaslicer.ini", _prusaslicer_ini(st.meta))
        with mp.Pool(workers, initializer=_init, initargs=(store_dir,)) as pool:
            for i, b in pool.imap(_png, range(n), chunksize=4):  # imap keeps order
                z.writestr("job%05d.png" % i, b)
    os.replace(tmp, sl1_path)
    return n


_PROPS = ("ResolutionX", "ResolutionY", "LayerHeight", "LayerCount")


def parse_properties(text):
    out = {}
    for k in _PROPS:
        m = re.search(r"\b%s\b[\s:=]+(-?[0-9][0-9.,]*)" % k, text)
        if m:
            out[k] = float(m.group(1).replace(",", ""))
    return out


def build_print_file(store_dir, out_dir, name, ref_pm7m=None, uvtools=C.UVTOOLS):
    from mkm_slice_store import SliceStore
    if os.path.basename(name) != name or not name:
        raise ValueError("name must be a bare file stem: %r" % name)
    os.makedirs(out_dir, exist_ok=True)
    sl1 = os.path.join(out_dir, name + ".sl1")
    pm7m = os.path.join(out_dir, name + ".pm7m")
    n = build_sl1(store_dir, sl1)
    meta = SliceStore(store_dir).meta
    r = C.run_hidden([uvtools, "convert", sl1, "pm7m", pm7m])
    if r.returncode != 0 or not os.path.exists(pm7m):
        raise RuntimeError("UVtools convert failed (%s): %s %s" % (r.returncode, r.stdout[-800:], r.stderr[-800:]))
    if ref_pm7m:
        r = C.run_hidden([uvtools, "copy-parameters", ref_pm7m, pm7m])
        if r.returncode != 0:
            raise RuntimeError("UVtools copy-parameters failed (%s): %s %s" % (r.returncode, r.stdout[-800:], r.stderr[-800:]))
    r = C.run_hidden([uvtools, "print-properties", pm7m])
    text = (r.stdout or "") + (r.stderr or "")
    open(os.path.join(out_dir, name + "_properties.txt"), "w", encoding="utf-8").write(text)
    got = parse_properties(text)
    want = dict(ResolutionX=meta["res"][0], ResolutionY=meta["res"][1], LayerHeight=meta["layer_mm"], LayerCount=n)
    bad = [k for k in _PROPS if k not in got or abs(got[k] - want[k]) > 1e-6]
    if bad:
        raise RuntimeError("print-properties mismatch on %s: wanted %s got %s\n%s" % (bad, want, got, text[-1500:]))
    return pm7m


if __name__ == "__main__":
    a = sys.argv[1:]
    ref = None
    if "--ref-pm7m" in a:
        j = a.index("--ref-pm7m")
        ref = a[j + 1]
        del a[j:j + 2]
    if len(a) != 3:
        sys.exit("usage: mkm_pm7m_build.py <store_dir> <out_dir> <name> [--ref-pm7m X]")
    print(build_print_file(a[0], a[1], a[2], ref_pm7m=ref))
