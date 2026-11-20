#!/usr/bin/env python3
"""Stand-in for UVtoolsCmd, used only by the plate_all smoke harness."""
import io
import json
import os
import re
import shutil
import sys
import zipfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
USAGE = "usage: fake_uvtools <convert|copy-parameters|print-properties|print-issues|run> ... (smoke fake)"
W, H = 6480, 3600


def encode_pw0(img):
    """img: 2-D uint8 (0/255) -> Anycubic pw0Img RLE bytes (2-byte tokens only)."""
    flat = (np.asarray(img).ravel() > 127)
    n = flat.size
    chg = np.flatnonzero(flat[1:] != flat[:-1]) + 1
    starts = np.concatenate(([0], chg))
    lens = np.diff(np.concatenate((starts, [n]))).astype(np.int64)
    cols = flat[starts]
    ntok = (lens + 4094) // 4095
    run_idx = np.repeat(np.arange(lens.size), ntok)
    first_tok = np.concatenate(([0], np.cumsum(ntok)[:-1]))
    pos = np.arange(int(ntok.sum())) - first_tok[run_idx]
    chunk = np.minimum(4095, lens[run_idx] - pos * 4095)
    hi = np.where(cols[run_idx], 0xF, 0x0).astype(np.int64)
    out = np.empty(chunk.size * 2, np.uint8)
    out[0::2] = (hi << 4) | ((chunk >> 8) & 0xF)
    out[1::2] = chunk & 0xFF
    return out.tobytes()


def decode_pw0(buf, shape=(H, W)):
    b = np.frombuffer(buf, np.uint8)
    hi = b[0::2] >> 4
    ln = ((b[0::2].astype(np.int64) & 0xF) << 8) | b[1::2]
    col = np.where(hi == 0xF, 255, 0).astype(np.uint8)
    return np.repeat(col, ln).reshape(shape)


def cmd_convert(a):
    from PIL import Image
    src, dst = a[0], a[2]
    with zipfile.ZipFile(src) as z:
        names = sorted(n for n in z.namelist() if re.match(r"job\d+\.png$", os.path.basename(n)))
        if not names:
            print("no job*.png in", src)
            return 1
        layers = []
        for n in names:
            im = Image.open(io.BytesIO(z.read(n))).convert("L")
            arr = np.asarray(im)
            if arr.shape != (H, W):
                print("unexpected layer size", arr.shape)
                return 1
            layers.append(encode_pw0(arr))
    paras = [{"layer_thickness": 0.03, "exposure_time": 50 if i < 6 else 2.45, "zup_height": 8, "zup_speed": 3}
             for i in range(len(layers))]
    with zipfile.ZipFile(dst, "w", zipfile.ZIP_STORED) as z:
        for i, b in enumerate(layers):
            z.writestr(f"layer_images/layer_{i}.pw0Img", b)
        z.writestr("layers_controller.conf", json.dumps({"paras": paras}))
        z.writestr("print_info.json", json.dumps({"print_time": 0, "volume": 0}))
        z.writestr("anycubic_photon_resins.pwsp", json.dumps(
            {"res_x": 6480, "res_y": 3600, "xy_pixel": 46, "print_xsize": 298.08, "print_ysize": 165.6,
             "print_zsize": 200}))
    print("ok")
    return 0


def _layer_names(z):
    ns = [n for n in z.namelist() if n.endswith(".pw0Img")]
    return sorted(ns, key=lambda n: int(re.search(r"layer_(\d+)", n).group(1)))


def cmd_props(a):
    with zipfile.ZipFile(a[0]) as z:
        n = len(_layer_names(z))
    print("ResolutionX: 6480")
    print("ResolutionY: 3600")
    print("LayerHeight: 0.03")
    print(f"LayerCount: {n}")
    return 0


def cmd_issues(a):
    try:
        from scipy import ndimage
    except Exception:
        ndimage = None
    hits = []
    prev = None
    with zipfile.ZipFile(a[0]) as z:
        for L, n in enumerate(_layer_names(z)):
            cur = decode_pw0(z.read(n))[::4, ::4] > 127
            if prev is not None and cur.any():
                if ndimage is not None:
                    lab, nl = ndimage.label(cur)
                    touched = np.unique(lab[cur & prev])
                    touched = touched[touched > 0]
                    cnt = nl - touched.size
                else:
                    cnt = 0 if (cur & prev).any() else 1
                hits += [L] * int(cnt)
            prev = cur
    print(f"Island: {len(hits)}")
    print("ResinTrap: 0")
    print("SuctionCup: 0")
    for L in hits:
        print(f"Layer {L}: Island")
    return 0


def _selftest():
    sys.path.insert(0, os.path.join(HERE, "..", "ref", "gpu_scan"))
    import mkm_slice_reader as R
    rng = np.random.RandomState(1)
    imgs = []
    a = np.zeros((32, 64), np.uint8)
    a[5:20, 10:40] = 255
    a[0, 0] = 255
    a[31, 63] = 255
    imgs.append(a)
    imgs.append((rng.rand(32, 64) > 0.5).astype(np.uint8) * 255)
    b = np.zeros((200, 150), np.uint8)       # long runs > 4095 get split
    b[120:, :] = 255
    imgs.append(b)
    imgs.append(np.zeros((32, 64), np.uint8))
    imgs.append(np.full((32, 64), 255, np.uint8))
    for img in imgs:
        buf = encode_pw0(img)
        col, reps = R.pw0_tokens(buf)
        got = np.repeat(col, reps).reshape(img.shape)
        assert np.array_equal(got, img), "ref reader mismatch"
        assert np.array_equal(decode_pw0(buf, img.shape), img), "own decoder mismatch"
    print("SELFTEST OK")


def main(argv):
    if not argv:
        print(USAGE)
        return 1
    c, a = argv[0], argv[1:]
    if c == "--version":
        print("UVtools 7.1.2 (smoke fake)")
        return 0
    if c == "--selftest":
        _selftest()
        return 0
    if c in ("print-issues", "run") and "--help" in a:
        print(f"{c}: smoke fake, usage: {c} FILE [options]")
        return 0
    if c == "convert" and len(a) >= 3:
        return cmd_convert(a)
    if c == "copy-parameters":
        print("ok")
        return 0
    if c == "print-properties" and a:
        return cmd_props(a)
    if c == "print-issues" and a:
        return cmd_issues(a)
    if c == "run" and len(a) >= 2 and "-o" in a:
        shutil.copyfile(a[0], a[a.index("-o") + 1])
        print("ok")
        return 0
    print(USAGE)
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
