"""MKM plater v3 - slice one plate ONCE on the GPU at printer pixels into the run-length store (step a, part 1).
Contract: plater/CONTRACT.md "mkm_plate_slice.py".   CLI: python mkm_plate_slice.py <recipe.json> <store_dir> [--batch N]

Flow: recipe.json sources[] -> load STL (binary or ascii, numpy) -> verify sha256 -> apply the 4x4 matrix once (float64, back to
float32) -> GSlicer per source (reused from mkm_gpuport, imported not copied) -> per batch of B layer midpoints z=(i+0.5)*0.03:
segments from every source, one gfill (non-zero winding = union of all sources) per layer into a (B,H,W) bool on the GPU,
flip columns once (mirror x), SliceStore.append_batch (the only device->host transfer, once per batch).

Rules kept: no printer path, mirror once (here; the store is already mirrored and the print builder does not mirror again),
every layer 0..top, no per-layer CPU work (the layer loop only launches GPU work; there is no host copy of any mask in this file).

ASSUMPTIONS - verify on MKM
  - gfill(P, x0, y0, r, W, H, d): (x0, y0) is the lower-left EDGE of pixel (0,0) in mm, r the square pixel pitch, pixel centre
    at x0+(i+0.5)*r (read from mkm_gpuport.gfill, ~l.225). So pixel-centre sampling is exactly what the contract asks for.
  - origin: recipe geometry.origin_mm when present (the planner writes -bed_px*pitch/2 = (-149.04, -82.8)); otherwise
    -SCREEN_MM/2. The origin used is stored in meta.json origin_mm. The pitch used is always mkm_v3_config.PIXEL_MM
    (0.046); a recipe whose geometry.bed_px is not (6480, 3600) is refused.
  - "mirror x" = flip of the column axis of the whole screen (column i -> 6479-i), done once here on the GPU.
  - n_layers = ceil(round(top,4)/0.03 - 1e-6) (float32 vertex noise, e.g. 0.30000001, must not add an empty extra layer).
  - Layers are numbered from the bed z=0 (not from the lowest triangle): a plate whose lowest point is above 0 starts with
    empty layers. The planner's recipe puts the lowest point at about 0 (matrix z drop).
  - Multiple sources are unioned by concatenating their segments before one gfill (as plate_heat.py does); overlapping
    sources therefore merge instead of cancelling.
  - mkm_gpuport.temp_gate() needs mkm_safety on the laptop path (gpuport imports it); if that is missing the slice fails loudly.
  - Slice-once skip: store meta recipe_sha256 equals the sha256 of the recipe.json bytes AND meta "complete" is true.
"""
import hashlib
import json
import math
import os
import re
import sys
import time

import numpy as np

import mkm_v3_config as C


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


_VERTEX = re.compile(rb"vertex\s+(\S+)\s+(\S+)\s+(\S+)", re.I)


def load_stl(path):
    """binary or ascii STL -> float32 (n,3,3); triangles with a non-finite coordinate are dropped (as mkm_gpuport.load_T32)."""
    size = os.path.getsize(path)
    with open(path, "rb") as fh:
        head = fh.read(84)
    if len(head) == 84 and size == 84 + 50 * int(np.frombuffer(head[80:84], "<u4")[0]):
        dt = np.dtype([("n", "<f4", (3,)), ("v", "<f4", (3, 3)), ("a", "<u2")])
        v = np.fromfile(path, dtype=dt, offset=84)["v"]
    else:
        with open(path, "rb") as fh:
            vals = _VERTEX.findall(fh.read())
        if not vals or len(vals) % 3:
            raise ValueError("not a readable STL (binary size mismatch, ascii vertices %d): %s" % (len(vals), path))
        v = np.array(vals, dtype=np.float64).astype(np.float32).reshape(-1, 3, 3)
    return np.ascontiguousarray(v[np.isfinite(v.reshape(-1, 9)).all(1)])


def load_plate(recipe):
    """-> (list of float32 (n,3,3) triangle arrays with the matrix applied once, top_mm). sha256 mismatch raises."""
    tris, top = [], 0.0
    for s in recipe["sources"]:
        want = s.get("sha256")
        if not want:
            raise ValueError("source without sha256: %s" % s.get("path"))
        got = _sha256(s["path"])
        if got != want:
            raise ValueError("sha256 mismatch for %s: recipe %s file %s" % (s["path"], want, got))
        T = load_stl(s["path"])
        M = np.array(s["matrix"], np.float64)
        T = (T.reshape(-1, 3).astype(np.float64) @ M[:3, :3].T + M[:3, 3]).astype(np.float32).reshape(-1, 3, 3)
        top = max(top, float(T[:, :, 2].max()))
        tris.append(T)
    return tris, top


def slice_plate(recipe_path, store_dir, batch=64, device="cuda"):
    from mkm_slice_store import SliceStore
    t0 = time.time()
    raw = open(recipe_path, "rb").read()
    rsha = hashlib.sha256(raw).hexdigest()
    recipe = json.loads(raw.decode("utf-8-sig"))
    geo = recipe.get("geometry") or {}
    if geo.get("bed_px") and tuple(int(x) for x in geo["bed_px"]) != tuple(C.RES):
        raise ValueError("recipe bed_px %r is not the printer grid %r" % (geo["bed_px"], C.RES))
    origin = tuple(float(x) for x in geo["origin_mm"]) if geo.get("origin_mm") else (-C.SCREEN_MM[0] / 2, -C.SCREEN_MM[1] / 2)
    W, H = C.RES
    r = C.PIXEL_MM[0]
    if abs(C.PIXEL_MM[0] - C.PIXEL_MM[1]) > 1e-9:
        raise ValueError("gfill needs square pixels")

    # slice-once: a finished store of this exact recipe is kept as is
    mp = os.path.join(store_dir, "meta.json")
    if os.path.exists(mp):
        try:
            m = json.load(open(mp, encoding="utf-8"))
            if m.get("recipe_sha256") == rsha and m.get("complete"):
                return dict(store_dir=store_dir, n_layers=m["n_layers"], skipped=True, recipe_sha256=rsha, origin_mm=m["origin_mm"],
                            seconds=round(time.time() - t0, 1))
        except (ValueError, OSError):
            pass

    import torch
    import mkm_gpuport as GP
    if device == "cuda":
        d = GP.configure()                                   # VRAM fit check / fraction cap lives in gpuport
        if d is None:
            raise RuntimeError("no usable GPU: %s" % GP._STATE.get("note"))
    else:
        d = torch.device(device)

    tris, top = load_plate(recipe)
    n_layers = int(math.ceil(round(top, 4) / C.LAYER_MM - 1e-6))   # top rounded to 0.1 um: float32 noise must not add a layer
    if n_layers < 1:
        raise ValueError("plate has no height (top %.4f mm)" % top)
    slicers = [GP.GSlicer(T, None, d) for T in tris]         # sorts by zmin on the GPU, as GSlicer expects
    del tris

    st = SliceStore(store_dir, "w")
    st.write_meta(res=C.RES, pixel_mm=C.PIXEL_MM, layer_mm=C.LAYER_MM, n_layers=n_layers, mirror_x=C.MIRROR_X, origin_mm=origin,
                  recipe_sha256=rsha)
    for i0 in range(0, n_layers, batch):
        i1 = min(n_layers, i0 + batch)
        if hasattr(GP, "temp_gate"):
            GP.temp_gate()
        masks = torch.zeros((i1 - i0, H, W), dtype=torch.bool, device=d)
        for k, i in enumerate(range(i0, i1)):
            z = (i + 0.5) * C.LAYER_MM
            sg = [p for p in (S.segs(z) for S in slicers) if p is not None and p[0].shape[0]]
            if sg:
                P = (torch.cat([p[0] for p in sg]), torch.cat([p[1] for p in sg]))
                masks[k] = GP.gfill(P, origin[0], origin[1], r, W, H, d)
        if C.MIRROR_X:
            masks = torch.flip(masks, dims=[2])              # mirror x, once, here
        st.append_batch(masks)                               # encode on the GPU, one transfer
        del masks
        if d.type == "cuda":
            torch.cuda.empty_cache()
        print("[slice] %s layers %d-%d / %d  %.0fs" % (os.path.basename(os.path.dirname(os.path.abspath(recipe_path))), i0, i1 - 1, n_layers,
                                                        time.time() - t0), flush=True)
    st.close()
    return dict(store_dir=store_dir, n_layers=n_layers, top_mm=top, skipped=False, recipe_sha256=rsha, origin_mm=list(origin),
                seconds=round(time.time() - t0, 1))


if __name__ == "__main__":
    a = [x for x in sys.argv[1:]]
    b = 64
    if "--batch" in a:
        j = a.index("--batch")
        b = int(a[j + 1])
        del a[j:j + 2]
    if len(a) != 2:
        sys.exit("usage: mkm_plate_slice.py <recipe.json> <store_dir> [--batch N]")
    print(json.dumps(slice_plate(a[0], a[1], batch=b), indent=1))
