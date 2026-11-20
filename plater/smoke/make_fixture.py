#!/usr/bin/env python3
"""Write a planner-style output folder (OUT/P1..P3 + OUT/_stl) for the plate_all smoke run."""
import hashlib
import json
import os
import sys

import numpy as np

# (x0, y0, z0) -> (x1, y1, z1) corner faces, CCW seen from outside
def box_tris(w, d, z0, z1, reverse=False, ox=0.0, oy=0.0):
    x0, x1, y0, y1 = ox, ox + w, oy, oy + d
    v = np.array([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
                  [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]], np.float64)
    quads = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    tris = []
    for a, b, c, e in quads:
        tris.append((v[a], v[b], v[c]))
        tris.append((v[a], v[c], v[e]))
    if reverse:
        tris = [(t[0], t[2], t[1]) for t in tris]
    return tris


def write_stl(path, tris):
    n = len(tris)
    rec = np.zeros(n, dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
    for i, (a, b, c) in enumerate(tris):
        nrm = np.cross(b - a, c - a)
        ln = np.linalg.norm(nrm)
        rec["n"][i] = nrm / ln if ln > 0 else 0
        rec["v"][i] = np.array([a, b, c])
    with open(path, "wb") as f:
        f.write(b"smoke fixture".ljust(80, b"\0"))
        f.write(np.uint32(n).tobytes())
        f.write(rec.tobytes())


def main(out):
    out = os.path.abspath(out)
    stl_dir = os.path.join(out, "_stl")
    os.makedirs(stl_dir, exist_ok=True)
    rng = np.random.RandomState(7)
    xs = [-130, -75, -20, 35, 90]
    ys = [-60, 0]
    plan = {"P1": 4, "P2": 3, "P3": 5}
    for name, nbox in plan.items():
        models = []  # (basename, tris, dx, dy, height)
        for i in range(nbox):
            w, d = (float(v) for v in rng.uniform(8, 25, 2))
            h = float(rng.uniform(1.5, 3.0))
            if i == 0:
                h = 3.0
            tris = box_tris(w, d, 0.0, h)
            models.append((f"{name}_m{i + 1}.stl", tris, xs[i], ys[i % 2], h))
        if name == "P2":
            models.append((f"{name}_float.stl", box_tris(5, 5, 1.0, 2.0), xs[3], ys[1], 2.0))
        if name == "P3":
            tris = box_tris(20, 20, 0.0, 2.4) + box_tris(12, 12, 0.6, 1.8, reverse=True, ox=4, oy=4)
            models[-1] = (f"{name}_hollow.stl", tris, xs[4], ys[0], 2.4)
        sources, pieces = [], []
        for k, (base, tris, dx, dy, h) in enumerate(models):
            p = os.path.join(stl_dir, base)
            write_stl(p, tris)
            raw = open(p, "rb").read()
            sources.append({"path": p, "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw),
                            "matrix": [[1, 0, 0, dx], [0, 1, 0, dy], [0, 0, 1, 0], [0, 0, 0, 1]]})
            pieces.append({"no": k + 1, "set": "SMOKE", "file": base, "path": p, "x": dx, "y": dy, "rot": 0})
        maxh = max(m[4] for m in models)
        pd = os.path.join(out, name)
        os.makedirs(pd, exist_ok=True)
        recipe = {"schema": 1, "name": name, "sources": sources,
                  "z_convention": "mm; exact affine applied once; layer midpoint sampling",
                  "geometry": {"bed_px": [6480, 3600], "pixel_pitch_mm": [0.046, 0.046], "dx_mm": 0.046,
                               "layer_mm": 0.03, "win_mm": [298.08, 165.6], "origin_mm": [-149.04, -82.8],
                               "layer_planes_mm": None, "layer_tops_mm": None},
                  "placement": "smoke fixture", "verdict": {"verdict": "UNSOLVED"}}
        with open(os.path.join(pd, "recipe.json"), "w") as f:
            json.dump(recipe, f, indent=1)
        manifest = [{"tag": name, "file": "", "height": maxh, "fill": 0.0, "safe_d": 0, "hold": 0,
                     "notes": "smoke", "leftover": [], "pieces": pieces, "peel": {}}]
        with open(os.path.join(pd, "RING_MANIFEST.json"), "w") as f:
            json.dump(manifest, f, indent=1)
    print("fixture written:", out)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: make_fixture.py OUT")
    main(sys.argv[1])
