"""MKM plater v3 - run-length slice store (step a, part 1).  Contract: plater/CONTRACT.md "mkm_slice_store.py".

Layout of one store directory:
  meta.json   resolution, pitch, layer height, n_layers, mirror_x, origin_mm, recipe_sha256, "complete"
  layers.bin  append-only little-endian uint32 run lengths, layers back to back
  index.npy   int64 (n_layers_written, 2): [offset in uint32 units into layers.bin, n_runs] per layer
A layer is the row-major flattening of the (H, W) mask (row = y, column = x). Run 0 has value 0 (it may have length 0),
runs then alternate 0/1, and the run lengths of a layer sum to H*W.

Encoding a batch happens where the masks live: torch masks (GPU) are encoded with diff/nonzero over the flattened batch and
leave the card in ONE transfer per batch; numpy masks use an equivalent CPU encoder (so the store works without torch).
torch is imported lazily so worker processes that only read layers never pay for it.

ASSUMPTIONS - verify on MKM
  - index offsets are counted in uint32 units (not bytes).
  - meta.json carries one key beyond the contract, "complete" (true once close() found every layer written); the slice-once
    skip in mkm_plate_slice only trusts a store that is complete.
  - uint32 transfers go through int32 on the GPU (run lengths are < 2**31) because torch uint32 support is partial.
"""
import json
import os

import numpy as np

import mkm_v3_config as C

_ENC_CHUNK = 8     # layers encoded per GPU step (bounds temporary VRAM); results are joined on the GPU, one transfer per batch


def _torch():
    import torch
    return torch


def _is_torch(x):
    return type(x).__module__.split(".")[0] == "torch"


def _encode_torch(masks):
    """(B,H,W) bool torch -> (lengths uint32 ndarray, n_runs int64 ndarray). One device->host transfer."""
    torch = _torch()
    B = masks.shape[0]
    flat = masks.reshape(B, -1)
    N = flat.shape[1]
    dev = flat.device
    lens, runs = [], []
    for a in range(0, B, _ENC_CHUNK):
        f = flat[a:a + _ENC_CHUNK]
        b = f.shape[0]
        ch = torch.empty((b, N), dtype=torch.bool, device=dev)          # True where the value differs from the previous pixel
        ch[:, 0] = f[:, 0]                                              # run 0 has value 0, so a lit first pixel is a change
        ch[:, 1:] = f[:, 1:] != f[:, :-1]
        nz = torch.nonzero(ch)                                          # (k,2) sorted by layer, then position
        del ch
        lay, pos = nz[:, 0], nz[:, 1]
        cnt = torch.bincount(lay, minlength=b)
        csum = torch.cumsum(cnt, 0)
        ends = torch.empty(nz.shape[0] + b, dtype=torch.int64, device=dev)     # run end positions, one sentinel (=N) per layer
        ends[torch.arange(nz.shape[0], device=dev) + lay] = pos
        sent = csum + torch.arange(b, device=dev)
        ends[sent] = N
        starts = torch.cat([torch.zeros(1, dtype=torch.int64, device=dev), ends[:-1]])
        starts[sent - cnt] = 0                                          # first run of every layer starts at 0
        lens.append((ends - starts).to(torch.int32))
        runs.append((cnt + 1).to(torch.int32))
    packed = torch.cat(lens + runs).cpu().numpy()                        # the one transfer for the whole batch
    nl = sum(int(r.shape[0]) for r in runs)
    return packed[:-nl].view(np.uint32), packed[-nl:].astype(np.int64)


def _encode_numpy(masks):
    lens, runs = [], []
    for m in masks:
        f = np.ascontiguousarray(m, dtype=bool).reshape(-1)
        pos = np.flatnonzero(np.diff(f.view(np.int8), prepend=np.int8(0)))
        ends = np.concatenate([pos, [f.size]])
        starts = np.concatenate([[0], pos])
        lens.append((ends - starts).astype(np.uint32))
        runs.append(len(ends))
    return np.concatenate(lens), np.array(runs, np.int64)


class SliceStore:
    def __init__(self, dir, mode="r"):
        if mode not in ("r", "w"):
            raise ValueError("mode must be 'r' or 'w'")
        self.dir, self.mode = dir, mode
        self._bin = os.path.join(dir, "layers.bin")
        self._idx = os.path.join(dir, "index.npy")
        self._meta = os.path.join(dir, "meta.json")
        self.meta = {}
        if mode == "w":
            os.makedirs(dir, exist_ok=True)
            open(self._bin, "wb").close()                                # new store: start empty
            for p in (self._idx, self._meta):
                if os.path.exists(p):
                    os.remove(p)
            self._index = np.zeros((0, 2), np.int64)
            self._fh = open(self._bin, "ab")
        else:
            self.meta = json.load(open(self._meta, encoding="utf-8"))
            self._index = np.load(self._idx) if os.path.exists(self._idx) else np.zeros((0, 2), np.int64)
            self._fh = None

    # ---- write side ----
    def write_meta(self, *, n_layers, origin_mm, recipe_sha256, res=C.RES, pixel_mm=C.PIXEL_MM, layer_mm=C.LAYER_MM,
                   mirror_x=C.MIRROR_X):
        self.meta = dict(res=[int(res[0]), int(res[1])], pixel_mm=[float(pixel_mm[0]), float(pixel_mm[1])],
                         layer_mm=float(layer_mm), n_layers=int(n_layers), mirror_x=bool(mirror_x),
                         origin_mm=[float(origin_mm[0]), float(origin_mm[1])], recipe_sha256=str(recipe_sha256),
                         encoding="rle-uint32-rowmajor-first-run-is-0", complete=False)
        self._save_meta()

    def _save_meta(self):
        tmp = self._meta + ".tmp"
        json.dump(self.meta, open(tmp, "w", encoding="utf-8"), indent=1)
        os.replace(tmp, self._meta)

    def append_batch(self, masks):
        """masks: torch bool or numpy bool (B,H,W), already mirrored. Appends B layers."""
        if self.mode != "w":
            raise RuntimeError("store opened read-only")
        if not self.meta:
            raise RuntimeError("write_meta() first")
        W, H = self.meta["res"]
        if tuple(masks.shape[1:]) != (H, W):
            raise ValueError("mask shape %r != (B, %d, %d)" % (tuple(masks.shape), H, W))
        lens, nruns = _encode_torch(masks) if _is_torch(masks) else _encode_numpy(np.asarray(masks, dtype=bool))
        base = self._index[-1, 0] + self._index[-1, 1] if len(self._index) else 0
        offs = base + np.cumsum(nruns) - nruns
        self._fh.write(lens.astype("<u4", copy=False).tobytes())
        self._fh.flush()
        self._index = np.concatenate([self._index, np.stack([offs, nruns], 1)])
        with open(self._idx + ".tmp", "wb") as f:
            np.save(f, self._index)
        os.replace(self._idx + ".tmp", self._idx)

    def close(self):
        if self._fh is not None:
            self._fh.close()
            self._fh = None
            self.meta["complete"] = bool(len(self._index) == self.meta.get("n_layers"))
            self._save_meta()

    # ---- read side ----
    @property
    def n_layers(self):
        return int(len(self._index)) if len(self._index) else int(self.meta.get("n_layers", 0))

    def _runs(self, i0, i1):
        o0 = int(self._index[i0, 0])
        o1 = int(self._index[i1 - 1, 0] + self._index[i1 - 1, 1])
        return np.fromfile(self._bin, dtype="<u4", count=o1 - o0, offset=o0 * 4)

    def read_layer(self, i):
        W, H = self.meta["res"]
        r = self._runs(i, i + 1)
        val = (np.arange(len(r)) & 1).astype(bool)
        return np.repeat(val, r).reshape(H, W)

    def read_batch(self, i0, i1, device=None):
        """layers i0..i1-1 decoded on the GPU (repeat_interleave) -> torch bool (B,H,W)."""
        torch = _torch()
        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        W, H = self.meta["res"]
        B = i1 - i0
        r = torch.from_numpy(self._runs(i0, i1).astype(np.int64)).to(dev)
        nr = torch.from_numpy(self._index[i0:i1, 1].copy()).to(dev)
        lay = torch.repeat_interleave(torch.arange(B, device=dev), nr)
        k = torch.arange(r.shape[0], device=dev) - (torch.cumsum(nr, 0) - nr)[lay]
        return torch.repeat_interleave((k & 1).bool(), r).reshape(B, H, W)

    def cured_px(self, i):
        """lit pixels of layer i, from the runs alone (odd runs are value 1)."""
        return int(self._runs(i, i + 1)[1::2].astype(np.int64).sum())

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()
