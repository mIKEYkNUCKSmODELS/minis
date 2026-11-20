# MKM slice reader - reads a sliced plate directly, no UVtools unpack. Read-only: never writes to a slice file.
# Format logic re-implemented from the UVtools project (github.com/sn4k3/UVtools) file-format code; this is a port, not a copy.
# 2026-11-20: Anycubic M7 Max (.pm7m = zip of layer_images/layer_N.pw0Img + layers_controller.conf + anycubic_photon_resins.pwsp).
# Verified pixel-exact against UVtoolsCmd on DRL2_PLATE_02 layers 0, 6, 99, 420, 2396 (0 mismatches). That file has no grey (AA) pixels.
import zipfile, json, re, os
import numpy as np

def pw0_tokens(buf):
    """Anycubic pw0Img run-length layer -> (grey values uint8, run lengths int64), in pixel order. Fully vectorised."""
    b = np.frombuffer(buf, np.uint8)
    n = b.size
    if n == 0:
        return np.zeros(0, np.uint8), np.zeros(0, np.int64)
    hi = b >> 4
    L = (hi == 0) | (hi == 15)                       # bytes that look like the start of a 2-byte (12-bit run) token
    idx = np.arange(n)
    first = L & ~np.concatenate(([False], L[:-1]))   # first byte of each run of such bytes is always a token start
    rs = np.maximum.accumulate(np.where(first, idx, 0))
    Lstart = L & (((idx - rs) & 1) == 0)             # 2-byte token starts
    cont = np.concatenate(([False], Lstart[:-1]))    # the byte after a 2-byte token start is its low byte
    start = ~cont
    isl = Lstart[start]
    si = idx[start]
    nxt = b[np.minimum(si + 1, n - 1)].astype(np.int64)
    low = b[si].astype(np.int64) & 15
    reps = np.where(isl, (low << 8) | nxt, low)
    h = hi[si]
    col = np.where(isl, np.where(h == 15, 255, 0), (h << 4) | h).astype(np.uint8)
    return col, reps

def _fit(col, reps, npix):
    tot = int(reps.sum())
    if tot > npix:
        c = np.cumsum(reps); k = int(np.searchsorted(c, npix)); reps = reps[:k + 1].copy(); col = col[:k + 1]
        reps[k] -= int(c[k] - npix)
    elif tot < npix:
        col = np.concatenate((col, np.zeros(1, np.uint8))); reps = np.concatenate((reps, np.array([npix - tot], np.int64)))
    return col, reps

class Pm7m:
    machine = "Anycubic M7 Max"
    def __init__(self, path):
        self.path = path
        self.z = zipfile.ZipFile(path)
        names = self.z.namelist()
        self.layer_names = sorted((n for n in names if n.lower().endswith(".pw0img")), key=lambda s: int(re.findall(r"(\d+)\.pw0img$", s.lower())[0]))
        self.n_layers = len(self.layer_names)
        self.ctrl = json.loads(self.z.read("layers_controller.conf")) if "layers_controller.conf" in names else {}
        self.info = json.loads(self.z.read("print_info.json")) if "print_info.json" in names else {}
        self.res = None; self.pixel_um = None; self.bed_mm = None
        if "anycubic_photon_resins.pwsp" in names:
            t = self.z.read("anycubic_photon_resins.pwsp").decode("utf-8", "replace")
            def g(k):
                m = re.search(r'"%s"\s*:\s*([-0-9.]+)' % k, t); return float(m.group(1)) if m else None
            if g("res_x") and g("res_y"): self.res = (int(g("res_x")), int(g("res_y")))
            self.pixel_um = g("xy_pixel"); self.bed_mm = (g("print_xsize"), g("print_ysize"), g("print_zsize"))
    def properties(self):
        p = self.ctrl.get("paras", [])
        return dict(machine=self.machine, layers=self.n_layers, resolution=self.res, pixel_um=self.pixel_um, bed_mm=self.bed_mm,
                    layer_thickness=p[0]["layer_thickness"] if p else None,
                    bottom_exposure=p[0]["exposure_time"] if p else None, normal_exposure=p[-1]["exposure_time"] if p else None,
                    lift_height=p[-1].get("zup_height") if p else None, lift_speed=p[-1].get("zup_speed") if p else None,
                    print_time_s=self.info.get("print_time"), volume_ml=self.info.get("volume"))
    def layer_params(self, i):
        return self.ctrl["paras"][i]
    def tokens(self, i):
        w, h = self.res
        return _fit(*pw0_tokens(self.z.read(self.layer_names[i])), w * h)
    def cured_pixels(self, i, thresh=127):
        col, reps = self.tokens(i)                    # area straight from the runs, no image built
        return int(reps[col > thresh].sum())
    def layer(self, i):
        """layer i (0-based) as uint8 numpy array H x W (CPU)"""
        w, h = self.res; col, reps = self.tokens(i)
        return np.repeat(col, reps).reshape(h, w)
    def layer_gpu(self, i, device="cuda", binary=True):
        """layer i expanded on the GPU: bool (or uint8) tensor H x W"""
        import torch
        w, h = self.res; col, reps = self.tokens(i)
        c = torch.from_numpy(col).to(device); r = torch.from_numpy(reps).to(device)
        if binary: c = c > 127
        return torch.repeat_interleave(c, r).reshape(h, w)
    def layers_gpu(self, idxs, device="cuda", binary=True):
        import torch
        return torch.stack([self.layer_gpu(i, device, binary) for i in idxs])

def open_slice(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pm7m":
        return Pm7m(path)
    raise ValueError("no reader yet for " + ext)
