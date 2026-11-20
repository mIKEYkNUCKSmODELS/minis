"""mkm_suction.py - the plate PREDICTOR entry point.  20261116 round 2: ONE predictor.

Mike (11/12): "never build a second predictor - the existing plate predictor is to be adjusted so it does what it was designed to do"; (11/12) "I want the math to determine whether it will print".
This module is the existing predictor's command line and Python entry.  Its legacy engine (calibrated suction/hold ratio Q_LIMIT, three-limit-state thresholds fitted to past plates, k_cup, sigma_green prior,
KNOWN.csv calibrate) is RETIRED to  _ARCHIVE_legacy_predictors/plate_packer/mkm_suction_legacy.py  (see the NOTE.md there).  Every verdict now comes from physics/mkm_plate_verdict.py:
    evaluate_masks_range()  ->  decide_range():  PASS at the worst-case end of every estimated range, FAIL at the best-case end, otherwise UNSOLVED + "needs measurement: <field>".
Nothing in this file decides anything.
usage:  mkm_suction.py predict FILE.pm7m|FILE.pwsz   (its own layer images + settings -> layers.npz -> the same solver)
        mkm_suction.py predict PLATE.stl            (slices the plate with the exact slicer, then the physics core)
        mkm_suction.py predict PLATE_DIR           (a folder with layers.npz + settings.json = the exact layer masks of a slice)
exit code 0 only on PASS."""
import os, sys, json

HERE = os.path.dirname(os.path.abspath(__file__))
CORE_PHYS = os.environ.get("MKM_PHYSICS_DIR") or (os.path.join(os.path.dirname(HERE), "physics") if os.path.isdir(os.path.join(os.path.dirname(HERE), "physics")) else r"F:\MKM_STAGE\physics")


def _phys():
    if CORE_PHYS not in sys.path:
        sys.path.insert(0, CORE_PHYS)
    import mkm_core_api as api, mkm_planner_params as PP, mkm_plate_verdict as PV
    return api, PP, PV


# ---- sliced printer file adapter (.pm7m/.pwsz): RESTORED from the 11/14 predictor (Plate Packer\mkm_suction.py: UVT, _KEYS, read_settings, raw_sliced extraction).
# Only the packing into the layers.npz + settings.json format that api.read_layers_npz / api.settings_from_slice_json read is glue; no solver or decision logic lives here.
import re as _re, glob as _glob, shutil as _sh, subprocess as _sp
UVT = r"C:\Program Files\UVtools\UVtoolsCmd.exe"
_KEYS = ["LayerCount", "LayerHeight", "BottomLayerCount", "TransitionLayerCount", "BottomExposureTime", "ExposureTime",
         "BottomLiftHeight", "BottomLiftSpeed", "BottomLiftHeight2", "BottomLiftSpeed2", "LiftHeight", "LiftSpeed", "LiftHeight2",
         "LiftSpeed2", "WaitTimeBeforeCure", "BottomWaitTimeBeforeCure", "WaitTimeAfterCure", "BottomWaitTimeAfterCure",
         "WaitTimeAfterLift", "BottomWaitTimeAfterLift", "IsUsingTSMC", "MachineName", "PixelWidth", "PixelHeight", "AntiAliasing"]
_STAGE = os.environ.get("MKM_PREDICT_STAGE", os.path.join(os.path.dirname(HERE), "cache", "predict"))


def read_settings(pm):
    """printer / exposure / lift settings stored IN the sliced file (UVtools print-properties; speeds in mm/min). (verbatim from the 11/14 predictor)"""
    out = _sp.run([UVT, "--no-progress", "print-properties", pm, "-n"] + _KEYS, capture_output=True, text=True, errors="replace").stdout
    s = {}
    for line in out.splitlines():
        m = _re.match(r"^(\w+): (.*)$", line.strip())
        if m and m.group(1) in _KEYS:
            s[m.group(1)] = m.group(2).strip()
    return s



def _area_resize_gpu(m, rows, cols):
    """exact area-average downscale of a bool mask on the GPU (same maths as cv2 INTER_AREA): cumulative sums sampled at fractional cell edges."""
    import torch
    x = m.float(); H, W = x.shape; dev = x.device
    def one(x, n_out, dim):
        n = x.shape[dim]
        cs = torch.cat([torch.zeros_like(x.narrow(dim, 0, 1)), x.cumsum(dim)], dim)
        e = torch.linspace(0, n, n_out + 1, device=dev, dtype=torch.float64)
        i0 = e.floor().long().clamp(max=n); fr = (e - i0).float(); i1 = (i0 + 1).clamp(max=n)
        a = cs.index_select(dim, i0); b = cs.index_select(dim, i1)
        shp = [1, 1]; shp[dim] = -1
        ce = a + (b - a) * fr.view(shp)
        return (ce.narrow(dim, 1, n_out) - ce.narrow(dim, 0, n_out)) / (n / n_out)
    return one(one(x, cols, 1), rows, 0)


def _fast_plate_dir(pm, s, out, cell_mm):
    """MKM_UVTOOLS=1/2: the same layers.npz + settings.json as below, but layers are decoded by gpu_scan\\mkm_slice_reader on the GPU (no UVtools extract)."""
    import zipfile, numpy as np, torch
    g = os.path.join(os.path.dirname(HERE), "gpu_scan")
    if g not in sys.path: sys.path.insert(0, g)
    import mkm_slice_reader as SR
    S = SR.open_slice(pm); W, H = S.res; n = S.n_layers
    lh = float(s["LayerHeight"]); pxw = float(s.get("PixelWidth", 0.0459)); pxh = float(s.get("PixelHeight", 0.0456))
    cols, rows = int(round(W * pxw / cell_mm)), int(round(H * pxh / cell_mm))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    with zipfile.ZipFile(os.path.join(out, "layers.npz"), "w", zipfile.ZIP_DEFLATED, allowZip64=True) as z:
        for k, v in (("layer_index", np.arange(n, dtype=np.int32)), ("z_top_mm", ((np.arange(n) + 1) * lh).astype(np.float32)),
                     ("layer_height_mm", np.float32(lh)), ("cell_mm", np.float32(cell_mm))):
            with z.open(k + ".npy", "w", force_zip64=True) as fh:
                np.lib.format.write_array(fh, np.asarray(v))
        with z.open("cov.npy", "w", force_zip64=True) as fh:
            np.lib.format.write_array_header_1_0(fh, dict(descr=np.lib.format.dtype_to_descr(np.dtype(np.float16)), fortran_order=False, shape=(n, rows, cols)))
            for i in range(n):
                fh.write(_area_resize_gpu(S.layer_gpu(i, dev), rows, cols).cpu().numpy().astype(np.float16).tobytes())
    info = dict(uvtools_decoded=s, source=pm, layers_decoded_by="mkm_slice_reader (GPU)")
    if os.environ.get("MKM_UVTOOLS") == "2":
        import mkm_uvtools_gpu as UG
        r = UG.plate_issues_gpu(pm); json.dump(r, open(os.path.join(out, "gpu_issues.json"), "w"))
        info["gpu_issues"] = {k: r[k] for k in ("suction", "suction_cup", "resin_trap", "island", "seconds", "grid_mm")}
    json.dump(info, open(os.path.join(out, "settings.json"), "w"), indent=1)
    return out


def sliced_to_plate_dir(pm, cell_mm=0.25, stage=None):
    """.pm7m/.pwsz -> folder with layers.npz (cov [K,rows(y),cols(x)] float16 0..1 at cell_mm cells; z_top_mm; cell_mm) + settings.json {uvtools_decoded}. Streams layer by layer (no stack in RAM)."""
    import zipfile, numpy as np, cv2
    s = read_settings(pm)
    if "LayerHeight" not in s:
        raise RuntimeError("UVtools returned no print properties for " + pm)
    tag = _re.sub(r"[^\w]+", "_", os.path.basename(pm))[:60]
    out = stage or os.path.join(_STAGE, "plate_" + tag); tmp = os.path.join(_STAGE, "x_" + tag)
    for d in (out, tmp):
        os.makedirs(d, exist_ok=True)
    if os.environ.get("MKM_UVTOOLS") in ("1", "2") and pm.lower().endswith(".pm7m"):
        _sh.rmtree(tmp, ignore_errors=True)
        return _fast_plate_dir(pm, s, out, cell_mm)
    _sp.run([UVT, "--no-progress", "extract", pm, tmp, "-c", "Layers"], capture_output=True)
    fs = _glob.glob(os.path.join(tmp, "**", "*.png"), recursive=True)
    fs = [f for f in fs if _re.search(r"layer", os.path.basename(f), _re.I)] or fs
    fs.sort(key=lambda f: int((_re.findall(r"(\d+)", os.path.basename(f)) or ["0"])[-1]))
    if not fs:
        raise RuntimeError("UVtools extracted no layer images from " + pm)
    lh = float(s["LayerHeight"]); pxw = float(s.get("PixelWidth", 0.0459)); pxh = float(s.get("PixelHeight", 0.0456))
    im0 = cv2.imread(fs[0], cv2.IMREAD_GRAYSCALE); H, W = im0.shape
    cols, rows = int(round(W * pxw / cell_mm)), int(round(H * pxh / cell_mm))
    with zipfile.ZipFile(os.path.join(out, "layers.npz"), "w", zipfile.ZIP_DEFLATED, allowZip64=True) as z:
        for k, v in (("layer_index", np.arange(len(fs), dtype=np.int32)), ("z_top_mm", ((np.arange(len(fs)) + 1) * lh).astype(np.float32)),
                     ("layer_height_mm", np.float32(lh)), ("cell_mm", np.float32(cell_mm))):
            with z.open(k + ".npy", "w", force_zip64=True) as fh:
                np.lib.format.write_array(fh, np.asarray(v))
        with z.open("cov.npy", "w", force_zip64=True) as fh:
            np.lib.format.write_array_header_1_0(fh, dict(descr=np.lib.format.dtype_to_descr(np.dtype(np.float16)), fortran_order=False, shape=(len(fs), rows, cols)))
            for f in fs:
                im = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
                _, b = cv2.threshold(im, 127, 1, cv2.THRESH_BINARY)                 # cured = grey > 127 (as the 11/14 predictor)
                fh.write(cv2.resize(b.astype(np.float32), (cols, rows), interpolation=cv2.INTER_AREA).astype(np.float16).tobytes())
    json.dump(dict(uvtools_decoded=s, source=pm), open(os.path.join(out, "settings.json"), "w"), indent=1)
    _sh.rmtree(tmp, ignore_errors=True)
    return out


def core_forces(plate_dir, profiles=None, device=None, bed_xy_mm=None, **kw):
    """Range verdict for one sliced plate folder (layers.npz + settings.json). Returns dict(core, checks, verdict, decision, banner, note)."""
    api, PP, PV = _phys()
    npz = os.path.join(plate_dir, "layers.npz"); sj = os.path.join(plate_dir, "settings.json")
    PRM = PP.load()
    prof = profiles or api.load_profiles(settings=api.settings_from_slice_json(sj) if os.path.exists(sj) else "settings_factory_m7max", choice="central")
    prof, _ = PP.apply_profile_overlay(prof, PRM)
    L = api.read_layers_npz(npz, indices="all", mask_thr=0.5)
    v, info = PV.evaluate_masks_range(L["cov"], L["z_top_mm"], prof, PRM, device=device, dx_m=L["cell_mm"] * 1e-3, plate_tag=os.path.basename(plate_dir), **kw)
    res = info["central"]; checks = PV.checks_from_core(res, prof)
    return dict(core=res, checks=checks, verdict=v["verdict"], decision=v, banner=res.get("banner", ""), note=PV.verdict_line(v))


def predict(path, device=None, **kw):
    """-> (None, decision dict with key 'verdict').  A .stl is sliced; a folder is read as layers.npz; a sliced .pm7m/.pwsz must first be turned into layers.npz (not done here: UNSOLVED says so)."""
    api, PP, PV = _phys()
    if os.path.isdir(path):
        return None, core_forces(path, device=device, **kw)
    if path.lower().endswith(".stl"):
        PRM = PP.load(); prof = api.load_profiles(settings="settings_mkm_mini")
        bed = kw.pop("bed_xy_mm", None) or (297.5, 164.0)   # M7 Max display 6480 x 3600 px (profile machine.pixel_pitch_um); caller may pass bed_xy_mm
        v, res = PV.evaluate_plate_stl(path, prof, PRM, bed, device=device, **kw)
        return None, dict(core=res, checks=(res or {}).get("checks", []), verdict=v["verdict"], decision=v, banner="PHYSICS CORE UNVERIFIED" if not v.get("verified_core") else "", note=PV.verdict_line(v))
    if path.lower().endswith((".pm7m", ".pwsz", ".pm7")):
        # sliced printer file: its OWN layer images + its OWN settings (restored from the 11/14 predictor, mkm_suction raw_sliced/read_settings) -> layers.npz folder -> the ONE solver (core_forces)
        stage = sliced_to_plate_dir(path)
        return None, core_forces(stage, device=device, **kw)
    v = PV.unsolved_no_solver("%s is neither .stl, .pm7m/.pwsz nor a layers.npz folder" % os.path.basename(path))
    return None, dict(core=None, checks=[], verdict=v["verdict"], decision=v, banner="", note=PV.verdict_line(v))


def report_lines(name, d):
    return ["PREDICT %s: %s" % (name, d["note"])]


def _main():
    a = sys.argv[1:]
    if len(a) >= 2 and a[0] == "predict":
        _raw, d = predict(a[1])
        print(d["banner"], "|", a[1], "|", d["note"], flush=True)
        for c in d["checks"][:40]:
            print("  ", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in c.items()}, flush=True)
        if "--json" in a:
            json.dump(d["decision"], open(a[a.index("--json") + 1], "w", encoding="utf-8"), indent=1, default=str)
        sys.exit(0 if d["verdict"] == "PASS" else 1)
    print(__doc__); sys.exit(2)


if __name__ == "__main__":
    _main()
