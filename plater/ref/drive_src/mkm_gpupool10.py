"""PLANNER FIX 20261116 (see PLANNER_FIX_REPORT_20261116.md): PLATE_CHECK verdict = physics solver output (PASS/FAIL/UNSOLVED + governing number; no hard-coded verdict);
hand-set constants (PULL, AREA, KN, raft band 0.30, GAP-0.6, blob sigma/stride, EFF, CARRY, SPL ...) now come from ONE parameter file (physics/profiles_filled/planner_params_*.json) or from the solver;
ntri = finite triangles; a figure's body+base are placed together; big models are FLAGGED, never held.
MKM GPU POOL PLANNER v10 = pool9 with the placement cost from the PHYSICS CORE (mkm_core_api; wiring pass 11/14 night; PHYSICS CORE UNVERIFIED).
Changed vs pool9: (1) bed size from the machine PROFILE (bed_mm), not 297.5/164; (2) film membrane = core mp.membrane_deflection with tension and film window from the
PROFILES (GB_FILM 12 mm assumption gone); (3) per-piece load = core force (rigid-film Reynolds squeeze with cavitation cap, mp.squeeze_film, coarse path; UPPER bound of the
compliant field solve) in newtons from profile mu, h0, p_c, lift stage - the hand constant KN is GONE (20261116: the allotment unit is now the same solver load in N);
(4) GB_SYNTH=1 = synthetic pieces, no STL, no files written. Every number still hand-set is printed at start (HAND-SET list).
MKM GPU POOL PLANNER v8 (Claude 11/14) - ONE process, a stack of NPOOL empty beds on the GPU, pieces placed DIRECTLY on the real geometry.
Mike 11/14 17:25: "the plater should have its own mock-up of the slicer ... a template that has 16 empty plates, and the math is a direct mirror of
reality, and then it will plate based off of its mock-up, and have all the models fit legally ... instantaneously." (11/8: yes to placing pieces directly
into the rings by the peel equations instead of a separate re-layout pass.)  Rounds (Mike 11/14 12:42): large pieces first (outer ring), then medium, then
small ringing inward to the 32 mm circle.  Legality, masks, zone_share and the load math are the same as mkm_gpulayout_fc2.py (shared fp_cache).
The three allotments (cured-area profile, raft pull, bed fill) are LIVE checks on each bed as pieces land. CARRY, POOL_CARRY.json, big-model gate kept.
usage: mkm_gpupool8.py PIECES.json OUT_DIR TAGPREFIX   (same call as the other planners; GB_PLANNER=mkm_gpupool8.py)"""
import sys, os, json, struct, time, hashlib, re as _re, numpy as np
from scipy import ndimage
HERE = os.environ.get("MKM_PACKER_DIR", r"C:\MKM_TMP\plater_one_build_20261117\tree\plate_packer"); sys.path.insert(0, HERE)        # env overrides: the laptop defaults are unchanged
PHYS_DIR = os.environ.get("MKM_PHYS_DIR", os.path.abspath(os.path.join(os.path.dirname(__file__),"..","physics"))); GPUSCAN_DIR = os.environ.get("MKM_GPUSCAN_DIR", r"C:\MKM_TMP\plater_one_build_20261117\tree\gpu_scan")
sys.path.insert(0, PHYS_DIR); sys.path.insert(0, GPUSCAN_DIR)
import mkm_planner_params as PP, mkm_planner_geom as GEO, mkm_plate_verdict as PV
import mkm_ringpack as RP, mkm_fullheight as FH
import mkm_calibration_rules as CAL
import torch, torch.nn.functional as F
# 20261115 Mike's GPU ruling (11/14 "everything that can done with the gpu needs to happen on the gpu utilizing its capacity safely"; 11/15 01:42 "That behavior needs to stop"):
# device = env MKM_DEV, default cuda when available (MKM_DEV=cpu forces the original CPU code everywhere). The piece-mask rasterisation (compute_piece) and the measuring step run on the GPU
# through mkm_gpuport (bit-identical to the CPU code, proven by _gpu_port\test_equiv1.py); VRAM is capped and no work is launched at >= 87 C.
MKM_DEV = os.environ.get("MKM_DEV", "cuda").lower()
DEVSTR = "cuda" if (MKM_DEV != "cpu" and torch.cuda.is_available()) else "cpu"
import mkm_gpuport as GP
GPU_RASTER = (DEVSTR == "cuda") and (GP.configure() is not None)
T0 = time.time()
NPOOL = int(os.environ["GB_NPOOL"]) if os.environ.get("GB_NPOOL") else None      # 20261117: no literal plate count. None = taken from the job below (beds the job has to fill, capped by the VRAM admission); GB_NPOOL = the job states its own count
print(f"planner=mkm_gpupool10.py position=film-membrane energy from the physics core (PHYSICS CORE UNVERIFIED) NPOOL={NPOOL}", flush=True)
import mkm_core_api as api
SYNTH = os.environ.get("GB_SYNTH") == "1"
_pn = os.environ.get("GB_PROFILES", "anycubic_m7max,film_acf,resin_sunlu_abslike,settings_mkm_mini").split(",")
PROF = api.load_profiles(*_pn, directory=os.environ.get("GB_PROFILE_DIR", api.PROFILE_DIR), choice=os.environ.get("GB_BRACKET", "central"))
PRM = PP.load(choice=os.environ.get("GB_BRACKET", "central"), printer=_pn[0], film=_pn[1], resin=_pn[2])          # ONE parameter file per printer+resin (GB_PARAMS overrides the path); null = pending, never a hidden default
PROF, _filled = PP.apply_profile_overlay(PROF, PRM)                     # film.poisson (estimate bracket) only while the film profile has it pending
print("profiles:", PROF.names, "bracket choice:", PROF.choice, "| parameter file:", PRM.path, "| overlay filled:", _filled, flush=True)
print("PLANNER PARAMETERS (every non-solver number; see the parameter file for meaning/source):"); [print("  " + l) for l in PRM.describe()]
LH_MM = PROF.get("settings", "layer_height")[0]      # layer height from the settings profile
if SYNTH:
    OUT, TAG = os.environ.get("GB_SYNTH_OUT", os.path.join(PHYS_DIR, "_synth_pool10")), "SYNTH"; rows = []
else:
    rows = json.load(open(sys.argv[1], encoding="utf-8")); OUT, TAG = sys.argv[2], sys.argv[3]
CALIBRATED = _pn[:3] == ["anycubic_m7max", "film_acf", "resin_sunlu_abslike"]
BOTTOM_S = PROF.get("settings", "bottom_exposure_time")[0]
if CALIBRATED:
    refusal = CAL.exposure_refusal(BOTTOM_S)
    if refusal:
        sys.exit(refusal)
    print(CAL.ASSUMPTIONS, flush=True)
os.makedirs(OUT, exist_ok=True)
_ls, _ = PROF.lift_stages(False); LIFT = _ls[0][1] if _ls else None      # first-stage lift speed mm/min from the settings profile (was the literal 60); used only to label the job file
PX, PY = [float(v) for v in PROF.raw["machine"]["_known_facts_not_used_by_core"]["bed_mm"]["value"]]      # bed from the machine profile
R, EDGE, GAP, CLEAR = PRM.get("raster_pitch_mm"), PRM.get("edge_margin_mm"), PRM.get("min_gap_mm"), PRM.get("centre_clear_diam_mm")      # search geometry: named parameters, sources in the parameter file
RAFT_BAND = PRM.get("raft_band_mm"); GP.RAFT_BAND_MM = RAFT_BAND                                            # raft/first-layer contact band (was the literal 0.30 here and in mkm_gpuport)
FBUD = PRM.get("plate_force_budget_N", pending_ok=True); ABUD = PRM.get("layer_area_budget_mm2", pending_ok=True)   # allotments: None = PENDING = inactive (were PULL 194 N, AREA 5100 mm2)
EFF = PRM.get("packing_efficiency"); CENTRE_SMALL = bool(PRM.get("centre_zone_open_to_small"))
USABLE = (PX - 2 * EDGE) * (PY - 2 * EDGE) - np.pi * (CLEAR / 2) ** 2; BED = USABLE * EFF
SPL = PRM.get("seconds_per_layer"); CARRY = PRM.get("min_plate_use_fraction"); MAXSPLIT = int(PRM.get("max_plates_per_model"))
W, H = int(PX / R), int(PY / R)
NAMES = ['back-left', 'back-centre', 'back-right', 'middle-left', 'middle-centre', 'middle-right', 'front-left', 'front-centre', 'front-right']
DT = np.dtype([('n', '<f4', (3,)), ('v', '<f4', (3, 3)), ('a', '<u2')]); CACHE = os.path.join(OUT, "piece_metrics_synth.json") if SYNTH else os.path.join(GPUSCAN_DIR, "piece_metrics.json")
# ---------------- piece masks: identical to fc2 (same cache dir/version) ----------------
def rot(a, k):
    c, s = [(1, 0), (0, 1), (-1, 0), (0, -1)][k]; o = a.copy(); o[..., 0] = c * a[..., 0] - s * a[..., 1]; o[..., 1] = s * a[..., 0] + c * a[..., 1]; return o
def dense_pts(tv):
    xy = tv[:, :, :2].astype(np.float64); e = np.stack([np.linalg.norm(xy[:, 1] - xy[:, 0], axis=1), np.linalg.norm(xy[:, 2] - xy[:, 1], axis=1), np.linalg.norm(xy[:, 0] - xy[:, 2], axis=1)], 1).max(1)
    n = np.ceil(e / 0.25).astype(int); out = [tv.reshape(-1, 3)]
    for lvl in (2, 4, 8, 16, 32, 64, 128, 256, 512, 1024):
        sel = np.nonzero((n > lvl // 2) & ((n <= lvl) | (lvl == 1024)))[0]
        if not sel.size: continue
        i, j = np.meshgrid(np.arange(lvl + 1), np.arange(lvl + 1), indexing="ij"); m = (i + j) <= lvl
        wa = (i[m] / lvl).astype(np.float32); wb = (j[m] / lvl).astype(np.float32); wc = 1 - wa - wb; step = max(1, 4000000 // wa.size)
        for c0 in range(0, sel.size, step):
            t = tv[sel[c0:c0 + step]].astype(np.float32)
            out.append((wa[None, :, None] * t[:, 0][:, None, :] + wb[None, :, None] * t[:, 1][:, None, :] + wc[None, :, None] * t[:, 2][:, None, :]).reshape(-1, 3))
    return np.concatenate(out)
CACHE_DIR = os.path.join(GPUSCAN_DIR, "fp_cache"); CACHE_VER = "fc3_raft%g" % RAFT_BAND        # the masks depend on the raft band: it is part of the cache key now
def load_raw(path):
    raw = np.fromfile(path, dtype=DT, offset=84); raw = raw[np.isfinite(raw['v'].reshape(-1, 9)).all(1)]; zmn = float(raw['v'][:, :, 2].min())
    if abs(zmn + 0.005) > 0.02:
        v_ = raw['v'].copy(); v_[:, :, 2] -= np.float32(zmn + 0.005); raw['v'] = v_
    return raw, zmn
_FILE_ID = {}
def file_identity(path):
    """One read. sha256 of the file bytes, finite-triangle count, and the z min load_raw returns before the drop. Same tests as load_raw and GEO.finite_tri_count. (Grok Build, loop cycle 1; applied by Claude 20261117 16:15)"""
    if path in _FILE_ID: return _FILE_ID[path]
    _st = os.stat(path); _k = f"{os.path.abspath(path)}|{_st.st_size}|{_st.st_mtime_ns}"
    if _FID_DISK[0] is None:
        try: _FID_DISK[0] = json.load(open(FID_FILE))
        except Exception: _FID_DISK[0] = {}
    if _k in _FID_DISK[0]:
        _v = _FID_DISK[0][_k]; _FILE_ID[path] = (int(_v[0]), float.fromhex(_v[1]), _v[2]); return _FILE_ID[path]
    h = hashlib.sha256(); nfin = 0; zmn = None
    size = os.path.getsize(path)
    with open(path, "rb") as fh:
        head = fh.read(84); h.update(head)
        n = int(np.frombuffer(head[80:84], "<u4")[0])
        if 84 + 50 * n != size: raise ValueError("%s: header says %d triangles but the file size says %s" % (path, n, (size - 84) / 50.0))
        left = n
        while left:
            take = min(20000, left); blk = fh.read(50 * take); h.update(blk); left -= take
            raw = np.frombuffer(blk, dtype=DT, count=take)
            fin = np.isfinite(raw["v"].reshape(-1, 9)).all(1); nfin += int(fin.sum())
            if fin.any():
                z = float(raw["v"][fin][:, :, 2].min()); zmn = z if zmn is None else min(zmn, z)
    if zmn is None: raise ValueError("no finite triangle in " + path)
    _FILE_ID[path] = (nfin, zmn, h.hexdigest()); _FID_DISK[0][_k] = [nfin, float(zmn).hex(), h.hexdigest()]; _FID_NEW[0] += 1; return _FILE_ID[path]
FID_FILE = os.path.join(os.environ.get('MKM_FID_DIR', r'C:\MKM_TMP\plater_fid_cache'), 'file_identity.json'); _FID_DISK = [None]; _FID_NEW = [0]
def file_identity_save():
    if _FID_NEW[0]:
        os.makedirs(os.path.dirname(FID_FILE), exist_ok=True); _t = FID_FILE + f'.tmp{os.getpid()}'; json.dump(_FID_DISK[0], open(_t, 'w')); os.replace(_t, FID_FILE)
    print(f"file identity: {len(_FILE_ID) - _FID_NEW[0]} from disk, {_FID_NEW[0]} read", flush=True)
def cache_file(path):
    import mkm_contentkey as CK; key = f"{CACHE_VER}|{CK.content_sha(path)}"      # 20261117 content key
    return os.path.join(CACHE_DIR, os.path.splitext(os.path.basename(path))[0][:40].replace(" ", "_") + "_" + hashlib.sha1(key.encode("utf-8")).hexdigest()[:20] + ".npz")
# 20261117 CACHE ALIAS (undo C:\MKM_TMP\backup\plater_cachealias_20261117\UNDO_cache_alias_and_placeonly.cmd): the content-key lookup falls back to the entry the SAME file already has under a LEGACY key
# (path|size|mtime, or the 11/16 name|size|mtime key), read-only from the pre-index folder; the mapping is recorded as a small alias file on C: so the next lookup is a direct hit. Nothing is written to the pre-index folder.
LEGACY_CACHE_DIR = os.environ.get("MKM_FP_LEGACY_DIR", r"F:\MKM_STAGE\gpu_scan\fp_cache"); ALIAS_DIR = os.environ.get("MKM_FP_ALIAS_DIR", r"C:\MKM_TMP\plater_fp_alias"); FP_MEASURED = [0]; FP_HOW = {"content": 0, "alias": 0, "legacy": 0}
LEGACY_DIRS = [LEGACY_CACHE_DIR] + [d_ for d_ in os.environ.get("MKM_FP_EXTRA_DIRS", r"F:\MKM_STAGE\PLATER_TEST_20261115\cache\fp_cache;F:\MKM_STAGE\PLATER_GPU_RUN_20261116\tree\PLATER_ONE_20261116\gpu_scan\fp_cache").split(";") if d_]      # run 3: other complete (fc3, rm0-3) caches, read only
def _has_rm(npz_path):
    import zipfile
    try: return "rm0.npy" in zipfile.ZipFile(npz_path).namelist()      # the raft-band masks this planner needs; fc1/fc2 files (the 11/16 night indexer wrote those) lack them
    except Exception: return False
def legacy_cache_files(path):
    st = os.stat(path); b = os.path.splitext(os.path.basename(path))[0][:40].replace(" ", "_")
    keys = []      # the key forms the pre-index was actually written under (found by trying them on the 8,904 files: fc1|path|size|mtime 5162, fc2|path|.. 223, fc2n|name|size|mtime 382 of 5781 indexed files); current version first
    for v in (CACHE_VER, "fc2", "fc1"): keys += [f"{v}|{os.path.abspath(path)}|{st.st_size}|{int(st.st_mtime)}", f"{v}n|{os.path.basename(path)}|{st.st_size}|{int(st.st_mtime)}"]
    return [os.path.join(dd, b + "_" + hashlib.sha1(k.encode("utf-8")).hexdigest()[:20] + ".npz") for k in keys for dd in [CACHE_DIR] + LEGACY_DIRS]
def resolve_cache(path):
    """-> (npz path, how) or (None, None). how = content | alias | legacy. Never measures."""
    st = os.stat(path); af = os.path.join(ALIAS_DIR, hashlib.sha1(f"{os.path.abspath(path)}|{st.st_size}|{st.st_mtime_ns}".encode("utf-8")).hexdigest()[:24] + ".json")      # alias keyed by the file's stat stamp: a direct hit needs no hashing (a 100 MB read on the slow drive)
    if os.path.exists(af):
        try:
            t = json.load(open(af, encoding="utf-8"))["npz"]
            if os.path.exists(t) and _has_rm(t): return t, "alias"
        except Exception: pass
    for lf in legacy_cache_files(path):
        if os.path.exists(lf) and _has_rm(lf):
            try:
                os.makedirs(ALIAS_DIR, exist_ok=True); tmp = af + f".tmp{os.getpid()}"; json.dump(dict(npz=lf, path=os.path.abspath(path)), open(tmp, "w", encoding="utf-8")); os.replace(tmp, af)
            except Exception as ex: print("alias write failed:", str(ex)[:100], flush=True)
            return lf, "legacy"
    cf = cache_file(path)      # content key (hashes the file once, memoised in plate_packer\content_sha_memo.jsonl); reached only when no alias / legacy entry exists
    if os.path.exists(cf): return cf, "content"
    return None, None
def compute_piece(q):
    m = RP.model(q["path"], q["set"]); raw, zmn = load_raw(q["path"])
    a = np.asarray(FH.piece(q["path"])[0], float)
    rots = []
    if GPU_RASTER:
        try: return dict(raft=float(m["raft"]), h=float(m["h"]), zmn=zmn, area=a, rots=GP.cp_masks(np.ascontiguousarray(raw["v"])))
        except Exception as ex: print("GPU raster failed, CPU path for this piece:", str(ex)[:120], flush=True)
    dp = dense_pts(raw["v"])
    for k in range(4):
        v = rot(dp, k); v = v[np.isfinite(v).all(1) & (np.abs(v) < 1e4).all(1)]; mn = v[:, :2].min(0); pad = 4
        ix = ((v[:, 0] - mn[0]) / R).astype(int) + pad; iy = ((v[:, 1] - mn[1]) / R).astype(int) + pad
        g = np.zeros((ix.max() + pad + 1, iy.max() + pad + 1), bool); g[ix, iy] = True; g = ndimage.binary_fill_holes(ndimage.binary_closing(g, iterations=3))
        xs, ys = np.nonzero(g); c0 = (xs.min(), ys.min()); g = g[xs.min():xs.max() + 1, ys.min():ys.max() + 1]
        sl = v[:, 2] <= v[:, 2].min() + RAFT_BAND; rx = ix[sl] - c0[0]; ry = iy[sl] - c0[1]; kp = (rx >= 0) & (rx < g.shape[0]) & (ry >= 0) & (ry < g.shape[1])
        rm = np.zeros(g.shape, bool); rm[rx[kp], ry[kp]] = True
        if not rm.any(): rm = g.copy()
        rots.append(dict(mask=g, rmask=rm, mn=mn, c0=c0, pad=pad))
    return dict(raft=float(m["raft"]), h=float(m["h"]), zmn=zmn, area=a, rots=rots)
def _synth_piece(spec):
    kind, a_, b_ = spec[:3]; pad = 4; n = int(round(2 * a_ / R)) if kind == "disc" else int(round(a_ / R)); m_ = int(round(2 * a_ / R)) if kind == "disc" else int(round(b_ / R))
    ii, jj = np.meshgrid(np.arange(n) + 0.5, np.arange(m_) + 0.5, indexing="ij"); g = ((ii - n / 2) ** 2 + (jj - m_ / 2) ** 2 <= (a_ / R) ** 2) if kind == "disc" else np.ones((n, m_), bool)
    rots = []
    for k in range(4):
        gk = np.rot90(g, k).copy(); rots.append(dict(mask=gk, rmask=gk.copy(), mn=np.zeros(2), c0=(0, 0), pad=pad))
    return dict(raft=float(g.sum() * R * R), h=(float(spec[5]) if len(spec) > 5 else 20.0), zmn=-0.005, area=np.full(40, float(g.sum() * R * R)), rots=rots)
FILL_DIR = os.environ.get("MKM_FP_FILL_DIR", r"C:\MKM_TMP\plater_fp_filled")      # complete (fc3, rm0-3) entries produced by the raft-band fill; on C:, the pre-index folder on F: is never written
def _rots_from_file(raw):
    """the 4 rotation dicts (mask, rmask, mn, c0, pad) by the SAME code a full measurement uses: GPU raster when available, else the CPU loop of compute_piece."""
    if GPU_RASTER:
        try: return GP.cp_masks(np.ascontiguousarray(raw["v"]))
        except Exception as ex: print("GPU raster failed, CPU path for this piece:", str(ex)[:120], flush=True)
    dp = dense_pts(raw["v"]); rots = []
    for k in range(4):
        v = rot(dp, k); v = v[np.isfinite(v).all(1) & (np.abs(v) < 1e4).all(1)]; mn = v[:, :2].min(0); pad = 4
        ix = ((v[:, 0] - mn[0]) / R).astype(int) + pad; iy = ((v[:, 1] - mn[1]) / R).astype(int) + pad
        g = np.zeros((ix.max() + pad + 1, iy.max() + pad + 1), bool); g[ix, iy] = True; g = ndimage.binary_fill_holes(ndimage.binary_closing(g, iterations=3))
        xs, ys = np.nonzero(g); c0 = (xs.min(), ys.min()); g = g[xs.min():xs.max() + 1, ys.min():ys.max() + 1]
        sl = v[:, 2] <= v[:, 2].min() + RAFT_BAND; rx = ix[sl] - c0[0]; ry = iy[sl] - c0[1]; kp = (rx >= 0) & (rx < g.shape[0]) & (ry >= 0) & (ry < g.shape[1])
        rm = np.zeros(g.shape, bool); rm[rx[kp], ry[kp]] = True
        if not rm.any(): rm = g.copy()
        rots.append(dict(mask=g, rmask=rm, mn=mn, c0=c0, pad=pad))
    return rots
def fill_raft_band(q):
    """-> piece dict, or None when the file has no footprint entry under any legacy key (then it is NOT measured here). Reuses raft/h/zmn/area from the existing entry; rasters the masks once; saves a complete entry to FILL_DIR + an alias."""
    st = os.stat(q["path"]); ff = os.path.join(FILL_DIR, os.path.splitext(os.path.basename(q["path"]))[0][:40].replace(" ", "_") + "_" + hashlib.sha1(f"{CACHE_VER}|{os.path.abspath(q['path'])}|{st.st_size}|{st.st_mtime_ns}".encode("utf-8")).hexdigest()[:20] + ".npz")
    if os.path.exists(ff) and _has_rm(ff):
        z = np.load(ff, allow_pickle=False)
        return dict(raft=float(z["raft"]), h=float(z["h"]), zmn=float(z["zmn"]), area=z["area"], rots=[dict(mask=z[f"mask{k}"], rmask=z[f"rm{k}"], mn=z[f"mn{k}"], c0=(z[f"c0{k}"][0], z[f"c0{k}"][1]), pad=int(z["pad"])) for k in range(4)])
    lf = next((f_ for f_ in legacy_cache_files(q["path"]) if os.path.exists(f_)), None)
    if lf is None: return None
    import psutil      # 20261117 14:27: a very large model file took the first run to 16 GB of commit (load + copies, then the CPU fallback). A file that does not fit the RAM room is LEFT OUT and logged, never forced through.
    _room = psutil.virtual_memory().available - 3 * 2 ** 30; _need = st.st_size * 5      # load_raw holds the raw array, a finite-filtered copy and a z-shifted copy; the raster adds the dense points
    if _need > _room: print(f"RAFT FILL: left out, file too large for the RAM room right now ({st.st_size / 2 ** 20:.0f} MB file, needs about {_need / 2 ** 30:.1f} GiB, room {_room / 2 ** 30:.1f} GiB): {q['path']}", flush=True); return None
    z = np.load(lf, allow_pickle=False); raw, zmn = load_raw(q["path"])
    try: rots = GP.cp_masks(np.ascontiguousarray(raw["v"])) if GPU_RASTER else _rots_from_file(raw)      # on the GPU a refusal (RAM / VRAM / temperature guard) leaves the piece out; it does NOT fall back to the much heavier CPU raster
    except Exception as ex: print(f"RAFT FILL: left out, raster refused or failed ({str(ex)[:120]}): {q['path']}", flush=True); del raw; return None
    del raw
    import gc; gc.collect()      # give the model's raw arrays back before the next piece loads
    try:
        import torch
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    except Exception: pass
    for k in range(4):      # the re-rastered footprint must be the stored footprint: same code, same pitch. A mismatch means the file changed since it was indexed: do not mix old numbers with new masks.
        if f"mask{k}" in z.files and (z[f"mask{k}"].shape != rots[k]["mask"].shape or not np.array_equal(z[f"mask{k}"], rots[k]["mask"])):
            print(f"RAFT FILL: footprint differs from the indexed one, piece left out (not measured): {q['path']} rot {k} stored {z[f'mask{k}'].shape} now {rots[k]['mask'].shape}", flush=True); return None
    d = dict(raft=float(z["raft"]), h=float(z["h"]), zmn=float(z["zmn"]), area=z["area"], rots=rots)
    os.makedirs(FILL_DIR, exist_ok=True); arrs = dict(raft=d["raft"], h=d["h"], zmn=d["zmn"], area=d["area"], pad=rots[0]["pad"])
    for k, r in enumerate(rots): arrs[f"mask{k}"] = r["mask"]; arrs[f"rm{k}"] = r["rmask"]; arrs[f"mn{k}"] = r["mn"]; arrs[f"c0{k}"] = np.array(r["c0"])
    tmp = ff[:-4] + f".tmp{os.getpid()}.npz"; np.savez_compressed(tmp, **arrs); os.replace(tmp, ff); FP_FILLED[0] += 1
    return d
FP_FILLED = [0]
def get_piece(q):
    if SYNTH: return _synth_piece(q["spec"]), True
    cf, _how = resolve_cache(q["path"])
    if cf is not None:
        try:
            z = np.load(cf, allow_pickle=False)
            d = dict(raft=float(z["raft"]), h=float(z["h"]), zmn=float(z["zmn"]), area=z["area"],
                     rots=[dict(mask=z[f"mask{k}"], rmask=z[f"rm{k}"], mn=z[f"mn{k}"], c0=(z[f"c0{k}"][0], z[f"c0{k}"][1]), pad=int(z["pad"])) for k in range(4)])      # an entry without the raft-band masks (rm0-3; fc1/fc2 files) cannot be used: KeyError -> treated as no entry
            FP_HOW[_how] += 1; return d, True
        except Exception: pass
    if os.environ.get("GB_FILL_RAFT") == "1":      # 20261117 14:25 RAFT-BAND FILL (undo C:\MKM_TMP\backup\plater_raftfill_20261117\UNDO_raftfill.cmd): the 11/16 indexer saved entries WITHOUT the raft-band masks (rm0-3).
        d = fill_raft_band(q)                      # They cannot be derived from the stored keys (they come from vertex heights), so the model file is opened ONCE for the mask raster only; raft/h/zmn/area are reused from the existing entry, nothing is re-measured.
        if d is not None: FP_HOW["filled"] = FP_HOW.get("filled", 0) + 1; return d, True
    if os.environ.get("GB_NO_MEASURE") == "1": raise RuntimeError("GB_NO_MEASURE=1: no cached entry (content, alias or legacy key) for " + q["path"])      # 20261117: never measure
    cf = cache_file(q["path"]); FP_MEASURED[0] += 1
    d = compute_piece(q); os.makedirs(CACHE_DIR, exist_ok=True); arrs = dict(raft=d["raft"], h=d["h"], zmn=d["zmn"], area=d["area"], pad=d["rots"][0]["pad"])
    for k, r in enumerate(d["rots"]): arrs[f"mask{k}"] = r["mask"]; arrs[f"rm{k}"] = r["rmask"]; arrs[f"mn{k}"] = r["mn"]; arrs[f"c0{k}"] = np.array(r["c0"])
    tmp = cf[:-4] + f".tmp{os.getpid()}.npz"; np.savez_compressed(tmp, **arrs); os.replace(tmp, cf)
    return d, False
_get_piece_once = get_piece; _GPM = {}
def get_piece(q):      # 20261117 15:26 loop cycle 2: same piece, same answer - the pre-pass and the load loop share one read
    if q["path"] not in _GPM: _GPM[q["path"]] = _get_piece_once(q)
    return _GPM[q["path"]]
# ---------------- (a) LOAD every piece's masks ONCE ----------------
if SYNTH:
    _spec = json.loads(os.environ.get("GB_SYNTH_PIECES", "[[\"disc\",4,0],[\"disc\",3,0],[\"rect\",8,4],[\"disc\",2,0],[\"rect\",4,4],[\"disc\",2.5,0]]"))
    for n_, sp in enumerate(_spec):
        dd = _synth_piece(sp); rows.append(dict(set=(sp[4] if len(sp) > 4 else f"SYN{n_}"), file=f"syn{n_}", path=f"SYN:{n_}", h=dd["h"], raft=dd["raft"], area=dd["area"].tolist(), spec=sp))
rows = [r for r in rows if not r["set"].upper().startswith("75MM")]
mcache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}; nhit = nmiss = 0; tL = time.time()
if os.environ.get("GB_NO_MEASURE") == "1" and not SYNTH:      # 20261117: a piece with no cached entry (footprint masks under any key, or foot/halo in piece_metrics) is skipped and logged, never measured
    _keep = []; _sk = []
    _bs = set()
    for r in rows:
        try: get_piece(dict(path=r["path"], set=r["set"], file=r["file"], spec=None))      # raises (never measures) when no entry under any key, or the entry cannot be read
        except Exception as ex: _sk.append((r["path"], "no usable footprint cache entry: " + str(ex)[:90])); _bs.add(r["set"]); continue
        if r["path"] not in mcache: _sk.append((r["path"], "no foot/halo entry in piece_metrics.json")); _bs.add(r["set"])
        else: _keep.append(r)
    _n0 = len(_keep); _keep = [r for r in _keep if r["set"] not in _bs]; _sk += [(r["path"], "complete sets only: another part of this set has no usable cache entry") for r in rows if r["set"] in _bs and r["path"] not in {s[0] for s in _sk}]; print(f"GB_NO_MEASURE: {_n0 - len(_keep)} further pieces dropped because their set is incomplete", flush=True)
    print(f"GB_NO_MEASURE: {len(_sk)} pieces skipped (not measured), {len(_keep)} kept", flush=True); json.dump(_sk, open(os.path.join(OUT, "SKIPPED_NO_CACHE.json"), "w"), indent=0); rows = _keep
for r in rows:
    d, hit = get_piece(dict(path=r["path"], set=r["set"], file=r["file"], spec=r.get("spec"))); nhit += hit; nmiss += (not hit); r["D"] = d
    a = np.zeros(600); a[:len(d["area"])] = d["area"]; r["prof"] = a; r["pull"] = 0.0      # load comes from the core below
    r["unit"] = ("SYN", r["set"]) if SYNTH else tuple(RP.P.model_unit(dict(path=r["path"], set=r["set"])))       # one figure = body + its own base (mkm_packer.model_unit); a SYNTH row may carry its own unit
    if SYNTH and r.get("spec") and len(r["spec"]) > 3: r["unit"] = ("SYN", r["spec"][3])
    if SYNTH: r["ntri"] = None
    else:
        try: r["ntri"], r["zfile"], r["sha256"] = file_identity(r["path"])      # one read: finite triangles, load_raw's z min, sha256
        except ValueError as _ex: print("file_identity fell back to the old count:", str(_ex)[:120], flush=True); r["ntri"] = GEO.finite_tri_count(r["path"])
    r["ntri"] = r["ntri"]         # ntri = triangles the plate file will contain (finite ones); never the header count
    if SYNTH and r["path"] not in mcache:
        g_ = r["D"]["rots"][0]["mask"]; mcache[r["path"]] = dict(foot=float(g_.sum() * R * R), halo=float(ndimage.binary_dilation(np.pad(g_, 8), iterations=int(GAP / 2 / R)).sum() * R * R))
    if r["path"] not in mcache and GPU_RASTER:
        try: mcache[r["path"]] = GP.foot_halo(np.fromfile(r["path"], dtype=DT, offset=84)["v"], R, int(GAP / 2 / R))
        except Exception as ex: print("GPU foot/halo failed, CPU path for this piece:", str(ex)[:100], flush=True)
    if r["path"] not in mcache:
        v = np.fromfile(r["path"], dtype=DT, offset=84)["v"].reshape(-1, 3); v = v[np.isfinite(v).all(1)]; mn = v[:, :2].min(0)
        ix = ((v[:, 0] - mn[0]) / R).astype(int) + 8; iy = ((v[:, 1] - mn[1]) / R).astype(int) + 8
        g = np.zeros((ix.max() + 9, iy.max() + 9), bool); g[ix, iy] = True; g = ndimage.binary_fill_holes(ndimage.binary_closing(g, iterations=3))
        mcache[r["path"]] = dict(foot=float(g.sum() * R * R), halo=float(ndimage.binary_dilation(g, iterations=int(GAP / 2 / R)).sum() * R * R))
    r["foot"], r["halo"] = mcache[r["path"]]["foot"], mcache[r["path"]]["halo"]
json.dump(mcache, open(CACHE, "w"))
file_identity_save()
T_LOAD = time.time() - tL
# ---- per-piece load from the CORE (coarse path: rigid-film Reynolds squeeze + cavitation cap on the raft footprint; upper bound of the field solve)
tC = time.time(); _cache_F = {}
import glob as _glob
_fbh = hashlib.sha1()
for _p in sorted(_glob.glob(os.path.join(os.path.dirname(os.path.abspath(api.__file__)), '**', '*'), recursive=True)):
    if _p.lower().endswith(('.py', '.json')) and os.path.isfile(_p) and '__pycache__' not in _p: _fbh.update(_p.encode('utf-8')); _fbh.update(open(_p, 'rb').read())
_fbh.update(json.dumps(PROF.raw, sort_keys=True, default=str).encode('utf-8')); _fbh.update(repr((R, 'bottom', 'pad2')).encode('utf-8'))
FB_FILE = os.path.join(os.environ.get('MKM_FB_DIR', r'C:\MKM_TMP\plater_fb_cache'), _fbh.hexdigest()[:24] + '.json'); _fb_disk = json.load(open(FB_FILE)) if os.path.exists(FB_FILE) else {}; _fb_new = 0
for r in rows:
    rm = r["D"]["rots"][0]["rmask"]; key = (rm.shape, int(rm.sum()), hashlib.sha1(np.packbits(rm).tobytes()).hexdigest())
    _dk = f"{key[0][0]}x{key[0][1]}|{key[1]}|{key[2]}"
    if key not in _cache_F and _dk in _fb_disk: _cache_F[key] = _fb_disk[_dk]
    if key not in _cache_F:
        Fv, info = api.piece_force_bound([np.pad(rm, 2)], PROF, R * 1e-3, bottom=True, device=DEVSTR)
        if Fv is None: sys.exit("PROFILE INCOMPLETE - no placement cost can be computed (no number invented). Pending: " + ", ".join(info["missing"]))
        _cache_F[key] = Fv[0]; _fb_disk[_dk] = float(Fv[0]); _fb_new += 1
    r["pull"] = _cache_F[key]
if _fb_new:
    os.makedirs(os.path.dirname(FB_FILE), exist_ok=True); _t = FB_FILE + f'.tmp{os.getpid()}'; json.dump(_fb_disk, open(_t, 'w')); os.replace(_t, FB_FILE)
print(f"coarse bound cache: {len(_cache_F) - _fb_new} from disk, {_fb_new} computed", flush=True)
print(f"core piece loads (coarse bound): {len(_cache_F)} distinct footprints, N min/median/max {min(r['pull'] for r in rows):.3g}/{np.median([r['pull'] for r in rows]):.3g}/{max(r['pull'] for r in rows):.3g}, {time.time() - tC:.1f}s", flush=True)
# force allotment reference (share of a plate's force that one piece uses): the parameter file's budget when it has one, else the pool's own mean plate load (solver numbers, no literal)
import mkm_dataflow as _DF
_need = int(NPOOL) if NPOOL else max(1, int(np.ceil(sum(r["halo"] for r in rows) / BED))) if rows else 1      # beds the job has to fill: total halo area / one bed's usable area (a lower bound; more beds only if the job says so)
_cap = max(1, int(_DF.vram_room() // (12 * W * H)))                                                              # the existing byte admission: per bed wf float64 (8 B) + occ float32 (4 B) per cell, the two NPOOL-sized tensors allocated below
NPOOL = min(_need, _cap)
print(f"NPOOL={NPOOL} beds: job needs {_need} ({'job-stated' if os.environ.get('GB_NPOOL') else 'sum halo / usable bed area'}), VRAM admission allows {_cap}", flush=True)
FREF = FBUD if FBUD else (sum(r["pull"] for r in rows) / NPOOL if rows else 1.0)
print("allotments: bed %.0f mm2 (usable %.0f x packing %.2f) | force budget %s | per-layer area budget %s | force reference for piece size %.4g N (%s)" % (
    BED, USABLE, EFF, "%.4g N" % FBUD if FBUD else "PENDING (inactive; solver verdict decides)", "%.4g mm2" % ABUD if ABUD else "PENDING (inactive)", FREF, "budget" if FBUD else "pool mean plate load from the core"), flush=True)
print(f"(a) LOAD masks: {len(rows)} pieces, cache {nhit} hit / {nmiss} miss, {T_LOAD:.1f}s | hit by content/alias/legacy key: {FP_HOW} | PIECES MEASURED THIS RUN: {FP_MEASURED[0]}", flush=True)
# ---------------- big-model FLAG (20261116: flagged, never held; every model stays in the pool) ----------------
M = {}
for r in rows: M.setdefault(r["set"], []).append(r)
TALL = PRM.get("tall_flag_mm"); ALLOW = PRM.get("big_model_allow_regex")
def _bed_share(g): return max([sum(x["halo"] for x in g) / BED] + ([sum(x["prof"] for x in g).max() / ABUD] if ABUD else []))
def _force_share(g): return sum(x["pull"] for x in g) / FBUD if FBUD else None
BIG = GEO.classify_big(M, TALL, ALLOW, _bed_share, _force_share)
gl = GEO.flag_report_lines(BIG, lambda h_: GEO.layer_hours(h_, LH_MM, SPL))
print("BIG-MODEL FLAGS (flagged for Mike before plating - NOT held, all models are plated):"); [print("  " + l) for l in gl]
open(os.path.join(OUT, "BUNDLE_FLAGS.txt"), "w", encoding="utf-8").write("\n".join(gl)); n = len(rows)
# ---------------- size = share of one bed's binding allotment (continuous); rounds from the pool's own terciles ----------------
def sz(x): return max(x["pull"] / FREF, x["halo"] / BED)       # share of one plate's binding allotment: solver force over the force reference, or halo over the bed
order = sorted(range(n), key=lambda i: -sz(rows[i])); rank = np.zeros(n)
for p, i in enumerate(order): rank[i] = p / max(n - 1, 1)            # 0 = largest piece ... 1 = smallest
_q2, _q1 = PRM.get("round_quantiles")
t2, t1 = np.quantile([sz(x) for x in rows], [_q2, _q1])
tier = [0 if sz(x) >= t2 else (1 if sz(x) >= t1 else 2) for x in rows]
# 20261116 body/base fix: a figure's pieces (body + its own base) travel together. The unit's round = the round of its BIGGEST piece; before, the body landed in the large round and its base in the
# small round, when the first plate was already full -> split. Units are placed atomically below (GEO.place_unit_atomic).
UNITS = {}
for i in range(n): UNITS.setdefault(GEO.unit_key(rows[i]), []).append(i)
utier = {u: min(tier[i] for i in idx) for u, idx in UNITS.items()}
rounds = [(nm, [u for u in sorted(UNITS, key=lambda u: -max(sz(rows[i]) for i in UNITS[u])) if utier[u] == tt]) for tt, nm in ((0, "large"), (1, "medium"), (2, "small"))]
print(f"rounds (units) from terciles of piece size (share of a bed's binding allotment): large >= {t2:.4f}, medium >= {t1:.4f}, small below | " + ", ".join(f"{a} {len(b)} units" for a, b in rounds), flush=True)
# ---------------- GPU: the mock-up = NPOOL empty beds, opened BEFORE the first piece ----------------
dev = torch.device(DEVSTR); xx = (torch.arange(W, device=dev) + 0.5) * R - PX / 2; yy = (torch.arange(H, device=dev) + 0.5) * R - PY / 2
circle = ((xx[:, None] ** 2 + yy[None, :] ** 2) <= (CLEAR / 2) ** 2).float() * float(CLEAR > 0)  # 32 mm centre floor removed - Mike 2026-11-17 10:09 (undo: C:\MKM_TMP\backup\centre_floor_20261117)
e = int(EDGE / R) + 1; g = int(round(GAP / R)); NB = 10
if CALIBRATED:
    # Separate full footprint cells, including a diagonal for their edge clearance.
    g = max(g, int(np.ceil(CAL.MIN_RAFT_GAP_MM / R + np.sqrt(2.0))))
BX = [0, int((xx < -PX / 6).sum()), W - int((xx > PX / 6).sum()), W]; BY = [0, int((yy < -PY / 6).sum()), H - int((yy > PY / 6).sum()), H]
OXg = torch.arange(W, device=dev)[:, None]; OYg = torch.arange(H, device=dev)[None, :]
OUTER = [0, 1, 2, 3, 5, 6, 7, 8]
# ---- FILM PHYSICS: membrane clamped at the vat frame, solved exactly on the grid (odd extension = zero deflection at the frame) ----
ENG = api.FilmCostEngine(PROF, W, H, R * 1e-3, device=DEVSTR)
if not ENG.ok: sys.exit("PROFILE INCOMPLETE for the film membrane: " + ", ".join(ENG.missing))
film = ENG.deflection
SC_DIR = os.environ.get("MKM_SC_DIR", r"C:\MKM_TMP\plater_sc_cache")
_SC = {}; SC_TOL = PRM.get("film_self_compliance_rel_tol")        # self-compliance of every footprint x rotation: exact membrane solves on a lattice refined to SC_TOL (no blob sigma / stride)
def self_c(rm_np):
    key = (rm_np.shape, hashlib.sha1(np.packbits(rm_np).tobytes()).hexdigest())
    if key not in _SC:
        _SC[key] = os.path.join(SC_DIR, hashlib.sha1(repr((key, W, H, R, float(ENG.T0), tuple(float(p_) for p_ in ENG.pad), SC_TOL)).encode("utf-8")).hexdigest()[:24] + ".npy")      # same footprint + same film + same grid + same tolerance = same map
        if os.path.exists(_SC[key]): return torch.from_numpy(np.load(_SC[key])).to(dev)
        m_, inf_ = GEO.self_compliance_map(ENG.deflection, W, H, R * 1e-3, rm_np, SC_TOL, device=DEVSTR)
        if not inf_["converged"]: sys.exit("UNSOLVED: film self-compliance lattice did not converge to %.3g (rel err %s, %d solves): no placement cost can be given" % (SC_TOL, inf_["rel_err"], inf_["solves"]))
        os.makedirs(SC_DIR, exist_ok=True); tmp_ = _SC[key][:-4] + f".tmp{os.getpid()}.npy"; np.save(tmp_, m_); os.replace(tmp_, _SC[key])
        return torch.from_numpy(m_).to(dev)
    return torch.from_numpy(np.load(_SC[key])).to(dev)      # 20261117 14:39: the map lives on DISK, and on the GPU only inside the bounded K cache. Held on the GPU for every piece it was 1.56 MB x 4 rotations x pieces = 2.3 GB at 368 pieces -> CUDA OOM under the 0.45 cap.
wf = torch.zeros((NPOOL, W, H), device=dev, dtype=torch.float64)
try: torch.backends.cuda.cufft_plan_cache.max_size = int(os.environ.get('GB_FFT_PLANS', 4))      # 20261117 14:50: every new batch shape (beds allowed, lattice chunk) left a cuFFT plan + its workspace on the GPU (default cache 4096 plans). A plan is a cache only: same numbers.
except Exception as _ex: print('cuFFT plan cache not limited:', _ex, flush=True)
print(f"film model: core membrane (mp.membrane_deflection), T0 {ENG.T0:.4g} N/m ({ENG.tags['T0']}), window {ENG.tags['window']}, pad m {tuple(round(p_, 4) for p_ in ENG.pad)}; assumptions: {ENG.assumptions}; self-compliance: exact lattice solves refined to rel tol {SC_TOL}", flush=True)
occ = torch.zeros((NPOOL, W, H), device=dev); load = torch.zeros((NPOOL, 9), device=dev)
prof = np.full((NPOOL, 600), ABUD if ABUD else np.inf); forceleft = np.full(NPOOL, FBUD if FBUD else np.inf); bedleft = np.full(NPOOL, BED)      # remaining allotments (inactive = infinite)
placements = [[] for _ in range(NPOOL)]; span = {}; seq = [0]
print(f"{NPOOL} empty beds open on the GPU before the first piece is placed ({W}x{H} px at {R} mm, gap {GAP} mm, edge {EDGE} mm, circle {CLEAR} mm)", flush=True)
def zone_share(K):
    C = K["C"]; G = [[C[(bx - OXg).clamp(0, K["sw"]), (by - OYg).clamp(0, K["sh"])] for by in BY] for bx in BX]; out = []
    for z in range(9):
        r_, c_ = divmod(z, 3); ci = 2 - c_; ri = 2 - r_
        out.append(G[ci + 1][ri + 1] - G[ci][ri + 1] - G[ci + 1][ri] + G[ci][ri])
    return torch.stack(out) / K["rt"]
from collections import OrderedDict
Kc = OrderedDict(); KCAP = int(os.environ.get("GB_KCACHE", 24))      # 20261115 PLUMBING FIX: bounded LRU (K is a pure function of piece i; recompute on a miss is identical). Unbounded, K held ~35 MB/piece = ~8.6 GB for 246 pieces -> CUDA OOM.
def get_K(i):
    if i in Kc: Kc.move_to_end(i); return Kc[i]
    out = []
    # 20261115: the 4 rotations' FFTs go through ONE batched rfft2/irfft2 (bit-identical to the per-rotation calls, 48/48 arrays equal, _gpu_port\test_getK.py)
    rots_ = rows[i]["D"]["rots"]; kers_ = torch.zeros((len(rots_), W, H), device=dev); krs_ = torch.zeros_like(kers_)
    for k_, r_ in enumerate(rots_):
        m_ = torch.from_numpy(r_["mask"]).to(dev); rm_ = torch.from_numpy(r_["rmask"]).to(dev).float(); kers_[k_, :m_.shape[0], :m_.shape[1]] = m_.float(); krs_[k_, :m_.shape[0], :m_.shape[1]] = rm_ / rm_.sum()
    F_ = torch.conj(torch.fft.rfft2(kers_)); RF_ = torch.conj(torch.fft.rfft2(krs_))
    for k, r in enumerate(rows[i]["D"]["rots"]):
        mk = torch.from_numpy(r["mask"]).to(dev); sw, sh = mk.shape
        ox = torch.arange(W, device=dev); oy = torch.arange(H, device=dev); cx = (ox + sw / 2) * R - PX / 2; cy = (oy + sh / 2) * R - PY / 2
        inb = ((ox >= e) & (ox <= W - sw - e))[:, None] & ((oy >= e) & (oy <= H - sh - e))[None, :]
        nz = torch.nonzero(mk); rmk = torch.from_numpy(r["rmask"]).to(dev).float(); Cm = F.pad(rmk.cumsum(0).cumsum(1), (1, 0, 1, 0)); arr_ = torch.nonzero(rmk).float()
        K = dict(f=F_[k], inb=inb, nz=nz, sw=sw, sh=sh, cx=cx, cy=cy, C=Cm, rt=Cm[-1, -1], rcx=float(arr_[:, 0].mean() + 0.5), rcy=float(arr_[:, 1].mean() + 0.5))
        K["SH"] = zone_share(K); K["rf"] = RF_[k]; K["rnz"] = torch.nonzero(rmk); K["hs"] = self_c(r_["rmask"]); out.append(K)
    Kc[i] = out
    while len(Kc) > KCAP: Kc.popitem(last=False)
    return out
def use(b): return max([1 - bedleft[b] / BED] + ([1 - prof[b].min() / ABUD] if ABUD else []) + ([1 - forceleft[b] / FBUD] if FBUD else []))      # binding ACTIVE allotment
def budget_ok(b, x): return bool((x["prof"] <= prof[b]).all() and x["pull"] <= forceleft[b] and x["halo"] <= bedleft[b])
def commit(rec):
    b, i = rec["b"], rec["i"]; x = rows[i]; nz = rec["nz"]; occ[b, rec["ox"] + nz[:, 0], rec["oy"] + nz[:, 1]] = 1.0; load[b] += rec["lv"]
    q_ = torch.zeros((W, H), device=dev, dtype=torch.float64); q_[rec["ox"] + rec["rnz"][:, 0], rec["oy"] + rec["rnz"][:, 1]] = rows[i]["pull"] / (rec["rnz"].shape[0] * (R * 1e-3) ** 2); _wq = film(q_); wf[b] += _wq; rec["wq"] = _wq.cpu(); del _wq      # 20261117 14:50: kept on the host (1.56 MB per placed piece), only needed again by uncommit
    prof[b] -= x["prof"]; forceleft[b] -= x["pull"]; bedleft[b] -= x["halo"]; span.setdefault(x["set"], set()).add(b); rec["seq"] = rec.get("seq", seq[0]); seq[0] += 1; placements[b].append(rec)
def uncommit(i):
    for b in range(NPOOL):
        for rec in placements[b]:
            if rec["i"] == i:
                placements[b].remove(rec); x = rows[i]; load[b] -= rec["lv"]; wf[b] -= rec["wq"].to(dev); prof[b] += x["prof"]; forceleft[b] += x["pull"]; bedleft[b] += x["halo"]
                occ[b] = 0
                for r2 in placements[b]: occ[b, r2["ox"] + r2["nz"][:, 0], r2["oy"] + r2["nz"][:, 1]] = 1.0
                if not any(rows[r2["i"]]["set"] == x["set"] for r2 in placements[b]): span[x["set"]].discard(b)
                return rec
def find(i, allowed):
    """evaluate every legal origin x rotation on ALL allowed beds at once; choose bed (model affinity, then most room left in the binding allotment),
    then the position with the lowest cost: exact zone-load math (outer spread + 3 x centre load, as fc2's final score) + ring target + neighbour hug."""
    x = rows[i]; S = span.get(x["set"], set()); Bs = [b for b in allowed if budget_ok(b, x)]
    if not Bs: return None
    Ks = get_K(i); pull = x["pull"]; ut = 1.0 - rank[i]; nb = len(Bs); lb = load[Bs]
    blocked = F.max_pool2d(occ[Bs][:, None], 2 * g + 1, 1, g)[:, 0] + (0.0 if (CENTRE_SMALL and tier[i] == 2) else circle[None]); bf = torch.fft.rfft2(blocked)
    wfft = torch.fft.rfft2(wf[Bs])
    best = [None] * nb; nlegal = [0] * nb
    for k, K in enumerate(Ks):
        legal = (torch.fft.irfft2(bf * K["f"][None], s=(W, H)) < 0.5) & K["inb"][None]
        if not bool(legal.any()): continue
        cost = pull * torch.fft.irfft2(wfft * K["rf"][None], s=(W, H)) + 0.5 * pull * pull * K["hs"][None]      # film energy this piece adds at each spot
        cost = torch.where(legal, cost, torch.full_like(cost, float("inf"))); mn, am = cost.reshape(nb, -1).min(1); cnt = legal.reshape(nb, -1).sum(1)
        for j in range(nb):
            nlegal[j] += int(cnt[j])
            if float(mn[j]) < float("inf") and (best[j] is None or float(mn[j]) < best[j][0]): best[j] = (float(mn[j]), int(am[j]) // H, int(am[j]) % H, k)
    cands = [(Bs[j], best[j], nlegal[j]) for j in range(nb) if best[j] is not None]
    if not cands: return None
    def key(c): return (0 if (not S or c[0] in S) else 1, min((abs(c[0] - s) for s in S), default=0) if (S and c[0] not in S) else 0, c[1][0], c[0])
    b, (cs, ox, oy, k), nlg = min(cands, key=key); K = Ks[k]
    return dict(b=b, i=i, ox=ox, oy=oy, k=k, nz=K["nz"].cpu(), rnz=K["rnz"].cpu(), lv=(K["SH"][:, ox, oy] * pull).clone(), cx=((ox + K["rcx"]) * R - PX / 2), cy=((oy + K["rcy"]) * R - PY / 2), nlegal=nlg)
# ---------------- (b) PLACE in rounds over all beds (20261116: by UNIT = a figure's body + its own base, placed together) ----------------
def drop_model(s):
    for i in [r["i"] for b in range(NPOOL) for r in placements[b] if rows[r["i"]]["set"] == s]: uncommit(i)
def place_unit(idx, beds):
    """all pieces of one unit on ONE bed of `beds` if any single bed holds them; else the fewest beds (<= MAXSPLIT). Returns recs or None (nothing committed)."""
    idx = sorted(idx, key=lambda i: -sz(rows[i])); r0 = find(idx[0], beds)
    if r0 is None: return None
    pref = [r0["b"]] + sorted([b for b in beds if b != r0["b"]], key=use)
    got, nb = GEO.place_unit_atomic(idx, pref, find, commit, uncommit, max_span=MAXSPLIT)
    if got is None: return None
    for i, rec in got: minlegal[i] = rec["nlegal"]
    return got
failed = []; tP = time.time(); minlegal = {}; unit_split = []
for rname, units in rounds:
    t_r = time.time(); done = 0
    for u in units:
        st_ = UNITS[u][0]; s = rows[st_]["set"]
        if s in failed: continue
        got = place_unit(UNITS[u], list(range(NPOOL)))
        if got is None: failed.append(s); drop_model(s); continue
        done += len(got)
        if len({rec["b"] for _, rec in got}) > 1: unit_split.append(u)
    print(f"round {rname}: {done} pieces of {sum(len(UNITS[u]) for u in units)} placed, {sum(len(p) for p in placements)} on beds, {len(failed)} models failed so far  [{time.time() - t_r:.1f}s]", flush=True)
for s in sorted(list(failed), key=lambda s: sum(x["halo"] for x in M[s])):        # deterministic top-up: freed room may now hold a whole failed model
    us = [u for u in UNITS if rows[UNITS[u][0]]["set"] == s]; ok = True; got_all = []
    for u in us:
        got = place_unit(UNITS[u], list(range(NPOOL)))
        if got is None: ok = False; break
        got_all += [i for i, _ in got]
    if ok: failed.remove(s); print(f"top-up: {s.split(' (')[0]} placed whole", flush=True)
    else: [uncommit(i) for i in got_all]
stuck = set(); moved_pcs = 0
while True:                                                        # no sparse plates: a bed under CARRY is emptied onto the others if everything legally fits
    cand = [b for b in range(NPOOL) if placements[b] and use(b) < CARRY and b not in stuck]
    if not cand: break
    b = min(cand, key=use); saved = list(placements[b]); moves = []; ok = True; others = [m for m in range(NPOOL) if m != b and placements[m]]
    byu = {}
    for rec0 in saved: byu.setdefault(GEO.unit_key(rows[rec0["i"]]), []).append(rec0["i"])
    for u, idx in sorted(byu.items(), key=lambda kv: -max(sz(rows[i]) for i in kv[1])):
        for i in idx: uncommit(i)
        got = place_unit(idx, others) if others else None
        if got is None: ok = False; break
        moves += [i for i, _ in got]
    if ok: moved_pcs += len(saved)
    else:
        [uncommit(i) for i in moves]
        for rec0 in saved:
            if not any(r["i"] == rec0["i"] for r in placements[b]): rec0.pop("seq", None); commit(rec0)
        stuck.add(b)
print("figures (body+base) split over more than one plate: " + (", ".join(f"{u[1]}" for u in unit_split) or "none") + " (only where no single plate could hold the figure)", flush=True)
T_PLACE = time.time() - tP
if SYNTH:
    for b in range(NPOOL):
        for rec in placements[b]: print(f"SYNTH bed {b + 1}: {rows[rec['i']]['set']} unit {rows[rec['i']]['unit'][1]} {rows[rec['i']]['spec'][:3]} at x {rec['cx']:+.1f} y {rec['cy']:+.1f} mm rot {rec['k']} load {rows[rec['i']]['pull']:.3g} N", flush=True)
    print(f"SYNTH done: {sum(len(p) for p in placements)} placed, {len(failed)} failed, place {T_PLACE:.2f}s, film peak per bed um: " + ", ".join(f"{float(wf[b].max()) * 1e6:.2f}" for b in range(NPOOL) if placements[b]), flush=True); sys.exit(0)
print(f"(b) PLACE: {sum(len(p) for p in placements)} pieces on {sum(1 for p in placements if p)} of {NPOOL} beds, {moved_pcs} pieces moved to clear sparse beds, {len(failed)} models carried, {T_PLACE:.1f}s", flush=True)
# ---------------- write every bed that is >= CARRY: same files as fc2 ----------------
class _NullFile:
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def write(self, b): return len(b)
    def seek(self, *a): return 0
class _Sha:      # the hash file_identity already took in the load; read the file only when there is none
    def __init__(self, x): self.x = x
    def hexdigest(self):
        if self.x.get("sha256"): return self.x["sha256"]
        h = hashlib.sha256()
        with open(self.x["path"], "rb") as fh:
            for blk in iter(lambda: fh.read(2 ** 20), b""): h.update(blk)
        return h.hexdigest()
NOEXP_ = os.environ.get("GB_NOEXPORT") == "1"      # 20261117 15:26 loop cycle 2
def _xform(raw, k):
    if raw is None: return None
    t = raw.copy(); t["v"] = rot(raw["v"], k); t["n"] = rot(raw["n"], k); return t
def _shift_write(t, dx, dy, fo):
    if t is None: return None
    v = t["v"]; v[:, :, 0] += np.float32(dx); v[:, :, 1] += np.float32(dy); t["v"] = v; fo.write(t.tobytes()); return v
def _ntri(t, x): return len(t) if t is not None else int(x["ntri"] or 0)
def _zmin(v, zraw):      # export off: the same z the load would give (file z min, dropped to -0.005 when it was off the plate)
    if v is not None: return float(v[:, :, 2].min())
    return float(np.float32(zraw) - np.float32(zraw + 0.005)) if abs(zraw + 0.005) > 0.02 else float(zraw)
def write_bed(b):
    recs = sorted(placements[b], key=lambda r: r["seq"]); idx = [r["i"] for r in recs]; Pc = [rows[i] for i in idx]; npc = len(Pc); tag = f"{TAG}_P{b + 1}"
    calibration = None
    if CALIBRATED:
        # These optional input-row measurements must come from piece geometry.
        connections = [x.get("min_load_carrying_connection_mm") for x in Pc]
        patches = [x.get("largest_solid_patch_mm2") for x in Pc]
        measured = [CAL.measurement(value) for value in connections if value is not None]
        connection = min(measured) if measured else None
        thin = connection is not None and connection < CAL.MIN_CONNECTION_MM
        calibration = CAL.placement_verdict(
            BOTTOM_S, raft_gap_mm=(g - np.sqrt(2.0)) * R if npc > 1 else None,
            connection_mm=connection if thin or len(measured) == npc else None,
            patch_mm2=max(map(CAL.measurement, patches)) if all(p is not None for p in patches) else None,
            gap_required=npc > 1,
        )
        if calibration.startswith("REFUSED"):
            raise ValueError(calibration)
    d = os.path.join(OUT, f"P{b + 1}"); os.makedirs(d, exist_ok=True)
    json.dump(dict(out=d, tag=tag, lift=LIFT, pieces=[dict(set=x["set"], file=x["file"], path=x["path"]) for x in Pc]), open(os.path.join(OUT, f"{tag}_job.json"), "w"), indent=1)
    hmax = max(x["h"] for x in Pc); fname = f"{tag}_{hmax:.0f}mm_{npc}pcs_RING.stl"; pcs = []; exact_src = []; off = 0; masks = []; dropped = []; prf = sum(x["prof"] for x in Pc); tot_pull = sum(x["pull"] for x in Pc)
    with (_NullFile() if os.environ.get("GB_NOEXPORT") == "1" else open(os.path.join(d, fname), "wb")) as fo:      # 20261117: GB_NOEXPORT=1 skips the 650 MB STL (default off); recipe, check file and PNG are still written
        from mkm_safety import checkpoint, Reservation
        fo.write(b"MKM PLATE GPU LAYOUT 20261114".ljust(80, b" ")); fo.write(struct.pack("<I", 0)); count_written = 0
        for n_, (x, rec) in enumerate(zip(Pc, recs)):
            with Reservation(1 if NOEXP_ else os.path.getsize(x['path'])*6):
                checkpoint(); raw, zraw_ = (None, float(x["zfile"] if x.get("zfile") is not None else load_raw(x["path"])[1])) if NOEXP_ else load_raw(x["path"]); count_written += _ntri(raw, x)
                ox, oy, k = rec["ox"], rec["oy"], rec["k"]; r = x["D"]["rots"][k]; t = _xform(raw, k)
                if abs(x["D"]["zmn"] + 0.005) > 0.02: dropped.append(f"{x['file']} was {x['D']['zmn']:+.2f} mm")
                dx = ox * R - PX / 2 - r["mn"][0] - (r["c0"][0] - r["pad"]) * R; dy = oy * R - PY / 2 - r["mn"][1] - (r["c0"][1] - r["pad"]) * R
                v = _shift_write(t, dx, dy, fo)
                # 20261117 14:43 EXACT RECIPE for the solver: the same affine this loop just applied (z drop from load_raw, quarter-turn k about Z, then the float32 shift), one 4x4 per source file, + the file's sha256.
                c_, s_ = [(1, 0), (0, 1), (-1, 0), (0, -1)][k]; dz_ = -float(np.float32(zraw_ + 0.005)) if abs(zraw_ + 0.005) > 0.02 else 0.0; h_ = _Sha(x)
                exact_src.append(dict(path=os.path.abspath(x["path"]), sha256=h_.hexdigest(), bytes=os.path.getsize(x["path"]), matrix=[[float(c_), float(-s_), 0.0, float(np.float32(dx))], [float(s_), float(c_), 0.0, float(np.float32(dy))], [0.0, 0.0, 1.0, dz_], [0.0, 0.0, 0.0, 1.0]]))
                if t is not None and x["ntri"] is not None and len(t) != x["ntri"]: raise SystemExit(f"ntri mismatch for {x['file']}: wrote {len(t)} triangles, finite count of the source is {x['ntri']}")      # ntri = finite triangles (GEO.finite_tri_count)
                sw, sh = r["mask"].shape; cx = (ox + sw / 2) * R - PX / 2; cy = (oy + sh / 2) * R - PY / 2
                col = 0 if cx > PX / 6 else (2 if cx < -PX / 6 else 1); row = 0 if cy > PY / 6 else (2 if cy < -PY / 6 else 1)
                full = np.zeros((W, H), bool); full[ox:ox + sw, oy:oy + sh] = r["mask"]; masks.append(full)
                pcs.append(dict(no=n_ + 1, set=x["set"], file=x["file"], path=x["path"], off=off, n=_ntri(t, x), ntri=_ntri(t, x), k=k, x=round(cx, 1), y=round(cy, 1), where=NAMES[row * 3 + col],
                                raft=round(x["D"]["raft"]), height=round(x["D"]["h"], 1), unit=str(x["unit"][1] if len(x["unit"]) > 1 else (x["unit"][0] if len(x["unit"]) else "")), pulled=False, zmin=_zmin(v, zraw_))); off += _ntri(t, x)
                del raw,t,v
        fo.seek(80); fo.write(struct.pack("<I", count_written))
    # schema-1 recipe (what physics\mkm_production.py and tests\grid_agreement_20261117.py read). Written BEFORE the checks below so a placed plate always leaves a recipe. Verdict stays UNSOLVED: only the solver may change it.
    # V2 field stages require square spacing. Use the profile nominal pitch for both axes.
    # The nominal LCD raster spans 298.08 x 165.60 mm at 46 um, separate from the packing bed.
    bed_px = [6480, 3600]
    pitch_mm = float(PROF.get("machine", "pixel_pitch_um")[0]) / 1000.0
    if not np.isfinite(pitch_mm) or pitch_mm <= 0:
        raise ValueError("recipe requires a finite positive machine.pixel_pitch_um")
    origin_mm = [-n * pitch_mm / 2 for n in bed_px]
    _rp = os.path.join(d, "recipe.json"); _rt = _rp + f".tmp{os.getpid()}"
    json.dump(dict(schema=1, name=tag, sources=exact_src, z_convention="mm; exact affine applied once; layer midpoint sampling",
                   geometry=dict(bed_px=bed_px, pixel_pitch_mm=[pitch_mm, pitch_mm], dx_mm=pitch_mm, layer_mm=float(PRM.get("film_contact_gap_um")) / 1000.0, win_mm=[float(PROF.get("machine", "film_window_mm")[0]["x"]), float(PROF.get("machine", "film_window_mm")[0]["y"])], origin_mm=origin_mm, layer_planes_mm=None, layer_tops_mm=None),
                   placement=f"MKM GPU planner mkm_gpupool10 run {TAG}, bed {b + 1}, {npc} pieces", verdict={"verdict": "UNSOLVED", "reasons": ["evaluation pending: placed by the planner, not yet solved"]}), open(_rt, "w"), indent=1); os.replace(_rt, _rp)
    gaps = []
    for a in range(npc):
        o = np.zeros((W, H), bool)
        for j2 in range(npc):
            if j2 != a: o |= masks[j2]
        gaps.append(float((ndimage.distance_transform_edt(~o) * R)[masks[a]].min()) if npc > 1 else 99.0)
    un = np.zeros((W, H), bool)
    for m_ in masks: un |= m_
    xs_, ys_ = np.nonzero(un); emp = float(np.sqrt((((xs_ + 0.5) * R - PX / 2) ** 2 + ((ys_ + 0.5) * R - PY / 2) ** 2).min()))
    fp = un.sum() * R * R; zn = load[b].cpu().numpy(); outer = zn[OUTER]; spread = float(outer.max() - outer.min())
    cop = float(np.hypot(sum(r["cx"] * x["pull"] for r, x in zip(recs, Pc)), sum(r["cy"] * x["pull"] for r, x in zip(recs, Pc))) / tot_pull)      # centre of pull weighted by the solver load of each piece
    note = (f"GPU layout search (direct pool placement, deterministic; Claude 11/14): {npc} pcs, {hmax:.1f} mm; peel force (core coarse bound, sum of pieces) {tot_pull:.0f} N at lift {LIFT} mm/min; "
            f"outer-zone spread {spread:.1f} N, centre zone {zn[4]:.1f} N, centre of pull {cop:.1f} mm off middle; smallest gap {min(gaps):.1f} mm; "
            f"real footprint {100 * fp / (PX * PY):.0f}% of bed; cured-area envelope peak {prf.max():.0f} mm2 at {10 + 0.5 * prf.argmax():.1f} mm")
    # ---- THE VERDICT: the physics solver's output for THIS plate (20261116). Nothing here can write PASS except PV.decide().
    if os.environ.get("GB_VERDICT", "defer" if calibration else "solve") == "defer":
        verdict = PV.unsolved_no_solver("solver not run for this plate (GB_VERDICT=defer); run mkm_suction core_forces / mkm_plate_verdict.evaluate_plate_stl on the file")
    else:
        verdict, _raw = PV.evaluate_plate_stl(os.path.join(d, fname), PROF, PRM, (PX, PY), device=DEVSTR)
        json.dump(dict(verdict=verdict, solver_plate=(_raw or {}).get("plate"), solver_missing=(_raw or {}).get("missing"), solver_assumptions=(_raw or {}).get("assumptions")), open(os.path.join(d, "SOLVER_VERDICT.json"), "w"), indent=1, default=str)
    json.dump([dict(tag=tag, file=fname, height=round(hmax, 1), fill=round(100 * fp / (PX * PY), 1), safe_d=CLEAR, hold=None, notes=[note], leftover=[], pieces=pcs, peel=dict(weak_supports=[]))], open(os.path.join(d, "RING_MANIFEST.json"), "w", encoding="utf-8"), indent=1)
    if calibration:
        note += " | " + CAL.ASSUMPTIONS
        verdict = dict(PV.unsolved_no_solver(CAL.ASSUMPTIONS), verdict=calibration)
    open(os.path.join(d, "PLATE_CHECK.txt"), "w", encoding="utf-8").write(PV.plate_check_text("MKM RING PACKER CHECK  (GPU layout search)", fname, npc, verdict, note,
                                                                                         [f"{p['set']} > {p['file']}" for p in pcs], sorted({p['set'] for p in pcs})))
    try:
        from PIL import Image, ImageDraw
        S = 4; im = Image.new("RGB", (W * S // 2, H * S // 2 + 22), (0, 0, 0)); dr = ImageDraw.Draw(im); cr = np.random.default_rng(7); px = np.zeros((H, W, 3), np.uint8)
        for m_ in masks: px[m_.T[::-1, ::-1]] = cr.integers(70, 240, 3)
        im.paste(Image.fromarray(px).resize((W * S // 2, H * S // 2), Image.NEAREST), (0, 22))
        for a in (1, 2): dr.line([(a * W * S // 6, 22), (a * W * S // 6, H * S // 2 + 22)], fill=(110, 110, 110)); dr.line([(0, 22 + a * H * S // 6), (W * S // 2, 22 + a * H * S // 6)], fill=(110, 110, 110))
        for z in range(9): dr.text(((z % 3) * W * S // 6 + 4, 24 + (z // 3) * H * S // 6), f"{zn[z]:.0f} N", fill=(255, 255, 0))
        for p in pcs: dr.text(((PX / 2 - p["x"]) / R * S / 2 - 4, 22 + (PY / 2 - p["y"]) / R * S / 2 - 5), str(p["no"]), fill=(255, 255, 255))
        rr = CLEAR / 2 / R * S / 2; dr.ellipse([W * S / 4 - rr, 22 + H * S / 4 - rr, W * S / 4 + rr, 22 + H * S / 4 + rr], outline=(255, 255, 255))
        dr.text((4, 4), f"{tag}  {npc} pcs  {hmax:.0f} mm  footprint {100 * fp / (PX * PY):.0f}%  F {tot_pull:.0f} N  {verdict["verdict"]}  spread {spread:.1f} N  min gap {min(gaps):.1f} mm  first layer {sum(1 for p in pcs if abs(p['zmin']) < 0.05)}/{npc}", fill=(255, 255, 255))
        im.save(os.path.join(d, f"{tag}.png"))
    except Exception as ex: print("picture failed:", ex)
    fl = sum(1 for p in pcs if abs(p["zmin"]) < 0.05); small = sum(1 for x in Pc if sz(x) < t1)
    return dict(verdict=verdict, npc=npc, small=small, emp=emp, centre=float(zn[4]), spread=spread, mingap=min(gaps), fl=fl, hmax=hmax, hmin=min(x["h"] for x in Pc), note=note, zn=zn, dropped=dropped, prf=prf, Pc=Pc, fp=fp)
# 20261117 GB_PLACE_ONLY=1 (ported from the 11/16 patched copy, same behaviour): print the per-bed placement result (same quantities as the P-lines) and exit BEFORE the write phase (no files, no PNG). Default off.
if os.environ.get("GB_PLACE_ONLY") == "1":
    _tot = 0.0; _wr = 0
    for b in range(NPOOL):
        if not placements[b]: continue
        Pc_ = [rows[r_["i"]] for r_ in placements[b]]; hh = int(np.ceil(max(x["h"] for x in Pc_) / LH_MM)) * SPL / 3600; wrote = use(b) >= CARRY; _tot += hh if wrote else 0; _wr += wrote
        print(f"PLACE_ONLY P{b + 1}: {len(Pc_)} pcs, {min(x['h'] for x in Pc_):.0f}-{max(x['h'] for x in Pc_):.0f} mm, ~{hh:.1f} h | area {('%.0f%%' % (100 * sum(x['prof'] for x in Pc_).max() / ABUD)) if ABUD else 'n/a'}, force {('%.0f%%' % (100 * sum(x['pull'] for x in Pc_) / FBUD)) if FBUD else 'n/a'} ({sum(x['pull'] for x in Pc_):.0f} N), bed {100 * sum(x['halo'] for x in Pc_) / BED:.0f}% | allot use {100 * use(b):.0f}% {'WRITTEN' if wrote else 'CARRIED'}", flush=True)
    print(f"PLACE_ONLY total: plates written {_wr}, printer-hours {_tot:.1f}, sets on >1 plate {sum(1 for v in span.values() if len(v) > 1)}, place {T_PLACE:.1f}s, total {time.time() - T0:.0f}s", flush=True); sys.exit(0)
carry = []; summary = []; hours = []; table = []; VERD = {}; BED_OK = []; BED_FAIL = []
for s in failed: carry += M[s]
for b in range(NPOOL):
    if not placements[b]: continue
    if use(b) < CARRY:
        print(f"P{b + 1}: carried over, only {100 * use(b):.0f}% of its allotment used ({len(placements[b])} pcs) - joins the next bundle", flush=True); carry += [rows[r["i"]] for r in placements[b]]; continue
    try: w = write_bed(b)      # 20261117 run 3: one failed bed must not stop the run (Mike 10:58). Any exception (incl. the admission MemoryError) is recorded for this bed only; the guard's decision is recorded verbatim, never softened.
    except (Exception, SystemExit) as ex_:
        import traceback as _tb; _fr = _tb.extract_tb(ex_.__traceback__); _g = any("mkm_safety" in f_.filename for f_ in _fr); _msg = f"{type(ex_).__name__}: {ex_}"
        BED_FAIL.append(dict(bed=b + 1, pieces=len(placements[b]), reason=("refused by guard: " if _g else "write failed: ") + _msg, where=(_fr[-1].filename.split("\\")[-1] + ":" + str(_fr[-1].lineno)) if _fr else ""))
        print(f"P{b + 1}: NOT WRITTEN - {BED_FAIL[-1]['reason']} [{BED_FAIL[-1]['where']}]", flush=True); carry += [rows[r["i"]] for r in placements[b]]; continue
    _d = os.path.join(OUT, f"P{b + 1}"); _need = ["RING_MANIFEST.json", "PLATE_CHECK.txt"] + ([] if os.environ.get("GB_NOEXPORT") == "1" else [w_ for w_ in os.listdir(_d) if w_.endswith("_RING.stl")][:1] or ["(stl)"])
    if not all(os.path.exists(os.path.join(_d, n_)) for n_ in _need):
        BED_FAIL.append(dict(bed=b + 1, pieces=len(placements[b]), reason="write returned but expected files are missing: " + ",".join(_need), where="")); print(f"P{b + 1}: NOT WRITTEN - files missing", flush=True); carry += [rows[r["i"]] for r in placements[b]]; continue
    BED_OK.append(b + 1); Pc = w["Pc"]; VERD[b] = w["verdict"]["verdict"]; hours.append(GEO.layer_hours(w["hmax"], LH_MM, SPL)); okk = GEO.gap_ok(w["mingap"], GAP, R) and w["fl"] == w["npc"]      # gap tolerance from the raster (g*R), not a literal
    line = (f"P{b + 1}: {w['npc']} pcs, {w['hmin']:.0f}-{w['hmax']:.0f} mm, ~{hours[-1]:.1f} h | allotment used: area {('%.0f%%' % (100 * w['prf'].max() / ABUD)) if ABUD else 'n/a (budget pending)'} (peak {w['prf'].max():.0f} mm2), force {('%.0f%%' % (100 * sum(x['pull'] for x in Pc) / FBUD)) if FBUD else 'n/a (budget pending)'} ({sum(x['pull'] for x in Pc):.0f} N), "
            f"bed {100 * sum(x['halo'] for x in Pc) / BED:.0f}% (real footprint {100 * w['fp'] / (PX * PY):.0f}% of bed) | placement check {'OK' if okk else 'FAILED'} | VERDICT {w['verdict']['verdict']} | " + "; ".join(sorted({x['set'].split(' (')[0].replace('Enemy - ', '').replace('Hero - ', '') for x in Pc})))
    tl = [w["note"], "zones N: " + " | ".join(f"{NAMES[z]} {w['zn'][z]:.1f}" for z in range(9)), f"first layer on plate (z min within 0.05 mm of 0): {w['fl']} of {w['npc']}", f"empty radius at centre {w['emp']:.1f} mm (supports included) | small pieces {w['small']} of {w['npc']} | smallest gap {w['mingap']:.1f} mm"]
    print(line, flush=True); [print("     " + l[:420], flush=True) for l in tl]; summary.append(line); summary += ["     " + l for l in tl]
    table.append((b + 1, w["npc"], w["small"], w["emp"], w["centre"], w["spread"], 100 * max([sum(x["halo"] for x in Pc) / BED] + ([w["prf"].max() / ABUD] if ABUD else []) + ([sum(x["pull"] for x in Pc) / FBUD] if FBUD else [])), min(minlegal.get(r["i"], 0) for r in placements[b])))
sp = "; ".join(f"{k.split(' (')[0]}: P{'/P'.join(str(j + 1) for j in sorted(v))}" for k, v in sorted(span.items()) if len(v) > 1 and any(use(j) >= CARRY for j in v))
print("models on more than one plate: " + (sp or "none")); summary.append("models on more than one plate: " + sp)
print("TABLE plate,pcs,small,empty_r_mm,centre_N,outer_spread_N,pct_allot,min_legal_spots: " + json.dumps([[round(v, 1) if isinstance(v, float) else v for v in t] for t in table]))
keep = ("set", "file", "path", "h", "raft", "area", "body_max", "unit", "ntri", "min_load_carrying_connection_mm", "largest_solid_patch_mm2")
json.dump([{k: x[k] for k in keep if k in x} for x in carry], open(os.path.join(OUT, "POOL_CARRY.json"), "w"), indent=0)
print(f"carried to the next bundle: {len(carry)} pieces from {len({x['set'] for x in carry})} models"); summary.append(f"carried to the next bundle: {len(carry)} pieces: " + "; ".join(sorted({x['set'] for x in carry})))
open(os.path.join(OUT, "BUNDLE_SUMMARY.txt"), "w", encoding="utf-8").write("\n".join(summary))
print("PLATE VERDICTS: " + ", ".join(f"P{b + 1} {VERD[b]}" for b in sorted(VERD)) if VERD else "PLATE VERDICTS: no plate written")
print("FILM peak deflection per bed (um, from the core membrane; lower = gentler peel): " + ", ".join(f"P{b + 1} {float(wf[b].max()) * 1e6:.1f}" for b in range(NPOOL) if placements[b]))
print(f"TIMES load masks {T_LOAD:.1f}s | place all pieces {T_PLACE:.1f}s | total {time.time() - T0:.0f}s")
json.dump(dict(written=BED_OK, failed=BED_FAIL), open(os.path.join(OUT, "BED_FAILURES.json"), "w"), indent=1)
print(f"WRITE SUMMARY: beds written {len(BED_OK)} {BED_OK} | beds failed {len(BED_FAIL)}" + (" (EVERY bed was refused/failed)" if BED_FAIL and not BED_OK else "") + "".join(f"\n   P{x['bed']}: {x['reason']} [{x['where']}]" for x in BED_FAIL), flush=True)
print(f"BUNDLEDONE {len(hours)} plates, {sum(hours):.0f} printer-hours total, {time.time() - T0:.0f}s")
