"""R14 (Mike 21:59 consensus answer): model accounting + the run's effective settings.  Read-only on every planner file.

model_accounting(pieces_json, planner_out) -> list of problems ([] = every model accounted for):
  1. every pieces.json source path is PLACED (in a P*/recipe.json source) or CARRIED (POOL_CARRY.json by path, LONE_CARRY.json by set+file);
  2. no source is both placed and carried;
  3. BED_FAILURES.json: failed == [];
  4. the number of P* folders that hold a recipe equals the number of plates the planner says it wrote (BED_FAILURES.written);
  5. no placed source that is not in pieces.json.
Match is by normalised absolute path (a path may repeat only if the pieces list repeats it).  Nothing is guessed: a missing file is a problem, not a pass.

effective_settings(planner_out) -> what the planner REALLY used, read from its own output (hard_checks.json footprint block), not from the config:
  fill_min_pct per plate (R14 planner), edge margin need per plate (hard_checks), so a stale label ('0.5 mm', 'carry off') cannot hide.
"""
import glob, json, os
from collections import Counter


def _norm(p):
    import mkm_archive_io as AIO
    return AIO.norm(p)


def _jload(p, default=None):
    try:
        return json.load(open(p, encoding="utf-8-sig"))
    except Exception:
        return default


def pieces_file_for(planner_out):
    """the pieces.json a plan was made from, via PLAN_RESULT.json up to two folders above the planner output; None if it was never recorded."""
    d = os.path.abspath(planner_out)
    for _ in range(3):
        rp = os.path.join(d, "PLAN_RESULT.json")
        if os.path.isfile(rp):
            pj = (_jload(rp, {}) or {}).get("pieces")
            if pj and os.path.isfile(pj):
                return pj
        d = os.path.dirname(d)
    return None


def model_accounting(pieces_json, planner_out):
    problems = []
    pieces = _jload(pieces_json)
    if not isinstance(pieces, list):
        return ["pieces.json unreadable or not a list: %s" % pieces_json]
    planner_out = os.path.abspath(planner_out)
    placed, plate_dirs = Counter(), []      # G1: Counter, not set - a path pieces.json lists k times must be placed/carried k times in total
    for rp in sorted(glob.glob(os.path.join(planner_out, "P*", "recipe.json"))):
        plate_dirs.append(os.path.dirname(rp))
        for s in (_jload(rp, {}) or {}).get("sources") or []:
            if s.get("path"):
                placed[_norm(s["path"])] += 1
    carried_paths, carried_sf = Counter(), Counter()
    for r in _jload(os.path.join(planner_out, "POOL_CARRY.json"), []) or []:
        if r.get("path"):
            carried_paths[_norm(r["path"])] += 1
    for r in _jload(os.path.join(planner_out, "LONE_CARRY.json"), []) or []:
        carried_sf[(str(r.get("set")), str(r.get("file")))] += 1
    want, sf_of = Counter(), {}
    for r in pieces:
        want[_norm(r["path"])] += 1; sf_of.setdefault(_norm(r["path"]), (str(r.get("set")), str(r.get("file"))))
    known = set(want)
    for pth, k in sorted(want.items()):
        sf = sf_of[pth]; npl = placed.get(pth, 0); nca = carried_paths.get(pth, 0) + carried_sf.get(sf, 0)
        if npl + nca < k:
            problems.append("model not accounted for (pieces.json lists it %d time(s), placed %d, carried %d): %s / %s" % (k, npl, nca, sf[0], sf[1]))
        if npl and nca:
            problems.append("model both placed and carried: %s / %s" % sf)
    for p in sorted(set(placed) - known):
        problems.append("placed source not in pieces.json: %s" % p)
    bf = _jload(os.path.join(planner_out, "BED_FAILURES.json"))
    if bf is None:
        problems.append("BED_FAILURES.json missing or unreadable: the planner's write result is unknown")
    else:
        if bf.get("failed"):
            problems.append("planner reports failed plates: %s" % "; ".join("P%s %s" % (x.get("bed"), str(x.get("reason"))[:80]) for x in bf["failed"]))
        if len(plate_dirs) != len(bf.get("written") or []):
            problems.append("%d plate folders with a recipe, but the planner says it wrote %d" % (len(plate_dirs), len(bf.get("written") or [])))
    return problems


def effective_settings(planner_out):
    out = dict(plates={}, carried=sorted({"%s (%s)" % (r.get("set"), r.get("file")) for r in (_jload(os.path.join(os.path.abspath(planner_out), "POOL_CARRY.json"), []) or [])}))
    for hc in sorted(glob.glob(os.path.join(os.path.abspath(planner_out), "P*", "hard_checks.json"))):
        r = _jload(hc, {}) or {}; fp = r.get("footprint") or {}
        out["plates"][os.path.basename(os.path.dirname(hc))] = dict(fp_pct=fp.get("fp_pct"), fill_min_pct=fp.get("fill_min_pct"), at_geometry_limit=fp.get("at_geometry_limit"), why=fp.get("why"),
                                                                    min_margin_mm=r.get("min_margin_mm"), edge_need_mm=r.get("edge_need_mm"))
    return out


def ssd_mode():
    """R19 (driver 0248 item 2, verifier R18 blocker 2): the SSD cache mode this process REALLY runs with, from the same switch the cache code reads (mkm_ssd_cache.switched_on), not from a label.
    -> dict(mode 'on'/'off', label 'SSD_CACHE=ON ...' / 'SSD_CACHE=OFF ...', root).  An invalid MKM_SSD_CACHE value is OFF (the cache code refuses it and the canonical store is used)."""
    raw = os.environ.get("MKM_SSD_CACHE", "0")
    try:
        import mkm_ssd_cache as SSD
        on = SSD.switched_on()
    except ValueError:
        return dict(mode="off", root=None, label="SSD_CACHE=OFF: MKM_SSD_CACHE=%r is not 0 or 1; the cache code refuses it, the canonical slice store is used" % raw)
    except Exception as ex_:
        return dict(mode="off", root=None, label="SSD_CACHE=OFF: the cache module did not load (%s); the canonical slice store is used" % str(ex_)[:80])
    if not on:
        return dict(mode="off", root=None, label="SSD_CACHE=OFF (MKM_SSD_CACHE=0, the default): the canonical RLE slice store only")
    root = str(SSD.active_root())
    return dict(mode="on", root=root, label="SSD_CACHE=ON: packed sidecar of the raw and repaired slice stores, cross-section cache for the planner, finished-plate checkpoints; cache folder %s" % root)


def case_mode():
    """R17 (verifier R16 blocker 4): the protective-case mode of THIS run, read from the same variable the case code reads (MKM_CASE_MODE, default 'on').  -> dict(mode, label)."""
    m = os.environ.get("MKM_CASE_MODE", "on").lower()
    if m == "off":
        return dict(mode="off", label="CASE=OFF: the models' own files are plated, no protective case and no case raft (A/B test run)")
    return dict(mode=m, label="CASE=%s: the protective case + raft is applied to every model (must be certified before layout)" % m.upper())
