"""shared synthetic fixtures for the G1-G4 gates (no GPU, no STLs, no slicer): tiny planner-output folders."""
import json, os, sys


def setup(tree):
    v3 = os.path.join(os.path.abspath(tree), "v3")
    sys.path.insert(0, v3)
    os.environ.pop("MKM_FILL_MIN", None); os.environ.pop("MKM_ALLOW_UNACCOUNTED", None)
    os.environ["MKM_CASE_MODE"] = "off"
    return v3


def jd(p, o):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    json.dump(o, open(p, "w"))


def make_plan(root, recipe_paths, fp=70.0, fmin=68.0, edge=2.0, gap=4.0):
    out = os.path.join(root, "out")
    jd(os.path.join(out, "P1", "recipe.json"), dict(sources=[dict(path=p, unit="U%d" % i) for i, p in enumerate(recipe_paths)]))
    jd(os.path.join(out, "P1", "hard_checks.json"), dict(ok=True, reasons=[], edge_need_mm=edge, gap_need_mm=gap, min_margin_mm=2.5,
                                                         footprint=dict(fp_pct=fp, fill_min_pct=fmin, at_geometry_limit=False, why="x")))
    jd(os.path.join(out, "BED_FAILURES.json"), dict(written=[1], failed=[]))
    return out


def run_main(root, out):
    """run plate_all.main on a planner folder; ('refused', 3) or ('went-on', what): the plan got past the accounting/settings refusal toward slicing."""
    import plate_all
    try:
        rc = plate_all.main([out, os.path.join(root, "run")])
    except BaseException as ex:      # past the refusal: the first slicing import (torch) or a later step
        return "went-on", repr(ex)[:120]
    return ("refused", rc) if rc == 3 else ("went-on", rc)


def finish(name, failures):
    if failures:
        print("%s FAIL: %s" % (name, "; ".join(failures))); sys.exit(1)
    print("%s PASS" % name)
