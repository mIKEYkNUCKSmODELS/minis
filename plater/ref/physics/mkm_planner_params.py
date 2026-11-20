"""mkm_planner_params.py - the ONE machine/resin parameter file the planner reads (20261116, planner fix).

Why: the planner (gpu_scan/mkm_gpupool10.py) used literals nobody approved (PULL 194 N, AREA 5100, KN = 0.012*1.025*LIFT/120, raft band 0.30 mm, gap tolerance 0.6 mm,
blob sigma 3 mm / stride 16, EFF, CARRY, SPL ...). Rules kept here (Mike: 'I want the math to determine whether it will print'):
  * every non-solver number is a NAMED parameter in profiles_filled/planner_params_<printer>_<film>_<resin>.json with unit, meaning, source and source class;
  * a bracket {low, central, high} is resolved by the same explicit policy as the core profiles ('central' | 'low' | 'high');
  * null = PENDING: get() raises ParamPending (a printability input) - nothing substitutes a number; allotments/optional reads use get(name, pending_ok=True) -> None (inactive);
  * a parameter whose role is 'search_heuristic' / 'allotment' / 'geometry' / 'report_only' never decides printability: only the solver verdict does.
Public: load(), PlannerParams.get/has/describe/validate, apply_profile_overlay()."""
import json, os, copy

HERE = os.path.dirname(os.path.abspath(__file__))
PARAM_DIR = os.path.join(HERE, "profiles_filled")
# 20261118 SPECS-ONLY SWITCH (Mike 00:25: "just the specs of the machinery and the resin"). Default OFF. env GB_SPECS_ONLY=1 neutralises every value that is not a machine/setting/resin/film datasheet spec.
# level 2 (Mike 00:31 "remove the 2 mm and 3 mm spacing"): gap and edge margin = one raster cell (raster_pitch_mm); the film-energy objective alone spreads the parts
# 20261118 DEFAULT = specs-only level 2 (packing_efficiency 1.0, min_plate_use_fraction 0, min_gap_mm and edge_margin_mm = one raster cell, calibration data labels off). Mike, 00:24: "plug those numbers in ... That's going to be the standard"; 00:25: "The only numbers that I gave a fuck about were ... the specs from the machine, the settings, the gap layer"; 00:31: "remove the 2 millimeter and 3 millimeter spacing ... The 2 and 3 were placeholders". Grok Build 01:13 recommends reverting this default until a force exists; the driver kept it on Mike's words. Legacy values (0.70 / 0.6 / 3.0 / 3.0 / calibration labels, still in the params file with their no-origin notes) only with GB_LEGACY_PLACEHOLDERS=1; GB_SPECS_ONLY=1 = 2 mm comparison level. Undo: UNDO_full_test_20261118.ps1.
SPECS_LEVEL = "" if os.environ.get("GB_LEGACY_PLACEHOLDERS") == "1" else (os.environ.get("GB_SPECS_ONLY") or "2")
SPECS_ONLY = SPECS_LEVEL in ("1", "2")
# neutral values: bed budget 100 % of the usable area (spec area less 3 mm edge), gap = Mike 11/5 2 mm, no min-plate-use carry rule (the 0.6 had no source)
SPECS_ONLY_VALUES = {"packing_efficiency": 1.0, "min_gap_mm": 2.0, "min_plate_use_fraction": 0.0}
# overlay fields that are estimates / other-brand values / print-derived: not filled in specs-only mode (the planner pull does not read them)
SPECS_ONLY_SKIP_OVERLAY = {"film.adhesion_energy_J_m2", "film.interface_strength_MPa", "film.stretch_closure", "resin.surface_tension_N_m", "resin.green_tensile_MPa"}
ROLES = {"printability_input", "geometry", "allotment", "search_heuristic", "report_only"}
CLASSES = {"published", "manufacturer spec", "independent measurement", "measured", "derived", "decided", "planner heuristic", "estimate", "pending"}


WORKING_SETTINGS_STAND_IN = "stand-in: proven at current working settings, Mike 11/17 16:31"
WORKING_SETTINGS_NOTE = (
    WORKING_SETTINGS_STAND_IN
    + " | applies to successful prints at the matching settings; numeric capacities require a solved reference"
)
WORKING_SETTINGS_ALLOTMENTS = {"plate_force_budget_N", "layer_area_budget_mm2"}


class ParamPending(RuntimeError):
    """A named planner/printability parameter has no value (null in the parameter file). No number is invented."""


class ParamError(ValueError):
    pass


def _is_bracket(v):
    return isinstance(v, dict) and ("low" in v or "high" in v)


class PlannerParams:
    def __init__(self, raw, choice="central", path=None):
        self.raw, self.choice, self.path = raw, choice, path
        self.p = raw.get("params", {})
        self.validate()

    def validate(self):
        bad = []
        for k, d in self.p.items():
            for need in ("role", "value", "unit", "meaning", "source", "source_class"):
                if need not in d:
                    bad.append("%s: missing '%s'" % (k, need))
            if d.get("role") not in ROLES:
                bad.append("%s: role %r not in %s" % (k, d.get("role"), sorted(ROLES)))
            if d.get("source_class") not in CLASSES:
                bad.append("%s: source_class %r not in the allowed list" % (k, d.get("source_class")))
            v = d.get("value")
            if v is None and d.get("source_class") != "pending":
                bad.append("%s: null value must have source_class 'pending'" % k)
            if v is not None and d.get("source_class") == "pending":
                bad.append("%s: has a value but source_class is 'pending'" % k)
            if _is_bracket(v):
                lo, hi, c = v.get("low"), v.get("high"), v.get("central")
                if lo is None or hi is None or lo > hi or (c is not None and not (lo <= c <= hi)):
                    bad.append("%s: bracket must satisfy low <= central <= high" % k)
            if d.get("source_class") in ("estimate", "decided", "planner heuristic") and len(str(d.get("source", ""))) < 20:
                bad.append("%s: source text too short to cite" % k)
        if bad:
            raise ParamError("planner parameter file %s invalid:\n  " % self.path + "\n  ".join(bad))

    def status(self, name):
        d = self.p.get(name)
        if d is None:
            raise KeyError("unknown planner parameter %r (not in %s)" % (name, self.path))
        return "pending" if d["value"] is None else ("bracket" if _is_bracket(d["value"]) else "value")

    def has(self, name):
        return name in self.p and self.p[name]["value"] is not None

    def get(self, name, pending_ok=False):
        d = self.p.get(name)
        if d is None:
            raise KeyError("unknown planner parameter %r (not in %s)" % (name, self.path))
        v = d["value"]
        if SPECS_LEVEL == "2" and name == "min_gap_mm":
            # 3 mm spacing is LOCKED by Mike (11/18) as the starting point; env GB_MIN_GAP_MM overrides for gap sweeps.
            return float(os.environ.get("GB_MIN_GAP_MM") or 3.0)
        if SPECS_LEVEL == "2" and name == "edge_margin_mm":
            # edge margin = one raster cell (0.5 mm): UNCONFIRMED with Mike, deliberately unchanged.
            return float(os.environ["GB_EDGE_MM"]) if os.environ.get("GB_EDGE_MM") else self.get("raster_pitch_mm")      # 20261120: GB_EDGE_MM lets the auto driver keep the bed-edge clearance equal to the gap it finds (unset = unchanged); undo C:\MKM_TMP\plate_heat_20261120\auto\UNDO_edge_env.cmd
        if SPECS_ONLY and name in SPECS_ONLY_VALUES:
            return SPECS_ONLY_VALUES[name]
        if v is None:
            if pending_ok:
                return None
            raise ParamPending("planner parameter %s is PENDING (%s)" % (name, d["source"]))
        if _is_bracket(v):
            if self.choice == "low":
                return v["low"]
            if self.choice == "high":
                return v["high"]
            return v["central"] if v.get("central") is not None else 0.5 * (v["low"] + v["high"])
        return v

    def bracket(self, name):
        v = self.p[name]["value"]
        return (v["low"], v["high"]) if _is_bracket(v) else None

    def describe(self):
        """one line per parameter: the replacement for the old 'HAND-SET still in this planner' print."""
        out = []
        for k, d in self.p.items():
            st = self.status(k)
            val = "PENDING" if st == "pending" else ("%s [%s..%s]" % (self.get(k), *self.bracket(k)) if st == "bracket" else str(d["value"]))
            if k in WORKING_SETTINGS_ALLOTMENTS and st == "pending":
                val += " (numeric allotment inactive)"
                out.append("%s | %s | %s" % (k, val, WORKING_SETTINGS_NOTE))
                continue
            out.append("%-30s %-10s %-18s %s %s  <- %s" % (k, d["role"], d["source_class"], val, d["unit"], d["source"][:110]))
        return out

    def estimated(self):
        """names of non-pending parameters that are estimates (carried into every verdict so an estimate can never pass as a measurement)."""
        return [k for k, d in self.p.items() if d["source_class"] == "estimate" and d["value"] is not None]


def default_path(printer="anycubic_m7max", film="film_acf", resin="resin_sunlu_abslike", directory=PARAM_DIR):
    short = {"anycubic_m7max": "m7max", "film_acf": "acf", "resin_sunlu_abslike": "sunlu"}
    return os.path.join(directory, "planner_params_%s_%s_%s.json" % (short.get(printer, printer), short.get(film, film).replace("film_", ""), short.get(resin, resin)))


def load(path=None, choice="central", **kw):
    path = path or os.environ.get("GB_PARAMS") or default_path(**kw)
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)
    return PlannerParams(raw, choice=choice, path=path)


def apply_profile_overlay(profiles, params):
    """Return (Profiles2, [fields filled]). 'profile_overlay': a field the profile has PENDING is filled from the parameter file (bracket -> source_class 'bracket (low, high)', text says ESTIMATE);
    a field the profile already has is never overwritten. 'profile_overlay_replace': Mike's own decisions that supersede a profile value (heater 30 C -> resin.temperature_C and the viscosity at 30 C);
    each replaced field is reported as 'section.field (replaced)'. A categorical value (string) is carried as a value. Every filled field keeps a source string pointing at the parameter file."""
    import mkm_core_api as api
    raws = copy.deepcopy(profiles.raw); filled = []
    for target, pname in params.raw.get("profile_overlay", {}).items():
        sec, field = target.split(".", 1)
        if SPECS_ONLY and target in SPECS_ONLY_SKIP_OVERLAY:
            continue
        if profiles.status(sec, field) != "pending" or not params.has(pname):
            continue
        raws[sec][field] = _field_from_param(params, pname); filled.append(target)
        if target == 'film.contact_gap_um' and params.p[pname].get('tracks_layer_height') and isinstance(raws[sec][field]['value'], dict):
            _lh = profiles.layer_height_mm()[0]                       # Mike 11/15: contact gap = layer height, dynamic
            if _lh:
                raws[sec][field]['value']['central'] = float(_lh) * 1000.0
                raws[sec][field]['source'] += ' | central = layer height (Mike 11/15), dynamic'
    for target, pname in ({} if SPECS_ONLY else params.raw.get("profile_overlay_replace", {})).items():      # specs-only: keep the Sunlu TDS viscosity at 25 C (the 30 C value is an Arrhenius estimate with an unsourced Ea)
        sec, field = target.split(".", 1)
        if not params.has(pname) or str(raws[sec].get(field, {}).get("source", "")).startswith("planner_params:%s |" % pname):
            continue                                                  # already replaced (idempotent)
        raws[sec][field] = _field_from_param(params, pname); filled.append(target + " (replaced)")
    if not filled:
        return profiles, []
    p2 = api.Profiles(raws["machine"], raws["film"], raws["resin"], raws["settings"], profiles.names, profiles.choice, None)
    p2.overrides = dict(profiles.overrides)
    return p2, filled


def _field_from_param(params, pname):
    d = params.p[pname]; v = d["value"]
    return dict(value=({k: v[k] for k in ("low", "central", "high") if v.get(k) is not None} if _is_bracket(v) else v), unit=d["unit"],
                source="planner_params:%s | %s" % (pname, d["source"]), source_class=("bracket (low, high)" if _is_bracket(v) else d["source_class"]))


def alternatives(params, pname):
    """categorical alternatives of a parameter (e.g. film_stretch_closure) that the corner sweep must also run."""
    return list(params.p[pname].get("categorical_alternatives", []))


_BAND = [None]
def raft_band_mm():
    """the raft / first-layer contact band (mm) from the parameter file, for every tool that used the literal 0.3 (mkm_ringpack raft_area / peel profile, mkm_gpuport). Raises if the file is missing:
    there is no hidden default."""
    if _BAND[0] is None:
        _BAND[0] = load(choice=os.environ.get("GB_BRACKET", "central")).get("raft_band_mm")
    return _BAND[0]

