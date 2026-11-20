"""MKM plater v3 - step (b): UVtools issue detection + repair on one plate's print file, and GPU parity.

Contract: plater/CONTRACT.md, "mkm_uv_check.py".  Nothing here prints or sends to a printer; the repaired file is
written to out_dir under a new name and the input is never touched.  One bad plate never raises: every failure is
returned in the result dict as status + message (+ per-step records).

    uv_check(pm7m, out_dir, uvtools=UVTOOLS, gate=False, timeout_s=1800, try_profiles=True) -> dict
    parity(pm7m, out_dir, gate=False, timeout_s=1800, parity_path=None) -> dict
    parse_issues(text) -> dict          # tolerant parser, pure function
    probe(out_dir, uvtools=UVTOOLS) -> dict   # writes out_dir/UVTOOLS_HELP.txt

CLI:  python mkm_uv_check.py <pm7m> <out_dir> [--parity] [--gate] [--probe] [--timeout S]

ASSUMPTIONS - verify on MKM
  1. UVtoolsCmd print-issues output format is NOT known.  parse_issues() is deliberately tolerant: JSON (whole text or
     the first {...}/[...] block), CSV with a type column, or text lines "<IssueType>: <count>", "<count> <IssueType>",
     and per-layer lines containing "Layer N" plus an issue-type word.  Unknown "<Name>: <int>" lines land in
     other_counts.  If a machine-readable option exists in `print-issues --help` (--json / --format json|csv / --csv)
     it is used for the scan; if that call fails fast or yields no counts, the plain call is run once instead.
     The full stdout is always kept (uv_log.txt and stdout_<step>.txt) so the parser can be fixed from real output.
  2. Flags that worked on MKM (REPORT_UVTOOLS_ROUTE.txt): `print-issues <f> -i -o -r -s` (111 s) and
     `run <f> OperationRepairLayers -o <out>` (69.5 s); print-issues on the repaired file took ~3+ min.
  3. Whether print-issues exits non-zero when it finds issues is unknown: a step counts as OK if rc == 0 OR counts parsed.
  4. The second repair profile `-p RepairIslands -p RepairResinTraps` is attempted only when `run --help` mentions
     -p / --profile, as `run <f> OperationRepairLayers -p RepairIslands -p RepairResinTraps -o <variant>`; the exact
     meaning of -p for that operation is unverified.  Its result is recorded under issues_repaired.json
     ["profile_variant"], never as the main repaired file.
  5. Per-call timeout (default 1800 s) kills the process handed to subprocess; under the gate wrapper (cmd /c) a
     grandchild UVtoolsCmd may outlive the timeout - check Task Manager after a TIMEOUT.
  6. Issue-type names are canonicalised to Island, ResinTrap, SuctionCup, Overhang, TouchingBounds, EmptyLayer,
     PrintHeight; other names are kept verbatim in other_counts.
  7. UVtools counts issue regions, the GPU scan reports per-type layer spans and mm2: parity therefore compares layers
     affected / first / last / layer overlap per type, not region counts.  GPU "resin_trap" and "suction_cup" are my
     reading of the reference definitions and may not equal UVtools' ResinTrap / SuctionCup.
  8. Parity runs the GPU scan on the RAW (pre-repair) file, to compare like with like with issues_raw.  It refuses to
     run on CPU (the reference falls back silently, which would take hours).  Under --gate the GPU scan is run in a
     worker subprocess with label UV_<plate>_GPU, ram 3 GB, vram 2 GB - the vram figure is a guess, not measured.
  9. PARITY.json goes to the parent of out_dir when out_dir is named P<digits> (layout uv/P{n}/ -> uv/PARITY.json),
     otherwise into out_dir; pass parity_path to override.
"""
import csv
import io
import json
import os
import re
import subprocess
import sys
import time
import traceback

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:          # needed under `python -I`, which drops the script directory from sys.path
    sys.path.insert(0, _HERE)

from mkm_v3_config import UVTOOLS, PY, run_hidden, gate_cmd, obs_running  # noqa: E402 (also puts REF_TREE on sys.path)

SCHEMA = 1
DEFAULT_TIMEOUT_S = 1800
PROBE_TIMEOUT_S = 120
HELP_TIMEOUT_S = 60
GATE_RAM_GB = 3
GATE_VRAM_GB = 0           # UVtoolsCmd is CPU only
GATE_VRAM_GB_GPU = 2       # GPU parity worker; unmeasured guess (ASSUMPTION 8)
PARITY_LABEL = "UNPROVEN until run on MKM"
RAW_ARGS = ["-i", "-o", "-r", "-s"]          # islands, overhangs, resin traps, suction cups (worked on MKM)
SECOND_PROFILE_ARGS = ["-p", "RepairIslands", "-p", "RepairResinTraps"]

# ---------------------------------------------------------------------------------------------- issue-type names
_CANON = {}
for _c, _names in {
    "Island": ["island"], "ResinTrap": ["resintrap"], "SuctionCup": ["suctioncup"], "Overhang": ["overhang"],
    "TouchingBounds": ["touchingbound", "touchingbounds"], "EmptyLayer": ["emptylayer"], "PrintHeight": ["printheight"],
}.items():
    for _n in _names:
        _CANON[_n] = _c
        _CANON[_n + "s"] = _c
_TYPE_WORD_RE = re.compile(r"\b(island|resin\s*trap|suction\s*cup|overhang|touching\s*bounds?|empty\s*layer|print\s*height)s?\b", re.I)


def canon_type(name):
    """Canonical issue-type name for a UVtools label, or None."""
    return _CANON.get(re.sub(r"[^a-z]", "", str(name).lower()))


# ---------------------------------------------------------------------------------------------- parser
_SUMMARY_A = re.compile(r"^\s*[-*•]?\s*([A-Za-z][A-Za-z _-]{2,40}?)\s*[:=]\s*(\d+)\s*(?:issues?)?\s*$")
_SUMMARY_B = re.compile(r"^\s*(\d+)\s+([A-Za-z][A-Za-z ]{2,40}?)\s*$")
_LAYER_RE = re.compile(r"\blayer\s*#?\s*(\d+)\b", re.I)
_JSON_TYPE_KEYS = ("type", "issuetype", "issue_type", "IssueType", "Type")
_JSON_LAYER_KEYS = ("layer", "layerindex", "layer_index", "LayerIndex", "Layer", "index")
_JSON_GROUP_KEYS = ("issues", "items", "results", "Issues", "Items", "Results")


def _new_acc():
    return {"summary": {}, "other": {}, "detail": {}, "warnings": []}


def _add_detail(acc, typ, layer, n=1):
    d = acc["detail"].setdefault(typ, {})
    d[layer] = d.get(layer, 0) + n


def _add_summary(acc, typ, n):
    if typ in acc["summary"]:
        acc["warnings"].append("type %s listed more than once; kept first (%d), ignored %d" % (typ, acc["summary"][typ], n))
    else:
        acc["summary"][typ] = n


def _walk_json(obj, acc):
    if isinstance(obj, list):
        for x in obj:
            _walk_json(x, acc)
        return
    if not isinstance(obj, dict):
        return
    tkey = next((k for k in _JSON_TYPE_KEYS if isinstance(obj.get(k), str) and canon_type(obj.get(k))), None)
    if tkey:                                     # one issue record (or a group of them)
        typ = canon_type(obj[tkey])
        group = next((obj[k] for k in _JSON_GROUP_KEYS if isinstance(obj.get(k), list)), None)
        if group is not None:
            _add_summary(acc, typ, len(group))
            for g in group:
                _walk_json_issue(acc, typ, g)
        else:
            _walk_json_issue(acc, typ, obj)
        return
    for k, v in obj.items():
        typ = canon_type(k)
        if typ and isinstance(v, bool):
            continue
        if typ and isinstance(v, (int, float)):
            _add_summary(acc, typ, int(v))
        elif typ and isinstance(v, list):
            _add_summary(acc, typ, len(v))
            for g in v:
                _walk_json_issue(acc, typ, g)
        else:
            _walk_json(v, acc)


def _walk_json_issue(acc, typ, rec):
    layer = None
    if isinstance(rec, dict):
        for k in _JSON_LAYER_KEYS:
            if isinstance(rec.get(k), int) and not isinstance(rec.get(k), bool):
                layer = rec[k]
                break
    elif isinstance(rec, int) and not isinstance(rec, bool):
        layer = rec
    if layer is not None:
        _add_detail(acc, typ, layer)
    else:
        acc["detail"].setdefault(typ, {})
        acc["detail"][typ]["_nolayer"] = acc["detail"][typ].get("_nolayer", 0) + 1


def _parse_json(text, acc):
    s = text.strip()
    cands = [s]
    for opener, closer in (("{", "}"), ("[", "]")):
        i, j = s.find(opener), s.rfind(closer)
        if i != -1 and j > i:
            cands.append(s[i:j + 1])
    for c in cands:
        try:
            obj = json.loads(c)
        except Exception:
            continue
        _walk_json(obj, acc)
        return True
    return False


def _parse_csv(text, acc):
    rows = list(csv.reader(io.StringIO(text)))
    rows = [r for r in rows if any(c.strip() for c in r)]
    if len(rows) < 2:
        return False
    head = [h.strip().lower() for h in rows[0]]
    tcol = next((i for i, h in enumerate(head) if "type" in h or h == "issue"), None)
    if tcol is None:
        return False
    lcol = next((i for i, h in enumerate(head) if "layer" in h), None)
    found = False
    for r in rows[1:]:
        if tcol >= len(r):
            continue
        typ = canon_type(r[tcol])
        if not typ:
            continue
        found = True
        layer = None
        if lcol is not None and lcol < len(r):
            m = re.search(r"\d+", r[lcol])
            layer = int(m.group()) if m else None
        if layer is None:
            acc["detail"].setdefault(typ, {})
            acc["detail"][typ]["_nolayer"] = acc["detail"][typ].get("_nolayer", 0) + 1
        else:
            _add_detail(acc, typ, layer)
    return found


def _parse_lines(text, acc):
    for line in text.splitlines():
        if not line.strip():
            continue
        m = _SUMMARY_A.match(line)
        if m:
            typ = canon_type(m.group(1))
            if typ:
                _add_summary(acc, typ, int(m.group(2)))
            else:
                acc["other"].setdefault(m.group(1).strip(), int(m.group(2)))
            continue
        m = _SUMMARY_B.match(line)
        if m and canon_type(m.group(2)):
            _add_summary(acc, canon_type(m.group(2)), int(m.group(1)))
            continue
        lm = _LAYER_RE.search(line)
        tm = _TYPE_WORD_RE.search(line)
        if lm and tm:
            _add_detail(acc, canon_type(tm.group(1)), int(lm.group(1)))


def parse_issues(text):
    """Tolerant parse of UVtoolsCmd print-issues output.

    Returns dict(format, counts, counts_source, other_counts, per_layer, layers, detail_counts, warnings, n_lines).
      counts        {IssueType: int}   from summary lines/keys when present, else derived from per-layer detail
      per_layer     {layer: {IssueType: n}}   only what the output printed
      layers        {IssueType: {layers_affected, first_layer, last_layer, layer_list}}
    Never raises; unparseable text yields empty counts and format 'unparsed'/'empty'."""
    text = text if isinstance(text, str) else ("" if text is None else str(text))
    acc = _new_acc()
    fmt = "empty"
    if text.strip():
        fmt = "unparsed"
        try:
            s = text.lstrip()
            if s[:1] in "{[" and _parse_json(text, acc):
                fmt = "json"
            elif _first_line_is_csv_header(text) and _parse_csv(text, acc):
                fmt = "csv"
            else:
                _parse_lines(text, acc)
                fmt = "text"
            if fmt == "json" and not acc["summary"] and not acc["detail"]:
                _parse_lines(text, acc)
                fmt = "text"
        except Exception as e:                      # parser must never take a plate down
            acc["warnings"].append("parser error: %r" % (e,))
    detail_counts = {t: sum(v for k, v in d.items()) for t, d in acc["detail"].items()}
    if acc["summary"]:
        counts, src = dict(acc["summary"]), "summary"
    elif detail_counts:
        counts, src = dict(detail_counts), "derived_from_detail"
    else:
        counts, src = {}, "none"
    if fmt == "text" and not counts:
        fmt = "unparsed" if text.strip() else "empty"
    per_layer, layers = {}, {}
    for typ, d in acc["detail"].items():
        ls = sorted(k for k in d if isinstance(k, int))
        for k in ls:
            per_layer.setdefault(k, {})[typ] = d[k]
        layers[typ] = dict(layers_affected=len(ls), first_layer=ls[0] if ls else None,
                           last_layer=ls[-1] if ls else None, layer_list=ls)
        if d.get("_nolayer"):
            layers[typ]["records_without_layer"] = d["_nolayer"]
    return dict(format=fmt, counts=counts, counts_source=src, other_counts=acc["other"], detail_counts=detail_counts,
                per_layer={str(k): v for k, v in sorted(per_layer.items())}, layers=layers,
                warnings=acc["warnings"], n_lines=len(text.splitlines()))


def _first_line_is_csv_header(text):
    for line in text.splitlines():
        if line.strip():
            return "," in line and bool(re.search(r"type|issue", line, re.I)) and not _SUMMARY_A.match(line)
    return False


def detect_format_flags(help_text):
    """argv extras that make print-issues emit JSON/CSV, from its --help text, or []. Prefers JSON."""
    h = help_text or ""
    if re.search(r"--json\b", h):
        return ["--json"]
    m = re.search(r"(--(?:output-)?format|--outformat)\b[^\n]*", h, re.I)
    if m:
        line = m.group(0)
        if re.search(r"json", line, re.I):
            return [m.group(1), "json"]
        if re.search(r"csv", line, re.I):
            return [m.group(1), "csv"]
    if re.search(r"--csv\b", h):
        return ["--csv"]
    return []


def profiles_supported(run_help_text):
    return bool(re.search(r"(^|\s)-p[,\s|]|--profile\b", run_help_text or "", re.M))


# ---------------------------------------------------------------------------------------------- run context
class _Ctx:
    def __init__(self, pm7m, out_dir, uvtools, gate, timeout_s):
        self.pm7m, self.out_dir, self.uvtools, self.gate, self.timeout_s = pm7m, out_dir, uvtools, gate, timeout_s
        self.plate = os.path.splitext(os.path.basename(pm7m))[0]
        self.log_path = os.path.join(out_dir, "uv_log.txt")
        self.steps = []

    def log(self, text):
        try:
            os.makedirs(self.out_dir, exist_ok=True)
            with open(self.log_path, "a", encoding="utf-8", errors="replace") as f:
                f.write(text if text.endswith("\n") else text + "\n")
        except Exception:
            pass

    def run(self, name, argv, heavy=True, timeout_s=None):
        """Run one UVtoolsCmd call. Returns (step_dict, stdout_text). Never raises."""
        timeout_s = timeout_s or self.timeout_s
        step = dict(name=name, argv=list(argv), heavy=heavy, gated=bool(self.gate and heavy), timeout_s=timeout_s,
                    status="FAILED", rc=None, seconds=None, message="", stdout_path=None)
        full = gate_cmd("UV_%s" % self.plate, GATE_RAM_GB, GATE_VRAM_GB, argv) if (self.gate and heavy) else list(argv)
        step["command"] = full
        out = err = ""
        t0 = time.time()
        self.log("=== [%s] %s start | timeout %ss | gated=%s\n    %s" % (name, time.strftime("%Y-%m-%d %H:%M:%S"), timeout_s, step["gated"], full))
        try:
            cp = run_hidden(full, timeout=timeout_s, encoding="utf-8", errors="replace")
            out, err, step["rc"] = cp.stdout or "", cp.stderr or "", cp.returncode
            step["status"] = "OK" if cp.returncode == 0 else "FAILED"
            if cp.returncode != 0:
                step["message"] = "exit code %s" % cp.returncode
        except subprocess.TimeoutExpired as e:
            out, err = _to_text(e.stdout), _to_text(e.stderr)
            step["status"] = "TIMEOUT"
            step["message"] = "timed out after %s s (process killed; with --gate a child may still run)" % timeout_s
        except Exception as e:                       # FileNotFoundError, OSError, ...
            step["status"] = "EXCEPTION"
            step["message"] = "%s: %s" % (type(e).__name__, e)
        step["seconds"] = round(time.time() - t0, 1)
        try:
            sp = os.path.join(self.out_dir, "stdout_%s.txt" % name)
            with open(sp, "w", encoding="utf-8", errors="replace") as f:
                f.write(out)
            step["stdout_path"] = sp
        except Exception:
            pass
        self.log("--- stdout ---\n%s\n--- stderr ---\n%s\n=== [%s] end status=%s rc=%s %.1fs %s" % (
            out, err, name, step["status"], step["rc"], step["seconds"], step["message"]))
        step["stdout_bytes"] = len(out.encode("utf-8", "replace"))
        step["stderr_tail"] = err[-2000:]
        self.steps.append(step)
        return step, out


def _to_text(x):
    if x is None:
        return ""
    return x.decode("utf-8", "replace") if isinstance(x, bytes) else str(x)


def _write_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1)


def _scan(ctx, name, pm7m, fmt_flags):
    """print-issues on one file. Returns (step, parsed, fmt_flags_used). Falls back to the plain call (once) if a
    machine-readable flag was tried and it failed fast or produced no counts."""
    base = [ctx.uvtools, "print-issues", pm7m] + RAW_ARGS
    if fmt_flags:
        step, out = ctx.run(name, base + fmt_flags)
        parsed = parse_issues(out)
        quick_fail = step["status"] in ("FAILED", "EXCEPTION") and (step["seconds"] or 0) < 30
        if step["status"] != "TIMEOUT" and (quick_fail or (step["status"] == "OK" and not parsed["counts"])):
            ctx.log("[%s] format flags %s gave no usable result; retrying plain" % (name, fmt_flags))
            step["message"] = (step["message"] + " | format flags %s unusable, plain call run next" % fmt_flags).strip(" |")
            step["status"] = "SUPERSEDED"
            step, out = ctx.run(name + "_plain", base)
            parsed = parse_issues(out)
            fmt_flags = []
    else:
        step, out = ctx.run(name, base)
        parsed = parse_issues(out)
    if step["status"] == "FAILED" and parsed["counts"]:
        step["status"] = "OK"
        step["message"] = (step["message"] + " (non-zero exit but counts parsed)").strip()
    if step["status"] == "OK" and not parsed["counts"]:
        step["message"] = (step["message"] + " no issue counts parsed (clean file, or format not recognised - see stdout)").strip()
    return step, parsed, fmt_flags


def _scan_record(ctx, label, pm7m, step, parsed, fmt_flags):
    return dict(schema=SCHEMA, label=label, pm7m=pm7m, uvtools=ctx.uvtools, status=step["status"], message=step["message"],
                command=step["argv"], rc=step["rc"], seconds=step["seconds"], output_format=parsed["format"],
                format_flags=fmt_flags, raw_stdout_path=step["stdout_path"], uv_log=ctx.log_path,
                counts=parsed["counts"], counts_source=parsed["counts_source"], other_counts=parsed["other_counts"],
                layers=parsed["layers"], per_layer=parsed["per_layer"], parse_warnings=parsed["warnings"],
                note="Raw UVtools counts are NOT a limit (merged rule); recorded only.")


def _result(ctx, status, message, **extra):
    r = dict(schema=SCHEMA, plate=ctx.plate if ctx else None, status=status, message=message, steps=ctx.steps if ctx else [])
    r.update(extra)
    return r


# ---------------------------------------------------------------------------------------------- uv_check
def uv_check(pm7m, out_dir, uvtools=UVTOOLS, gate=False, timeout_s=DEFAULT_TIMEOUT_S, try_profiles=True):
    """Detect (raw) -> repair -> detect (repaired) with UVtoolsCmd for one plate. Returns a result dict, never raises.

    status: OK | PARTIAL (raw scan ok, a later step failed) | FAILED_RAW_SCAN | WAITING_OBS | FAILED_INPUT |
            FAILED_NO_UVTOOLS | FAILED_EXCEPTION.  Per-step detail in result['steps'].
    Files in out_dir: issues_raw.json, issues_repaired.json, <plate>_repaired.pm7m, uv_log.txt, stdout_<step>.txt."""
    ctx = None
    t_all = time.time()
    try:
        pm7m = os.path.abspath(str(pm7m))
        out_dir = os.path.abspath(str(out_dir))
        ctx = _Ctx(pm7m, out_dir, uvtools, gate, timeout_s)
        if not os.path.isfile(pm7m):
            return _result(ctx, "FAILED_INPUT", "input file not found: %s" % pm7m, pm7m=pm7m, out_dir=out_dir)
        if not os.path.isfile(uvtools):
            return _result(ctx, "FAILED_NO_UVTOOLS", "UVtoolsCmd not found: %s" % uvtools, pm7m=pm7m, out_dir=out_dir)
        if obs_running():
            return _result(ctx, "WAITING_OBS", "OBS (obs64.exe) is running; heavy UVtools work not started. Re-run when OBS is closed.",
                           pm7m=pm7m, out_dir=out_dir)
        os.makedirs(out_dir, exist_ok=True)
        repaired = os.path.join(out_dir, "%s_repaired.pm7m" % ctx.plate)
        if os.path.realpath(repaired) == os.path.realpath(pm7m):
            return _result(ctx, "FAILED_INPUT", "refusing: repaired path would overwrite the input", pm7m=pm7m, out_dir=out_dir)
        res = _result(ctx, "OK", "", pm7m=pm7m, out_dir=out_dir, gate=bool(gate), timeout_s=timeout_s, uv_log=ctx.log_path,
                      input_bytes=os.path.getsize(pm7m), repaired_pm7m=None,
                      issues_raw_json=os.path.join(out_dir, "issues_raw.json"),
                      issues_repaired_json=os.path.join(out_dir, "issues_repaired.json"))
        ctx.log("uv_check %s -> %s | gate=%s timeout=%ss | %s" % (pm7m, out_dir, gate, timeout_s, time.strftime("%Y-%m-%d %H:%M:%S")))

        # cheap help probes: decide machine-readable output and second profile at runtime
        s_ph, ph = ctx.run("help_print_issues", [uvtools, "print-issues", "--help"], heavy=False, timeout_s=HELP_TIMEOUT_S)
        s_rh, rh = ctx.run("help_run", [uvtools, "run", "--help"], heavy=False, timeout_s=HELP_TIMEOUT_S)
        fmt_flags = detect_format_flags(ph)
        prof_ok = profiles_supported(rh)
        res["format_flags_detected"], res["profiles_supported"] = fmt_flags, prof_ok

        def waiting(step):
            return step["status"] == "WAITING_OBS"

        def guard(name):
            if obs_running():
                st = dict(name=name, status="WAITING_OBS", message="OBS started mid-run; step not started", rc=None, seconds=0.0)
                ctx.steps.append(st)
                return st
            return None

        # 1. raw scan
        g = guard("print_issues_raw")
        if g:
            res.update(status="WAITING_OBS", message=g["message"])
            return res
        s_raw, p_raw, used_flags = _scan(ctx, "print_issues_raw", pm7m, fmt_flags)
        _write_json(res["issues_raw_json"], _scan_record(ctx, "raw", pm7m, s_raw, p_raw, used_flags))
        res["raw"] = dict(status=s_raw["status"], counts=p_raw["counts"], seconds=s_raw["seconds"], output_format=p_raw["format"])
        raw_ok = s_raw["status"] == "OK"

        # 2. repair (default profile) -> out_dir/<plate>_repaired.pm7m, never over the input
        g = guard("repair")
        if g:
            res.update(status="WAITING_OBS", message=g["message"])
            return res
        if os.path.isfile(repaired):
            os.remove(repaired)                      # our own earlier output only; stops a stale file passing the size check
        s_rep, _ = ctx.run("repair", [uvtools, "run", pm7m, "OperationRepairLayers", "-o", repaired])
        rep_ok = _repair_ok(s_rep, repaired)
        if rep_ok:
            res["repaired_pm7m"] = repaired
            res["repaired_bytes"] = os.path.getsize(repaired)

        # 3. re-scan the repaired file
        p_after, s_after = None, None
        if rep_ok:
            g = guard("print_issues_repaired")
            if g:
                res.update(status="WAITING_OBS", message=g["message"])
            else:
                s_after, p_after, _f = _scan(ctx, "print_issues_repaired", repaired, used_flags)
        rec = dict(schema=SCHEMA, label="repaired", source_pm7m=pm7m, repaired_pm7m=res["repaired_pm7m"],
                   repair_status=s_rep["status"], repair_message=s_rep["message"], repair_seconds=s_rep["seconds"],
                   repair_command=s_rep["argv"], repair_stdout_path=s_rep["stdout_path"],
                   counts_before=p_raw["counts"])
        if p_after is not None:
            rec.update(_scan_record(ctx, "repaired", repaired, s_after, p_after, used_flags))
            rec.update(repair_status=s_rep["status"], repair_message=s_rep["message"], repair_seconds=s_rep["seconds"],
                       repair_command=s_rep["argv"], repair_stdout_path=s_rep["stdout_path"], source_pm7m=pm7m,
                       repaired_pm7m=repaired, counts_before=p_raw["counts"])
            rec["counts_after"] = p_after["counts"]
        else:
            rec.update(status="NOT_SCANNED", message="repair failed or re-scan did not run", counts_after=None, counts=None)
        res["repaired"] = dict(status=rec.get("status"), counts=rec.get("counts_after"), repair_status=s_rep["status"])

        # 4. optional second profile, only if `run --help` mentions -p/--profile
        if try_profiles and prof_ok and rep_ok and not any(waiting(s) for s in ctx.steps):
            variant = os.path.join(out_dir, "%s_repaired_profiles.pm7m" % ctx.plate)
            g = guard("repair_profiles")
            if not g:
                if os.path.isfile(variant):
                    os.remove(variant)
                s_v, _ = ctx.run("repair_profiles", [uvtools, "run", pm7m, "OperationRepairLayers"] + SECOND_PROFILE_ARGS + ["-o", variant])
                pv = dict(repair_status=s_v["status"], repair_message=s_v["message"], repaired_pm7m=None, counts=None)
                if _repair_ok(s_v, variant):
                    pv["repaired_pm7m"] = variant
                    if not guard("print_issues_repaired_profiles"):
                        s_vs, p_vs, _f = _scan(ctx, "print_issues_repaired_profiles", variant, used_flags)
                        pv.update(scan_status=s_vs["status"], counts=p_vs["counts"], raw_stdout_path=s_vs["stdout_path"])
                rec["profile_variant"] = pv
        elif try_profiles and not prof_ok:
            rec["profile_variant"] = dict(status="SKIPPED", message="`run --help` does not mention -p/--profile")
        _write_json(res["issues_repaired_json"], rec)

        # overall status
        if any(s["status"] == "WAITING_OBS" for s in ctx.steps):
            res["status"] = "WAITING_OBS"
            res["message"] = "OBS started during the run; later steps not run (see steps)"
        elif not raw_ok:
            res["status"] = "FAILED_RAW_SCAN"
            res["message"] = "raw print-issues %s: %s" % (s_raw["status"], s_raw["message"])
        else:
            bad = [s for s in ctx.steps if s["name"] in ("repair", "print_issues_repaired", "print_issues_repaired_plain",
                                                           "repair_profiles", "print_issues_repaired_profiles")
                   and s["status"] not in ("OK",)]
            if not rep_ok:
                bad = bad or [s_rep]
            if bad:
                res["status"] = "PARTIAL"
                res["message"] = "; ".join("%s %s %s" % (s["name"], s["status"], s.get("message", "")) for s in bad)
            else:
                res["message"] = "raw %s -> repaired %s" % (p_raw["counts"], p_after["counts"] if p_after else None)
        res["seconds"] = round(time.time() - t_all, 1)
        return res
    except Exception as e:                           # last line of defence: one plate never raises
        tb = traceback.format_exc()
        if ctx is not None:
            ctx.log("EXCEPTION in uv_check:\n" + tb)
        return _result(ctx, "FAILED_EXCEPTION", "%s: %s" % (type(e).__name__, e), pm7m=str(pm7m), out_dir=str(out_dir),
                       traceback=tb[-3000:])


def _repair_ok(step, path):
    if step["status"] != "OK":
        return False
    if not (os.path.isfile(path) and os.path.getsize(path) > 0):
        step["status"] = "FAILED"
        step["message"] = (step["message"] + " repair exited 0 but wrote no output file: %s" % path).strip()
        return False
    return True


# ---------------------------------------------------------------------------------------------- probe
def probe(out_dir, uvtools=UVTOOLS):
    """Run `print-issues --help`, `run --help`, `--version` and save everything to out_dir/UVTOOLS_HELP.txt,
    so the first MKM run tells us the real flags. Never raises."""
    out_dir = os.path.abspath(str(out_dir))
    res = dict(schema=SCHEMA, status="OK", message="", help_path=os.path.join(out_dir, "UVTOOLS_HELP.txt"), calls=[])
    try:
        os.makedirs(out_dir, exist_ok=True)
        chunks = ["UVtools probe %s\nexe: %s\n" % (time.strftime("%Y-%m-%d %H:%M:%S"), uvtools)]
        for argv in ([uvtools, "print-issues", "--help"], [uvtools, "run", "--help"], [uvtools, "--version"]):
            t0 = time.time()
            c = dict(argv=argv, rc=None, status="FAILED", seconds=None)
            out = err = ""
            try:
                cp = run_hidden(argv, timeout=PROBE_TIMEOUT_S, encoding="utf-8", errors="replace")
                out, err, c["rc"] = cp.stdout or "", cp.stderr or "", cp.returncode
                c["status"] = "OK" if cp.returncode == 0 else "FAILED"
            except subprocess.TimeoutExpired as e:
                out, err, c["status"] = _to_text(e.stdout), _to_text(e.stderr), "TIMEOUT"
            except Exception as e:
                c["status"] = "EXCEPTION"
                err = "%s: %s" % (type(e).__name__, e)
            c["seconds"] = round(time.time() - t0, 1)
            res["calls"].append(c)
            chunks.append("\n%s\n$ %s\n[status=%s rc=%s %.1fs]\n--- stdout ---\n%s\n--- stderr ---\n%s\n" % (
                "=" * 78, " ".join('"%s"' % a if " " in a else a for a in argv), c["status"], c["rc"], c["seconds"], out, err))
            if argv[1:] == ["print-issues", "--help"]:
                res["format_flags_detected"] = detect_format_flags(out + err)
            if argv[1:] == ["run", "--help"]:
                res["profiles_supported"] = profiles_supported(out + err)
        with open(res["help_path"], "w", encoding="utf-8", errors="replace") as f:
            f.write("".join(chunks))
        bad = [c for c in res["calls"] if c["status"] != "OK"]
        if bad:
            res["status"] = "PARTIAL" if len(bad) < len(res["calls"]) else "FAILED"
            res["message"] = "; ".join("%s: %s" % (" ".join(c["argv"][1:]), c["status"]) for c in bad)
    except Exception as e:
        res.update(status="FAILED_EXCEPTION", message="%s: %s" % (type(e).__name__, e), traceback=traceback.format_exc()[-3000:])
    return res


# ---------------------------------------------------------------------------------------------- parity
_GPU_TYPES = {"Island": "island", "ResinTrap": "resin_trap", "SuctionCup": "suction_cup", "Overhang": "overhang"}
_GPU_LAYER_ARRAYS = {"island": "island_mm2", "resin_trap": "trap_mm2", "suction_cup": "cup_mm2"}   # overhang has no per-layer array


def compare_issues(uv_raw, gpu):
    """Pure comparison of UVtools raw issues (issues_raw.json dict or None) against plate_issues_gpu() output.
    Compares layers affected / first / last / layer overlap per type; UVtools counts regions, the GPU scan does not."""
    out = {}
    uv_counts = (uv_raw or {}).get("counts") or {}
    uv_layers = (uv_raw or {}).get("layers") or {}
    per_layer = (gpu or {}).get("per_layer") or {}
    for uvt, gk in _GPU_TYPES.items():
        g = (gpu or {}).get(gk) or {}
        ul = uv_layers.get(uvt)
        row = dict(uvtools_count=uv_counts.get(uvt),
                   uvtools_layers_affected=ul["layers_affected"] if ul else None,
                   uvtools_first_layer=ul["first_layer"] if ul else None, uvtools_last_layer=ul["last_layer"] if ul else None,
                   gpu_layers_affected=g.get("layers_affected"), gpu_first_layer=g.get("first_layer"),
                   gpu_last_layer=g.get("last_layer"), gpu_total_mm2_layers=g.get("total_mm2_layers"))
        arr = per_layer.get(_GPU_LAYER_ARRAYS.get(gk, ""))
        if ul and arr is not None:
            us, gs = set(ul["layer_list"]), {i for i, v in enumerate(arr) if v and v > 0}
            both = len(us & gs)
            row["layer_overlap"] = dict(both=both, only_uvtools=len(us - gs), only_gpu=len(gs - us),
                                        jaccard=round(both / len(us | gs), 4) if (us | gs) else 1.0,
                                        sample_only_uvtools=sorted(us - gs)[:20], sample_only_gpu=sorted(gs - us)[:20])
        else:
            row["layer_overlap"] = None
        if row["uvtools_first_layer"] is not None and row["gpu_first_layer"] is not None:
            row["first_last_match"] = (row["uvtools_first_layer"], row["uvtools_last_layer"]) == (row["gpu_first_layer"], row["gpu_last_layer"])
        else:
            row["first_last_match"] = None
        if not ul and uvt in uv_counts:
            row["note"] = "UVtools per-layer detail not found in its output; only the count is known"
        out[uvt] = row
    return out


def _gpu_scan_inprocess(pm7m):
    import torch  # noqa: WPS433 - lazy: not needed for UVtools-only runs
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available; refusing the CPU fallback of plate_issues_gpu")
    import mkm_uvtools_gpu as G                      # found via REF_TREE on sys.path (mkm_v3_config)
    return G.plate_issues_gpu(pm7m)


def _parity_path(out_dir, parity_path):
    if parity_path:
        return parity_path
    if re.fullmatch(r"P\d+", os.path.basename(out_dir.rstrip("\\/"))):
        return os.path.join(os.path.dirname(out_dir.rstrip("\\/")), "PARITY.json")
    return os.path.join(out_dir, "PARITY.json")


def parity(pm7m, out_dir, gate=False, timeout_s=DEFAULT_TIMEOUT_S, parity_path=None):
    """GPU scan (mkm_uvtools_gpu.plate_issues_gpu) on the RAW (pre-repair) file, compared with UVtools' issues_raw.json.
    Writes out_dir/gpu_issues.json and PARITY.json. Never raises. UNPROVEN until run on MKM."""
    ctx = None
    try:
        pm7m, out_dir = os.path.abspath(str(pm7m)), os.path.abspath(str(out_dir))
        ctx = _Ctx(pm7m, out_dir, UVTOOLS, gate, timeout_s)
        res = dict(schema=SCHEMA, plate=ctx.plate, status="OK", message="", comparison_status=PARITY_LABEL, pm7m=pm7m,
                   scanned="RAW (pre-repair) file, same as issues_raw", gpu_issues_json=os.path.join(out_dir, "gpu_issues.json"),
                   parity_json=_parity_path(out_dir, parity_path), steps=ctx.steps)
        if not os.path.isfile(pm7m):
            res.update(status="FAILED_INPUT", message="input file not found: %s" % pm7m)
            return res
        if obs_running():
            res.update(status="WAITING_OBS", message="OBS (obs64.exe) is running; GPU parity not started")
            return res
        os.makedirs(out_dir, exist_ok=True)
        uv_raw = None
        try:
            with open(os.path.join(out_dir, "issues_raw.json"), encoding="utf-8") as f:
                uv_raw = json.load(f)
        except Exception as e:
            res["uvtools_raw_missing"] = "%s: %s (run uv_check first)" % (type(e).__name__, e)
        gpu = None
        t0 = time.time()
        try:
            if gate:
                py = PY if os.path.isfile(PY) else sys.executable
                argv = [py, "-I", "-u", os.path.abspath(__file__), "--gpu-worker", pm7m, res["gpu_issues_json"]]
                full = gate_cmd("UV_%s_GPU" % ctx.plate, GATE_RAM_GB, GATE_VRAM_GB_GPU, argv)
                try:
                    cp = run_hidden(full, timeout=timeout_s, encoding="utf-8", errors="replace")
                    ctx.log("gpu worker rc=%s\n%s\n%s" % (cp.returncode, cp.stdout, cp.stderr))
                    if cp.returncode != 0 or not os.path.isfile(res["gpu_issues_json"]):
                        raise RuntimeError("gpu worker rc=%s: %s" % (cp.returncode, (cp.stderr or cp.stdout or "")[-500:]))
                except subprocess.TimeoutExpired:
                    raise RuntimeError("gpu worker timed out after %s s" % timeout_s)
                with open(res["gpu_issues_json"], encoding="utf-8") as f:
                    gpu = json.load(f)
            else:
                gpu = _gpu_scan_inprocess(pm7m)
                _write_json(res["gpu_issues_json"], gpu)
        except Exception as e:
            res.update(status="FAILED_GPU_SCAN", message="%s: %s" % (type(e).__name__, e), traceback=traceback.format_exc()[-3000:])
        res["gpu_seconds"] = round(time.time() - t0, 1)
        if gpu is not None:
            res["gpu_summary"] = {k: v for k, v in gpu.items() if k != "per_layer"}
        res["by_type"] = compare_issues(uv_raw, gpu)
        res["uvtools_counts_raw"] = (uv_raw or {}).get("counts")
        res["caveat"] = ("UVtools counts issue regions, the GPU scan reports layers/area; compared on layers affected, first/last "
                         "layer and layer overlap. " + PARITY_LABEL + ".")
        try:
            _write_json(res["parity_json"], res)
        except Exception as e:
            res["message"] = (res["message"] + " | could not write PARITY.json: %s" % e).strip(" |")
        return res
    except Exception as e:
        tb = traceback.format_exc()
        if ctx is not None:
            ctx.log("EXCEPTION in parity:\n" + tb)
        return dict(schema=SCHEMA, status="FAILED_EXCEPTION", message="%s: %s" % (type(e).__name__, e),
                    comparison_status=PARITY_LABEL, traceback=tb[-3000:])


# ---------------------------------------------------------------------------------------------- CLI
def _main(argv):
    import argparse
    if len(argv) >= 4 and argv[0] == "--gpu-worker":          # internal: GPU scan inside the gate
        _write_json(argv[2], _gpu_scan_inprocess(argv[1]))
        return 0
    ap = argparse.ArgumentParser(description="MKM plater v3 step (b): UVtools detect + repair (+ GPU parity) on one print file")
    ap.add_argument("pm7m")
    ap.add_argument("out_dir")
    ap.add_argument("--parity", action="store_true", help="also run the GPU scan on the raw file and write PARITY.json")
    ap.add_argument("--gate", action="store_true", help="wrap UVtools calls in the MKM gate (label UV_<plate>)")
    ap.add_argument("--probe", action="store_true", help="only run --help/--version probes into out_dir/UVTOOLS_HELP.txt")
    ap.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_S, help="per-call timeout in seconds (default 1800)")
    a = ap.parse_args(argv)
    if a.probe:
        r = probe(a.out_dir)
    else:
        r = uv_check(a.pm7m, a.out_dir, gate=a.gate, timeout_s=a.timeout)
        if a.parity and r.get("status") not in ("WAITING_OBS", "FAILED_INPUT", "FAILED_NO_UVTOOLS"):
            r["parity"] = parity(a.pm7m, a.out_dir, gate=a.gate, timeout_s=a.timeout)
    print(json.dumps(r, indent=1, default=str))
    st = r.get("status")
    return 0 if st == "OK" else (2 if st == "WAITING_OBS" else 1)


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
