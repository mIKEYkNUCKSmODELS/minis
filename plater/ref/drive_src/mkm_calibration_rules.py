"""Calibration screen for the M7 Max / ACF / Sunlu ABS-like build."""
import json
import math
from zipfile import ZipFile

MIN_BOTTOM_S = 50.0
MIN_RAFT_GAP_MM = 1.5
MIN_CONNECTION_MM = 0.60
TESTED_PATCH_MM2 = 1600.0
ASSUMPTIONS = "Assumes level plate and good film. Real peel force and upper capacity UNSOLVED."


def measurement(value):
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError("expected a numeric measurement")
    if not math.isfinite(value) or value < 0:
        raise ValueError("expected a finite nonnegative measurement")
    return value


def saved_bottom(path):
    """Use the saved layer schedule, which can disagree with the embedded profile."""
    with ZipFile(path) as archive:
        member = archive.getinfo("layers_controller.conf")
        if member.file_size > 8 * 1024**2:
            raise ValueError("layer schedule exceeds metadata read limit")
        schedule = json.loads(archive.read(member))
    layers = schedule["paras"]
    if not layers or len(layers) != schedule["count"]:
        raise ValueError("empty or inconsistent layer schedule")
    exposure = measurement(layers[0]["exposure_time"])
    count = 0
    for index, layer in enumerate(layers):
        if layer["layer_index"] != index:
            raise ValueError("invalid layer index in bottom schedule")
        if measurement(layer["exposure_time"]) != exposure:
            break
        count += 1
    return exposure, count


def exposure_refusal(exposure):
    try:
        exposure = measurement(exposure)
    except ValueError:
        return "REFUSED (bottom exposure unavailable or invalid)"
    if exposure < MIN_BOTTOM_S:
        return f"REFUSED (bottom exposure {exposure:g} s < {MIN_BOTTOM_S:g} s)"
    return None


def placement_verdict(exposure, *, raft_gap_mm=None, connection_mm=None,
                      patch_mm2=None, gap_required=True):
    refusal = exposure_refusal(exposure)
    if refusal:
        return refusal
    if raft_gap_mm is not None and measurement(raft_gap_mm) < MIN_RAFT_GAP_MM:
        return f"REFUSED (raft gap {raft_gap_mm:g} mm < {MIN_RAFT_GAP_MM:g} mm)"
    if connection_mm is not None and measurement(connection_mm) < MIN_CONNECTION_MM:
        return f"REFUSED (load-carrying connection {connection_mm:g} mm < {MIN_CONNECTION_MM:.2f} mm)"

    reasons = []
    if patch_mm2 is None:
        reasons.append("largest solid patch unverified")
    elif measurement(patch_mm2) > TESTED_PATCH_MM2:
        reasons.append(f"largest solid patch {patch_mm2:g} mm2 > {TESTED_PATCH_MM2:g} tested")
    if gap_required and raft_gap_mm is None:
        reasons.append("raft clearance unverified")
    if connection_mm is None:
        reasons.append("load-carrying connections unverified")
    if exposure != MIN_BOTTOM_S:
        reasons.append(f"{exposure:g} s bottom exposure outside the 50 s reference")
    if reasons:
        return "PLACED - beyond tested limits (" + ", ".join(reasons) + ")"
    return "PLACED - within tested limits"

