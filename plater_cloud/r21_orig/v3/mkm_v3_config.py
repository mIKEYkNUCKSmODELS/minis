"""MKM plater v3 - shared paths and constants (laptop MKM). Values are Mike's constants; do not tune them."""
import os, subprocess, sys

UVTOOLS = os.environ.get("MKM_UVTOOLS", r"C:\Program Files\UVtools\UVtoolsCmd.exe")
PY = r"C:\Users\confi\AI\ComfyUI_windows_portable\python_embeded\python.exe"
REF_PM7M = r"F:\MKM_STAGE\UVTOOLS_SCOUT_20261120\copy.pm7m"   # a real M7 Max plate file (scout 11/20); lifts and bottom layers are copied from it
GATE = r"F:\MKM_STAGE\MKM_GATE\mkm_gate.cmd"
REF_TREE = os.environ.get("MKM_GPU_SCAN", r"C:\MKM_TMP\plater_one_build_20261117\tree\gpu_scan")
PHYSICS_TREE = os.path.join(os.path.dirname(REF_TREE), "physics")
for _p in (REF_TREE, PHYSICS_TREE):
    if _p not in sys.path and os.path.isdir(_p):
        sys.path.insert(0, _p)

RES = (6480, 3600)                 # printer pixels x, y
SCREEN_MM = (298.08, 165.6)        # screen size mm
PIXEL_MM = (SCREEN_MM[0] / RES[0], SCREEN_MM[1] / RES[1])   # 0.046 mm
LAYER_MM = 0.03                    # every layer 0..top
GAP_MM = 4.0                       # locked 11/20
BOTTOM_EXP_S = 50.0                # Mike's settings, constants
NORMAL_EXP_S = 2.45
MIRROR_X = True
# Printable area (Mike 11/21 14:16): the real machine definition (Photon Workshop pm7m header of the real M7 Max file copy.pm7m: DisplayWidth 297.5 x
# DisplayHeight 164, PixelWidthMicrons 46; planner machine profile anycubic_m7max.json bed_mm 297.5 x 164) is smaller than the LCD (6480 x 3600 square
# 46 um pixels = 298.08 x 165.6). 297.5/6480 and 164/3600 are not equal pitches, so 297.5 x 164 is a window INSIDE the LCD, not the screen: it is the
# printable area (centred on the screen). The LCD stays the pixel grid. Physical plate size itself is UNPROVEN by software.
PRINTABLE_MM = (297.5, 164.0)
FILL_MIN = 0.68                    # Mike 11/21 22:00: every printed plate has a real model footprint of at least 68% of the usable bed (a VETO; metric = the planner log line "real footprint X% of bed"); the planner reads MKM_FILL_MIN, this is its default
EDGE_MARGIN_MM = 2.0               # Mike 11/21 14:18: every model at least 2 mm inside the printable area on all four sides
NO_WINDOW = 0x08000000 if os.name == "nt" else 0

def obs_running():
    """True if OBS is running (a show). Heavy jobs must not start then."""
    if os.name != "nt":
        return False
    try:
        out = subprocess.run(["tasklist", "/FI", "IMAGENAME eq obs64.exe", "/NH"], capture_output=True, text=True,
                             creationflags=NO_WINDOW, timeout=20).stdout
        return "obs64.exe" in out.lower()
    except Exception:
        return False

def gate_cmd(label, ram_gb, vram_gb, argv, lane=None):
    """Wrap argv so it waits its turn in the MKM gate as a heavy job. lane: 'plater' pins the job to the plater lane (R11: the gate's name rule only knows names containing PLATE)."""
    return ["cmd", "/c", GATE, "run", "--name", label, "--ram-gb", str(ram_gb), "--vram-gb", str(vram_gb),
            "--class", "heavy"] + (["--lane", lane] if lane else []) + ["--"] + list(argv)

def run_hidden(argv, timeout=None, **kw):
    """Run a command with no console window; returns CompletedProcess (text)."""
    return subprocess.run(argv, capture_output=True, text=True, creationflags=NO_WINDOW, timeout=timeout, **kw)

# ---- Rubric C3/C5 constants (Mike's, never re-solved; each carries its source and status). Verifier R3 #2: they were absent from v3.
# The consensus (hub, 11/21 14:25) says these are for RANKING the layout's peel/suction load, never pass/fail lines. The rating records them in rating.json.
PHYSICS = dict(
    peel_stress_kPa=dict(value=12.0, source="rubric C3 (ACF film ceiling, AmeraLabs); a stand-in, unmeasured on Mike's vat", use="USED: peel_rank() = per-layer peel load (cured area x this stress), plate rank = its peak (mkm_rate_px -> rating.json peel_rank). Ranking only, never a pass/fail line."),
    fep_standin=dict(value=dict(thickness_um=50, peel_stress_kPa=13.0, E_MPa=480), source="rubric C3 'FEP specs stand in for ACF'; physics/INPUTS_RESEARCH_20261114.md (AmeraLabs 50 um FEP 13 kPa, Chemours E 480 MPa)", use="NOT USED: recorded only (no ranking consumes it)"),
    film_tension_N_per_m=dict(value=dict(standin_final_250_300_350Hz=(4889, 7041, 9583), conflicting_estimate=(983, 1343)), status="NOT USED by any ranking. UNRESOLVED: two estimates conflict (hub 'MIKE 1403 CONSENSUS', FILM STAND-IN FINAL 20261117); tap tone 112.1 Hz measured 11/18; film opening unmeasured",
                             source="rubric C3 'film stiffness comes from his vat tap video'", use="NOT USED: no ranking in the rating consumes it (C5 open); recorded only"),
    resin_mix=dict(value="80% Anycubic 16K Standard + 20% Anycubic Tough 2.0", viscosity_mPa_s_25C=(489.6, 534.1, 578.5), source="hub 'ANSWER CODEX - resin mix viscosity 80-20 - 20261118-1747' (estimate low/central/high, not measured; vat runs at 30 C, no temperature correction)", accepted_standin="Sunlu ABS-like (resin_sunlu_abslike)", use="ACCEPTED STAND-IN (Mike 11/21 20:10: 'The sunlu is fine. Thats not that a big deal.'): the planner loads resin_sunlu_abslike (Sunlu ABS-like) and that is the accepted resin profile; the 80/20 mix is kept here for reference only and nothing ranks by viscosity"),
    vat_mm=dict(value=dict(frame_short_end=18.81, frame_long_edge=21.94, vat_height=35.51, resin_depth=22.93, edge_to_screws=7.61), source="hub 'MIKE MEASURED 20261118-1710 - M7 Max vat dimensions'; inside opening of the frame still unmeasured", use="NOT USED: recorded only"),
)

def physics_record():
    """The C3/C5 constants as written into every rating.json (so a rating that lost them is visible)."""
    import copy
    return copy.deepcopy(PHYSICS)


def peel_rank(cured_mm2, phys=None):
    """C3 (R6, verifier R5 #3): the peel load that RANKS plates, from the peel stress in PHYSICS. cured_mm2 = cured area of every layer (mm^2).
    load_N[layer] = area_m2 x stress_Pa. The plate's rank value is the PEAK load. It never rejects a plate: the 12 kPa figure is an unmeasured stand-in."""
    phys = phys or PHYSICS
    kpa = float(phys["peel_stress_kPa"]["value"])
    if kpa <= 0:
        raise ValueError("peel_stress_kPa must be positive")
    load = [float(a) * 1e-6 * kpa * 1e3 for a in cured_mm2]
    if not load:
        return dict(peak_peel_load_N=0.0, peak_layer=None, stress_kPa_used=kpa, use="rank only")
    pk = max(range(len(load)), key=lambda i: load[i])
    return dict(peak_peel_load_N=round(load[pk], 4), peak_layer=int(pk), stress_kPa_used=kpa, use="rank only, never a pass/fail line (12 kPa is unmeasured)")

