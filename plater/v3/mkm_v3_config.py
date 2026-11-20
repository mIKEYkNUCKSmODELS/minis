"""MKM plater v3 - shared paths and constants (laptop MKM). Values are Mike's constants; do not tune them."""
import os, subprocess, sys

UVTOOLS = os.environ.get("MKM_UVTOOLS", r"C:\Program Files\UVtools\UVtoolsCmd.exe")
PY = r"C:\Users\confi\AI\ComfyUI_windows_portable\python_embeded\python.exe"
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

def gate_cmd(label, ram_gb, vram_gb, argv):
    """Wrap argv so it waits its turn in the MKM gate as a heavy job."""
    return ["cmd", "/c", GATE, "run", "--name", label, "--ram-gb", str(ram_gb), "--vram-gb", str(vram_gb),
            "--class", "heavy", "--"] + list(argv)

def run_hidden(argv, timeout=None, **kw):
    """Run a command with no console window; returns CompletedProcess (text)."""
    return subprocess.run(argv, capture_output=True, text=True, creationflags=NO_WINDOW, timeout=timeout, **kw)
