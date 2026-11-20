import shutil, os, sys, ast
P = r"C:\MKM_TMP\plater_one_build_20261117\tree\physics\mkm_planner_params.py"; B = P + ".backup_20261120-0520_before_edge_env"
s = open(P, encoding="utf-8").read()
if "GB_EDGE_MM" in s: sys.exit("already patched")
a = '            return self.get("raster_pitch_mm")'
assert s.count(a) == 1
shutil.copy2(P, B)
s = s.replace(a, '            return float(os.environ["GB_EDGE_MM"]) if os.environ.get("GB_EDGE_MM") else self.get("raster_pitch_mm")      # 20261120: GB_EDGE_MM lets the auto driver keep the bed-edge clearance equal to the gap it finds (unset = unchanged); undo C:\\MKM_TMP\\plate_heat_20261120\\auto\\UNDO_edge_env.cmd', 1)
ast.parse(s); open(P, "w", encoding="utf-8", newline="").write(s)
open(r"C:\MKM_TMP\plate_heat_20261120\auto\UNDO_edge_env.cmd", "w").write('@echo off\r\ncopy /y "%s" "%s"\r\n' % (B, P)); print("patched")
