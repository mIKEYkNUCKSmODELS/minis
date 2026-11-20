import shutil, os, sys, ast
P = r"C:\MKM_TMP\plater_one_build_20261117\tree\gpu_scan\mkm_gpupool10.py"; B = P + ".backup_20261120-0245_before_auto_count_gap"
s = open(P, encoding="utf-8").read()
if "MKM_AUTO" in s: sys.exit("already patched")
shutil.copy2(P, B)
a1 = 'if os.environ.get("MKM_FULLNESS_MM2"): ABUD = float(os.environ["MKM_FULLNESS_MM2"])'
a2 = '    if os.environ.get("MKM_FULLNESS_MM2"): r["prof_layers"] = a; r["prof"] = np.full(600, float(a[4:60].mean()))'
a3 = '_need = int(NPOOL) if NPOOL else max(1, int(np.ceil(sum(r["halo"] for r in rows) / BED))) if rows else 1'
for a in (a1, a2, a3): assert s.count(a) == 1, a
def after(s, a, ins):
    i = s.index("\n", s.index(a)) + 1; return s[:i] + ins + s[i:]
s = after(s, a1, 'if os.environ.get("MKM_AUTO") == "1": ABUD = float(os.environ.get("MKM_AUTO_LOW", "3364"))      # 20261120 auto plate count (gated; undo: C:\\MKM_TMP\\plate_heat_20261120\\auto\\UNDO_auto.cmd): allotment = fullest plate that has printed clean, at every height\n')
s = after(s, a2, '    if os.environ.get("MKM_AUTO") == "1":      # 20261120: each piece spends its own 28 mm running mean of cured area at EVERY height (window j = bins 4+j..59+j of the 0.5 mm profile); window 0 is the 1.8-30 mm measure, higher windows are scaled so the one allotment ABUD stands for MKM_AUTO_HIGH there\n        _c = np.concatenate([[0.0], np.cumsum(np.concatenate([a, np.zeros(60)]))]); _rm = (_c[60:660] - _c[4:604]) / 56.0; _rm[1:] *= ABUD / float(os.environ.get("MKM_AUTO_HIGH", "4832")); r["prof_layers"] = a; r["prof"] = _rm\n')
s = after(s, a3, 'if os.environ.get("MKM_AUTO") == "1" and not os.environ.get("GB_NPOOL") and rows and ABUD: _need = max(_need, int(np.ceil(float(sum(r["prof"] for r in rows).max()) / ABUD)))      # 20261120: the plate count comes from the cured-area allotment as well as the bed area\n')
ast.parse(s); open(P, "w", encoding="utf-8", newline="").write(s)
open(r"C:\MKM_TMP\plate_heat_20261120\auto\UNDO_auto.cmd", "w").write('@echo off\r\ncopy /y "%s" "%s"\r\n' % (B, P))
print("patched; backup", B)
