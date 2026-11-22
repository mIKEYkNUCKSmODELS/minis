"""G2: NOT CHECKED accounting is a hard refusal before slicing; MKM_ALLOW_UNACCOUNTED=1 is the only escape.  usage: python gate_g2_not_checked.py <tree>"""
import os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _fx
_fx.setup(sys.argv[1])
fails = []
d = tempfile.mkdtemp(); out = _fx.make_plan(d, [os.path.join(d, "a.stl"), os.path.join(d, "b.stl")])      # no PLAN_RESULT / pieces.json recorded -> accounting NOT CHECKED
st, info = _fx.run_main(d, out)
if st != "refused":
    fails.append("plan with no recorded pieces.json went on toward slicing (%s)" % (info,))
os.environ["MKM_ALLOW_UNACCOUNTED"] = "1"
d2 = tempfile.mkdtemp(); out2 = _fx.make_plan(d2, [os.path.join(d2, "a.stl"), os.path.join(d2, "b.stl")])
st2, info2 = _fx.run_main(d2, out2)
if st2 == "refused":
    fails.append("MKM_ALLOW_UNACCOUNTED=1 did not allow the run")
_fx.finish("G2", fails)
