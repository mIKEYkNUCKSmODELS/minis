"""G3: edge distance, gap and fill minimum are enforced (refused before slicing), not only logged.  usage: python gate_g3_settings.py <tree>"""
import json, os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _fx
_fx.setup(sys.argv[1])
fails = []


def fixture(**kw):
    d = tempfile.mkdtemp(); a = os.path.join(d, "a.stl"); b = os.path.join(d, "b.stl"); pj = os.path.join(d, "pieces.json")
    json.dump([dict(set="S", file="a.stl", path=a), dict(set="T", file="b.stl", path=b)], open(pj, "w"))
    out = _fx.make_plan(d, [a, b], **kw)      # fully accounted plan: only the setting under test can be wrong
    json.dump(dict(pieces=pj), open(os.path.join(d, "PLAN_RESULT.json"), "w"))
    return d, out


for label, kw in (("edge 1.0 mm < 2.0", dict(edge=1.0)), ("gap 3.0 mm != 4.0", dict(gap=3.0)),
                  ("fill min 50% < 68%", dict(fmin=50.0, fp=50.0)), ("plate fp 60% < its own min 68%", dict(fp=60.0))):
    d, out = fixture(**kw)
    st, info = _fx.run_main(d, out)
    if st != "refused":
        fails.append("%s: plan went on toward slicing (%s)" % (label, info))
d, out = fixture()
st, info = _fx.run_main(d, out)
if os.path.exists(os.path.join(d, "run", "REFUSED_ACCOUNTING.json")):
    fails.append("control: a clean plan was refused: " + open(os.path.join(d, "run", "REFUSED_ACCOUNTING.json")).read()[:200])
_fx.finish("G3", fails)
