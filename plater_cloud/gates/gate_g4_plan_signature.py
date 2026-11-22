"""G4: a saved PLAN_RESULT is reused only when the plan settings (fill minimum) and pieces match.  usage: python gate_g4_plan_signature.py <tree>"""
import json, os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _fx
_fx.setup(sys.argv[1])
import mkm_plan
fails = []
d = tempfile.mkdtemp(); pj = os.path.join(d, "pieces.json"); json.dump([dict(set="S", file="a.stl", path="a.stl")], open(pj, "w"))
mkm_plan._case_input = lambda p, o: p
calls = []


def fake_one(pieces, out_dir, n, gate, deadline=None):
    calls.append(n); out = os.path.join(out_dir, "N%s" % (n or "math"), "out"); os.makedirs(out, exist_ok=True)
    return dict(pieces=os.path.abspath(pieces), n=1, n_math=1, done=True, carried=0, lone_carried=0, fill_carried=0, hard_fail=[], carry_reasons=[], hard_dropped=[], fill=[], fill2=[], out=out)


mkm_plan._one = fake_one
od = os.path.join(d, "plan")
os.environ["MKM_FILL_MIN"] = "0.5"; mkm_plan.plan(pj, od); n1 = len(calls)
mkm_plan.plan(pj, od)
if len(calls) != n1:
    fails.append("control: same pieces + same settings were re-planned")
os.environ["MKM_FILL_MIN"] = "0.68"; mkm_plan.plan(pj, od)
if len(calls) == n1:
    fails.append("a plan made at fill minimum 0.5 was reused at 0.68 (stale PLAN_RESULT)")
n2 = len(calls); json.dump([dict(set="S", file="b.stl", path="b.stl")], open(pj, "w")); mkm_plan.plan(pj, od)
if len(calls) == n2:
    fails.append("a plan was reused after pieces.json content changed")
_fx.finish("G4", fails)
