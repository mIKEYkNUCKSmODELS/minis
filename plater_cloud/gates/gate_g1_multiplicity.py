"""G1: accounting counts with multiplicity.  usage: python gate_g1_multiplicity.py <tree>"""
import json, os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _fx
_fx.setup(sys.argv[1])
import mkm_accounting as MA
fails = []
d = tempfile.mkdtemp(); a, b = os.path.join(d, "a.stl"), os.path.join(d, "b.stl"); pj = os.path.join(d, "pieces.json")
json.dump([dict(set="S", file="a.stl", path=a), dict(set="S", file="a.stl", path=a), dict(set="T", file="b.stl", path=b)], open(pj, "w"))
if not any("not accounted" in p for p in MA.model_accounting(pj, _fx.make_plan(d, [a, b]))):
    fails.append("one of two copies of a.stl vanished and was not flagged")
d2 = tempfile.mkdtemp(); a2, b2 = os.path.join(d2, "a.stl"), os.path.join(d2, "b.stl"); pj2 = os.path.join(d2, "pieces.json")
json.dump([dict(set="S", file="a.stl", path=a2), dict(set="S", file="a.stl", path=a2), dict(set="T", file="b.stl", path=b2)], open(pj2, "w"))
if MA.model_accounting(pj2, _fx.make_plan(d2, [a2, a2, b2])):
    fails.append("control: every copy placed but flagged")
d3 = tempfile.mkdtemp(); a3, b3 = os.path.join(d3, "a.stl"), os.path.join(d3, "b.stl"); pj3 = os.path.join(d3, "pieces.json")
json.dump([dict(set="S", file="a.stl", path=a3), dict(set="S", file="a.stl", path=a3), dict(set="T", file="b.stl", path=b3)], open(pj3, "w"))
out3 = _fx.make_plan(d3, [b3]); _fx.jd(os.path.join(out3, "LONE_CARRY.json"), [dict(set="S", file="a.stl"), dict(set="S", file="a.stl")])
if MA.model_accounting(pj3, out3):
    fails.append("control: both copies carried (none placed) but flagged")
_fx.finish("G1", fails)
