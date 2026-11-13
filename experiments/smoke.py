import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__)); _EXP0 = _os.path.join(_HERE, '..', 'exp0'); _RES = _os.path.join(_HERE, '..', 'results')
import sys, time; sys.path.insert(0, _HERE)
import numpy as np, r2lib as L
trajs, lab = L.generate(1234); ho = L.heldout_set()
nodes, lp = L.make_grid()
t0 = time.time(); m = L.TrajLevel(lp); res = L.run(m, nodes, trajs, 0); s = L.summarize(res, nodes, lab, m, ho)
print(round(time.time() - t0, 1), {k: (round(v, 4) if isinstance(v, float) else v) for k, v in s.items() if k != 'mae'})
