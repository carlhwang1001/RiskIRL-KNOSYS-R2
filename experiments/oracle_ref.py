import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__)); _EXP0 = _os.path.join(_HERE, '..', 'exp0'); _RES = _os.path.join(_HERE, '..', 'results')
import sys; sys.path.insert(0, _HERE)
import numpy as np, r2lib as L, em_core as E
trajs, lab = L.generate(L.TRAIN_SEED); ho = L.heldout_set(); nodes, lp = L.make_grid(); L.set_grid(nodes)
qc = E.QCache(); qc.refresh(L.THETA_TRUE, tol=1e-6, max_iter=2000); m = L.TrajLevel(lp)
w, obj = m.weights(qc.loglik_matrix(trajs))
res = dict(theta=L.THETA_TRUE.copy(), w=w, obj=obj, epochs=0, conflicts=[0], viol=0, qc=qc)
s = L.summarize(res, nodes, lab, m, ho); print({k: v for k, v in s.items() if k in ('r', 'worst_mae', 'cover90', 'heldout_post_pred', 'regret_rs', 'obj')})
