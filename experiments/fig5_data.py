import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__)); _EXP0 = _os.path.join(_HERE, '..', 'exp0'); _RES = _os.path.join(_HERE, '..', 'results')
import sys, json; sys.path.insert(0, _HERE)
import numpy as np, r2lib as L, em_core as E
trajs, lab = L.generate(L.TRAIN_SEED); ht, hl = L.heldout_set()
R = [json.loads(l) for l in open(_os.path.join(_RES, 'e1e2.jsonl'))]
out = {}
for v in ['full', 'beta1', 'known_beta']:
    for r in [x for x in R if x['variant'] == v]:
        th = np.array(r['theta'])
        if v == 'full':
            nodes, lp = L.make_grid(); L.set_grid(nodes); qc = E.QCache(); qc.refresh(th, tol=1e-6, max_iter=2000)
            w, _ = L.TrajLevel(lp).weights(qc.loglik_matrix(trajs)); pred = np.log(w.mean(0))
        elif v == 'beta1':
            nodes = np.array([1.0]); L.set_grid(nodes); qc = E.QCache(); qc.refresh(th, tol=1e-6, max_iter=2000); pred = np.zeros(1)
        else:
            nodes = np.array(L.BETAS_TRUE, float); L.set_grid(nodes); qc = E.QCache(); qc.refresh(th, tol=1e-6, max_iter=2000); pred = np.log(np.full(4, .25))
        LLh = qc.loglik_matrix(ht); ll = L.lse(pred[None] + LLh, 1); steps = np.array([len(t) for t in ht])
        out.setdefault(v, []).append(dict(seed=r['seed'], goal=th[0], group={str(b): float(ll[hl == b].sum() / steps[hl == b].sum()) for b in L.BETAS_TRUE}))
json.dump(out, open(_os.path.join(_RES, 'fig5_data.json'), 'w'))
for v, rs in out.items(): print(v, {b: round(np.mean([r['group'][b] for r in rs]), 3) for b in rs[0]['group']}, round(np.mean([r['goal'] for r in rs]), 2))
