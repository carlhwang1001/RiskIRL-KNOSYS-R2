import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__)); _EXP0 = _os.path.join(_HERE, '..', 'exp0'); _RES = _os.path.join(_HERE, '..', 'results')
"""Final-protocol replica: grid [0.02,10], theta_goal>=0 projection, Adam lr 0.3, <=100 iters.
usage: python3 -I final_proto.py <op> <seed> ..."""
import sys, json, time; sys.path.insert(0, _EXP0)
import numpy as np
from scipy.stats import pearsonr
import em_core as E
from gen_demos import generate
E.NODES = np.exp(np.linspace(np.log(0.02), np.log(10.0), 64))
_pl = -0.5 * (np.log(E.NODES) / 0.75) ** 2
E.PRIOR_LOGW = _pl - (_pl.max() + np.log(np.exp(_pl - _pl.max()).sum()))
trajs, lab = generate(1234)
op = sys.argv[1]
for seed in map(int, sys.argv[2:]):
    t0 = time.time(); rng = np.random.default_rng(seed)
    th = rng.normal(0, 0.5, 4); th[0] = max(th[0], 0.0)
    qc = E.QCache(); adam = E.Adam(4, 0.3); el = []; conf = []
    for ep in range(100):
        qc.refresh(th); LL = qc.loglik_matrix(trajs); bh, s2, ml, w = E.e_step(LL)
        G = E.per_traj_grads(trajs, w, th, qc); conf.append(E.count_conflicts(G))
        d = E.SURGERIES[op](G, rng); new = adam.step(th, d); new[0] = max(new[0], 0.0)
        el.append(ml); st = np.linalg.norm(new - th); th = new
        if st < 1e-3 and ep > 10: break
    qc.refresh(th); bh, s2, ml, w = E.e_step(qc.loglik_matrix(trajs))
    mae = {str(b): float(np.abs(bh[lab == b] - b).mean()) for b in np.unique(lab)}
    rec = dict(op=op, seed=seed, r=float(pearsonr(bh, lab)[0]), theta=th.round(4).tolist(), elbo=float(ml), epochs=ep + 1,
               mae=mae, conf_mean=float(np.mean(conf)), secs=round(time.time() - t0, 1))
    with open(_os.path.join(_RES, 'final_runs.jsonl'), 'a') as f: f.write(json.dumps(rec) + '\n')
    print(json.dumps(rec), flush=True)
