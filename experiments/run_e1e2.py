import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__)); _EXP0 = _os.path.join(_HERE, '..', 'exp0'); _RES = _os.path.join(_HERE, '..', 'results')
"""E1 (within-framework beta ablation) and E2 (oracle identity). usage: run_e1e2.py <worker_id> <n_workers> <out.jsonl>"""
import sys, json, time; sys.path.insert(0, _HERE)
import numpy as np, r2lib as L
wid, nw, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
trajs, lab = L.generate(L.TRAIN_SEED); ho = L.heldout_set()
VARIANTS = ['full', 'uniform', 'K7', 'K3', 'population', 'kmeans4', 'beta1', 'oracle_identity', 'known_beta']
def build(v, seed):
    if v == 'full': n, lp = L.make_grid(); return L.TrajLevel(lp), n
    if v == 'uniform': n, lp = L.make_grid(prior='uniform'); return L.TrajLevel(lp), n
    if v == 'K7': n, lp = L.make_grid(K=7); return L.TrajLevel(lp), n
    if v == 'K3': n, lp = L.make_grid(K=3); return L.TrajLevel(lp), n
    if v == 'population': n, lp = L.make_grid(); return L.Grouped(lp, np.zeros(len(trajs), int), v), n
    if v == 'kmeans4': n, lp = L.make_grid(); return L.Grouped(lp, L.kmeans_pp(L.traj_features(trajs), 4, seed), v), n
    if v == 'oracle_identity': n, lp = L.make_grid(); return L.Grouped(lp, np.searchsorted(L.BETAS_TRUE, lab), v), n
    if v == 'beta1': return L.Fixed(np.ones((len(trajs), 1)), v), np.array([1.0])
    if v == 'known_beta':
        n = np.array(L.BETAS_TRUE, float); W = (lab[:, None] == n[None]).astype(float); return L.Fixed(W, v), n
jobs = [(v, s) for s in range(10) for v in VARIANTS]
for k, (v, s) in enumerate(jobs):
    if k % nw != wid: continue
    t0 = time.time(); m, nodes = build(v, s); res = L.run(m, nodes, trajs, s)
    rec = dict(variant=v, seed=s, **L.summarize(res, nodes, lab, m, ho), secs=round(time.time() - t0, 1))
    with open(out, 'a') as f: f.write(json.dumps(rec) + '\n')
    print(v, s, round(rec['r'], 3) if rec['r'] == rec['r'] else None, rec['theta'], rec['secs'], flush=True)
