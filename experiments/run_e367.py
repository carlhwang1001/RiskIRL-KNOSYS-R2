import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__)); _EXP0 = _os.path.join(_HERE, '..', 'exp0'); _RES = _os.path.join(_HERE, '..', 'results')
"""E3 (group-level mixture EM), E6 (identifiability diagnostics), E7 (aggregation diagnostics).
usage: run_e367.py <worker_id> <n_workers> <out.jsonl>"""
import sys, json, time; sys.path.insert(0, _HERE)
import numpy as np, r2lib as L, em_core as E
wid, nw, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
trajs, lab = L.generate(L.TRAIN_SEED); ho = L.heldout_set()

def agg_clip(G, rng=None):            # per-trajectory gradient clipping at the median norm, then mean
    n = np.linalg.norm(G, axis=1); c = np.median(n) + 1e-12
    return (G * np.minimum(1.0, c / (n + 1e-12))[:, None]).mean(0)
def agg_randproj(G, rng):             # control: remove the component along one random other gradient, regardless of sign
    N = len(G); Gp = G.copy()
    for i in range(N):
        j = rng.integers(N - 1); j += j >= i
        Gp[i] = Gp[i] - (Gp[i] @ G[j]) / (G[j] @ G[j] + 1e-12) * G[j]
    return Gp.mean(0)
E.SURGERIES['clip'] = agg_clip; E.SURGERIES['randproj'] = agg_randproj

jobs = []
for s in range(10):
    for G in [2, 3, 4, 5, 6]: jobs.append(('E3', f'mix{G}', s, {}))
    jobs.append(('E6', 'noproj', s, {}))
    for op in ['pcgrad', 'cagrad', 'clip', 'randproj']: jobs.append(('E7', op, s, {}))
for s in range(3):
    for lam in [1.5, 2.0, 4.0, 6.0]: jobs.append(('E6', f'lam{lam}', s, {'lam': lam}))
for k, (exp, v, s, kw) in enumerate(jobs):
    if k % nw != wid: continue
    t0 = time.time(); nodes, lp = L.make_grid(); E.RATIONALITY = kw.get('lam', 3.0)
    op, project = 'vanilla', True
    if exp == 'E3': m = L.Mixture(nodes, int(v[3:]))
    else: m = L.TrajLevel(lp)
    if exp == 'E6' and v == 'noproj': project = False
    if exp == 'E7': op = v
    res = L.run(m, nodes, trajs, s, op=op, project=project)
    rec = dict(exp=exp, variant=v, seed=s, lam=E.RATIONALITY, **L.summarize(res, nodes, lab, m, ho), secs=round(time.time() - t0, 1))
    if exp == 'E3': rec['mix_beta'] = nodes[m.idx].round(3).tolist(); rec['mix_pi'] = np.exp(m.logpi).round(3).tolist()
    rec['objs'] = [round(x, 2) for x in res['objs']]
    with open(out, 'a') as f: f.write(json.dumps(rec) + '\n')
    print(exp, v, s, rec['theta'], rec['secs'], flush=True)
