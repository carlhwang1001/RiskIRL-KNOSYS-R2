"""Experiment 0': smoke test. 5 seeds x {vanilla, pcgrad, cagrad}, 60 epochs.
Acceptance: (a) ELBO-proxy monotone-ish ascent & theta convergence,
(b) beta-recovery Pearson r (report as-is), (c) full-batch conflict counts.
"""
import numpy as np, json, time
from gen_demos import generate
from em_core import run_em

METHODS = ['vanilla', 'pcgrad', 'cagrad']
SEEDS = [0, 1, 2, 3, 4]

def main():
    trajs, labels = generate(1234)          # data seed fixed, separate from training seeds
    out = {}
    for m in METHODS:
        out[m] = []
        for s in SEEDS:
            t0 = time.time()
            res = run_em(trajs, labels, m, seed=s, epochs=100, lr=0.3)
            res['time'] = time.time() - t0
            out[m].append(res)
            e = np.array(res['elbo'])
            mono = float((np.diff(e) >= -1e-6).mean())
            print(f"{m:8s} seed{s} r={res['r']:.3f} ep={res['epochs']:3d} "
                  f"conf(first/last)={res['conflicts'][0]}/{res['conflicts'][-1]} "
                  f"mono%={mono:.2f} th={np.round(res['theta'],2)} {res['time']:.0f}s")
    np.savez('exp0_results.npz',
             **{f"{m}_r": [r['r'] for r in out[m]] for m in METHODS},
             **{f"{m}_theta": [r['theta'] for r in out[m]] for m in METHODS})
    with open('exp0_summary.json', 'w') as f:
        json.dump({m: [{k: v for k, v in r.items() if k in ('r','mae','epochs','conflicts','elbo')}
                       for r in out[m]] for m in METHODS}, f, default=float)
    print("\n=== summary (mean±std over 5 seeds) ===")
    for m in METHODS:
        rs = np.array([r['r'] for r in out[m]])
        cf = np.array([np.mean(r['conflicts']) for r in out[m]])
        print(f"{m:8s} r = {rs.mean():.3f} ± {rs.std():.3f}   avg conflicts/epoch = {cf.mean():.1f}")

if __name__ == '__main__':
    main()
