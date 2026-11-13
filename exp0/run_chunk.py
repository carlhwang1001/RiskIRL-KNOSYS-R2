"""Chunked, resumable runner: python3 run_chunk.py <method> <seed> [<seed>...]
Appends one JSON line per completed run to exp0_runs.jsonl."""
import sys, json, time
import numpy as np
from gen_demos import generate
from em_core import run_em

def main():
    method = sys.argv[1]
    seeds = [int(x) for x in sys.argv[2:]]
    trajs, labels = generate(1234)
    for s in seeds:
        t0 = time.time()
        r = run_em(trajs, labels, method, seed=s, epochs=100, lr=0.3)
        e = np.array(r['elbo'])
        rec = dict(method=method, seed=s, r=r['r'], mae=r['mae'],
                   theta=list(map(float, r['theta'])),
                   conflicts_mean=float(np.mean(r['conflicts'])),
                   conflicts_first=int(r['conflicts'][0]), conflicts_last=int(r['conflicts'][-1]),
                   mono=float((np.diff(e) >= -1e-6).mean()),
                   elbo_final=float(e[-1]), epochs=r['epochs'], secs=round(time.time()-t0, 1))
        with open('exp0_runs.jsonl', 'a') as f:
            f.write(json.dumps(rec) + '\n')
        print(json.dumps({k: rec[k] for k in ('method','seed','r','theta','conflicts_mean','mono','secs')}), flush=True)

if __name__ == '__main__':
    main()
