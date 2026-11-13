import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__)); _EXP0 = _os.path.join(_HERE, '..', 'exp0'); _RES = _os.path.join(_HERE, '..', 'results')
import sys, json; sys.path.insert(0, _HERE)
import numpy as np, r2lib as L, em_core as E
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
trajs, lab = L.generate(L.TRAIN_SEED); ht, hl = L.heldout_set()
nodes, lp = L.make_grid(); L.set_grid(nodes); qc = E.QCache(); qc.refresh(L.THETA_TRUE, tol=1e-6, max_iter=2000)
w, _ = L.TrajLevel(lp).weights(qc.loglik_matrix(trajs)); ll = L.lse(np.log(w.mean(0))[None] + qc.loglik_matrix(ht), 1); steps = np.array([len(t) for t in ht])
oracle = {str(b): ll[hl == b].sum() / steps[hl == b].sum() for b in L.BETAS_TRUE}
D = json.load(open(_os.path.join(_RES, 'fig5_data.json')))
E1 = [json.loads(l) for l in open(_os.path.join(_RES, 'e1e2.jsonl'))]; E7 = [json.loads(l) for l in open(_os.path.join(_RES, 'e367.jsonl'))]
def kept(rs):
    best = max(r['obj'] for r in rs); return [(r['theta'][0], r['obj'] >= best - 100) for r in rs]
groups = [('Averaging\n(default)', kept([r for r in E1 if r['variant'] == 'full']), '#1f77b4'),
          ('PCGrad', kept([r for r in E7 if r['variant'] == 'pcgrad']), '#e8622c'),
          ('CAGrad', kept([r for r in E7 if r['variant'] == 'cagrad']), '#1aaf7a'),
          ('Known-β\noracle', kept([r for r in E1 if r['variant'] == 'known_beta']), '#444444'),
          ('Homog.\n(β = 1)', kept([r for r in E1 if r['variant'] == 'beta1']), '#8c8c87')]
plt.rcParams.update({'font.family': 'DejaVu Serif', 'font.size': 10})
fig, ax = plt.subplots(2, 1, figsize=(5.2, 5.6), dpi=300, gridspec_kw={'hspace': 0.55})
rng = np.random.default_rng(0)
for k, (name, vals, c) in enumerate(groups):
    x = k + rng.uniform(-0.15, 0.15, len(vals)); g = np.array([v for v, _ in vals]); keep = np.array([kp for _, kp in vals])
    ax[0].scatter(x[keep], g[keep], s=22, color=c, edgecolor='white', linewidth=0.5, zorder=3)
    ax[0].scatter(x[~keep], g[~keep], s=22, facecolor='none', edgecolor=c, linewidth=0.9, zorder=3)
    ax[0].hlines(g[keep].mean(), k - 0.27, k + 0.27, color='black', lw=2, zorder=4)
ax[0].axhline(10, ls='--', color='#555555', lw=1); ax[0].text(4.45, 10.4, 'true value', ha='right', fontsize=8, color='#555555')
ax[0].set_xticks(range(5)); ax[0].set_xticklabels([g[0] for g in groups], fontsize=8.5); ax[0].set_ylabel(r'recovered $\theta_{goal}$')
ax[0].set_title('(a) Goal weight per seed (bar = mean of retained runs; hollow = screened)', fontsize=9, loc='left', fontweight='bold')
ax[0].set_ylim(-1.5, 13.5); ax[0].grid(axis='y', color='#e5e5e5'); [ax[0].spines[s].set_visible(False) for s in ('top', 'right')]
bs = [str(b) for b in L.BETAS_TRUE]; xb = np.arange(4); wd = 0.26
series = [('Oracle (θ*)', [oracle[b] for b in bs], None, '#c8c7c2'),
          ('Framework (averaging)', [np.mean([r['group'][b] for r in D['full']]) for b in bs], [np.std([r['group'][b] for r in D['full']], ddof=1) for b in bs], '#1aaf7a'),
          ('Homogeneous (β = 1)', [np.mean([r['group'][b] for r in D['beta1']]) for b in bs], [np.std([r['group'][b] for r in D['beta1']], ddof=1) for b in bs], '#8c8c87')]
for j, (nm, mu, sd, c) in enumerate(series):
    ax[1].bar(xb + (j - 1) * wd, mu, wd * 0.95, color=c, label=nm, yerr=sd, error_kw=dict(lw=0.8, capsize=1.5))
ax[1].set_xticks(xb); ax[1].set_xticklabels([f'β = {b}' for b in bs]); ax[1].set_ylabel('held-out log-lik.\n(nats/step)')
ax[1].set_title('(b) Held-out likelihood per group (common posterior predictive rule)', fontsize=9, loc='left', fontweight='bold')
ax[1].grid(axis='y', color='#e5e5e5'); [ax[1].spines[s].set_visible(False) for s in ('top', 'right')]
ax[1].legend(ncol=3, fontsize=7.5, frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.15))
fig.savefig(sys.argv[1], bbox_inches='tight'); print('saved', {k: round(v, 3) for k, v in oracle.items()})
