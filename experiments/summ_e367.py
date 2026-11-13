import json, numpy as np, sys
from scipy import stats
R = [json.loads(l) for l in open(sys.argv[1])]
F = {json.loads(l)['seed']: json.loads(l) for l in open(sys.argv[2]) if json.loads(l)['variant'] == 'full'}
def ci(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) == 0: return 'n/a'
    if len(x) < 2: return f"{x.mean():.3f}"
    h = stats.t.ppf(.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)); return f"{x.mean():.3f}±{h:.3f}"
def block(rs, label):
    best = max(r['obj'] for r in rs); keep = [r for r in rs if r['obj'] >= best - 100]
    g = lambda k: [r[k] for r in keep]
    print(f"{label:12}{len(keep):>3}/{len(rs)} r {ci(g('r')):>13} worstMAE {ci(g('worst_mae')):>13} goal {ci([r['theta'][0] for r in keep]):>14} MSE|sh {ci(g('mse_mod_shaping')):>12} regRS {ci(g('regret_rs')):>12} HO {ci(g('heldout_post_pred')):>14} viol {np.mean(g('viol')):5.1f} obj {ci(g('obj')):>16}")
    return keep
print('== E3 mixture EM (group-level beta) ==')
for G in [2, 3, 4, 5, 6]:
    rs = [r for r in R if r['variant'] == f'mix{G}']; k = block(rs, f'mix{G}')
    print('     betas (seed0..):', [r['mix_beta'] for r in rs[:3]])
print('== E6 no projection ==')
rs = [r for r in R if r['variant'] == 'noproj']; block(rs, 'noproj')
print('   per seed goal/obj:', [(r['seed'], round(r['theta'][0], 2), round(r['obj'], 1)) for r in sorted(rs, key=lambda r: r['seed'])])
best = max(r['obj'] for r in rs); mir = [r for r in rs if r['theta'][0] < 0]
print(f"   mirror rate {len(mir)}/10; mirror obj gap to best: {[round(best - r['obj'], 1) for r in mir]}; mirror regretRS {[round(r['regret_rs'], 3) for r in mir]}; selected-by-ELBO goal sign: {'+' if max(rs, key=lambda r: r['obj'])['theta'][0] > 0 else '-'}")
print('== E6 lambda profile (3 seeds each) ==')
for lam in [1.5, 2.0, 3.0, 4.0, 6.0]:
    rs = [r for r in R if r['variant'] == f'lam{lam}'] if lam != 3.0 else [F[s] for s in range(3)]
    print(f"   lam {lam}: obj {ci([r['obj'] for r in rs]):>18} r {ci([r['r'] for r in rs]):>12} goal {ci([r['theta'][0] for r in rs]):>14} theta {np.mean([r['theta'] for r in rs], 0).round(2).tolist()} HO {ci([r['heldout_post_pred'] for r in rs])}")
print('== E7 aggregation (full model) ==')
block([F[s] for s in range(10)], 'mean')
for op in ['pcgrad', 'cagrad', 'clip', 'randproj']:
    block([r for r in R if r['variant'] == op], op)
