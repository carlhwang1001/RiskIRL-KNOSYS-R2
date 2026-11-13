import json, numpy as np, sys
from scipy import stats
R = [json.loads(l) for l in open(sys.argv[1])]
V = ['full', 'uniform', 'K7', 'K3', 'population', 'kmeans4', 'beta1', 'oracle_identity', 'known_beta']
def ci(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) < 2: return f"{x.mean():.3f}" if len(x) else 'n/a'
    h = stats.t.ppf(.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)); return f"{x.mean():.3f}±{h:.3f}"
full = {r['seed']: r for r in R if r['variant'] == 'full'}
print(f"{'variant':16}{'kept':>5} {'r':>13} {'worstMAE':>13} {'goal':>13} {'MSE':>13} {'MSE|shap':>13} {'regretRS':>13} {'HO post':>15} {'cover90':>8}")
for v in V:
    rs = [r for r in R if r['variant'] == v]; best = max(r['obj'] for r in rs); keep = [r for r in rs if r['obj'] >= best - 100]
    g = lambda k: [r[k] for r in keep]
    print(f"{v:16}{len(keep):>3}/10 {ci(g('r')):>13} {ci(g('worst_mae')):>13} {ci([r['theta'][0] for r in keep]):>13} {ci(g('mse')):>13} {ci(g('mse_mod_shaping')):>13} {ci(g('regret_rs')):>13} {ci(g('heldout_post_pred')):>15} {np.mean(g('cover90')):8.2f}")
# paired held-out difference vs full (all seeds where both kept)
print('\nPaired held-out (post-pred, nats/step) full minus variant, seeds kept in both:')
for v in V[1:]:
    rs = {r['seed']: r for r in R if r['variant'] == v}
    d = [full[s]['heldout_post_pred'] - rs[s]['heldout_post_pred'] for s in range(10)]
    t = stats.ttest_1samp(d, 0)
    print(f"  {v:16} diff {ci(d):>15}  t={t.statistic:7.2f} p={t.pvalue:.2g}")
