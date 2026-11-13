import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__)); _EXP0 = _os.path.join(_HERE, '..', 'exp0'); _RES = _os.path.join(_HERE, '..', 'results')
import sys; sys.path.insert(0, _EXP0)
import numpy as np
from scipy.stats import pearsonr
import em_core as E
from gen_demos import generate
from env_riskgrid import THETA_TRUE
def setgrid(lo, hi):
    E.NODES = np.exp(np.linspace(np.log(lo), np.log(hi), 64))
    pl = -0.5 * (np.log(E.NODES) / 0.75) ** 2
    E.PRIOR_LOGW = pl - (pl.max() + np.log(np.exp(pl - pl.max()).sum()))
trajs, lab = generate(1234)
for lo, hi in [(0.05, 5.0), (0.02, 10.0)]:
    setgrid(lo, hi); qc = E.QCache(); qc.refresh(THETA_TRUE, tol=1e-6, max_iter=2000)
    bhat, s2, ml, w = E.e_step(qc.loglik_matrix(trajs))
    print(lo, hi, 'oracle r =', round(pearsonr(bhat, lab)[0], 4), 'ELBO at theta* =', round(ml, 2))
