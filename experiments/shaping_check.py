import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__)); _EXP0 = _os.path.join(_HERE, '..', 'exp0'); _RES = _os.path.join(_HERE, '..', 'results')
import sys; sys.path.insert(0, _EXP0)
import numpy as np
from solver_riskq import value_iteration, traj_loglik
from gen_demos import generate
from env_riskgrid import THETA_TRUE, GAMMA
trajs, lab = generate(1234)
g = GAMMA; v = np.array([-g, 0, 0, -(1 - g)])
for kappa in [5.0, 20.76]:
    th2 = THETA_TRUE + kappa * v
    out = []
    for b in [0.3, 0.7, 1.0, 1.5, 2.5]:
        Q1 = value_iteration(THETA_TRUE, b, tol=1e-9, max_iter=5000); Q2 = value_iteration(th2, b, tol=1e-9, max_iter=5000)
        d = sum(traj_loglik(t, Q2) - traj_loglik(t, Q1) for t in trajs)
        out.append(f"beta={b}: dLL={d:+.4f}")
    print('kappa', kappa, 'theta2', np.round(th2, 3), ' | '.join(out))
