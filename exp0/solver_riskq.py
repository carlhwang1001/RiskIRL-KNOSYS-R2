"""Risk-sensitive value iteration (mean-variance form), unified model:
  Q_beta(s,a) = E[R] - 0.5*(beta-1)*Var[R] + gamma * E_{s'}[V(s')],
  V(s) = max_a Q(s,a);  policy pi(a|s) = softmax(Q_beta(s,.)) (temperature 1,
  beta appears ONLY inside Q -- the unified form of paper eq.(1)).
Goal state is absorbing with zero continuation.
"""
import numpy as np
from env_riskgrid import P, GOAL_S, GAMMA, mean_R, var_R, NS, NA

RATIONALITY = 3.0  # fixed Boltzmann rationality lambda, shared by all experts;
# pi(a|s) = softmax(RATIONALITY * Q_beta). Separates rationality (known constant)
# from risk attitude beta (latent). Unified-model form of paper eq.(1).

def value_iteration(theta, beta, Q0=None, tol=1e-6, max_iter=500):
    ER = mean_R(theta)
    VR = var_R(theta)
    base = ER - 0.5 * (beta - 1.0) * VR      # (NS,NA)
    Q = np.zeros((NS, NA)) if Q0 is None else Q0.copy()
    cont = np.ones(NS); cont[GOAL_S] = 0.0   # absorb at goal
    for _ in range(max_iter):
        V = Q.max(axis=1) * cont
        Qn = base + GAMMA * np.einsum('sap,p->sa', P, V)
        if np.max(np.abs(Qn - Q)) < tol:
            return Qn
        Q = Qn
    return Q

def policy_from_Q(Q, lam=None):
    lam = RATIONALITY if lam is None else lam
    z = lam * (Q - Q.max(axis=1, keepdims=True))
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)

def traj_loglik(traj, Q, lam=None):
    """traj: list of (s,a). log prod softmax(lam*Q)[s,a]."""
    lam = RATIONALITY if lam is None else lam
    z = lam * (Q - Q.max(axis=1, keepdims=True))
    logZ = np.log(np.exp(z).sum(axis=1))
    ll = 0.0
    for s, a in traj:
        ll += z[s, a] - logZ[s]
    return ll
