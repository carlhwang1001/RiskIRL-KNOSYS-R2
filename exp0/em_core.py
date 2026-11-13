"""EM core for Experiment 0'.
E-step: deterministic quadrature on a log-spaced beta grid (64 nodes,
[0.05, 5.0]) under a log-normal prior (mu0=0, sig0=0.75). Posterior
q(beta_i)=N(bhat_i, sig2_i) by moment matching.  (Revision item #15.)
M-step: per-trajectory feature-matching gradients (myopic dQ/dtheta ~ phi),
discounted by 1/(1+sig2_i); aggregated by {vanilla | pcgrad | cagrad}.
"""
import numpy as np
from scipy.stats import pearsonr
from env_riskgrid import PHI_SA
from solver_riskq import value_iteration, RATIONALITY

# ---------- quadrature grid ----------
N_NODES = 64
LOG_LO, LOG_HI = np.log(0.05), np.log(5.0)
NODES = np.exp(np.linspace(LOG_LO, LOG_HI, N_NODES))
MU0, SIG0 = 0.0, 0.75
_pl = -0.5 * ((np.log(NODES) - MU0) / SIG0) ** 2   # log-normal density in log-space; grid uniform in log
PRIOR_LOGW = _pl - (_pl.max() + np.log(np.exp(_pl - _pl.max()).sum()))  # normalized log prior weights

def _logsumexp(x, axis=None):
    m = np.max(x, axis=axis, keepdims=True)
    return (m + np.log(np.sum(np.exp(x - m), axis=axis, keepdims=True))).squeeze()

from env_riskgrid import P as _P, GOAL_S as _GOAL, GAMMA as _GAMMA, mean_R as _meanR, var_R as _varR, NS as _NS, NA as _NA

class QCache:
    """Q tables at all quadrature nodes, batched vectorized VI, warm-started.
    self.Qb has shape (N_NODES, NS, NA); self.Q[n] views into it."""
    def __init__(self):
        self.Qb = None
    @property
    def Q(self):
        return self.Qb
    @Q.setter
    def Q(self, v):
        self.Qb = np.array(v)
    def refresh(self, theta, tol=1e-5, max_iter=400):
        ER = _meanR(theta); VR = _varR(theta)
        base = ER[None] - 0.5 * (NODES[:, None, None] - 1.0) * VR[None]  # (K,NS,NA)
        Q = np.zeros((N_NODES, _NS, _NA)) if self.Qb is None else self.Qb
        cont = np.ones(_NS); cont[_GOAL] = 0.0
        P2T = _P.reshape(_NS * _NA, _NS).T                       # (NS, NS*NA)
        for _ in range(max_iter):
            V = Q.max(axis=2) * cont[None]                       # (K,NS)
            Qn = base + _GAMMA * (V @ P2T).reshape(N_NODES, _NS, _NA)
            if np.max(np.abs(Qn - Q)) < tol:
                self.Qb = Qn
                return
            Q = Qn
        self.Qb = Q
    def logpi(self):
        z = RATIONALITY * (self.Qb - self.Qb.max(axis=2, keepdims=True))
        return z - np.log(np.exp(z).sum(axis=2, keepdims=True))
    def loglik_matrix(self, trajs):
        """(N_traj, N_NODES) log-likelihoods, vectorized over nodes."""
        LP = self.logpi()                                        # (K,NS,NA)
        LL = np.zeros((len(trajs), N_NODES))
        for i, tr in enumerate(trajs):
            S = np.array([s for s, a in tr]); A = np.array([a for s, a in tr])
            LL[i] = LP[:, S, A].sum(axis=1)
        return LL

def e_step(LL):
    """Returns bhat (N,), sig2 (N,), marg_ll scalar (quadrature marginal loglik)."""
    logw = PRIOR_LOGW[None, :] + LL
    ml = _logsumexp(logw, axis=1)                    # (N,)
    w = np.exp(logw - ml[:, None])
    bhat = w @ NODES
    sig2 = (w * (NODES[None, :] - bhat[:, None]) ** 2).sum(axis=1)
    return bhat, sig2, ml.sum(), w

def per_traj_grads(trajs, w_post, theta, qc, eps=0.02):
    """Numerical central-difference gradient of the per-trajectory M-step
    objective f_i(theta) = sum_n w_in * loglik_i(theta, node_n), with
    posterior weights w_post held fixed (proper EM M-step gradient).
    Exact through the risk-sensitive value iteration (no myopic dQ/dtheta
    approximation). theta is 4-dim -> 8 warm-started evaluations."""
    N = len(trajs)
    G = np.zeros((N, 4))
    for k in range(4):
        fs = []
        for sgn in (+1.0, -1.0):
            th = theta.copy(); th[k] += sgn * eps
            qp = QCache()
            qp.Qb = qc.Qb.copy()                 # warm start from base
            qp.refresh(th, max_iter=60)          # capped; truncation bias cancels in central diff
            LLp = qp.loglik_matrix(trajs)
            fs.append((w_post * LLp).sum(axis=1))
        G[:, k] = (fs[0] - fs[1]) / (2 * eps)
    return G

# ---------- gradient surgeries ----------
def count_conflicts(G):
    M = G @ G.T
    iu = np.triu_indices(len(G), 1)
    return int((M[iu] < 0).sum())

def agg_vanilla(G, rng=None):
    return G.mean(axis=0)

def agg_pcgrad(G, rng):
    N = len(G)
    Gp = G.copy()
    for i in range(N):
        order = rng.permutation(N)
        for j in order:
            if j == i:
                continue
            d = Gp[i] @ G[j]
            if d < 0:
                Gp[i] = Gp[i] - d / (G[j] @ G[j] + 1e-12) * G[j]
    return Gp.mean(axis=0)

def agg_cagrad(G, rng=None, c=0.5, iters=200, lr=0.5):
    """min_w on simplex: w^T m0 + c*||g0|| * sqrt(w^T M w);  d = g0 + (c||g0||/||g_w||) g_w."""
    N = len(G)
    M = G @ G.T
    g0 = G.mean(axis=0)
    g0n = np.linalg.norm(g0) + 1e-12
    m0 = M.mean(axis=1)
    w = np.ones(N) / N
    def proj_simplex(v):
        u = np.sort(v)[::-1]
        css = np.cumsum(u)
        rho = np.nonzero(u * np.arange(1, N + 1) > (css - 1))[0][-1]
        tau = (css[rho] - 1) / (rho + 1.0)
        return np.maximum(v - tau, 0)
    for _ in range(iters):
        Mw = M @ w
        nw = np.sqrt(max(w @ Mw, 1e-12))
        grad = m0 + c * g0n * Mw / nw
        w = proj_simplex(w - lr * grad / (np.linalg.norm(grad) + 1e-12))
    gw = G.T @ w
    nw = np.linalg.norm(gw) + 1e-12
    return g0 + (c * g0n / nw) * gw

SURGERIES = {'vanilla': agg_vanilla, 'pcgrad': agg_pcgrad, 'cagrad': agg_cagrad}

class Adam:
    def __init__(self, dim, lr=0.05):
        self.m = np.zeros(dim); self.v = np.zeros(dim)
        self.t = 0; self.lr = lr
    def step(self, theta, grad):
        self.t += 1
        self.m = 0.9 * self.m + 0.1 * grad
        self.v = 0.999 * self.v + 0.001 * grad * grad
        mh = self.m / (1 - 0.9 ** self.t); vh = self.v / (1 - 0.999 ** self.t)
        return theta + self.lr * mh / (np.sqrt(vh) + 1e-8)

def run_em(trajs, beta_true, method, seed, epochs=60, lr=0.05, verbose=False):
    rng = np.random.default_rng(seed)
    theta = rng.normal(0, 0.5, 4)
    qc = QCache(); adam = Adam(4, lr)
    hist = {'elbo': [], 'conflicts': [], 'dtheta': []}
    for ep in range(epochs):
        qc.refresh(theta)
        LL = qc.loglik_matrix(trajs)
        bhat, sig2, ml, w_post = e_step(LL)
        G = per_traj_grads(trajs, w_post, theta, qc)
        hist['conflicts'].append(count_conflicts(G))
        d = SURGERIES[method](G, rng)
        theta_new = adam.step(theta, d)
        hist['elbo'].append(ml)
        hist['dtheta'].append(float(np.linalg.norm(theta_new - theta)))
        theta = theta_new
        if verbose and ep % 10 == 0:
            print(f"  ep{ep:3d} ml={ml:9.2f} conf={hist['conflicts'][-1]:3d} th={np.round(theta,2)}")
        if hist['dtheta'][-1] < 1e-3 and ep > 10:
            break
    qc.refresh(theta)
    bhat, sig2, ml, _ = e_step(qc.loglik_matrix(trajs))
    r = pearsonr(bhat, beta_true)[0]
    mae = {b: float(np.abs(bhat[beta_true == b] - b).mean()) for b in np.unique(beta_true)}
    return dict(theta=theta, bhat=bhat, sig2=sig2, r=float(r), mae=mae,
                elbo=hist['elbo'], conflicts=hist['conflicts'], epochs=ep + 1)
