import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__)); _EXP0 = _os.path.join(_HERE, '..', 'exp0'); _RES = _os.path.join(_HERE, '..', 'results')
"""R2 experiment library for KNOSYS-D-26-17431 (risk-sensitive IRL, identity-unlabeled demonstrations).

Builds on the exp0 code (env_riskgrid, solver_riskq, em_core, gen_demos) and the final protocol:
beta grid log-spaced on [0.02, 10], K = 64, LogNormal(0, 0.75^2) prior, Adam lr 0.3, <= 100 EM iterations,
stop when ||dtheta|| < 1e-3 after iteration 10, theta_goal >= 0 projection after every update,
ELBO screening (drop runs whose final objective is > 100 nats below the best run of the same condition).

Every model variant is expressed through the weight table w (N x K) that the M-step uses:
    g_i = sum_n w_in d/dtheta log P(tau_i | theta, beta_n)
so that all variants share the same likelihood, features, optimizer, finite-difference gradient and projection.
"""
import sys, os
sys.path.insert(0, os.environ.get('EXP0_SRC', _EXP0))
import numpy as np
from scipy.stats import pearsonr, spearmanr
import em_core as E
from gen_demos import generate, rollout, BETAS_TRUE
from env_riskgrid import THETA_TRUE, GAMMA, P as PT, GOAL_S, NS, NA, mean_R, var_R, START, sid
from solver_riskq import value_iteration, policy_from_Q

TRAIN_SEED, HELDOUT_SEED = 1234, 5678
SHAPING_DIR = np.array([-GAMMA, 0.0, 0.0, -(1.0 - GAMMA)]); SHAPING_DIR /= np.linalg.norm(SHAPING_DIR)

def lse(x, axis=None):
    m = np.max(x, axis=axis, keepdims=True)
    return np.squeeze(m + np.log(np.sum(np.exp(x - m), axis=axis, keepdims=True)), axis=axis)

def make_grid(K=64, lo=0.02, hi=10.0, prior='lognormal', nodes=None):
    nodes = np.exp(np.linspace(np.log(lo), np.log(hi), K)) if nodes is None else np.asarray(nodes, float)
    if prior == 'lognormal': lp = -0.5 * (np.log(nodes) / 0.75) ** 2
    elif prior == 'uniform': lp = np.zeros(len(nodes))
    else: raise ValueError(prior)
    return nodes, lp - lse(lp)

def set_grid(nodes):
    E.NODES = np.asarray(nodes, float); E.N_NODES = len(nodes)

def heldout_set(n_per=10, seed=HELDOUT_SEED):
    rng = np.random.default_rng(seed); trajs, lab = [], []
    for b in BETAS_TRUE:
        pi = policy_from_Q(value_iteration(THETA_TRUE, b))
        for _ in range(n_per): trajs.append(rollout(pi, rng)); lab.append(b)
    return trajs, np.array(lab)

def traj_features(trajs):
    from env_riskgrid import E_S
    out = []
    for t in trajs:
        S = np.array([s for s, a in t]); out.append([E_S[S, 1].sum(), E_S[S, 2].sum(), len(t)])
    X = np.array(out, float); return (X - X.mean(0)) / (X.std(0) + 1e-9)

def kmeans_pp(X, k, seed, iters=100):
    rng = np.random.default_rng(seed); C = [X[rng.integers(len(X))]]
    for _ in range(1, k):
        d2 = np.min([((X - c) ** 2).sum(1) for c in C], axis=0); C.append(X[rng.choice(len(X), p=d2 / d2.sum())])
    C = np.array(C)
    for _ in range(iters):
        lab = np.argmin(((X[:, None] - C[None]) ** 2).sum(2), 1)
        Cn = np.array([X[lab == j].mean(0) if np.any(lab == j) else C[j] for j in range(k)])
        if np.allclose(Cn, C): break
        C = Cn
    return lab

# ---------------- weight models ----------------
class TrajLevel:
    """Proposed model: independent trajectory-level beta, discrete posterior on the grid."""
    name = 'traj'
    def __init__(self, logprior): self.lp = logprior
    def weights(self, LL):
        lw = self.lp[None] + LL; ml = lse(lw, 1); return np.exp(lw - ml[:, None]), float(ml.sum())
    def predictive(self, w): return w.mean(0)          # learned beta distribution for a new trajectory

class Grouped:
    """Group-level beta with a fixed assignment (population: one group; oracle identity: true groups; k-means clusters)."""
    def __init__(self, logprior, groups, name):
        self.lp = logprior; self.g = np.asarray(groups); self.name = name
    def weights(self, LL):
        w = np.zeros_like(LL); obj = 0.0
        for gid in np.unique(self.g):
            m = self.g == gid; lw = self.lp + LL[m].sum(0); ml = lse(lw); w[m] = np.exp(lw - ml); obj += ml
        return w, float(obj)
    def predictive(self, w):
        gs, cnt = np.unique(self.g, return_counts=True)
        return sum(c * w[np.argmax(self.g == gid)] for gid, c in zip(gs, cnt)) / cnt.sum()

class Fixed:
    """Fixed per-trajectory weights (beta = 1 point mass, or known-beta oracle)."""
    def __init__(self, W, name): self.W = np.asarray(W, float); self.name = name
    def weights(self, LL): return self.W, float((self.W * LL).sum())
    def predictive(self, w): return w.mean(0)

class Mixture:
    """Mixture EM with G group-level risk coefficients (each a grid node) and soft responsibilities."""
    def __init__(self, nodes, G, name=None):
        self.nodes = nodes; self.G = G; self.name = name or f'mix{G}'
        tgt = np.exp(np.linspace(np.log(0.2), np.log(5.0), G))
        self.idx = np.array([np.argmin(np.abs(np.log(nodes) - np.log(t))) for t in tgt]); self.logpi = np.full(G, -np.log(G))
    def weights(self, LL):
        for _ in range(3):
            lr = self.logpi[None] + LL[:, self.idx]; ml = lse(lr, 1); R = np.exp(lr - ml[:, None])
            Nk = R.sum(0) + 1e-12; self.logpi = np.log(Nk / Nk.sum())
            self.idx = np.argmax(R.T @ LL, axis=1)
        lr = self.logpi[None] + LL[:, self.idx]; ml = lse(lr, 1); R = np.exp(lr - ml[:, None]); self.R = R
        W = np.zeros_like(LL)
        for g in range(self.G): W[:, self.idx[g]] += R[:, g]
        return W, float(ml.sum())
    def predictive(self, w):
        p = np.zeros(len(self.nodes))
        for g in range(self.G): p[self.idx[g]] += np.exp(self.logpi[g])
        return p

# ---------------- EM runner ----------------
def run(model, nodes, trajs, seed, epochs=100, lr=0.3, op='vanilla', project=True, theta0=None):
    set_grid(nodes); rng = np.random.default_rng(seed)
    th = rng.normal(0, 0.5, 4) if theta0 is None else np.array(theta0, float)
    if project: th[0] = max(th[0], 0.0)
    qc = E.QCache(); adam = E.Adam(4, lr); objs, confs, viol = [], [], 0
    for ep in range(epochs):
        qc.refresh(th); LL = qc.loglik_matrix(trajs); w, obj = model.weights(LL)
        if objs and obj < objs[-1] - 1e-6: viol += 1
        G = E.per_traj_grads(trajs, w, th, qc); confs.append(E.count_conflicts(G))
        d = E.SURGERIES[op](G, rng); new = adam.step(th, d)
        if project: new[0] = max(new[0], 0.0)
        objs.append(obj); st = np.linalg.norm(new - th); th = new
        if st < 1e-3 and ep > 10: break
    qc.refresh(th); LL = qc.loglik_matrix(trajs); w, obj = model.weights(LL)
    return dict(theta=th, w=w, obj=obj, objs=objs + [obj], epochs=ep + 1, conflicts=confs, viol=viol, qc=qc)

# ---------------- metrics ----------------
def policy_value(theta_eval, beta_eval, pi):
    """Mean-variance (immediate-risk) value of policy pi from START under reward theta_eval and risk beta_eval.
    beta_eval = 1 gives the expected discounted return under the true reward."""
    b = mean_R(theta_eval) - 0.5 * (beta_eval - 1.0) * var_R(theta_eval)
    cont = np.ones(NS); cont[GOAL_S] = 0.0
    Ppi = np.einsum('sa,sap->sp', pi, PT) * cont[None, :]       # absorbing goal: no continuation after entering it
    bpi = (pi * b).sum(1)
    V = np.linalg.solve(np.eye(NS) - GAMMA * Ppi * 1.0, bpi)
    return V[sid(*START)]

def greedy(Q):
    pi = np.zeros_like(Q); pi[np.arange(len(Q)), Q.argmax(1)] = 1.0; return pi

def regrets(theta_hat):
    """Per risk group: regret of the greedy policy planned with theta_hat and the true beta,
    evaluated (a) under the true risk-sensitive objective and (b) as expected return under the true reward."""
    out = {}
    for b in BETAS_TRUE:
        pi_star = greedy(value_iteration(THETA_TRUE, b, tol=1e-8, max_iter=3000))
        pi_hat = greedy(value_iteration(theta_hat, b, tol=1e-8, max_iter=3000))
        out[b] = (policy_value(THETA_TRUE, b, pi_star) - policy_value(THETA_TRUE, b, pi_hat),
                  policy_value(THETA_TRUE, 1.0, pi_star) - policy_value(THETA_TRUE, 1.0, pi_hat))
    return out

def summarize(res, nodes, lab, model, heldout=None):
    w = res['w']; th = res['theta']
    if isinstance(model, Mixture): bhat = model.R @ nodes[model.idx]
    else: bhat = w @ nodes
    r = pearsonr(bhat, lab)[0] if np.std(bhat) > 1e-9 else np.nan
    rho = spearmanr(bhat, lab)[0] if np.std(bhat) > 1e-9 else np.nan
    mae = {b: float(np.abs(bhat[lab == b] - b).mean()) for b in np.unique(lab)}
    cw = np.cumsum(w, 1); lo = nodes[np.argmax(cw >= 0.05, 1)]; hi = nodes[np.argmax(cw >= 0.95, 1)]
    cover = float(np.mean((lab >= lo) & (lab <= hi)))
    dth = th - THETA_TRUE; resid = dth - (dth @ SHAPING_DIR) * SHAPING_DIR
    rg = regrets(th)
    out = dict(r=float(r), spearman=float(rho), mae=mae, worst_mae=max(mae.values()), cover90=cover,
               theta=th.round(4).tolist(), mse=float((dth ** 2).mean()), mse_mod_shaping=float((resid ** 2).mean()),
               regret_rs=float(np.mean([v[0] for v in rg.values()])), regret_ret=float(np.mean([v[1] for v in rg.values()])),
               regret_rs_max=float(max(v[0] for v in rg.values())), obj=res['obj'], epochs=res['epochs'], viol=res['viol'],
               conf_mean=float(np.mean(res['conflicts'])))
    if heldout is not None:
        ht, _ = heldout; LLh = res['qc'].loglik_matrix(ht); steps = sum(len(t) for t in ht)
        pred = model.predictive(w); pred = pred / pred.sum()
        with np.errstate(divide='ignore'):
            lp = np.log(np.maximum(pred, 0))
        out['heldout_post_pred'] = float(lse(lp[None] + LLh, 1).sum() / steps)      # common rule: learned beta distribution
        if hasattr(model, 'lp'):
            out['heldout_prior_pred'] = float(lse(model.lp[None] + LLh, 1).sum() / steps)
    return out
