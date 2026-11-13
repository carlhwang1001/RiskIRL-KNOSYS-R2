# -*- coding: utf-8 -*-
"""Phase A：變分 EM（論文式 (4)–(13)）。
E 步：K 個對數等距候選 β 上的離散後驗（式 (7)），並報告摘要（式 (8)）。
M 步：逐軌跡梯度（式 (12)，以中央差分計算），梯度衝突消解（式 (9) PCGrad、式 (10) CAGrad），
      彙總後以 Adam 更新（式 (13)），再做正目標投影 θ_goal ≥ 0。
"""
import numpy as np
from riskmdp import value_iteration, log_policy, loglik_matrix

def make_grid(K=64, lo=0.02, hi=10.0, mu0=0.0, sig0=0.75):
    nodes = np.exp(np.linspace(np.log(lo), np.log(hi), K))
    lp = -0.5 * ((np.log(nodes) - mu0) / sig0) ** 2; lp -= lp.max(); lp -= np.log(np.exp(lp).sum())
    return nodes, lp                                   # 候選值 β⁽ⁿ⁾ 與先驗權重 log p_n

def lse(x, axis): m = x.max(axis=axis, keepdims=True); return (m + np.log(np.exp(x - m).sum(axis=axis, keepdims=True))).squeeze(axis)

def e_step(LL, logprior, nodes):
    """式 (7)：w_in ∝ p_n · P(τ_i | β⁽ⁿ⁾, θ)；式 (8)：β̂_i = Σ w_in β⁽ⁿ⁾，σ_i² = Σ w_in (β⁽ⁿ⁾ − β̂_i)²。
    回傳權重 w、β̂、σ²，以及格點上的邊際對數似然（等於 E 步後的 ELBO，式 (5)(6)）。"""
    lw = logprior[None] + LL; ml = lse(lw, 1); w = np.exp(lw - ml[:, None])
    bhat = w @ nodes; sig2 = (w * (nodes[None] - bhat[:, None]) ** 2).sum(1)
    return w, bhat, sig2, float(ml.sum())

def per_traj_grads(model, trajs, w, theta, nodes, eps=0.02):
    """式 (12) 的逐軌跡梯度 g_i = ∂/∂θ Σ_n w_in log P(τ_i | θ, β⁽ⁿ⁾)，以中央差分計算（w 固定，式 (11) 的 M 步目標）。"""
    d = len(theta); G = np.zeros((len(trajs), d))
    for k in range(d):
        f = []
        for sg in (1.0, -1.0):
            th = theta.copy(); th[k] += sg * eps
            f.append((w * loglik_matrix(log_policy(value_iteration(model, th, nodes)), trajs)).sum(1))
        G[:, k] = (f[0] - f[1]) / (2 * eps)
    return G

def agg_mean(G, rng=None): return G.mean(0)

def agg_pcgrad(G, rng):
    """式 (9)：若 g_i·g_j < 0，把 g_i 投影到 g_j 的法平面上。"""
    Gp = G.copy()
    for i in range(len(G)):
        for j in rng.permutation(len(G)):
            if j != i and Gp[i] @ G[j] < 0: Gp[i] = Gp[i] - (Gp[i] @ G[j]) / (G[j] @ G[j] + 1e-12) * G[j]
    return Gp.mean(0)

def agg_cagrad(G, rng=None, c=0.5, iters=200):
    """式 (10)：w⋆ = argmin_{w∈Δ} g_wᵀg₀ + c‖g₀‖‖g_w‖，d⋆ = g₀ + c‖g₀‖/‖g_w⋆‖ · g_w⋆。"""
    N = len(G); Mx = G @ G.T; g0 = G.mean(0); n0 = np.linalg.norm(g0) + 1e-12; m0 = Mx.mean(1); w = np.ones(N) / N
    def proj(v):
        u = np.sort(v)[::-1]; cs = np.cumsum(u); rho = np.nonzero(u * np.arange(1, N + 1) > cs - 1)[0][-1]; return np.maximum(v - (cs[rho] - 1) / (rho + 1), 0)
    for _ in range(iters):
        Mw = Mx @ w; nw = np.sqrt(max(w @ Mw, 1e-12)); g = m0 + c * n0 * Mw / nw; w = proj(w - 0.5 * g / (np.linalg.norm(g) + 1e-12))
    gw = G.T @ w; return g0 + c * n0 / (np.linalg.norm(gw) + 1e-12) * gw

OPS = {'mean': agg_mean, 'pcgrad': agg_pcgrad, 'cagrad': agg_cagrad}

class Adam:
    def __init__(self, d, lr): self.m = np.zeros(d); self.v = np.zeros(d); self.t = 0; self.lr = lr
    def step(self, th, g):
        self.t += 1; self.m = 0.9 * self.m + 0.1 * g; self.v = 0.999 * self.v + 0.001 * g * g
        return th + self.lr * (self.m / (1 - 0.9 ** self.t)) / (np.sqrt(self.v / (1 - 0.999 ** self.t)) + 1e-8)

def run_em(model, trajs, seed, op='cagrad', epochs=100, lr=0.3, theta0=None, K=64, lo=0.02, hi=10.0, fixed=None, verbose=False):
    """一次 Phase A（一個起點）。fixed = {分量編號: 固定值}，用來錨定已知的成本（例如每次派工的固定成本）。
    回傳 θ̂、β̂、σ²、每輪 ELBO 與衝突數。"""
    nodes, lp = make_grid(K, lo, hi); rng = np.random.default_rng(seed)
    theta = rng.normal(0, 0.5, 4) if theta0 is None else np.array(theta0, float); theta[0] = max(theta[0], 0.0)
    fixed = fixed or {}
    for k, v in fixed.items(): theta[k] = v
    adam = Adam(len(theta), lr); elbo, conf = [], []
    for ep in range(epochs):
        LL = loglik_matrix(log_policy(value_iteration(model, theta, nodes)), trajs)       # 步驟 A：每條軌跡在每個候選 β 下的對數似然
        w, bhat, s2, ml = e_step(LL, lp, nodes)                                            # 式 (7)(8)
        G = per_traj_grads(model, trajs, w, theta, nodes)                                  # 式 (12)
        conf.append(int(((G @ G.T)[np.triu_indices(len(G), 1)] < 0).sum()))
        for k in fixed: G[:, k] = 0.0                                                      # 錨定的分量不更新
        d = OPS[op](G, rng); new = adam.step(theta, d); new[0] = max(new[0], 0.0)         # 式 (9)/(10)、(13)、正目標投影
        for k, v in fixed.items(): new[k] = v
        elbo.append(ml); step = np.linalg.norm(new - theta); theta = new
        if verbose and ep % 10 == 0: print(ep, round(ml, 1), np.round(theta, 2), flush=True)
        if step < 1e-3 and ep > 10: break
    LL = loglik_matrix(log_policy(value_iteration(model, theta, nodes)), trajs); w, bhat, s2, ml = e_step(LL, lp, nodes)
    return dict(theta=theta, bhat=bhat, sig2=s2, w=w, elbo=elbo + [ml], conflicts=conf, epochs=ep + 1, seed=seed)

def screen(runs, gap=100.0):
    """多起點篩選：最終 ELBO 低於同組最佳值超過 gap 的 run 視為退化。"""
    best = max(r['elbo'][-1] for r in runs); return [r for r in runs if r['elbo'][-1] >= best - gap], best
