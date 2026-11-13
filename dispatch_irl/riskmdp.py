# -*- coding: utf-8 -*-
"""通用的風險敏感表格 MDP（論文式 (1)–(3)）與派工環境模型的估計。
環境模型以三個陣列表示：
  P[s,a,s']    轉移機率
  MU[s,a,:]    一步特徵向量 e 的平均值（獎勵 R = θᵀe，故 E[R] = θᵀ MU）
  SIG[s,a,:,:] 一步特徵向量 e 的共變異數（Var[R] = θᵀ SIG θ）
"""
import numpy as np
import mes_sim as M

GAMMA = 0.95; LAMBDA = 3.0; R0 = 1.0          # 折扣、理性常數 λ、獎勵尺度 R0（式 (1)）

class Model:
    def __init__(self, P, MU, SIG, goal):
        self.P, self.MU, self.SIG, self.goal = P, MU, SIG, goal; self.NS, self.NA = P.shape[:2]
        self.cont = np.ones(self.NS); self.cont[goal] = 0.0
    def mean_var(self, theta):
        return self.MU @ theta, np.einsum('saij,i,j->sa', self.SIG, theta, theta)

def value_iteration(model, theta, betas, Q0=None, tol=1e-6, max_iter=2000):
    """式 (1)：Q_β(s,a) = E[R] − ½(β−1)Var[R]/R0 + γ E[V(s')]，V(s) = max_a Q_β(s,a)。betas 可一次給多個（向量化）。"""
    betas = np.atleast_1d(betas).astype(float); ER, VR = model.mean_var(theta)
    base = ER[None] - 0.5 * (betas[:, None, None] - 1.0) * VR[None] / R0
    Q = np.zeros_like(base) if Q0 is None else Q0.copy()
    PT = model.P.reshape(model.NS * model.NA, model.NS).T
    for _ in range(max_iter):
        V = Q.max(axis=2) * model.cont[None]
        Qn = base + GAMMA * (V @ PT).reshape(base.shape)
        if np.max(np.abs(Qn - Q)) < tol: return Qn
        Q = Qn
    return Q

def log_policy(Q, lam=LAMBDA):
    """式 (2)：π(a|s) = exp(λQ) / Σ exp(λQ)，回傳 log π。"""
    z = lam * (Q - Q.max(axis=-1, keepdims=True)); return z - np.log(np.exp(z).sum(axis=-1, keepdims=True))

def loglik_matrix(logpi, trajs):
    """式 (3)：log P(τ|θ,β) = Σ_t log π(a_t|s_t)（轉移機率與 θ 無關，視為常數）。回傳 (N, K)。"""
    out = np.zeros((len(trajs), logpi.shape[0]))
    for i, tr in enumerate(trajs):
        S = np.array([s for s, a in tr]); A = np.array([a for s, a in tr]); out[i] = logpi[:, S, A].sum(axis=1)
    return out

# ---------------- 由 MES 紀錄估計環境模型（實作步驟 3） ----------------
def estimate(logs, min_count=5):
    """因子化估計：p_fix(q,a)、停機成本平均與變異數 (q,a)、成功後新故障的情境分布、失敗後的情境分布 (q,a)。
    樣本少於 min_count 的格子退回同一動作的整體估計。"""
    NQ, NA = M.N_Q, M.NA
    cnt = np.zeros((NQ, NA)); fx = np.zeros((NQ, NA)); d1 = np.zeros((NQ, NA)); d2 = np.zeros((NQ, NA))
    newq = np.ones(NQ) * 0.5; failq = np.ones((NQ, NA, NQ)) * 0.5 / NQ
    rows = sorted(logs, key=lambda x: (x['shift'], x['seq']))
    for x, y in zip(rows, rows[1:] + [None]):
        q, a = x['situation'], x['rule']; d = (x['travel_cost'] + x['repair_cost']) / M.COST_UNIT
        cnt[q, a] += 1; fx[q, a] += x['fixed']; d1[q, a] += d; d2[q, a] += d * d
        if y is not None and y['shift'] == x['shift']:
            if x['fixed']: newq[y['situation']] += 1
            else: failq[q, a, y['situation']] += 1
    tot = cnt.sum(0); pa = fx.sum(0) / np.maximum(tot, 1); da = d1.sum(0) / np.maximum(tot, 1); va = d2.sum(0) / np.maximum(tot, 1) - da ** 2
    ok = cnt >= min_count
    p = np.where(ok, fx / np.maximum(cnt, 1), pa[None]); dm = np.where(ok, d1 / np.maximum(cnt, 1), da[None])
    dv = np.maximum(np.where(ok, d2 / np.maximum(cnt, 1) - dm ** 2, va[None]), 0.0)
    return dict(p=p, dm=dm, dv=dv, newq=newq / newq.sum(), failq=failq / failq.sum(axis=2, keepdims=True), count=cnt)

def build_model(est):
    """由因子化估計組出 P、MU、SIG。特徵 e = [完成, 停機成本, 重工, 每步]。"""
    NS, NA, NQ = M.NS, M.NA, M.N_Q; d = 4
    P = np.zeros((NS, NA, NS)); MU = np.zeros((NS, NA, d)); SIG = np.zeros((NS, NA, d, d))
    P[M.GOAL_S, :, M.GOAL_S] = 1.0
    for r in range(1, M.N_BREAK + 1):
        for q in range(NQ):
            s = M.state_id(r, q)
            for a in range(NA):
                p = est['p'][q, a]; last = float(r == 1); v = p * (1 - p)
                if r == 1: P[s, a, M.GOAL_S] += p
                else: P[s, a, [M.state_id(r - 1, q2) for q2 in range(NQ)]] += p * est['newq']
                P[s, a, [M.state_id(r, q2) for q2 in range(NQ)]] += (1 - p) * est['failq'][q, a]
                MU[s, a] = [p * last, est['dm'][q, a], 1 - p, 1.0]
                S_ = np.zeros((d, d)); S_[0, 0] = v * last; S_[2, 2] = v; S_[0, 2] = S_[2, 0] = -v * last; S_[1, 1] = est['dv'][q, a]
                SIG[s, a] = S_
    return Model(P, MU, SIG, M.GOAL_S)

def true_model(n_shifts=20000, seed=0):
    """「真實」環境模型：以大量隨機派工的模擬估計（實務上未知，只用來產生示範與做最終評估）。"""
    rng = np.random.default_rng(seed); logs = []
    for i in range(n_shifts): M.run_shift(lambda s, m, t, loc, rng: int(rng.integers(0, 3)), rng, log=logs, shift_code=i)
    return build_model(estimate(logs)), estimate(logs)
