"""E5: controlled counterexample for the immediate-risk likelihood (Reviewer 3, point 1 and Q1).
Two-step family indexed by mu. At s0: 'safe' (reward 0, then reward 0) or 'risky' (R0 = m +/- 1 w.p. 1/2, then a forced
second step whose reward is mu*(R0 - m) +/- 1). Every visited (s,a) has the same one-step conditional variance (1 for the
risky branch) for every mu, and E[R1] = 0, so the immediate-risk Q is identical for all mu. Return variance of 'risky' is
(1 + gamma*mu)^2 + gamma^2 and depends on mu.  mu = 0: independent steps; mu = +1: amplifying continuation risk;
mu = -1: hedging continuation (high local variance, low continuation risk)."""
import numpy as np
lam, gam, m = 3.0, 0.9, 0.5
sig = lambda x: 1 / (1 + np.exp(-x))
nodes = np.exp(np.linspace(np.log(0.02), np.log(10), 64)); lp = -0.5 * (np.log(nodes) / 0.75) ** 2; lp -= np.log(np.exp(lp).sum())
dq_ir = lambda b: m - 0.5 * (b - 1.0) * 1.0                           # immediate-risk: one-step variance 1, same for all mu
rv = lambda mu: (1 + gam * mu) ** 2 + gam ** 2                         # return variance of the risky branch
dq_rv = lambda b, mu: m - 0.5 * (b - 1.0) * rv(mu)                     # return-variance (mean-variance of G) expert
rng = np.random.default_rng(0); n = 400
print(f"{'mu':>4} {'RetVar':>7} | {'expert':>14} {'true b':>6} | {'P(risky)':>8} | {'b_hat (post. mean)':>18} {'90% CI':>14} | {'asymptotic b_hat':>16}")
for expert in ['immediate-risk', 'return-variance']:
    for b in [0.5, 2.0]:
        for mu in [-1.0, 0.0, 1.0]:
            p = sig(lam * (dq_ir(b) if expert == 'immediate-risk' else dq_rv(b, mu)))
            k = rng.binomial(n, p)
            LL = k * np.log(sig(lam * dq_ir(nodes))) + (n - k) * np.log(sig(-lam * dq_ir(nodes)))   # immediate-risk likelihood: no mu
            w = np.exp(lp + LL - (lp + LL).max()); w /= w.sum(); cw = np.cumsum(w)
            bhat = w @ nodes; ci = (nodes[np.argmax(cw >= .05)], nodes[np.argmax(cw >= .95)])
            asym = b if expert == 'immediate-risk' else 1 + (b - 1) * rv(mu)
            print(f"{mu:4.0f} {rv(mu):7.2f} | {expert:>14} {b:6.1f} | {p:8.3f} | {bhat:18.2f} {('[%.2f, %.2f]' % ci):>14} | {asym:16.2f}")
