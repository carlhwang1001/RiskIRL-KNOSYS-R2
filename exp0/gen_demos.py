"""Generate anonymous heterogeneous demonstrations.
4 expert groups beta in {0.3, 0.7, 1.5, 2.5}, 20 rollouts each (N=80).
Rollout follows softmax(Q_beta) policy under slip dynamics.
"""
import numpy as np
from env_riskgrid import P, START, GOAL_S, MAX_STEPS, THETA_TRUE, sid
from solver_riskq import value_iteration, policy_from_Q

BETAS_TRUE = [0.3, 0.7, 1.5, 2.5]
N_PER = 20

def rollout(pi, rng):
    s = sid(*START)
    traj = []
    for _ in range(MAX_STEPS):
        a = rng.choice(4, p=pi[s])
        traj.append((s, a))
        s = rng.choice(P.shape[2], p=P[s, a])
        if s == GOAL_S:
            break
    return traj

def generate(seed=1234):
    rng = np.random.default_rng(seed)
    trajs, beta_labels = [], []
    for b in BETAS_TRUE:
        Q = value_iteration(THETA_TRUE, b)
        pi = policy_from_Q(Q)
        for _ in range(N_PER):
            trajs.append(rollout(pi, rng))
            beta_labels.append(b)
    return trajs, np.array(beta_labels)

if __name__ == '__main__':
    trajs, labels = generate()
    lens = np.array([len(t) for t in trajs])
    from env_riskgrid import RISK, W
    risk_ids = {r * W + c for r, c in RISK}
    hits = np.array([sum(1 for s, a in t if s in risk_ids) for t in trajs])
    reach = np.array([t[-1][0] for t in trajs])
    for i, b in enumerate(BETAS_TRUE):
        m = labels == b
        print(f"beta={b}: mean len={lens[m].mean():.1f}  risk-cell visits/traj={hits[m].mean():.2f}")
