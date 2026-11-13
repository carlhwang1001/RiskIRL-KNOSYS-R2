"""10x10 risk grid-world for Experiment 0' (EM-CAGrad smoke test).

Layout (fixed):
  start (0,0), goal (9,9) absorbing with terminal feature reward.
  8 obstacles (impassable), 4 risk cells (stochastic penalty -4 w.p. 0.4),
  3 mild cells (+0.2 deterministic; kept small so that wall-bumping on a mild cell cannot be farmed). Living cost -0.3 per step. Slip 0.1.

NOTE (deviation from spec, to be reported): risk penalty -8 -> -4 and
mild +3 -> +0.2. Reason: with the original magnitudes, a risk-seeking
agent (beta=0.3) earns a net-positive variance premium per risk-cell
visit (and any agent earns net-positive mild-cell loops), so the optimal
policy degenerates into reward farming and never reaches the goal.
Adjusted values keep all loops net-negative while preserving the
shortcut-vs-detour dilemma.

Reward model:
  R(s,a,s') = theta_goal*1[s'=goal] + X*1[s' in risk] + theta_mild*1[s' in mild] + theta_step
  X in {-4, 0} w.p. {0.4, 0.6}; E[X] = -1.6, Var[X] = 0.4*0.6*16 = 3.84.
  Learner learns mean weights theta (4-dim); Var[X] is a known env property.
"""
import numpy as np

H = W = 10
NS, NA = H * W, 4  # actions: 0=up 1=down 2=left 3=right
SLIP = 0.1
GAMMA = 0.95
MAX_STEPS = 60
VAR_X = 0.4 * 0.6 * 16.0  # 3.84

OBSTACLES = {(2, 2), (2, 3), (3, 6), (5, 1), (6, 6), (7, 3), (8, 5), (4, 4)}
RISK = {(3, 3), (5, 4), (6, 7), (8, 8)}
MILD = {(4, 6), (2, 7), (7, 5)}
START, GOAL = (0, 0), (9, 9)

THETA_TRUE = np.array([10.0, -1.6, 0.2, -0.3])  # goal, risk(mean), mild, step

def sid(r, c): return r * W + c
GOAL_S = sid(*GOAL)

def _valid(r, c):
    return 0 <= r < H and 0 <= c < W and (r, c) not in OBSTACLES

# state indicator features e(s') = (goal, risk, mild, 1)
E_S = np.zeros((NS, 4))
for r in range(H):
    for c in range(W):
        s = sid(r, c)
        E_S[s, 0] = 1.0 if (r, c) == GOAL else 0.0
        E_S[s, 1] = 1.0 if (r, c) in RISK else 0.0
        E_S[s, 2] = 1.0 if (r, c) in MILD else 0.0
        E_S[s, 3] = 1.0

_MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_PERP = {0: (2, 3), 1: (2, 3), 2: (0, 1), 3: (0, 1)}

def build_P():
    """P[s,a,s'] with slip; goal absorbing (self-loop, no reward via features:
    absorption handled by zeroing continuation in the solver)."""
    P = np.zeros((NS, NA, NS))
    for r in range(H):
        for c in range(W):
            s = sid(r, c)
            if (r, c) in OBSTACLES:
                P[s, :, s] = 1.0
                continue
            if (r, c) == GOAL:
                P[s, :, s] = 1.0
                continue
            for a in range(NA):
                outs = [(a, 1.0 - SLIP)] + [(p, SLIP / 2) for p in _PERP[a]]
                for mv, pr in outs:
                    dr, dc = _MOVES[mv]
                    nr, nc = r + dr, c + dc
                    if not _valid(nr, nc):
                        nr, nc = r, c
                    P[s, a, sid(nr, nc)] += pr
    return P

P = build_P()
PHI_SA = np.einsum('sap,pf->saf', P, E_S)  # E_{s'}[e(s')] : (NS,NA,4)

def var_R(theta):
    """Var[R | s,a] = sum p*(v(s')+m(s')^2) - (sum p*m(s'))^2,
    v(s') = VAR_X * 1[risk]."""
    m = E_S @ theta                      # (NS,)
    v = VAR_X * E_S[:, 1]                # (NS,)
    ex = np.einsum('sap,p->sa', P, m)
    ex2 = np.einsum('sap,p->sa', P, v + m * m)
    return np.maximum(ex2 - ex * ex, 0.0)

def mean_R(theta):
    return PHI_SA @ theta                # (NS,NA)
