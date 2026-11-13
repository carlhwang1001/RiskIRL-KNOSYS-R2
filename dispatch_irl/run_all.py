# -*- coding: utf-8 -*-
"""產線派工案例：從模擬的 MES 派工紀錄走完 Phase A 與 Phase B。
執行：python3 run_all.py   （約 3 到 5 分鐘；結果寫入 out/）
"""
import os, json, csv, time, numpy as np
from scipy.stats import pearsonr
import mes_sim as M, riskmdp as RM, irl

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out'); os.makedirs(OUT, exist_ok=True)
THETA_TRUE = np.array([10.0, -2.0, -3.0, -0.3])       # 模擬用的真實獎勵：[班次完成, 停機成本, 重工, 每次派工]（實務上未知）
BETAS = [0.3, 0.7, 1.5, 2.5]                           # 四種派工員的風險係數（實務上未知）
FEAT = ['班次完成', '停機成本', '重工', '每次派工']
R = {}

def policy_sampler(pi):
    return lambda s, m, t, loc, rng: int(rng.choice(3, p=pi[s]))
def greedy(Q):
    return lambda s, m, t, loc, rng: int(np.argmax(Q[s]))
def write_csv(rows, fn):
    with open(os.path.join(OUT, fn), 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
def history(model, n, seed, codes='H'):
    pi = np.exp(RM.log_policy(RM.value_iteration(model, THETA_TRUE, BETAS))); rng = np.random.default_rng(seed); logs = []
    for i in range(n): M.run_shift(policy_sampler(pi[i % 4]), rng, log=logs, shift_code='%s%05d' % (codes, i))
    return logs

# ---------------- 0. 真實環境（只在模擬中存在） ----------------
t0 = time.time()
TM, _ = RM.true_model(20000, seed=0)
for k in (5, 6): TM = RM.build_model(RM.estimate(history(TM, 20000, k)))   # 讓環境模型與派工員的行為一致
pi_true = np.exp(RM.log_policy(RM.value_iteration(TM, THETA_TRUE, BETAS)))

# ---------------- 1. 資料：MES 派工紀錄 ----------------
rng = np.random.default_rng(1234); trajs, lab, demo_log = [], [], []
codes = np.random.default_rng(99).permutation(80)                            # 匿名：班次編號打亂，不含派工員身分
for g, b in enumerate(BETAS):
    for j in range(20):
        tr, _ = M.run_shift(policy_sampler(pi_true[g]), rng, log=demo_log, shift_code='S%03d' % codes[g * 20 + j]); trajs.append(tr); lab.append(b)
lab = np.array(lab)
held, held_lab = [], []
rng_h = np.random.default_rng(777)
for g, b in enumerate(BETAS):
    for j in range(10): tr, _ = M.run_shift(policy_sampler(pi_true[g]), rng_h); held.append(tr); held_lab.append(b)
hist_log = history(TM, 1000, 77)                                              # 另外一份歷史紀錄，用來估計環境模型
write_csv(demo_log, 'mes_dispatch_log_demo.csv'); write_csv(hist_log[:3000], 'mes_dispatch_log_history_sample.csv')
R['data'] = dict(n_traj=len(trajs), n_steps=len(demo_log), mean_len=float(np.mean([len(t) for t in trajs])), n_hist_shifts=1000, n_hist_steps=len(hist_log),
                 rule_share={M.RULES[a]: float(np.mean([x['rule'] == a for x in demo_log])) for a in range(3)},
                 rule_share_by_type={str(b): [float(np.mean([a == k for tr, l in zip(trajs, lab) if l == b for (_, a) in tr])) for k in range(3)] for b in BETAS})

# ---------------- 2. 環境模型（實作步驟 3） ----------------
est = RM.estimate(hist_log); EM_ = RM.build_model(est)
est_small = RM.estimate(history(TM, 200, 78)); EM_small = RM.build_model(est_small)
R['env'] = dict(p_fix=est['p'].round(3).tolist(), downtime=est['dm'].round(3).tolist(), count=est['count'].astype(int).tolist(), newq=est['newq'].round(3).tolist())

# ---------------- 3. Phase A：多起點變分 EM ----------------
nodes, lp = irl.make_grid()
def elbo_at(model, th, tr):
    return irl.e_step(RM.loglik_matrix(RM.log_policy(RM.value_iteration(model, th, nodes)), tr), lp, nodes)
runs = []
for s in range(5):
    r = irl.run_em(EM_, trajs, s, op='cagrad', epochs=100, fixed={3: -0.3}); runs.append(r)
    print('start', s, np.round(r['theta'], 2), 'ELBO %.1f' % r['elbo'][-1], 'r %.3f' % pearsonr(r['bhat'], lab)[0], flush=True)
keep, best = irl.screen(runs); bestrun = max(keep, key=lambda r: r['elbo'][-1]); th_hat = bestrun['theta']
w, bhat, sig2, ml = elbo_at(EM_, th_hat, trajs)
_, bh_or, _, ml_or = elbo_at(EM_, THETA_TRUE, trajs)
R['phaseA'] = dict(runs=[dict(seed=r['seed'], curve=r['elbo'], theta=r['theta'].round(3).tolist(), elbo=round(r['elbo'][-1], 1), r=float(pearsonr(r['bhat'], lab)[0]), epochs=r['epochs'],
                              kept=bool(any(r is k for k in keep))) for r in runs],
                   theta_hat=th_hat.round(3).tolist(), elbo=ml, r=float(pearsonr(bhat, lab)[0]), r_oracle=float(pearsonr(bh_or, lab)[0]), elbo_at_true=ml_or,
                   group_mean={str(b): float(bhat[lab == b].mean()) for b in BETAS}, group_mae={str(b): float(np.abs(bhat[lab == b] - b).mean()) for b in BETAS},
                   sig_mean=float(np.sqrt(sig2).mean()), elbo_curve=bestrun['elbo'], conflicts=bestrun['conflicts'])
# 小樣本環境模型的對照（只估 200 班）
rs = irl.run_em(EM_small, trajs, 0, op='cagrad', epochs=100, fixed={3: -0.3})
R['phaseA_small_env'] = dict(theta=rs['theta'].round(3).tolist(), r=float(pearsonr(rs['bhat'], lab)[0]))
# 同質基準：所有派工員都當成 β = 1
hom = irl.run_em(EM_, trajs, 0, op='mean', epochs=100, K=1, lo=1.0, hi=1.0, fixed={3: -0.3})
def heldout(model, th, K, lo, hi):
    nd, lpp = irl.make_grid(K, lo, hi); LL = RM.loglik_matrix(RM.log_policy(RM.value_iteration(model, th, nd)), held)
    return float(irl.lse(lpp[None] + LL, 1).sum() / sum(len(t) for t in held))
R['baseline'] = dict(theta_hom=hom['theta'].round(3).tolist(), heldout_ours=heldout(EM_, th_hat, 64, 0.02, 10.0), heldout_hom=heldout(EM_, hom['theta'], 1, 1.0, 1.0),
                     heldout_true=heldout(EM_, THETA_TRUE, 64, 0.02, 10.0))
np.save(os.path.join(OUT, 'bhat.npy'), np.c_[lab, bhat, np.sqrt(sig2)])

# ---------------- 4. Phase B：重建獎勵、產生並評估新派工策略 ----------------
# 4a. 式 (14)：r̂(s,a) = θ̂ᵀ φ(s,a)；式 (15)：只到正向縮放與平移，比較時以「班次完成」權重正規化
R['phaseB'] = dict(theta_hat_norm=(th_hat / th_hat[0]).round(3).tolist(), theta_true_norm=(THETA_TRUE / THETA_TRUE[0]).round(3).tolist())
# 4b. 風險指標：β̂ 的分布
R['phaseB']['bhat_quantiles'] = np.quantile(bhat, [0.1, 0.25, 0.5, 0.75, 0.9]).round(3).tolist()
R['phaseB']['share_seeking'] = float((bhat < 0.8).mean()); R['phaseB']['share_averse'] = float((bhat > 1.25).mean())
# 4c. 以 r̂ 與指定 β 代回式 (1)(2) 產生新策略（部署時取最大值，即 λ → ∞）
targets = [0.5, 1.0, 2.0]
Qhat = RM.value_iteration(EM_, th_hat, targets); Qtrue = RM.value_iteration(TM, THETA_TRUE, [1.0])
R['phaseB']['rule_table'] = {str(b): [[int(np.argmax(Qhat[i][M.state_id(r, q)])) for q in range(M.N_Q)] for r in range(1, M.N_BREAK + 1)] for i, b in enumerate(targets)}
# 4d. 評估：同一批 2000 個班次（共同隨機數）上比較 KPI 與式 (16) 的 J(π; r*)
ev_rng = np.random.default_rng(2024); shifts = [M.new_shift(ev_rng) for _ in range(2000)]
disc = RM.GAMMA ** np.arange(M.MAX_STEPS)
def kpis(feats_list):
    J = np.array([disc[:len(f)] @ (f @ THETA_TRUE) for f in feats_list])
    cost = np.array([f[:, 1].sum() * M.COST_UNIT for f in feats_list]); rw = np.array([f[:, 2].sum() for f in feats_list])
    done = np.array([f[:, 0].sum() > 0 for f in feats_list]); n = np.array([len(f) for f in feats_list])
    return dict(J=float(J.mean()), J_se=float(J.std(ddof=1) / np.sqrt(len(J))), cost=float(cost.mean()), cost_sd=float(cost.std(ddof=1)),
                cost_p95=float(np.quantile(cost, 0.95)), rework=float(rw.mean()), done=float(done.mean()), steps=float(n.mean()))
def evaluate(fn, seed):
    rr = np.random.default_rng(seed); return kpis([M.run_shift(fn, rr, shift=sh)[1] for sh in shifts])
pols = {}
for i, b in enumerate(targets): pols['IRL 策略 β=%.1f' % b] = greedy(Qhat[i])
pols['真實獎勵策略 β=1（參考上限）'] = greedy(Qtrue[0])
pols['派工員（示範者混合）'] = None
pols['永遠成本最低'] = lambda s, m, t, loc, rng: 2
pols['永遠技能優先'] = lambda s, m, t, loc, rng: 1
res = {}
for k, fn in pols.items():
    if fn is None:
        rr = np.random.default_rng(11); res[k] = kpis([M.run_shift(policy_sampler(pi_true[i % 4]), rr, shift=sh)[1] for i, sh in enumerate(shifts)])
    else: res[k] = evaluate(fn, 11)
rr = np.random.default_rng(11); res['匈牙利派工（dispatcher.docx）'] = kpis([M.hungarian_shift(rr, shift=sh) for sh in shifts])
R['phaseB']['kpi'] = res
R['secs'] = round(time.time() - t0, 1)
json.dump(R, open(os.path.join(OUT, 'results.json'), 'w'), ensure_ascii=False, indent=1, default=float)
print(json.dumps({k: R[k] for k in ('data', 'baseline', 'phaseA_small_env')}, ensure_ascii=False, default=float))
print(json.dumps({k: v for k, v in R['phaseA'].items() if k not in ('elbo_curve', 'conflicts')}, ensure_ascii=False, default=float))
print(json.dumps({k: v for k, v in R['phaseB'].items()}, ensure_ascii=False, default=float, indent=0))
