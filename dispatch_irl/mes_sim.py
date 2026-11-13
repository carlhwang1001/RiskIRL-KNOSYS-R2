# -*- coding: utf-8 -*-
"""產線派工模擬器（MES 派工紀錄產生器）。
參數取自學生報告 dispatcher.docx：10 種問題、20 位工程師、4 個技能等級、12 個機組（13 個地點含 Office）、
TaskSkillCost、MachineSetEngrLocationCost、EngrSkill、初始工程師位置。
本檔另外加入兩項「使用者需自行定義」的假設（實務上應由 MES 的首次修復率統計取代）：
  REQ_SKILL[t]  : 可靠修好問題 t 所需的技能等級
  P_FIX[t][l]   : 技能等級 l 的工程師一次修好問題 t 的機率
"""
import numpy as np

# ---------------- dispatcher.docx 的參數（原樣照抄） ----------------
TASK_SKILL_COST = np.array([
    [68, 43, 71, 0], [97, 97, 18, 61], [10, 95, 65, 69], [40, 69, 19, 79], [62, 92, 20, 36],
    [16, 96, 53, 3], [42, 73, 96, 75], [34, 60, 54, 27], [11, 95, 10, 5], [87, 32, 36, 69]], float)   # [問題, 技能等級]
LOC_COST = np.array([                                   # [發生問題的機組, 工程師所在地]；最後一欄為 Office
    [0, 21, 34, 19, 45, 24, 1, 38, 18, 22, 0, 17, 16], [42, 0, 32, 3, 12, 37, 21, 36, 14, 0, 15, 38, 41],
    [11, 41, 0, 23, 32, 33, 26, 29, 19, 45, 4, 14, 40], [25, 28, 17, 0, 4, 11, 19, 34, 31, 44, 46, 47, 43],
    [28, 47, 20, 9, 0, 39, 40, 10, 6, 32, 4, 30, 18], [34, 21, 40, 2, 9, 0, 25, 22, 0, 39, 4, 1, 22],
    [38, 47, 13, 18, 32, 28, 0, 15, 15, 46, 27, 13, 9], [35, 9, 17, 38, 20, 20, 39, 0, 9, 0, 6, 32, 5],
    [43, 13, 33, 1, 3, 34, 8, 41, 0, 48, 22, 35, 49], [45, 34, 38, 9, 15, 46, 22, 14, 27, 0, 13, 40, 7],
    [44, 49, 3, 31, 3, 26, 28, 23, 5, 9, 0, 3, 3], [0, 48, 29, 2, 32, 23, 22, 46, 34, 16, 28, 0, 42]], float)
ENGR_SKILL = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3])
INIT_LOC = np.array([9, 6, 7, 0, 9, 3, 12, 0, 9, 9, 6, 8, 9, 8, 10, 7, 2, 4, 10, 3])
N_TASK, N_ENGR, N_SKILL, N_MSET = 10, 20, 4, 12; OFFICE = 12

# ---------------- 本專題加入的假設（需以 MES 統計取代） ----------------
REQ_SKILL = np.array([3, 3, 2, 3, 3, 3, 2, 3, 3, 2])
P_FIX = np.clip(0.95 - 0.25 * np.maximum(REQ_SKILL[:, None] - np.arange(N_SKILL)[None, :], 0), 0.2, 0.95)   # [問題, 技能]
N_BREAK = 10          # 每班（4 小時）待處理的故障數，對應 dispatcher.docx 每班約 10 到 12 個 OnlineTask
MAX_STEPS = 30        # 一班內最多派工次數（含重工）
COST_UNIT = 100.0     # 停機成本特徵 = (移動成本 + 修復成本) / 100
EXTRA = 30.0          # 改派技能優先人選的額外成本門檻（原始成本單位）

RULES = ['就近派工', '技能優先', '成本最低']     # 動作 a = 0, 1, 2
N_Q = 6                                          # 情境類別數

def pick(rule, m, t, loc):
    """依派工規則選出工程師編號。"""
    travel = LOC_COST[m, loc]
    if rule == 0: return int(np.argmin(travel))
    if rule == 1:
        ok = np.where(ENGR_SKILL >= REQ_SKILL[t])[0]; return int(ok[np.argmin(travel[ok])])
    return int(np.argmin(TASK_SKILL_COST[t, ENGR_SKILL] + travel))

def situation(m, t, loc):
    """情境類別 q（0..5）＝「成本最低」人選的技能缺口（0、1、2 以上）× 改派「技能優先」人選要多花的成本是否偏高。"""
    e2, e1 = pick(2, m, t, loc), pick(1, m, t, loc)
    gap = min(max(REQ_SKILL[t] - ENGR_SKILL[e2], 0), 2)
    c = lambda e: TASK_SKILL_COST[t, ENGR_SKILL[e]] + LOC_COST[m, loc[e]]
    extra = int(c(e1) - c(e2) > EXTRA)
    return gap * 2 + extra

def state_id(r, q): return (r - 1) * N_Q + q          # r = 尚未修好的故障數（1..10）
GOAL_S = N_BREAK * N_Q; NS = GOAL_S + 1; NA = 3

def new_shift(rng):
    ms = rng.choice(N_MSET, N_BREAK, replace=False); ts = rng.integers(0, N_TASK, N_BREAK)
    loc = rng.integers(0, N_MSET + 1, N_ENGR)       # 班次開始時工程師散在各機組或 Office
    return list(zip(ms.tolist(), ts.tolist())), loc

def run_shift(policy_fn, rng, shift=None, log=None, shift_code=None):
    """執行一班。policy_fn(s, m, t, loc, rng) 回傳派工規則 a。
    回傳 (s,a) 軌跡與每步的特徵 e = [完成, 停機成本, 重工, 每步]；log 不為 None 時寫入 MES 紀錄。"""
    jobs, loc = shift if shift is not None else new_shift(rng); loc = loc.copy(); k = 0; traj, feats = [], []
    for step in range(MAX_STEPS):
        m, t = jobs[k]; r = N_BREAK - k; q = situation(m, t, loc); s = state_id(r, q)
        a = policy_fn(s, m, t, loc, rng); e_id = pick(a, m, t, loc)
        travel, repair = LOC_COST[m, loc[e_id]], TASK_SKILL_COST[t, ENGR_SKILL[e_id]]
        fixed = rng.random() < P_FIX[t, ENGR_SKILL[e_id]]; loc[e_id] = m
        done = fixed and k == N_BREAK - 1
        traj.append((s, a)); feats.append([float(done), (travel + repair) / COST_UNIT, float(not fixed), 1.0])
        if log is not None:
            log.append(dict(shift=shift_code, seq=step + 1, machine_set=m, task=t, remaining=r, situation=q, rule=a,
                            engineer_skill=int(ENGR_SKILL[e_id]), travel_cost=travel, repair_cost=repair, fixed=int(fixed)))
        if fixed: k += 1
        if k == N_BREAK: break
    return traj, np.array(feats)

def hungarian_shift(rng, shift=None):
    """dispatcher.docx 的做法：班次開始時以匈牙利演算法指派（成本 = TaskSkillCost + 移動成本），
    每位工程師最多一件；修不好時由同一位工程師重修。回傳特徵序列（供 KPI 比較）。"""
    from scipy.optimize import linear_sum_assignment
    jobs, loc = shift if shift is not None else new_shift(rng); loc = loc.copy()
    C = np.array([[TASK_SKILL_COST[t, ENGR_SKILL[e]] + LOC_COST[m, loc[e]] for (m, t) in jobs] for e in range(N_ENGR)])
    rows, cols = linear_sum_assignment(C); who = {c: r_ for r_, c in zip(rows, cols)}
    feats, k, step = [], 0, 0
    while k < N_BREAK and step < MAX_STEPS:
        m, t = jobs[k]; e_id = who[k]
        travel, repair = LOC_COST[m, loc[e_id]], TASK_SKILL_COST[t, ENGR_SKILL[e_id]]
        fixed = rng.random() < P_FIX[t, ENGR_SKILL[e_id]]; loc[e_id] = m
        feats.append([float(fixed and k == N_BREAK - 1), (travel + repair) / COST_UNIT, float(not fixed), 1.0])
        if fixed: k += 1
        step += 1
    return np.array(feats)
