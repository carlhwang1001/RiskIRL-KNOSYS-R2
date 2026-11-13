# dispatch_irl：產線派工的風險敏感逆強化學習範例

對應《風險敏感逆強化學習：產學合作專題導讀與實作指引》第八節與附錄 B。

## 執行

```
pip install numpy scipy matplotlib
python3 run_all.py
```

約兩分鐘；結果寫入 `out/`（`results.json` 為導讀第八節的所有數字）。

## 檔案

| 檔案 | 內容 | 對應論文 |
|---|---|---|
| mes_sim.py | dispatcher.docx 的參數、本專題假設（REQ_SKILL、P_FIX）、三種派工規則、情境類別、班次模擬、MES 紀錄、匈牙利派工基準 | 實作步驟 1、3、4 |
| riskmdp.py | 風險敏感表格 MDP：值迭代、選擇機率、軌跡似然；由紀錄估計環境模型 | 式 (1)(2)(3) |
| irl.py | 候選格點、E 步、逐軌跡梯度、PCGrad、CAGrad、Adam、正目標投影與錨定、多起點篩選 | 式 (4)–(13) |
| run_all.py | 端到端：資料、環境模型、Phase A、同質基準、Phase B（規則表與 KPI） | 式 (14)–(16) |

## 換成真實資料

1. 在 mes_sim.py 換上工廠的 TaskSkillCost、移動成本矩陣、工程師技能，並以 MES 首次修復率取代 REQ_SKILL 與 P_FIX。
2. 把 MES 匯出的紀錄整理成與 `out/mes_dispatch_log_demo.csv` 相同的欄位（shift 用隨機碼）。
3. 和資深派工員確認「規則」與「情境類別」的定義。
4. 把已知成本寫進 `run_em(..., fixed={分量: 值})` 錨定。

注意：真實獎勵 THETA_TRUE 與四種派工員的 BETAS 只在模擬中存在，用來產生資料與最後評分；方法本身看不到它們。
