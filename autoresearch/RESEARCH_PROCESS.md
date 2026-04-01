# Auto Research 人工研究流程手冊（v2 — 防造假版）

> 防止 p-hacking、參數竄改、事後湊答案。

---

## 🔴 核心原則

1. **參數一寫定，不能改**：每次實驗的參數在跑回測前就寫進 `research_log.md`，**事後嚴禁修改**
2. **邏輯定義必須完整無歧義**：YAML 規格必須包含每一個指標的計算公式，不能用「突破」這樣的模糊詞
3. **QA 拿到的 spec 必須是 log 裡的原文**：不能從記憶/口頭描述推斷
4. **4/4 達標才算成功**：任一項未達就 DISCARD，不准降低標準

---

## 目標參數（Target）

| 指標 | 門檻 |
|------|-------|
| 勝率 (Win Rate) | ≥ 50% |
| 盈虧比 (Profit Factor) | ≥ 2.0 |
| 最大回撤 (Max Drawdown) | ≤ 30% |
| Sharpe Ratio | ≥ 1.5 |

---

## 完整流程（5 步，嚴格順序）

```
STEP 1: 市場分析
  └→ 寫入 research_log.md（分析視角、方向）

STEP 2: 策略設計（team-researcher）
  └→ 每一個條件都要有明確計算公式
  └→ 寫入 research_log.md（實驗開始前，不可事後補）
  └→ 不能用「突破」、「確認」這類模糊詞

STEP 3: 回測驗證（team-researcher）
  └→ 用 backtest_engine.py 跑，參數必須與 STEP 2 完全一致
  └→ 結果寫入 research_log.md（實驗結束後）
  └→ 對比 Target：4/4 才 PASS

    └→ 達標 → 進入 STEP 4
    └→ 未達標 → DISCARD，重新 STEP 1（不得修改參數重跑）

STEP 4: QA 驗證（team-qa）
  └→ researcher 把完整 YAML 規格交給 QA
  └→ QA 獨立實作（不看 researcher 代碼）
  └→ 數據來源相同（btcusdt_1d.parquet）
  └→ 雙方結果對比：
      - 勝率誤差 < 5%
      - 盈虧比誤差 < 10%
      - 一致 → VERIFIED ✅
      - 不一致 → FAIL，需找出原因（可能是 spec 歧義）

STEP 5: 入庫部署（team-ceo 裁決）
  └→ CEO 審核 VERIFIED 結果
  └→ 通過 → 更新 best_strategies.json（狀態：VERIFIED）
  └→ 同步更新 factor_library.json（新策略進 Dashboard）
  └→ 更新 research_log.md（最終記錄）
```

---

## 實驗記錄格式（research_log.md）

每一個實驗都必須即時寫入，格式如下：

```markdown
### 實驗 #N — YYYY-MM-DD HH:MM

#### 分析視角
（市場結構、方向、為什麼這個策略可能有效）

#### 策略邏輯（完整定義）
- 指標 1：公式（例如 RSI(14) = 100 - 100/(1 + RS)，RS = SMA(漲幅,14)/SMA(跌幅,14)）
- 指標 2：公式（例如 MA20 = SMA(close, 20)）
- 進場條件 LONG：
  - C1: RSI(14) >= 50
  - C2: close > SMA(close, 20)
- 進場條件 SHORT：（同上）
- 止盈：5%（固定百分比）
- 止損：3%（固定百分比）
- 最大持有：15 根 K 線

#### 實驗參數（不可事後修改）
- rsi_min: 50
- rsi_max: 75
- vol_mult: 2.0
- tp: 5
- sl: 3
- regime: true

#### 回測結果
- 總交易：N 筆
- 勝率：XX%
- 盈虧比：X.XX
- 最大回撤：XX%
- Sharpe：X.XX
- TP/SL 次數：N/N

#### Target 對比
- 勝率 ≥ 50% → ✅ PASS / ❌ FAIL
- 盈虧比 ≥ 2.0 → ✅ PASS / ❌ FAIL
- 最大回撤 ≤ 30% → ✅ PASS / ❌ FAIL
- Sharpe ≥ 1.5 → ✅ PASS / ❌ FAIL

#### 評估
✅ KEEP → 進入 QA / ❌ DISCARD
```

---

## best_strategies.json 格式

```json
{
  "strategy_id": "Strategy_Name_v1",
  "strategy_name": "策略顯示名稱",
  "focus_area": "策略類型",
  "entry_description": "完整進場邏輯描述",
  "params": {
    "rsi_min": 50,
    "vol_mult": 2.0,
    "tp": 5.0,
    "sl": 3.0
  },
  "metrics": {
    "win_rate": 65.0,
    "profit_factor": 2.5,
    "max_drawdown": 15.0,
    "sharpe": 3.0
  },
  "status": "VERIFIED",
  "verified_at": "2026-04-01T09:30:00",
  "research_log_ref": "research_log.md #實驗編號"
}
```

---

## Dashboard 顯示邏輯

- `match_pct` = 當前市場滿足的條件數 / 總條件數
- 信心度不是回測勝率，是當前信號強度
- `match_pct ≥ 50%` 才顯示
- VERIFIED 狀態表示「該策略歷史回測通過 4 項指標」

---

## 數據檔案位置

| 用途 | 檔案 |
|------|------|
| 歷史 K 線（1D） | `data/btcusdt_1d.parquet` |
| 因子策略庫 | `autoresearch/factor_library.json` |
| 最佳策略庫 | `autoresearch/memory/best_strategies.json` |
| 研究日誌 | `autoresearch/memory/research_log.md` |
| QA 結果 | `autoresearch/memory/qa_*.json` |
| 回測引擎 | `backtest/backtest_engine.py` |

---

## QA 驗證清單（每次都要檢查）

- [ ] researcher 在實驗前就把完整邏輯寫進 research_log.md
- [ ] QA 拿到的 YAML 與 research_log.md 的 spec 完全一致
- [ ] QA 獨立實作，不看 researcher 代碼
- [ ] 雙方誤差在範圍內
- [ ] 4 項指標全部達標
- [ ] CEO 審核通過
- [ ] status 更新為 VERIFIED
- [ ] best_strategies.json + factor_library.json 同步更新

---

## ⚠️ 嚴禁事項

- **不准事後改參數**：發現結果不好就 DISCARD，不准改 threshold 重跑
- **不准模糊詞**：不能寫「價格突破」就停下，要寫「close > SMA(close, 20)」
- **不准湊答案**：不能根據目標反推參數
- **不准補 log**：實驗記錄必須即時寫入，不能事後補寫

---

*Last updated: 2026-04-01 v2*
