# Auto Research 人工研究流程手冊

> 確保每次研究運作都是可重複、可審計的。

---

## 核心原則

1. **獨立實作**：研究 Agent 與 QA Agent 必須各自獨立實作，不能互相看代碼
2. **數據一致**：雙方使用同一組歷史 K 線數據
3. **結果可重現**：每次實驗的參數、結果、決策全部寫入日誌
4. **4/4 達標才算成功**：勝率、盈虧比、最大回撤、Sharpe 全部達標才入庫

---

## 目標參數（Target）

| 指標 | 門檻 | 說明 |
|------|-------|------|
| 勝率 (Win Rate) | ≥ 50% | 盈利交易筆數 / 總交易筆數 |
| 盈虧比 (Profit Factor) | ≥ 2.0 | 總盈利 / 總虧損 |
| 最大回撤 (Max Drawdown) | ≤ 30% | 從峰值最大跌幅 |
| Sharpe Ratio | ≥ 1.5 | 風險调整後收益 |

---

## 完整流程（5 步）

```
STEP 1: 市場分析
  └→ 找出當前市場結構（多頭/空頭/區間）
  └→ 識別機會方向（哪些策略類型可能有效）

STEP 2: 策略設計（team-researcher）
  └→ 參考 strategy_template_format.yaml 格式
  └→ 寫出完整策略規格（指標、參數、進場/出場邏輯）
  └→ 規格寫入 research_log.md

STEP 3: 回測驗證（team-researcher）
  └→ 用 backtest/backtest_engine.py 跑歷史數據
  └→ 數據來源：data/btcusdt_1d.parquet（1D）
  └→ 也可用 Binance API 即時抓 4H 數據
  └→ 結果對比 Target：4/4 達標？

    └→ 達標 → 進入 STEP 4
    └→ 未達標 → 回到 STEP 2，調整參數重新設計

STEP 4: QA 驗證（team-qa）
  └→ team-qa 收到 YAML 策略規格
  └→ QA 獨立實作回測代碼（不看 researcher 的代碼）
  └→ 雙方結果對比：
      - 一致（誤差 < 5%）→ VERIFIED ✅
      - 不一致 → 回去檢查 YAML 歧義，修復後重新 STEP 3

STEP 5: 入庫與部署（team-ceo 裁決）
  └→ CEO 審核 VERIFIED 結果
  └→ 通過 → 更新 best_strategies.json（狀態：VERIFIED）
  └→ 同步更新 factor_library.json（新策略寫入 Dashboard）
  └→ 更新 research_log.md（最終記錄）
```

---

## 策略描述格式（Strategy Template Format）

詳見 `strategy_template_format.yaml`，核心欄位：

```yaml
StrategyTemplate:
  name: "策略名稱"
  version: "1.0"

  indicator_calculation:
    # 每個指標必須獨立定義，包含 type、price、period、formula
    ma20:
      type: "SMA"
      price: "close"
      period: 20
      formula: "SMA = (C[0] + ... + C[-19]) / 20"

    rsi14:
      type: "RSI"
      price: "close"
      period: 14
      formula: |
        RS = 平均漲幅 / 平均跌幅
        RSI = 100 - (100 / (1 + RS))

  entry_conditions:
    long:
      - condition_name: "golden_cross"
        description: "MA20 上穿 MA50"
      - condition_name: "rsi_above_50"
        description: "RSI > 50"

  exit_conditions:
    take_profit:
      - type: "percentage"
        value: 7.0  # 7%
    stop_loss:
      - type: "percentage"
        value: 3.0  # 3%
    max_holding:
      - type: "bars"
        value: 15

  backtest_config:
    data_range: "2015-2026"
    timeframe: "1D"
```

---

## 數據檔案位置

| 用途 | 檔案 |
|------|------|
| 歷史 K 線（1D） | `data/btcusdt_1d.parquet` |
| 因子策略庫 | `autoresearch/factor_library.json` |
| 最佳策略庫 | `autoresearch/memory/best_strategies.json` |
| 研究日誌 | `autoresearch/memory/research_log.md` |
| 失敗策略記錄 | `autoresearch/memory/failed_strategies.json` |
| 回測引擎 | `backtest/backtest_engine.py` |

---

## 結果記錄格式（research_log.md）

```markdown
### 實驗 #N - YYYY-MM-DD HH:MM

- **策略**: StrategyName
- **進場邏輯**: 具體描述
- **止盈**: X% 或 Nx ATR
- **止損**: X% 或 Nx ATR
- **結果**: WR=XX% | PF=X.XX | DD=XX% | Sharpe=X.XX
- **評估**: ✅ KEEP / ⚠️ PARTIAL / ❌ DISCARD
- **備註**: 未達標原因或關鍵洞察
```

---

## best_strategies.json 格式

```json
{
  "strategy_id": "Regime_MA_Cross_v2",
  "strategy_name": "Regime-Aware MA Cross v2",
  "focus_area": "市場結構/趨勢",
  "round": 2,
  "entry_description": "Golden Cross + RSI>50 + 7日動量>0.5% + 價格>MA200 (多頭)...",
  "params": {
    "stop_loss_pct": 3.0,
    "take_profit_pct": 7.0,
    "max_holding_days": 15,
    "rsi_long_thr": 50,
    "rsi_short_thr": 48,
    "momentum_thr_pct": 0.5
  },
  "metrics": {
    "win_rate": 73.33,
    "profit_factor": 2.75,
    "max_drawdown": 3.69,
    "sharpe": 3.39
  },
  "status": "VERIFIED",
  "verified_at": "2026-03-27T11:31:00"
}
```

---

## Dashboard 顯示邏輯

- `match_pct` = 當前市場滿足的條件數 / 總條件數
- 信心度不是回測的勝率，而是「當前有多少條件滿足」
- 只有 `match_pct ≥ 50%` 才會顯示信號
- VERIFIED 狀態表示「該策略的歷史回測通過 4 項指標」

---

## 啟動研究任務

```bash
# 由 team-researcher 或 CEO 人工發起
# 派任務給 team-researcher，格式如下：

sessions_send(sessionKey="team-researcher", message="""
## 任務：人工研究新量化交易策略

### Target
- 勝率 ≥ 50%
- 盈虧比 ≥ 2.0
- 最大回撤 ≤ 30%
- Sharpe ≥ 1.5

### 現有策略
（列出現有 VERIFIED 策略及問題）

### 需要方向
（多頭策略 / 4H 策略 / 突破型 / 均值回歸型等）

### 可用 Indicators
RSI, MA20/50/200, 7日動量, Regime, Golden/Death Cross, ATR 等

### 執行步驟
1. 設計策略 → 寫入 research_log.md
2. 回測驗證 → 對比 Target
3. 達標 → 更新 best_strategies.json + factor_library.json
""")
```

---

## QA 驗證清單

- [ ] team-researcher 提出策略 YAML
- [ ] team-qa 獨立實作（不看 researcher 代碼）
- [ ] 雙方結果誤差 < 5%
- [ ] 4 項指標全部達標
- [ ] CEO 審核通過
- [ ] 狀態更新為 `VERIFIED`
- [ ] `best_strategies.json` 更新
- [ ] `factor_library.json` 更新（新策略進 Dashboard）
- [ ] `research_log.md` 最終記錄

---

*Last updated: 2026-04-01*
