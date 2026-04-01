# Auto Research 日誌 - 2026-04-01 新實驗

## 研究方向分析

### 現有 Dashboard 缺口
- Dashboard 只有 Regime-Aware MA Cross (SHORT 為主) 兩個 VERIFIED 策略
- **缺口：沒有 LONG 策略**
- 需要找到 4/4 達標的 LONG 策略

### 關鍵洞察（從失敗實驗學習）
1. **為什麼過去 130+ 實驗都失敗？** → 大多使用極 tight SL (0.5%-1.5%)，被市場噪音止損
2. **Regime-Aware MA Cross v2 如何成功？** → SL=3%, TP=7% → 給予市場呼吸空間
3. **PF 公式**: PF = (WR × Avg_win) / ((1-WR) × Avg_loss)
4. **50% WR + 2.0 PF → 需要 TP/SL 比 = 2.0** → TP=6%, SL=3% 或 TP=7%, SL=3.5%

### 設計方向
- 學習 MA Cross v2 的成功參數：SL=3%, TP=6-7%
- 設計 LONG 專屬策略（市場牛市或區間震盪時有效）

---

## 實驗 #N+1 — 2026-04-01 09:40

### 策略名稱: VolMomentumBreakout_Long_v1

#### 分析視角
比特幣常見「成交量突破後動量延續」型態。當價格在均線上方整理後，
突然放量突破往往預示新一輪上漲。這個策略結合成交量放大 + 價格動量 +
趨勢確認，捕捉突破後的動量延續。

#### 策略邏輯（完整定義）
- **指標 1 - 成交量動量**:
  `vol_sma20 = SMA(volume, 20)`
  `vol_ratio = volume.iloc[-1] / vol_sma20.iloc[-1]`
- **指標 2 - 價格動量**:
  `mom7 = (close.iloc[-1] - close.iloc[-7]) / close.iloc[-7] * 100`（7日動量%）
  `SMA20 = SMA(close, 20)`
  `SMA50 = SMA(close, 50)`
- **指標 3 - 趨勢確認**:
  `trend_up = SMA20.iloc[-1] > SMA50.iloc[-1]`（20日均 > 50日均 = 多頭排列）
- **進場條件 LONG**:
  - C1: `vol_ratio >= 2.0`（當根成交量 ≥ 20日均量的 2 倍）
  - C2: `mom7 >= 1.0`（7日動量 ≥ 1%，確認向上動量）
  - C3: `trend_up == True`（多頭排列確認）
  - C4: `close.iloc[-1] > SMA20.iloc[-1]`（價格站上 20 日均）
- **進場條件 SHORT**: 全部反向（N/A，本策略專注 LONG）
- **止盈**: `tp = 6.0`%（固定 6%）
- **止損**: `sl = 3.0`%（固定 3%）
- **最大持有**: `max_hold = 15` 根 K 線（1D 數據 = 15 天）

#### 實驗參數（不可事後修改）
- vol_mult: 2.0
- mom_thr: 1.0
- tp: 6.0
- sl: 3.0
- max_hold: 15
- sma_fast: 20
- sma_slow: 50
- mom_period: 7

#### Target 對比
- 勝率 ≥ 50% → ❓
- 盈虧比 ≥ 2.0 → ❓
- 最大回撤 ≤ 30% → ❓
- Sharpe ≥ 1.5 → ❓

---

## 實驗 #N+2 — 2026-04-01 09:40

### 策略名稱: RSIStackedBull_Long_v1

#### 分析視角
比特幣在牛市中常見 RSI 從低位反彈並持續上升的模式。
當價格維持在長期均線上方時，RSI 的超賣反彈有很高的成功概率。
這個策略結合多週期趨勢確認 + RSI 均值回歸。

#### 策略邏輯（完整定義）
- **指標 1 - RSI(14)**:
  `RSI(14) = 100 - 100/(1 + RS)`
  `RS = SMA(漲幅,14) / SMA(跌幅,14)`
- **指標 2 - 長期均線確認**:
  `SMA200 = SMA(close, 200)`
- **指標 3 - 短期動量**:
  `mom5 = (close.iloc[-1] - close.iloc[-5]) / close.iloc[-5] * 100`
- **進場條件 LONG**:
  - C1: `RSI(14) >= 55 AND RSI(14) <= 75`（RSI 處於強勢區間但未超買）
  - C2: `close.iloc[-1] > SMA200.iloc[-1]`（價格在 200 日均上方）
  - C3: `mom5 >= 0.5`（5日動量為正）
- **進場條件 SHORT**: 全部反向（N/A）
- **止盈**: `tp = 6.0`%
- **止損**: `sl = 3.0`%
- **最大持有**: `max_hold = 15` 根 K 線

#### 實驗參數（不可事後修改）
- rsi_entry_min: 55
- rsi_entry_max: 75
- sma_long: 200
- mom_thr: 0.5
- mom_period: 5
- tp: 6.0
- sl: 3.0
- max_hold: 15

#### Target 對比
- 勝率 ≥ 50% → ❓
- 盈虧比 ≥ 2.0 → ❓
- 最大回撤 ≤ 30% → ❓
- Sharpe ≥ 1.5 → ❓

---

## 實驗 #N+3 — 2026-04-01 09:40

### 策略名稱: EMAGoldenCross_4H_Long_v1

#### 分析視角
均線黃金交叉是經典的趨勢追蹤信號。4H 級別的黃金交叉
比 1D 更敏感但比 1H 更穩定。這個策略結合 EMA 交叉 +
 RSI 確認 + 價格位置過濾，只做 LONG。

#### 策略邏輯（完整定義）
- **指標 1 - EMA 交叉**:
  `EMA9 = EMA(close, 9)`
  `EMA21 = EMA(close, 21)`
  `prev_EMA9 = EMA9.iloc[-2]`
  `prev_EMA21 = EMA21.iloc[-2]`
  `gc_bull = (EMA9.iloc[-1] > EMA21.iloc[-1]) AND (prev_EMA9 <= prev_EMA21)`（黃金交叉）
- **指標 2 - RSI 確認**:
  `RSI(14) = 100 - 100/(1 + RS)`
- **指標 3 - ATR（真實波幅）**:
  `TR = MAX(high-low, ABS(high-prev_close), ABS(low-prev_close))`
  `ATR14 = SMA(TR, 14)`
- **進場條件 LONG**:
  - C1: `gc_bull == True`（當根 K 線發生黃金交叉）
  - C2: `RSI(14) >= 50`（RSI 確認多頭）
  - C3: `close.iloc[-1] > EMA21.iloc[-1]`（價格在慢線上方）
- **進場條件 SHORT**: 全部反向（N/A）
- **止盈**: `tp = 7.0`%（擴大 TP 配合 EMA 趨勢特性）
- **止損**: `sl = 3.0`%
- **最大持有**: `max_hold = 20` 根 K 線（4H × 20 = 80 小時 ≈ 3.3 天）

#### 實驗參數（不可事後修改）
- ema_fast: 9
- ema_slow: 21
- rsi_thr: 50
- tp: 7.0
- sl: 3.0
- max_hold: 20

#### Target 對比
- 勝率 ≥ 50% → ❓
- 盈虧比 ≥ 2.0 → ❓
- 最大回撤 ≤ 30% → ❓
- Sharpe ≥ 1.5 → ❓
