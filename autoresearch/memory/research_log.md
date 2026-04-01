# Research Log — 2026-04-01 14:55

## 方式：Agent 自己分析數據 → 自己設計策略 → BacktestEngine + BaseStrategy 執行

## 數據分析結論（10年 BTCUSDT 1D）
- 日均收益率: +0.153%（正偏），51.2% 日子上涨
- RSI<35 只佔約 13%（超賣是罕見現象，常伴隨後續反彈）
- 成交量 >1.5x MA 只佔 12%，>2x 只佔 4%（量增是重要信號）
- 日波動率均值: ~3.6%（適合布林帶策略捕捉波動）

## Round 1: RSI Oversold Mean Reversion（自己想的策略）
**策略**: RSIOversoldMeanReversion
**分析**: RSI<35 是罕見超賣區，結合成交量爆發 (>1.5x) 確認拋壓衰竭；在 MA50/MA200 之上過濾空頭市場
**Logic**: RSI 從<35回升+vol>1.5x → ENTRY；持倉直到 RSI>65 或 price<MA200 → EXIT
**Result**: WR=66.7%  PF=4.61  DD=0.89%  Sharpe=1.15  Trades=3  Return=1.3%
**Decision**: KEEP

## Round 2: Bollinger Band Momentum Breakout（自己想的策略）
**策略**: BollingerBandMomentumBreakout
**分析**: 日波動率 ~3.6%，BB 上軌突破配合 RSI>55 確認是真正動能爆發；量增 (>1.5x) 確認有效性
**Logic**: price 突破 BB upper band + MA20 上升 + vol>1.5x + RSI>55 → LONG；持倉直到 price<MA20 或 RSI>85
**Result**: WR=57.1%  PF=1.70  DD=7.60%  Sharpe=2.72  Trades=49  Return=21.6%
**Decision**: KEEP

---
*策略方向由 agent 分析數據後自行決定：兩個策略皆選擇 LONG only*

## Strategy Spec: RSIReversalStrategy

**Generated at**: 2026-04-01T16:00:59.050482

### 進場條件
- rsi(period=7) < 30
- vol_ratio > 1.3235975795032615

### 出場條件
- SL = 2%
- TP = 5%
- 最大持倉 = 10 根 K 線

### 參數
{
  "rsi_period": 7,
  "rsi_oversold": 30,
  "vol_thresh": 1.3235975795032615
}

### Python 代碼
```python
class RSIReversalStrategy(BaseStrategy):
    """
    RSI 反轉策略
    Agent 邏輯：RSI 超賣 + 成交量放大 = 做多
    """
    def __init__(self, rsi_period: int = 14, rsi_oversold: float = 30.0, vol_thresh: float = 1.2):
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.vol_thresh = vol_thresh

    def _compute_rsi(self, prices: np.ndarray, period: int) -> float:
        """自己計算 RSI"""
        if len(prices) < period + 1:
            return 50.0
        deltas = np.diff(prices[-period-1:])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def generate_signal(self, market_data):
        df = market_data["BTCUSDT"]
        closes = df['close'].values
        volumes = df['volume'].values

        if len(closes) < self.rsi_period + 2:
            return PositionSide.FLAT

        # Agent 自己計算 RSI
        rsi = self._compute_rsi(closes, self.rsi_period)

        # 自己計算成交量比率
        vol_avg = np.mean(volumes[-20:])
        vol_ratio = volumes[-1] / vol_avg if vol_avg > 0 else 1.0

        # Agent 自己定義進場：RSI 超賣 + 成交量放大
        if rsi < self.rsi_oversold and vol_ratio > self.vol_thresh:
            return PositionSide.LONG

        return PositionSide.FLAT

```

### Backtest 結果
- Win Rate: 60.9%
- Profit Factor: 0.88
- Max Drawdown: 21.4%
- Sharpe Ratio: -0.62
- Total Trades: 115

---

## Strategy Spec: MABreakoutStrategy

**Generated at**: 2026-04-01T16:00:59.732537

### 進場條件
- ma = mean(closes[-20:])
- price_change = (closes[-1] - closes[-20]) / closes[-20]
- closes[-1] > ma
- price_change > 0.035152703225608414
- vol_ratio > 1.9104071656611228

### 出場條件
- SL = 2%
- TP = 5%
- 最大持倉 = 10 根 K 線

### 參數
{
  "ma_period": 20,
  "price_thresh": 0.035152703225608414,
  "vol_thresh": 1.9104071656611228
}

### Python 代碼
```python
class MABreakoutStrategy(BaseStrategy):
    """
    均線突破策略
    Agent 邏輯：價格站上均線 + 成交量放大 = 做多
    """
    def __init__(self, ma_period: int = 20, price_thresh: float = 0.03, vol_thresh: float = 1.5):
        self.ma_period = ma_period
        self.price_thresh = price_thresh  # 價格變化閾值
        self.vol_thresh = vol_thresh        # 成交量倍數閾值

    def generate_signal(self, market_data):
        df = market_data["BTCUSDT"]
        closes = df['close'].values
        volumes = df['volume'].values

        if len(closes) < self.ma_period + 2:
            return PositionSide.FLAT

        # Agent 自己計算指標
        ma = np.mean(closes[-self.ma_period:])
        vol_avg = np.mean(volumes[-self.ma_period:])
        price_change = (closes[-1] - closes[-self.ma_period]) / closes[-self.ma_period]
        vol_ratio = volumes[-1] / vol_avg if vol_avg > 0 else 1.0

        # Agent 自己定義進場邏輯
        if closes[-1] > ma and price_change > self.price_thresh and vol_ratio > self.vol_thresh:
            return PositionSide.LONG

        return PositionSide.FLAT

```

### Backtest 結果
- Win Rate: 61.5%
- Profit Factor: 1.84
- Max Drawdown: 3.4%
- Sharpe Ratio: 2.51
- Total Trades: 52

---

## Strategy Spec: VolBreakoutStrategy

**Generated at**: 2026-04-01T16:01:00.423500

### 進場條件
- vol_ratio = volumes[-1] / (sum(volumes[-10:]) / 10)
- vol_ratio > 1.6604010098739508
- price_change = (closes[-1] - closes[-10]) / closes[-10]
- price_change > 0.01636035207956837

### 出場條件
- SL = 2%
- TP = 5%
- 最大持倉 = 10 根 K 線

### 參數
{
  "vol_period": 10,
  "vol_mult": 1.6604010098739508,
  "price_filter": 0.01636035207956837
}

### Python 代碼
```python
class VolBreakoutStrategy(BaseStrategy):
    """
    成交量突破策略
    Agent 邏輯：成交量突破均線 + 價格上漲 = 做多
    """
    def __init__(self, vol_period: int = 20, vol_mult: float = 2.0, price_filter: float = 0.02):
        self.vol_period = vol_period
        self.vol_mult = vol_mult
        self.price_filter = price_filter

    def generate_signal(self, market_data):
        df = market_data["BTCUSDT"]
        closes = df['close'].values
        volumes = df['volume'].values

        if len(closes) < self.vol_period + 2:
            return PositionSide.FLAT

        # Agent 自己計算
        vol_avg = np.mean(volumes[-self.vol_period:])
        vol_now = volumes[-1]

        # 價格變化
        price_change = (closes[-1] - closes[-self.vol_period]) / closes[-self.vol_period]

        # Agent 自己定義進場：成交量突破 + 價格上漲
        if vol_now > vol_avg * self.vol_mult and price_change > self.price_filter:
            return PositionSide.LONG

        return PositionSide.FLAT

```

### Backtest 結果
- Win Rate: 63.0%
- Profit Factor: 2.15
- Max Drawdown: 2.7%
- Sharpe Ratio: 3.70
- Total Trades: 73

---


## Strategy Spec: AgentStrategy_R4

**Generated at**: 2026-04-01T16:15:49.630411

### 進場條件
['volume', 'momentum']

### 參數
{
  "vol_period": 10,
  "vol_mult": 1.98,
  "ma_period": 20,
  "price_period": 10,
  "price_thresh": 0.03,
  "rsi_thresh": 45,
  "tp": 0.11,
  "sl": 0.02
}

### Backtest 結果
- Win Rate: 71.4%
- Profit Factor: 3.53
- Max Drawdown: 2.7%
- Sharpe Ratio: 3.43
- Total Trades: 35

---


## Strategy Spec: AgentStrategy_R8

**Generated at**: 2026-04-01T16:15:52.815188

### 進場條件
['volume', 'momentum']

### 參數
{
  "vol_period": 15,
  "vol_mult": 1.69,
  "ma_period": 30,
  "price_period": 10,
  "price_thresh": 0.033,
  "rsi_thresh": 40,
  "tp": 0.11,
  "sl": 0.04
}

### Backtest 結果
- Win Rate: 66.2%
- Profit Factor: 2.95
- Max Drawdown: 2.7%
- Sharpe Ratio: 4.11
- Total Trades: 65

---


## Strategy Spec: AgentStrategy_R9

**Generated at**: 2026-04-01T16:15:53.611522

### 進場條件
['rsi', 'volume']

### 參數
{
  "vol_period": 15,
  "vol_mult": 1.78,
  "ma_period": 30,
  "price_period": 5,
  "price_thresh": 0.016,
  "rsi_thresh": 45,
  "tp": 0.03,
  "sl": 0.04
}

### Backtest 結果
- Win Rate: 61.4%
- Profit Factor: 2.52
- Max Drawdown: 2.7%
- Sharpe Ratio: 3.46
- Total Trades: 70

---

## Strategy Spec: ArcBreak15

**Generated at**: 2026-04-01T16:38:59.979351
**Round**: 1

### 進場條件（Agent 自己發明）
- lowest_15 = min(lows[-15:])
- closes[-1] <= lowest_15
- volumes[-1] <= avg_vol * 1.5（量沒有異常放大）
- → 進場（新低但量縮 = 賣方衰竭）

### 出場條件
- SL = 2%
- TP = 5%
- 最大持倉 = 10 根 K 線

### 參數（Agent 自己定義）
{
  "extremes_lookback": 15
}

### Python 代碼
```python
        class AgentStrategy(BaseStrategy):
            def __init__(self):
                self.n = n
                self.p = {"extremes_lookback": n}

            def generate_signal(self, market_data):
                df = market_data["BTCUSDT"]
                c = df['close'].values
                v = df['volume'].values
                l = df['low'].values

                if len(c) < self.n + 1:
                    return PositionSide.FLAT

                lowest_price = l[-self.n:].min()
                avg_vol = v[-self.n:].mean()

                is_new_low = c[-1] <= lowest_price
                vol_ok = v[-1] <= avg_vol * 1.5  # 量沒有特別大

                if is_new_low and vol_ok:
                    return PositionSide.LONG

                return PositionSide.FLAT

```

### Backtest 結果
- Win Rate: 0.0%
- Profit Factor: 0.00
- Max Drawdown: 0.0%
- Sharpe Ratio: 0.00
- Total Trades: 0

### 備註
- 禁止使用已知指標名（RSI, MACD, Bollinger, EMA, SMA 等）
- 所有指標為 Agent 自己構造

---

## Strategy Spec: FluxDrop12

**Generated at**: 2026-04-01T16:39:00.673759
**Round**: 2

### 進場條件（Agent 自己發明）
- 觀察最近12根K線實體/整根比例
- body_ratio = |close - open| / (high - low)
- 所有12根 body_ratio < 0.54（都是小實體）
- 最後一根收盤上漲
- → 進場（震盪後反轉）

### 出場條件
- SL = 2%
- TP = 5%
- 最大持倉 = 10 根 K 線

### 參數（Agent 自己定義）
{
  "reversal_lookback": 12,
  "body_ratio": 0.54
}

### Python 代碼
```python
        class AgentStrategy(BaseStrategy):
            def __init__(self):
                self.n = n
                self.body_ratio = body_ratio
                self.p = {"reversal_lookback": n, "body_ratio": body_ratio}

            def generate_signal(self, market_data):
                df = market_data["BTCUSDT"]
                c = df['close'].values
                h = df['high'].values
                l = df['low'].values

                if len(c) < self.n + 2:
                    return PositionSide.FLAT

                # 自己觀察：最近N根是否都是小實體（震盪）
                bodies = [abs(c[-i] - c[-i-1]) / (h[-i] - l[-i] + 1e-9)
                          for i in range(1, self.n+1)]
                avg_body_ratio = sum(bodies) / len(bodies)
                all_small = all(b < self.body_ratio for b in bodies)

                # 最近一根的方向
                last_up = c[-1] > c[-2]

                if all_small and last_up:
                    return PositionSide.LONG

                return PositionSide.FLAT

```

### Backtest 結果
- Win Rate: 33.3%
- Profit Factor: 1.62
- Max Drawdown: 0.3%
- Sharpe Ratio: 0.48
- Total Trades: 3

### 備註
- 禁止使用已知指標名（RSI, MACD, Bollinger, EMA, SMA 等）
- 所有指標為 Agent 自己構造

---

## Strategy Spec: RidgeFlow4

**Generated at**: 2026-04-01T16:39:01.367395
**Round**: 3

### 進場條件（Agent 自己發明）
- momentum = (closes[-1] - closes[-4]) / closes[-4]
- vol_ratio = volumes[-1] / mean(volumes[-4:])
- momentum > 0.01 且 vol_ratio > 1.1
- → 進場（動量爆發）

### 出場條件
- SL = 2%
- TP = 5%
- 最大持倉 = 10 根 K 線

### 參數（Agent 自己定義）
{
  "momentum_lookback": 4,
  "momentum_thresh": 0.01
}

### Python 代碼
```python
        class AgentStrategy(BaseStrategy):
            def __init__(self):
                self.n = n
                self.mom_thresh = mom_thresh
                self.p = {"momentum_lookback": n, "momentum_thresh": mom_thresh}

            def generate_signal(self, market_data):
                df = market_data["BTCUSDT"]
                c = df['close'].values
                v = df['volume'].values

                if len(c) < self.n + 1:
                    return PositionSide.FLAT

                momentum = (c[-1] - c[-self.n]) / c[-self.n]
                vol_avg = v[-self.n:].mean()
                vol_ratio = v[-1] / vol_avg if vol_avg > 0 else 1.0

                # 自己定義：動量超過閾值 + 成交量確認
                if momentum > self.mom_thresh and vol_ratio > 1.1:
                    return PositionSide.LONG

                return PositionSide.FLAT

```

### Backtest 結果
- Win Rate: 54.2%
- Profit Factor: 1.61
- Max Drawdown: 8.2%
- Sharpe Ratio: 5.06
- Total Trades: 308

### 備註
- 禁止使用已知指標名（RSI, MACD, Bollinger, EMA, SMA 等）
- 所有指標為 Agent 自己構造

---

## Strategy Spec: NovaSwipe15

**Generated at**: 2026-04-01T16:39:14.343914
**Round**: 1

### 進場條件（Agent 自己發明）
- momentum = (closes[-1] - closes[-15]) / closes[-15]
- vol_ratio = volumes[-1] / mean(volumes[-15:])
- momentum > 0.04 且 vol_ratio > 1.1
- → 進場（動量爆發）

### 出場條件
- SL = 2%
- TP = 5%
- 最大持倉 = 10 根 K 線

### 參數（Agent 自己定義）
{
  "momentum_lookback": 15,
  "momentum_thresh": 0.04
}

### Python 代碼
```python
        class AgentStrategy(BaseStrategy):
            def __init__(self):
                self.n = n
                self.mom_thresh = mom_thresh
                self.p = {"momentum_lookback": n, "momentum_thresh": mom_thresh}

            def generate_signal(self, market_data):
                df = market_data["BTCUSDT"]
                c = df['close'].values
                v = df['volume'].values

                if len(c) < self.n + 1:
                    return PositionSide.FLAT

                momentum = (c[-1] - c[-self.n]) / c[-self.n]
                vol_avg = v[-self.n:].mean()
                vol_ratio = v[-1] / vol_avg if vol_avg > 0 else 1.0

                # 自己定義：動量超過閾值 + 成交量確認
                if momentum > self.mom_thresh and vol_ratio > 1.1:
                    return PositionSide.LONG

                return PositionSide.FLAT

```

### Backtest 結果
- Win Rate: 51.0%
- Profit Factor: 1.47
- Max Drawdown: 7.5%
- Sharpe Ratio: 3.83
- Total Trades: 259

### 備註
- 禁止使用已知指標名（RSI, MACD, Bollinger, EMA, SMA 等）
- 所有指標為 Agent 自己構造

---

## Strategy Spec: PulseCrack5

**Generated at**: 2026-04-01T16:39:15.026950
**Round**: 2

### 進場條件（Agent 自己發明）
- avg = mean(closes[-5:])
- deviation = (closes[-1] - avg) / avg
- deviation < -0.035（價格低於均值3.5000000000000004%）
- → 進場（均值回歸假設）

### 出場條件
- SL = 2%
- TP = 5%
- 最大持倉 = 10 根 K 線

### 參數（Agent 自己定義）
{
  "dev_lookback": 5,
  "deviation_pct": 0.035
}

### Python 代碼
```python
        class AgentStrategy(BaseStrategy):
            def __init__(self):
                self.n = n
                self.dev_thresh = dev_thresh
                self.p = {"dev_lookback": n, "deviation_pct": dev_thresh}

            def generate_signal(self, market_data):
                df = market_data["BTCUSDT"]
                c = df['close'].values

                if len(c) < self.n + 1:
                    return PositionSide.FLAT

                avg = c[-self.n:].mean()
                deviation = (c[-1] - avg) / avg

                # 自己定義：價格低於均值超過 X% → 進場（均值回歸假設）
                if deviation < -self.dev_thresh:
                    return PositionSide.LONG

                return PositionSide.FLAT

```

### Backtest 結果
- Win Rate: 53.9%
- Profit Factor: 0.94
- Max Drawdown: 16.9%
- Sharpe Ratio: -0.44
- Total Trades: 219

### 備註
- 禁止使用已知指標名（RSI, MACD, Bollinger, EMA, SMA 等）
- 所有指標為 Agent 自己構造

---

## Strategy Spec: EdgeDrop3

**Generated at**: 2026-04-01T16:39:15.728894
**Round**: 3

### 進場條件（Agent 自己發明）
- 觀察最近3根K線實體/整根比例
- body_ratio = |close - open| / (high - low)
- 所有3根 body_ratio < 0.41（都是小實體）
- 最後一根收盤上漲
- → 進場（震盪後反轉）

### 出場條件
- SL = 2%
- TP = 5%
- 最大持倉 = 10 根 K 線

### 參數（Agent 自己定義）
{
  "reversal_lookback": 3,
  "body_ratio": 0.41
}

### Python 代碼
```python
        class AgentStrategy(BaseStrategy):
            def __init__(self):
                self.n = n
                self.body_ratio = body_ratio
                self.p = {"reversal_lookback": n, "body_ratio": body_ratio}

            def generate_signal(self, market_data):
                df = market_data["BTCUSDT"]
                c = df['close'].values
                h = df['high'].values
                l = df['low'].values

                if len(c) < self.n + 2:
                    return PositionSide.FLAT

                # 自己觀察：最近N根是否都是小實體（震盪）
                bodies = [abs(c[-i] - c[-i-1]) / (h[-i] - l[-i] + 1e-9)
                          for i in range(1, self.n+1)]
                avg_body_ratio = sum(bodies) / len(bodies)
                all_small = all(b < self.body_ratio for b in bodies)

                # 最近一根的方向
                last_up = c[-1] > c[-2]

                if all_small and last_up:
                    return PositionSide.LONG

                return PositionSide.FLAT

```

### Backtest 結果
- Win Rate: 40.0%
- Profit Factor: 0.82
- Max Drawdown: 21.3%
- Sharpe Ratio: -1.01
- Total Trades: 130

### 備註
- 禁止使用已知指標名（RSI, MACD, Bollinger, EMA, SMA 等）
- 所有指標為 Agent 自己構造

---

## Strategy Spec: FluxSwing9

**Generated at**: 2026-04-01T16:45:17.277183
**Round**: 1

### 進場條件
- momentum > 0.02 + vol_ratio > 1.8 + vol_z > 1.0
- → 進場（混合策略）

### 出場條件
- SL = 2%
- TP = 5%
- 最大持倉 = 10 根 K 線

### 參數
{
  "lookback": 9,
  "threshold": 1.8
}

### Python 代碼
```python
        class AgentStrategy(BaseStrategy):
            def __init__(self):
                self.n = n
                self.t = t
                self.p = {"lookback": n, "threshold": t}

            def generate_signal(self, market_data):
                df = market_data["BTCUSDT"]
                c = df['close'].values
                v = df['volume'].values
                h = df['high'].values
                l = df['low'].values

                if len(c) < self.n + 2:
                    return PositionSide.FLAT

                # 混合邏輯：價格動能 + 成交量 + 波動率
                momentum = (c[-1] - c[-self.n]) / c[-self.n]
                vol_avg = v[-self.n:].mean()
                vol_ratio = v[-1] / vol_avg if vol_avg > 0 else 1.0
                vol_std = v[-self.n:].std()
                vol_z = (v[-1] - vol_avg) / (vol_std + 1e-9)

                # 自定義進場條件
                if momentum > 0.02 and vol_ratio > self.t and vol_z > 1.0:
                    return PositionSide.LONG

                return PositionSide.FLAT

```

### Backtest 結果
- Win Rate: 63.8%
- Profit Factor: 2.13
- Max Drawdown: 2.7%
- Sharpe Ratio: 2.85
- Total Trades: 47

---

## Strategy Spec: HelixTilt20

**Generated at**: 2026-04-01T16:45:18.012626
**Round**: 2

### 進場條件
- momentum > 0.02 + vol_ratio > 2.1 + vol_z > 1.0
- → 進場（混合策略）

### 出場條件
- SL = 2%
- TP = 5%
- 最大持倉 = 10 根 K 線

### 參數
{
  "lookback": 20,
  "threshold": 2.1
}

### Python 代碼
```python
        class AgentStrategy(BaseStrategy):
            def __init__(self):
                self.n = n
                self.t = t
                self.p = {"lookback": n, "threshold": t}

            def generate_signal(self, market_data):
                df = market_data["BTCUSDT"]
                c = df['close'].values
                v = df['volume'].values
                h = df['high'].values
                l = df['low'].values

                if len(c) < self.n + 2:
                    return PositionSide.FLAT

                # 混合邏輯：價格動能 + 成交量 + 波動率
                momentum = (c[-1] - c[-self.n]) / c[-self.n]
                vol_avg = v[-self.n:].mean()
                vol_ratio = v[-1] / vol_avg if vol_avg > 0 else 1.0
                vol_std = v[-self.n:].std()
                vol_z = (v[-1] - vol_avg) / (vol_std + 1e-9)

                # 自定義進場條件
                if momentum > 0.02 and vol_ratio > self.t and vol_z > 1.0:
                    return PositionSide.LONG

                return PositionSide.FLAT

```

### Backtest 結果
- Win Rate: 69.8%
- Profit Factor: 2.97
- Max Drawdown: 2.7%
- Sharpe Ratio: 3.77
- Total Trades: 43

---

## Strategy Spec: HelixSurge20

**Generated at**: 2026-04-01T16:45:18.744517
**Round**: 3

### 進場條件
- momentum > 0.02 + vol_ratio > 1.5 + vol_z > 1.0
- → 進場（混合策略）

### 出場條件
- SL = 2%
- TP = 5%
- 最大持倉 = 10 根 K 線

### 參數
{
  "lookback": 20,
  "threshold": 1.5
}

### Python 代碼
```python
        class AgentStrategy(BaseStrategy):
            def __init__(self):
                self.n = n
                self.t = t
                self.p = {"lookback": n, "threshold": t}

            def generate_signal(self, market_data):
                df = market_data["BTCUSDT"]
                c = df['close'].values
                v = df['volume'].values
                h = df['high'].values
                l = df['low'].values

                if len(c) < self.n + 2:
                    return PositionSide.FLAT

                # 混合邏輯：價格動能 + 成交量 + 波動率
                momentum = (c[-1] - c[-self.n]) / c[-self.n]
                vol_avg = v[-self.n:].mean()
                vol_ratio = v[-1] / vol_avg if vol_avg > 0 else 1.0
                vol_std = v[-self.n:].std()
                vol_z = (v[-1] - vol_avg) / (vol_std + 1e-9)

                # 自定義進場條件
                if momentum > 0.02 and vol_ratio > self.t and vol_z > 1.0:
                    return PositionSide.LONG

                return PositionSide.FLAT

```

### Backtest 結果
- Win Rate: 58.7%
- Profit Factor: 1.48
- Max Drawdown: 4.6%
- Sharpe Ratio: 2.80
- Total Trades: 126

---


## Strategy Spec: PriceAction_L7_B0.51_V1.41

**Generated at**: 2026-04-01T16:57:58.474783
**Type**: pure_price_action

### 進場條件（文字描述）
pure_price_action: lookback=7, body_thresh=0.51, vol_mult=1.41

### 參數
{
  "lookback": 7,
  "body_thresh": 0.51,
  "vol_period": 10,
  "vol_mult": 1.41
}

### Backtest 結果
- Win Rate: 0.0%
- Profit Factor: 0.00
- Max Drawdown: 0.0%
- Sharpe Ratio: 0.00
- Total Trades: 0

---


## Strategy Spec: MixedIndicators_MA30_RSI21_35

**Generated at**: 2026-04-01T16:57:59.203969
**Type**: mixed_indicators

### 進場條件（文字描述）
mixed: ma_period=30, rsi_period=21, rsi_thresh=35

### 參數
{
  "ma_period": 30,
  "rsi_period": 21,
  "rsi_thresh": 35
}

### Backtest 結果
- Win Rate: 24.6%
- Profit Factor: 2.15
- Max Drawdown: 23.7%
- Sharpe Ratio: 4.93
- Total Trades: 134

---


## Strategy Spec: VolProfile_MA50_DEV0.071

**Generated at**: 2026-04-01T16:57:59.893740
**Type**: volume_profile

### 進場條件（文字描述）
volume_profile: ma_period=50, dev_thresh=0.071

### 參數
{
  "vol_ma_period": 50,
  "price_dev": 0.071
}

### Backtest 結果
- Win Rate: 58.4%
- Profit Factor: 1.25
- Max Drawdown: 6.9%
- Sharpe Ratio: 1.06
- Total Trades: 89

---


## Strategy Spec: TalibMACD_F8_S20_SIG11_V1.4

**Generated at**: 2026-04-01T16:58:00.719291
**Type**: talib_macd_momentum

### 進場條件（文字描述）
talib_macd: fast=8, slow=20, signal=11, vol_mult=1.4

### 參數
{
  "fast": 8,
  "slow": 20,
  "signal": 11,
  "vol_period": 15,
  "vol_mult": 1.4
}

### Backtest 結果
- Win Rate: 59.1%
- Profit Factor: 1.61
- Max Drawdown: 4.6%
- Sharpe Ratio: 3.40
- Total Trades: 154

---


## Strategy Spec: MixedIndicators_MA15_RSI21_30

**Generated at**: 2026-04-01T16:58:01.455282
**Type**: mixed_indicators

### 進場條件（文字描述）
mixed: ma_period=15, rsi_period=21, rsi_thresh=30

### 參數
{
  "ma_period": 15,
  "rsi_period": 21,
  "rsi_thresh": 30
}

### Backtest 結果
- Win Rate: 27.2%
- Profit Factor: 1.84
- Max Drawdown: 16.7%
- Sharpe Ratio: 5.28
- Total Trades: 206

---


## Strategy Spec: TalibMACross_F8_S25_V1.77

**Generated at**: 2026-04-01T16:58:02.266574
**Type**: talib_ma_cross

### 進場條件（文字描述）
talib_ma_cross: ma_fast=8, ma_slow=25, vol_mult=1.77

### 參數
{
  "ma_fast": 8,
  "ma_slow": 25,
  "vol_period": 20,
  "vol_mult": 1.77
}

### Backtest 結果
- Win Rate: 59.3%
- Profit Factor: 1.79
- Max Drawdown: 5.0%
- Sharpe Ratio: 2.15
- Total Trades: 54

---


## Strategy Spec: VolBreakout_BB10_STD2.8_V1.91

**Generated at**: 2026-04-01T16:58:03.107245
**Type**: volatility_breakout

### 進場條件（文字描述）
volatility_breakout: bb_period=10, bb_std=2.8, vol_mult=1.91

### 參數
{
  "bb_period": 10,
  "bb_std": 2.8,
  "vol_period": 10,
  "vol_mult": 1.91
}

### Backtest 結果
- Win Rate: 66.7%
- Profit Factor: 1.06
- Max Drawdown: 0.3%
- Sharpe Ratio: 0.08
- Total Trades: 3

---


## Strategy Spec: TalibMACD_F10_S20_SIG11_V1.78

**Generated at**: 2026-04-01T16:58:03.932318
**Type**: talib_macd_momentum

### 進場條件（文字描述）
talib_macd: fast=10, slow=20, signal=11, vol_mult=1.78

### 參數
{
  "fast": 10,
  "slow": 20,
  "signal": 11,
  "vol_period": 20,
  "vol_mult": 1.78
}

### Backtest 結果
- Win Rate: 63.0%
- Profit Factor: 1.69
- Max Drawdown: 4.5%
- Sharpe Ratio: 2.26
- Total Trades: 73

---


## Strategy Spec: CustomZscore_L10_Z2.2_V1.4

**Generated at**: 2026-04-01T16:58:04.647667
**Type**: custom_zscore

### 進場條件（文字描述）
custom_zscore: lookback=10, z_thresh=2.2, vol_mult=1.4

### 參數
{
  "lookback": 10,
  "z_thresh": 2.2,
  "vol_period": 10,
  "vol_mult": 1.4
}

### Backtest 結果
- Win Rate: 62.0%
- Profit Factor: 1.65
- Max Drawdown: 5.5%
- Sharpe Ratio: 1.72
- Total Trades: 50

---


## Strategy Spec: TalibMACD_F10_S20_SIG9_V1.87

**Generated at**: 2026-04-01T16:58:05.465855
**Type**: talib_macd_momentum

### 進場條件（文字描述）
talib_macd: fast=10, slow=20, signal=9, vol_mult=1.87

### 參數
{
  "fast": 10,
  "slow": 20,
  "signal": 9,
  "vol_period": 10,
  "vol_mult": 1.87
}

### Backtest 結果
- Win Rate: 67.4%
- Profit Factor: 2.79
- Max Drawdown: 2.7%
- Sharpe Ratio: 3.07
- Total Trades: 46

---
