"""
Research Experiment Runner - 2026-04-01 (Round 2)
位置: crypto-agent-platform/research_run_v3.py

## 第一輪失敗分析
所有 3 個策略 Sharpe 都 > 1.5，但 DD > 30% (position sizing = 20% 造成)
Root cause: 20% position × 3% SL × 連續虧損 = 60%+ DD

## 第二輪設計原則
1. 降低 position size → 改為 10% (max_position_size)
2. 加入止損保護（engine.stop_loss）
3. 設計信號更頻繁的策略（平滑 equity curve）
4. TP/SL 比維持在 2:1

## 實驗設計

策略 A: TrendFollowing_4H (4H 順勢)
- 組合: SMA20 + RSI(14) + 動量
- Position size: 10% (降一半)
- TP=6%, SL=3%

策略 B: TrendFollowing_1D (1D 順勢)
- 組合: SMA50 + RSI(14) + ATR
- Position size: 10%
- TP=8%, SL=4% (擴大 TP/SL 給予市場空間)

策略 C: MeanReversion_4H (4H 均值回歸)
- 組合: Bollinger Band + RSI(14)
- Position size: 10%
- TP=4%, SL=2%

策略 D: EMACross_4H_Strict (4H EMA 交叉 - 嚴格過濾)
- EMA(9,21) 黃金交叉 + RSI(14) > 55 + 價格 > EMA21
- Position size: 10%
- TP=7%, SL=3.5%

策略 E: Supertrend_4H (4H 超級趨勢)
- 使用 ATR-based 超級趨勢指標
- Position size: 10%
- TP=8%, SL=4%
"""

import sys
import os
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

pkg_root = os.path.dirname(os.path.abspath(__file__))
if pkg_root not in sys.path:
    sys.path.insert(0, pkg_root)

from backtest.backtest_engine import (
    BacktestEngine, BaseStrategy, PositionSide
)


# ============================================================
# Strategy A: TrendFollowing_4H
# ============================================================
class TrendFollowing4H(BaseStrategy):
    """
    順勢策略 (4H)
    
    進場邏輯 (完整數學定義):
    - SMA20 = SMA(close, 20)
    - SMA50 = SMA(close, 50)
    - trend_up = SMA20[-1] > SMA50[-1]  (多頭排列)
    - mom5 = (close[-1] - close[-5]) / close[-5] * 100  (5日動量)
    - RSI14 = 100 - 100/(1 + RS), RS = SMA(漲幅,14)/SMA(跌幅,14)
    
    LONG 進場:
    - C1: SMA20[-1] > SMA50[-1]  (多頭排列)
    - C2: close[-1] > SMA20[-1]  (價格站上快線)
    - C3: RSI14 >= 45 AND RSI14 <= 70  (RSI 處於健康區間)
    - C4: mom5 >= 0.0  (動量為正)
    
    SHORT 進場（允許，根據市場方向）:
    - C1: SMA20[-1] < SMA50[-1]
    - C2: close[-1] < SMA20[-1]
    - C3: RSI14 <= 55 AND RSI14 >= 30
    - C4: mom5 <= 0.0
    
    風險參數:
    - TP=6%, SL=3%, position=10%
    """
    
    def __init__(self):
        self.sma_fast = 20
        self.sma_slow = 50
        self.mom_period = 5
        self.max_hold = 20  # 4H × 20 = 80h ≈ 3.3 days
    
    def _compute_rsi(self, series, period=14):
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta.where(delta < 0, 0.0))
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))
    
    def generate_signal(self, market_data):
        df = list(market_data.values())[0]
        
        if len(df) < max(self.sma_slow + 1, self.mom_period + 1, 15):
            return PositionSide.FLAT
        
        close = df['close']
        sma20 = close.rolling(window=self.sma_fast).mean()
        sma50 = close.rolling(window=self.sma_slow).mean()
        rsi = self._compute_rsi(close, 14)
        mom5 = (close.iloc[-1] - close.iloc[-self.mom_period]) / close.iloc[-self.mom_period] * 100
        
        trend_up = sma20.iloc[-1] > sma50.iloc[-1]
        trend_down = sma20.iloc[-1] < sma50.iloc[-1]
        price_above_fast = close.iloc[-1] > sma20.iloc[-1]
        price_below_fast = close.iloc[-1] < sma20.iloc[-1]
        rsi_healthy = 45 <= rsi.iloc[-1] <= 70
        rsi_weak = 30 <= rsi.iloc[-1] <= 55
        
        # LONG
        c1 = trend_up
        c2 = price_above_fast
        c3 = rsi_healthy
        c4 = mom5 >= 0.0
        if c1 and c2 and c3 and c4:
            return PositionSide.LONG
        
        # SHORT
        c1s = trend_down
        c2s = price_below_fast
        c3s = rsi.iloc[-1] <= 55 and rsi.iloc[-1] >= 30
        c4s = mom5 <= 0.0
        if c1s and c2s and c3s and c4s:
            return PositionSide.SHORT
        
        return PositionSide.FLAT


# ============================================================
# Strategy B: TrendFollowing_1D (擴大 TP/SL)
# ============================================================
class TrendFollowing1D(BaseStrategy):
    """
    順勢策略 (1D, 擴大 TP/SL)
    
    進場邏輯:
    - SMA50 = SMA(close, 50)
    - SMA200 = SMA(close, 200)
    - RSI14 = 100 - 100/(1 + RS)
    - ATR14 = SMA(TR, 14), TR = MAX(high-low, |high-prev_close|, |low-prev_close|)
    - mom10 = (close[-1] - close[-10]) / close[-10] * 100
    
    LONG 進場:
    - C1: SMA50[-1] > SMA200[-1]  (長期多頭)
    - C2: close[-1] > SMA50[-1]
    - C3: RSI14 >= 50
    - C4: mom10 >= 2.0  (動量確認)
    
    風險: TP=8%, SL=4%, position=10%
    """
    
    def __init__(self):
        self.sma_med = 50
        self.sma_long = 200
        self.mom_period = 10
        self.mom_thr = 2.0
        self.max_hold = 15  # 1D × 15 = 15 days
    
    def _compute_rsi(self, series, period=14):
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta.where(delta < 0, 0.0))
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))
    
    def _compute_atr(self, df, period=14):
        high = df['high']
        low = df['low']
        prev_close = df['close'].shift(1)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()
    
    def generate_signal(self, market_data):
        df = list(market_data.values())[0]
        
        if len(df) < max(self.sma_long + 1, self.mom_period + 1, 15):
            return PositionSide.FLAT
        
        close = df['close']
        sma50 = close.rolling(window=self.sma_med).mean()
        sma200 = close.rolling(window=self.sma_long).mean()
        rsi = self._compute_rsi(close, 14)
        mom10 = (close.iloc[-1] - close.iloc[-self.mom_period]) / close.iloc[-self.mom_period] * 100
        
        # LONG
        c1 = sma50.iloc[-1] > sma200.iloc[-1]
        c2 = close.iloc[-1] > sma50.iloc[-1]
        c3 = rsi.iloc[-1] >= 50
        c4 = mom10 >= self.mom_thr
        if c1 and c2 and c3 and c4:
            return PositionSide.LONG
        
        # SHORT
        c1s = sma50.iloc[-1] < sma200.iloc[-1]
        c2s = close.iloc[-1] < sma50.iloc[-1]
        c3s = rsi.iloc[-1] <= 50
        c4s = mom10 <= -self.mom_thr
        if c1s and c2s and c3s and c4s:
            return PositionSide.SHORT
        
        return PositionSide.FLAT


# ============================================================
# Strategy C: MeanReversion_4H (均值回歸)
# ============================================================
class MeanReversion4H(BaseStrategy):
    """
    均值回歸策略 (4H)
    
    進場邏輯:
    - SMA20 = SMA(close, 20)
    - STD20 = STD(close, 20)
    - UpperBand = SMA20 + 2*STD20
    - LowerBand = SMA20 - 2*STD20
    - RSI14 = 100 - 100/(1 + RS)
    
    LONG 進場:
    - C1: close[-1] <= LowerBand  (價格觸及下軌)
    - C2: RSI14 <= 35  (超賣)
    - C3: close[-1] > SMA20[-1]  (已在均線上方，否則確認反彈)
    
    SHORT 進場:
    - C1: close[-1] >= UpperBand
    - C2: RSI14 >= 65
    - C3: close[-1] < SMA20[-1]
    
    風險: TP=4%, SL=2%, position=10%
    """
    
    def __init__(self):
        self.sma_period = 20
        self.bb_std = 2
        self.max_hold = 30  # 4H × 30 = 120h = 5 days
    
    def _compute_rsi(self, series, period=14):
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta.where(delta < 0, 0.0))
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))
    
    def generate_signal(self, market_data):
        df = list(market_data.values())[0]
        
        if len(df) < self.sma_period + 1:
            return PositionSide.FLAT
        
        close = df['close']
        sma20 = close.rolling(window=self.sma_period).mean()
        std20 = close.rolling(window=self.sma_period).std()
        upper = sma20 + self.bb_std * std20
        lower = sma20 - self.bb_std * std20
        rsi = self._compute_rsi(close, 14)
        
        # LONG: 價格接觸下軌 + RSI 超賣
        c1 = close.iloc[-1] <= lower.iloc[-1]
        c2 = rsi.iloc[-1] <= 35
        c3 = close.iloc[-1] > sma20.iloc[-1]  # 已在均線上方
        if c1 and c2 and c3:
            return PositionSide.LONG
        
        # SHORT
        c1s = close.iloc[-1] >= upper.iloc[-1]
        c2s = rsi.iloc[-1] >= 65
        c3s = close.iloc[-1] < sma20.iloc[-1]
        if c1s and c2s and c3s:
            return PositionSide.SHORT
        
        return PositionSide.FLAT


# ============================================================
# Strategy D: EMACross_4H_Strict
# ============================================================
class EMACross4HStrict(BaseStrategy):
    """
    EMA 交叉嚴格版 (4H)
    
    進場邏輯:
    - EMA9 = EMA(close, 9)
    - EMA21 = EMA(close, 21)
    - 黃金交叉: EMA9[-1] > EMA21[-1] AND EMA9[-2] <= EMA21[-2]
    - 死亡交叉: EMA9[-1] < EMA21[-1] AND EMA9[-2] >= EMA21[-2]
    - RSI14 >= 55 (多頭確認) / <= 45 (空頭確認)
    - 額外確認: ADX(14) > 20  (確認是趨勢市場，不是盤整)
    
    風險: TP=7%, SL=3.5%, position=10%
    """
    
    def __init__(self):
        self.ema_fast = 9
        self.ema_slow = 21
        self.adx_period = 14
        self.max_hold = 24  # 4H × 24 = 96h = 4 days
    
    def _compute_ema(self, series, period):
        return series.ewm(span=period, adjust=False).mean()
    
    def _compute_rsi(self, series, period=14):
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta.where(delta < 0, 0.0))
        avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))
    
    def _compute_adx(self, df, period=14):
        high = df['high']
        low = df['low']
        close = df['close']
        prev_close = close.shift(1)
        
        plus_dm = high.diff()
        minus_dm = -low.diff()
        
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
        
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        atr = tr.rolling(window=period).mean()
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
        
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()
        
        return adx
    
    def generate_signal(self, market_data):
        df = list(market_data.values())[0]
        
        if len(df) < max(self.ema_slow + 1, self.adx_period + 1, 15):
            return PositionSide.FLAT
        
        close = df['close']
        ema9 = self._compute_ema(close, self.ema_fast)
        ema21 = self._compute_ema(close, self.ema_slow)
        rsi = self._compute_rsi(close, 14)
        adx = self._compute_adx(df, self.adx_period)
        
        gc = (ema9.iloc[-1] > ema21.iloc[-1]) and (ema9.iloc[-2] <= ema21.iloc[-2])
        dc = (ema9.iloc[-1] < ema21.iloc[-1]) and (ema9.iloc[-2] >= ema21.iloc[-2])
        adx_trend = adx.iloc[-1] > 20
        
        # LONG: 黃金交叉 + RSI 確認 + ADX 確認
        if gc and rsi.iloc[-1] >= 55 and adx_trend:
            return PositionSide.LONG
        
        # SHORT: 死亡交叉 + RSI 確認 + ADX 確認
        if dc and rsi.iloc[-1] <= 45 and adx_trend:
            return PositionSide.SHORT
        
        return PositionSide.FLAT


# ============================================================
# Backtest Runner
# ============================================================

async def run_backtest(strategy_name, strategy_instance, symbol, interval,
                       start_date, end_date, tp, sl, pos_size, initial_capital=10000.0):
    
    engine = BacktestEngine()
    engine.initial_capital = initial_capital
    engine.cash = initial_capital
    engine.equity = initial_capital
    engine.max_position_size = pos_size  # 10% (reduced from 20%)
    engine.stop_loss = sl / 100
    engine.take_profit = tp / 100
    
    parquet_path = f"data/{symbol.lower()}_{interval}.parquet"
    
    if os.path.exists(parquet_path):
        print(f"📂 從本地加載: {parquet_path}")
        df = pd.read_parquet(parquet_path)
        df = df[(df['open_time'] >= pd.Timestamp(start_date)) & 
                 (df['open_time'] <= pd.Timestamp(end_date))]
        engine.load_dataframe(symbol, df)
    else:
        print(f"📥 下載數據: {symbol} {interval} {start_date} ~ {end_date}")
        await engine.load_data(symbol, interval, start_date, end_date)
    
    engine.set_strategy(strategy_instance)
    result = await engine.run()
    engine.close()
    
    return result


def print_result(name, result, tp, sl, pos_size):
    print(f"\n{'─' * 50}")
    print(f"  {name}")
    print(f"{'─' * 50}")
    print(f"  總交易:   {result.Total_Trades} 筆")
    print(f"  勝率:     {result.Win_Rate:.2f}%")
    print(f"  盈虧比:   {result.Profit_Factor:.4f}")
    print(f"  最大回撤: {result.Max_Drawdown_Pct:.2f}%")
    print(f"  Sharpe:   {result.Sharpe_Ratio:.4f}")
    print(f"  TP={tp}%, SL={sl}%, Pos={pos_size*100:.0f}%")
    if result.trades:
        avg_dur = np.mean([t.duration for t in result.trades]) / 86400
        print(f"  平均持倉: {avg_dur:.1f} 天")


async def main():
    print("=" * 70)
    print("  Research Experiment Runner - Round 2 (2026-04-01)")
    print("  分析: 第一輪 DD>30% 來自 position=20%")
    print("  改進: position=10%, 擴大 TP/SL, 加入 ADX 過濾")
    print("  Target: WR≥50%, PF≥2.0, DD≤30%, Sharpe≥1.5")
    print("=" * 70)
    
    pos_size = 0.10  # 10% position size (reduced from 20%)
    initial_capital = 10000.0
    
    start_1d = "2020-01-01"
    end_1d = "2024-12-31"
    start_4h = "2023-01-01"
    end_4h = "2024-12-31"
    
    results = {}
    
    # ---------- 實驗 A: TrendFollowing_4H ----------
    print("\n" + "=" * 70)
    print("實驗 A: TrendFollowing_4H")
    print("數據: BTCUSDT 4H | TP=6%, SL=3%, Pos=10%")
    print("=" * 70)
    
    sa = TrendFollowing4H()
    ra = await run_backtest("TrendFollowing_4H", sa, "BTCUSDT", "4h",
                            start_4h, end_4h, tp=6.0, sl=3.0, pos_size=pos_size)
    results['TrendFollowing_4H'] = ra
    print_result("TrendFollowing_4H", ra, tp=6.0, sl=3.0, pos_size=pos_size)
    
    # ---------- 實驗 B: TrendFollowing_1D ----------
    print("\n" + "=" * 70)
    print("實驗 B: TrendFollowing_1D")
    print("數據: BTCUSDT 1D | TP=8%, SL=4%, Pos=10%")
    print("=" * 70)
    
    sb = TrendFollowing1D()
    rb = await run_backtest("TrendFollowing_1D", sb, "BTCUSDT", "1d",
                            start_1d, end_1d, tp=8.0, sl=4.0, pos_size=pos_size)
    results['TrendFollowing_1D'] = rb
    print_result("TrendFollowing_1D", rb, tp=8.0, sl=4.0, pos_size=pos_size)
    
    # ---------- 實驗 C: MeanReversion_4H ----------
    print("\n" + "=" * 70)
    print("實驗 C: MeanReversion_4H")
    print("數據: BTCUSDT 4H | TP=4%, SL=2%, Pos=10%")
    print("=" * 70)
    
    sc = MeanReversion4H()
    rc = await run_backtest("MeanReversion_4H", sc, "BTCUSDT", "4h",
                            start_4h, end_4h, tp=4.0, sl=2.0, pos_size=pos_size)
    results['MeanReversion_4H'] = rc
    print_result("MeanReversion_4H", rc, tp=4.0, sl=2.0, pos_size=pos_size)
    
    # ---------- 實驗 D: EMACross_4H_Strict ----------
    print("\n" + "=" * 70)
    print("實驗 D: EMACross_4H_Strict")
    print("數據: BTCUSDT 4H | TP=7%, SL=3.5%, Pos=10%")
    print("=" * 70)
    
    sd = EMACross4HStrict()
    rd = await run_backtest("EMACross_4H_Strict", sd, "BTCUSDT", "4h",
                            start_4h, end_4h, tp=7.0, sl=3.5, pos_size=pos_size)
    results['EMACross_4H_Strict'] = rd
    print_result("EMACross_4H_Strict", rd, tp=7.0, sl=3.5, pos_size=pos_size)
    
    # ---------- 總結 ----------
    print("\n" + "=" * 70)
    print("  實驗總結")
    print("=" * 70)
    
    summary = []
    for name, r in results.items():
        wr_ok = r.Win_Rate >= 50.0
        pf_ok = r.Profit_Factor >= 2.0
        dd_ok = r.Max_Drawdown_Pct <= 30.0
        sharpe_ok = r.Sharpe_Ratio >= 1.5
        all_pass = wr_ok and pf_ok and dd_ok and sharpe_ok
        
        status = "✅ PASS" if all_pass else "❌ FAIL"
        print(f"\n{name}: {status}")
        print(f"  WR={r.Win_Rate:.2f}% (≥50% {'✅' if wr_ok else '❌'})")
        print(f"  PF={r.Profit_Factor:.4f} (≥2.0 {'✅' if pf_ok else '❌'})")
        print(f"  DD={r.Max_Drawdown_Pct:.2f}% (≤30% {'✅' if dd_ok else '❌'})")
        print(f"  Sharpe={r.Sharpe_Ratio:.4f} (≥1.5 {'✅' if sharpe_ok else '❌'})")
        print(f"  Trades={r.Total_Trades}")
        print(f"  Return={r.total_return_pct:.2f}%")
        
        summary.append({
            'name': name,
            'wr': r.Win_Rate,
            'pf': r.Profit_Factor,
            'dd': r.Max_Drawdown_Pct,
            'sharpe': r.Sharpe_Ratio,
            'trades': r.Total_Trades,
            'ret': r.total_return_pct,
            'pass': all_pass
        })
    
    passed = [s for s in summary if s['pass']]
    if passed:
        print(f"\n🎉 找到 {len(passed)} 個 4/4 達標策略!")
        for p in passed:
            print(f"  → {p['name']}: WR={p['wr']:.1f}%, PF={p['pf']:.2f}, DD={p['dd']:.1f}%, Sharpe={p['sharpe']:.2f}, Trades={p['trades']}")
    else:
        print("\n😢 沒有策略達到 4/4 標準。")
    
    return summary


if __name__ == "__main__":
    summary = asyncio.run(main())
