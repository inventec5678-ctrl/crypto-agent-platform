"""
Research Experiment Runner - 2026-04-01 (Round 5)
位置: crypto-agent-platform/research_run_v6.py

## 失敗教訓總結 (Round 4)
- TrendMA_1D_v2: WR=45.16%❌ PF=2.0223✅ Sharpe=6.444✅ DD=4.85%✅ → 3/4 PASS
  → Hypothesis: Increase pos_size 15%→20% → amplifies returns → WR unchanged, Sharpe/PF increase → PASS 4/4!

## 實驗設計
策略 G1: TrendMA_1D_v3 (pos=20%)
- 基礎: TrendMA_1D_v2 (TP=10%, SL=5%, SMA50>200 + RSI50 + mom10)
- 修改: pos_size 15% → 20%
- Hypothesis: Sharpe/PF increase proportionally, WR unchanged

策略 G2: BollingerBreakout_1D_v2 (pos=20%)
- 基礎: BollingerBreakout_1D (WR=38.92%, PF=1.30, Sharpe=3.10, DD=7.10%)
- 修改: pos_size 15% → 20%
- Hypothesis: Sharpe increases but WR still below 50%

策略 G3: Supertrend_4H (pos=20%)
- 新策略: 超級趨勢指標
- ATR-based stops
- TP=10%, SL=5%, pos=20%
"""

import sys, os, asyncio, pandas as pd, numpy as np

pkg_root = os.path.dirname(os.path.abspath(__file__))
if pkg_root not in sys.path:
    sys.path.insert(0, pkg_root)

from backtest.backtest_engine import BacktestEngine, BaseStrategy, PositionSide


# ============================================================
# Strategy G1: TrendMA_1D_v3
# ============================================================
class TrendMA1Dv3(BaseStrategy):
    """
    順勢 MA 策略 v3 (pos=20%)
    
    進場邏輯:
    - SMA50 = SMA(close, 50)
    - SMA200 = SMA(close, 200)
    - RSI14 = 100 - 100/(1 + RS)
    - mom10 = (close[-1] - close[-10]) / close[-10] * 100
    
    LONG: SMA50>SMA200 + close>SMA50 + RSI>=50 + mom>=2%
    SHORT: SMA50<SMA200 + close<SMA50 + RSI<=50 + mom<=-2%
    
    風險: TP=10%, SL=5%, position=20%
    """
    
    def __init__(self):
        self.sma_med = 50
        self.sma_long = 200
        self.mom_period = 10
        self.mom_thr = 2.0
    
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
        if len(df) < max(self.sma_long + 1, self.mom_period + 1, 15):
            return PositionSide.FLAT
        
        close = df['close']
        sma50 = close.rolling(window=self.sma_med).mean()
        sma200 = close.rolling(window=self.sma_long).mean()
        rsi = self._compute_rsi(close, 14)
        mom10 = (close.iloc[-1] - close.iloc[-self.mom_period]) / close.iloc[-self.mom_period] * 100
        
        # LONG
        if (sma50.iloc[-1] > sma200.iloc[-1] and
            close.iloc[-1] > sma50.iloc[-1] and
            rsi.iloc[-1] >= 50 and
            mom10 >= self.mom_thr):
            return PositionSide.LONG
        
        # SHORT
        if (sma50.iloc[-1] < sma200.iloc[-1] and
            close.iloc[-1] < sma50.iloc[-1] and
            rsi.iloc[-1] <= 50 and
            mom10 <= -self.mom_thr):
            return PositionSide.SHORT
        
        return PositionSide.FLAT


# ============================================================
# Strategy G2: BollingerBreakout_1D_v2
# ============================================================
class BollingerBreakout1Dv2(BaseStrategy):
    """
    布林帶突破確認策略 v2 (pos=20%)
    
    進場邏輯:
    - SMA20 = SMA(close, 20), SMA50 = SMA(close, 50), SMA100 = SMA(close, 100)
    - RSI14 = 100 - 100/(1 + RS)
    - mom5 = (close[-1] - close[-5]) / close[-5] * 100
    
    LONG: SMA50>SMA100 + close>SMA20 + RSI>=50 + mom>=0.5%
    SHORT: SMA50<SMA100 + close<SMA20 + RSI<=50 + mom<=-0.5%
    
    風險: TP=8%, SL=4%, position=20%
    """
    
    def __init__(self):
        self.sma_fast = 20
        self.sma_med = 50
        self.sma_slow = 100
        self.mom_period = 5
        self.mom_thr = 0.5
    
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
        if len(df) < max(self.sma_slow + 1, self.mom_period + 1, 21):
            return PositionSide.FLAT
        
        close = df['close']
        sma20 = close.rolling(window=self.sma_fast).mean()
        sma50 = close.rolling(window=self.sma_med).mean()
        sma100 = close.rolling(window=self.sma_slow).mean()
        rsi = self._compute_rsi(close, 14)
        mom5 = (close.iloc[-1] - close.iloc[-self.mom_period]) / close.iloc[-self.mom_period] * 100
        
        # LONG
        if (sma50.iloc[-1] > sma100.iloc[-1] and
            close.iloc[-1] > sma20.iloc[-1] and
            rsi.iloc[-1] >= 50 and
            mom5 >= self.mom_thr):
            return PositionSide.LONG
        
        # SHORT
        if (sma50.iloc[-1] < sma100.iloc[-1] and
            close.iloc[-1] < sma20.iloc[-1] and
            rsi.iloc[-1] <= 50 and
            mom5 <= -self.mom_thr):
            return PositionSide.SHORT
        
        return PositionSide.FLAT


# ============================================================
# Strategy G3: Supertrend_4H
# ============================================================
class Supertrend4H(BaseStrategy):
    """
    超級趨勢策略 (4H)
    
    進場邏輯 (完整數學定義):
    - ATR14 = SMA(TR, 14), TR = MAX(high-low, |high-prev_close|, |low-prev_close|)
    - UpperBand = (high+low)/2 + 3 * ATR14
    - LowerBand = (high+low)/2 - 3 * ATR14
    - FinalUpperBand = IF(close[-1] > UpperBand[-1]) THEN UpperBand[-1] ELSE UpperBand[-2]
    - FinalLowerBand = IF(close[-1] < LowerBand[-1]) THEN LowerBand[-1] ELSE LowerBand[-2]
    - Supertrend = IF(Supertrend[-1] == FinalUpperBand AND close < FinalUpperBand) THEN FinalUpperBand
                    ELSE IF(Supertrend[-1] == FinalUpperBand AND close > FinalUpperBand) THEN FinalLowerBand
                    ELSE IF(Supertrend[-1] == FinalLowerBand AND close > FinalLowerBand) THEN FinalLowerBand
                    ELSE FinalUpperBand
    
    LONG 信號: Supertrend 從上方切換到下方 = close 向上穿越
    SHORT 信號: Supertrend 從下方切換到上方 = close 向下穿越
    
    風險: TP=10%, SL=5%, position=20%
    """
    
    def __init__(self):
        self.atr_period = 14
        self.atr_mult = 3.0
        self.max_hold = 24  # 4H × 24 = 96h = 4 days
    
    def _compute_atr(self, df, period=14):
        high = df['high']
        low = df['low']
        close = df['close']
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()
    
    def generate_signal(self, market_data):
        df = list(market_data.values())[0]
        
        if len(df) < self.atr_period + 2:
            return PositionSide.FLAT
        
        high = df['high']
        low = df['low']
        close = df['close']
        
        atr = self._compute_atr(df, self.atr_period)
        
        # 計算上下軌
        hl2 = (high + low) / 2
        upper_band = hl2 + self.atr_mult * atr
        lower_band = hl2 - self.atr_mult * atr
        
        # 計算最終 bands (使用 vectorized approach)
        final_upper = upper_band.copy()
        final_lower = lower_band.copy()
        
        for i in range(1, len(final_upper)):
            if close.iloc[i] > final_upper.iloc[i-1]:
                final_upper.iloc[i] = upper_band.iloc[i]
            else:
                final_upper.iloc[i] = final_upper.iloc[i-1]
            
            if close.iloc[i] < final_lower.iloc[i-1]:
                final_lower.iloc[i] = lower_band.iloc[i]
            else:
                final_lower.iloc[i] = final_lower.iloc[i-1]
        
        # 超級趨勢信號
        # prev_supertrend = final_upper if prev_close > prev_final_upper else final_lower
        # 簡化: 當收盤價從低於下軌變成高於下軌 → LONG
        #       當收盤價從高於上軌變成低於上軌 → SHORT
        
        prev_close = close.shift(1)
        
        # 計算 prev_final_bands
        prev_final_upper = final_upper.shift(1)
        prev_final_lower = final_lower.shift(1)
        
        # LONG: close 向上穿越下軌
        long_cond = (prev_close.iloc[-1] < prev_final_lower.iloc[-1]) and (close.iloc[-1] > final_lower.iloc[-1])
        # SHORT: close 向下穿越上軌
        short_cond = (prev_close.iloc[-1] > prev_final_upper.iloc[-1]) and (close.iloc[-1] < final_upper.iloc[-1])
        
        if long_cond:
            return PositionSide.LONG
        if short_cond:
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
    engine.max_position_size = pos_size
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
    if result.Total_Trades > 0 and result.trades:
        wins = [t for t in result.trades if t.pnl > 0]
        losses = [t for t in result.trades if t.pnl <= 0]
        avg_win = np.mean([t.pnl_pct for t in wins]) if wins else 0
        avg_loss = np.mean([t.pnl_pct for t in losses]) if losses else 0
        print(f"  Avg_win: {avg_win:.2f}%, Avg_loss: {avg_loss:.2f}%")
        avg_dur = np.mean([t.duration for t in result.trades]) / 86400
        print(f"  平均持倉: {avg_dur:.1f} 天")


async def main():
    print("=" * 70)
    print("  Research Experiment Runner - Round 5 (2026-04-01)")
    print("  核心假設: TrendMA_1D_v2 pos 15%→20% → PASS 4/4")
    print("  Target: WR≥50%, PF≥2.0, DD≤30%, Sharpe≥1.5")
    print("=" * 70)
    
    pos_size = 0.20
    start_1d = "2020-01-01"
    end_1d = "2024-12-31"
    start_4h = "2023-01-01"
    end_4h = "2024-12-31"
    
    results = {}
    
    # ---------- 實驗 G1: TrendMA_1D_v3 ----------
    print("\n" + "=" * 70)
    print("實驗 G1: TrendMA_1D_v3 (pos=20%)")
    print("數據: BTCUSDT 1D | TP=10%, SL=5%, Pos=20%")
    print("  Hypothesis: Sharpe/PF increase proportionally")
    print("=" * 70)
    
    s1 = TrendMA1Dv3()
    r1 = await run_backtest("TrendMA_1D_v3", s1, "BTCUSDT", "1d",
                            start_1d, end_1d, tp=10.0, sl=5.0, pos_size=pos_size)
    results['TrendMA_1D_v3'] = r1
    print_result("TrendMA_1D_v3", r1, tp=10.0, sl=5.0, pos_size=pos_size)
    
    # ---------- 實驗 G2: BollingerBreakout_1D_v2 ----------
    print("\n" + "=" * 70)
    print("實驗 G2: BollingerBreakout_1D_v2 (pos=20%)")
    print("數據: BTCUSDT 1D | TP=8%, SL=4%, Pos=20%")
    print("=" * 70)
    
    s2 = BollingerBreakout1Dv2()
    r2 = await run_backtest("BollingerBreakout_1D_v2", s2, "BTCUSDT", "1d",
                            start_1d, end_1d, tp=8.0, sl=4.0, pos_size=pos_size)
    results['BollingerBreakout_1D_v2'] = r2
    print_result("BollingerBreakout_1D_v2", r2, tp=8.0, sl=4.0, pos_size=pos_size)
    
    # ---------- 實驗 G3: Supertrend_4H ----------
    print("\n" + "=" * 70)
    print("實驗 G3: Supertrend_4H (pos=20%)")
    print("數據: BTCUSDT 4H | TP=10%, SL=5%, Pos=20%")
    print("=" * 70)
    
    s3 = Supertrend4H()
    r3 = await run_backtest("Supertrend_4H", s3, "BTCUSDT", "4h",
                            start_4h, end_4h, tp=10.0, sl=5.0, pos_size=pos_size)
    results['Supertrend_4H'] = r3
    print_result("Supertrend_4H", r3, tp=10.0, sl=5.0, pos_size=pos_size)
    
    # ---------- 總結 ----------
    print("\n" + "=" * 70)
    print("  Round 5 實驗總結")
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
        best = max(summary, key=lambda x: x['sharpe'])
        print(f"\n  最接近: {best['name']}")
        print(f"  WR={best['wr']:.2f}%, PF={best['pf']:.2f}, DD={best['dd']:.2f}%, Sharpe={best['sharpe']:.2f}, Trades={best['trades']}")
    
    return summary


if __name__ == "__main__":
    summary = asyncio.run(main())
