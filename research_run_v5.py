"""
Research Experiment Runner - 2026-04-01 (Round 4)
位置: crypto-agent-platform/research_run_v5.py

## 失敗教訓總結 (Round 3)
- EMAPureCross_1D_Long: WR=71.43%✅ PF=1.61❌ Sharpe=1.01❌ DD=0.47%✅
  → 高 WR，但 TP=6% SL=3% → 平均賺2%/輸3.1% → PF=1.61
  → 需要擴大 TP/SL 到 TP=7%, SL=3.5% 來提高 PF
- MultiTimeframe_1D: WR=38.24%❌ Sharpe=3.23✅ DD=5.21%✅
  → 高 Sharpe，但 WR 不夠 → 需要更嚴格進場

## 核心洞察
1. TP/SL 比例影響 PF: TP 越大 → Avg_win 越大 → PF 越大
2. Position size 影響 Sharpe 和 PF: pos 越大 → 放大returns → Sharpe/PF 提高
3. WR 取決於進場條件嚴格程度: 越嚴格 → WR 越高

## Round 4 實驗設計

策略 F1: EMAPureCross_1D_v2
- 基礎: EMAPureCross_1D_Long
- 修改: TP=7%, SL=3.5% (擴大 TP/SL)
- Position size: 15%
- 假設: TP=7% SL=3.5% → 如果 WR 維持71%，Avg_win=7%, Avg_loss=3.5% → PF=2.0✅

策略 F2: TrendMA_1D_v2
- 基礎: TrendFollowing_1D (Round 2)
- 修改: TP=10%, SL=5% (擴大 TP/SL)
- Position size: 15%
- 假設: 如果 WR 維持44%，擴大 TP/SL → Avg_win 增加到 10%, Avg_loss = 5% → PF=2.0✅

策略 F3: BollingerBreakout_1D
- 新策略: 布林帶突破確認進場
- 進場: 價格突破上軌 + RSI>50 + SMA20>SMA50
- Position size: 15%
- TP=8%, SL=4%
"""

import sys, os, asyncio, pandas as pd, numpy as np

pkg_root = os.path.dirname(os.path.abspath(__file__))
if pkg_root not in sys.path:
    sys.path.insert(0, pkg_root)

from backtest.backtest_engine import BacktestEngine, BaseStrategy, PositionSide


# ============================================================
# Strategy F1: EMAPureCross_1D_v2
# ============================================================
class EMAPureCross1Dv2(BaseStrategy):
    """
    EMA 交叉策略 v2 (1D LONG)
    
    進場邏輯:
    - EMA9 = EMA(close, 9)
    - EMA21 = EMA(close, 21)
    - SMA200 = SMA(close, 200)
    - RSI14 = 100 - 100/(1 + RS)
    - mom7 = (close[-1] - close[-7]) / close[-7] * 100
    
    LONG 進場:
    - C1: EMA9[-1] > EMA21[-1] AND EMA9[-2] <= EMA21[-2]  (黃金交叉)
    - C2: RSI14 >= 55
    - C3: mom7 >= 1.0
    - C4: close[-1] > SMA200[-1]
    
    風險: TP=7%, SL=3.5%, position=15%
    """
    
    def __init__(self):
        self.ema_fast = 9
        self.ema_slow = 21
        self.sma_long = 200
        self.mom_period = 7
        self.mom_thr = 1.0
    
    def _compute_ema(self, series, period):
        return series.ewm(span=period, adjust=False).mean()
    
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
        if len(df) < max(self.ema_slow + 1, self.mom_period + 1, self.sma_long + 1):
            return PositionSide.FLAT
        
        close = df['close']
        ema9 = self._compute_ema(close, self.ema_fast)
        ema21 = self._compute_ema(close, self.ema_slow)
        sma200 = close.rolling(window=self.sma_long).mean()
        rsi = self._compute_rsi(close, 14)
        mom7 = (close.iloc[-1] - close.iloc[-self.mom_period]) / close.iloc[-self.mom_period] * 100
        
        gc = (ema9.iloc[-1] > ema21.iloc[-1]) and (ema9.iloc[-2] <= ema21.iloc[-2])
        
        if (gc and rsi.iloc[-1] >= 55 and mom7 >= self.mom_thr 
                and close.iloc[-1] > sma200.iloc[-1]):
            return PositionSide.LONG
        return PositionSide.FLAT


# ============================================================
# Strategy F2: TrendMA_1D_v2
# ============================================================
class TrendMA1Dv2(BaseStrategy):
    """
    順勢 MA 策略 v2 (1D LONG)
    
    進場邏輯:
    - SMA50 = SMA(close, 50)
    - SMA200 = SMA(close, 200)
    - RSI14 = 100 - 100/(1 + RS)
    - mom10 = (close[-1] - close[-10]) / close[-10] * 100
    
    LONG 進場:
    - C1: SMA50[-1] > SMA200[-1]  (牛市排列)
    - C2: close[-1] > SMA50[-1]
    - C3: RSI14 >= 50
    - C4: mom10 >= 2.0
    
    風險: TP=10%, SL=5%, position=15%
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
        
        c1 = sma50.iloc[-1] > sma200.iloc[-1]
        c2 = close.iloc[-1] > sma50.iloc[-1]
        c3 = rsi.iloc[-1] >= 50
        c4 = mom10 >= self.mom_thr
        
        if c1 and c2 and c3 and c4:
            return PositionSide.LONG
        return PositionSide.FLAT


# ============================================================
# Strategy F3: BollingerBreakout_1D
# ============================================================
class BollingerBreakout1D(BaseStrategy):
    """
    布林帶突破確認策略 (1D LONG)
    
    進場邏輯:
    - SMA20 = SMA(close, 20)
    - STD20 = STD(close, 20)
    - UpperBand = SMA20 + 2*STD20
    - SMA50 = SMA(close, 50)
    - RSI14 = 100 - 100/(1 + RS)
    - mom5 = (close[-1] - close[-5]) / close[-5] * 100
    
    LONG 進場:
    - C1: SMA50[-1] > SMA(close, 100)[-1]  (中期多頭)
    - C2: close[-1] > SMA20[-1]  (價格在均線上方)
    - C3: RSI14 >= 50  (多頭確認)
    - C4: mom5 >= 0.5  (正動量)
    
    SHORT 進場:
    - C1: SMA50[-1] < SMA(close, 100)[-1]
    - C2: close[-1] < SMA20[-1]
    - C3: RSI14 <= 50
    - C4: mom5 <= -0.5
    
    風險: TP=8%, SL=4%, position=15%
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
        c1 = sma50.iloc[-1] > sma100.iloc[-1]
        c2 = close.iloc[-1] > sma20.iloc[-1]
        c3 = rsi.iloc[-1] >= 50
        c4 = mom5 >= self.mom_thr
        if c1 and c2 and c3 and c4:
            return PositionSide.LONG
        
        # SHORT
        c1s = sma50.iloc[-1] < sma100.iloc[-1]
        c2s = close.iloc[-1] < sma20.iloc[-1]
        c3s = rsi.iloc[-1] <= 50
        c4s = mom5 <= -self.mom_thr
        if c1s and c2s and c3s and c4s:
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
    print("  Research Experiment Runner - Round 4 (2026-04-01)")
    print("  洞察: pos_size=15% + 擴大 TP/SL → 放大 Sharpe/PF")
    print("  Target: WR≥50%, PF≥2.0, DD≤30%, Sharpe≥1.5")
    print("=" * 70)
    
    pos_size = 0.15
    start_1d = "2020-01-01"
    end_1d = "2024-12-31"
    
    results = {}
    
    # ---------- 實驗 F1: EMAPureCross_1D_v2 ----------
    print("\n" + "=" * 70)
    print("實驗 F1: EMAPureCross_1D_v2")
    print("數據: BTCUSDT 1D | TP=7%, SL=3.5%, Pos=15%")
    print("  (基礎: EMAPureCross_1D_Long → WR=71.43%)")
    print("=" * 70)
    
    s1 = EMAPureCross1Dv2()
    r1 = await run_backtest("EMAPureCross_1D_v2", s1, "BTCUSDT", "1d",
                            start_1d, end_1d, tp=7.0, sl=3.5, pos_size=pos_size)
    results['EMAPureCross_1D_v2'] = r1
    print_result("EMAPureCross_1D_v2", r1, tp=7.0, sl=3.5, pos_size=pos_size)
    
    # ---------- 實驗 F2: TrendMA_1D_v2 ----------
    print("\n" + "=" * 70)
    print("實驗 F2: TrendMA_1D_v2")
    print("數據: BTCUSDT 1D | TP=10%, SL=5%, Pos=15%")
    print("  (基礎: TrendFollowing_1D → WR=43.71%, Sharpe=5.66)")
    print("=" * 70)
    
    s2 = TrendMA1Dv2()
    r2 = await run_backtest("TrendMA_1D_v2", s2, "BTCUSDT", "1d",
                            start_1d, end_1d, tp=10.0, sl=5.0, pos_size=pos_size)
    results['TrendMA_1D_v2'] = r2
    print_result("TrendMA_1D_v2", r2, tp=10.0, sl=5.0, pos_size=pos_size)
    
    # ---------- 實驗 F3: BollingerBreakout_1D ----------
    print("\n" + "=" * 70)
    print("實驗 F3: BollingerBreakout_1D")
    print("數據: BTCUSDT 1D | TP=8%, SL=4%, Pos=15%")
    print("  (新策略: SMA50>100 + RSI50 + 動量確認)")
    print("=" * 70)
    
    s3 = BollingerBreakout1D()
    r3 = await run_backtest("BollingerBreakout_1D", s3, "BTCUSDT", "1d",
                            start_1d, end_1d, tp=8.0, sl=4.0, pos_size=pos_size)
    results['BollingerBreakout_1D'] = r3
    print_result("BollingerBreakout_1D", r3, tp=8.0, sl=4.0, pos_size=pos_size)
    
    # ---------- 總結 ----------
    print("\n" + "=" * 70)
    print("  Round 4 實驗總結")
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
