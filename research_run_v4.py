"""
Research Experiment Runner - 2026-04-01 (Round 3)
位置: crypto-agent-platform/research_run_v4.py

## 失敗教訓總結
- TrendFollowing_1D (Round 2): Sharpe=5.66✅ DD=4.54%✅ WR=43.71% PF=1.65
  → WR 和 PF 略低，需要更嚴格進場過濾
- 核心問題: 信號不夠嚴格，導致太多弱勢進場降低 WR

## 第三輪設計核心洞察
成功公式（來自已驗證的 Regime MA Cross）:
  嚴格的多重確認 = 高 WR (70%+) + 適度 PF
  關鍵: 4 個條件 ALL 滿足 → WR 70%+

## 實驗設計

策略 E1: EMAPureCross_1D_Long (1D 純 EMA 交叉)
- 核心: EMA9/21 黃金交叉（只做多）
- 多重確認: +RSI55 + 動量1% + 價格>MA200
- 參考: 成功策略的進場邏輯，但調整為 TP=6%, SL=3%

策略 E2: PullbackRSI_1D (1D RSI 回調進場)
- 核心: 上漲後 RSI 回調至 50 附近再次向上
- 多重確認: SMA50>MA200 + RSI回調再啟 + 動量2%
- 捕捉: 慣性上漲中的再次啟動點

策略 E3: MultiTimeframe_1D (1D 多時間框架)
- 核心: 4H EMA 交叉 + 1D 趨勢確認
- 4H: EMA9/21 交叉
- 1D 確認: SMA20 > SMA50 + RSI > 50 + 動量1%
- 確保: 順大趨勢交易
"""

import sys, os, asyncio, pandas as pd, numpy as np

pkg_root = os.path.dirname(os.path.abspath(__file__))
if pkg_root not in sys.path:
    sys.path.insert(0, pkg_root)

from backtest.backtest_engine import BacktestEngine, BaseStrategy, PositionSide


# ============================================================
# Strategy E1: EMAPureCross_1D_Long
# ============================================================
class EMAPureCross1DLong(BaseStrategy):
    """
    純 EMA 交叉策略 (1D LONG)
    
    進場邏輯 (完整數學定義):
    - EMA9 = EMA(close, 9)
    - EMA21 = EMA(close, 21)
    - SMA200 = SMA(close, 200)
    - RSI14 = 100 - 100/(1 + RS), RS = SMA(漲幅,14)/SMA(跌幅,14)
    - mom7 = (close[-1] - close[-7]) / close[-7] * 100
    
    LONG 進場:
    - C1: EMA9[-1] > EMA21[-1] AND EMA9[-2] <= EMA21[-2]  (黃金交叉)
    - C2: RSI14 >= 55  (RSI 確認多頭)
    - C3: mom7 >= 1.0  (7日動量正向)
    - C4: close[-1] > SMA200[-1]  (長期多頭市場)
    
    風險: TP=6%, SL=3%, position=10%
    """
    
    def __init__(self):
        self.ema_fast = 9
        self.ema_slow = 21
        self.sma_long = 200
        self.mom_period = 7
        self.mom_thr = 1.0
        self.max_hold = 15
    
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
        
        # 黃金交叉
        gc = (ema9.iloc[-1] > ema21.iloc[-1]) and (ema9.iloc[-2] <= ema21.iloc[-2])
        
        # LONG: 4 個條件全部滿足
        if (gc and rsi.iloc[-1] >= 55 and mom7 >= self.mom_thr 
                and close.iloc[-1] > sma200.iloc[-1]):
            return PositionSide.LONG
        
        return PositionSide.FLAT


# ============================================================
# Strategy E2: PullbackRSI_1D
# ============================================================
class PullbackRSI1D(BaseStrategy):
    """
    RSI 回調進場策略 (1D LONG)
    
    進場邏輯:
    - SMA50 = SMA(close, 50)
    - SMA200 = SMA(close, 200)
    - RSI14 = 100 - 100/(1 + RS)
    - mom7 = (close[-1] - close[-7]) / close[-7] * 100
    
    LONG 進場:
    - C1: SMA50[-1] > SMA200[-1]  (長期多頭排列)
    - C2: RSI14 >= 45 AND RSI14 <= 60  (RSI 處於回升區間，不是超買)
    - C3: mom7 >= 1.0  (正在上漲)
    - C4: RSI14[-2] < RSI14[-1]  (RSI 正在改善/鉤頭)
    
    這個策略捕捉: 上漲趨勢中的短暫回調後再次啟動
    風險: TP=6%, SL=3%, position=10%
    """
    
    def __init__(self):
        self.sma_med = 50
        self.sma_long = 200
        self.mom_period = 7
        self.mom_thr = 1.0
        self.max_hold = 15
    
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
        
        if len(df) < max(self.sma_long + 1, self.mom_period + 1, 16):
            return PositionSide.FLAT
        
        close = df['close']
        sma50 = close.rolling(window=self.sma_med).mean()
        sma200 = close.rolling(window=self.sma_long).mean()
        rsi = self._compute_rsi(close, 14)
        mom7 = (close.iloc[-1] - close.iloc[-self.mom_period]) / close.iloc[-self.mom_period] * 100
        
        c1 = sma50.iloc[-1] > sma200.iloc[-1]
        c2 = 45 <= rsi.iloc[-1] <= 60
        c3 = mom7 >= self.mom_thr
        c4 = rsi.iloc[-2] < rsi.iloc[-1]  # RSI鉤頭
        
        if c1 and c2 and c3 and c4:
            return PositionSide.LONG
        
        return PositionSide.FLAT


# ============================================================
# Strategy E3: MultiTimeframe_1D
# ============================================================
class MultiTimeframe1D(BaseStrategy):
    """
    多時間框架策略 (1D 主框架)
    
    由於 engine 只支援單一 timeframe data傳入，
    我用不同周期的 SMA 模擬多時間框架:
    - 1D 等效: SMA20 (4H×5 = 20 個 4H bars)
    - 1W 等效: SMA60 (1D×60 = 60 天)
    
    進場邏輯:
    - SMA20 > SMA50 (日線多頭排列)
    - RSI14 >= 50 (日線確認)
    - mom5 >= 1.0% (日線動量)
    - close > SMA20 (價格在均線上方)
    
    這是多個時間框架的綜合確認:
    - SMA20>50 類似 4H EMA9>21 的日線版本
    - RSI+50 確認強度
    - 動量確認趨勢
    
    風險: TP=7%, SL=3.5%, position=10%
    """
    
    def __init__(self):
        self.sma_fast = 20
        self.sma_slow = 50
        self.mom_period = 5
        self.mom_thr = 1.0
        self.max_hold = 20
    
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
        sma50 = close.rolling(window=self.sma_slow).mean()
        rsi = self._compute_rsi(close, 14)
        mom5 = (close.iloc[-1] - close.iloc[-self.mom_period]) / close.iloc[-self.mom_period] * 100
        
        c1 = sma20.iloc[-1] > sma50.iloc[-1]
        c2 = rsi.iloc[-1] >= 50
        c3 = mom5 >= self.mom_thr
        c4 = close.iloc[-1] > sma20.iloc[-1]
        
        if c1 and c2 and c3 and c4:
            return PositionSide.LONG
        
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
    print("  Research Experiment Runner - Round 3 (2026-04-01)")
    print("  核心洞察: 嚴格多重確認 = 高 WR")
    print("  設計: 模仿 Regime MA Cross 的成功公式")
    print("  Target: WR≥50%, PF≥2.0, DD≤30%, Sharpe≥1.5")
    print("=" * 70)
    
    pos_size = 0.10
    start_1d = "2020-01-01"
    end_1d = "2024-12-31"
    
    results = {}
    
    # ---------- 實驗 E1: EMAPureCross_1D_Long ----------
    print("\n" + "=" * 70)
    print("實驗 E1: EMAPureCross_1D_Long")
    print("數據: BTCUSDT 1D | TP=6%, SL=3%, Pos=10%")
    print("  (模仿 Regime MA Cross 成功公式)")
    print("=" * 70)
    
    s1 = EMAPureCross1DLong()
    r1 = await run_backtest("EMAPureCross_1D_Long", s1, "BTCUSDT", "1d",
                            start_1d, end_1d, tp=6.0, sl=3.0, pos_size=pos_size)
    results['EMAPureCross_1D_Long'] = r1
    print_result("EMAPureCross_1D_Long", r1, tp=6.0, sl=3.0, pos_size=pos_size)
    
    # ---------- 實驗 E2: PullbackRSI_1D ----------
    print("\n" + "=" * 70)
    print("實驗 E2: PullbackRSI_1D")
    print("數據: BTCUSDT 1D | TP=6%, SL=3%, Pos=10%")
    print("  (RSI 回調進場: 捕捉上漲中的短暫回調)")
    print("=" * 70)
    
    s2 = PullbackRSI1D()
    r2 = await run_backtest("PullbackRSI_1D", s2, "BTCUSDT", "1d",
                            start_1d, end_1d, tp=6.0, sl=3.0, pos_size=pos_size)
    results['PullbackRSI_1D'] = r2
    print_result("PullbackRSI_1D", r2, tp=6.0, sl=3.0, pos_size=pos_size)
    
    # ---------- 實驗 E3: MultiTimeframe_1D ----------
    print("\n" + "=" * 70)
    print("實驗 E3: MultiTimeframe_1D")
    print("數據: BTCUSDT 1D | TP=7%, SL=3.5%, Pos=10%")
    print("  (多時間框架: SMA20>50 + RSI50 + 動量)")
    print("=" * 70)
    
    s3 = MultiTimeframe1D()
    r3 = await run_backtest("MultiTimeframe_1D", s3, "BTCUSDT", "1d",
                            start_1d, end_1d, tp=7.0, sl=3.5, pos_size=pos_size)
    results['MultiTimeframe_1D'] = r3
    print_result("MultiTimeframe_1D", r3, tp=7.0, sl=3.5, pos_size=pos_size)
    
    # ---------- 總結 ----------
    print("\n" + "=" * 70)
    print("  Round 3 實驗總結")
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
