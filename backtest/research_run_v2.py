"""
Research Experiment Runner - 2026-04-01
Runs three new LONG strategy experiments per RESEARCH_PROCESS.md v2

Strategies:
1. VolMomentumBreakout_Long_v1  (1D BTCUSDT)
2. RSIStackedBull_Long_v1       (1D BTCUSDT)
3. EMAGoldenCross_4H_Long_v1   (4H BTCUSDT)
"""

import sys
import os
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 確保可以正確導入模組
if __name__ == "__main__":
    # 當直接執行此腳本時
    pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if pkg_root not in sys.path:
        sys.path.insert(0, pkg_root)

from backtest.backtest_engine import (
    BacktestEngine, BaseStrategy, PositionSide
)


# ============================================================
# Strategy 1: VolMomentumBreakout_Long_v1
# ============================================================
class VolMomentumBreakoutLong(BaseStrategy):
    """
    成交量動量突破策略 (LONG)
    
    進場邏輯（完整數學定義）:
    - vol_ratio = volume[-1] / SMA(volume, 20)[-1]  >= 2.0
    - mom7 = (close[-1] - close[-7]) / close[-7] * 100  >= 1.0
    - SMA(close, 20)[-1] > SMA(close, 50)[-1]  (多頭排列)
    - close[-1] > SMA(close, 20)[-1]
    
    風險參數（固定，不優化）:
    - TP = 6%, SL = 3%, max_hold = 15 天
    """
    
    def __init__(self):
        # 參數（不可事後修改）
        self.vol_mult = 2.0
        self.mom_thr = 1.0
        self.sma_fast = 20
        self.sma_slow = 50
        self.mom_period = 7
        self.max_hold = 15  # K 線數
    
    def generate_signal(self, market_data):
        df = list(market_data.values())[0]
        
        if len(df) < max(self.sma_slow, self.mom_period + 1, 21):
            return PositionSide.FLAT
        
        close = df['close']
        volume = df['volume']
        
        # 指標計算
        sma20 = close.rolling(window=self.sma_fast).mean()
        sma50 = close.rolling(window=self.sma_slow).mean()
        vol_sma20 = volume.rolling(window=20).mean()
        
        mom7 = (close.iloc[-1] - close.iloc[-self.mom_period]) / close.iloc[-self.mom_period] * 100
        
        vol_ratio = volume.iloc[-1] / vol_sma20.iloc[-1] if vol_sma20.iloc[-1] > 0 else 0
        
        # 多頭排列
        trend_up = sma20.iloc[-1] > sma50.iloc[-1]
        
        # 進場條件
        c1 = vol_ratio >= self.vol_mult
        c2 = mom7 >= self.mom_thr
        c3 = trend_up
        c4 = close.iloc[-1] > sma20.iloc[-1]
        
        if c1 and c2 and c3 and c4:
            return PositionSide.LONG
        
        return PositionSide.FLAT


# ============================================================
# Strategy 2: RSIStackedBull_Long_v1
# ============================================================
class RSIStackedBullLong(BaseStrategy):
    """
    RSI 均值回歸策略 (LONG)
    
    進場邏輯:
    - RSI(14) = 100 - 100/(1 + RS), RS = SMA(漲幅,14)/SMA(跌幅,14)
    - RSI(14) >= 55 AND <= 75
    - close[-1] > SMA(close, 200)[-1]  (牛市環境)
    - mom5 >= 0.5%
    
    風險參數: TP=6%, SL=3%, max_hold=15 天
    """
    
    def __init__(self):
        self.rsi_entry_min = 55
        self.rsi_entry_max = 75
        self.sma_long = 200
        self.mom_thr = 0.5
        self.mom_period = 5
        self.max_hold = 15
    
    def _compute_rsi(self, series, period=14):
        """計算 RSI"""
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def generate_signal(self, market_data):
        df = list(market_data.values())[0]
        
        if len(df) < max(self.sma_long + 1, self.mom_period + 1, 15):
            return PositionSide.FLAT
        
        close = df['close']
        
        # RSI(14)
        rsi = self._compute_rsi(close, 14)
        
        # 長期均線
        sma200 = close.rolling(window=self.sma_long).mean()
        
        # 5日動量
        mom5 = (close.iloc[-1] - close.iloc[-self.mom_period]) / close.iloc[-self.mom_period] * 100
        
        # 進場條件
        c1 = rsi.iloc[-1] >= self.rsi_entry_min and rsi.iloc[-1] <= self.rsi_entry_max
        c2 = close.iloc[-1] > sma200.iloc[-1]
        c3 = mom5 >= self.mom_thr
        
        if c1 and c2 and c3:
            return PositionSide.LONG
        
        return PositionSide.FLAT


# ============================================================
# Strategy 3: EMAGoldenCross_4H_Long_v1
# ============================================================
class EMAGoldenCross4HLong(BaseStrategy):
    """
    EMA 黃金交叉策略 (LONG, 4H)
    
    進場邏輯:
    - EMA9 = EMA(close, 9), EMA21 = EMA(close, 21)
    - 黃金交叉: EMA9[-1] > EMA21[-1] AND EMA9[-2] <= EMA21[-2]
    - RSI(14) >= 50
    - close[-1] > EMA21[-1]
    
    風險參數: TP=7%, SL=3%, max_hold=20 根 (4H)
    """
    
    def __init__(self):
        self.ema_fast = 9
        self.ema_slow = 21
        self.rsi_thr = 50
        self.tp = 7.0
        self.sl = 3.0
        self.max_hold = 20
    
    def _compute_ema(self, series, period):
        """計算 EMA"""
        return series.ewm(span=period, adjust=False).mean()
    
    def _compute_rsi(self, series, period=14):
        """計算 RSI"""
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def generate_signal(self, market_data):
        df = list(market_data.values())[0]
        
        if len(df) < max(self.ema_slow + 1, 15):
            return PositionSide.FLAT
        
        close = df['close']
        
        # EMA 計算
        ema9 = self._compute_ema(close, self.ema_fast)
        ema21 = self._compute_ema(close, self.ema_slow)
        
        # RSI
        rsi = self._compute_rsi(close, 14)
        
        # 黃金交叉
        gc = (ema9.iloc[-1] > ema21.iloc[-1]) and (ema9.iloc[-2] <= ema21.iloc[-2])
        
        # 進場條件
        c1 = gc
        c2 = rsi.iloc[-1] >= self.rsi_thr
        c3 = close.iloc[-1] > ema21.iloc[-1]
        
        if c1 and c2 and c3:
            return PositionSide.LONG
        
        return PositionSide.FLAT


# ============================================================
# Backtest Runner
# ============================================================

async def run_backtest(strategy_name, strategy_instance, symbol, interval,
                       start_date, end_date, tp, sl, max_hold, initial_capital=10000.0):
    """執行單次回測"""
    
    engine = BacktestEngine()
    engine.initial_capital = initial_capital
    engine.cash = initial_capital
    engine.equity = initial_capital
    engine.max_position_size = 0.2
    engine.stop_loss = sl / 100
    engine.take_profit = tp / 100
    
    # 嘗試從本地 parquet 加載
    parquet_path = f"/Users/changrunlin/.openclaw/workspace/crypto-agent-platform/data/{symbol.lower()}_{interval}.parquet"
    
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
    
    # 手動計算 max_hold 裁切（如果引擎不支持）
    # 這裡我們使用引擎原生結果
    
    engine.close()
    
    return result


async def main():
    print("=" * 70)
    print("  Research Experiment Runner - 2026-04-01")
    print("  目標: 找到 4/4 達標的 LONG 策略")
    print("  Target: WR≥50%, PF≥2.0, DD≤30%, Sharpe≥1.5")
    print("=" * 70)
    
    # 通用參數（學習 MA Cross v2 的成功配置）
    tp = 6.0   # 6% 止盈（PF=2 需要 TP/SL=2）
    sl = 3.0   # 3% 止損
    initial_capital = 10000.0
    
    # 數據範圍（使用完整可用數據）
    start_1d = "2020-01-01"
    end_1d = "2024-12-31"
    start_4h = "2023-01-01"
    end_4h = "2024-12-31"
    
    results = {}
    
    # ---------- 實驗 1: VolMomentumBreakout_Long_v1 ----------
    print("\n" + "=" * 70)
    print("實驗 1: VolMomentumBreakout_Long_v1")
    print("數據: BTCUSDT 1D")
    print("=" * 70)
    
    s1 = VolMomentumBreakoutLong()
    r1 = await run_backtest(
        "VolMomentumBreakout_Long_v1", s1,
        "BTCUSDT", "1d",
        start_1d, end_1d,
        tp=6.0, sl=3.0, max_hold=15
    )
    
    results['VolMomentumBreakout_Long_v1'] = r1
    
    print_result("VolMomentumBreakout_Long_v1", r1, tp, sl)
    
    # ---------- 實驗 2: RSIStackedBull_Long_v1 ----------
    print("\n" + "=" * 70)
    print("實驗 2: RSIStackedBull_Long_v1")
    print("數據: BTCUSDT 1D")
    print("=" * 70)
    
    s2 = RSIStackedBullLong()
    r2 = await run_backtest(
        "RSIStackedBull_Long_v1", s2,
        "BTCUSDT", "1d",
        start_1d, end_1d,
        tp=6.0, sl=3.0, max_hold=15
    )
    
    results['RSIStackedBull_Long_v1'] = r2
    
    print_result("RSIStackedBull_Long_v1", r2, tp, sl)
    
    # ---------- 實驗 3: EMAGoldenCross_4H_Long_v1 ----------
    print("\n" + "=" * 70)
    print("實驗 3: EMAGoldenCross_4H_Long_v1")
    print("數據: BTCUSDT 4H")
    print("=" * 70)
    
    s3 = EMAGoldenCross4HLong()
    r3 = await run_backtest(
        "EMAGoldenCross_4H_Long_v1", s3,
        "BTCUSDT", "4h",
        start_4h, end_4h,
        tp=7.0, sl=3.0, max_hold=20
    )
    
    results['EMAGoldenCross_4H_Long_v1'] = r3
    
    print_result("EMAGoldenCross_4H_Long_v1", r3, tp=7.0, sl=3.0)
    
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
        print(f"  PF={r.Profit_Factor:.2f} (≥2.0 {'✅' if pf_ok else '❌'})")
        print(f"  DD={r.Max_Drawdown_Pct:.2f}% (≤30% {'✅' if dd_ok else '❌'})")
        print(f"  Sharpe={r.Sharpe_Ratio:.3f} (≥1.5 {'✅' if sharpe_ok else '❌'})")
        print(f"  Trades={r.Total_Trades}")
        
        summary.append({
            'name': name,
            'wr': r.Win_Rate,
            'pf': r.Profit_Factor,
            'dd': r.Max_Drawdown_Pct,
            'sharpe': r.Sharpe_Ratio,
            'trades': r.Total_Trades,
            'pass': all_pass
        })
    
    return summary


def print_result(name, result, tp, sl):
    """打印單個結果"""
    print(f"\n{'─' * 50}")
    print(f"  {name}")
    print(f"{'─' * 50}")
    print(f"  總交易:   {result.Total_Trades} 筆")
    print(f"  勝率:     {result.Win_Rate:.2f}%")
    print(f"  盈虧比:   {result.Profit_Factor:.4f}")
    print(f"  最大回撤: {result.Max_Drawdown_Pct:.2f}%")
    print(f"  Sharpe:   {result.Sharpe_Ratio:.4f}")
    print(f"  TP:       {tp}% / SL: {sl}%")
    if result.trades:
        avg_dur = np.mean([t.duration for t in result.trades]) / 86400 if result.trades else 0
        print(f"  平均持倉: {avg_dur:.1f} 天")


if __name__ == "__main__":
    summary = asyncio.run(main())
    
    # 檢查是否有 PASS 的策略
    passed = [s for s in summary if s['pass']]
    if passed:
        print(f"\n🎉 找到 {len(passed)} 個 4/4 達標策略!")
        for p in passed:
            print(f"  → {p['name']}")
    else:
        print("\n😢 沒有策略達到 4/4 標準。")
