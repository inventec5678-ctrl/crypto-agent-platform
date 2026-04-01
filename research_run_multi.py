"""
Multi-Asset Test of best strategy: TrendMA_1D_v2
Testing on ETH, SOL, BNB to see if WR improves across different assets.
"""

import sys, os, asyncio, pandas as pd, numpy as np

pkg_root = os.path.dirname(os.path.abspath(__file__))
if pkg_root not in sys.path:
    sys.path.insert(0, pkg_root)

from backtest.backtest_engine import BacktestEngine, BaseStrategy, PositionSide


class TrendMA1DMulti(BaseStrategy):
    """Best performing strategy from rounds 1-5: SMA50>200 + close>SMA50 + RSI>=50 + mom>=2%"""
    
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


async def run_backtest(symbol, interval, tp, sl, pos_size, start_date, end_date):
    engine = BacktestEngine()
    engine.initial_capital = 10000.0
    engine.cash = 10000.0
    engine.equity = 10000.0
    engine.max_position_size = pos_size
    engine.stop_loss = sl / 100
    engine.take_profit = tp / 100
    
    parquet_path = f"data/{symbol.lower()}_{interval}.parquet"
    
    if os.path.exists(parquet_path):
        df = pd.read_parquet(parquet_path)
        # Convert open_time to datetime if it's int64 (milliseconds timestamp)
        if df['open_time'].dtype == 'int64':
            df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
        # Also ensure close_time is datetime if present
        if 'close_time' in df.columns:
            if df['close_time'].dtype == 'int64':
                df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
        
        df = df[(df['open_time'] >= pd.Timestamp(start_date)) & 
                 (df['open_time'] <= pd.Timestamp(end_date))]
        if df.empty:
            print(f"  ⚠️ No data for {symbol} {interval}")
            return None
        print(f"  📂 {parquet_path}: {len(df)} bars, {df['open_time'].min()} ~ {df['open_time'].max()}")
        engine.load_dataframe(symbol, df)
    else:
        print(f"  ⚠️ File not found: {parquet_path}")
        return None
    
    engine.set_strategy(TrendMA1DMulti())
    result = await engine.run()
    engine.close()
    
    return result


async def main():
    print("=" * 70)
    print("  Multi-Asset Test: TrendMA_1D_v2")
    print("  Best strategy (3/4 PASS) on ETH, SOL, BNB")
    print("=" * 70)
    
    tp = 10.0
    sl = 5.0
    pos_size = 0.15
    start = "2020-01-01"
    end = "2024-12-31"
    
    assets = ["ETHUSDT", "SOLUSDT", "BNBUSDT", "AVAXUSDT"]
    all_results = {}
    
    for asset in assets:
        print(f"\n{'='*50}")
        print(f"  {asset} 1D | TP={tp}%, SL={sl}%, Pos={pos_size*100:.0f}%")
        print(f"{'='*50}")
        
        result = await run_backtest(asset, "1d", tp, sl, pos_size, start, end)
        
        if result:
            wr_ok = result.Win_Rate >= 50.0
            pf_ok = result.Profit_Factor >= 2.0
            dd_ok = result.Max_Drawdown_Pct <= 30.0
            sharpe_ok = result.Sharpe_Ratio >= 1.5
            all_pass = wr_ok and pf_ok and dd_ok and sharpe_ok
            
            print(f"\n  {asset}: {'✅ PASS 4/4!' if all_pass else '❌ FAIL'}")
            print(f"  WR={result.Win_Rate:.2f}% (≥50% {'✅' if wr_ok else '❌'})")
            print(f"  PF={result.Profit_Factor:.4f} (≥2.0 {'✅' if pf_ok else '❌'})")
            print(f"  DD={result.Max_Drawdown_Pct:.2f}% (≤30% {'✅' if dd_ok else '❌'})")
            print(f"  Sharpe={result.Sharpe_Ratio:.4f} (≥1.5 {'✅' if sharpe_ok else '❌'})")
            print(f"  Trades={result.Total_Trades}")
            
            all_results[asset] = {
                'wr': result.Win_Rate, 'pf': result.Profit_Factor,
                'dd': result.Max_Drawdown_Pct, 'sharpe': result.Sharpe_Ratio,
                'trades': result.Total_Trades, 'ret': result.total_return_pct,
                'pass': all_pass
            }
    
    # Also rerun BTC for comparison
    print(f"\n{'='*50}")
    print(f"  BTCUSDT 1D (reference)")
    print(f"{'='*50}")
    result = await run_backtest("BTCUSDT", "1d", tp, sl, pos_size, start, end)
    if result:
        wr_ok = result.Win_Rate >= 50.0
        pf_ok = result.Profit_Factor >= 2.0
        dd_ok = result.Max_Drawdown_Pct <= 30.0
        sharpe_ok = result.Sharpe_Ratio >= 1.5
        all_pass = wr_ok and pf_ok and dd_ok and sharpe_ok
        
        print(f"\n  BTCUSDT: {'✅ PASS 4/4!' if all_pass else '❌ FAIL'}")
        print(f"  WR={result.Win_Rate:.2f}% (≥50% {'✅' if wr_ok else '❌'})")
        print(f"  PF={result.Profit_Factor:.4f} (≥2.0 {'✅' if pf_ok else '❌'})")
        print(f"  DD={result.Max_Drawdown_Pct:.2f}% (≤30% {'✅' if dd_ok else '❌'})")
        print(f"  Sharpe={result.Sharpe_Ratio:.4f} (≥1.5 {'✅' if sharpe_ok else '❌'})")
        print(f"  Trades={result.Total_Trades}")
        
        all_results['BTCUSDT'] = {
            'wr': result.Win_Rate, 'pf': result.Profit_Factor,
            'dd': result.Max_Drawdown_Pct, 'sharpe': result.Sharpe_Ratio,
            'trades': result.Total_Trades, 'ret': result.total_return_pct,
            'pass': all_pass
        }
    
    print("\n" + "=" * 70)
    print("  Multi-Asset Summary")
    print("=" * 70)
    
    for asset, r in all_results.items():
        status = "✅ PASS" if r['pass'] else "❌ FAIL"
        print(f"  {asset}: {status} | WR={r['wr']:.1f}%, PF={r['pf']:.2f}, DD={r['dd']:.1f}%, Sharpe={r['sharpe']:.2f}, Trades={r['trades']}")
    
    passed = [a for a, r in all_results.items() if r['pass']]
    if passed:
        print(f"\n🎉 {len(passed)} assets PASS 4/4!")
        for a in passed:
            print(f"  → {a}")
    else:
        print("\n😢 No assets pass 4/4.")


if __name__ == "__main__":
    asyncio.run(main())
