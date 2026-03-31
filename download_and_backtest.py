#!/usr/bin/env python3
"""
download_and_backtest.py
下載幣安歷史 K 線數據，跑真實回測，更新排行榜
"""

import asyncio
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import aiohttp
import pandas as pd
import numpy as np

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from backtest.backtest_engine import (
    BacktestEngine, PositionSide, SimpleMovingAverageCrossover,
    BaseStrategy
)
from strategy_ranking import get_ranking_service, TradeDirection
from backtest.performance_metrics import PerformanceMetrics

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

BINANCE_BASE = "https://api.binance.com"
INTERVALS = {
    "1m":  ("1m",  365 * 24 * 60,  "btcusdt_1m.parquet"),
    "4h":  ("4h",  365 * 24 * 3,   "btcusdt_4h.parquet"),
    "1d":  ("1d",  365 * 10,       "btcusdt_1d.parquet"),
    "15m": ("15m", 365 * 24 * 4,   "btcusdt_15m.parquet"),
}


# ─────────────────────────────────────────────
# 1. Download historical K-lines
# ─────────────────────────────────────────────

async def fetch_klines(session, symbol, interval, start_ts, end_ts, limit=1000):
    """Fetch a single page of klines."""
    url = f"{BINANCE_BASE}/api/v3/klines"
    params = {
        "symbol": symbol.upper(),
        "interval": interval,
        "startTime": start_ts,
        "endTime": end_ts,
        "limit": limit,
    }
    async with session.get(url, params=params) as resp:
        if resp.status == 429:
            raise Exception("Rate limited")
        if resp.status != 200:
            raise Exception(f"Binance API error: {resp.status}")
        return await resp.json()


async def download_interval(symbol, interval_key, days_back, filename, pbar=None):
    """Download all historical data for a given interval."""
    filepath = DATA_DIR / filename
    
    # Check if already exists
    if filepath.exists():
        df_existing = pd.read_parquet(filepath)
        print(f"  [Cache] {filename} 已存在: {len(df_existing)} 行, {df_existing['open_time'].min()} ~ {df_existing['open_time'].max()}")
        return df_existing
    
    interval = interval_key
    end_ts = int(datetime.now().timestamp() * 1000)
    start_ts = int((datetime.now() - timedelta(days=days_back)).timestamp() * 1000)
    
    all_klines = []
    current_ts = start_ts
    total_expected = days_back * (60 if "m" in interval else (240 if "h" in interval else 1440)) // (1 if "m" in interval else (4 if "h" in interval else 1))
    total_expected = min(total_expected, 10000)
    
    cols = ["open_time","open","high","low","close","volume","close_time","quote_volume","trades","taker_buy_base","taker_buy_quote","ignore"]
    
    headers = {
        "open_time": "open_time",
        "open": float, "high": float, "low": float, "close": float,
        "volume": float, "close_time": float
    }
    
    count = 0
    while current_ts < end_ts:
        try:
            async with aiohttp.ClientSession() as session:
                data = await fetch_klines(session, symbol, interval, current_ts, end_ts)
                if not data:
                    break
                all_klines.extend(data)
                count += len(data)
                current_ts = data[-1][6] + 1  # close_time + 1ms
                
                if pbar:
                    pbar.update(len(data))
                
                # Rate limit: sleep 1-2 seconds
                await asyncio.sleep(1.2)
                
        except Exception as e:
            print(f"  Error at {current_ts}: {e}, retrying in 5s...")
            await asyncio.sleep(5)
    
    if not all_klines:
        print(f"  [Warn] No data fetched for {interval}")
        return pd.DataFrame()
    
    df = pd.DataFrame(all_klines, columns=cols)
    
    # Convert types
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")
    
    # Keep only needed columns
    df = df[["open_time", "open", "high", "low", "close", "volume"]].copy()
    df = df.sort_values("open_time").reset_index(drop=True)
    
    # Save as parquet
    df.to_parquet(filepath, index=False)
    print(f"  [Saved] {filename}: {len(df)} rows, {df['open_time'].min()} ~ {df['open_time'].max()}")
    
    return df


async def download_all():
    """Download all intervals."""
    print("\n" + "="*60)
    print("📥 下載 BTCUSDT 歷史 K 線數據")
    print("="*60)
    
    symbol = "BTCUSDT"
    
    for interval_key, (interval, days, filename) in INTERVALS.items():
        print(f"\n下載 {symbol} {interval} (最近 {days} 天)...")
        await download_interval(symbol, interval, days, filename)


# ─────────────────────────────────────────────
# 2. Load historical data from local files
# ─────────────────────────────────────────────

def load_historical_data(symbol: str, interval: str, 
                          start: datetime = None, 
                          end: datetime = None) -> pd.DataFrame:
    """
    從本地 Parquet 檔案讀取歷史數據。
    """
    filename_map = {
        "1m":  "btcusdt_1m.parquet",
        "4h":  "btcusdt_4h.parquet",
        "1d":  "btcusdt_1d.parquet",
        "15m": "btcusdt_15m.parquet",
    }
    
    filename = filename_map.get(interval)
    if not filename:
        raise ValueError(f"Unknown interval: {interval}")
    
    filepath = DATA_DIR / filename
    if not filepath.exists():
        raise FileNotFoundError(f"Local data not found: {filepath}")
    
    df = pd.read_parquet(filepath)
    df = df.sort_values("open_time").reset_index(drop=True)
    
    if start:
        df = df[df["open_time"] >= pd.Timestamp(start)]
    if end:
        df = df[df["open_time"] <= pd.Timestamp(end)]
    
    return df


# ─────────────────────────────────────────────
# 3. Backtest strategies on local data
# ─────────────────────────────────────────────

class MA_Cross_Strategy(BaseStrategy):
    """均線交叉策略 (MA_Cross)"""
    
    def __init__(self, fast=10, slow=30):
        self.fast = fast
        self.slow = slow
        self.name = f"MA_Cross({fast},{slow})"
    
    def generate_signal(self, market_data):
        df = list(market_data.values())[0]
        if len(df) < self.slow + 1:
            return PositionSide.FLAT
        
        fast_ma = df["close"].rolling(self.fast).mean()
        slow_ma = df["close"].rolling(self.slow).mean()
        
        if (fast_ma.iloc[-1] > slow_ma.iloc[-1] and 
            fast_ma.iloc[-2] <= slow_ma.iloc[-2]):
            return PositionSide.LONG
        elif (fast_ma.iloc[-1] < slow_ma.iloc[-1] and 
              fast_ma.iloc[-2] >= slow_ma.iloc[-2]):
            return PositionSide.SHORT
        return PositionSide.FLAT


class RSI_Reversal_Strategy(BaseStrategy):
    """RSI 反轉策略"""
    
    def __init__(self, period=14, oversold=30, overbought=70):
        self.period = period
        self.oversold = oversold
        self.overbought = overbought
        self.name = f"RSI_Reversal({period})"
    
    def calc_rsi(self, prices, period):
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def generate_signal(self, market_data):
        df = list(market_data.values())[0]
        if len(df) < self.period + 1:
            return PositionSide.FLAT
        
        rsi = self.calc_rsi(df["close"], self.period)
        current_rsi = rsi.iloc[-1]
        prev_rsi = rsi.iloc[-2]
        
        if prev_rsi <= self.oversold and current_rsi > self.oversold:
            return PositionSide.LONG
        elif prev_rsi >= self.overbought and current_rsi < self.overbought:
            return PositionSide.SHORT
        return PositionSide.FLAT


class BB_Breakout_Strategy(BaseStrategy):
    """布林帶突破策略"""
    
    def __init__(self, period=20, std_dev=2.0):
        self.period = period
        self.std_dev = std_dev
        self.name = f"BB_Breakout({period},{std_dev})"
    
    def generate_signal(self, market_data):
        df = list(market_data.values())[0]
        if len(df) < self.period + 1:
            return PositionSide.FLAT
        
        middle = df["close"].rolling(self.period).mean()
        std = df["close"].rolling(self.period).std()
        upper = middle + std * self.std_dev
        lower = middle - std * self.std_dev
        
        curr_close = df["close"].iloc[-1]
        prev_close = df["close"].iloc[-2]
        
        if prev_close <= upper.iloc[-2] and curr_close > upper.iloc[-1]:
            return PositionSide.LONG
        elif prev_close >= lower.iloc[-2] and curr_close < lower.iloc[-1]:
            return PositionSide.SHORT
        return PositionSide.FLAT


def run_backtest_on_data(strategy, symbol, interval, start_date, end_date):
    """使用本地數據跑回測。"""
    df = load_historical_data(symbol, interval, start_date, end_date)
    
    if df.empty:
        print(f"  [Warn] No data for {symbol} {interval} {start_date} ~ {end_date}")
        return None
    
    print(f"\n跑 {strategy.name} 回測: {len(df)} 根K線, {df['open_time'].min()} ~ {df['open_time'].max()}")
    
    # Create backtest engine
    engine = BacktestEngine()
    engine.initial_capital = 10000.0
    engine.max_position_size = 0.2
    engine.stop_loss = 0.02
    engine.take_profit = 0.05
    
    # Load dataframe
    engine.load_dataframe(symbol, df)
    engine.set_strategy(strategy)
    
    # Run synchronously
    import asyncio
    result = asyncio.get_event_loop().run_until_complete(engine.run())
    
    engine.close()
    
    return result, engine.trades


def run_all_backtests():
    """跑所有策略的回測。"""
    print("\n" + "="*60)
    print("📊 跑真實歷史數據回測")
    print("="*60)
    
    results = {}
    
    # Test on 1d interval with 10 years of data
    strategies = [
        MA_Cross_Strategy(fast=10, slow=30),
        RSI_Reversal_Strategy(period=14),
        BB_Breakout_Strategy(period=20, std_dev=2.0),
    ]
    
    for strategy in strategies:
        try:
            result, trades = run_backtest_on_data(
                strategy, "BTCUSDT", "1d",
                start_date="2017-04-01",
                end_date="2026-03-26"
            )
            results[strategy.name] = {
                "result": result,
                "trades": trades,
            }
        except Exception as e:
            print(f"  [Error] {strategy.name}: {e}")
            import traceback
            traceback.print_exc()
    
    return results


# ─────────────────────────────────────────────
# 4. Register results to ranking service
# ─────────────────────────────────────────────

def register_to_ranking(results):
    """把回測結果寫入排行榜。"""
    print("\n" + "="*60)
    print("🏆 更新排行榜")
    print("="*60)
    
    ranker = get_ranking_service()
    
    for strategy_name, data in results.items():
        result = data["result"]
        trades = data["trades"]
        
        if result is None:
            continue
        
        # Register strategy
        ranker.register_strategy(strategy_name)
        tracker = ranker.get_tracker(strategy_name)
        
        if tracker is None:
            print(f"  [Warn] No tracker for {strategy_name}")
            continue
        
        # Open and close trades from backtest
        for trade in trades:
            direction = TradeDirection.LONG if trade.side == PositionSide.LONG else TradeDirection.SHORT
            entry_price = trade.entry_price
            quantity = trade.quantity
            
            trade_id = tracker.open_trade(
                entry_price=entry_price,
                quantity=quantity,
                direction=direction
            )
            
            tracker.close_trade(
                trade_id,
                exit_price=trade.exit_price,
                commission=trade.commission
            )
        
        print(f"  ✅ {strategy_name}: {len(trades)} trades, Win Rate: {result.Win_Rate:.2f}%, Sharpe: {result.Sharpe_Ratio:.3f}")
    
    # Show rankings
    rankings = ranker.get_rankings()
    print(f"\n📋 排行榜 (updated: {rankings['updated_at']})")
    for entry in rankings["rankings"]:
        print(f"  #{entry['rank']} {entry['strategy']}: Score={entry['score']:.2f}, "
              f"Win={entry['win_rate']*100:.1f}%, Sharpe={entry['sharpe_ratio']:.3f}, "
              f"MaxDD={entry['max_drawdown']*100:.2f}%, PnL=${entry['total_pnl']:.2f}")


# ─────────────────────────────────────────────
# 5. Main
# ─────────────────────────────────────────────

async def main():
    print("="*60)
    print("🚀 幣安歷史數據下載 + 真實數據回測")
    print(f"   Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    # Step 1: Download data
    await download_all()
    
    # Step 2: Run backtests
    results = run_all_backtests()
    
    # Step 3: Register to ranking
    if results:
        register_to_ranking(results)
    
    print("\n" + "="*60)
    print("✅ 完成！")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(main())
