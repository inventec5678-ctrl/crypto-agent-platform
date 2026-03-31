#!/usr/bin/env python3
"""
Run backtests with local parquet data and populate the ranking system via API.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple
import sys
import os
import requests
import asyncio

# Add the platform directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backtest.backtest_engine import (
    BacktestEngine, BaseStrategy, PositionSide, 
    Trade as BacktestTrade
)


# ─── Strategy Implementations ────────────────────────────────────────────────

class MA_Cross_Strategy(BaseStrategy):
    """MA Crossover Strategy - Golden Cross / Death Cross"""
    
    def __init__(self, fast_period: int = 10, slow_period: int = 30):
        self.fast_period = fast_period
        self.slow_period = slow_period
    
    def generate_signal(self, market_data: Dict[str, pd.DataFrame]) -> PositionSide:
        df = list(market_data.values())[0]
        
        if len(df) < self.slow_period + 1:
            return PositionSide.FLAT
        
        close = df['close'].values
        fast_ma = pd.Series(close).rolling(window=self.fast_period).mean().values
        slow_ma = pd.Series(close).rolling(window=self.slow_period).mean().values
        
        if np.isnan(fast_ma[-1]) or np.isnan(slow_ma[-1]):
            return PositionSide.FLAT
        
        # Golden cross: fast crosses above slow
        if fast_ma[-1] > slow_ma[-1] and fast_ma[-2] <= slow_ma[-2]:
            return PositionSide.LONG
        # Death cross: fast crosses below slow
        elif fast_ma[-1] < slow_ma[-1] and fast_ma[-2] >= slow_ma[-2]:
            return PositionSide.FLAT
        
        return PositionSide.FLAT


class RSI_Reversal_Strategy(BaseStrategy):
    """RSI Reversal Strategy - Buy oversold, Sell overbought"""
    
    def __init__(self, period: int = 14, oversold: float = 30.0, overbought: float = 70.0):
        self.period = period
        self.oversold = oversold
        self.overbought = overbought
    
    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def generate_signal(self, market_data: Dict[str, pd.DataFrame]) -> PositionSide:
        df = list(market_data.values())[0]
        
        if len(df) < self.period + 1:
            return PositionSide.FLAT
        
        close = df['close']
        rsi = self._calculate_rsi(close, self.period)
        
        if np.isnan(rsi.iloc[-1]):
            return PositionSide.FLAT
        
        # RSI crosses above oversold -> BUY
        if rsi.iloc[-1] > self.oversold and rsi.iloc[-2] <= self.oversold:
            return PositionSide.LONG
        # RSI crosses below overbought -> SELL (close long)
        elif rsi.iloc[-1] < self.overbought and rsi.iloc[-2] >= self.overbought:
            return PositionSide.FLAT
        
        return PositionSide.FLAT


class BB_Breakout_Strategy(BaseStrategy):
    """Bollinger Bands Breakout Strategy"""
    
    def __init__(self, period: int = 20, std_dev: float = 2.0):
        self.period = period
        self.std_dev = std_dev
    
    def generate_signal(self, market_data: Dict[str, pd.DataFrame]) -> PositionSide:
        df = list(market_data.values())[0]
        
        if len(df) < self.period + 1:
            return PositionSide.FLAT
        
        close = df['close']
        middle = close.rolling(window=self.period).mean()
        std = close.rolling(window=self.period).std()
        upper = middle + (std * self.std_dev)
        lower = middle - (std * self.std_dev)
        
        if np.isnan(upper.iloc[-1]):
            return PositionSide.FLAT
        
        # Price breaks above upper band -> BUY
        if close.iloc[-1] > upper.iloc[-1] and close.iloc[-2] <= upper.iloc[-2]:
            return PositionSide.LONG
        # Price breaks below lower band -> SELL (close long)
        elif close.iloc[-1] < lower.iloc[-1] and close.iloc[-2] >= lower.iloc[-2]:
            return PositionSide.FLAT
        
        return PositionSide.FLAT


# ─── API Client ──────────────────────────────────────────────────────────────

API_BASE = "http://localhost:8000"

def register_strategy_via_api(strategy_name: str) -> bool:
    """Register a strategy via API."""
    try:
        resp = requests.post(f"{API_BASE}/api/strategies", json={"strategy_name": strategy_name})
        return resp.status_code in (200, 201)
    except Exception as e:
        print(f"  Error registering {strategy_name}: {e}")
        return False

def open_trade_via_api(strategy_name: str, entry_price: float, quantity: float, direction: str, entry_time: str) -> str:
    """Open a trade via API and return trade_id."""
    resp = requests.post(f"{API_BASE}/api/trades/open", json={
        "strategy_name": strategy_name,
        "entry_price": entry_price,
        "quantity": quantity,
        "direction": direction,
    })
    if resp.status_code != 200:
        print(f"  Error opening trade: {resp.text}")
        return None
    return resp.json().get("trade_id")

def close_trade_via_api(strategy_name: str, trade_id: str, exit_price: float, commission: float = 0.0) -> dict:
    """Close a trade via API."""
    resp = requests.post(f"{API_BASE}/api/trades/close", json={
        "strategy_name": strategy_name,
        "trade_id": trade_id,
        "exit_price": exit_price,
        "commission": commission,
    })
    if resp.status_code != 200:
        print(f"  Error closing trade: {resp.text}")
        return None
    return resp.json()


# ─── Main Backtest Runner ─────────────────────────────────────────────────────

def run_backtest_for_strategy(strategy_name: str, strategy_instance, df: pd.DataFrame, initial_capital: float = 10000.0) -> List[BacktestTrade]:
    """Run backtest for a single strategy and return trades."""
    
    print(f"\n{'='*60}")
    print(f"Running backtest for: {strategy_name}")
    print(f"{'='*60}")
    
    # Create engine
    engine = BacktestEngine()
    engine.initial_capital = initial_capital
    engine.cash = initial_capital
    engine.equity = initial_capital
    engine.max_position_size = 0.95
    
    # Load data
    engine.load_dataframe('BTCUSDT', df)
    
    # Set strategy
    engine.set_strategy(strategy_instance)
    
    # Run backtest
    async def run():
        return await engine.run()
    
    result = asyncio.run(run())
    
    print(f"\nBacktest Results for {strategy_name}:")
    print(f"  Initial Capital: ${result.initial_capital:,.2f}")
    print(f"  Final Equity:    ${result.final_equity:,.2f}")
    print(f"  Total Return:    {result.total_return_pct:.2f}%")
    print(f"  Total Trades:    {result.Total_Trades}")
    print(f"  Win Rate:         {result.Win_Rate:.2f}%")
    print(f"  Sharpe Ratio:     {result.Sharpe_Ratio:.4f}")
    print(f"  Max Drawdown:     {result.Max_Drawdown_Pct:.2f}%")
    print(f"  Profit Factor:    {result.Profit_Factor:.4f}")
    
    engine.close()
    
    return result.trades


def populate_ranking_via_api(strategy_name: str, trades: List[BacktestTrade], initial_capital: float = 10000.0):
    """Populate the ranking system via API with backtest trades."""
    
    print(f"\nPopulating ranking via API for: {strategy_name}")
    
    # Register strategy
    register_strategy_via_api(strategy_name)
    
    # Process trades
    success_count = 0
    for trade in trades:
        # Determine direction
        direction = "long" if trade.side == PositionSide.LONG else "short"
        
        # Calculate quantity
        if trade.exit_price > 0 and trade.entry_price > 0:
            price_change_pct = abs(trade.exit_price - trade.entry_price) / trade.entry_price
            if price_change_pct > 0:
                quantity = abs(trade.pnl / (price_change_pct * trade.entry_price)) if trade.pnl != 0 else 1000 / trade.entry_price
            else:
                quantity = 1000 / trade.entry_price
        else:
            quantity = 1000 / trade.entry_price if trade.entry_price > 0 else 1.0
        
        # Format entry time
        entry_time = trade.entry_time.isoformat() if hasattr(trade.entry_time, 'isoformat') else str(trade.entry_time)
        
        # Open trade
        trade_id = open_trade_via_api(strategy_name, trade.entry_price, quantity, direction, entry_time)
        
        if trade_id:
            # Close trade
            exit_time = trade.exit_time.isoformat() if hasattr(trade.exit_time, 'isoformat') else str(trade.exit_time)
            result = close_trade_via_api(strategy_name, trade_id, trade.exit_price, trade.commission)
            if result:
                success_count += 1
    
    print(f"  Successfully recorded {success_count}/{len(trades)} trades")


def get_rankings_via_api():
    """Get rankings from API."""
    resp = requests.get(f"{API_BASE}/api/rankings")
    if resp.status_code == 200:
        return resp.json()
    return None


def main():
    print("="*60)
    print("Backtest Runner - Using Local Parquet Data via API")
    print("="*60)
    
    # Load local data
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'btcusdt_1d.parquet')
    print(f"\nLoading data from: {data_path}")
    
    df = pd.read_parquet(data_path)
    print(f"Loaded {len(df)} rows of daily data")
    print(f"Date range: {df['open_time'].min()} to {df['open_time'].max()}")
    
    initial_capital = 10000.0
    
    # Define strategies
    strategies = [
        ("MA_Cross", MA_Cross_Strategy(fast_period=10, slow_period=30)),
        ("RSI_Reversal", RSI_Reversal_Strategy(period=14)),
        ("BB_Breakout", BB_Breakout_Strategy(period=20, std_dev=2.0)),
    ]
    
    # Run backtests and populate via API
    for display_name, strategy in strategies:
        trades = run_backtest_for_strategy(display_name, strategy, df, initial_capital)
        populate_ranking_via_api(display_name, trades, initial_capital)
    
    # Print final rankings
    print("\n" + "="*60)
    print("FINAL RANKINGS (from API)")
    print("="*60)
    
    rankings = get_rankings_via_api()
    
    if rankings:
        for entry in rankings['rankings']:
            print(f"\n  #{entry['rank']} {entry['strategy']}")
            print(f"      Score:        {entry['score']:.2f}")
            print(f"      Total Trades: {entry['total_trades']}")
            print(f"      Win Rate:     {entry['win_rate']:.2%}")
            print(f"      Profit Factor:{entry['profit_factor']:.4f}")
            print(f"      Sharpe Ratio: {entry['sharpe_ratio']:.4f}")
            print(f"      Max Drawdown: {entry['max_drawdown']:.2%}")
            print(f"      Total PnL:    ${entry['total_pnl']:.2f}")
        
        print(f"\n  Updated: {rankings['updated_at']}")
    else:
        print("  Failed to fetch rankings from API")
    
    print("\n" + "="*60)
    print("Backtest complete!")
    print("="*60)
    
    return rankings


if __name__ == "__main__":
    main()
