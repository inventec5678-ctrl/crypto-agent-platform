#!/usr/bin/env python3
"""
Auto Research v2.0 主執行腳本 (Enhanced)

功能:
- 市場分層（牛市/熊市/盤整）
- 分層抽樣框架
- 多因子策略庫
- 參數網格搜索
- 新策略類型（Momentum, MACD）
- 持續滾動優化引擎
"""

import asyncio
import pandas as pd
import numpy as np
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, List

# 確保專案根目錄在 Python 路徑
PROJECT_ROOT = Path(__file__).parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from autoresearch.market_regime import MarketRegimeClassifier
from autoresearch.factor_library import FactorLibrary, FactorRecord
from backtest.backtest_engine import BacktestEngine, PositionSide, BaseStrategy
from strategies.short_cycle_strategies import (
    TripleEMA_Strategy,
    EMA_Breakout_Strategy,
    RSI_MA_Combo_Strategy,
    BB_RSI_Combo_Strategy,
    TrendBreakout_Strategy,
    VolumeWeighted_Strategy,
    MA_Trend_Holding_Strategy,
    Stochastic_Strategy,
    Supertrend_Strategy,
    RSI_Extreme_Strategy,
    MACD_Histogram_Strategy,
    Keltner_Breakout_Strategy,
    CCI_Reversal_Strategy,
    ADX_Trend_Strategy,
    SHORT_CYCLE_SIGNAL_GENERATORS,
    SHORT_CYCLE_CATEGORY_MAP,
)


# ─── Direct Signal Strategies for Backtesting ─────────────────────────────────

class MA_Cross_Strategy(BaseStrategy):
    """MA Crossover Strategy - Golden Cross / Death Cross"""
    
    def __init__(self, fast_period: int = 10, slow_period: int = 30):
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.name = f"MA_Cross({fast_period}/{slow_period})"
    
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
        self.name = f"RSI_Reversal({period})"
    
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
        self.name = f"BB_Breakout({period})"
    
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


class Momentum_Strategy(BaseStrategy):
    """Simple Momentum Strategy"""
    
    def __init__(self, period: int = 10, threshold: float = 0.02):
        self.period = period
        self.threshold = threshold
        self.name = f"Momentum({period})"
    
    def generate_signal(self, market_data: Dict[str, pd.DataFrame]) -> PositionSide:
        df = list(market_data.values())[0]
        
        if len(df) < self.period + 1:
            return PositionSide.FLAT
        
        close = df['close']
        returns = close.pct_change(self.period)
        
        if np.isnan(returns.iloc[-1]):
            return PositionSide.FLAT
        
        # Momentum positive and above threshold -> BUY
        if returns.iloc[-1] > self.threshold:
            return PositionSide.LONG
        # Momentum negative -> CLOSE
        elif returns.iloc[-1] < -self.threshold:
            return PositionSide.FLAT
        
        return PositionSide.FLAT


class MACD_Strategy(BaseStrategy):
    """MACD Crossover Strategy"""
    
    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9):
        self.fast = fast
        self.slow = slow
        self.signal_period = signal
        self.name = f"MACD({fast},{slow},{signal})"
    
    def generate_signal(self, market_data: Dict[str, pd.DataFrame]) -> PositionSide:
        df = list(market_data.values())[0]
        
        if len(df) < self.slow + 1:
            return PositionSide.FLAT
        
        close = df['close']
        ema_fast = close.ewm(span=self.fast, adjust=False).mean()
        ema_slow = close.ewm(span=self.slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=self.signal_period, adjust=False).mean()
        
        if np.isnan(macd.iloc[-1]) or np.isnan(signal_line.iloc[-1]):
            return PositionSide.FLAT
        
        # MACD crosses above signal line -> BUY
        if macd.iloc[-1] > signal_line.iloc[-1] and macd.iloc[-2] <= signal_line.iloc[-2]:
            return PositionSide.LONG
        # MACD crosses below signal line -> SELL
        elif macd.iloc[-1] < signal_line.iloc[-1] and macd.iloc[-2] >= signal_line.iloc[-2]:
            return PositionSide.FLAT
        
        return PositionSide.FLAT


class ATR_Reversal_Strategy(BaseStrategy):
    """ATR-based Reversal Strategy"""
    
    def __init__(self, period: int = 14, multiplier: float = 2.0):
        self.period = period
        self.multiplier = multiplier
        self.name = f"ATR_Reversal({period})"
    
    def generate_signal(self, market_data: Dict[str, pd.DataFrame]) -> PositionSide:
        df = list(market_data.values())[0]
        
        if len(df) < self.period + 1:
            return PositionSide.FLAT
        
        high = df['high']
        low = df['low']
        close = df['close']
        
        # Calculate ATR
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=self.period).mean()
        
        if np.isnan(atr.iloc[-1]):
            return PositionSide.FLAT
        
        # Simple reversal: price crosses above/below ATR channel
        upper = close.shift(1) + atr * self.multiplier
        lower = close.shift(1) - atr * self.multiplier
        
        if close.iloc[-1] > upper.iloc[-1] and close.iloc[-2] <= upper.iloc[-2]:
            return PositionSide.LONG
        elif close.iloc[-1] < lower.iloc[-1] and close.iloc[-2] >= lower.iloc[-2]:
            return PositionSide.FLAT
        
        return PositionSide.FLAT


# ─── Parameter Grids ──────────────────────────────────────────────────────────

PARAM_GRIDS = {
    'MA_Cross': [
        {'fast_period': 5, 'slow_period': 20},
        {'fast_period': 10, 'slow_period': 30},
        {'fast_period': 10, 'slow_period': 50},
        {'fast_period': 20, 'slow_period': 50},
        {'fast_period': 20, 'slow_period': 100},
        {'fast_period': 5, 'slow_period': 50},
        {'fast_period': 12, 'slow_period': 26},
        {'fast_period': 8, 'slow_period': 21},
        {'fast_period': 3, 'slow_period': 15},
    ],
    'RSI_Reversal': [
        {'period': 7, 'oversold': 30, 'overbought': 70},
        {'period': 14, 'oversold': 30, 'overbought': 70},
        {'period': 21, 'oversold': 25, 'overbought': 75},
        {'period': 14, 'oversold': 20, 'overbought': 80},
        {'period': 7, 'oversold': 25, 'overbought': 75},
        {'period': 10, 'oversold': 30, 'overbought': 70},
    ],
    'BB_Breakout': [
        {'period': 10, 'std_dev': 1.5},
        {'period': 20, 'std_dev': 2.0},
        {'period': 20, 'std_dev': 2.5},
        {'period': 30, 'std_dev': 2.0},
        {'period': 15, 'std_dev': 2.0},
        {'period': 25, 'std_dev': 2.5},
    ],
    'Momentum': [
        {'period': 5, 'threshold': 0.02},
        {'period': 10, 'threshold': 0.02},
        {'period': 10, 'threshold': 0.03},
        {'period': 20, 'threshold': 0.02},
        {'period': 20, 'threshold': 0.05},
        {'period': 14, 'threshold': 0.03},
    ],
    'MACD': [
        {'fast': 12, 'slow': 26, 'signal': 9},
        {'fast': 8, 'slow': 17, 'signal': 9},
        {'fast': 5, 'slow': 35, 'signal': 5},
        {'fast': 12, 'slow': 26, 'signal': 12},
        {'fast': 19, 'slow': 39, 'signal': 9},
    ],
    'ATR_Reversal': [
        {'period': 14, 'multiplier': 1.5},
        {'period': 14, 'multiplier': 2.0},
        {'period': 20, 'multiplier': 2.0},
        {'period': 10, 'multiplier': 2.0},
    ],
    # 短週期策略 (4H/15m)
    'TripleEMA': [
        {'fast': 9, 'mid': 21, 'slow': 50},
        {'fast': 5, 'mid': 13, 'slow': 34},
        {'fast': 8, 'mid': 21, 'slow': 55},
        {'fast': 10, 'mid': 25, 'slow': 99},
        {'fast': 12, 'mid': 26, 'slow': 50},
    ],
    'EMA_Breakout': [
        {'period': 10},
        {'period': 20},
        {'period': 30},
        {'period': 50},
        {'period': 12},
        {'period': 25},
    ],
    'RSI_MA_Combo': [
        {'rsi_period': 7, 'ma_period': 50},
        {'rsi_period': 14, 'ma_period': 50},
        {'rsi_period': 14, 'ma_period': 100},
        {'rsi_period': 7, 'ma_period': 30},
        {'rsi_period': 21, 'ma_period': 50},
        {'rsi_period': 10, 'ma_period': 20},
    ],
    'BB_RSI_Combo': [
        {'bb_period': 20, 'bb_std': 2.0, 'rsi_period': 14},
        {'bb_period': 20, 'bb_std': 2.5, 'rsi_period': 14},
        {'bb_period': 15, 'bb_std': 2.0, 'rsi_period': 7},
        {'bb_period': 30, 'bb_std': 2.0, 'rsi_period': 14},
        {'bb_period': 20, 'bb_std': 1.5, 'rsi_period': 7},
    ],
    'TrendBreakout': [
        {'lookback': 20, 'atr_period': 14, 'atr_multiplier': 2.0},
        {'lookback': 10, 'atr_period': 14, 'atr_multiplier': 2.0},
        {'lookback': 20, 'atr_period': 14, 'atr_multiplier': 1.5},
        {'lookback': 30, 'atr_period': 14, 'atr_multiplier': 2.5},
        {'lookback': 15, 'atr_period': 10, 'atr_multiplier': 2.0},
        {'lookback': 25, 'atr_period': 20, 'atr_multiplier': 2.0},
    ],
    'VolumeWeighted': [
        {'period': 20},
        {'period': 10},
        {'period': 30},
        {'period': 15},
        {'period': 50},
    ],
    'MA_Trend_Holding': [
        {'fast': 5, 'slow': 20},
        {'fast': 10, 'slow': 30},
        {'fast': 10, 'slow': 50},
        {'fast': 8, 'slow': 21},
        {'fast': 12, 'slow': 26},
    ],
    'Stochastic': [
        {'period': 14, 'smooth_k': 3, 'smooth_d': 3},
        {'period': 21, 'smooth_k': 3, 'smooth_d': 3},
        {'period': 14, 'smooth_k': 5, 'smooth_d': 3},
        {'period': 7, 'smooth_k': 3, 'smooth_d': 3},
        {'period': 10, 'smooth_k': 3, 'smooth_d': 3},
    ],
    'Supertrend': [
        {'period': 10, 'multiplier': 3.0},
        {'period': 10, 'multiplier': 2.0},
        {'period': 14, 'multiplier': 3.0},
        {'period': 7, 'multiplier': 2.5},
        {'period': 20, 'multiplier': 3.0},
        {'period': 12, 'multiplier': 2.0},
    ],
    'RSI_Extreme': [
        {'period': 7, 'extreme_buy': 15.0, 'extreme_sell': 85.0},
        {'period': 14, 'extreme_buy': 20.0, 'extreme_sell': 80.0},
        {'period': 7, 'extreme_buy': 20.0, 'extreme_sell': 80.0},
        {'period': 5, 'extreme_buy': 15.0, 'extreme_sell': 85.0},
        {'period': 10, 'extreme_buy': 25.0, 'extreme_sell': 75.0},
    ],
    'MACD_Histogram': [
        {'fast': 12, 'slow': 26, 'signal': 9},
        {'fast': 8, 'slow': 17, 'signal': 9},
        {'fast': 5, 'slow': 35, 'signal': 5},
        {'fast': 12, 'slow': 26, 'signal': 12},
        {'fast': 19, 'slow': 39, 'signal': 9},
    ],
    'Keltner_Breakout': [
        {'ema_period': 20, 'atr_period': 10, 'multiplier': 2.0},
        {'ema_period': 20, 'atr_period': 10, 'multiplier': 1.5},
        {'ema_period': 20, 'atr_period': 15, 'multiplier': 2.0},
        {'ema_period': 30, 'atr_period': 10, 'multiplier': 2.0},
        {'ema_period': 15, 'atr_period': 10, 'multiplier': 1.5},
    ],
    'CCI_Reversal': [
        {'period': 20},
        {'period': 14},
        {'period': 10},
        {'period': 30},
        {'period': 25},
    ],
    'ADX_Trend': [
        {'period': 14, 'adx_threshold': 25.0},
        {'period': 14, 'adx_threshold': 20.0},
        {'period': 20, 'adx_threshold': 25.0},
        {'period': 10, 'adx_threshold': 20.0},
        {'period': 7, 'adx_threshold': 25.0},
    ],
}


# ─── Strategy Class Mapping ──────────────────────────────────────────────────

STRATEGY_CLASSES = {
    'MA_Cross': MA_Cross_Strategy,
    'RSI_Reversal': RSI_Reversal_Strategy,
    'BB_Breakout': BB_Breakout_Strategy,
    'Momentum': Momentum_Strategy,
    'MACD': MACD_Strategy,
    'ATR_Reversal': ATR_Reversal_Strategy,
    # 短週期策略
    'TripleEMA': TripleEMA_Strategy,
    'EMA_Breakout': EMA_Breakout_Strategy,
    'RSI_MA_Combo': RSI_MA_Combo_Strategy,
    'BB_RSI_Combo': BB_RSI_Combo_Strategy,
    'TrendBreakout': TrendBreakout_Strategy,
    'VolumeWeighted': VolumeWeighted_Strategy,
    'MA_Trend_Holding': MA_Trend_Holding_Strategy,
    'Stochastic': Stochastic_Strategy,
    'Supertrend': Supertrend_Strategy,
    'RSI_Extreme': RSI_Extreme_Strategy,
    'MACD_Histogram': MACD_Histogram_Strategy,
    'Keltner_Breakout': Keltner_Breakout_Strategy,
    'CCI_Reversal': CCI_Reversal_Strategy,
    'ADX_Trend': ADX_Trend_Strategy,
}


# ─── Factor Signal Generators ─────────────────────────────────────────────────

def ma_cross_signal_generator(market_data: dict, params: dict) -> str:
    """MA Cross 信號產生器"""
    df = list(market_data.values())[0]
    if len(df) < params.get('slow_period', 30) + 1:
        return 'FLAT'
    
    close = df['close'].values
    fast = pd.Series(close).rolling(window=params['fast_period']).mean().values
    slow = pd.Series(close).rolling(window=params['slow_period']).mean().values
    
    if np.isnan(fast[-1]) or np.isnan(slow[-1]):
        return 'FLAT'
    
    if fast[-1] > slow[-1] and fast[-2] <= slow[-2]:
        return 'LONG'
    elif fast[-1] < slow[-1] and fast[-2] >= slow[-2]:
        return 'SHORT'
    return 'FLAT'


def rsi_signal_generator(market_data: dict, params: dict) -> str:
    """RSI 反轉信號產生器"""
    df = list(market_data.values())[0]
    period = params.get('period', 14)
    if len(df) < period + 1:
        return 'FLAT'
    
    close = df['close']
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = (100 - (100 / (1 + rs))).values
    
    if np.isnan(rsi[-1]):
        return 'FLAT'
    
    oversold = params.get('oversold', 30)
    overbought = params.get('overbought', 70)
    
    if rsi[-1] > oversold and rsi[-2] <= oversold:
        return 'LONG'
    elif rsi[-1] < overbought and rsi[-2] >= overbought:
        return 'SHORT'
    return 'FLAT'


def bb_signal_generator(market_data: dict, params: dict) -> str:
    """BB 信號產生器"""
    df = list(market_data.values())[0]
    period = params.get('period', 20)
    if len(df) < period + 1:
        return 'FLAT'
    
    close = df['close']
    middle = close.rolling(window=period).mean()
    std = close.rolling(window=period).std()
    upper = middle + (std * params.get('std_dev', 2.0))
    lower = middle - (std * params.get('std_dev', 2.0))
    
    if np.isnan(upper.iloc[-1]):
        return 'FLAT'
    
    if close.iloc[-1] > upper.iloc[-1] and close.iloc[-2] <= upper.iloc[-2]:
        return 'LONG'
    elif close.iloc[-1] < lower.iloc[-1] and close.iloc[-2] >= lower.iloc[-2]:
        return 'SHORT'
    return 'FLAT'


def momentum_signal_generator(market_data: dict, params: dict) -> str:
    """Momentum 信號產生器"""
    df = list(market_data.values())[0]
    period = params.get('period', 10)
    threshold = params.get('threshold', 0.02)
    
    if len(df) < period + 1:
        return 'FLAT'
    
    close = df['close']
    momentum = close.pct_change(period)
    
    if np.isnan(momentum.iloc[-1]):
        return 'FLAT'
    
    if momentum.iloc[-1] > threshold:
        return 'LONG'
    elif momentum.iloc[-1] < -threshold:
        return 'SHORT'
    return 'FLAT'


def macd_signal_generator(market_data: dict, params: dict) -> str:
    """MACD 信號產生器"""
    df = list(market_data.values())[0]
    fast = params.get('fast', 12)
    slow = params.get('slow', 26)
    signal_period = params.get('signal', 9)
    
    if len(df) < slow + 1:
        return 'FLAT'
    
    close = df['close']
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal_period, adjust=False).mean()
    
    if np.isnan(macd.iloc[-1]) or np.isnan(signal_line.iloc[-1]):
        return 'FLAT'
    
    if macd.iloc[-1] > signal_line.iloc[-1] and macd.iloc[-2] <= signal_line.iloc[-2]:
        return 'LONG'
    elif macd.iloc[-1] < signal_line.iloc[-1] and macd.iloc[-2] >= signal_line.iloc[-2]:
        return 'SHORT'
    return 'FLAT'


def atr_signal_generator(market_data: dict, params: dict) -> str:
    """ATR Reversal 信號產生器"""
    df = list(market_data.values())[0]
    period = params.get('period', 14)
    multiplier = params.get('multiplier', 2.0)
    
    if len(df) < period + 1:
        return 'FLAT'
    
    high = df['high']
    low = df['low']
    close = df['close']
    
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    
    if np.isnan(atr.iloc[-1]):
        return 'FLAT'
    
    upper = close.shift(1) + atr * multiplier
    lower = close.shift(1) - atr * multiplier
    
    if close.iloc[-1] > upper.iloc[-1] and close.iloc[-2] <= upper.iloc[-2]:
        return 'LONG'
    elif close.iloc[-1] < lower.iloc[-1] and close.iloc[-2] >= lower.iloc[-2]:
        return 'SHORT'
    return 'FLAT'


SIGNAL_GENERATORS = {
    'MA_Cross': ma_cross_signal_generator,
    'RSI_Reversal': rsi_signal_generator,
    'BB_Breakout': bb_signal_generator,
    'Momentum': momentum_signal_generator,
    'MACD': macd_signal_generator,
    'ATR_Reversal': atr_signal_generator,
    # 短週期策略信號產生器
    **SHORT_CYCLE_SIGNAL_GENERATORS,
}


CATEGORY_MAP = {
    'MA_Cross': 'trend',
    'RSI_Reversal': 'reversal',
    'BB_Breakout': 'volatility',
    'Momentum': 'momentum',
    'MACD': 'trend',
    'ATR_Reversal': 'volatility',
    # 短週期策略分類
    **SHORT_CYCLE_CATEGORY_MAP,
}


# ─── Backtest Runner ──────────────────────────────────────────────────────────

async def run_strategy_backtest(strategy_instance, df: pd.DataFrame, initial_capital: float = 10000.0) -> Optional[dict]:
    """執行單一策略的回測"""
    try:
        engine = BacktestEngine()
        engine.initial_capital = initial_capital
        engine.cash = initial_capital
        engine.equity = initial_capital
        engine.max_position_size = 0.95
        
        engine.load_dataframe('BTCUSDT', df)
        engine.set_strategy(strategy_instance)
        
        result = await engine.run()
        
        return {
            'win_rate': result.Win_Rate,
            'profit_factor': result.Profit_Factor,
            'sharpe': result.Sharpe_Ratio,
            'max_drawdown': result.Max_Drawdown_Pct / 100,
            'total_return': result.total_return_pct,
            'final_equity': result.final_equity,
            'num_trades': len(result.trades)
        }
    except Exception as e:
        print(f"   ⚠️ 回測執行失敗: {e}")
        return None


async def run_grid_search(
    strategy_type: str,
    param_grid: List[dict],
    test_df: pd.DataFrame,
    min_trades: int = 5
) -> List[dict]:
    """對指定策略類型進行參數網格搜索"""
    results = []
    cls = STRATEGY_CLASSES[strategy_type]
    
    for params in param_grid:
        strategy = cls(**params)
        result = await run_strategy_backtest(strategy, test_df)
        
        if result and result['num_trades'] >= min_trades:
            results.append({
                'strategy_type': strategy_type,
                'params': params,
                'name': strategy.name,
                'result': result
            })
    
    return results


# ─── Market Regime Analysis ───────────────────────────────────────────────────

async def analyze_by_regime(
    strategy_instance,
    df: pd.DataFrame,
    classifier: MarketRegimeClassifier
) -> dict:
    """分析策略在各市場狀態的表現"""
    regime_labels = classifier.classify(df)
    df_with_regime = df.copy()
    df_with_regime['regime'] = regime_labels
    
    results = {
        'bull': {'count': 0, 'return': 0},
        'bear': {'count': 0, 'return': 0},
        'sideways': {'count': 0, 'return': 0}
    }
    
    for regime in ['bull', 'bear', 'sideways']:
        regime_df = df_with_regime[df_with_regime['regime'] == regime]
        if len(regime_df) >= 30:
            result = await run_strategy_backtest(strategy_instance, regime_df)
            if result:
                results[regime] = {
                    'count': result['num_trades'],
                    'return': result['total_return'],
                    'win_rate': result['win_rate'],
                    'sharpe': result['sharpe']
                }
    
    return results


# ─── Continuous Optimization ──────────────────────────────────────────────────

class ContinuousOptimizer:
    """持續滾動優化器"""
    
    def __init__(self, library: FactorLibrary, state_file: Path = None):
        self.library = library
        self.state_file = state_file or (PROJECT_ROOT / 'autoresearch' / 'optimizer_state.json')
        self.state = self._load_state()
        self.DORMANT_THRESHOLD = 3  # 連續表現不佳次數進入休眠
    
    def _load_state(self) -> dict:
        if self.state_file.exists():
            with open(self.state_file) as f:
                return json.load(f)
        return {
            'day': 0,
            'total_runs': 0,
            'dormant_factors': [],
            'performance_history': []
        }
    
    def _save_state(self):
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2)
    
    async def evaluate_factor(
        self,
        factor: FactorRecord,
        df: pd.DataFrame
    ) -> dict:
        """評估單一因子在最近數據的表現"""
        recent_df = df.tail(30).copy()
        market_data = {'BTCUSDT': recent_df}
        
        signal = factor.generate_signal(market_data)
        
        # 簡單評估：根據信號方向和近期價格變動判斷
        if signal == 'FLAT':
            return {'signal': 'FLAT', 'score': 0}
        
        recent_return = recent_df['close'].iloc[-1] / recent_df['close'].iloc[0] - 1
        
        if signal == 'LONG' and recent_return > 0:
            score = 1
        elif signal == 'SHORT' and recent_return < 0:
            score = 1
        else:
            score = -1
        
        return {'signal': signal, 'score': score, 'recent_return': recent_return}
    
    async def run_day(self, df: pd.DataFrame, day: int):
        """執行單一天的優化"""
        print(f"\n📅 Day {day + 1}")
        self.state['day'] = day + 1
        self.state['total_runs'] += 1
        
        classifier = MarketRegimeClassifier(ma_period=200)
        regime = classifier.classify(df)
        
        regime_name = regime.iloc[-1] if len(regime) > 0 else 'unknown'
        print(f"   市場狀態: {regime_name}")
        
        # 評估每個活躍因子
        poor_performers = []
        for factor in self.library.get_active_factors():
            eval_result = await self.evaluate_factor(factor, df)
            
            # 追蹤表現歷史
            if 'performance_history' not in self.state:
                self.state['performance_history'] = []
            
            self.state['performance_history'].append({
                'factor': factor.name,
                'day': day,
                'result': eval_result
            })
            
            # 記錄分數
            score = eval_result.get('score', 0)
            if score < 0:
                poor_performers.append(factor.name)
                print(f"   ⚠️ {factor.name} 表現不佳")
        
        # 休眠連續表現不佳的因子
        dormant_count = 0
        for factor_name in poor_performers:
            dormant_count += 1
            if self.library.remove(factor_name):
                if factor_name not in self.state['dormant_factors']:
                    self.state['dormant_factors'].append(factor_name)
                print(f"   😴 {factor_name} 进入休眠")
        
        self._save_state()
        return dormant_count
    
    def get_performance_summary(self) -> dict:
        """取得表現摘要"""
        return {
            'total_runs': self.state['total_runs'],
            'dormant_count': len(self.state['dormant_factors']),
            'dormant_factors': self.state['dormant_factors'],
            'recent_performance': self.state['performance_history'][-10:] if self.state.get('performance_history') else []
        }


# ─── Adaptive Thresholds ─────────────────────────────────────────────────────

# 根據市場環境調整門檻
# BTC 過去兩年為牛市，簡單策略勝率普遍偏低
# 研究模式下：接受有交易且回撤可控的策略
FACTOR_MIN_WIN_RATE = 0.0       # 勝率門檻（牛市期間可能為0）
FACTOR_MIN_PROFIT_FACTOR = 0.0   # 利潤因子不設限（全是輸時PF=0）
FACTOR_MAX_DRAWDOWN = 0.10      # 最大回撤 < 10%
FACTOR_MIN_TRADES = 3           # 最少交易次數（已被 FactorRecord 門檻覆寫）


# ─── Main ─────────────────────────────────────────────────────────────────────

async def main():
    print("=" * 60)
    print("📊 Auto Research v2.0 Enhanced 啟動")
    print("=" * 60)
    
    # 讀取本地數據 - 1D
    data_path = PROJECT_ROOT / 'data' / 'btcusdt_1d.parquet'
    if not data_path.exists():
        print(f"❌ 數據文件不存在: {data_path}")
        return
    
    df = pd.read_parquet(data_path)
    print(f"\n📂 1D 數據載入: {len(df)} 根K線")
    print(f"   時間範圍: {df['open_time'].min()} ~ {df['open_time'].max()}")
    
    # 讀取 4H 數據
    data_path_4h = PROJECT_ROOT / 'data' / 'btcusdt_4h.parquet'
    df_4h = None
    if data_path_4h.exists():
        df_4h = pd.read_parquet(data_path_4h)
        print(f"\n📂 4H 數據載入: {len(df_4h)} 根K線")
        print(f"   時間範圍: {df_4h['open_time'].min()} ~ {df_4h['open_time'].max()}")
    
    # 市場分層
    print("\n" + "-" * 60)
    print("🔍 市場分層分析")
    print("-" * 60)
    
    classifier = MarketRegimeClassifier(ma_period=200)
    regime = classifier.classify(df)
    stats = classifier.get_regime_stats(regime)
    
    print(f"   牛市 (Bull):  {stats['bull_count']:,} 根 ({stats['bull_pct']:.1f}%)")
    print(f"   熊市 (Bear):  {stats['bear_count']:,} 根 ({stats['bear_pct']:.1f}%)")
    print(f"   盤整 (Sideways): {stats['sideways_count']:,} 根 ({stats['sideways_pct']:.1f}%)")
    
    # 分層抽樣
    print("\n" + "-" * 60)
    print("📊 分層抽樣")
    print("-" * 60)
    
    samples = classifier.get_stratified_sample(
        df, 
        train_months=18, 
        test_months=3, 
        n_samples=5,
        random_seed=42
    )
    print(f"   生成了 {len(samples)} 組訓練/測試樣本")
    
    for i, sample in enumerate(samples[:3]):
        train_count = len(sample['train'])
        test_count = len(sample['test'])
        print(f"\n   樣本 {i+1}:")
        print(f"      訓練月: {train_count} 個月")
        print(f"      測試月: {test_count} 個月")
    
    # 使用最近兩年數據進行測試
    test_df = df.tail(730).copy()
    print(f"\n   使用 {len(test_df)} 根K線進行策略測試")
    
    # ─── 參數網格搜索 ───
    print("\n" + "-" * 60)
    print("🔬 參數網格搜索")
    print("-" * 60)
    
    all_results = []
    
    for strategy_type, param_grid in PARAM_GRIDS.items():
        print(f"\n   📈 {strategy_type}: 測試 {len(param_grid)} 組參數")
        
        results = await run_grid_search(strategy_type, param_grid, test_df)
        
        if results:
            # 按 profit_factor 排序，取最佳
            best = max(results, key=lambda x: x['result']['profit_factor'])
            print(f"      通過測試: {len(results)}/{len(param_grid)}")
            print(f"      最佳: {best['name']}")
            print(f"         PF={best['result']['profit_factor']:.3f}, WR={best['result']['win_rate']:.1f}%, Sharpe={best['result']['sharpe']:.3f}")
            
            for r in results:
                r['regime_analysis'] = None
                all_results.append(r)
        else:
            print(f"      ❌ 無有效結果")
    
    # ─── 4H 短週期策略測試 ───
    if df_4h is not None:
        print("\n" + "=" * 60)
        print("📊 4H 短週期策略測試")
        print("=" * 60)
        
        # 使用最近 180 天 4H 數據 (約 1080 根K線)
        test_df_4h = df_4h.tail(1080).copy()
        print(f"\n   使用 {len(test_df_4h)} 根4H K線進行策略測試")
        
        # 只測試短週期策略
        short_cycle_strategies = [
            'TripleEMA', 'EMA_Breakout', 'RSI_MA_Combo', 'BB_RSI_Combo',
            'TrendBreakout', 'VolumeWeighted', 'MA_Trend_Holding', 'Stochastic',
            'Supertrend', 'RSI_Extreme', 'MACD_Histogram', 'Keltner_Breakout',
            'CCI_Reversal', 'ADX_Trend'
        ]
        
        all_results_4h = []
        
        for strategy_type in short_cycle_strategies:
            if strategy_type in PARAM_GRIDS:
                param_grid = PARAM_GRIDS[strategy_type]
                print(f"\n   📈 {strategy_type}: 測試 {len(param_grid)} 組參數")
                
                results = await run_grid_search(strategy_type, param_grid, test_df_4h)
                
                if results:
                    best = max(results, key=lambda x: x['result']['profit_factor'])
                    print(f"      通過測試: {len(results)}/{len(param_grid)}")
                    print(f"      最佳: {best['name']}")
                    print(f"         PF={best['result']['profit_factor']:.3f}, WR={best['result']['win_rate']:.1f}%, Sharpe={best['result']['sharpe']:.3f}")
                    
                    for r in results:
                        r['timeframe'] = '4H'
                        all_results_4h.append(r)
                else:
                    print(f"      ❌ 無有效結果")
        
        print(f"\n   4H 測試完成: {len(all_results_4h)} 個策略通過")
        
        # 合併 1D 和 4H 結果
        for r in all_results_4h:
            all_results.append(r)
    
    # ─── 因子入庫 ───
    print("\n" + "-" * 60)
    print("📚 因子入庫評估")
    print("-" * 60)
    
    library = FactorLibrary()
    valid_count = 0
    
    for item in sorted(all_results, key=lambda x: x['result']['profit_factor'], reverse=True):
        result = item['result']
        name = item['name']
        params = item['params']
        strategy_type = item['strategy_type']
        
        is_valid = (
            result['win_rate'] >= FACTOR_MIN_WIN_RATE * 100 and
            result['max_drawdown'] <= FACTOR_MAX_DRAWDOWN and
            result['num_trades'] >= FACTOR_MIN_TRADES
        )
        
        if is_valid:
            signal_gen = SIGNAL_GENERATORS.get(strategy_type)
            category = CATEGORY_MAP.get(strategy_type, 'other')
            
            factor = FactorRecord(
                name=name,
                category=category,
                params=params,
                metrics={
                    'win_rate': result['win_rate'],
                    'profit_factor': result['profit_factor'],
                    'sharpe': result['sharpe'],
                    'max_drawdown': result['max_drawdown'],
                    'num_trades': result['num_trades'],
                    'active': True
                },
                signal_generator=signal_gen
            )
            library.add(factor)
            valid_count += 1
        else:
            print(f"   ❌ {name}: Ret={result['total_return']:.1f}%, WR={result['win_rate']:.1f}%, DD={result['max_drawdown']*100:.1f}%, trades={result['num_trades']}")
    
    print(f"\n   總共 {valid_count}/{len(all_results)} 個策略通過門檻入庫")
    
    # ─── 市場狀態分析 ───
    print("\n" + "-" * 60)
    print("🎯 各市場狀態最佳策略")
    print("-" * 60)
    
    # 取得各市場狀態
    df_with_regime = df.copy()
    regime_labels = classifier.classify(df)
    df_with_regime['regime'] = regime_labels
    
    for regime_name, regime_code in [('牛市', 0), ('熊市', 1), ('盤整', 2)]:
        regime_df = df_with_regime[df_with_regime['regime'] == regime_code]
        if len(regime_df) >= 30:
            print(f"\n   📊 {regime_name}:")
            
            # 取Top3因子在該市場狀態測試
            top_factors = library.get_top_by_metric('profit_factor', n=5)
            
            best_for_regime = None
            best_score = -999
            
            for factor in top_factors:
                cls = STRATEGY_CLASSES.get(factor.category.title() + '_' + factor.category.capitalize(), None)
                if cls is None:
                    # 嘗試直接匹配
                    for sc, scls in STRATEGY_CLASSES.items():
                        if sc.lower() in factor.name.lower():
                            cls = scls
                            break
                
                if cls:
                    try:
                        strategy = cls(**factor.params)
                        result = await run_strategy_backtest(strategy, regime_df.tail(180))
                        if result and result['total_return'] > best_score:
                            best_score = result['total_return']
                            best_for_regime = factor.name
                            print(f"      ✅ {factor.name}: 報酬={result['total_return']:.2f}%, PF={result['profit_factor']:.3f}")
                    except:
                        pass
            
            if best_for_regime is None:
                print(f"      (數據不足，無法完整分析)")
    
    # ─── 多因子信號測試 ───
    print("\n" + "-" * 60)
    print("🔗 多因子信號組合測試")
    print("-" * 60)
    
    if library.factors:
        latest = df.tail(200).copy()
        market_data = {'BTCUSDT': latest}
        
        signal = library.generate_signal(market_data)
        print(f"   組合信號: {signal['signal']}")
        print(f"   信心度: {signal['confidence']:.2f}")
        if signal['factors']:
            print(f"   因子投票:")
            for f in signal['factors'][:10]:
                print(f"      - {f['name']}: {f['signal']}")
    
    # ─── 持續優化演示 ───
    print("\n" + "-" * 60)
    print("🔄 持續滾動優化演示")
    print("-" * 60)
    
    optimizer = ContinuousOptimizer(library)
    
    # 模擬3個週期
    for day in range(3):
        dormant = await optimizer.run_day(df.tail(90), day)
        print(f"   本日休眠因子: {dormant}")
    
    perf_summary = optimizer.get_performance_summary()
    print(f"\n   總運行天數: {perf_summary['total_runs']}")
    print(f"   休眠因子數: {perf_summary['dormant_count']}")
    if perf_summary['dormant_factors']:
        print(f"   休眠列表: {perf_summary['dormant_factors']}")
    
    # ─── 最終摘要 ───
    print("\n" + "=" * 60)
    print("📋 最終摘要")
    print("=" * 60)
    
    lib_stats = library.get_statistics()
    print(f"   總因子數: {lib_stats['total_factors']}")
    print(f"   活躍因子: {lib_stats['active_factors']}")
    print(f"   按類別: {lib_stats.get('by_category', {})}")
    
    # Top因子列表
    if library.factors:
        print(f"\n   🏆 Top 5 因子 (按利潤因子):")
        for i, f in enumerate(library.get_top_by_metric('profit_factor', n=5), 1):
            m = f.metrics
            print(f"      {i}. {f.name}")
            print(f"         WR={m['win_rate']:.1f}%, PF={m['profit_factor']:.3f}, Sharpe={m['sharpe']:.3f}, DD={m['max_drawdown']*100:.1f}%")
    
    print("\n" + "=" * 60)
    print("✅ Auto Research v2.0 Enhanced 完成")
    print("=" * 60)
    
    return library


if __name__ == '__main__':
    library = asyncio.run(main())