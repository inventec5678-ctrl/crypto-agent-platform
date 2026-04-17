"""
新策略研究 - 完整回測腳本
研究目標：找到 4/4 達標的新策略 (WR≥50%, PF≥2.0, DD≤30%, Sharpe≥1.5)

重點方向：
1. LONG 信號策略
2. 4H 時間軸策略  
3. 突破型/均值回歸型等不同類型
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime
import sys

np.random.seed(42)

# ============================================================
# 數據載入
# ============================================================

DATA_1D = "data/btcusdt_1d.parquet"
DATA_4H = "data/btcusdt_4h.parquet"

def load_1d():
    df = pd.read_parquet(DATA_1D)
    df['open_time'] = pd.to_datetime(df['open_time'])
    df = df.sort_values('open_time').reset_index(drop=True)
    
    # === 指標計算 ===
    df['MA20'] = df['close'].rolling(20).mean()
    df['MA50'] = df['close'].rolling(50).mean()
    df['MA200'] = df['close'].rolling(200).mean()
    
    # RSI
    delta = df['close'].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).abs().rolling(14).mean()
    rs = gain / loss
    df['RSI14'] = 100 - (100 / (1 + rs))
    
    # EMA
    df['EMA8'] = df['close'].ewm(span=8, adjust=False).mean()
    df['EMA9'] = df['close'].ewm(span=9, adjust=False).mean()
    df['EMA21'] = df['close'].ewm(span=21, adjust=False).mean()
    df['EMA40'] = df['close'].ewm(span=40, adjust=False).mean()
    df['EMA50'] = df['close'].ewm(span=50, adjust=False).mean()
    
    # 動量
    df['momentum_7d'] = df['close'].pct_change(7) * 100
    df['momentum_3d'] = df['close'].pct_change(3) * 100
    
    # ATR
    tr1 = df['high'] - df['low']
    tr2 = (df['high'] - df['close'].shift()).abs()
    tr3 = (df['low'] - df['close'].shift()).abs()
    df['ATR14'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).rolling(14).mean()
    df['ATR_pct'] = df['ATR14'] / df['close'] * 100  # ATR as % of price
    
    # Regime
    df['regime'] = np.where(df['close'] > df['MA200'], 'bull', 'bear')
    df['regime_bull'] = df['regime'] == 'bull'
    df['regime_bear'] = df['regime'] == 'bear'
    
    # Golden/Death Cross
    df['MA20_above_MA50'] = df['MA20'] > df['MA50']
    df['golden_cross'] = df['MA20_above_MA50'] & ~df['MA20_above_MA50'].shift(1).fillna(False)
    df['death_cross'] = ~df['MA20_above_MA50'] & df['MA20_above_MA50'].shift(1).fillna(False)
    
    # EMA crosses
    df['EMA8_above_EMA40'] = df['EMA8'] > df['EMA40']
    df['EMA8_cross_above_EMA40'] = df['EMA8_above_EMA40'] & ~df['EMA8_above_EMA40'].shift(1).fillna(False)
    df['EMA8_cross_below_EMA40'] = ~df['EMA8_above_EMA40'] & df['EMA8_above_EMA40'].shift(1).fillna(False)
    
    df['EMA9_above_EMA21'] = df['EMA9'] > df['EMA21']
    df['EMA9_cross_above_EMA21'] = df['EMA9_above_EMA21'] & ~df['EMA9_above_EMA21'].shift(1).fillna(False)
    df['EMA9_cross_below_EMA21'] = ~df['EMA9_above_EMA21'] & df['EMA9_above_EMA21'].shift(1).fillna(False)
    
    # Volume
    df['volume_ma20'] = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma20']
    
    # MA slope
    df['MA50_slope'] = df['MA50'].diff(5) / df['MA50'].shift(5) * 100
    
    # High/Low
    df['high_20'] = df['high'].rolling(20).max()
    df['low_20'] = df['low'].rolling(20).min()
    
    return df


def load_4h():
    df = pd.read_parquet(DATA_4H)
    df['open_time'] = pd.to_datetime(df['open_time'])
    df = df.sort_values('open_time').reset_index(drop=True)
    
    df['MA20'] = df['close'].rolling(20).mean()
    df['MA50'] = df['close'].rolling(50).mean()
    df['MA200'] = df['close'].rolling(200).mean()
    
    delta = df['close'].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).abs().rolling(14).mean()
    rs = gain / loss
    df['RSI14'] = 100 - (100 / (1 + rs))
    
    df['EMA8'] = df['close'].ewm(span=8, adjust=False).mean()
    df['EMA21'] = df['close'].ewm(span=21, adjust=False).mean()
    df['EMA40'] = df['close'].ewm(span=40, adjust=False).mean()
    
    tr1 = df['high'] - df['low']
    tr2 = (df['high'] - df['close'].shift()).abs()
    tr3 = (df['low'] - df['close'].shift()).abs()
    df['ATR14'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).rolling(14).mean()
    df['ATR_pct'] = df['ATR14'] / df['close'] * 100
    
    df['regime'] = np.where(df['close'] > df['MA200'], 'bull', 'bear')
    df['EMA8_above_EMA40'] = df['EMA8'] > df['EMA40']
    df['EMA8_cross_above_EMA40'] = df['EMA8_above_EMA40'] & ~df['EMA8_above_EMA40'].shift(1).fillna(False)
    df['EMA8_cross_below_EMA40'] = ~df['EMA8_above_EMA40'] & df['EMA8_above_EMA40'].shift(1).fillna(False)
    
    df['volume_ma20'] = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma20']
    
    df['momentum_7d'] = df['close'].pct_change(7 * 6) * 100  # 7 days in 4h = 42 bars
    
    return df


# ============================================================
# 向量化回測引擎
# ============================================================

def backtest_vectorized(df, entry_long, entry_short, sl_pct, tp_pct, max_bars=15,
                        initial_capital=10000, fee=0.0004):
    """
    向量化回測引擎
    
    Args:
        df: DataFrame with OHLCV data
        entry_long: boolean series - long entry signals
        entry_short: boolean series - short entry signals
        sl_pct: stop loss percentage
        tp_pct: take profit percentage
        max_bars: max holding period in bars
        initial_capital: starting capital
        fee: trading fee (one-way)
    
    Returns:
        dict with metrics
    """
    n = len(df)
    
    # Trade tracking arrays
    trade_entry_idx = np.full(n, np.nan)
    trade_entry_price = np.full(n, np.nan)
    trade_side = np.full(n, 0)  # 1=long, -1=short
    trade_active = np.zeros(n, dtype=bool)
    trade_exit_idx = np.full(n, np.nan)
    trade_pnl_pct = np.full(n, np.nan)
    trade_exit_reason = np.empty(n, dtype=object)
    
    # Simulate
    pos_active = False
    pos_entry_idx = None
    pos_entry_price = None
    pos_side = 0
    pos_entry_bar = None
    
    for i in range(n):
        # Close existing position
        if pos_active:
            bars_held = i - pos_entry_bar
            entry_p = pos_entry_price
            close_p = df['close'].iloc[i]
            high_p = df['high'].iloc[i]
            low_p = df['low'].iloc[i]
            
            if pos_side == 1:  # Long
                pnl_pct = (close_p - entry_p) / entry_p
                hit_sl = low_p <= entry_p * (1 - sl_pct)
                hit_tp = high_p >= entry_p * (1 + tp_pct)
            else:  # Short
                pnl_pct = (entry_p - close_p) / entry_p
                hit_sl = high_p >= entry_p * (1 + sl_pct)
                hit_tp = low_p <= entry_p * (1 - tp_pct)
            
            hit_max = bars_held >= max_bars
            
            should_close = hit_sl or hit_tp or hit_max
            
            if should_close:
                if hit_sl:
                    reason = 'SL'
                    if pos_side == 1:
                        exit_p = entry_p * (1 - sl_pct)
                    else:
                        exit_p = entry_p * (1 + sl_pct)
                elif hit_tp:
                    reason = 'TP'
                    if pos_side == 1:
                        exit_p = entry_p * (1 + tp_pct)
                    else:
                        exit_p = entry_p * (1 - tp_pct)
                else:
                    reason = 'MAX'
                    exit_p = close_p
                
                actual_pnl_pct = (exit_p - entry_p) / entry_p if pos_side == 1 else (entry_p - exit_p) / entry_p
                actual_pnl_pct -= fee * 2  # entry + exit fees
                
                trade_active[pos_entry_idx] = True
                trade_entry_idx[pos_entry_idx] = pos_entry_idx
                trade_entry_price[pos_entry_idx] = entry_p
                trade_side[pos_entry_idx] = pos_side
                trade_exit_idx[pos_entry_idx] = i
                trade_pnl_pct[pos_entry_idx] = actual_pnl_pct
                trade_exit_reason[pos_entry_idx] = reason
                
                pos_active = False
                pos_entry_idx = None
        
        # Open new position
        if not pos_active:
            if entry_long.iloc[i]:
                pos_active = True
                pos_entry_idx = i
                pos_entry_price = df['close'].iloc[i]
                pos_side = 1
                pos_entry_bar = i
            elif entry_short.iloc[i]:
                pos_active = True
                pos_entry_idx = i
                pos_entry_price = df['close'].iloc[i]
                pos_side = -1
                pos_entry_bar = i
    
    # Collect trades
    trade_mask = trade_active
    trade_count = trade_mask.sum()
    
    if trade_count == 0:
        return {
            'total_trades': 0, 'wins': 0, 'losses': 0,
            'win_rate': 0, 'profit_factor': 0, 'max_drawdown': 0,
            'sharpe': 0, 'total_return': 0, 'long_trades': 0, 'short_trades': 0,
            'tp_count': 0, 'sl_count': 0, 'timeout_count': 0
        }
    
    pnls = trade_pnl_pct[trade_mask]
    sides = trade_side[trade_mask]
    reasons = trade_exit_reason[trade_mask]
    
    wins = pnls > 0
    long_wins = (sides == 1) & wins
    short_wins = (sides == -1) & wins
    
    long_mask = sides == 1
    short_mask = sides == -1
    
    win_rate = wins.sum() / len(pnls) * 100
    gross_profit = pnls[wins].sum()
    gross_loss = abs(pnls[~wins].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    # Equity curve for drawdown/sharpe
    equity = [1.0]
    peak = 1.0
    max_dd = 0.0
    for pnl in pnls:
        equity.append(equity[-1] * (1 + pnl))
    
    for eq in equity:
        if eq > peak:
            peak = eq
        dd = (peak - eq) / peak
        if dd > max_dd:
            max_dd = dd
    
    # Sharpe
    if len(pnls) > 1 and np.std(pnls) > 0:
        sharpe = (np.mean(pnls) / np.std(pnls)) * np.sqrt(365)
    else:
        sharpe = 0.0
    
    total_return = (equity[-1] - 1) * 100
    
    return {
        'total_trades': int(trade_count),
        'wins': int(wins.sum()),
        'losses': int((~wins).sum()),
        'long_trades': int(long_mask.sum()),
        'short_trades': int(short_mask.sum()),
        'win_rate': float(win_rate),
        'profit_factor': float(profit_factor) if profit_factor != float('inf') else 999.0,
        'max_drawdown': float(max_dd * 100),
        'sharpe': float(sharpe),
        'total_return': float(total_return),
        'tp_count': int((reasons == 'TP').sum()),
        'sl_count': int((reasons == 'SL').sum()),
        'timeout_count': int((reasons == 'MAX').sum()),
    }


# ============================================================
# 策略定義工廠
# ============================================================

class StrategyFactory:
    """策略工廠"""
    
    @staticmethod
    def EMA840_Regime(df, rsi_long_min=48, rsi_long_max=65, 
                      rsi_short_max=52, rsi_short_min=35,
                      momentum_min=0, momentum_max=0,
                      regime_filter=True):
        """EMA8/40 + Regime Filter"""
        long_entry = df['EMA8_cross_above_EMA40'].copy()
        short_entry = df['EMA8_cross_below_EMA40'].copy()
        
        long_ok = (df['RSI14'] > rsi_long_min) & (df['RSI14'] < rsi_long_max) & \
                  (df['momentum_7d'] > momentum_min)
        short_ok = (df['RSI14'] < rsi_short_max) & (df['RSI14'] > rsi_short_min) & \
                    (df['momentum_7d'] < momentum_max)
        
        if regime_filter:
            long_ok = long_ok & df['regime_bull'] & (df['close'] > df['MA200'])
            short_ok = short_ok & df['regime_bear'] & (df['close'] < df['MA200'])
        
        return long_ok, short_ok
    
    @staticmethod
    def GoldenCross_Strict(df, rsi_long_min=50, rsi_short_max=48,
                           momentum_min=0.5, regime_filter=True):
        """MA20/50 Golden/Death Cross + Strict Filters"""
        long_entry = df['golden_cross'].copy()
        short_entry = df['death_cross'].copy()
        
        long_ok = (df['RSI14'] > rsi_long_min) & (df['momentum_7d'] > momentum_min)
        short_ok = (df['RSI14'] < rsi_short_max) & (df['momentum_7d'] < -momentum_min)
        
        if regime_filter:
            long_ok = long_ok & df['regime_bull']
            short_ok = short_ok & df['regime_bear']
        
        return long_ok, short_ok
    
    @staticmethod
    def LONG_DipBuy(df, rsi_max=50, rsi_min=40, ma50_slope_min=0.01, 
                    momentum_min=0, regime_filter=True):
        """LONG-ONLY: Dip-buy after confirmed uptrend"""
        # MA50 rising for 5+ days
        ma50_rising = df['MA50_slope'] > ma50_slope_min
        
        # RSI in recovery zone
        rsi_recovering = (df['RSI14'] > rsi_min) & (df['RSI14'] < rsi_max)
        
        # Price above MA50 (not broken)
        price_above_ma50 = df['close'] > df['MA50']
        
        # Positive momentum
        mom_ok = df['momentum_7d'] > momentum_min
        
        long_entry = ma50_rising & rsi_recovering & price_above_ma50 & mom_ok
        
        if regime_filter:
            long_entry = long_entry & df['regime_bull']
        
        return long_entry, pd.Series(False, index=df.index)
    
    @staticmethod
    def LONG_EMA9_RSIDip(df, rsi_max=55, rsi_min=40, regime_filter=True):
        """LONG-ONLY: EMA9/21 cross with RSI dip"""
        long_entry = df['EMA9_cross_above_EMA21'].copy()
        short_entry = pd.Series(False, index=df.index)
        
        long_ok = (df['RSI14'] > rsi_min) & (df['RSI14'] < rsi_max)
        
        if regime_filter:
            long_ok = long_ok & df['regime_bull'] & (df['close'] > df['MA50'])
        
        return long_ok, short_entry
    
    @staticmethod
    def Dual_RSI_Momentum(df, rsi_long_min=48, rsi_short_max=52,
                           momentum_min=0.3, regime_filter=True):
        """雙向: RSI對稱 + 動量"""
        # Long: RSI exits oversold, momentum turns positive
        # Short: RSI exits overbought, momentum turns negative
        rsi_turning_up = df['RSI14'] > rsi_long_min
        rsi_turning_down = df['RSI14'] < rsi_short_max
        
        mom_up = df['momentum_7d'] > momentum_min
        mom_down = df['momentum_7d'] < -momentum_min
        
        # Also require MA50 trending
        ma50_ok = df['MA50_slope'] > 0
        
        long_entry = rsi_turning_up & mom_up & ma50_ok
        short_entry = rsi_turning_down & mom_down & ~ma50_ok
        
        if regime_filter:
            long_entry = long_entry & df['regime_bull']
            short_entry = short_entry & df['regime_bear']
        
        return long_entry, short_entry
    
    @staticmethod
    def LONG_BreakoutHigh(df, rsi_min=50, rsi_max=70, vol_mult=1.2,
                          regime_filter=True):
        """LONG-ONLY: 20日高低點突破 + 成交量確認"""
        # Price breaks above 20-day high
        price_break = df['close'] > df['high_20'].shift(1)
        
        # RSI confirming strength
        rsi_ok = (df['RSI14'] > rsi_min) & (df['RSI14'] < rsi_max)
        
        # Volume confirmation
        vol_ok = df['volume_ratio'] > vol_mult
        
        # Price above MA50 (trend confirmation)
        trend_ok = df['close'] > df['MA50']
        
        long_entry = price_break & rsi_ok & vol_ok & trend_ok
        
        if regime_filter:
            long_entry = long_entry & df['regime_bull']
        
        return long_entry, pd.Series(False, index=df.index)
    
    @staticmethod
    def EMA840_NoRegime(df, rsi_long_min=45, rsi_short_max=55,
                        momentum_min=0):
        """EMA8/40 WITHOUT regime filter - pure trend"""
        long_entry = df['EMA8_cross_above_EMA40'].copy()
        short_entry = df['EMA8_cross_below_EMA40'].copy()
        
        long_ok = (df['RSI14'] > rsi_long_min) & (df['momentum_7d'] > momentum_min)
        short_ok = (df['RSI14'] < rsi_short_max) & (df['momentum_7d'] < -momentum_min)
        
        return long_ok, short_ok


# ============================================================
# 參數網格搜索
# ============================================================

def grid_search_ema840_regime(df, regime_filter=True):
    """網格搜索 EMA8/40 + Regime 策略"""
    print("\n🔍 Grid Search: EMA8/40 + Regime...")
    
    best = None
    best_sharpe = -999
    
    for rsi_l_min in [45, 48, 50, 52]:
        for rsi_l_max in [60, 65, 70]:
            for rsi_s_max in [48, 50, 52, 55]:
                for rsi_s_min in [30, 35, 40]:
                    for mom_min in [0, 0.3, 0.5]:
                        for tp in [5, 7, 10]:
                            for sl in [3, 4, 5]:
                                long_e, short_e = StrategyFactory.EMA840_Regime(
                                    df, rsi_l_min, rsi_l_max, rsi_s_max, rsi_s_min,
                                    mom_min, -mom_min, regime_filter
                                )
                                r = backtest_vectorized(df, long_e, short_e, sl/100, tp/100)
                                
                                if r['total_trades'] < 10:
                                    continue
                                
                                if r['sharpe'] > best_sharpe:
                                    best_sharpe = r['sharpe']
                                    best = {
                                        **r,
                                        'strategy': 'EMA840_Regime',
                                        'params': {
                                            'rsi_long_min': rsi_l_min, 'rsi_long_max': rsi_l_max,
                                            'rsi_short_max': rsi_s_max, 'rsi_short_min': rsi_s_min,
                                            'momentum_min': mom_min,
                                            'tp': tp, 'sl': sl, 'regime': regime_filter
                                        }
                                    }
    
    return best


def grid_search_golden_strict(df, regime_filter=True):
    """網格搜索 Golden/Death Cross 嚴格版"""
    print("\n🔍 Grid Search: Golden/Death Cross Strict...")
    
    best = None
    best_sharpe = -999
    
    for rsi_l in [48, 50, 52, 55]:
        for rsi_s in [45, 48, 50]:
            for mom in [0.3, 0.5, 0.8]:
                for tp in [5, 7, 9]:
                    for sl in [3, 4, 5]:
                        long_e, short_e = StrategyFactory.GoldenCross_Strict(
                            df, rsi_l, rsi_s, mom, regime_filter
                        )
                        r = backtest_vectorized(df, long_e, short_e, sl/100, tp/100)
                        
                        if r['total_trades'] < 10:
                            continue
                        
                        if r['sharpe'] > best_sharpe:
                            best_sharpe = r['sharpe']
                            best = {
                                **r,
                                'strategy': 'GoldenCross_Strict',
                                'params': {
                                    'rsi_long': rsi_l, 'rsi_short': rsi_s,
                                    'momentum_min': mom,
                                    'tp': tp, 'sl': sl, 'regime': regime_filter
                                }
                            }
    
    return best


def grid_search_longrsi_dip(df, regime_filter=True):
    """網格搜索 LONG RSI Dip策略"""
    print("\n🔍 Grid Search: LONG RSI DipBuy...")
    
    best = None
    best_sharpe = -999
    
    for rsi_min in [35, 40, 45]:
        for rsi_max in [50, 55, 60]:
            for ma_slope in [0.005, 0.01, 0.02]:
                for mom_min in [0, 0.3, 0.5]:
                    for tp in [5, 7, 10, 12]:
                        for sl in [3, 4, 5]:
                            long_e, _ = StrategyFactory.LONG_DipBuy(
                                df, rsi_max, rsi_min, ma_slope, mom_min, regime_filter
                            )
                            r = backtest_vectorized(df, long_e, pd.Series(False, index=df.index), sl/100, tp/100)
                            
                            if r['total_trades'] < 8:
                                continue
                            
                            if r['sharpe'] > best_sharpe:
                                best_sharpe = r['sharpe']
                                best = {
                                    **r,
                                    'strategy': 'LONG_DipBuy',
                                    'params': {
                                        'rsi_min': rsi_min, 'rsi_max': rsi_max,
                                        'ma50_slope': ma_slope, 'momentum_min': mom_min,
                                        'tp': tp, 'sl': sl, 'regime': regime_filter
                                    }
                                }
    
    return best


def grid_search_breakout_high(df, regime_filter=True):
    """網格搜索 突破高點策略"""
    print("\n🔍 Grid Search: Breakout High...")
    
    best = None
    best_sharpe = -999
    
    for rsi_min in [45, 50, 55]:
        for rsi_max in [65, 70, 75]:
            for vol_mult in [1.0, 1.2, 1.5]:
                for tp in [5, 7, 10, 12]:
                    for sl in [3, 4, 5]:
                        long_e, _ = StrategyFactory.LONG_BreakoutHigh(
                            df, rsi_min, rsi_max, vol_mult, regime_filter
                        )
                        r = backtest_vectorized(df, long_e, pd.Series(False, index=df.index), sl/100, tp/100)
                        
                        if r['total_trades'] < 8:
                            continue
                        
                        if r['sharpe'] > best_sharpe:
                            best_sharpe = r['sharpe']
                            best = {
                                **r,
                                'strategy': 'LONG_BreakoutHigh',
                                'params': {
                                    'rsi_min': rsi_min, 'rsi_max': rsi_max,
                                    'vol_mult': vol_mult,
                                    'tp': tp, 'sl': sl, 'regime': regime_filter
                                }
                            }
    
    return best


def grid_search_rsi_momentum(df, regime_filter=True):
    """網格搜索 RSI Momentum 策略"""
    print("\n🔍 Grid Search: RSI Momentum...")
    
    best = None
    best_sharpe = -999
    
    for rsi_l_min in [45, 48, 50, 55]:
        for rsi_s_max in [48, 50, 52, 55]:
            for mom_min in [0.3, 0.5, 0.8]:
                for tp in [5, 7, 10]:
                    for sl in [3, 4, 5]:
                        long_e, short_e = StrategyFactory.Dual_RSI_Momentum(
                            df, rsi_l_min, rsi_s_max, mom_min, regime_filter
                        )
                        r = backtest_vectorized(df, long_e, short_e, sl/100, tp/100)
                        
                        if r['total_trades'] < 10:
                            continue
                        
                        if r['sharpe'] > best_sharpe:
                            best_sharpe = r['sharpe']
                            best = {
                                **r,
                                'strategy': 'Dual_RSI_Momentum',
                                'params': {
                                    'rsi_long_min': rsi_l_min, 'rsi_short_max': rsi_s_max,
                                    'momentum_min': mom_min,
                                    'tp': tp, 'sl': sl, 'regime': regime_filter
                                }
                            }
    
    return best


def grid_search_ema840_noregime(df):
    """網格搜索 EMA8/40 無Regime版本"""
    print("\n🔍 Grid Search: EMA8/40 No-Regime...")
    
    best = None
    best_sharpe = -999
    
    for rsi_l_min in [40, 45, 48, 50]:
        for rsi_s_max in [50, 55, 60]:
            for mom_min in [0, 0.3]:
                for tp in [5, 7, 10]:
                    for sl in [3, 4, 5]:
                        long_e, short_e = StrategyFactory.EMA840_NoRegime(
                            df, rsi_l_min, rsi_s_max, mom_min
                        )
                        r = backtest_vectorized(df, long_e, short_e, sl/100, tp/100)
                        
                        if r['total_trades'] < 10:
                            continue
                        
                        if r['sharpe'] > best_sharpe:
                            best_sharpe = r['sharpe']
                            best = {
                                **r,
                                'strategy': 'EMA840_NoRegime',
                                'params': {
                                    'rsi_long_min': rsi_l_min, 'rsi_short_max': rsi_s_max,
                                    'momentum_min': mom_min,
                                    'tp': tp, 'sl': sl
                                }
                            }
    
    return best


# ============================================================
# 目標檢查
# ============================================================

def check_targets(r):
    if r is None or r.get('total_trades', 0) == 0:
        return {'WR_50': '❌ FAIL', 'PF_2.0': '❌ FAIL', 
                'DD_30': '❌ FAIL', 'Sharpe_1.5': '❌ FAIL'}, False
    
    checks = {
        'WR_50': '✅ PASS' if r['win_rate'] >= 50 else '❌ FAIL',
        'PF_2.0': '✅ PASS' if r['profit_factor'] >= 2.0 else '❌ FAIL',
        'DD_30': '✅ PASS' if r['max_drawdown'] <= 30 else '❌ FAIL',
        'Sharpe_1.5': '✅ PASS' if r['sharpe'] >= 1.5 else '❌ FAIL',
    }
    
    all_pass = all(v == '✅ PASS' for v in checks.values())
    return checks, all_pass


# ============================================================
# 主程序
# ============================================================

def main():
    print("=" * 70)
    print("🔬 新策略研究 - 向量化回測引擎")
    print("=" * 70)
    
    # 載入數據
    df_1d = load_1d()
    print(f"\n📊 1D 數據: {len(df_1d)} bars, {df_1d['open_time'].min()} ~ {df_1d['open_time'].max()}")
    
    all_results = []
    
    # 策略1: EMA8/40 + Regime
    r1 = grid_search_ema840_regime(df_1d, regime_filter=True)
    if r1: all_results.append(r1)
    
    # 策略2: Golden/Death Cross 嚴格版
    r2 = grid_search_golden_strict(df_1d, regime_filter=True)
    if r2: all_results.append(r2)
    
    # 策略3: LONG RSI Dip
    r3 = grid_search_longrsi_dip(df_1d, regime_filter=True)
    if r3: all_results.append(r3)
    
    # 策略4: 突破高點
    r4 = grid_search_breakout_high(df_1d, regime_filter=True)
    if r4: all_results.append(r4)
    
    # 策略5: RSI Momentum
    r5 = grid_search_rsi_momentum(df_1d, regime_filter=True)
    if r5: all_results.append(r5)
    
    # 策略6: EMA8/40 無Regime
    r6 = grid_search_ema840_noregime(df_1d)
    if r6: all_results.append(r6)
    
    # ============================================================
    # 結果分析
    # ============================================================
    print("\n" + "=" * 70)
    print("📊 所有結果摘要")
    print("=" * 70)
    
    # Sort by Sharpe descending
    all_results.sort(key=lambda x: x.get('sharpe', 0), reverse=True)
    
    passed_results = []
    
    for r in all_results:
        checks, passed = check_targets(r)
        r['target_checks'] = checks
        r['passed'] = passed
        
        status = "✅ PASS 4/4" if passed else "❌ FAIL"
        print(f"\n{r['strategy']}")
        print(f"  Params: {r.get('params', {})}")
        print(f"  Trades: {r['total_trades']} (L={r['long_trades']}, S={r['short_trades']})")
        print(f"  WR: {r['win_rate']:.2f}%, PF: {r['profit_factor']:.3f}, DD: {r['max_drawdown']:.2f}%, Sharpe: {r['sharpe']:.3f}")
        print(f"  TP/SL/MAX: {r['tp_count']}/{r['sl_count']}/{r['timeout_count']}")
        print(f"  Status: {status}")
        for k, v in checks.items():
            print(f"    {k}: {v}")
        
        if passed:
            passed_results.append(r)
    
    # ============================================================
    # 最佳策略詳細分析
    # ============================================================
    if passed_results:
        print("\n✅ 找到以下通過所有目標的策略:")
        for r in passed_results:
            print(f"\n  【{r['strategy']}】")
            print(f"  Params: {r.get('params', {})}")
            print(f"  WR={r['win_rate']:.2f}%, PF={r['profit_factor']:.3f}, DD={r['max_drawdown']:.2f}%, Sharpe={r['sharpe']:.3f}")
            print(f"  Trades: {r['total_trades']} (L={r['long_trades']}, S={r['short_trades']})")
            print(f"  TP/SL/Timeout: {r['tp_count']}/{r['sl_count']}/{r['timeout_count']}")
    else:
        print("\n❌ 沒有策略通過所有4個目標")
        print("\n展示最佳前5名:")
        for i, r in enumerate(all_results[:5]):
            print(f"\n  #{i+1} {r['strategy']}: WR={r['win_rate']:.2f}%, PF={r['profit_factor']:.3f}, DD={r['max_drawdown']:.2f}%, Sharpe={r['sharpe']:.3f}")
            print(f"     Params: {r.get('params', {})}")
            print(f"     Trades: {r['total_trades']}")
    
    # ============================================================
    # 保存結果
    # ============================================================
    output = {
        'timestamp': datetime.now().isoformat(),
        'data_info': {
            '1d_range': f"{df_1d['open_time'].min()} to {df_1d['open_time'].max()}",
            '1d_bars': len(df_1d)
        },
        'all_results': [{
            'strategy': r['strategy'],
            'params': r.get('params', {}),
            'metrics': {k: v for k, v in r.items() 
                       if k not in ['strategy', 'params', 'target_checks', 'passed']},
            'passed': r.get('passed', False),
            'target_checks': r.get('target_checks', {})
        } for r in all_results]
    }
    
    with open('autoresearch/memory/new_research_results.json', 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"\n💾 結果已保存到 autoresearch/memory/new_research_results.json")
    
    return passed_results, all_results


if __name__ == "__main__":
    passed, all_res = main()
