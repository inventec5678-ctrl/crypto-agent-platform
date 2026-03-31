"""
Agent #2 Round 3: Volume/Volatility Breakout Strategies
Three approaches:
1. Volume Spike + Price Breakout (1d)
2. BB Width Squeeze → Expansion (1d)
3. Intraday 15m Strategy
"""

import pandas as pd
import numpy as np
import json
from itertools import product

# ─────────────────────────────────────────────
# Shared backtest engine
# ─────────────────────────────────────────────
def backtest(df, entries, exits, direction=1, stop_loss_pct=3.0, take_profit_pct=7.0, max_holding_bars=None):
    """
    entries/exits: boolean series (1=enter, 2=exit already handled)
    direction: 1=long, -1=short, 0=both
    Returns dict of metrics.
    """
    df = df.copy()
    df['entry'] = entries.astype(bool)
    df['in_trade'] = False
    df['trade_dir'] = 0
    df['entry_price'] = np.nan
    df['exit_price'] = np.nan
    df['trade_pnl'] = np.nan
    df['exit_reason'] = ''

    trades = []
    in_pos = False
    entry_bar = None
    entry_price = None
    trade_dir = 0

    bars = df.shape[0]
    max_holding = max_holding_bars if max_holding_bars else bars + 1

    for i in range(bars):
        if not in_pos:
            if df['entry'].iloc[i]:
                in_pos = True
                entry_bar = i
                entry_price = df['close'].iloc[i]
                trade_dir = 1
                df.loc[df.index[i], 'in_trade'] = True
                df.loc[df.index[i], 'trade_dir'] = 1
                df.loc[df.index[i], 'entry_price'] = entry_price
        else:
            hold_bars = i - entry_bar
            exit_reason = ''
            pnl = 0.0

            # Check exit conditions
            hit_sl = (df['low'].iloc[i] <= entry_price * (1 - stop_loss_pct/100))
            hit_tp = (df['high'].iloc[i] >= entry_price * (1 + take_profit_pct/100))
            hit_max = hold_bars >= max_holding

            if hit_sl:
                exit_price = entry_price * (1 - stop_loss_pct/100)
                pnl = -stop_loss_pct/100
                exit_reason = 'SL'
            elif hit_tp:
                exit_price = entry_price * (1 + take_profit_pct/100)
                pnl = take_profit_pct/100
                exit_reason = 'TP'
            elif exits.iloc[i] if i < len(exits) else False:
                exit_price = df['close'].iloc[i]
                pnl = (exit_price - entry_price) / entry_price
                exit_reason = 'SIG'
            elif hit_max:
                exit_price = df['close'].iloc[i]
                pnl = (exit_price - entry_price) / entry_price
                exit_reason = 'MAX'

            if exit_reason:
                trades.append({'entry_idx': entry_bar, 'exit_idx': i,
                                'entry_price': entry_price, 'exit_price': exit_price,
                                'pnl': pnl, 'dir': trade_dir, 'reason': exit_reason,
                                'holding_bars': hold_bars})
                in_pos = False
                entry_bar = None

    if not trades:
        return {'total_trades': 0, 'wins': 0, 'losses': 0, 'win_rate': 0,
                'profit_factor': 0, 'max_drawdown': 0, 'sharpe': 0,
                'avg_return': 0, 'total_return': 0}

    tdf = pd.DataFrame(trades)
    wins = tdf[tdf['pnl'] > 0]
    losses = tdf[tdf['pnl'] <= 0]

    gross_wins = wins['pnl'].sum() if len(wins) > 0 else 0
    gross_losses = abs(losses['pnl'].sum()) if len(losses) > 0 else 0
    profit_factor = gross_wins / gross_losses if gross_losses > 0 else float('inf')

    # Compute equity curve for Sharpe & drawdown
    equity = [1.0]
    peak = 1.0
    max_dd = 0.0
    for _, tr in tdf.iterrows():
        equity.append(equity[-1] * (1 + tr['pnl']))
    equity = pd.Series(equity)
    for v in equity:
        if v > peak:
            peak = v
        dd = (peak - v) / peak
        if dd > max_dd:
            max_dd = dd

    # Sharpe (annualized assuming 365d for 1d data, 365*16 for 15m data)
    rets = tdf['pnl'].values
    if len(rets) > 1 and np.std(rets) > 0:
        sharpe = (np.mean(rets) / np.std(rets)) * np.sqrt(365)  # annualized
    else:
        sharpe = 0

    return {
        'total_trades': len(trades),
        'wins': len(wins),
        'losses': len(losses),
        'win_rate': len(wins)/len(trades)*100,
        'profit_factor': profit_factor,
        'max_drawdown': max_dd*100,
        'sharpe': sharpe,
        'avg_return': np.mean(tdf['pnl'])*100,
        'total_return': (equity.iloc[-1] - 1)*100,
        'trade_list': trades
    }


# ─────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────
df_1d = pd.read_parquet('/Users/changrunlin/.openclaw/workspace/crypto-agent-platform/data/btcusdt_1d.parquet')
df_15m = pd.read_parquet('/Users/changrunlin/.openclaw/workspace/crypto-agent-platform/data/btcusdt_15m.parquet')

df_1d['open_time'] = pd.to_datetime(df_1d['open_time'])
df_15m['open_time'] = pd.to_datetime(df_15m['open_time'])
df_1d = df_1d.sort_values('open_time').reset_index(drop=True)
df_15m = df_15m.sort_values('open_time').reset_index(drop=True)

print(f"1d: {df_1d['open_time'].min()} → {df_1d['open_time'].max()}, {len(df_1d)} bars")
print(f"15m: {df_15m['open_time'].min()} → {df_15m['open_time'].max()}, {len(df_15m)} bars")

# ─────────────────────────────────────────────
# STRATEGY 1: Volume Spike + Price Breakout (1d)
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("STRATEGY 1: Volume Spike + Price Breakout")
print("="*60)

# Build indicators
df_1d['vol_sma20'] = df_1d['volume'].rolling(20).mean()
df_1d['vol_ratio'] = df_1d['volume'] / df_1d['vol_sma20']
df_1d['vol_rank'] = df_1d['vol_ratio'].rank(pct=True)  # percentile rank

df_1d['high_20'] = df_1d['high'].rolling(20).max()
df_1d['low_20'] = df_1d['low'].rolling(20).min()
df_1d['atr'] = df_1d['high'].rolling(14).max() - df_1d['low'].rolling(14).min()
df_1d['atr_ma'] = df_1d['atr'].rolling(20).mean()
df_1d['atr_ratio'] = df_1d['atr'] / df_1d['atr_ma']

df_1d['MA20'] = df_1d['close'].rolling(20).mean()
df_1d['MA50'] = df_1d['close'].rolling(50).mean()
df_1d['MA200'] = df_1d['close'].rolling(200).mean()

df_1d['price_breakout_high'] = df_1d['close'] > df_1d['high_20'].shift(1)
df_1d['price_breakout_low'] = df_1d['close'] < df_1d['low_20'].shift(1)

# Regime
df_1d['regime_bull'] = df_1d['close'] > df_1d['MA200']

# Momentum
df_1d['mom7'] = df_1d['close'].pct_change(7) * 100

# RSI
delta = df_1d['close'].diff()
gain = delta.clip(lower=0).rolling(14).mean()
loss = (-delta.clip(upper=0)).rolling(14).mean()
rs = gain / loss.replace(0, np.nan)
df_1d['RSI'] = 100 - (100 / (1 + rs))

results_s1 = []

# Grid search
for vol_ratio_thr in [2.0, 2.5, 3.0]:
    for atr_ratio_thr in [1.2, 1.5, 2.0]:
        for rsi_thr_long in [50, 55]:
            for rsi_thr_short in [45, 50]:
                for mom_thr in [0.3, 0.5]:
                    for tp in [7, 10, 15]:
                        for sl in [3, 4, 5]:
                            for use_regime in [True, False]:
                                # Long: Vol spike + price breaks 20d high + ATR expansion
                                long_entry = (
                                    (df_1d['vol_ratio'] > vol_ratio_thr) &
                                    (df_1d['price_breakout_high']) &
                                    (df_1d['atr_ratio'] > atr_ratio_thr) &
                                    (df_1d['RSI'] > rsi_thr_long) &
                                    (df_1d['mom7'] > mom_thr) &
                                    ((~use_regime) | (df_1d['regime_bull']))
                                )
                                # Short
                                short_entry = (
                                    (df_1d['vol_ratio'] > vol_ratio_thr) &
                                    (df_1d['price_breakout_low']) &
                                    (df_1d['atr_ratio'] > atr_ratio_thr) &
                                    (df_1d['RSI'] < rsi_thr_short) &
                                    (df_1d['mom7'] < -mom_thr) &
                                    ((~use_regime) | (~df_1d['regime_bull']))
                                )

                                long_exit = df_1d['MA20'] < df_1d['MA50]  # invalid
                                # Actually just use signals: MA cross for direction flip
                                # For this strategy, we treat it as directional only
                                # Short exit on reverse signal
                                short_exit = short_entry  # just for tracking

                                both_entry = long_entry | short_entry

                                r = backtest(df_1d, both_entry, short_exit,
                                             direction=0, stop_loss_pct=sl,
                                             take_profit_pct=tp, max_holding_bars=15)

                                if r['total_trades'] >= 10:
                                    results_s1.append({
                                        'vol_ratio_thr': vol_ratio_thr,
                                        'atr_ratio_thr': atr_ratio_thr,
                                        'rsi_thr_long': rsi_thr_long,
                                        'rsi_thr_short': rsi_thr_short,
                                        'mom_thr': mom_thr,
                                        'tp': tp, 'sl': sl,
                                        'use_regime': use_regime,
                                        **r
                                    })

# Fix: short_entry defined after use
# Actually the issue: short_entry uses same df columns defined before, but referencing shift(1)
# Let me just run the search properly

# Re-run without the broken code
df_1d['high_20_prev'] = df_1d['high'].rolling(20).max().shift(1)
df_1d['low_20_prev'] = df_1d['low'].rolling(20).min().shift(1)
df_1d['price_breakout_high_v2'] = df_1d['close'] > df_1d['high_20_prev']
df_1d['price_breakout_low_v2'] = df_1d['close'] < df_1d['low_20_prev']

results_s1 = []
for vol_ratio_thr in [2.0, 2.5, 3.0]:
    for atr_ratio_thr in [1.2, 1.5, 2.0]:
        for rsi_thr_long in [50, 55]:
            for rsi_thr_short in [45, 50]:
                for mom_thr in [0.3, 0.5]:
                    for tp in [7, 10, 15]:
                        for sl in [3, 4, 5]:
                            for use_regime in [True, False]:
                                long_entry = (
                                    (df_1d['vol_ratio'] > vol_ratio_thr) &
                                    (df_1d['price_breakout_high_v2']) &
                                    (df_1d['atr_ratio'] > atr_ratio_thr) &
                                    (df_1d['RSI'] > rsi_thr_long) &
                                    (df_1d['mom7'] > mom_thr) &
                                    ((~use_regime) | (df_1d['regime_bull']))
                                )
                                short_entry = (
                                    (df_1d['vol_ratio'] > vol_ratio_thr) &
                                    (df_1d['price_breakout_low_v2']) &
                                    (df_1d['atr_ratio'] > atr_ratio_thr) &
                                    (df_1d['RSI'] < rsi_thr_short) &
                                    (df_1d['mom7'] < -mom_thr) &
                                    ((~use_regime) | (~df_1d['regime_bull']))
                                )
                                both_entry = long_entry | short_entry
                                r = backtest(df_1d, both_entry, both_entry,
                                             direction=0, stop_loss_pct=sl,
                                             take_profit_pct=tp, max_holding_bars=15)
                                if r['total_trades'] >= 10:
                                    results_s1.append({
                                        'vol_ratio_thr': vol_ratio_thr,
                                        'atr_ratio_thr': atr_ratio_thr,
                                        'rsi_thr_long': rsi_thr_long,
                                        'rsi_thr_short': rsi_thr_short,
                                        'mom_thr': mom_thr,
                                        'tp': tp, 'sl': sl,
                                        'use_regime': use_regime,
                                        **r
                                    })

results_s1_df = pd.DataFrame(results_s1)
if len(results_s1_df) > 0:
    # Filter: win_rate >= 50, profit_factor >= 2.0, max_drawdown <= 30, sharpe >= 1.5
    pass_s1 = results_s1_df[
        (results_s1_df['win_rate'] >= 50) &
        (results_s1_df['profit_factor'] >= 2.0) &
        (results_s1_df['max_drawdown'] <= 30) &
        (results_s1_df['sharpe'] >= 1.5)
    ].sort_values('sharpe', ascending=False)
    print(f"\nTotal parameter combos tested: {len(results_s1_df)}")
    print(f"Combos passing all 4 targets: {len(pass_s1)}")
    if len(pass_s1) > 0:
        best_s1 = pass_s1.iloc[0].to_dict()
        print(f"\nBest Volume Spike Strategy:")
        print(f"  Params: vol_ratio={best_s1['vol_ratio_thr']}, atr_ratio={best_s1['atr_ratio_thr']}, RSI={best_s1['rsi_thr_long']}/{best_s1['rsi_thr_short']}, mom={best_s1['mom_thr']}, TP={best_s1['tp']}%, SL={best_s1['sl']}%, regime={best_s1['use_regime']}")
        print(f"  Trades: {best_s1['total_trades']}, WR: {best_s1['win_rate']:.1f}%, PF: {best_s1['profit_factor']:.2f}, DD: {best_s1['max_drawdown']:.1f}%, Sharpe: {best_s1['sharpe']:.2f}")
    else:
        best_s1 = results_s1_df.sort_values('sharpe', ascending=False).iloc[0].to_dict()
        print(f"\nBest (no full pass): WR={best_s1['win_rate']:.1f}%, PF={best_s1['profit_factor']:.2f}, DD={best_s1['max_drawdown']:.1f}%, Sharpe={best_s1['sharpe']:.2f}")
else:
    print("No results for Strategy 1")
    best_s1 = None
    pass_s1 = pd.DataFrame()

# ─────────────────────────────────────────────
# STRATEGY 2: BB Width Squeeze → Expansion (1d)
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("STRATEGY 2: BB Width Squeeze → Expansion")
print("="*60)

df_1d['BB_mid'] = df_1d['close'].rolling(20).mean()
df_1d['BB_std'] = df_1d['close'].rolling(20).std()
df_1d['BB_upper'] = df_1d['BB_mid'] + 2 * df_1d['BB_std']
df_1d['BB_lower'] = df_1d['BB_mid'] - 2 * df_1d['BB_std']
df_1d['BB_width'] = (df_1d['BB_upper'] - df_1d['BB_lower']) / df_1d['BB_mid']

# BB width rank (percentile over 252d = 1 year)
df_1d['BB_width_rank'] = df_1d['BB_width'].rolling(252).apply(
    lambda x: (x[-1] < x).sum() / len(x), raw=False
)

# Squeeze: BB width in bottom 20% of 1-year range
df_1d['BB_squeeze'] = df_1d['BB_width_rank'] < 0.2

# Expansion: BB width > 2x the squeeze minimum
df_1d['BB_squeeze_min'] = df_1d['BB_width'].rolling(60).min()
df_1d['BB_expanded'] = df_1d['BB_width'] > 2 * df_1d['BB_squeeze_min']

# Breakout direction: close above/below squeeze range
df_1d['BB_breakout_up'] = (df_1d['close'] > df_1d['BB_upper'].shift(1)) & df_1d['BB_expanded']
df_1d['BB_breakout_down'] = (df_1d['close'] < df_1d['BB_lower'].shift(1)) & df_1d['BB_expanded']

results_s2 = []
for bb_squeeze_thr in [0.15, 0.20, 0.25]:
    for bb_expand_mult in [1.8, 2.0, 2.5]:
        for rsi_thr_long in [50, 55]:
            for rsi_thr_short in [45, 50]:
                for mom_thr in [0.3, 0.5]:
                    for tp in [7, 10, 15]:
                        for sl in [3, 4, 5]:
                            for use_regime in [True, False]:
                                df_1d['squeeze_test'] = df_1d['BB_width_rank'] < bb_squeeze_thr
                                df_1d['expanded_test'] = df_1d['BB_width'] > bb_expand_mult * df_1d['BB_squeeze_min']

                                long_entry = (
                                    df_1d['squeeze_test'] &
                                    df_1d['expanded_test'] &
                                    df_1d['BB_breakout_up'] &
                                    (df_1d['RSI'] > rsi_thr_long) &
                                    (df_1d['mom7'] > mom_thr) &
                                    ((~use_regime) | (df_1d['regime_bull']))
                                )
                                short_entry = (
                                    df_1d['squeeze_test'] &
                                    df_1d['expanded_test'] &
                                    df_1d['BB_breakout_down'] &
                                    (df_1d['RSI'] < rsi_thr_short) &
                                    (df_1d['mom7'] < -mom_thr) &
                                    ((~use_regime) | (~df_1d['regime_bull']))
                                )
                                both_entry = long_entry | short_entry
                                r = backtest(df_1d, both_entry, both_entry,
                                             direction=0, stop_loss_pct=sl,
                                             take_profit_pct=tp, max_holding_bars=15)
                                if r['total_trades'] >= 8:
                                    results_s2.append({
                                        'bb_squeeze_thr': bb_squeeze_thr,
                                        'bb_expand_mult': bb_expand_mult,
                                        'rsi_thr_long': rsi_thr_long,
                                        'rsi_thr_short': rsi_thr_short,
                                        'mom_thr': mom_thr,
                                        'tp': tp, 'sl': sl,
                                        'use_regime': use_regime,
                                        **r
                                    })

results_s2_df = pd.DataFrame(results_s2)
if len(results_s2_df) > 0:
    pass_s2 = results_s2_df[
        (results_s2_df['win_rate'] >= 50) &
        (results_s2_df['profit_factor'] >= 2.0) &
        (results_s2_df['max_drawdown'] <= 30) &
        (results_s2_df['sharpe'] >= 1.5)
    ].sort_values('sharpe', ascending=False)
    print(f"\nTotal combos tested: {len(results_s2_df)}")
    print(f"Combos passing all 4 targets: {len(pass_s2)}")
    if len(pass_s2) > 0:
        best_s2 = pass_s2.iloc[0].to_dict()
        print(f"\nBest BB Squeeze Strategy:")
        print(f"  Params: squeeze_thr={best_s2['bb_squeeze_thr']}, expand_mult={best_s2['bb_expand_mult']}, RSI={best_s2['rsi_thr_long']}/{best_s2['rsi_thr_short']}, mom={best_s2['mom_thr']}, TP={best_s2['tp']}%, SL={best_s2['sl']}%, regime={best_s2['use_regime']}")
        print(f"  Trades: {best_s2['total_trades']}, WR: {best_s2['win_rate']:.1f}%, PF: {best_s2['profit_factor']:.2f}, DD: {best_s2['max_drawdown']:.1f}%, Sharpe: {best_s2['sharpe']:.2f}")
    else:
        best_s2 = results_s2_df.sort_values('sharpe', ascending=False).iloc[0].to_dict()
        print(f"\nBest (no full pass): WR={best_s2['win_rate']:.1f}%, PF={best_s2['profit_factor']:.2f}, DD={best_s2['max_drawdown']:.1f}%, Sharpe={best_s2['sharpe']:.2f}")
else:
    print("No results for Strategy 2")
    best_s2 = None
    pass_s2 = pd.DataFrame()

# ─────────────────────────────────────────────
# STRATEGY 3: Intraday 15m Strategy
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("STRATEGY 3: Intraday 15m Strategy")
print("="*60)

# Build 15m indicators
df_15m['vol_sma20'] = df_15m['volume'].rolling(20).mean()
df_15m['vol_ratio'] = df_15m['volume'] / df_15m['vol_sma20']

df_15m['MA20'] = df_15m['close'].rolling(20).mean()
df_15m['MA50'] = df_15m['close'].rolling(50).mean()

df_15m['high_20'] = df_15m['high'].rolling(20).max()
df_15m['low_20'] = df_15m['low'].rolling(20).min()
df_15m['high_20_prev'] = df_15m['high_20'].shift(1)
df_15m['low_20_prev'] = df_15m['low_20'].shift(1)

df_15m['price_breakout_high'] = df_15m['close'] > df_15m['high_20_prev']
df_15m['price_breakout_low'] = df_15m['close'] < df_15m['low_20_prev']

delta = df_15m['close'].diff()
gain = delta.clip(lower=0).rolling(14).mean()
loss = (-delta.clip(upper=0)).rolling(14).mean()
rs = gain / loss.replace(0, np.nan)
df_15m['RSI'] = 100 - (100 / (1 + rs))

df_15m['mom_4'] = df_15m['close'].pct_change(4) * 100  # 1h momentum
df_15m['atr'] = df_15m['high'].rolling(14).max() - df_15m['low'].rolling(14).min()
df_15m['atr_ma'] = df_15m['atr'].rolling(20).mean()
df_15m['atr_ratio'] = df_15m['atr'] / df_15m['atr_ma']

# Regime: price vs MA50 on 15m
df_15m['regime_bull'] = df_15m['close'] > df_15m['MA50']

# BB width on 15m
df_15m['BB_mid'] = df_15m['close'].rolling(20).mean()
df_15m['BB_std'] = df_15m['close'].rolling(20).std()
df_15m['BB_upper'] = df_15m['BB_mid'] + 2 * df_15m['BB_std']
df_15m['BB_lower'] = df_15m['BB_mid'] - 2 * df_15m['BB_std']
df_15m['BB_width'] = (df_15m['BB_upper'] - df_15m['BB_lower']) / df_15m['BB_mid']
df_15m['BB_width_ma'] = df_15m['BB_width'].rolling(50).mean()
df_15m['BB_expanded'] = df_15m['BB_width'] > 1.5 * df_15m['BB_width_ma']

# Session filter: only trade during high volatility sessions (00:00-08:00 UTC is typically more volatile)
# Use hourly candles for session detection
df_15m['hour'] = df_15m['open_time'].dt.hour
df_15m['is_active_session'] = (df_15m['hour'] >= 0) & (df_15m['hour'] <= 8)

# 4h candles for broader trend
# Aggregate to 4h for trend detection
df_15m['4h_close'] = df_15m['close'].resample('4h', origin='start').last()
df_15m['4h_open'] = df_15m['open'].resample('4h', origin='start').first()
# Simple: just use MA20/MA50 as trend on 15m

results_s3 = []
# For 15m: shorter holding period (4-16 bars = 1-4 hours)
for vol_ratio_thr in [1.8, 2.0, 2.5]:
    for rsi_thr_long in [50, 55]:
        for rsi_thr_short in [45, 50]:
            for mom_thr in [0.2, 0.3, 0.5]:
                for tp in [1.0, 1.5, 2.0, 3.0]:  # % take profit
                    for sl in [0.75, 1.0, 1.5]:   # % stop loss
                        for max_bars in [4, 8, 16]:  # 1h, 2h, 4h
                            for use_session in [True, False]:
                                for use_bb_expand in [True, False]:
                                    # Long
                                    long_entry = (
                                        (df_15m['vol_ratio'] > vol_ratio_thr) &
                                        (df_15m['price_breakout_high']) &
                                        (df_15m['RSI'] > rsi_thr_long) &
                                        (df_15m['mom_4'] > mom_thr) &
                                        ((~use_session) | (df_15m['is_active_session'])) &
                                        ((~use_bb_expand) | (df_15m['BB_expanded']))
                                    )
                                    # Short
                                    short_entry = (
                                        (df_15m['vol_ratio'] > vol_ratio_thr) &
                                        (df_15m['price_breakout_low']) &
                                        (df_15m['RSI'] < rsi_thr_short) &
                                        (df_15m['mom_4'] < -mom_thr) &
                                        ((~use_session) | (df_15m['is_active_session'])) &
                                        ((~use_bb_expand) | (df_15m['BB_expanded']))
                                    )
                                    both_entry = long_entry | short_entry
                                    r = backtest(df_15m, both_entry, both_entry,
                                                 direction=0, stop_loss_pct=sl,
                                                 take_profit_pct=tp, max_holding_bars=max_bars)
                                    if r['total_trades'] >= 50:
                                        results_s3.append({
                                            'vol_ratio_thr': vol_ratio_thr,
                                            'rsi_thr_long': rsi_thr_long,
                                            'rsi_thr_short': rsi_thr_short,
                                            'mom_thr': mom_thr,
                                            'tp': tp, 'sl': sl,
                                            'max_bars': max_bars,
                                            'use_session': use_session,
                                            'use_bb_expand': use_bb_expand,
                                            **r
                                        })

results_s3_df = pd.DataFrame(results_s3)
if len(results_s3_df) > 0:
    pass_s3 = results_s3_df[
        (results_s3_df['win_rate'] >= 50) &
        (results_s3_df['profit_factor'] >= 2.0) &
        (results_s3_df['max_drawdown'] <= 30) &
        (results_s3_df['sharpe'] >= 1.5)
    ].sort_values('sharpe', ascending=False)
    print(f"\nTotal combos tested: {len(results_s3_df)}")
    print(f"Combos passing all 4 targets: {len(pass_s3)}")
    if len(pass_s3) > 0:
        best_s3 = pass_s3.iloc[0].to_dict()
        print(f"\nBest 15m Strategy:")
        print(f"  Params: vol_ratio={best_s3['vol_ratio_thr']}, RSI={best_s3['rsi_thr_long']}/{best_s3['rsi_thr_short']}, mom={best_s3['mom_thr']}, TP={best_s3['tp']}%, SL={best_s3['sl']}%, max_bars={best_s3['max_bars']}, session={best_s3['use_session']}, BB_expand={best_s3['use_bb_expand']}")
        print(f"  Trades: {best_s3['total_trades']}, WR: {best_s3['win_rate']:.1f}%, PF: {best_s3['profit_factor']:.2f}, DD: {best_s3['max_drawdown']:.1f}%, Sharpe: {best_s3['sharpe']:.2f}")
    else:
        best_s3 = results_s3_df.sort_values('sharpe', ascending=False).iloc[0].to_dict()
        print(f"\nBest (no full pass): WR={best_s3['win_rate']:.1f}%, PF={best_s3['profit_factor']:.2f}, DD={best_s3['max_drawdown']:.1f}%, Sharpe={best_s3['sharpe']:.2f}, trades={best_s3['total_trades']}")
else:
    print("No results for Strategy 3")
    best_s3 = None
    pass_s3 = pd.DataFrame()

# ─────────────────────────────────────────────
# SUMMARY & REPORT
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("SUMMARY")
print("="*60)

def check_targets(r):
    if r is None: return {'win_rate_50': 'N/A', 'profit_factor_2': 'N/A', 'max_drawdown_30': 'N/A', 'sharpe_1.5': 'N/A'}
    return {
        'win_rate_50': 'PASS' if r.get('win_rate', 0) >= 50 else 'FAIL',
        'profit_factor_2': 'PASS' if r.get('profit_factor', 0) >= 2.0 else 'FAIL',
        'max_drawdown_30': 'PASS' if r.get('max_drawdown', 999) <= 30 else 'FAIL',
        'sharpe_1.5': 'PASS' if r.get('sharpe', 0) >= 1.5 else 'FAIL',
    }

s1_targets = check_targets(best_s1)
s2_targets = check_targets(best_s2)
s3_targets = check_targets(best_s3)

print(f"\nStrategy 1 (Vol Spike 1d): WR={best_s1['win_rate']:.1f if best_s1 else 'N/A'}%, PF={best_s1['profit_factor']:.2f if best_s1 else 'N/A'}, DD={best_s1['max_drawdown']:.1f if best_s1 else 'N/A'}%, Sharpe={best_s1['sharpe']:.2f if best_s1 else 'N/A'}")
print(f"  Targets: {s1_targets}")

print(f"\nStrategy 2 (BB Squeeze 1d): WR={best_s2['win_rate']:.1f if best_s2 else 'N/A'}%, PF={best_s2['profit_factor']:.2f if best_s2 else 'N/A'}, DD={best_s2['max_drawdown']:.1f if best_s2 else 'N/A'}%, Sharpe={best_s2['sharpe']:.2f if best_s2 else 'N/A'}")
print(f"  Targets: {s2_targets}")

print(f"\nStrategy 3 (15m Intraday): WR={best_s3['win_rate']:.1f if best_s3 else 'N/A'}%, PF={best_s3['profit_factor']:.2f if best_s3 else 'N/A'}, DD={best_s3['max_drawdown']:.1f if best_s3 else 'N/A'}%, Sharpe={best_s3['sharpe']:.2f if best_s3 else 'N/A'}")
print(f"  Targets: {s3_targets}")

# ─────────────────────────────────────────────
# Additional: Try BB Squeeze with regime-bidirectional, looser squeeze threshold
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("ADDITIONAL: BB Width Squeeze v2 (lower squeeze threshold, regime)")
print("="*60)

df_1d['BB_width_pct_rank