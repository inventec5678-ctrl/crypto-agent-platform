#!/usr/bin/env python3
"""
Quick backtest runner - fills ranking service with real trades from historical data.
"""
import sys
sys.path.insert(0, '/Users/changrunlin/.openclaw/workspace/crypto-agent-platform')

import pandas as pd
from datetime import datetime
from strategy_ranking import get_ranking_service, TradeDirection

# Load data
df = pd.read_parquet('/Users/changrunlin/.openclaw/workspace/crypto-agent-platform/data/btcusdt_1d.parquet')
print(f"Loaded {len(df)} rows from {df['open_time'].min()} to {df['open_time'].max()}")

ranker = get_ranking_service()

# Register strategies
strategies = ['MA_Cross', 'RSI_Reversal', 'BB_Breakout']
for name in strategies:
    ranker.register_strategy(name)

# ── MA Cross Strategy ──────────────────────────────────────────────
tracker_ma = ranker.get_tracker('MA_Cross')
df['ma_fast'] = df['close'].rolling(10).mean()
df['ma_slow'] = df['close'].rolling(30).mean()

position = None
for i in range(31, len(df)):
    row = df.iloc[i]
    prev_fast = df.iloc[i-1]['ma_fast']
    prev_slow = df.iloc[i-1]['ma_slow']
    curr_fast = row['ma_fast']
    curr_slow = row['ma_slow']

    if curr_fast > curr_slow and prev_fast <= prev_slow and position is None:
        # Golden cross - BUY
        position = tracker_ma.open_trade(
            entry_price=row['close'],
            quantity=0.1,  # 0.1 BTC
            direction=TradeDirection.LONG,
            entry_time=pd.Timestamp(row['open_time']).to_pydatetime()
        )
    elif curr_fast < curr_slow and prev_fast >= prev_slow and position is not None:
        # Death cross - SELL
        tracker_ma.close_trade(
            trade_id=position,
            exit_price=row['close'],
            exit_time=pd.Timestamp(row['open_time']).to_pydatetime(),
            commission=row['close'] * 0.1 * 0.001  # 0.1% fee
        )
        position = None

# Close any remaining position at last price
if position is not None:
    last_row = df.iloc[-1]
    tracker_ma.close_trade(
        trade_id=position,
        exit_price=last_row['close'],
        exit_time=pd.Timestamp(last_row['open_time']).to_pydatetime(),
        commission=last_row['close'] * 0.1 * 0.001
    )

metrics_ma = tracker_ma.get_metrics()
print(f"MA_Cross: {metrics_ma.total_trades} trades, PnL={metrics_ma.total_pnl:.2f}, WinRate={metrics_ma.win_rate:.2%}")

# ── RSI Reversal Strategy ───────────────────────────────────────────
tracker_rsi = ranker.get_tracker('RSI_Reversal')
delta = df['close'].diff()
gain = delta.clip(lower=0).rolling(14).mean()
loss = (-delta.clip(upper=0)).rolling(14).mean()
rs = gain / loss
df['rsi'] = 100 - (100 / (1 + rs))

position = None
for i in range(20, len(df)):
    row = df.iloc[i]
    prev_rsi = df.iloc[i-1]['rsi']
    curr_rsi = row['rsi']

    if prev_rsi < 30 and curr_rsi >= 30 and position is None:
        # RSI crosses above 30 - BUY (oversold reversal)
        position = tracker_rsi.open_trade(
            entry_price=row['close'],
            quantity=0.1,
            direction=TradeDirection.LONG,
            entry_time=pd.Timestamp(row['open_time']).to_pydatetime()
        )
    elif prev_rsi > 70 and curr_rsi <= 70 and position is not None:
        # RSI crosses below 70 - SELL (overbought)
        tracker_rsi.close_trade(
            trade_id=position,
            exit_price=row['close'],
            exit_time=pd.Timestamp(row['open_time']).to_pydatetime(),
            commission=row['close'] * 0.1 * 0.001
        )
        position = None

if position is not None:
    last_row = df.iloc[-1]
    tracker_rsi.close_trade(
        trade_id=position,
        exit_price=last_row['close'],
        exit_time=pd.Timestamp(last_row['open_time']).to_pydatetime(),
        commission=last_row['close'] * 0.1 * 0.001
    )

metrics_rsi = tracker_rsi.get_metrics()
print(f"RSI_Reversal: {metrics_rsi.total_trades} trades, PnL={metrics_rsi.total_pnl:.2f}, WinRate={metrics_rsi.win_rate:.2%}")

# ── Bollinger Bands Breakout Strategy ──────────────────────────────
tracker_bb = ranker.get_tracker('BB_Breakout')
df['bb_mid'] = df['close'].rolling(20).mean()
df['bb_std'] = df['close'].rolling(20).std()
df['bb_upper'] = df['bb_mid'] + 2 * df['bb_std']
df['bb_lower'] = df['bb_mid'] - 2 * df['bb_std']

position = None
for i in range(21, len(df)):
    row = df.iloc[i]
    prev_close = df.iloc[i-1]['close']
    curr_close = row['close']

    # Breakout above upper band
    if prev_close <= row['bb_upper'] and curr_close > row['bb_upper'] and position is None:
        position = tracker_bb.open_trade(
            entry_price=curr_close,
            quantity=0.1,
            direction=TradeDirection.LONG,
            entry_time=pd.Timestamp(row['open_time']).to_pydatetime()
        )
    # Mean reversion: price crosses back below upper band
    elif prev_close > row['bb_upper'] and curr_close <= row['bb_upper'] and position is not None:
        tracker_bb.close_trade(
            trade_id=position,
            exit_price=curr_close,
            exit_time=pd.Timestamp(row['open_time']).to_pydatetime(),
            commission=curr_close * 0.1 * 0.001
        )
        position = None

if position is not None:
    last_row = df.iloc[-1]
    tracker_bb.close_trade(
        trade_id=position,
        exit_price=last_row['close'],
        exit_time=pd.Timestamp(last_row['open_time']).to_pydatetime(),
        commission=last_row['close'] * 0.1 * 0.001
    )

metrics_bb = tracker_bb.get_metrics()
print(f"BB_Breakout: {metrics_bb.total_trades} trades, PnL={metrics_bb.total_pnl:.2f}, WinRate={metrics_bb.win_rate:.2%}")

print("\n── Rankings ──")
rankings = ranker.get_rankings()
for e in rankings['rankings']:
    print(f"  #{e['rank']} {e['strategy']}: score={e['score']:.4f}, trades={e['total_trades']}, win_rate={e['win_rate']:.2%}, pnl={e['total_pnl']:.2f}")
