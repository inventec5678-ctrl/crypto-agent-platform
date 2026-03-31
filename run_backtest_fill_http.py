#!/usr/bin/env python3
"""
Quick backtest runner - fills ranking service via HTTP API.
"""
import sys
sys.path.insert(0, '/Users/changrunlin/.openclaw/workspace/crypto-agent-platform')

import pandas as pd
import requests
from datetime import datetime

API = "http://localhost:8000"

# Load data
df = pd.read_parquet('/Users/changrunlin/.openclaw/workspace/crypto-agent-platform/data/btcusdt_1d.parquet')
print(f"Loaded {len(df)} rows from {df['open_time'].min()} to {df['open_time'].max()}")

strategies = ['MA_Cross', 'RSI_Reversal', 'BB_Breakout']

# ── Register strategies ────────────────────────────────────────────
for name in strategies:
    resp = requests.post(f"{API}/api/strategies", json={"strategy_name": name})
    print(f"Registered {name}: {resp.status_code}")

def open_trade_http(strategy, entry_price, quantity, direction):
    resp = requests.post(f"{API}/api/trades/open", json={
        "strategy_name": strategy,
        "entry_price": float(entry_price),
        "quantity": float(quantity),
        "direction": direction,
    })
    if resp.status_code != 200:
        print(f"  ERROR opening trade: {resp.status_code} {resp.text}")
        return None
    return resp.json()["trade_id"]

def close_trade_http(strategy, trade_id, exit_price, commission=0.0):
    resp = requests.post(f"{API}/api/trades/close", json={
        "strategy_name": strategy,
        "trade_id": trade_id,
        "exit_price": float(exit_price),
        "commission": float(commission),
    })
    if resp.status_code != 200:
        print(f"  ERROR closing trade: {resp.status_code} {resp.text}")
        return None
    return resp.json().get("pnl")

# ── MA Cross Strategy ─────────────────────────────────────────────
print("\n── MA_Cross ──")
df['ma_fast'] = df['close'].rolling(10).mean()
df['ma_slow'] = df['close'].rolling(30).mean()

position = None
trade_count = 0
for i in range(31, len(df)):
    row = df.iloc[i]
    prev_fast = df.iloc[i-1]['ma_fast']
    prev_slow = df.iloc[i-1]['ma_slow']
    curr_fast = row['ma_fast']
    curr_slow = row['ma_slow']

    if curr_fast > curr_slow and prev_fast <= prev_slow and position is None:
        position = open_trade_http('MA_Cross', row['close'], 0.1, 'long')
    elif curr_fast < curr_slow and prev_fast >= prev_slow and position is not None:
        comm = row['close'] * 0.1 * 0.001
        close_trade_http('MA_Cross', position, row['close'], comm)
        position = None
        trade_count += 1

if position:
    last_row = df.iloc[-1]
    close_trade_http('MA_Cross', position, last_row['close'], last_row['close'] * 0.1 * 0.001)
    trade_count += 1
print(f"  Recorded {trade_count} trades")

# ── RSI Reversal Strategy ─────────────────────────────────────────
print("\n── RSI_Reversal ──")
delta = df['close'].diff()
gain = delta.clip(lower=0).rolling(14).mean()
loss = (-delta.clip(upper=0)).rolling(14).mean()
rs = gain / loss
df['rsi'] = 100 - (100 / (1 + rs))

position = None
trade_count = 0
for i in range(20, len(df)):
    row = df.iloc[i]
    prev_rsi = df.iloc[i-1]['rsi']
    curr_rsi = row['rsi']

    if prev_rsi < 30 and curr_rsi >= 30 and position is None:
        position = open_trade_http('RSI_Reversal', row['close'], 0.1, 'long')
    elif prev_rsi > 70 and curr_rsi <= 70 and position is not None:
        comm = row['close'] * 0.1 * 0.001
        close_trade_http('RSI_Reversal', position, row['close'], comm)
        position = None
        trade_count += 1

if position:
    last_row = df.iloc[-1]
    close_trade_http('RSI_Reversal', position, last_row['close'], last_row['close'] * 0.1 * 0.001)
    trade_count += 1
print(f"  Recorded {trade_count} trades")

# ── BB Breakout Strategy ──────────────────────────────────────────
print("\n── BB_Breakout ──")
df['bb_mid'] = df['close'].rolling(20).mean()
df['bb_std'] = df['close'].rolling(20).std()
df['bb_upper'] = df['bb_mid'] + 2 * df['bb_std']
df['bb_lower'] = df['bb_mid'] - 2 * df['bb_std']

position = None
trade_count = 0
for i in range(21, len(df)):
    row = df.iloc[i]
    prev_close = df.iloc[i-1]['close']
    curr_close = row['close']

    if prev_close <= row['bb_upper'] and curr_close > row['bb_upper'] and position is None:
        position = open_trade_http('BB_Breakout', curr_close, 0.1, 'long')
    elif prev_close > row['bb_upper'] and curr_close <= row['bb_upper'] and position is not None:
        comm = curr_close * 0.1 * 0.001
        close_trade_http('BB_Breakout', position, curr_close, comm)
        position = None
        trade_count += 1

if position:
    last_row = df.iloc[-1]
    close_trade_http('BB_Breakout', position, last_row['close'], last_row['close'] * 0.1 * 0.001)
    trade_count += 1
print(f"  Recorded {trade_count} trades")

# ── Verify ─────────────────────────────────────────────────────────
print("\n── Rankings ──")
resp = requests.get(f"{API}/api/rankings")
data = resp.json()
for e in data['rankings']:
    print(f"  #{e['rank']} {e['strategy']}: score={e['score']:.4f}, trades={e['total_trades']}, win_rate={e['win_rate']:.2%}, pnl={e['total_pnl']:.2f}")
