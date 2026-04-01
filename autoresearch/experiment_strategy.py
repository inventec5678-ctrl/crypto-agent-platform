
"""
experiment_strategy.py — Round 3
Market: BEAR | RSI=48.8
"""
from dataclasses import dataclass

def get_entry(snap) -> bool:
    return (snap.rsi < 35 and snap.vol_ratio > 1.5 and snap.trend_7d > 0.4)

STOP_LOSS = 0.0267
TAKE_PROFIT = 0.1088
MAX_HOLDING_BARS = 6
DIRECTION = "LONG"
