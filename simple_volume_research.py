#!/usr/bin/env python3
"""
成交量突破策略研究 - 獨立版本
直接使用本地數據進行回測
"""

import pandas as pd
import numpy as np
from pathlib import Path
import itertools
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent

# 載入數據
df = pd.read_parquet(PROJECT_ROOT / "data" / "btcusdt_1d.parquet")
print(f"📊 載入 {len(df)} 根 K 線 ({df['open_time'].min()} ~ {df['open_time'].max()})")

# 計算 ATR
def calculate_atr(df, period=14):
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    
    trs = []
    for i in range(1, min(len(df), period + 1)):
        tr = max(
            high[-i] - low[-i],
            abs(high[-i] - close[-i-1]),
            abs(low[-i] - close[-i-1])
        )
        trs.append(tr)
    return np.mean(trs) if trs else 0


def backtest_volume_strategy(df, volume_ma_period, volume_multiplier, trend_period, stop_loss_atr, take_profit_atr):
    """回測成交量策略"""
    trades = []
    position = None
    entry_price = 0
    entry_bar = 0
    
    volumes = df['volume'].values
    closes = df['close'].values
    highs = df['high'].values
    lows = df['low'].values
    
    for i in range(volume_ma_period + trend_period + 5, len(df)):
        # 計算成交量移動平均
        vol_ma = np.mean(volumes[i - volume_ma_period:i])
        current_vol = volumes[i]
        vol_ratio = current_vol / vol_ma if vol_ma > 0 else 1
        
        # 計算價格趨勢
        trend_start = i - trend_period
        trend_change = (closes[i] - closes[trend_start]) / closes[trend_start] * 100 if closes[trend_start] != 0 else 0
        
        # 計算 ATR
        atr_window = df.iloc[max(0, i-14):i]
        atr = calculate_atr(atr_window)
        
        current_price = closes[i]
        
        # 檢查持倉
        if position is not None:
            # 計算進場後的價格變化
            pnl_pct = (current_price - entry_price) / entry_price * 100
            
            # 止損
            if pnl_pct <= -stop_loss_atr * atr / entry_price * 100:
                trades.append({'type': 'LOSS', 'pnl': pnl_pct, 'holding': i - entry_bar})
                position = None
            # 止盈
            elif pnl_pct >= take_profit_atr * atr / entry_price * 100:
                trades.append({'type': 'WIN', 'pnl': pnl_pct, 'holding': i - entry_bar})
                position = None
            # 持倉過長
            elif i - entry_bar > 50:
                trades.append({'type': 'TIMEOUT', 'pnl': pnl_pct, 'holding': i - entry_bar})
                position = None
        
        # 進場信號
        if position is None:
            if vol_ratio >= volume_multiplier and trend_change > 0.5:
                position = 'LONG'
                entry_price = current_price
                entry_bar = i
            elif vol_ratio >= volume_multiplier and trend_change < -0.5:
                position = 'SHORT'
                entry_price = current_price
                entry_bar = i
    
    if position is not None and len(df) > 0:
        pnl_pct = (closes[-1] - entry_price) / entry_price * 100
        trades.append({'type': 'END', 'pnl': pnl_pct, 'holding': len(df) - entry_bar})
    
    return trades


# 參數空間
param_combinations = []
for vol_ma in [10, 20, 30]:
    for vol_mult in [1.5, 2.0, 2.5, 3.0]:
        for trend_p in [3, 5, 10]:
            for sl in [1.0, 2.0]:
                for tp in [2.0, 3.0, 4.0]:
                    param_combinations.append({
                        'volume_ma_period': vol_ma,
                        'volume_multiplier': vol_mult,
                        'trend_period': trend_p,
                        'stop_loss_atr': sl,
                        'take_profit_atr': tp,
                    })

print(f"\n🔬 測試 {len(param_combinations)} 種參數組合...")

results = []
for i, params in enumerate(param_combinations):
    if i % 20 == 0:
        print(f"  進度: {i}/{len(param_combinations)}")
    
    trades = backtest_volume_strategy(df, **params)
    
    if len(trades) >= 5:
        wins = [t for t in trades if t['type'] == 'WIN']
        win_rate = len(wins) / len(trades) * 100 if trades else 0
        avg_win = np.mean([t['pnl'] for t in wins]) if wins else 0
        losses = [t for t in trades if t['type'] == 'LOSS']
        avg_loss = np.mean([t['pnl'] for t in losses]) if losses else 0
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else 0
        
        # Sharpe-like ratio
        pnls = [t['pnl'] for t in trades]
        sharpe = np.mean(pnls) / np.std(pnls) * np.sqrt(len(pnls)) if np.std(pnls) > 0 else 0
        
        results.append({
            **params,
            'trades': len(trades),
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'sharpe': sharpe,
        })

print(f"\n{'='*80}")
print("📊 成交量策略回測結果（按 Sharpe 排序）")
print(f"{'='*80}")
print(f"{'Vol MA':>8} {'Vol Mult':>10} {'Trend':>7} {'SL':>5} {'TP':>5} {'交易':>6} {'勝率':>8} {'PF':>8} {'Sharpe':>8}")
print(f"{'-'*80}")

# 按 Sharpe 排序
results.sort(key=lambda x: x['sharpe'], reverse=True)

for r in results[:20]:
    print(f"{r['volume_ma_period']:>8} {r['volume_multiplier']:>10.1f} {r['trend_period']:>7} {r['stop_loss_atr']:>5.1f} {r['take_profit_atr']:>5.1f} {r['trades']:>6} {r['win_rate']:>7.1f}% {r['profit_factor']:>8.2f} {r['sharpe']:>8.2f}")

# 按勝率排序
print(f"\n{'='*80}")
print("📊 按勝率排序（勝率 ≥ 50%）")
print(f"{'='*80}")

high_wr = [r for r in results if r['win_rate'] >= 50]
high_wr.sort(key=lambda x: x['win_rate'], reverse=True)

for r in high_wr[:15]:
    print(f"{r['volume_ma_period']:>8} {r['volume_multiplier']:>10.1f} {r['trend_period']:>7} {r['stop_loss_atr']:>5.1f} {r['take_profit_atr']:>5.1f} {r['trades']:>6} {r['win_rate']:>7.1f}% {r['profit_factor']:>8.2f} {r['sharpe']:>8.2f}")

print(f"\n共找到 {len(high_wr)} 個勝率≥50%的策略")
