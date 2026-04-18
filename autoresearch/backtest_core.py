#!/usr/bin/env python3
"""
backtest_core.py - 標準化回測框架 v2

新增功能：
- Trailing Stop（追蹤止損）

使用方法：
```python
from backtest_core import BacktestEngine, load_data, compute_indicators

df = load_data()
df = compute_indicators(df)

engine = BacktestEngine(df)

# 固定 SL/TP
result = engine.run(signals, sl_pct=0.02, tp_pct=0.04, max_bars=20)

# 追蹤止損（推薦！）
result = engine.run(signals, sl_pct=0.02, tp_pct=0.04, max_bars=20,
                    use_trailing_stop=True, trail_pct=0.02)
```

Trailing Stop：
- 進場後，SL 會跟隨市場移動
- 做多：SL = 峰值 × (1 - trail_pct)
- 做空：SL = 谷值 × (1 + trail_pct)
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Tuple

# ============ 常數 ============
DATA_PATH = "/Users/changrunlin/.openclaw/workspace/crypto-agent-platform/data/btcusdt_15m.parquet"
COST = 0.0003  # 0.03% 單邊（手續費 + 滑點）


# ============ 數據載入 ============

def load_data(symbol: str = "btcusdt_15m") -> pd.DataFrame:
    """載入 K 線數據"""
    path = DATA_PATH if symbol == "btcusdt_15m" else f"data/{symbol}_15m.parquet"
    df = pd.read_parquet(path)
    return df


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """計算常用指標"""
    closes = df['close'].values.astype(float)
    highs = df['high'].values.astype(float)
    lows = df['low'].values.astype(float)
    volumes = df['volume'].values.astype(float)
    n = len(closes)
    
    # ATR
    tr = np.zeros(n - 1)
    for i in range(1, n):
        hl = highs[i] - lows[i]
        hc = abs(highs[i] - closes[i-1])
        lc = abs(lows[i] - closes[i-1])
        tr[i-1] = max(hl, hc, lc)
    atr = np.zeros(n)
    for i in range(14, n):
        atr[i] = np.mean(tr[i-14:i])
    df['atr'] = atr
    
    # RSI (14)
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    rsi = np.zeros(n)
    for i in range(14, n):
        gain_avg = np.mean(gains[i-14:i])
        loss_avg = np.mean(losses[i-14:i])
        rs = gain_avg / (loss_avg + 1e-10)
        rsi[i] = 100 - (100 / (1 + rs))
    df['rsi'] = rsi
    
    # MA
    df['ma20'] = pd.Series(closes).rolling(20).mean().values
    df['ma50'] = pd.Series(closes).rolling(50).mean().values
    
    # 成交量均線
    df['vol_ma20'] = pd.Series(volumes).rolling(20).mean().values
    
    # CCI
    typical = (highs + lows + closes) / 3
    sma = pd.Series(typical).rolling(20).mean().values
    mad = pd.Series(np.abs(typical - sma)).rolling(20).mean().values
    cci = np.zeros(n)
    for i in range(20, n):
        if mad[i] != 0:
            cci[i] = (typical[i] - sma[i]) / (0.015 * mad[i])
    df['cci'] = cci
    
    # ADX
    dm_pos = np.zeros(n - 1)
    dm_neg = np.zeros(n - 1)
    for i in range(1, n):
        hl = highs[i] - lows[i]
        hc = abs(highs[i] - closes[i-1])
        lc = abs(lows[i] - closes[i-1])
        dm_pos[i-1] = highs[i] - highs[i-1] if (highs[i] - highs[i-1]) > (lows[i-1] - lows[i]) else 0
        dm_neg[i-1] = lows[i-1] - lows[i] if (lows[i-1] - lows[i]) > (highs[i] - highs[i-1]) else 0
    
    di_pos = np.zeros(n)
    di_neg = np.zeros(n)
    for i in range(14, n):
        s_pos = np.sum(dm_pos[i-14:i])
        s_neg = np.sum(dm_neg[i-14:i])
        s_tr = np.sum(tr[i-14:i])
        if s_tr > 0:
            di_pos[i] = s_pos / s_tr * 100
            di_neg[i] = s_neg / s_tr * 100
    
    dx = np.zeros(n)
    for i in range(14, n):
        if (di_pos[i] + di_neg[i]) > 0:
            dx[i] = abs(di_pos[i] - di_neg[i]) / (di_pos[i] + di_neg[i]) * 100
    
    adx = np.zeros(n)
    adx[14] = np.mean(dx[1:15])
    for i in range(15, n):
        adx[i] = (adx[i-1] * 13 + dx[i]) / 14
    df['adx'] = adx
    df['di_pos'] = di_pos
    df['di_neg'] = di_neg
    
    return df


# ============ 回測引擎 ============

@dataclass
class Trade:
    entry_price: float
    exit_price: float
    direction: int  # 1=long, -1=short
    pnl_pct: float
    won: bool
    exit_reason: str  # 'STOP_LOSS', 'TAKE_PROFIT', 'TIMEOUT', 'TRAILING_STOP'
    holding_bars: int
    peak_price: float = 0.0  # 做多時的最高價
    trough_price: float = 0.0  # 做空時的最低價


class BacktestEngine:
    """標準化回測引擎 v2"""
    
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.closes = data['close'].values.astype(float)
        self.highs = data['high'].values.astype(float)
        self.lows = data['low'].values.astype(float)
        self.n = len(self.closes)
    
    def run(self, 
            signals: np.ndarray,
            sl_pct: float,
            tp_pct: float,
            max_bars: int = 20,
            initial_capital: float = 10000.0,
            use_trailing_stop: bool = False,
            trail_pct: float = 0.02,
            activation_pct: float = 0.01) -> Dict:
        """
        執行回測
        
        成本計算：進場 0.03% + 出場 0.03% = 0.06% 總成本
        
        Args:
            signals: 信號陣列（1=做多, -1=做空, 0=無信號）
            sl_pct: 止損百分比（相對於進場價格）
            tp_pct: 止盈百分比
            max_bars: 最大持倉 K 線數
            use_trailing_stop: 是否使用追蹤止損
            trail_pct: 追蹤止損百分比
            activation_pct: 追蹤止損激活百分比（從進場價格的盈利超過此值時才激活）
        
        Returns:
            dict: 包含 wr, pf, dd, sharpe, trades, monthly_trades 等
        """
        trades = []
        position = None
        
        for i in range(self.n - 1):
            # 沒有倉位時，檢查是否需要進場
            if position is None:
                if signals[i] != 0:
                    # Next-bar Entry：信號在 bar i 產生，以 bar i+1 開盤價進場
                    entry_price = self.closes[i + 1]
                    position = {
                        'direction': signals[i],
                        'entry_price': entry_price,
                        'entry_bar': i + 1,
                        '_cost_paid': False,
                        '_peak_price': entry_price,  # 做多的最高價
                        '_trough_price': entry_price,  # 做空的最低價
                        '_trail_activated': False,  # 追蹤止損是否已激活
                        '_trail_stop_price': 0,  # 追蹤止損價格
                    }
            else:
                # 有倉位，檢查是否需要出場
                direction = position['direction']
                entry_price = position['entry_price']
                current_price = self.closes[i]
                holding_bars = i - position['entry_bar']
                
                # 更新 peak/trough
                if direction > 0:
                    if self.highs[i] > position['_peak_price']:
                        position['_peak_price'] = self.highs[i]
                    # 計算當前 SL（追蹤止損）
                    if use_trailing_stop:
                        profit_pct = (position['_peak_price'] - entry_price) / entry_price
                        if profit_pct >= activation_pct:
                            position['_trail_activated'] = True
                            position['_trail_stop_price'] = position['_peak_price'] * (1 - trail_pct)
                            # 如果價格跌破 trail_stop，立即止損
                            if self.lows[i] <= position['_trail_stop_price']:
                                trades.append(Trade(
                                    entry_price=entry_price,
                                    exit_price=position['_trail_stop_price'],
                                    direction=direction,
                                    pnl_pct=(position['_trail_stop_price'] - entry_price) / entry_price - COST,
                                    won=False,
                                    exit_reason='TRAILING_STOP',
                                    holding_bars=holding_bars,
                                    peak_price=position['_peak_price']
                                ))
                                position = None
                                continue
                else:  # short
                    if self.lows[i] < position['_trough_price']:
                        position['_trough_price'] = self.lows[i]
                    if use_trailing_stop:
                        profit_pct = (entry_price - position['_trough_price']) / entry_price
                        if profit_pct >= activation_pct:
                            position['_trail_activated'] = True
                            position['_trail_stop_price'] = position['_trough_price'] * (1 + trail_pct)
                            if self.highs[i] >= position['_trail_stop_price']:
                                trades.append(Trade(
                                    entry_price=entry_price,
                                    exit_price=position['_trail_stop_price'],
                                    direction=direction,
                                    pnl_pct=(entry_price - position['_trail_stop_price']) / entry_price - COST,
                                    won=False,
                                    exit_reason='TRAILING_STOP',
                                    holding_bars=holding_bars,
                                    trough_price=position['_trough_price']
                                ))
                                position = None
                                continue
                
                # 計算未實現損益
                if direction > 0:
                    pnl_pct = (current_price - entry_price) / entry_price
                else:
                    pnl_pct = (entry_price - current_price) / entry_price
                
                # 進場成本（只在第一次檢查時扣）
                if not position['_cost_paid']:
                    pnl_pct -= COST
                    position['_cost_paid'] = True
                
                exit_reason = None
                final_pnl = pnl_pct
                
                # 檢查止損
                if pnl_pct <= -sl_pct:
                    exit_reason = 'STOP_LOSS'
                    final_pnl = -sl_pct - COST
                # 檢查止盈
                elif pnl_pct >= tp_pct:
                    exit_reason = 'TAKE_PROFIT'
                    final_pnl = tp_pct - COST
                # 檢查超時
                elif holding_bars >= max_bars:
                    exit_reason = 'TIMEOUT'
                    final_pnl = final_pnl - COST
                
                if exit_reason:
                    trades.append(Trade(
                        entry_price=entry_price,
                        exit_price=current_price,
                        direction=direction,
                        pnl_pct=final_pnl,
                        won=final_pnl > 0,
                        exit_reason=exit_reason,
                        holding_bars=holding_bars,
                        peak_price=position.get('_peak_price', 0),
                        trough_price=position.get('_trough_price', 0)
                    ))
                    position = None
        
        return self._calculate_metrics(trades)
    
    def _calculate_metrics(self, trades: list) -> Dict:
        """計算策略指標"""
        if not trades:
            return {
                'wr': 0, 'pf': 0, 'dd': 0, 'sharpe': 0,
                'trades': 0, 'monthly_trades': 0,
                'avg_win': 0, 'avg_loss': 0,
                'won': 0, 'lost': 0,
                'passed': False
            }
        
        n_trades = len(trades)
        won_trades = [t for t in trades if t.won]
        lost_trades = [t for t in trades if not t.won]
        
        wr = len(won_trades) / n_trades * 100
        
        avg_win = np.mean([t.pnl_pct for t in won_trades]) if won_trades else 0
        avg_loss = abs(np.mean([t.pnl_pct for t in lost_trades])) if lost_trades else 1e-10
        
        # Profit Factor
        gross_win = avg_win * len(won_trades)
        gross_loss = avg_loss * len(lost_trades)
        pf = gross_win / gross_loss if gross_loss > 0 else 0
        
        # Max Drawdown
        equity = [1.0]
        for t in trades:
            equity.append(equity[-1] * (1 + t.pnl_pct))
        
        peak = 1.0
        max_dd = 0.0
        for e in equity:
            if e > peak:
                peak = e
            dd = (peak - e) / peak
            if dd > max_dd:
                max_dd = dd
        dd_pct = max_dd * 100
        
        # Sharpe Ratio（年化）
        returns = [t.pnl_pct for t in trades]
        if np.std(returns) > 0:
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252 * 96)
        else:
            sharpe = 0
        
        # 月交易量（假設 24 個月）
        monthly_trades = n_trades / 24
        
        passed = (
            wr >= 60 and
            pf >= 1.8 and
            dd_pct <= 25 and
            sharpe >= 1.5 and
            monthly_trades >= 100
        )
        
        return {
            'wr': wr,
            'pf': pf,
            'dd': dd_pct,
            'sharpe': sharpe,
            'trades': n_trades,
            'monthly_trades': monthly_trades,
            'avg_win': avg_win * 100,
            'avg_loss': avg_loss * 100,
            'won': len(won_trades),
            'lost': len(lost_trades),
            'passed': passed,
            'exit_stats': self._exit_stats(trades)
        }
    
    def _exit_stats(self, trades: list) -> Dict:
        """計算 exit 分佈"""
        sl_count = sum(1 for t in trades if t.exit_reason == 'STOP_LOSS')
        tp_count = sum(1 for t in trades if t.exit_reason == 'TAKE_PROFIT')
        to_count = sum(1 for t in trades if t.exit_reason == 'TIMEOUT')
        ts_count = sum(1 for t in trades if t.exit_reason == 'TRAILING_STOP')
        total = len(trades)
        
        return {
            'stop_loss': f"{sl_count/total*100:.1f}%" if total > 0 else "0%",
            'take_profit': f"{tp_count/total*100:.1f}%" if total > 0 else "0%",
            'timeout': f"{to_count/total*100:.1f}%" if total > 0 else "0%",
            'trailing_stop': f"{ts_count/total*100:.1f}%" if total > 0 else "0%"
        }


# ============ 便捷函數 ============

def quick_backtest(signals: np.ndarray,
                   closes: np.ndarray,
                   sl_pct: float,
                   tp_pct: float,
                   max_bars: int = 20,
                   use_trailing_stop: bool = False,
                   trail_pct: float = 0.02) -> Dict:
    """快速回測（只需信號和收盤價）"""
    data = pd.DataFrame({'close': closes, 'high': closes, 'low': closes})
    engine = BacktestEngine(data)
    return engine.run(signals, sl_pct, tp_pct, max_bars,
                      use_trailing_stop=use_trailing_stop, trail_pct=trail_pct)


# ============ 測試 ============

if __name__ == "__main__":
    print("載入數據...")
    df = load_data()
    print(f"數據筆數: {len(df)}")
    
    print("\n計算指標...")
    df = compute_indicators(df)
    
    print("\n測試均線交叉策略（無 Trailing Stop）...")
    signals = np.zeros(len(df))
    closes = df['close'].values
    
    for i in range(50, len(df) - 1):
        ma20 = df['ma20'].iloc[i]
        ma50 = df['ma50'].iloc[i]
        ma20_prev = df['ma20'].iloc[i-1]
        ma50_prev = df['ma50'].iloc[i-1]
        
        if ma20 > ma50 and ma20_prev <= ma50_prev:
            signals[i] = 1
        elif ma20 < ma50 and ma20_prev >= ma50_prev:
            signals[i] = -1
    
    engine = BacktestEngine(df)
    results = engine.run(signals, sl_pct=0.02, tp_pct=0.04, max_bars=20)
    
    print(f"\n結果（無 Trailing Stop）:")
    print(f"  WR: {results['wr']:.2f}%")
    print(f"  PF: {results['pf']:.2f}")
    print(f"  DD: {results['dd']:.2f}%")
    print(f"  Sharpe: {results['sharpe']:.2f}")
    print(f"  月交易: {results['monthly_trades']:.1f}")
    print(f"  Exit: {results['exit_stats']}")
    print(f"  通過: {results['passed']}")
    
    print("\n測試均線交叉策略（有 Trailing Stop）...")
    results_trail = engine.run(signals, sl_pct=0.02, tp_pct=0.04, max_bars=20,
                               use_trailing_stop=True, trail_pct=0.02, activation_pct=0.01)
    
    print(f"\n結果（有 Trailing Stop）:")
    print(f"  WR: {results_trail['wr']:.2f}%")
    print(f"  PF: {results_trail['pf']:.2f}")
    print(f"  DD: {results_trail['dd']:.2f}%")
    print(f"  Sharpe: {results_trail['sharpe']:.2f}")
    print(f"  月交易: {results_trail['monthly_trades']:.1f}")
    print(f"  Exit: {results_trail['exit_stats']}")
    print(f"  通過: {results_trail['passed']}")
