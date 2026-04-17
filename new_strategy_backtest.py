"""
新策略研究回測腳本
用於測試團隊研究的新型交易策略

Target: WR≥50%, PF≥2.0, DD≤30%, Sharpe≥1.5
"""

import pandas as pd
import numpy as np
import json
import sys
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# ============ 載入數據並計算指標 ============
DATA_PATH = "data/btcusdt_1d.parquet"

def load_and_compute():
    df = pd.read_parquet(DATA_PATH)
    df = df.sort_values('open_time').reset_index(drop=True)
    
    # === 基本指標 ===
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
    df['EMA8'] = df['close'].ewm(span=8).mean()
    df['EMA9'] = df['close'].ewm(span=9).mean()
    df['EMA21'] = df['close'].ewm(span=21).mean()
    df['EMA40'] = df['close'].ewm(span=40).mean()
    
    # 7日動量
    df['momentum_7d'] = df['close'].pct_change(7) * 100
    
    # ATR
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR14'] = tr.rolling(14).mean()
    
    # Regime
    df['regime'] = np.where(df['close'] > df['MA200'], 'bull', 'bear')
    
    # Golden/Death Cross (MA20/MA50)
    df['MA20_above_MA50'] = df['MA20'] > df['MA50']
    df['golden_cross'] = (df['MA20_above_MA50']) & (~df['MA20_above_MA50'].shift(1).fillna(False))
    df['death_cross'] = (~df['MA20_above_MA50']) & (df['MA20_above_MA50'].shift(1).fillna(False))
    
    # EMA8/EMA9 cross
    df['EMA8_above_EMA9'] = df['EMA8'] > df['EMA9']
    df['EMA8_cross_above_EMA9'] = (df['EMA8_above_EMA9']) & (~df['EMA8_above_EMA9'].shift(1).fillna(False))
    df['EMA8_cross_below_EMA9'] = (~df['EMA8_above_EMA9']) & (df['EMA8_above_EMA9'].shift(1).fillna(False))
    
    # EMA8/EMA40 cross
    df['EMA8_above_EMA40'] = df['EMA8'] > df['EMA40']
    df['EMA8_cross_above_EMA40'] = (df['EMA8_above_EMA40']) & (~df['EMA8_above_EMA40'].shift(1).fillna(False))
    df['EMA8_cross_below_EMA40'] = (~df['EMA8_above_EMA40']) & (df['EMA8_above_EMA40'].shift(1).fillna(False))
    
    # EMA10/EMA15 cross
    df['EMA10'] = df['close'].ewm(span=10).mean()
    df['EMA15'] = df['close'].ewm(span=15).mean()
    df['EMA10_above_EMA15'] = df['EMA10'] > df['EMA15']
    df['EMA10_cross_above_EMA15'] = (df['EMA10_above_EMA15']) & (~df['EMA10_above_EMA15'].shift(1).fillna(False))
    df['EMA10_cross_below_EMA15'] = (~df['EMA10_above_EMA15']) & (df['EMA10_above_EMA15'].shift(1).fillna(False))
    
    # MA200 slope
    df['MA200_slope'] = df['MA200'].diff(5) / df['MA200'].shift(5) * 100
    
    # Volume MA
    df['volume_ma20'] = df['volume'].rolling(20).mean()
    
    return df


# ============ 策略定義 ============

class Strategy:
    def __init__(self, name, df, params):
        self.name = name
        self.df = df
        self.params = params
        
    def check_long(self, i):
        raise NotImplementedError
    def check_short(self, i):
        raise NotImplementedError


class LONG_RegimeMA_RSIPullback(Strategy):
    """LONG-ONLY: Regime + Golden Cross + RSI Pullback"""
    def check_long(self, i):
        p = self.params
        d = self.df.iloc[i]
        
        if d['regime'] != 'bull': return False
        if not d['golden_cross']: return False
        if d['RSI14'] < p.get('rsi_min', 40): return False
        if d['RSI14'] > p.get('rsi_max', 55): return False
        if d['momentum_7d'] < p.get('momentum_min', 0): return False
        return True
    
    def check_short(self, i):
        return False  # LONG only


class Dual_EMA8_40_Regime(Strategy):
    """雙向: EMA8/40 + Regime Filter + RSI"""
    def check_long(self, i):
        p = self.params
        d = self.df.iloc[i]
        
        if not d['EMA8_cross_above_EMA40']: return False
        if d['regime'] != 'bull': return False
        if d['RSI14'] < p.get('rsi_long_min', 45): return False
        if d['RSI14'] > p.get('rsi_long_max', 70): return False
        if d['momentum_7d'] < p.get('momentum_min', 0): return False
        return True
    
    def check_short(self, i):
        p = self.params
        d = self.df.iloc[i]
        
        if not d['EMA8_cross_below_EMA40']: return False
        if d['regime'] != 'bear': return False
        if d['RSI14'] > p.get('rsi_short_max', 55): return False
        if d['RSI14'] < p.get('rsi_short_min', 30): return False
        if d['momentum_7d'] > p.get('momentum_max', 0): return False
        return True


class LONG_RSIDip_GoldenCross(Strategy):
    """LONG-ONLY: RSI Dip after Golden Cross"""
    def check_long(self, i):
        p = self.params
        d = self.df.iloc[i]
        
        # 需要Golden Cross發生在過去N天內
        gc_recent = self.df.iloc[max(0, i-p.get('gc_lookback', 5)):i+1]['golden_cross'].any()
        if not gc_recent: return False
        if d['regime'] != 'bull': return False
        if d['RSI14'] < p.get('rsi_max', 40): return False  # RSI not oversold
        if d['RSI14'] > p.get('rsi_min', 60): return False  # RSI confirming strength
        if d['close'] < d['MA20']: return False  # price above MA20
        return True
    
    def check_short(self, i):
        return False


class LONG_Breakout_MA50_Confirm(Strategy):
    """LONG-ONLY: EMA10/15 Breakout + MA50 Confirm"""
    def check_long(self, i):
        p = self.params
        d = self.df.iloc[i]
        
        if not d['EMA10_cross_above_EMA15']: return False
        if d['regime'] != 'bull': return False
        if d['RSI14'] < p.get('rsi_min', 45): return False
        if d['RSI14'] > p.get('rsi_max', 70): return False
        if d['close'] < d['MA50']: return False  # price above MA50
        if d['volume'] < p.get('vol_mult', 1.0) * d['volume_ma20']: return False
        return True
    
    def check_short(self, i):
        return False


class Dual_MA_Cluster_Break(Strategy):
    """雙向: 多均線彙聚後突破"""
    def check_long(self, i):
        p = self.params
        d = self.df.iloc[i]
        
        # 均線收斂：MA20和MA50接近
        ma_spread = abs(d['MA20'] - d['MA50']) / d['MA50'] * 100
        if ma_spread > p.get('ma_spread_max', 2.0): return False
        
        # 突破確認
        if d['close'] <= d['MA20']: return False
        if not d['EMA10_cross_above_EMA15']: return False
        
        if d['regime'] != 'bull': return False
        if d['RSI14'] < p.get('rsi_min', 50): return False
        return True
    
    def check_short(self, i):
        p = self.params
        d = self.df.iloc[i]
        
        ma_spread = abs(d['MA20'] - d['MA50']) / d['MA50'] * 100
        if ma_spread > p.get('ma_spread_max', 2.0): return False
        
        if d['close'] >= d['MA20']: return False
        if not d['EMA10_cross_below_EMA15']: return False
        
        if d['regime'] != 'bear': return False
        if d['RSI14'] > p.get('rsi_max', 50): return False
        return True


class LONG_RSIBreak_50_Regime(Strategy):
    """LONG-ONLY: RSI從極低區反轉 + Regime Filter"""
    def check_long(self, i):
        p = self.params
        d = self.df.iloc[i]
        
        if d['regime'] != 'bull': return False
        if d['RSI14'] < p.get('rsi_oversold', 40): return False  # RSI recovering from oversold
        if d['RSI14'] > p.get('rsi_entry', 55): return False    # RSI not too high
        if d['momentum_7d'] < p.get('momentum_min', 0): return False
        
        # 需要MA50向上
        ma50_slope = self.df.iloc[max(0, i-5):i+1]['MA50'].diff().sum()
        if ma50_slope <= 0: return False
        
        return True
    
    def check_short(self, i):
        return False


class Dual_EMA8_40_TightRSI(Strategy):
    """雙向: EMA8/40交叉 + 嚴格RSI對稱"""
    def check_long(self, i):
        p = self.params
        d = self.df.iloc[i]
        
        if not d['EMA8_cross_above_EMA40']: return False
        if d['regime'] != 'bull': return False
        if d['RSI14'] < p.get('rsi_long_min', 48): return False
        if d['RSI14'] > p.get('rsi_long_max', 65): return False
        if d['momentum_7d'] < p.get('momentum_min', 0): return False
        if d['close'] < d['MA200']: return False
        return True
    
    def check_short(self, i):
        p = self.params
        d = self.df.iloc[i]
        
        if not d['EMA8_cross_below_EMA40']: return False
        if d['regime'] != 'bear': return False
        if d['RSI14'] > p.get('rsi_short_max', 52): return False
        if d['RSI14'] < p.get('rsi_short_min', 35): return False
        if d['momentum_7d'] > p.get('momentum_max', 0): return False
        if d['close'] > d['MA200']: return False
        return True


# ============ 止損止盈計算 ============

def calc_sl_tp_pct(entry_price, sl_pct, tp_pct):
    """計算止損止盈價格"""
    sl_price = entry_price * (1 - sl_pct)
    tp_price = entry_price * (1 + tp_pct)
    return sl_price, tp_price


def calc_sl_tp_atr(entry_price, atr, sl_atr, tp_atr):
    """計算ATR-based止損止盈"""
    sl_price = entry_price - sl_atr * atr
    tp_price = entry_price + tp_atr * atr
    return sl_price, tp_price


# ============ 核心回測引擎 ============

def run_backtest(df, strategy, sl_pct=0.03, tp_pct=0.07, max_holding=15, initial_capital=10000.0, maker_fee=0.0002, taker_fee=0.0004):
    """
    跑回測
    
    Args:
        df: 計算過指標的DataFrame
        strategy: Strategy 實例
        sl_pct: 止損百分比 (如 0.03 = 3%)
        tp_pct: 止盈百分比 (如 0.07 = 7%)
        max_holding: 最大持倉天數
        initial_capital: 初始資金
        maker_fee: Maker費用
        taker_fee: Taker費用
    
    Returns:
        dict: 回測結果
    """
    trades = []
    position = None
    cash = initial_capital
    equity_curve = []
    
    start_i = max(200, 250)  # 需要足夠歷史數據計算指標
    
    for i in range(start_i, len(df)):
        d = df.iloc[i]
        ts = d['open_time']
        
        # === 計算權益 ===
        if position:
            # 未實現PnL
            if position['side'] == 'LONG':
                unrealized = (d['close'] - position['entry_price']) * position['qty']
            else:
                unrealized = (position['entry_price'] - d['close']) * position['qty']
            equity = cash + unrealized
        else:
            equity = cash
        
        peak = max([e[1] for e in equity_curve] + [equity]) if equity_curve else equity
        dd = (peak - equity) / peak * 100 if peak > 0 else 0
        equity_curve.append((ts, equity, dd))
        
        # === 持倉中 ===
        if position:
            bars_held = position['bars_held'] + 1
            entry_price = position['entry_price']
            qty = position['qty']
            
            # 計算止損止盈
            if position['atr_based']:
                sl_price = position['sl_price']
                tp_price = position['tp_price']
            else:
                sl_price = position['sl_price']
                tp_price = position['tp_price']
            
            # 檢查是否觸發止損/止盈
            hit_sl = (position['side'] == 'LONG' and d['low'] <= sl_price) or \
                     (position['side'] == 'SHORT' and d['high'] >= sl_price)
            hit_tp = (position['side'] == 'LONG' and d['high'] >= tp_price) or \
                     (position['side'] == 'SHORT' and d['low'] <= tp_price)
            
            # 平倉條件
            should_exit = (bars_held >= max_holding) or hit_sl or hit_tp
            
            if should_exit:
                if position['side'] == 'LONG':
                    exit_price = min(d['close'], sl_price) if hit_sl else tp_price
                    pnl = (exit_price - entry_price) * qty
                else:
                    exit_price = max(d['close'], sl_price) if hit_sl else tp_price
                    pnl = (entry_price - exit_price) * qty
                
                # 扣除費用
                fees = d['close'] * qty * taker_fee
                net_pnl = pnl - fees
                cash += net_pnl
                
                trades.append({
                    'entry_time': position['entry_time'],
                    'exit_time': ts,
                    'side': position['side'],
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'qty': qty,
                    'pnl': net_pnl,
                    'pnl_pct': net_pnl / (entry_price * qty) * 100,
                    'fees': fees,
                    'bars': bars_held,
                    'exit_reason': 'timeout' if bars_held >= max_holding else ('sl' if hit_sl else 'tp')
                })
                position = None
        
        # === 開倉 ===
        if not position:
            long_signal = strategy.check_long(i)
            short_signal = strategy.check_short(i)
            
            if long_signal or short_signal:
                side = 'LONG' if long_signal else 'SHORT'
                entry_price = d['close']
                
                # 計算倉位大小 (20%風險)
                risk_amount = cash * 0.2
                atr = d['ATR14']
                if atr and not np.isnan(atr) and atr > 0:
                    qty = risk_amount / (sl_pct * entry_price)  # 根據止損金額
                    qty = min(qty, cash * 0.95 / entry_price)  # 不超過cash
                else:
                    qty = cash * 0.2 / entry_price
                qty = max(qty, 0.001)
                
                # 止損止盈
                if 'atr' in strategy.params and strategy.params['atr']:
                    sl_price, tp_price = calc_sl_tp_atr(entry_price, atr, sl_pct, tp_pct)
                    atr_based = True
                else:
                    sl_price, tp_price = calc_sl_tp_pct(entry_price, sl_pct, tp_pct)
                    atr_based = False
                
                position = {
                    'side': side,
                    'entry_time': ts,
                    'entry_price': entry_price,
                    'qty': qty,
                    'sl_price': sl_price,
                    'tp_price': tp_price,
                    'bars_held': 0,
                    'atr_based': atr_based
                }
    
    # === 計算績效 ===
    if not trades:
        return {
            'total_trades': 0, 'wins': 0, 'losses': 0,
            'win_rate': 0, 'profit_factor': 0, 'max_drawdown': 0,
            'sharpe': 0, 'total_return': 0
        }
    
    wins = [t for t in trades if t['pnl'] > 0]
    losses = [t for t in trades if t['pnl'] <= 0]
    
    win_rate = len(wins) / len(trades) * 100
    gross_profit = sum(t['pnl'] for t in wins)
    gross_loss = abs(sum(t['pnl'] for t in losses))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    # Max Drawdown
    equity_series = [initial_capital]
    for t in trades:
        equity_series.append(equity_series[-1] + t['pnl'])
    
    peak = equity_series[0]
    max_dd = 0
    for eq in equity_series:
        if eq > peak:
            peak = eq
        dd = (peak - eq) / peak * 100
        if dd > max_dd:
            max_dd = dd
    
    # Sharpe (使用日權益曲線)
    if len(equity_curve) > 10:
        eq_series = pd.Series([e[1] for e in equity_curve])
        rets = eq_series.pct_change().dropna()
        sharpe = rets.mean() / rets.std() * np.sqrt(365) if rets.std() > 0 else 0
    else:
        sharpe = 0
    
    total_return = (cash - initial_capital) / initial_capital * 100
    
    # TP/SL timeout ratio
    tp_exits = sum(1 for t in trades if t['exit_reason'] == 'tp')
    sl_exits = sum(1 for t in trades if t['exit_reason'] == 'sl')
    timeout_exits = sum(1 for t in trades if t['exit_reason'] == 'timeout')
    
    return {
        'strategy': strategy.name,
        'total_trades': len(trades),
        'wins': len(wins),
        'losses': len(losses),
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'max_drawdown': max_dd,
        'sharpe': sharpe,
        'total_return': total_return,
        'tp_exits': tp_exits,
        'sl_exits': sl_exits,
        'timeout_exits': timeout_exits,
        'final_capital': cash,
        'trades': trades
    }


# ============ 目標達標檢查 ============

TARGETS = {
    'win_rate': ('WR', 50, 'gte'),
    'profit_factor': ('PF', 2.0, 'gte'),
    'max_drawdown': ('DD', 30, 'lte'),
    'sharpe': ('Sharpe', 1.5, 'gte')
}

def check_targets(metrics):
    """檢查是否通過所有目標"""
    results = {}
    all_pass = True
    
    for key, (label, threshold, direction) in TARGETS.items():
        value = metrics.get(key, 0)
        if direction == 'gte':
            passed = value >= threshold
        else:
            passed = value <= threshold
        
        results[f"{label}_{threshold}"] = "✅ PASS" if passed else "❌ FAIL"
        if not passed:
            all_pass = False
    
    return all_pass, results


# ============ 主程序 ============

def main():
    print("=" * 70)
    print("🔬 新策略研究 - 回測引擎")
    print("=" * 70)
    
    # 載入數據
    df = load_and_compute()
    print(f"\n📊 數據加載完成: {len(df)} 根K線")
    print(f"   時間範圍: {df['open_time'].min()} ~ {df['open_time'].max()}")
    
    # ============ 策略定義 ============
    strategies = [
        # Strategy 1: LONG-ONLY RSI Pullback after Golden Cross
        (LONG_RegimeMA_RSIPullback, {
            'name': 'LONG_RSIPullback_GoldenCross',
            'rsi_min': 45,
            'rsi_max': 55,
            'momentum_min': 0
        }, 0.03, 0.07, 15),
        
        # Strategy 2: Dual EMA8/40 + Regime (tight RSI)
        (Dual_EMA8_40_TightRSI, {
            'name': 'Dual_EMA8_40_TightRSI',
            'rsi_long_min': 48,
            'rsi_long_max': 65,
            'rsi_short_max': 52,
            'rsi_short_min': 35,
            'momentum_min': 0,
            'momentum_max': 0,
        }, 0.03, 0.07, 15),
        
        # Strategy 3: LONG RSI Dip + Golden Cross
        (LONG_RSIDip_GoldenCross, {
            'name': 'LONG_RSIDip_GoldenCross',
            'rsi_min': 50,
            'rsi_max': 60,
            'gc_lookback': 5,
        }, 0.03, 0.07, 15),
        
        # Strategy 4: LONG Breakout MA50 Confirm
        (LONG_Breakout_MA50_Confirm, {
            'name': 'LONG_Breakout_MA50_Confirm',
            'rsi_min': 45,
            'rsi_max': 70,
            'vol_mult': 1.0,
        }, 0.03, 0.07, 15),
        
        # Strategy 5: Dual MA Cluster Break
        (Dual_MA_Cluster_Break, {
            'name': 'Dual_MA_Cluster_Break',
            'ma_spread_max': 2.0,
            'rsi_min': 50,
            'rsi_max': 70,
        }, 0.03, 0.07, 15),
        
        # Strategy 6: LONG RSI Break from 40 + MA50 Up
        (LONG_RSIBreak_50_Regime, {
            'name': 'LONG_RSIBreak_50_Regime',
            'rsi_oversold': 40,
            'rsi_entry': 55,
            'momentum_min': 0,
        }, 0.03, 0.07, 15),
        
        # Strategy 7: Dual EMA8/40 + Regime with higher TP
        (Dual_EMA8_40_TightRSI, {
            'name': 'Dual_EMA8_40_HighTP',
            'rsi_long_min': 48,
            'rsi_long_max': 65,
            'rsi_short_max': 52,
            'rsi_short_min': 35,
            'momentum_min': 0,
            'momentum_max': 0,
        }, 0.03, 0.10, 20),  # Higher TP, longer hold
        
        # Strategy 8: LONG-only, stricter RSI entry
        (LONG_RegimeMA_RSIPullback, {
            'name': 'LONG_GoldenCross_StrictRSI',
            'rsi_min': 48,
            'rsi_max': 58,
            'momentum_min': 0.5,
        }, 0.025, 0.08, 20),  # Tight SL, High TP
    ]
    
    # ============ 執行回測 ============
    results = []
    
    for strat_class, params, sl, tp, max_hold in strategies:
        name = params.pop('name')
        
        # 重新包裝為可迭代格式
        strat_params = {k: v for k, v in params.items()}
        
        # 使用修改後的策略
        class NamedStrategy(strat_class):
            pass
        NamedStrategy.__name__ = name
        
        strat = NamedStrategy(name, df, strat_params)
        
        # 重新設置name屬性
        strat.name = name
        
        metrics = run_backtest(df, strat, sl_pct=sl, tp_pct=tp, max_holding=max_hold)
        
        # 恢復params中的name
        params['name'] = name
        
        passed, target_results = check_targets(metrics)
        
        result = {
            'strategy': name,
            'sl': sl,
            'tp': tp,
            'max_hold': max_hold,
            'metrics': {k: v for k, v in metrics.items() if k != 'trades'},
            'passed': passed,
            'targets': target_results
        }
        results.append(result)
        
        print(f"\n{'='*70}")
        print(f"📈 策略: {name}")
        print(f"   SL={sl*100:.1f}%, TP={tp*100:.1f}%, MaxHold={max_hold}d")
        print(f"   總交易: {metrics['total_trades']} | 勝: {metrics['wins']} | 負: {metrics['losses']}")
        print(f"   勝率: {metrics['win_rate']:.2f}%")
        print(f"   盈虧比: {metrics['profit_factor']:.3f}")
        print(f"   最大回撤: {metrics['max_drawdown']:.2f}%")
        print(f"   夏普值: {metrics['sharpe']:.3f}")
        print(f"   TP/SL/Timeout: {metrics['tp_exits']}/{metrics['sl_exits']}/{metrics['timeout_exits']}")
        print(f"   Target Status: {'✅ ALL PASS' if passed else '❌ FAIL'}")
        
        for k, v in target_results.items():
            print(f"     {k}: {v}")
        
        results.append(result)
    
    # ============ 找出最佳策略 ============
    print(f"\n{'='*70}")
    print("🏆 結果摘要")
    print("="*70)
    
    passed_results = [r for r in results if r.get('passed', False)]
    
    if passed_results:
        print(f"\n✅ {len(passed_results)} 個策略通過所有目標:\n")
        for r in sorted(passed_results, key=lambda x: -x['metrics']['sharpe']):
            m = r['metrics']
            print(f"  {r['strategy']}: WR={m['win_rate']:.1f}%, PF={m['profit_factor']:.2f}, DD={m['max_drawdown']:.2f}%, Sharpe={m['sharpe']:.2f}")
    else:
        print("\n❌ 沒有策略通過所有4個目標")
        print("\n嘗試參數優化...")
    
    # ============ 保存結果 ============
    output = {
        'timestamp': datetime.now().isoformat(),
        'data_range': f"{df['open_time'].min()} to {df['open_time'].max()}",
        'results': [{
            'strategy': r['strategy'],
            'sl': r['sl'],
            'tp': r['tp'],
            'max_hold': r['max_hold'],
            'metrics': {k: v for k, v in r['metrics'].items() if k != 'trades'},
            'passed': r['passed'],
            'targets': r['targets']
        } for r in results]
    }
    
    with open('autoresearch/memory/new_strategy_results.json', 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"\n💾 結果已保存到 autoresearch/memory/new_strategy_results.json")
    
    return results


if __name__ == "__main__":
    results = main()
