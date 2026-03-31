"""
多因子信號系統 - Multi-Factor Signal Engine
"""

import json
import pandas as pd
from pathlib import Path
from typing import List, Dict
from dataclasses import dataclass


@dataclass
class Factor:
    factor_id: str
    name: str
    sl_atr: float
    win_rate: float
    profit_factor: float
    sharpe: float
    trade_count: int
    weight: float
    avg_entry_price: float = 0
    conditions: List[Dict] = None


class MultiFactorEngine:
    def __init__(self, factor_library_path: str = None):
        if factor_library_path is None:
            factor_library_path = Path(__file__).parent / "factor_library.json"
        
        with open(factor_library_path) as f:
            data = json.load(f)
        
        self.factors = []
        for item in data:
            conditions = self._get_conditions(item.get('factor_id', item.get('name', '')))
            self.factors.append(Factor(
                factor_id=item.get('factor_id', item.get('name', '')),
                name=item.get('name', ''),
                sl_atr=item.get('sl_atr', 1.0),
                win_rate=item.get('win_rate', 0),
                profit_factor=item.get('profit_factor', 0),
                sharpe=item.get('sharpe', 0),
                trade_count=item.get('trade_count', 0),
                weight=item.get('weight', item.get('win_rate', 0) / 100),
                avg_entry_price=item.get('avg_entry_price', 0),
                conditions=conditions
            ))
        
        self.factors.sort(key=lambda x: x.weight, reverse=True)
        self.total_weight = sum(f.weight for f in self.factors) if self.factors else 1.0
    
    def _get_conditions(self, factor_id: str) -> List[Dict]:
        conditions_map = {
            "bull": [{"field": "regime", "op": "eq", "value": "bull"}],
            "bear": [{"field": "regime", "op": "eq", "value": "bear"}],
            "sideways": [{"field": "regime", "op": "eq", "value": "sideways"}],
            "atr_low": [{"field": "atr_percentile", "op": "lt", "value": 25}],
            "atr_high": [{"field": "atr_percentile", "op": "gt", "value": 75}],
            "trend12h_up": [{"field": "trend_12h_pct", "op": "gt", "value": 0}],
            "trend12h_strong": [{"field": "trend_12h_pct", "op": "gt", "value": 1}],
            "trend24h_up": [{"field": "trend_24h_pct", "op": "gt", "value": 0}],
            "trend24h_strong": [{"field": "trend_24h_pct", "op": "gt", "value": 1}],
            "bull_low_atr": [{"field": "regime", "op": "eq", "value": "bull"}, {"field": "atr_percentile", "op": "lt", "value": 50}],
            "bull_trend": [{"field": "regime", "op": "eq", "value": "bull"}, {"field": "trend_12h_pct", "op": "gt", "value": 0}],
            "lowatr_trend": [{"field": "atr_percentile", "op": "lt", "value": 25}, {"field": "trend_12h_pct", "op": "gt", "value": 0}],
            "bull_lowatr_trend": [{"field": "regime", "op": "eq", "value": "bull"}, {"field": "atr_percentile", "op": "lt", "value": 30}, {"field": "trend_12h_pct", "op": "gt", "value": 0}],
        }
        
        if factor_id in conditions_map:
            return conditions_map[factor_id]
        
        fid_lower = factor_id.lower()
        for key, conds in conditions_map.items():
            if key in fid_lower:
                return conds
        
        return []
    
    def evaluate_factor(self, factor: Factor, context) -> bool:
        for cond in factor.conditions:
            field = cond.get('field', '')
            op = cond.get('op', '')
            value = cond.get('value', 0)
            
            ctx_value = getattr(context, field, None)
            if ctx_value is None:
                return False
            
            try:
                if op == 'gt' and not (ctx_value > value):
                    return False
                elif op == 'lt' and not (ctx_value < value):
                    return False
                elif op == 'eq' and not (ctx_value == value):
                    return False
            except (TypeError, ValueError):
                return False
        
        return True
    
    def calculate_signal(self, context, current_price: float = 0) -> Dict:
        active_factors = []
        details = []
        
        # 取得市場指標
        trend_24h = getattr(context, 'trend_24h_pct', 0) or 0
        trend_12h = getattr(context, 'trend_12h_pct', 0) or 0
        atr_pct = getattr(context, 'atr_percentile', 50) or 50
        
        for factor in self.factors:
            is_active = self.evaluate_factor(factor, context)
            
            # 計算這個因子在當前市場的觸發進場價
            trigger_price = None
            if is_active and current_price > 0:
                trigger_price = self._calculate_trigger_price(factor, current_price, trend_24h, trend_12h)
            
            details.append({
                'name': factor.name,
                'factor_id': factor.factor_id,
                'win_rate': factor.win_rate,
                'weight': factor.weight,
                'sharpe': factor.sharpe,
                'active': is_active,
                'trigger_price': trigger_price
            })
            if is_active:
                active_factors.append(factor)
        
        if not active_factors:
            return {
                'signal': 'FLAT',
                'confidence': 0,
                'active_factors': [],
                'weighted_score': 0,
                'avg_entry_price': 0,
                'details': details
            }
        
        total_weight = sum(f.weight for f in active_factors)
        max_possible = sum(f.weight for f in self.factors[:5]) if self.factors else 1.0
        confidence = min(total_weight / max_possible, 1.0) if max_possible > 0 else 0
        signal = 'LONG' if confidence >= 0.2 else 'FLAT'
        
        # 計算加權平均進場價（基於當前市場觸發價格）
        total_entry_weight = 0
        for f in active_factors:
            for d in details:
                if d['factor_id'] == f.factor_id and d.get('trigger_price'):
                    total_entry_weight += f.weight * d['trigger_price']
                    break  # 只取第一個匹配的
        
        avg_entry_price = total_entry_weight / total_weight if total_weight > 0 else 0
        
        return {
            'signal': signal,
            'confidence': round(confidence, 3),
            'active_factors': [
                {'name': f.name, 'factor_id': f.factor_id, 'win_rate': f.win_rate, 'weight': f.weight, 'sharpe': f.sharpe, 'trigger_price': next((d['trigger_price'] for d in details if d['factor_id'] == f.factor_id), None)}
                for f in active_factors
            ],
            'weighted_score': round(total_weight / self.total_weight, 3) if self.total_weight > 0 else 0,
            'total_weight': round(total_weight, 3),
            'avg_entry_price': round(avg_entry_price, 4) if avg_entry_price > 0 else 0,
            'details': details
        }
    
    def _calculate_trigger_price(self, factor, current_price, trend_24h, trend_12h) -> float:
        """根據因子條件計算觸發進場價"""
        fid = factor.factor_id.lower()
        
        # 成交量策略：進場價基於趨勢調整
        if 'vol_' in fid:
            # 成交量策略，進場價接近現價但考慮趨勢
            if trend_24h != 0:
                return current_price / (1 + trend_24h / 100)
            return current_price
        
        # 根據因子類型計算進場價
        # 進場價 = 現價 / (1 + 趨勢%)
        if 'trend24h' in fid and trend_24h != 0:
            # 24H 趨勢進場價
            return current_price / (1 + trend_24h / 100)
        elif 'trend12h' in fid and trend_12h != 0:
            # 12H 趨勢進場價
            return current_price / (1 + trend_12h / 100)
        elif 'atr_low' in fid or 'lowatr' in fid:
            # 低 ATR 通常在低波動時進場，進場價接近現價
            return current_price * 0.995
        elif 'atr_high' in fid:
            # 高 ATR 在高波動時進場
            return current_price * 1.005
        elif 'bull' in fid:
            # 多頭市場進場
            return current_price * 0.99
        elif 'bear' in fid:
            # 空頭市場進場
            return current_price * 1.01
        else:
            # 默認使用 24H 趨勢
            if trend_24h != 0:
                return current_price / (1 + trend_24h / 100)
            return current_price
    
    def analyze_dataframe(self, df: pd.DataFrame, symbol: str = 'BTCUSDT') -> Dict:
        import pendulum
        from tradememory.data import OHLCVSeries, Timeframe, OHLCV
        from tradememory.data.context_builder import build_context, ContextConfig
        
        bars = []
        df_tail = df.tail(100)
        n = len(df_tail)
        
        for i in range(n):
            row = df_tail.iloc[i]
            raw_ts = row['open_time']
            
            if isinstance(raw_ts, bytes):
                raw_ts = raw_ts.decode('utf-8')
            if isinstance(raw_ts, str):
                ts = pendulum.from_timestamp(pd.Timestamp(raw_ts).timestamp())
            elif hasattr(raw_ts, 'timestamp'):
                ts = pendulum.from_timestamp(raw_ts.timestamp())
            else:
                ts = pendulum.now()
            
            bar = OHLCV(
                timestamp=ts,
                open=float(row['open']),
                high=float(row['high']),
                low=float(row['low']),
                close=float(row['close']),
                volume=float(row['volume']),
            )
            bars.append(bar)
        
        series = OHLCVSeries(symbol=symbol, timeframe=Timeframe.D1, bars=bars, source='dataframe')
        config = ContextConfig()
        context = build_context(series, bar_index=-1, config=config)
        
        # 取得當前價格（最新收盤價）
        current_price = context.price if hasattr(context, 'price') and context.price else (bars[-1].close if bars else 0)
        
        signal_result = self.calculate_signal(context, current_price)
        
        return {
            'symbol': symbol,
            'market_context': {
                'regime': str(context.regime.value) if context.regime else 'unknown',
                'atr_percentile': context.atr_percentile if hasattr(context, 'atr_percentile') else 50,
                'trend_12h_pct': context.trend_12h_pct if hasattr(context, 'trend_12h_pct') else 0,
                'trend_24h_pct': context.trend_24h_pct if hasattr(context, 'trend_24h_pct') else 0,
                'price': current_price,
            },
            'signal': signal_result
        }
