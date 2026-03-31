"""
多因子策略庫
每個因子只要滿足基本門檻就入庫
"""

from dataclasses import dataclass, field
from typing import Optional, Callable


@dataclass
class FactorRecord:
    """因子紀錄"""
    name: str
    category: str  # trend/reversal/volatility/volume
    params: dict
    metrics: dict  # win_rate, profit_factor, sharpe, max_drawdown
    regimes_performed: dict = field(default_factory=dict)  # 各市場狀態的表現
    
    # 門檻
    MIN_WIN_RATE: float = 0.0
    MIN_PROFIT_FACTOR: float = 0.0
    MAX_DRAWDOWN: float = 0.10
    
    # 信號產生器（可選）
    signal_generator: Optional[Callable] = None
    
    def is_valid(self) -> bool:
        """通過基本門檻"""
        return (
            self.metrics.get('win_rate', 0) >= self.MIN_WIN_RATE and
            self.metrics.get('profit_factor', 0) >= self.MIN_PROFIT_FACTOR and
            self.metrics.get('max_drawdown', 1) <= self.MAX_DRAWDOWN
        )
    
    def passes_regimes(self, regimes_passed: int) -> bool:
        """通過至少2個市場狀態"""
        return regimes_passed >= 2
    
    def generate_signal(self, market_data: dict) -> str:
        """產生交易信號"""
        if self.signal_generator is None:
            return 'FLAT'
        try:
            return self.signal_generator(market_data, self.params)
        except Exception:
            return 'FLAT'


class FactorLibrary:
    """因子庫管理器"""
    
    def __init__(self):
        self.factors: list[FactorRecord] = []
        self._history: list[dict] = []  # 記錄添加歷史
    
    def add(self, factor: FactorRecord, verbose: bool = True):
        """添加因子到庫"""
        if factor.is_valid():
            self.factors.append(factor)
            self._history.append({
                'name': factor.name,
                'action': 'added',
                'metrics': factor.metrics.copy()
            })
            if verbose:
                print(f"✅ 新因子入庫: {factor.name}")
        else:
            self._history.append({
                'name': factor.name,
                'action': 'rejected',
                'metrics': factor.metrics.copy()
            })
            if verbose:
                print(f"❌ 因子未通過門檻: {factor.name}")
    
    def remove(self, factor_name: str) -> bool:
        """從庫中移除因子"""
        for i, f in enumerate(self.factors):
            if f.name == factor_name:
                self.factors.pop(i)
                self._history.append({
                    'name': factor_name,
                    'action': 'removed',
                    'metrics': {}
                })
                return True
        return False
    
    def get_active_factors(self) -> list[FactorRecord]:
        """取得目前活躍的因子"""
        return [f for f in self.factors if f.metrics.get('active', True)]
    
    def get_by_category(self, category: str) -> list[FactorRecord]:
        """按類別取得因子"""
        return [f for f in self.factors if f.category == category]
    
    def get_top_by_metric(self, metric: str, n: int = 5) -> list[FactorRecord]:
        """按指標取得Top N因子"""
        sorted_factors = sorted(
            self.factors, 
            key=lambda f: f.metrics.get(metric, 0), 
            reverse=True
        )
        return sorted_factors[:n]
    
    def generate_signal(self, market_data: dict) -> dict:
        """
        多因子信號組合
        返回: {'signal': 'LONG/SHORT/FLAT', 'confidence': 0-1, 'factors': [...]}
        """
        active = self.get_active_factors()
        if not active:
            return {'signal': 'FLAT', 'confidence': 0, 'factors': []}
        
        long_votes = 0
        short_votes = 0
        factor_details = []
        
        for factor in active:
            signal = factor.generate_signal(market_data)
            factor_details.append({'name': factor.name, 'signal': signal})
            
            if signal == 'LONG':
                long_votes += 1
            elif signal == 'SHORT':
                short_votes += 1
        
        total = len(active)
        confidence = max(long_votes, short_votes) / total if total > 0 else 0
        
        if long_votes > short_votes and confidence >= 0.5:
            return {'signal': 'LONG', 'confidence': confidence, 'factors': factor_details}
        elif short_votes > long_votes and confidence >= 0.5:
            return {'signal': 'SHORT', 'confidence': confidence, 'factors': factor_details}
        
        return {'signal': 'FLAT', 'confidence': 0, 'factors': factor_details}
    
    def get_statistics(self) -> dict:
        """取得因子庫統計"""
        by_category = {}
        for f in self.factors:
            by_category[f.category] = by_category.get(f.category, 0) + 1
        
        return {
            'total_factors': len(self.factors),
            'active_factors': len(self.get_active_factors()),
            'by_category': by_category,
            'history_count': len(self._history),
            'top_sharpe': max(self.factors, key=lambda f: f.metrics.get('sharpe', 0)).name if self.factors else None,
            'top_win_rate': max(self.factors, key=lambda f: f.metrics.get('win_rate', 0)).name if self.factors else None,
        }
    
    def filter_factors(
        self,
        min_win_rate: Optional[float] = None,
        min_profit_factor: Optional[float] = None,
        max_drawdown: Optional[float] = None,
        category: Optional[str] = None
    ) -> list[FactorRecord]:
        """根據條件過濾因子"""
        filtered = self.factors.copy()
        
        if min_win_rate is not None:
            filtered = [f for f in filtered if f.metrics.get('win_rate', 0) >= min_win_rate]
        
        if min_profit_factor is not None:
            filtered = [f for f in filtered if f.metrics.get('profit_factor', 0) >= min_profit_factor]
        
        if max_drawdown is not None:
            filtered = [f for f in filtered if f.metrics.get('max_drawdown', 1) <= max_drawdown]
        
        if category is not None:
            filtered = [f for f in filtered if f.category == category]
        
        return filtered
