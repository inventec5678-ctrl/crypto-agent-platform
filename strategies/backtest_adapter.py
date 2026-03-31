"""
策略回測適配器

為 live 策略添加 backtest 模式支援
使用方式:
    strategy = MACrossoverStrategy(fast_period=10, slow_period=30)
    strategy.enable_backtest_mode(df)  # df from backtest engine
    # 之後 analyze() 會使用提供的 df 而非網路請求
"""

from typing import Optional, Dict
import pandas as pd


class BacktestModeMixin:
    """
    混入類：為策略添加回測模式
    
    策略繼承此 mixin 後可用:
        strategy.enable_backtest_mode(df)
        signal = await strategy.analyze()  # 使用本地 df
    """
    
    _backtest_df: Optional[pd.DataFrame] = None
    _backtest_enabled: bool = False
    _original_analyze: Optional[callable] = None
    
    def enable_backtest_mode(self, df: pd.DataFrame):
        """
        啟用回測模式
        
        Args:
            df: 歷史 K線 DataFrame（需包含 open/high/low/close/volume 列）
        """
        self._backtest_df = df.copy()
        self._backtest_enabled = True
        
        # 保存原來的 klines source（binance_client）
        # 之後 analyze() 會被 patch 為直接用 self._backtest_df
        self._patch_analyze()
    
    def disable_backtest_mode(self):
        """關閉回測模式"""
        if self._original_analyze:
            # 恢復原來的 analyze
            self.__class__.analyze = self._original_analyze
            self._original_analyze = None
        self._backtest_df = None
        self._backtest_enabled = False
    
    def _patch_analyze(self):
        """Patch analyze() 使用本地數據"""
        # 避免重複 patch
        if self._original_analyze is not None:
            return
        
        original = self.__class__.analyze
        self._original_analyze = original
        
        strat = self  # closure reference
        
        async def patched_analyze(self=None):
            """修補後的 analyze"""
            if self is None:
                self = strat
            
            if not getattr(self, '_backtest_enabled', False):
                # 非回測模式，調用原方法
                return await original(self)
            
            # 回測模式：直接使用本地數據
            return await self._backtest_analyze()
        
        self.__class__.analyze = patched_analyze
    
    async def _backtest_analyze(self) -> Optional[Dict]:
        """
        回測模式分析 - 子類覆寫
        
        預設行為：直接調用原有邏輯，但用 _backtest_df 替換 API 獲取
        """
        raise NotImplementedError("子類必須實作 _backtest_analyze()")


# 便捷函數：為策略啟用回測模式
def enable_backtest_mode(strategy, df: pd.DataFrame):
    """為任意策略啟用回測模式"""
    if isinstance(strategy, BacktestModeMixin):
        strategy.enable_backtest_mode(df)
    else:
        # 動態注入
        strategy._backtest_df = df.copy()
        strategy._backtest_enabled = True
        
        # Patch the strategy's analyze
        original = strategy.__class__.analyze
        strategy._original_analyze = original
        
        async def patched(self):
            if not getattr(self, '_backtest_enabled', False):
                return await original(self)
            return await self._backtest_analyze()
        
        strategy.__class__.analyze = patched
