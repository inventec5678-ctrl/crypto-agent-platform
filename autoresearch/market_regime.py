"""
市場分層模組
- 識別牛市/熊市/盤整
- 分層抽樣確保各市場狀態都有樣本
"""

import pandas as pd
import numpy as np
from typing import Optional


class MarketRegimeClassifier:
    """市場狀態分類器"""
    
    BULL = 'bull'
    BEAR = 'bear'
    SIDEWAYS = 'sideways'
    
    def __init__(self, ma_period: int = 200):
        self.ma_period = ma_period
    
    def classify(self, df: pd.DataFrame) -> pd.Series:
        """
        對每根K線分類市場狀態
        - 牛市：MA向上 + 價格 > MA
        - 熊市：MA向下 + 價格 < MA
        - 盤整：其他情況
        """
        df = df.copy()
        df['ma'] = df['close'].rolling(self.ma_period).mean()
        
        uptrend = df['ma'] > df['ma'].shift(1)
        price_above_ma = df['close'] > df['ma']
        downtrend = df['ma'] < df['ma'].shift(1)
        price_below_ma = df['close'] < df['ma']
        
        conditions = [
            (uptrend & price_above_ma, self.BULL),
            (downtrend & price_below_ma, self.BEAR),
        ]
        
        regime = pd.Series(self.SIDEWAYS, index=df.index)
        for condition, value in conditions:
            regime[condition] = value
        
        return regime
    
    def get_stratified_sample(
        self, 
        df: pd.DataFrame,
        train_months: int = 18,
        test_months: int = 3,
        n_samples: int = 10,
        random_seed: Optional[int] = None
    ) -> list:
        """
        分層抽樣
        每次隨機挑選訓練和測試區間
        """
        if random_seed is not None:
            np.random.seed(random_seed)
        
        regime = self.classify(df)
        
        # 按月分組
        df = df.copy()
        df['regime'] = regime
        df['year_month'] = df['open_time'].dt.to_period('M')
        
        # 分層
        bull_months = df[df['regime'] == self.BULL]['year_month'].unique()
        bear_months = df[df['regime'] == self.BEAR]['year_month'].unique()
        sideways_months = df[df['regime'] == self.SIDEWAYS]['year_month'].unique()
        
        samples = []
        for i in range(n_samples):
            # 隨機選擇訓練月份（分層抽樣）
            n_train = min(train_months, len(bull_months) + len(bear_months) + len(sideways_months))
            
            train_bull_size = min(3, len(bull_months))
            train_bear_size = min(2, len(bear_months))
            train_sideways_size = min(1, len(sideways_months))
            
            train_bull = np.random.choice(bull_months, size=train_bull_size, replace=False) if len(bull_months) > 0 else np.array([])
            train_bear = np.random.choice(bear_months, size=train_bear_size, replace=False) if len(bear_months) > 0 else np.array([])
            train_sideways = np.random.choice(sideways_months, size=train_sideways_size, replace=False) if len(sideways_months) > 0 else np.array([])
            
            train_months_selected = list(train_bull) + list(train_bear) + list(train_sideways)
            
            # 選擇測試月份
            all_months = set(bull_months) | set(bear_months) | set(sideways_months)
            remaining = all_months - set(train_months_selected)
            remaining_list = list(remaining)
            
            if len(remaining_list) == 0:
                continue
                
            test_months_selected = list(np.random.choice(
                remaining_list, 
                size=min(test_months, len(remaining_list)), 
                replace=False
            ))
            
            samples.append({
                'train': train_months_selected,
                'test': test_months_selected,
                'sample_id': i
            })
        
        return samples
    
    def get_regime_stats(self, regime: pd.Series) -> dict:
        """取得市場分層統計"""
        return {
            'bull_count': int(sum(regime == self.BULL)),
            'bear_count': int(sum(regime == self.BEAR)),
            'sideways_count': int(sum(regime == self.SIDEWAYS)),
            'bull_pct': float(sum(regime == self.BULL) / len(regime) * 100),
            'bear_pct': float(sum(regime == self.BEAR) / len(regime) * 100),
            'sideways_pct': float(sum(regime == self.SIDEWAYS) / len(regime) * 100),
        }
