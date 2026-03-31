"""
策略參數優化模組
位置: crypto-agent-platform/backtest/optimization.py

功能:
- 網格搜索（Grid Search）
- Walk-Forward 分析
- 過擬合檢測
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Callable, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from itertools import product
import concurrent.futures
import warnings

from .backtest_engine import BacktestEngine, BaseStrategy, PositionSide
from .performance_metrics import PerformanceMetrics


@dataclass
class OptimizationResult:
    """優化結果"""
    best_params: Dict[str, Any]
    best_metrics: Dict[str, float]
    all_results: List[Dict[str, Any]]
    overfitting_score: float = 0.0
    overfitting_warning: str = ""
    
    def summary(self) -> str:
        """生成摘要"""
        lines = []
        lines.append("=" * 60)
        lines.append("           策略參數優化結果")
        lines.append("=" * 60)
        lines.append("")
        lines.append(f"最佳參數:")
        for k, v in self.best_params.items():
            lines.append(f"  {k}: {v}")
        lines.append("")
        lines.append(f"最佳指標:")
        for k, v in self.best_metrics.items():
            if isinstance(v, float):
                lines.append(f"  {k}: {v:.4f}")
            else:
                lines.append(f"  {k}: {v}")
        lines.append("")
        if self.overfitting_warning:
            lines.append(f"⚠️ 過擬合警告: {self.overfitting_warning}")
        lines.append("")
        lines.append("=" * 60)
        return "\n".join(lines)


class GridSearchOptimizer:
    """
    網格搜索參數優化器
    
    遍歷所有參數組合，找出最佳參數
    """
    
    def __init__(
        self,
        strategy_class: type,
        data: Dict[str, pd.DataFrame],
        param_space: Dict[str, List[Any]],
        metric: str = "sharpe_ratio",
        maximize: bool = True,
        n_jobs: int = 1
    ):
        """
        初始化網格搜索
        
        Args:
            strategy_class: 策略類別
            data: 市場數據
            param_space: 參數空間，如 {'fast_period': [5, 10], 'slow_period': [20, 30]}
            metric: 優化目標指標
            maximize: 是否最大化該指標
            n_jobs: 並行任務數
        """
        self.strategy_class = strategy_class
        self.data = data
        self.param_space = param_space
        self.metric = metric
        self.maximize = maximize
        self.n_jobs = n_jobs
        
        # 生成參數組合
        self.param_combinations = self._generate_combinations()
    
    def _generate_combinations(self) -> List[Dict[str, Any]]:
        """生成所有參數組合"""
        keys = list(self.param_space.keys())
        values = list(self.param_space.values())
        
        combinations = []
        for combo in product(*values):
            param_dict = dict(zip(keys, combo))
            combinations.append(param_dict)
        
        return combinations
    
    def _evaluate_params(self, params: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, float]]:
        """評估單一參數組合"""
        try:
            # 建立策略實例
            strategy = self.strategy_class(**params)
            
            # 建立回測引擎
            engine = BacktestEngine()
            
            # 載入數據
            for symbol, df in self.data.items():
                engine.load_dataframe(symbol, df)
            
            # 設定策略
            engine.set_strategy(strategy)
            
            # 執行回測（同步版本）
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(engine.run())
            finally:
                loop.close()
            
            # 提取指標
            metrics = {
                'total_return': result.total_return_pct,
                'sharpe_ratio': result.Sharpe_Ratio,
                'max_drawdown': result.Max_Drawdown_Pct,
                'win_rate': result.Win_Rate,
                'profit_factor': result.Profit_Factor,
                'total_trades': result.Total_Trades,
            }
            
            return params, metrics
            
        except Exception as e:
            warnings.warn(f"參數組合 {params} 評估失敗: {e}")
            return params, {'error': str(e)}
    
    def optimize(self, progress_callback: Optional[Callable] = None) -> OptimizationResult:
        """
        執行網格搜索
        
        Args:
            progress_callback: 進度回調函數
            
        Returns:
            OptimizationResult: 優化結果
        """
        total = len(self.param_combinations)
        all_results = []
        
        print(f"🔍 開始網格搜索，共 {total} 種參數組合...")
        
        if self.n_jobs == 1:
            # 順序執行
            for i, params in enumerate(self.param_combinations):
                _, metrics = self._evaluate_params(params)
                result = {**params, **metrics}
                all_results.append(result)
                
                if progress_callback:
                    progress_callback((i + 1) / total)
                elif (i + 1) % 10 == 0:
                    print(f"   進度: {i + 1}/{total}")
        else:
            # 並行執行
            with concurrent.futures.ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
                futures = [
                    executor.submit(self._evaluate_params, params) 
                    for params in self.param_combinations
                ]
                
                for i, future in enumerate(concurrent.futures.as_completed(futures)):
                    _, metrics = future.result()
                    result = {**self.param_combinations[i], **metrics}
                    all_results.append(result)
                    
                    if progress_callback:
                        progress_callback((i + 1) / total)
                    elif (i + 1) % 10 == 0:
                        print(f"   進度: {i + 1}/{total}")
        
        # 找出最佳參數
        valid_results = [r for r in all_results if 'error' not in r]
        
        if not valid_results:
            raise ValueError("所有參數組合評估失敗")
        
        if self.maximize:
            best = max(valid_results, key=lambda x: x.get(self.metric, -float('inf')))
        else:
            best = min(valid_results, key=lambda x: x.get(self.metric, float('inf')))
        
        best_params = {k: v for k, v in best.items() if k in self.param_space}
        best_metrics = {k: v for k, v in best.items() if k not in self.param_space}
        
        # 過擬合檢測
        overfitting_score, warning = self._check_overfitting(all_results)
        
        return OptimizationResult(
            best_params=best_params,
            best_metrics=best_metrics,
            all_results=all_results,
            overfitting_score=overfitting_score,
            overfitting_warning=warning
        )
    
    def _check_overfitting(self, results: List[Dict]) -> Tuple[float, str]:
        """檢測過擬合"""
        if len(results) < 2:
            return 0.0, ""
        
        metric_values = [r.get(self.metric, 0) for r in results if 'error' not in r]
        
        if not metric_values:
            return 0.0, ""
        
        # 計算指標分佈
        std = np.std(metric_values)
        mean = np.mean(metric_values)
        cv = std / abs(mean) if mean != 0 else 0  # 變異係數
        
        # 過擬合指標
        # 如果變異係數太高，說明結果對參數很敏感，可能過擬合
        if cv > 1.0:
            return cv, f"參數敏感性高 (CV={cv:.2f})，可能有過擬合風險"
        elif cv > 0.5:
            return cv, f"建議進行 Walk-Forward 分析驗證"
        
        return cv, ""


class WalkForwardAnalyzer:
    """
    Walk-Forward 分析
    
    將歷史數據分為多個視窗，在每個視窗中:
    1. 使用前期數據優化參數（訓練期）
    2. 使用最佳參數在後續數據上測試（測試期）
    
    這種方法可以更好地評估策略的真實表現
    """
    
    def __init__(
        self,
        strategy_class: type,
        data: Dict[str, pd.DataFrame],
        param_space: Dict[str, List[Any]],
        metric: str = "sharpe_ratio",
        n_windows: int = 5,
        train_ratio: float = 0.7,
        step_size: Optional[int] = None
    ):
        """
        初始化 Walk-Forward 分析
        
        Args:
            strategy_class: 策略類別
            data: 市場數據
            param_space: 參數空間
            metric: 優化目標指標
            n_windows: 分析視窗數量
            train_ratio: 訓練期佔比
            step_size: 滑動步長（預設為測試期長度）
        """
        self.strategy_class = strategy_class
        self.data = data
        self.param_space = param_space
        self.metric = metric
        self.n_windows = n_windows
        self.train_ratio = train_ratio
        self.step_size = step_size
    
    def _split_dataframe(self, df: pd.DataFrame) -> Tuple[List[pd.DataFrame], List[pd.DataFrame]]:
        """
        分割數據為訓練期和測試期
        
        Returns:
            (train_periods, test_periods): 訓練期和測試期列表
        """
        n_rows = len(df)
        
        # 計算視窗大小
        if self.step_size is None:
            # 視窗不重疊
            window_size = n_rows // self.n_windows
            train_size = int(window_size * self.train_ratio)
            test_size = window_size - train_size
        else:
            test_size = self.step_size
            train_size = int(test_size * self.train_ratio / (1 - self.train_ratio))
        
        train_periods = []
        test_periods = []
        
        for i in range(self.n_windows):
            if self.step_size is None:
                # 非滑動視窗
                start_idx = i * window_size
                train_end = start_idx + train_size
                test_end = start_idx + window_size
            else:
                # 滑動視窗
                start_idx = i * test_size
                train_end = start_idx + train_size
                test_end = start_idx + train_size + test_size
            
            if test_end > n_rows:
                break
            
            train_periods.append(df.iloc[start_idx:train_end])
            test_periods.append(df.iloc[train_end:test_end])
        
        return train_periods, test_periods
    
    def analyze(self, progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        執行 Walk-Forward 分析
        
        Returns:
            Dict: 分析結果，包含每個視窗的訓練/測試結果
        """
        print(f"🔄 開始 Walk-Forward 分析 ({self.n_windows} 個視窗)...")
        
        # 取得第一個 symbol 的數據進行分割
        symbol = list(self.data.keys())[0]
        full_df = self.data[symbol]
        
        train_periods, test_periods = self._split_dataframe(full_df)
        
        results = {
            'windows': [],
            'train_metrics': [],
            'test_metrics': [],
            'best_params_per_window': [],
            'consistency_score': 0.0,
            'average_train_metric': 0.0,
            'average_test_metric': 0.0,
            'overfitting_ratio': 0.0
        }
        
        for i, (train_df, test_df) in enumerate(zip(train_periods, test_periods)):
            print(f"\n📊 視窗 {i + 1}/{len(train_periods)}")
            print(f"   訓練期: {train_df['open_time'].min()} ~ {train_df['open_time'].max()}")
            print(f"   測試期: {test_df['open_time'].min()} ~ {test_df['open_time'].max()}")
            
            # 準備訓練數據
            train_data = {symbol: train_df}
            
            # 網格搜索找出最佳參數
            optimizer = GridSearchOptimizer(
                self.strategy_class,
                train_data,
                self.param_space,
                self.metric,
                maximize=True,
                n_jobs=1
            )
            
            train_result = optimizer.optimize()
            best_params = train_result.best_params
            
            print(f"   訓練期最佳參數: {best_params}")
            print(f"   訓練期 {self.metric}: {train_result.best_metrics.get(self.metric, 0):.4f}")
            
            # 在測試期評估
            test_data = {symbol: test_df}
            strategy = self.strategy_class(**best_params)
            engine = BacktestEngine()
            engine.load_dataframe(symbol, test_df)
            engine.set_strategy(strategy)
            
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                test_result = loop.run_until_complete(engine.run())
            finally:
                loop.close()
            
            test_metric = getattr(test_result, self.metric.replace('_', '_'), 0)
            
            print(f"   測試期 {self.metric}: {test_metric:.4f}")
            
            results['windows'].append({
                'train_start': train_df['open_time'].min(),
                'train_end': train_df['open_time'].max(),
                'test_start': test_df['open_time'].min(),
                'test_end': test_df['open_time'].max()
            })
            results['train_metrics'].append(train_result.best_metrics)
            results['test_metrics'].append({
                self.metric: test_metric,
                'total_return': test_result.total_return_pct,
                'sharpe_ratio': test_result.Sharpe_Ratio,
                'max_drawdown': test_result.Max_Drawdown_Pct,
                'win_rate': test_result.Win_Rate
            })
            results['best_params_per_window'].append(best_params)
            
            if progress_callback:
                progress_callback((i + 1) / len(train_periods))
        
        # 計算總結指標
        test_metric_values = [m[self.metric] for m in results['test_metrics']]
        train_metric_values = [m.get(self.metric, 0) for m in results['train_metrics']]
        
        results['average_train_metric'] = np.mean(train_metric_values)
        results['average_test_metric'] = np.mean(test_metric_values)
        
        # 過擬合比率 = (訓練期表現 - 測試期表現) / 訓練期表現
        if results['average_train_metric'] != 0:
            results['overfitting_ratio'] = (
                (results['average_train_metric'] - results['average_test_metric']) 
                / abs(results['average_train_metric'])
            )
        
        # 一致性分數：測試期表現的穩定性
        results['consistency_score'] = 1.0 / (1.0 + np.std(test_metric_values)) if len(test_metric_values) > 1 else 1.0
        
        # 參數穩定性：最佳參數在視窗間的一致性
        results['param_stability'] = self._calculate_param_stability(
            results['best_params_per_window']
        )
        
        print("\n" + "=" * 60)
        print("           Walk-Forward 分析結果")
        print("=" * 60)
        print(f"平均訓練期 {self.metric}: {results['average_train_metric']:.4f}")
        print(f"平均測試期 {self.metric}: {results['average_test_metric']:.4f}")
        print(f"過擬合比率: {results['overfitting_ratio']*100:.1f}%")
        print(f"一致性分數: {results['consistency_score']:.4f}")
        print(f"參數穩定性: {results['param_stability']:.4f}")
        
        if results['overfitting_ratio'] > 0.3:
            print("⚠️ 警告: 過擬合比率較高，建議調整策略")
        elif results['overfitting_ratio'] > 0.1:
            print("⚡ 注意: 存在一定程度過擬合，可接受範圍")
        else:
            print("✅ 策略表現穩定")
        
        print("=" * 60)
        
        return results
    
    def _calculate_param_stability(self, all_params: List[Dict]) -> float:
        """計算參數穩定性"""
        if len(all_params) < 2:
            return 1.0
        
        # 對於數值型參數，計算標準化後的變異係數
        numeric_params = {}
        for params in all_params:
            for k, v in params.items():
                if isinstance(v, (int, float)):
                    if k not in numeric_params:
                        numeric_params[k] = []
                    numeric_params[k].append(v)
        
        if not numeric_params:
            return 1.0
        
        # 計算每個參數的變異係數
        cvs = []
        for values in numeric_params.values():
            mean = np.mean(values)
            std = np.std(values)
            cv = std / mean if mean != 0 else 0
            cvs.append(cv)
        
        # 平均 CV，越低越穩定
        avg_cv = np.mean(cvs)
        
        # 轉換為穩定性分數 (0-1, 1為最穩定)
        stability = 1.0 / (1.0 + avg_cv)
        
        return stability


class OverfittingDetector:
    """
    過擬合檢測器
    
    提供多種方法檢測策略是否過擬合
    """
    
    @staticmethod
    def detect_by_split(
        strategy_class: type,
        data: Dict[str, pd.DataFrame],
        param_space: Dict[str, List[Any]],
        test_ratio: float = 0.3
    ) -> Dict[str, Any]:
        """
        通過訓練/測試分割檢測過擬合
        
        在訓練數據上優化參數，然後在測試數據上評估
        如果差異過大，說明可能過擬合
        """
        symbol = list(data.keys())[0]
        df = data[symbol]
        
        # 分割數據
        split_idx = int(len(df) * (1 - test_ratio))
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]
        
        train_data = {symbol: train_df}
        test_data = {symbol: test_df}
        
        # 訓練
        optimizer = GridSearchOptimizer(
            strategy_class,
            train_data,
            param_space,
            n_jobs=1
        )
        train_result = optimizer.optimize()
        
        # 測試
        best_params = train_result.best_params
        strategy = strategy_class(**best_params)
        engine = BacktestEngine()
        engine.load_dataframe(symbol, test_df)
        engine.set_strategy(strategy)
        
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            test_result = loop.run_until_complete(engine.run())
        finally:
            loop.close()
        
        # 計算過擬合指標
        train_sharpe = train_result.best_metrics.get('sharpe_ratio', 0)
        test_sharpe = test_result.Sharpe_Ratio
        
        overfitting_ratio = (
            (train_sharpe - test_sharpe) / train_sharpe 
            if train_sharpe > 0 else 0
        )
        
        return {
            'train_sharpe': train_sharpe,
            'test_sharpe': test_sharpe,
            'overfitting_ratio': overfitting_ratio,
            'best_params': best_params,
            'is_overfitting': overfitting_ratio > 0.3,
            'recommendation': (
                '策略可能過擬合，建議簡化策略或增加訓練數據'
                if overfitting_ratio > 0.3 else
                '策略表現穩定'
            )
        }
    
    @staticmethod
    def detect_by_montecarlo(
        strategy_class: type,
        data: Dict[str, pd.DataFrame],
        params: Dict[str, Any],
        n_simulations: int = 100,
        sample_size: float = 0.8
    ) -> Dict[str, Any]:
        """
        通過蒙特卡洛模擬檢測過擬合
        
        對數據進行多次隨機抽樣，評估策略表現的穩定性
        如果標準差過大，說明可能過擬合
        """
        symbol = list(data.keys())[0]
        df = data[symbol]
        n = len(df)
        sample_len = int(n * sample_size)
        
        sharpe_values = []
        
        for i in range(n_simulations):
            # 隨機選擇起始點
            start_idx = np.random.randint(0, n - sample_len)
            sampled_df = df.iloc[start_idx:start_idx + sample_len].reset_index(drop=True)
            
            # 評估
            strategy = strategy_class(**params)
            engine = BacktestEngine()
            engine.load_dataframe(symbol, sampled_df)
            engine.set_strategy(strategy)
            
            try:
                import asyncio
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                result = loop.run_until_complete(engine.run())
                loop.close()
                
                sharpe_values.append(result.Sharpe_Ratio)
            except:
                continue
        
        if not sharpe_values:
            return {'error': '模擬全部失敗'}
        
        sharpe_std = np.std(sharpe_values)
        sharpe_mean = np.mean(sharpe_values)
        cv = sharpe_std / abs(sharpe_mean) if sharpe_mean != 0 else float('inf')
        
        return {
            'mean_sharpe': sharpe_mean,
            'std_sharpe': sharpe_std,
            'cv': cv,
            'min_sharpe': np.min(sharpe_values),
            'max_sharpe': np.max(sharpe_values),
            'is_overfitting': cv > 0.5,
            'recommendation': (
                '策略對數據樣本敏感，可能過擬合'
                if cv > 0.5 else
                '策略表現穩定'
            )
        }


# 便捷函數
def optimize_strategy(
    strategy_class: type,
    data: Dict[str, pd.DataFrame],
    param_space: Dict[str, List[Any]],
    method: str = "grid_search",
    **kwargs
) -> OptimizationResult:
    """
    便捷的策略優化函數
    
    Args:
        strategy_class: 策略類別
        data: 市場數據
        param_space: 參數空間
        method: 優化方法 ("grid_search" | "walk_forward")
    """
    if method == "grid_search":
        optimizer = GridSearchOptimizer(strategy_class, data, param_space, **kwargs)
        return optimizer.optimize()
    elif method == "walk_forward":
        analyzer = WalkForwardAnalyzer(strategy_class, data, param_space, **kwargs)
        result = analyzer.analyze()
        return result
    else:
        raise ValueError(f"未知優化方法: {method}")


if __name__ == "__main__":
    # 測試範例
    from backtest_engine import SimpleMovingAverageCrossover, BacktestEngine
    
    # 模擬數據
    dates = pd.date_range('2024-01-01', periods=500, freq='h')
    np.random.seed(42)
    price = 50000 + np.cumsum(np.random.randn(500) * 100)
    
    df = pd.DataFrame({
        'open_time': dates,
        'open': price,
        'high': price * 1.01,
        'low': price * 0.99,
        'close': price,
        'volume': np.random.randint(100, 1000, 500)
    })
    
    data = {'BTCUSDT': df}
    
    # 網格搜索
    param_space = {
        'fast_period': [5, 10, 15],
        'slow_period': [20, 30, 50]
    }
    
    print("測試網格搜索...")
    optimizer = GridSearchOptimizer(
        SimpleMovingAverageCrossover,
        data,
        param_space,
        n_jobs=1
    )
    
    result = optimizer.optimize()
    print(result.summary())
