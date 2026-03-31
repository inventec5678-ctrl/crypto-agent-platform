"""
Walk-Forward 分析模組
位置: crypto-agent-platform/backtest/walk_forward_analysis.py

功能:
- 滑動窗口 Walk-Forward 分析
- 訓練窗口參數優化
- 測試窗口績效驗證
- 過擬合風險檢測

這種方法可以更好地評估策略在未見數據上的真實表現，
避免傳統回測的過擬合問題。
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from itertools import product
import warnings
import asyncio

from .backtest_engine import BacktestEngine, BaseStrategy, PositionSide
from .performance_metrics import PerformanceMetrics


@dataclass
class WalkForwardConfig:
    """Walk-Forward 分析配置"""
    total_data: int = 1000      # 總數據點
    train_window: int = 500     # 訓練窗口大小
    test_window: int = 100      # 測試窗口大小
    step: int = 50             # 滑動步長
    
    def __post_init__(self):
        """驗證配置參數"""
        if self.train_window + self.test_window > self.total_data:
            raise ValueError(
                f"訓練窗口({self.train_window}) + 測試窗口({self.test_window}) "
                f"不能超過總數據點({self.total_data})"
            )
        if self.step <= 0:
            raise ValueError(f"滑動步長必須為正數: {self.step}")
    
    def calculate_total_windows(self) -> int:
        """計算總視窗數量"""
        # 第一個視窗: [0, train_window) 訓練, [train_window, train_window+test_window) 測試
        # 後續視窗每次滑動 step 個數據點
        max_start = self.total_data - self.train_window - self.test_window
        if max_start < 0:
            return 0
        return (max_start // self.step) + 1


@dataclass
class WindowResult:
    """單一視窗的分析結果"""
    window_index: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int
    best_params: Dict[str, Any]
    train_sharpe: float
    test_sharpe: float
    train_win_rate: float
    test_win_rate: float
    train_total_return: float
    test_total_return: float
    overfitting_risk: bool = False
    overfitting_message: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """轉換為字典格式"""
        return {
            "window_index": self.window_index,
            "train_start": self.train_start,
            "train_end": self.train_end,
            "test_start": self.test_start,
            "test_end": self.test_end,
            "best_params": self.best_params,
            "train_sharpe": round(self.train_sharpe, 4),
            "test_sharpe": round(self.test_sharpe, 4),
            "train_win_rate": round(self.train_win_rate, 4),
            "test_win_rate": round(self.test_win_rate, 4),
            "train_total_return": round(self.train_total_return, 4),
            "test_total_return": round(self.test_total_return, 4),
            "overfitting_risk": self.overfitting_risk,
            "overfitting_message": self.overfitting_message
        }


@dataclass
class WalkForwardResult:
    """Walk-Forward 分析總結果"""
    total_windows: int
    results: List[WindowResult]
    avg_test_sharpe: float
    avg_test_win_rate: float
    consistency: float  # 參數穩定性 (0-1)
    avg_train_sharpe: float = 0.0
    avg_train_win_rate: float = 0.0
    overfitting_windows_count: int = 0
    overfitting_ratio: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """轉換為字典格式"""
        return {
            "total_windows": self.total_windows,
            "results": [r.to_dict() for r in self.results],
            "avg_test_sharpe": round(self.avg_test_sharpe, 4),
            "avg_test_win_rate": round(self.avg_test_win_rate, 4),
            "consistency": round(self.consistency, 4),
            "avg_train_sharpe": round(self.avg_train_sharpe, 4),
            "avg_train_win_rate": round(self.avg_train_win_rate, 4),
            "overfitting_windows_count": self.overfitting_windows_count,
            "overfitting_ratio": round(self.overfitting_ratio, 4)
        }
    
    def summary(self) -> str:
        """生成摘要報告"""
        lines = []
        lines.append("=" * 70)
        lines.append("                    Walk-Forward 分析報告")
        lines.append("=" * 70)
        lines.append("")
        lines.append(f"總視窗數: {self.total_windows}")
        lines.append(f"過擬合風險視窗數: {self.overfitting_windows_count}")
        lines.append("")
        lines.append("績效摘要:")
        lines.append(f"  平均訓練 Sharpe: {self.avg_train_sharpe:.4f}")
        lines.append(f"  平均測試 Sharpe: {self.avg_test_sharpe:.4f}")
        lines.append(f"  平均訓練勝率: {self.avg_train_win_rate:.2%}")
        lines.append(f"  平均測試勝率: {self.avg_test_win_rate:.2%}")
        lines.append(f"  參數穩定性 (Consistency): {self.consistency:.4f}")
        lines.append("")
        
        if self.overfitting_windows_count > 0:
            pct = self.overfitting_windows_count / self.total_windows * 100
            lines.append(f"⚠️ 警告: {pct:.1f}% 視窗存在過擬合風險")
        
        if self.consistency < 0.5:
            lines.append("⚠️ 警告: 參數穩定性低，策略可能過度擬合歷史數據")
        elif self.consistency > 0.7:
            lines.append("✅ 參數穩定性良好")
        
        lines.append("")
        lines.append("各視窗詳細結果:")
        lines.append("-" * 70)
        for r in self.results:
            status = "⚠️ 過擬合" if r.overfitting_risk else "✅ 正常"
            lines.append(
                f"視窗 {r.window_index}: "
                f"Train Sharpe={r.train_sharpe:.3f}, "
                f"Test Sharpe={r.test_sharpe:.3f} "
                f"[{status}]"
            )
            lines.append(f"  最佳參數: {r.best_params}")
        lines.append("")
        lines.append("=" * 70)
        return "\n".join(lines)


class WalkForwardAnalysisEngine:
    """
    Walk-Forward 分析引擎
    
    將歷史數據分為「訓練窗口」和「測試窗口」:
    1. 在訓練窗口使用網格搜索優化參數
    2. 在測試窗口驗證最佳參數的效果
    3. 滑動窗口逐步前進，重複上述過程
    
    這種方法可以:
    - 更好地評估策略的真實表現
    - 檢測參數的穩定性
    - 識別過擬合風險
    """
    
    def __init__(
        self,
        strategy_class: type,
        param_space: Dict[str, List[Any]],
        metric: str = "sharpe_ratio",
        overfitting_threshold: float = 0.7
    ):
        """
        初始化 Walk-Forward 分析引擎
        
        Args:
            strategy_class: 策略類別
            param_space: 參數空間，如 {'fast_period': [5, 10, 15], 'slow_period': [20, 30, 50]}
            metric: 優化目標指標
            overfitting_threshold: 過擬合閾值 (測試 < 訓練 * threshold 視為過擬合)
        """
        self.strategy_class = strategy_class
        self.param_space = param_space
        self.metric = metric
        self.overfitting_threshold = overfitting_threshold
    
    def _generate_param_combinations(self) -> List[Dict[str, Any]]:
        """生成所有參數組合"""
        keys = list(self.param_space.keys())
        values = list(self.param_space.values())
        
        combinations = []
        for combo in product(*values):
            combinations.append(dict(zip(keys, combo)))
        return combinations
    
    def _evaluate_on_data(
        self,
        df: pd.DataFrame,
        params: Dict[str, Any],
        config: WalkForwardConfig
    ) -> Tuple[float, float, float]:
        """
        在指定數據上評估策略
        
        Returns:
            (sharpe_ratio, win_rate, total_return)
        """
        try:
            # 建立策略
            strategy = self.strategy_class(**params)
            
            # 建立引擎
            engine = BacktestEngine()
            engine.load_dataframe("BACKTEST", df.copy())
            engine.set_strategy(strategy)
            
            # 執行回測
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(engine.run())
            finally:
                loop.close()
            
            sharpe = result.Sharpe_Ratio
            win_rate = result.Win_Rate / 100.0 if result.Win_Rate > 1 else result.Win_Rate
            total_return = result.total_return_pct / 100.0 if result.total_return_pct > 1 else result.total_return_pct
            
            return sharpe, win_rate, total_return
            
        except Exception as e:
            warnings.warn(f"參數 {params} 評估失敗: {e}")
            return 0.0, 0.0, 0.0
    
    def _grid_search_on_train_data(
        self,
        train_df: pd.DataFrame,
        config: WalkForwardConfig
    ) -> Tuple[Dict[str, Any], float, float, float]:
        """
        在訓練數據上進行網格搜索
        
        Returns:
            (best_params, best_sharpe, best_win_rate, best_return)
        """
        combinations = self._generate_param_combinations()
        
        best_params = None
        best_metric_value = -float('inf')
        best_sharpe = 0.0
        best_win_rate = 0.0
        best_return = 0.0
        
        for params in combinations:
            sharpe, win_rate, total_return = self._evaluate_on_data(
                train_df, params, config
            )
            
            # 根據 metric 選擇最佳參數
            if self.metric == "sharpe_ratio":
                metric_value = sharpe
            elif self.metric == "win_rate":
                metric_value = win_rate
            elif self.metric == "total_return":
                metric_value = total_return
            else:
                metric_value = sharpe
            
            if metric_value > best_metric_value:
                best_metric_value = metric_value
                best_params = params
                best_sharpe = sharpe
                best_win_rate = win_rate
                best_return = total_return
        
        return best_params, best_sharpe, best_win_rate, best_return
    
    def analyze(
        self,
        data: pd.DataFrame,
        config: Optional[WalkForwardConfig] = None,
        progress_callback: Optional[Callable] = None
    ) -> WalkForwardResult:
        """
        執行 Walk-Forward 分析
        
        Args:
            data: 市場數據 DataFrame
            config: Walk-Forward 配置，如果為 None 則使用預設配置
            progress_callback: 進度回調函數 (progress: float) -> None
            
        Returns:
            WalkForwardResult: 分析結果
        """
        # 處理默認配置
        if config is None:
            config = WalkForwardConfig(total_data=len(data))
        else:
            # 使用實際數據長度
            actual_data_len = len(data)
            if config.total_data > actual_data_len:
                warnings.warn(
                    f"配置的總數據點({config.total_data}) > 實際數據({actual_data_len})，"
                    f"將使用實際數據長度"
                )
                config.total_data = actual_data_len
        
        total_windows = config.calculate_total_windows()
        
        if total_windows == 0:
            raise ValueError(
                f"數據點不足，無法進行 Walk-Forward 分析。"
                f"需要至少 train_window + test_window = "
                f"{config.train_window + config.test_window} 個數據點"
            )
        
        print(f"🔄 開始 Walk-Forward 分析...")
        print(f"   總數據點: {config.total_data}")
        print(f"   訓練窗口: {config.train_window}")
        print(f"   測試窗口: {config.test_window}")
        print(f"   滑動步長: {config.step}")
        print(f"   總視窗數: {total_windows}")
        print("")
        
        window_results: List[WindowResult] = []
        test_sharpes = []
        test_win_rates = []
        train_sharpes = []
        train_win_rates = []
        overfitting_count = 0
        
        for i in range(total_windows):
            # 計算當前視窗的範圍
            train_start = i * config.step
            train_end = train_start + config.train_window
            test_start = train_end
            test_end = test_start + config.test_window
            
            # 確保不超出範圍
            if test_end > config.total_data:
                break
            
            print(f"📊 視窗 {i + 1}/{total_windows}")
            print(f"   訓練: [{train_start}:{train_end}]")
            print(f"   測試: [{test_start}:{test_end}]")
            
            # 分割數據
            train_df = data.iloc[train_start:train_end].reset_index(drop=True)
            test_df = data.iloc[test_start:test_end].reset_index(drop=True)
            
            # 在訓練數據上網格搜索
            best_params, train_sharpe, train_win_rate, train_return = \
                self._grid_search_on_train_data(train_df, config)
            
            print(f"   訓練期最佳參數: {best_params}")
            print(f"   訓練 Sharpe: {train_sharpe:.4f}, 勝率: {train_win_rate:.2%}")
            
            # 在測試數據上評估
            test_sharpe, test_win_rate, test_return = self._evaluate_on_data(
                test_df, best_params, config
            )
            
            print(f"   測試 Sharpe: {test_sharpe:.4f}, 勝率: {test_win_rate:.2%}")
            
            # 過擬合檢測
            overfitting_risk = False
            overfitting_message = ""
            
            if train_sharpe > 0 and test_sharpe < train_sharpe * self.overfitting_threshold:
                overfitting_risk = True
                overfitting_message = (
                    f"測試 Sharpe ({test_sharpe:.4f}) < "
                    f"訓練 Sharpe ({train_sharpe:.4f}) * {self.overfitting_threshold}"
                )
                overfitting_count += 1
                print(f"   ⚠️ 過擬合風險: {overfitting_message}")
            
            # 記錄結果
            window_result = WindowResult(
                window_index=i + 1,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                best_params=best_params,
                train_sharpe=train_sharpe,
                test_sharpe=test_sharpe,
                train_win_rate=train_win_rate,
                test_win_rate=test_win_rate,
                train_total_return=train_return,
                test_total_return=test_return,
                overfitting_risk=overfitting_risk,
                overfitting_message=overfitting_message
            )
            window_results.append(window_result)
            
            # 收集指標
            test_sharpes.append(test_sharpe)
            test_win_rates.append(test_win_rate)
            train_sharpes.append(train_sharpe)
            train_win_rates.append(train_win_rate)
            
            if progress_callback:
                progress_callback((i + 1) / total_windows)
            
            print("")
        
        # 計算總結指標
        avg_test_sharpe = np.mean(test_sharpes) if test_sharpes else 0.0
        avg_test_win_rate = np.mean(test_win_rates) if test_win_rates else 0.0
        avg_train_sharpe = np.mean(train_sharpes) if train_sharpes else 0.0
        avg_train_win_rate = np.mean(train_win_rates) if train_win_rates else 0.0
        
        # 計算參數穩定性 (consistency)
        consistency = self._calculate_consistency(window_results)
        
        # 計算過擬合比率
        overfitting_ratio = overfitting_count / total_windows if total_windows > 0 else 0.0
        
        result = WalkForwardResult(
            total_windows=total_windows,
            results=window_results,
            avg_test_sharpe=avg_test_sharpe,
            avg_test_win_rate=avg_test_win_rate,
            consistency=consistency,
            avg_train_sharpe=avg_train_sharpe,
            avg_train_win_rate=avg_train_win_rate,
            overfitting_windows_count=overfitting_count,
            overfitting_ratio=overfitting_ratio
        )
        
        return result
    
    def _calculate_consistency(self, window_results: List[WindowResult]) -> float:
        """
        計算參數穩定性 (Consistency)
        
        評估各視窗最佳參數的一致性:
        - 1.0 = 完全穩定（所有視窗參數相同）
        - 0.0 = 完全不穩定（所有視窗參數都不同）
        """
        if len(window_results) < 2:
            return 1.0
        
        # 收集所有參數名稱
        all_param_names = set()
        for r in window_results:
            all_param_names.update(r.best_params.keys())
        
        # 對於每個參數，計算其在視窗間的變異係數
        param_stabilities = []
        
        for param_name in all_param_names:
            values = []
            for r in window_results:
                if param_name in r.best_params:
                    v = r.best_params[param_name]
                    if isinstance(v, (int, float)):
                        values.append(v)
            
            if len(values) >= 2:
                mean = np.mean(values)
                std = np.std(values)
                # 計算變異係數 (CV)
                cv = std / abs(mean) if mean != 0 else 0
                # 將 CV 轉換為穩定性分數 (1 = 完全穩定, CV=0)
                # 使用 1/(1+cv) 映射: CV=0 -> 1, CV=1 -> 0.5, CV=∞ -> 0
                stability = 1.0 / (1.0 + cv)
                param_stabilities.append(stability)
        
        if not param_stabilities:
            return 1.0
        
        # 返回所有參數的平均穩定性
        return np.mean(param_stabilities)


# 便捷函數
def run_walk_forward_analysis(
    strategy_class: type,
    data: pd.DataFrame,
    param_space: Dict[str, List[Any]],
    config: Optional[WalkForwardConfig] = None,
    metric: str = "sharpe_ratio",
    overfitting_threshold: float = 0.7,
    verbose: bool = True
) -> WalkForwardResult:
    """
    便捷的 Walk-Forward 分析函數
    
    Args:
        strategy_class: 策略類別
        data: 市場數據 DataFrame
        param_space: 參數空間
        config: Walk-Forward 配置
        metric: 優化目標指標
        overfitting_threshold: 過擬合閾值
        verbose: 是否輸出詳細報告
        
    Returns:
        WalkForwardResult: 分析結果
    """
    engine = WalkForwardAnalysisEngine(
        strategy_class=strategy_class,
        param_space=param_space,
        metric=metric,
        overfitting_threshold=overfitting_threshold
    )
    
    result = engine.analyze(
        data=data,
        config=config,
        progress_callback=None
    )
    
    if verbose:
        print(result.summary())
    
    return result


if __name__ == "__main__":
    # 測試範例
    from backtest_engine import SimpleMovingAverageCrossover
    
    # 模擬數據
    print("生成模擬數據...")
    n = 2000
    dates = pd.date_range('2024-01-01', periods=n, freq='1h')
    np.random.seed(42)
    
    # 模擬價格走勢
    returns = np.random.randn(n) * 0.02
    price = 50000 * np.exp(np.cumsum(returns))
    
    df = pd.DataFrame({
        'open_time': dates,
        'open': price,
        'high': price * 1.01,
        'low': price * 0.99,
        'close': price,
        'volume': np.random.randint(100, 1000, n)
    })
    
    # Walk-Forward 分析配置
    config = WalkForwardConfig(
        total_data=1500,
        train_window=500,
        test_window=100,
        step=50
    )
    
    # 參數空間
    param_space = {
        'fast_period': [5, 10, 15, 20],
        'slow_period': [30, 50, 70]
    }
    
    print("\n執行 Walk-Forward 分析...")
    print("-" * 50)
    
    result = run_walk_forward_analysis(
        strategy_class=SimpleMovingAverageCrossover,
        data=df,
        param_space=param_space,
        config=config,
        metric="sharpe_ratio",
        verbose=True
    )
    
    # 輸出 JSON 格式結果
    print("\n\nJSON 格式結果:")
    import json
    print(json.dumps(result.to_dict(), indent=2))
