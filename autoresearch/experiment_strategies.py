"""
實驗策略模組 - 參數探索策略

提供三種策略:
1. MutationStrategy  - 隨機扰動（推薦，永不重複）
2. GridSearchStrategy - 網格搜索（全面但可能重複）
3. BayesianOptimizer  - 貝葉斯優化（高效但需要額外套件）
"""

import random
import itertools
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
import numpy as np

from .models import ExperimentResult, StrategyParamSpec


@dataclass
class MutationConfig:
    """突變配置"""
    mutation_ratio: float = 0.1      # 每個參數扰動比例 (10%)
    mutation_probability: float = 0.8  # 每個參數被突變的機率
    min_experiments_before_revisit: int = 10  # 至少跑幾次後才能重複參數組合


class MutationStrategy:
    """
    隨機扰動策略
    
    借鑒 Karpathy Autoresearch 概念：
    - 對當前最佳參數做隨機扰動
    - 避免完全隨機搜索（Epsilon-Greedy 變體）
    """

    def __init__(self, specs: Dict[str, StrategyParamSpec], config: Optional[MutationConfig] = None):
        self.specs = specs  # {param_name: spec}
        self.config = config or MutationConfig()
        self.history: List[Dict[str, Any]] = []  # 記錄所有嘗試過的參數

    def mutate(self, base_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        對基礎參數進行隨機扰動
        
        Args:
            base_params: 基礎參數
            
        Returns:
            新的参數組合
        """
        new_params = base_params.copy()
        
        for param_name, spec in self.specs.items():
            if param_name not in new_params:
                continue
            
            # Epsilon-greedy: 以一定機率扰動
            if random.random() < self.config.mutation_probability:
                current = new_params[param_name]
                new_params[param_name] = spec.mutate_value(current)
        
        return new_params

    def suggest(self, base_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        建議下一個實驗參數
        
        80% 機率：基於 base_params 扰動
        20% 機率：隨機探索（避免局部最優）
        
        Args:
            base_params: 當前最佳參數（可為 None，則隨機）
        """
        if base_params is None or random.random() < 0.2:
            # 隨機探索
            return {name: spec.random_value() for name, spec in self.specs.items()}
        
        # 扰動當前最佳
        attempts = 0
        while attempts < 100:
            candidate = self.mutate(base_params)
            if not self._is_duplicate(candidate):
                self.history.append(candidate)
                return candidate
            attempts += 1
        
        # 找不到新參數，隨機
        return {name: spec.random_value() for name, spec in self.specs.items()}

    def _is_duplicate(self, params: Dict[str, Any]) -> bool:
        """檢查參數組合是否重複"""
        for hist in self.history[-self.config.min_experiments_before_revisit:]:
            if all(
                hist.get(k) == params.get(k) 
                for k in params.keys()
            ):
                return True
        return False

    def record(self, params: Dict[str, Any]):
        """記錄已使用的參數"""
        self.history.append(params.copy())

    def random_explore(self) -> Dict[str, Any]:
        """完全隨機探索"""
        return {name: spec.random_value() for name, spec in self.specs.items()}


class GridSearchStrategy:
    """
    網格搜索策略
    
    產生所有參數組合，用於確保覆蓋所有可能
    """

    def __init__(self, specs: Dict[str, StrategyParamSpec]):
        self.specs = specs
        self._combinations: Optional[List[Dict[str, Any]]] = None
        self._index = 0

    def build_grid(self) -> List[Dict[str, Any]]:
        """構建完整網格"""
        if self._combinations is not None:
            return self._combinations
        
        keys = list(self.specs.keys())
        value_lists = []
        
        for name in keys:
            spec = self.specs[name]
            values = []
            if spec.param_type == "int":
                v = spec.min_val
                while v <= spec.max_val:
                    values.append(int(v))
                    v += spec.step
            else:
                v = spec.min_val
                while v <= spec.max_val:
                    values.append(round(v, 4))
                    v += spec.step
            value_lists.append(values)
        
        self._combinations = []
        for combo in itertools.product(*value_lists):
            self._combinations.append(dict(zip(keys, combo)))
        
        random.shuffle(self._combinations)  # 隨機順序避免偏見
        return self._combinations

    def suggest(self) -> Optional[Dict[str, Any]]:
        """建議下一個實驗參數（按順序）"""
        grid = self.build_grid()
        if self._index >= len(grid):
            return None  # 網格搜索完成
        params = grid[self._index]
        self._index += 1
        return params

    def reset(self):
        """重置搜索指標"""
        self._index = 0

    @property
    def progress(self) -> float:
        """搜索進度"""
        grid = self.build_grid()
        return self._index / len(grid) if grid else 1.0

    @property
    def remaining(self) -> int:
        """剩餘組合數"""
        grid = self.build_grid()
        return len(grid) - self._index


class BayesianOptimizer:
    """
    貝葉斯優化策略（可選）
    
    使用歷史實驗結果指導下一個實驗點的選擇
    需要 sklearn / scipy
    
    適用場景：
    - 連續參數空間
    - 已經跑了 10+ 實驗後
    """

    def __init__(self, specs: Dict[str, StrategyParamSpec]):
        self.specs = specs
        self.experiments: List[ExperimentResult] = []
        self.param_names = list(specs.keys())
        self._initialized = False

    def record(self, result: ExperimentResult):
        """記錄實驗結果"""
        self.experiments.append(result)
        if len(self.experiments) >= 5:
            self._initialized = True

    def suggest(self) -> Optional[Dict[str, Any]]:
        """建議下一個實驗參數"""
        if not self._initialized:
            # 前期隨機探索
            return {name: spec.random_value() for name, spec in self.specs.items()}
        
        try:
            return self._bayesian_suggest()
        except ImportError:
            # fallback 到隨機
            return {name: spec.random_value() for name, spec in self.specs.items()}

    def _bayesian_suggest(self) -> Dict[str, Any]:
        """貝葉斯優化選擇"""
        try:
            from sklearn.gaussian_process import GaussianProcessRegressor
            from sklearn.gaussian_process.kernels import RBF, ConstantKernel
        except ImportError:
            raise ImportError("需要 scikit-learn: pip install scikit-learn")
        
        # 準備數據
        X = []
        y = []
        for exp in self.experiments:
            if exp.status == "crash":
                continue
            vec = [exp.params.get(name, 0) for name in self.param_names]
            X.append(vec)
            # 優化目標：夏普值（越高越好）
            y.append(exp.sharpe)
        
        if len(X) < 5:
            return {name: spec.random_value() for name, spec in self.specs.items()}
        
        X = np.array(X)
        y = np.array(y)
        
        # 標準化
        X_mean = X.mean(axis=0)
        X_std = X.std(axis=0) + 1e-8
        X_norm = (X - X_mean) / X_std
        
        # 高斯過程
        kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)
        gpr = GaussianProcessRegressor(kernel, n_restarts_optimizer=2, random_state=42)
        gpr.fit(X_norm, y)
        
        # 生成候選點
        candidates = []
        for _ in range(100):
            cand = np.array([[spec.random_value() for spec in self.specs.values()]])
            cand_norm = (cand - X_mean) / X_std
            mu, sigma = gpr.predict(cand_norm, return_std=True)
            
            # UCB 獲取函數
            acquisition = mu + 2.0 * sigma
            candidates.append((cand[0], acquisition[0]))
        
        # 選擇最大 UCB
        best = max(candidates, key=lambda x: x[1])
        
        result = {}
        for i, name in enumerate(self.param_names):
            val = best[0][i]
            spec = self.specs[name]
            val = max(spec.min_val, min(spec.max_val, val))
            result[name] = int(val) if spec.param_type == "int" else round(val, 4)
        
        return result


class EnsembleStrategy:
    """
    集成策略 - 同時使用多種探索方法
    
    平衡：
    - MutationStrategy (40%) - 利用當前最佳
    - RandomExplore (30%)   - 全域探索  
    - BayesianOpt (30%)    - 高效優化（如果可用）
    """

    def __init__(
        self,
        specs: Dict[str, StrategyParamSpec],
        mutation_weight: float = 0.4,
        random_weight: float = 0.3,
        bayesian_weight: float = 0.3
    ):
        self.specs = specs
        self.mutation = MutationStrategy(specs)
        self.grid = GridSearchStrategy(specs)
        self.bayesian = BayesianOptimizer(specs)
        
        self.mutation_weight = mutation_weight
        self.random_weight = random_weight
        self.bayesian_weight = bayesian_weight
        
        self.best_params: Optional[Dict[str, Any]] = None
        self.best_sharpe: float = -999.0

    def suggest(self) -> Dict[str, Any]:
        """建議下一個實驗參數"""
        if self.best_params is None:
            # 第一次：隨機
            return self.mutation.random_explore()
        
        roll = random.random()
        
        if roll < self.mutation_weight:
            # 突變
            return self.mutation.suggest(self.best_params)
        elif roll < self.mutation_weight + self.random_weight:
            # 隨機
            return self.mutation.random_explore()
        else:
            # 貝葉斯
            return self.bayesian.suggest()

    def update_best(self, params: Dict[str, Any], sharpe: float):
        """更新最佳參數"""
        if sharpe > self.best_sharpe:
            self.best_sharpe = sharpe
            self.best_params = params.copy()
            print(f"  🏆 New best Sharpe: {sharpe:.4f} with {params}")

    def record(self, result: ExperimentResult):
        """記錄實驗結果用於貝葉斯"""
        self.bayesian.record(result)
        if result.status != "crash":
            self.update_best(result.params, result.sharpe)
