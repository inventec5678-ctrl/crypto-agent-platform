"""
Autoresearch - 24/7 自動策略實驗系統

模組結構:
  autoresearch/
    models.py          - 數據模型 (ExperimentResult, BestParams, ...)
    experiment_strategies.py - 參數探索策略
    experiment_engine.py    - 實驗執行引擎
    persistence.py     - 結果持久化
    loop.py            - 24/7 自動循環控制器
    registry.py         - 策略參數註冊表
    cli.py             - 命令行入口
    market_regime.py   - 市場分層模組 (牛市/熊市/盤整)
    factor_library.py  - 多因子策略庫
"""

from .models import (
    ExperimentResult,
    BestParams,
    AutoresearchState,
    StrategyParamSpec,
)
from .experiment_strategies import (
    MutationStrategy,
    GridSearchStrategy,
    BayesianOptimizer,
    EnsembleStrategy,
)
from .persistence import Persistence
from .experiment_engine import ExperimentEngine, BacktestStrategyWrapper
from .registry import ALL_SPECS, get_specs, get_default_params
from .market_regime import MarketRegimeClassifier
from .factor_library import FactorLibrary, FactorRecord

__all__ = [
    "ExperimentResult",
    "BestParams", 
    "AutoresearchState",
    "StrategyParamSpec",
    "MutationStrategy",
    "GridSearchStrategy",
    "BayesianOptimizer",
    "EnsembleStrategy",
    "Persistence",
    "ExperimentEngine",
    "BacktestStrategyWrapper",
    "ALL_SPECS",
    "get_specs",
    "get_default_params",
    "MarketRegimeClassifier",
    "FactorLibrary",
    "FactorRecord",
]
