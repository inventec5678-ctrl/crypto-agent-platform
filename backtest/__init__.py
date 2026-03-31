"""
加密貨幣回測系統
位置: crypto-agent-platform/backtest/

提供完整的回測功能:
- 回測引擎核心 (backtest_engine.py)
- 績效指標計算 (performance_metrics.py)
- 策略參數優化 (optimization.py)
- Autoresearch 整合介面 (autoresearch_interface.py)
"""

from .backtest_engine import (
    BacktestEngine,
    BaseStrategy,
    SimpleMovingAverageCrossover,
    PositionSide,
    Order,
    Position,
    Trade,
    BacktestResult,
    AutoResearchInterface,
    quick_backtest
)

from .performance_metrics import (
    PerformanceMetrics,
    quick_metrics
)

from .optimization import (
    GridSearchOptimizer,
    WalkForwardAnalyzer,
    OverfittingDetector,
    OptimizationResult,
    optimize_strategy
)

from .walk_forward_analysis import (
    WalkForwardConfig,
    WalkForwardResult,
    WindowResult,
    WalkForwardAnalysisEngine,
    run_walk_forward_analysis
)

__all__ = [
    # 引擎
    'BacktestEngine',
    'BaseStrategy', 
    'SimpleMovingAverageCrossover',
    'PositionSide',
    'Order',
    'Position',
    'Trade',
    'BacktestResult',
    'AutoResearchInterface',
    'quick_backtest',
    
    # 績效指標
    'PerformanceMetrics',
    'quick_metrics',
    
    # 優化
    'GridSearchOptimizer',
    'WalkForwardAnalyzer',
    'OverfittingDetector',
    'OptimizationResult',
    'optimize_strategy',
    
    # Walk-Forward 分析
    'WalkForwardConfig',
    'WalkForwardResult',
    'WindowResult',
    'WalkForwardAnalysisEngine',
    'run_walk_forward_analysis',
]
