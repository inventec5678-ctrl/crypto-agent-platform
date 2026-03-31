"""
Autoresearch 數據模型
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Optional, Any
from datetime import datetime
import hashlib
import json


@dataclass
class ExperimentResult:
    """單次實驗結果"""
    experiment_id: str
    commit_hash: str           # 實驗版本/commit
    strategy_name: str          # 策略名稱
    params: Dict[str, Any]     # 使用的參數
    metrics: Dict[str, float]  # 夏普值、勝率、回撤等
    status: str                # "keep" / "discard" / "crash"
    description: str           # 實驗描述
    timestamp: str             # ISO 時間戳
    duration_seconds: float    # 實驗耗時
    backtest_period: Optional[Dict[str, str]] = None  # 回測期間

    def __post_init__(self):
        if not self.experiment_id:
            self.experiment_id = self._generate_id()

    def _generate_id(self) -> str:
        """根據參數和時間生成唯一 ID"""
        content = f"{self.strategy_name}:{json.dumps(self.params, sort_keys=True)}:{self.timestamp}"
        return hashlib.sha256(content.encode()).hexdigest()[:12]

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ExperimentResult":
        return cls(**d)

    @property
    def sharpe(self) -> float:
        return self.metrics.get("sharpe_ratio", 0.0)

    @property
    def win_rate(self) -> float:
        return self.metrics.get("win_rate", 0.0)

    @property
    def max_drawdown(self) -> float:
        return self.metrics.get("max_drawdown_pct", 0.0)

    @property
    def total_return(self) -> float:
        return self.metrics.get("total_return_pct", 0.0)


@dataclass
class BestParams:
    """最佳參數追蹤"""
    strategy_name: str
    params: Dict[str, Any]
    metrics: Dict[str, float]
    experiment_id: str
    updated_at: str

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "BestParams":
        return cls(**d)


@dataclass
class AutoresearchState:
    """Autoresearch 全局狀態"""
    total_experiments: int = 0
    successful_experiments: int = 0
    discarded_experiments: int = 0
    crashed_experiments: int = 0
    best_by_strategy: Dict[str, BestParams] = field(default_factory=dict)
    recent_results: list = field(default_factory=list)  # 最近實驗 ID 列表
    started_at: Optional[str] = None
    last_experiment_at: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "total_experiments": self.total_experiments,
            "successful_experiments": self.successful_experiments,
            "discarded_experiments": self.discarded_experiments,
            "crashed_experiments": self.crashed_experiments,
            "best_by_strategy": {k: v.to_dict() for k, v in self.best_by_strategy.items()},
            "recent_results": self.recent_results,
            "started_at": self.started_at,
            "last_experiment_at": self.last_experiment_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "AutoresearchState":
        d = d.copy()
        if "best_by_strategy" in d:
            d["best_by_strategy"] = {
                k: BestParams.from_dict(v) for k, v in d["best_by_strategy"].items()
            }
        return cls(**d)


@dataclass
class StrategyParamSpec:
    """策略參數規格"""
    name: str
    param_name: str
    min_val: float
    max_val: float
    step: float
    param_type: str = "int"  # "int" | "float"
    mutate_ratio: float = 0.1  # 突變比例 10%

    def mutate_value(self, current: float) -> float:
        """對當前值進行隨機扰動"""
        import random
        delta = current * random.uniform(-self.mutate_ratio, self.mutate_ratio)
        new_val = current + delta

        # 限制範圍
        new_val = max(self.min_val, min(self.max_val, new_val))

        if self.param_type == "int":
            # 對齊 step
            new_val = round(new_val / self.step) * self.step
            return int(new_val)
        else:
            return round(new_val, 4)

    def random_value(self) -> float:
        """產生隨機值"""
        import random
        if self.param_type == "int":
            steps = int((self.max_val - self.min_val) / self.step)
            return int(self.min_val + random.randint(0, steps) * self.step)
        else:
            return random.uniform(self.min_val, self.max_val)
