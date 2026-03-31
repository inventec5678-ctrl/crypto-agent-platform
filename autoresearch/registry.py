"""
策略參數註冊表

定義所有可用策略及其參數空間
"""

from typing import Dict, Type
from .models import StrategyParamSpec

# ── MA 交叉策略參數 ──
MA_CROSS_SPECS: Dict[str, StrategyParamSpec] = {
    "fast_period": StrategyParamSpec(
        name="MACrossoverStrategy",
        param_name="fast_period",
        min_val=2,
        max_val=50,
        step=1,
        param_type="int",
        mutate_ratio=0.15
    ),
    "slow_period": StrategyParamSpec(
        name="MACrossoverStrategy",
        param_name="slow_period",
        min_val=10,
        max_val=200,
        step=5,
        param_type="int",
        mutate_ratio=0.10
    ),
}

# ── RSI 反轉策略參數 ──
RSI_SPECS: Dict[str, StrategyParamSpec] = {
    "period": StrategyParamSpec(
        name="RSIReversalStrategy",
        param_name="period",
        min_val=5,
        max_val=30,
        step=1,
        param_type="int",
        mutate_ratio=0.10
    ),
    "oversold": StrategyParamSpec(
        name="RSIReversalStrategy",
        param_name="oversold",
        min_val=10.0,
        max_val=40.0,
        step=5.0,
        param_type="float",
        mutate_ratio=0.10
    ),
    "overbought": StrategyParamSpec(
        name="RSIReversalStrategy",
        param_name="overbought",
        min_val=60.0,
        max_val=90.0,
        step=5.0,
        param_type="float",
        mutate_ratio=0.10
    ),
}

# ── 布林帶策略參數 ──
BB_SPECS: Dict[str, StrategyParamSpec] = {
    "period": StrategyParamSpec(
        name="BBBreakoutStrategy",
        param_name="period",
        min_val=10,
        max_val=50,
        step=5,
        param_type="int",
        mutate_ratio=0.10
    ),
    "std_dev": StrategyParamSpec(
        name="BBBreakoutStrategy",
        param_name="std_dev",
        min_val=1.0,
        max_val=3.0,
        step=0.25,
        param_type="float",
        mutate_ratio=0.10
    ),
}

# ── 全部註冊表 ──
ALL_SPECS: Dict[str, Dict[str, StrategyParamSpec]] = {
    "MACrossoverStrategy": MA_CROSS_SPECS,
    "RSIReversalStrategy": RSI_SPECS,
    "BBBreakoutStrategy": BB_SPECS,
}

# ── 默認值 ──
DEFAULT_PARAMS: Dict[str, Dict[str, any]] = {
    "MACrossoverStrategy": {"fast_period": 10, "slow_period": 30},
    "RSIReversalStrategy": {"period": 14, "oversold": 30.0, "overbought": 70.0},
    "BBBreakoutStrategy": {"period": 20, "std_dev": 2.0},
}


def get_specs(strategy_name: str) -> Dict[str, StrategyParamSpec]:
    """獲取策略參數規格"""
    return ALL_SPECS.get(strategy_name, {})


def get_default_params(strategy_name: str) -> Dict[str, any]:
    """獲取策略默認參數"""
    return DEFAULT_PARAMS.get(strategy_name, {})


# ── Volume Breakout 策略參數 ──
VOLUME_SPECS: Dict[str, StrategyParamSpec] = {
    "volume_ma_period": StrategyParamSpec(
        name="VolumeBreakoutStrategy",
        param_name="volume_ma_period",
        min_val=5,
        max_val=50,
        step=5,
        param_type="int",
        mutate_ratio=0.15
    ),
    "volume_multiplier": StrategyParamSpec(
        name="VolumeBreakoutStrategy",
        param_name="volume_multiplier",
        min_val=1.2,
        max_val=4.0,
        step=0.2,
        param_type="float",
        mutate_ratio=0.1
    ),
    "trend_period": StrategyParamSpec(
        name="VolumeBreakoutStrategy",
        param_name="trend_period",
        min_val=2,
        max_val=20,
        step=2,
        param_type="int",
        mutate_ratio=0.15
    ),
    "stop_loss_atr": StrategyParamSpec(
        name="VolumeBreakoutStrategy",
        param_name="stop_loss_atr",
        min_val=0.5,
        max_val=5.0,
        step=0.5,
        param_type="float",
        mutate_ratio=0.2
    ),
    "take_profit_atr": StrategyParamSpec(
        name="VolumeBreakoutStrategy",
        param_name="take_profit_atr",
        min_val=1.0,
        max_val=8.0,
        step=0.5,
        param_type="float",
        mutate_ratio=0.2
    ),
}

# 添加到 ALL_SPECS
ALL_SPECS["VolumeBreakoutStrategy"] = VOLUME_SPECS

DEFAULT_PARAMS["VolumeBreakoutStrategy"] = {
    "volume_ma_period": 20,
    "volume_multiplier": 2.0,
    "trend_period": 5,
    "stop_loss_atr": 2.0,
    "take_profit_atr": 3.0,
}
