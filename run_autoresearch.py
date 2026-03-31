#!/usr/bin/env python3
"""
Auto Research 腳本 - 成交量策略研究
"""

import asyncio
import json
import pandas as pd
import sys
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from autoresearch.loop import AutoresearchLoop, LoopConfig
from autoresearch.models import StrategyParamSpec


def load_local_parquet_data(symbol="BTCUSDT", interval="1d"):
    """載入本地 parquet 數據"""
    path = PROJECT_ROOT / "data" / f"{symbol.lower()}_{interval}.parquet"
    if path.exists():
        df = pd.read_parquet(path)
        print(f"  📁 已載入本地數據: {path} ({len(df)} rows)")
        return {symbol: df}
    else:
        print(f"  ⚠️ 本地數據不存在: {path}")
        return None


def main():
    print("=" * 60)
    print("🚀 Auto Research 啟動 - 成交量策略研究")
    print("=" * 60)
    
    # 載入本地數據
    local_data = load_local_parquet_data("BTCUSDT", "1d")
    if local_data is None:
        print("❌ 無法載入本地數據，退出")
        return
    
    # 只使用成交量策略
    strategies = {
        "VolumeBreakoutStrategy": __import__('strategies.volume_strategy', fromlist=['VolumeBreakoutStrategy']).VolumeBreakoutStrategy
    }
    
    # 定義參數空間
    param_specs = {
        "VolumeBreakoutStrategy": {
            "volume_ma_period": StrategyParamSpec(
                name="VolumeBreakoutStrategy", param_name="volume_ma_period",
                min_val=5, max_val=50, step=5, param_type="int", mutate_ratio=0.15
            ),
            "volume_multiplier": StrategyParamSpec(
                name="VolumeBreakoutStrategy", param_name="volume_multiplier",
                min_val=1.2, max_val=4.0, step=0.2, param_type="float", mutate_ratio=0.1
            ),
            "trend_period": StrategyParamSpec(
                name="VolumeBreakoutStrategy", param_name="trend_period",
                min_val=2, max_val=20, step=2, param_type="int", mutate_ratio=0.15
            ),
            "stop_loss_atr": StrategyParamSpec(
                name="VolumeBreakoutStrategy", param_name="stop_loss_atr",
                min_val=0.5, max_val=5.0, step=0.5, param_type="float", mutate_ratio=0.2
            ),
            "take_profit_atr": StrategyParamSpec(
                name="VolumeBreakoutStrategy", param_name="take_profit_atr",
                min_val=1.0, max_val=8.0, step=0.5, param_type="float", mutate_ratio=0.2
            ),
        }
    }
    
    print(f"  📊 使用策略: VolumeBreakoutStrategy")
    
    # 配置
    config = LoopConfig(
        max_experiments=None,  # 無限運行
        pause_between_experiments=3.0,
        enable_bayesian=True,
        keep_best_params_from_persistence=True,
    )
    
    # Autoresearch 目錄
    base_dir = PROJECT_ROOT / "autoresearch"
    base_dir.mkdir(exist_ok=True)
    
    # 創建循環
    loop = AutoresearchLoop(
        base_dir=str(base_dir),
        strategies=strategies,
        param_specs=param_specs,
        config=config,
        symbol="BTCUSDT",
        interval="1d",
        start_date="2017-01-01",
        end_date="2026-12-31",
        data_fetcher=local_data,
    )
    
    # 啟動
    print("\n📊 開始自動研究成交量策略...")
    loop.start(max_experiments=None, background=True)


if __name__ == "__main__":
    main()
