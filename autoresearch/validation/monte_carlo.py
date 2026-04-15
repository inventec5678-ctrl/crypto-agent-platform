# monte_carlo.py
# 用 block bootstrap 評估策略穩定性

import random
import numpy as np


def block_bootstrap_returns(returns, n_bootstrap=1000, block_size=5):
    """Block bootstrap for time series returns"""
    results = []
    for _ in range(n_bootstrap):
        # 隨機選擇 block 起點
        indices = []
        while len(indices) < len(returns):
            start = random.randint(0, len(returns) - block_size)
            indices.extend(range(start, start + block_size))
        indices = indices[:len(returns)]
        bootstrap = [returns[i] for i in indices]
        results.append(sum(bootstrap) / len(bootstrap))
    return np.array(results)


def evaluate_strategy_stability(returns, n_bootstrap=1000):
    bs_returns = block_bootstrap_returns(returns, n_bootstrap)
    return {
        'mean': np.mean(bs_returns),
        'std': np.std(bs_returns),
        'sharpe': np.mean(bs_returns) / np.std(bs_returns) * np.sqrt(365),
        'var_5pct': np.percentile(bs_returns, 5),
        'sharpe_confidence_interval': (
            np.percentile(bs_returns, 2.5) / np.std(bs_returns),
            np.percentile(bs_returns, 97.5) / np.std(bs_returns)
        )
    }
