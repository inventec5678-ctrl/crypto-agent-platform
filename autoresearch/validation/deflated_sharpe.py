# deflated_sharpe.py

import numpy as np


def deflated_sharpe_ratio(all_sharpes, target_sr=None):
    """
    計算 Deflated Sharpe Ratio
    考慮多重比較偏差後的真實夏普值
    """
    n = len(all_sharpes)
    sorted_sharpes = sorted(all_sharpes, reverse=True)
    if target_sr is None:
        # 使用第 N/2 分位數作為 benchmark
        target_sr = np.percentile(sorted_sharpes, 50)
    # 非參數估計：計算 Sharpe > target 的比例
    prob_above = sum(1 for s in all_sharpes if s > target_sr) / n
    # DSR = (mean - target) / std * sqrt(n)
    mean_sr = np.mean(all_sharpes)
    std_sr = np.std(all_sharpes)
    dsr = (mean_sr - target_sr) / std_sr * np.sqrt(n) if std_sr > 0 else 0
    return {
        'dsr': dsr,
        'prob_above_median': prob_above,
        'n_strategies': n
    }
