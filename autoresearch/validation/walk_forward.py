# walk_forward.py
# 滾動視窗驗證

import numpy as np


def walk_forward_validation(prices, train_days=300, test_days=30):
    """
    訓練窗口 train_days，測試窗口 test_days
    產生多個 OOS 區間
    """
    results = []
    start = 0
    while start + train_days + test_days <= len(prices):
        train = prices[start:start+train_days]
        test = prices[start+train_days:start+train_days+test_days]
        # 計算 train 期間夏普值
        train_returns = np.diff(train) / train[:-1]
        train_sharpe = np.mean(train_returns) / np.std(train_returns) * np.sqrt(365)
        # 計算 test 期間夏普值
        test_returns = np.diff(test) / test[:-1]
        test_sharpe = np.mean(test_returns) / np.std(test_returns) * np.sqrt(365)
        results.append({
            'train_sharpe': train_sharpe,
            'test_sharpe': test_sharpe,
            'overfitting_ratio': (train_sharpe - test_sharpe) / abs(train_sharpe) if train_sharpe != 0 else 0
        })
        start += test_days
    return results


def summarize_wfv(results):
    """WFV 結果摘要"""
    test_sharpes = [r['test_sharpe'] for r in results]
    oos_positive_ratio = sum(1 for s in test_sharpes if s > 0) / len(test_sharpes)
    avg_overfitting = np.mean([r['overfitting_ratio'] for r in results])
    return {
        'oos_positive_ratio': oos_positive_ratio,
        'avg_overfitting_ratio': avg_overfitting,
        'max_oos_sharpe': max(test_sharpes),
        'min_oos_sharpe': min(test_sharpes)
    }
