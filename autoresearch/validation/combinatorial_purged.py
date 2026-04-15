# combinatorial_purged.py

import numpy as np
from itertools import combinations


def combinatorial_purged_cv(n_samples, n_splits=6, purge_pct=0.1):
    """
    標準 CPCV：
    1. 把資料分成 K 份（每份 n/K 筆）
    2. 枚舉所有從 K 份中選 2 份當 test 的組合（K×(K-1)/2 種）
    3. 其餘 K-2 份當 train
    4. 加入 purge buffer（防止 look-ahead）
    """
    k = n_splits
    fold_size = n_samples // k
    purge_size = int(fold_size * purge_pct)  # 10% of fold = purge

    # 建立 K 個 fold 的 index
    folds = []
    for i in range(k):
        start = i * fold_size
        end = start + fold_size if i < k - 1 else n_samples
        folds.append(list(range(start, end)))

    # 枚舉所有從 K 份選 2 份當 test 的組合
    splits = []
    for test_pair in combinations(range(k), 2):
        test_set = set()
        for fold_idx in test_pair:
            test_set.update(folds[fold_idx])

        train_set = set()
        for fold_idx in range(k):
            if fold_idx not in test_pair:
                train_set.update(folds[fold_idx])

        # Purge：移除 test 邊緣附近的 train 樣本
        test_min = min(test_set)
        test_max = max(test_set)
        train_set_purged = {
            i for i in train_set
            if not (test_min - purge_size <= i <= test_max + purge_size)
        }

        splits.append({
            'train': sorted(list(train_set_purged)),
            'test': sorted(list(test_set))
        })

    return splits


def evaluate_cpcv(strategy_func, prices, n_splits=6):
    """對策略進行 CPCV 評估"""
    returns = np.diff(prices) / prices[:-1]
    splits = combinatorial_purged_cv(len(returns), n_splits)
    results = []
    for split in splits:
        train_returns = [returns[i] for i in split['train'] if i < len(returns)]
        test_returns = [returns[i] for i in split['test'] if i < len(returns)]
        train_sharpe = np.mean(train_returns) / np.std(train_returns) * np.sqrt(365) if np.std(train_returns) > 0 else 0
        test_sharpe = np.mean(test_returns) / np.std(test_returns) * np.sqrt(365) if np.std(test_returns) > 0 else 0
        results.append({
            'train_sharpe': train_sharpe,
            'test_sharpe': test_sharpe
        })
    return results
