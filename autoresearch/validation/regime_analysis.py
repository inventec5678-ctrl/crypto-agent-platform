# regime_analysis.py
# 用簡單滾動 Sharpe 識別市場regime

import numpy as np


def detect_regimes(prices, window=30):
    """
    用 rolling Sharpe 識別 regime
    regime 1: Sharpe > 1.5 (多頭)
    regime 2: Sharpe < -1.5 (空頭)
    regime 3: 其他 (橫盤)
    """
    returns = np.diff(prices) / prices[:-1]
    regimes = []
    for i in range(window, len(returns)+1):
        window_returns = returns[i-window:i]
        sharpe = np.mean(window_returns) / np.std(window_returns) * np.sqrt(365) if np.std(window_returns) > 0 else 0
        if sharpe > 1.5:
            regimes.append('bull')
        elif sharpe < -1.5:
            regimes.append('bear')
        else:
            regimes.append('neutral')
    return regimes


def regime_performance(prices, regimes):
    """各regime的績效統計"""
    returns = np.diff(prices) / prices[:-1]
    stats = {}
    for r in ['bull', 'bear', 'neutral']:
        mask = [regimes[i] == r for i in range(len(regimes))]
        r_returns = [returns[i] for i in range(len(mask)) if mask[i]]
        if r_returns:
            stats[r] = {
                'count': len(r_returns),
                'mean_return': np.mean(r_returns),
                'win_rate': sum(1 for x in r_returns if x > 0) / len(r_returns)
            }
    return stats
