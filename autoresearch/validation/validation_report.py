# validation_report.py
# 產生驗證報告

from .monte_carlo import evaluate_strategy_stability
from .walk_forward import walk_forward_validation, summarize_wfv
from .deflated_sharpe import deflated_sharpe_ratio
from .regime_analysis import detect_regimes, regime_performance


def generate_validation_report(strategy_name, returns, prices, all_strategy_sharpes=None):
    """整合所有驗證指標的報告"""
    # Monte Carlo
    mc = evaluate_strategy_stability(returns)
    # WFV
    wfv_results = walk_forward_validation(prices, train_days=300, test_days=30)
    wfv_summary = summarize_wfv(wfv_results)
    # DSR
    dsr = deflated_sharpe_ratio(all_strategy_sharpes) if all_strategy_sharpes else None
    # Regime
    regimes = detect_regimes(prices)
    regime_stats = regime_performance(prices, regimes)

    report = f"""
## Validation Report: {strategy_name}

### Monte Carlo (1000 bootstrap)
- Mean Return: {mc['mean']:.4f}
- Sharpe: {mc['sharpe']:.3f}
- VaR 5%: {mc['var_5pct']:.4f}

### Walk Forward Validation
- OOS Positive Ratio: {wfv_summary['oos_positive_ratio']:.1%}
- Avg Overfitting: {wfv_summary['avg_overfitting_ratio']:.2%}

### Deflated Sharpe Ratio
- DSR: {dsr['dsr']:.3f} (if > 1 = significant)
- Prob above median: {dsr['prob_above_median']:.1%}

### Regime Stats
{regime_stats}
"""
    return report
