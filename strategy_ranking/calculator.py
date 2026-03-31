"""
Metrics calculation utilities.
"""

import math
from typing import List, Optional
from .models import Trade, StrategyMetrics


def calculate_win_rate(trades: List[Trade]) -> float:
    """Calculate win rate from closed trades."""
    closed = [t for t in trades if t.pnl is not None]
    if not closed:
        return 0.0
    winners = sum(1 for t in closed if t.is_winner)
    return winners / len(closed)


def calculate_avg_profit_loss(trades: List[Trade]) -> tuple[float, float]:
    """Calculate average profit and loss from closed trades."""
    closed = [t for t in trades if t.pnl is not None]
    if not closed:
        return 0.0, 0.0

    profits = [t.pnl for t in closed if t.pnl > 0]
    losses = [t.pnl for t in closed if t.pnl < 0]

    avg_profit = sum(profits) / len(profits) if profits else 0.0
    avg_loss = abs(sum(losses) / len(losses)) if losses else 0.0
    return avg_profit, avg_loss


def calculate_profit_factor(trades: List[Trade]) -> float:
    """Calculate profit factor = gross profit / gross loss."""
    closed = [t for t in trades if t.pnl is not None]
    if not closed:
        return 0.0

    gross_profit = sum(t.pnl for t in closed if t.pnl > 0)
    gross_loss = abs(sum(t.pnl for t in closed if t.pnl < 0))

    if gross_loss == 0:
        # Cap at 999.0 instead of infinity to allow JSON serialization
        return 999.0 if gross_profit > 0 else 0.0
    return gross_profit / gross_loss


def calculate_sharpe_ratio(trades: List[Trade], risk_free_rate: float = 0.0) -> float:
    """
    Calculate Sharpe Ratio from closed trades.
    Uses trade PnLs as returns.
    """
    closed = [t for t in trades if t.pnl is not None]
    if len(closed) < 2:
        return 0.0

    pnls = [t.pnl for t in closed]
    avg_return = sum(pnls) / len(pnls)
    variance = sum((p - avg_return) ** 2 for p in pnls) / len(pnls)
    std_dev = math.sqrt(variance)

    if std_dev == 0:
        return 0.0
    return (avg_return - risk_free_rate) / std_dev


def calculate_max_drawdown(equity_curve: List[float]) -> float:
    """
    Calculate maximum drawdown from equity curve.
    Returns drawdown as a positive percentage (e.g., 15.5 means 15.5% drawdown).
    """
    if not equity_curve:
        return 0.0

    peak = equity_curve[0]
    max_dd = 0.0

    for equity in equity_curve:
        if equity > peak:
            peak = equity
        drawdown = (peak - equity) / peak * 100 if peak > 0 else 0
        if drawdown > max_dd:
            max_dd = drawdown

    return max_dd


def calculate_drawdown_curve(equity_curve: List[float]) -> List[float]:
    """
    Calculate drawdown curve (percentage drawdown at each point).
    """
    if not equity_curve:
        return []

    drawdowns = []
    peak = equity_curve[0]

    for equity in equity_curve:
        if equity > peak:
            peak = equity
        dd = (peak - equity) / peak * 100 if peak > 0 else 0
        drawdowns.append(dd)

    return drawdowns


def calculate_expectancy(trades: List[Trade]) -> float:
    """
    Calculate expectancy = (win_rate * avg_win) - (loss_rate * avg_loss)
    Returns expected profit per trade.
    """
    closed = [t for t in trades if t.pnl is not None]
    if not closed:
        return 0.0

    win_rate = calculate_win_rate(closed)
    avg_profit, avg_loss = calculate_avg_profit_loss(closed)

    loss_rate = 1 - win_rate
    expectancy = (win_rate * avg_profit) - (loss_rate * avg_loss)
    return expectancy


def calculate_consecutive_wins_losses(trades: List[Trade]) -> tuple[int, int]:
    """Calculate max consecutive wins and losses."""
    closed = [t for t in trades if t.pnl is not None]
    if not closed:
        return 0, 0

    max_wins = curr_wins = 0
    max_losses = curr_losses = 0

    for t in closed:
        if t.is_winner:
            curr_wins += 1
            curr_losses = 0
            max_wins = max(max_wins, curr_wins)
        else:
            curr_losses += 1
            curr_wins = 0
            max_losses = max(max_losses, curr_losses)

    return max_wins, max_losses


def calculate_score(metrics: StrategyMetrics) -> float:
    """
    综合评分公式
    score = (win_rate * 30) + (profit_factor * 20) + (sharpe_ratio * 30) + (expectancy * 20)

    Note: win_rate and expectancy are normalized to 0-100 scale,
    profit_factor is capped at reasonable ranges.
    """
    # Normalize win_rate to 0-100
    win_rate_score = metrics.win_rate * 100

    # Normalize profit_factor (cap at 5.0 = very good)
    pf = min(metrics.profit_factor, 5.0)
    profit_factor_score = (pf / 5.0) * 100

    # Normalize sharpe_ratio (cap at 3.0 = very good)
    sr = min(max(metrics.sharpe_ratio, 0), 3.0)
    sharpe_score = (sr / 3.0) * 100

    # Normalize expectancy (cap at 100)
    exp = min(max(metrics.expectancy, 0), 100)
    expectancy_score = (exp / 100) * 100

    score = (win_rate_score * 0.30) + \
            (profit_factor_score * 0.20) + \
            (sharpe_score * 0.30) + \
            (expectancy_score * 0.20)

    return round(score, 2)


def build_metrics(strategy_name: str, trades: List[Trade], initial_capital: float = 10000.0) -> StrategyMetrics:
    """Build complete StrategyMetrics from list of trades."""
    closed = [t for t in trades if t.pnl is not None]

    win_rate = calculate_win_rate(trades)
    avg_profit, avg_loss = calculate_avg_profit_loss(trades)
    profit_factor = calculate_profit_factor(trades)
    sharpe_ratio = calculate_sharpe_ratio(trades)
    expectancy = calculate_expectancy(trades)
    consecutive_wins, consecutive_losses = calculate_consecutive_wins_losses(trades)
    total_pnl = sum(t.pnl for t in closed)

    # Build equity curve
    equity_curve = [initial_capital]
    for t in closed:
        equity_curve.append(equity_curve[-1] + t.pnl)

    max_drawdown = calculate_max_drawdown(equity_curve)

    return StrategyMetrics(
        strategy_name=strategy_name,
        total_trades=len(closed),
        win_rate=win_rate,
        avg_profit=avg_profit,
        avg_loss=avg_loss,
        profit_factor=profit_factor,
        sharpe_ratio=sharpe_ratio,
        max_drawdown=max_drawdown,
        expectancy=expectancy,
        consecutive_wins=consecutive_wins,
        consecutive_losses=consecutive_losses,
        total_pnl=total_pnl,
        equity_curve=equity_curve,
    )
