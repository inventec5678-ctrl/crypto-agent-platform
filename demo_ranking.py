#!/usr/bin/env python3
"""
Demo script for Strategy Ranking Module.

Shows the system in action with simulated trades.
"""

import random
from datetime import datetime, timedelta
from strategy_ranking import (
    RankingService,
    TradeDirection,
    calculate_score,
    StrategyMetrics,
)


def simulate_trades(tracker, num_trades=20, initial_price=100.0):
    """Simulate random trades for a tracker."""
    entry_price = initial_price

    for i in range(num_trades):
        direction = random.choice([TradeDirection.LONG, TradeDirection.SHORT])
        quantity = random.uniform(0.1, 1.0)

        # Open trade
        trade_id = tracker.open_trade(
            entry_price=entry_price,
            quantity=quantity,
            direction=direction,
        )

        # Simulate price movement and close
        price_change = random.uniform(-0.05, 0.06)  # -5% to +6%
        exit_price = entry_price * (1 + price_change)

        # Higher win rate for demo (55%)
        if random.random() < 0.55:
            exit_price = entry_price * (1 + abs(price_change) * random.uniform(0.5, 1.5))

        commission = entry_price * quantity * 0.001
        pnl = tracker.close_trade(trade_id, exit_price, commission=commission)

        entry_price = exit_price

        pass  # trade closed


def main():
    print("=" * 60)
    print("Strategy Ranking Module Demo")
    print("=" * 60)

    # Create ranking service
    service = RankingService(initial_capital=10000.0)

    # Define strategies
    strategies = [
        "MA_Cross(10/30)",
        "RSI_Reversal(14)",
        "Bollinger_Breakout(20)",
        "MACD_Signal(12/26/9)",
        "Supertrend(10/3)",
    ]

    # Register strategies
    print("\n[1] Registering strategies...")
    for name in strategies:
        service.register_strategy(name)
        print(f"    ✓ {name}")

    # Simulate trades for each strategy
    print("\n[2] Simulating trades...")
    for name in strategies:
        tracker = service.get_tracker(name)
        # Different characteristics per strategy
        num_trades = random.randint(15, 30)
        simulate_trades(tracker, num_trades=num_trades)
        print(f"    ✓ {name}: {num_trades} trades")

    # Get rankings
    print("\n[3] Rankings:")
    print("-" * 60)
    rankings = service.get_rankings()

    for entry in rankings["rankings"]:
        print(f"  #{entry['rank']:2d} | {entry['strategy']:25s} | Score: {entry['score']:6.2f} | "
              f"Win: {entry['win_rate']*100:5.1f}% | PF: {entry['profit_factor']:5.2f} | "
              f"PnL: ${entry['total_pnl']:8.2f}")

    print(f"\n  Updated: {rankings['updated_at']}")

    # Show equity curve for top strategy
    print("\n[4] Equity Curve - Top Strategy:")
    print("-" * 60)
    top_strategy = rankings["rankings"][0]["strategy"]
    curve = service.get_equity_curve(top_strategy)

    print(f"  Strategy: {top_strategy}")
    print(f"  Equity curve points: {len(curve['equity_curve'])}")

    # Show first 10 and last 5 points
    print("\n  First 10 equity points:")
    for i, (eq, dd) in enumerate(zip(curve['equity_curve'][:10], curve['drawdown_curve'][:10])):
        print(f"    [{i:2d}] Equity: ${eq:10.2f} | Drawdown: {dd:6.2f}%")

    print(f"\n  ... ({len(curve['equity_curve'])-15} points) ...\n")

    print("  Last 5 equity points:")
    for i, (eq, dd) in enumerate(zip(curve['equity_curve'][-5:], curve['drawdown_curve'][-5:])):
        idx = len(curve['equity_curve']) - 5 + i
        print(f"    [{idx:2d}] Equity: ${eq:10.2f} | Drawdown: {dd:6.2f}%")

    # Show max drawdown
    metrics = service.get_strategy_metrics(top_strategy)
    print(f"\n  Max Drawdown: {metrics['max_drawdown']:.2f}%")
    print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.4f}")
    print(f"  Expectancy:   ${metrics['expectancy']:.2f} per trade")

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
