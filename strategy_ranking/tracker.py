"""
Strategy tracker - manages trades and metrics for individual strategies.
"""

import uuid
from datetime import datetime
from typing import Dict, List, Optional
from threading import Lock

from .models import Trade, TradeDirection, StrategyMetrics, EquityCurveData
from .calculator import (
    build_metrics,
    calculate_drawdown_curve,
    calculate_score,
)


class StrategyTracker:
    """Tracks trades and performance for a single strategy."""

    def __init__(self, strategy_name: str, initial_capital: float = 10000.0):
        self.strategy_name = strategy_name
        self.initial_capital = initial_capital
        self._trades: List[Trade] = []
        self._open_trades: Dict[str, Trade] = {}  # trade_id -> Trade
        self._lock = Lock()

    def open_trade(
        self,
        entry_price: float,
        quantity: float,
        direction: TradeDirection,
        entry_time: Optional[datetime] = None,
    ) -> str:
        """Open a new trade and return its ID."""
        with self._lock:
            trade_id = str(uuid.uuid4())
            trade = Trade(
                trade_id=trade_id,
                strategy_name=self.strategy_name,
                entry_time=entry_time or datetime.now(),
                exit_time=None,
                entry_price=entry_price,
                exit_price=None,
                quantity=quantity,
                direction=direction,
            )
            self._open_trades[trade_id] = trade
            return trade_id

    def close_trade(
        self,
        trade_id: str,
        exit_price: float,
        exit_time: Optional[datetime] = None,
        commission: float = 0.0,
    ) -> Optional[float]:
        """Close an open trade and return PnL."""
        with self._lock:
            if trade_id not in self._open_trades:
                return None

            trade = self._open_trades.pop(trade_id)
            trade.close(exit_price, exit_time or datetime.now(), commission)
            self._trades.append(trade)
            return trade.pnl

    def get_metrics(self) -> StrategyMetrics:
        """Calculate and return current metrics."""
        with self._lock:
            return build_metrics(self.strategy_name, self._trades.copy(), self.initial_capital)

    def get_equity_curve_data(self) -> EquityCurveData:
        """Get equity curve data for charting."""
        with self._lock:
            metrics = build_metrics(self.strategy_name, self._trades.copy(), self.initial_capital)
            drawdown_curve = calculate_drawdown_curve(metrics.equity_curve)

            trade_points = []
            for i, trade in enumerate(self._trades):
                trade_points.append({
                    "index": i + 1,  # equity_curve[0] is initial, index 1 is first trade
                    "equity": round(metrics.equity_curve[i + 1], 2),
                    "drawdown": round(drawdown_curve[i + 1], 4) if i + 1 < len(drawdown_curve) else 0,
                    "trade_id": trade.trade_id,
                    "pnl": round(trade.pnl, 2) if trade.pnl else 0,
                    "is_winner": trade.is_winner,
                })

            return EquityCurveData(
                equity_curve=metrics.equity_curve,
                drawdown_curve=drawdown_curve,
                trade_points=trade_points,
            )

    @property
    def open_trade_count(self) -> int:
        return len(self._open_trades)

    @property
    def total_trades(self) -> int:
        return len(self._trades)


class MultiStrategyTracker:
    """Manages multiple strategy trackers simultaneously."""

    def __init__(self, initial_capital: float = 10000.0):
        self.initial_capital = initial_capital
        self._trackers: Dict[str, StrategyTracker] = {}
        self._lock = Lock()

    def register_strategy(self, strategy_name: str) -> StrategyTracker:
        """Register a new strategy tracker."""
        with self._lock:
            if strategy_name not in self._trackers:
                self._trackers[strategy_name] = StrategyTracker(
                    strategy_name, self.initial_capital
                )
            return self._trackers[strategy_name]

    def get_tracker(self, strategy_name: str) -> Optional[StrategyTracker]:
        """Get a strategy tracker by name."""
        with self._lock:
            return self._trackers.get(strategy_name)

    def get_all_metrics(self) -> Dict[str, StrategyMetrics]:
        """Get metrics for all strategies."""
        with self._lock:
            return {name: tracker.get_metrics() for name, tracker in self._trackers.items()}

    def get_all_equity_curves(self) -> Dict[str, EquityCurveData]:
        """Get equity curve data for all strategies."""
        with self._lock:
            return {name: tracker.get_equity_curve_data() for name, tracker in self._trackers.items()}

    @property
    def strategy_names(self) -> List[str]:
        return list(self._trackers.keys())
