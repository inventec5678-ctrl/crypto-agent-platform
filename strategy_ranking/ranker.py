"""
Ranking service - provides ranking API and data.
"""

from datetime import datetime
from typing import Dict, List, Optional
from threading import Lock

from .models import RankingEntry, StrategyMetrics, EquityCurveData
from .tracker import MultiStrategyTracker
from .calculator import calculate_score


class RankingService:
    """
    Ranking service with REST API simulation.
    Provides GET /api/rankings and GET /api/rankings/{strategy}/equity_curve
    """

    def __init__(self, initial_capital: float = 10000.0):
        self._tracker = MultiStrategyTracker(initial_capital)
        self._lock = Lock()
        self._last_updated: Optional[datetime] = None

    # ─── Strategy Management ───────────────────────────────────────────────

    def register_strategy(self, strategy_name: str) -> None:
        """Register a new strategy."""
        self._tracker.register_strategy(strategy_name)
        self._update_timestamp()

    def get_tracker(self, strategy_name: str):
        """Get a strategy tracker."""
        return self._tracker.get_tracker(strategy_name)

    # ─── API: Rankings ──────────────────────────────────────────────────────

    def get_rankings(self) -> dict:
        """
        GET /api/rankings

        Returns ranked list of all strategies with scores.
        """
        with self._lock:
            all_metrics = self._tracker.get_all_metrics()

            # Build ranking entries
            entries: List[RankingEntry] = []
            for name, metrics in all_metrics.items():
                score = calculate_score(metrics)
                entry = RankingEntry(
                    rank=0,  # Will be set after sorting
                    strategy=name,
                    score=score,
                    total_trades=metrics.total_trades,
                    win_rate=metrics.win_rate,
                    profit_factor=metrics.profit_factor,
                    sharpe_ratio=metrics.sharpe_ratio,
                    max_drawdown=metrics.max_drawdown,
                    expectancy=metrics.expectancy,
                    total_pnl=metrics.total_pnl,
                )
                entries.append(entry)

            # Sort by score descending
            entries.sort(key=lambda e: e.score, reverse=True)

            # Assign ranks
            for i, entry in enumerate(entries):
                entry.rank = i + 1

            return {
                "rankings": [e.to_dict() for e in entries],
                "updated_at": self._last_updated.isoformat() if self._last_updated else None,
            }

    # ─── API: Equity Curve ─────────────────────────────────────────────────

    def get_equity_curve(self, strategy_name: str) -> Optional[dict]:
        """
        GET /api/rankings/{strategy}/equity_curve

        Returns equity curve data for a specific strategy.
        """
        with self._lock:
            tracker = self._tracker.get_tracker(strategy_name)
            if not tracker:
                return None

            data = tracker.get_equity_curve_data()
            return data.to_dict()

    # ─── API: Single Strategy Metrics ───────────────────────────────────────

    def get_strategy_metrics(self, strategy_name: str) -> Optional[dict]:
        """Get metrics for a single strategy."""
        with self._lock:
            tracker = self._tracker.get_tracker(strategy_name)
            if not tracker:
                return None
            metrics = tracker.get_metrics()
            return metrics.to_dict()

    # ─── Helpers ────────────────────────────────────────────────────────────

    def _update_timestamp(self) -> None:
        self._last_updated = datetime.now()

    @property
    def strategy_names(self) -> List[str]:
        return self._tracker.strategy_names


# ─── Singleton instance ──────────────────────────────────────────────────────

_service: Optional[RankingService] = None


def get_ranking_service(initial_capital: float = 10000.0) -> RankingService:
    """Get or create the global ranking service instance."""
    global _service
    if _service is None:
        _service = RankingService(initial_capital)
    return _service
