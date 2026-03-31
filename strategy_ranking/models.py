"""
Data models for strategy ranking.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional
from enum import Enum


class TradeDirection(Enum):
    LONG = "long"
    SHORT = "short"


@dataclass
class Trade:
    """Individual trade record."""
    trade_id: str
    strategy_name: str
    entry_time: datetime
    exit_time: Optional[datetime]
    entry_price: float
    exit_price: Optional[float]
    quantity: float
    direction: TradeDirection
    pnl: Optional[float] = None  # Calculated on close
    commission: float = 0.0

    def close(self, exit_price: float, exit_time: datetime, commission: float = 0.0) -> float:
        """Close the trade and calculate PnL."""
        self.exit_price = exit_price
        self.exit_time = exit_time
        self.commission = commission

        if self.direction == TradeDirection.LONG:
            self.pnl = (exit_price - self.entry_price) * self.quantity - commission
        else:
            self.pnl = (self.entry_price - exit_price) * self.quantity - commission
        return self.pnl

    @property
    def is_winner(self) -> bool:
        return self.pnl is not None and self.pnl > 0


@dataclass
class StrategyMetrics:
    """Performance metrics for a single strategy."""
    strategy_name: str
    total_trades: int = 0
    win_rate: float = 0.0          # 勝率
    avg_profit: float = 0.0        # 平均獲利
    avg_loss: float = 0.0          # 平均虧損
    profit_factor: float = 0.0     # 盈虧比 = avg_profit / avg_loss
    sharpe_ratio: float = 0.0      # 夏普值
    max_drawdown: float = 0.0      # 最大回撤
    expectancy: float = 0.0       # 期望值
    consecutive_wins: int = 0      # 最大連勝
    consecutive_losses: int = 0    # 最大連虧
    total_pnl: float = 0.0         # 總損益
    equity_curve: List[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "strategy_name": self.strategy_name,
            "total_trades": self.total_trades,
            "win_rate": round(self.win_rate, 4),
            "avg_profit": round(self.avg_profit, 4),
            "avg_loss": round(self.avg_loss, 4),
            "profit_factor": round(self.profit_factor, 4),
            "sharpe_ratio": round(self.sharpe_ratio, 4),
            "max_drawdown": round(self.max_drawdown, 4),
            "expectancy": round(self.expectancy, 4),
            "consecutive_wins": self.consecutive_wins,
            "consecutive_losses": self.consecutive_losses,
            "total_pnl": round(self.total_pnl, 4),
        }


@dataclass
class RankingEntry:
    """Single entry in the ranking list."""
    rank: int
    strategy: str
    score: float
    total_trades: int
    win_rate: float
    profit_factor: float
    sharpe_ratio: float
    max_drawdown: float
    expectancy: float
    total_pnl: float

    def to_dict(self) -> dict:
        return {
            "rank": self.rank,
            "strategy": self.strategy,
            "score": round(self.score, 2),
            "total_trades": self.total_trades,
            "win_rate": round(self.win_rate, 4),
            "profit_factor": round(self.profit_factor, 4),
            "sharpe_ratio": round(self.sharpe_ratio, 4),
            "max_drawdown": round(self.max_drawdown, 4),
            "expectancy": round(self.expectancy, 4),
            "total_pnl": round(self.total_pnl, 2),
        }


@dataclass
class EquityCurveData:
    """Equity curve and drawdown data for charting."""
    equity_curve: List[float]
    drawdown_curve: List[float]
    trade_points: List[dict]  # [{index, equity, drawdown, trade_id}]

    def to_dict(self) -> dict:
        return {
            "equity_curve": [round(e, 2) for e in self.equity_curve],
            "drawdown_curve": [round(d, 4) for d in self.drawdown_curve],
            "trade_points": self.trade_points,
        }
