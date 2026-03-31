"""
FastAPI integration for strategy ranking.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime

from .ranker import get_ranking_service
from .models import TradeDirection


app = FastAPI(title="Strategy Ranking API", version="1.0.0")
_rankings = get_ranking_service()


# ─── Request Models ─────────────────────────────────────────────────────────

class OpenTradeRequest(BaseModel):
    strategy_name: str
    entry_price: float
    quantity: float
    direction: str  # "long" or "short"


class CloseTradeRequest(BaseModel):
    strategy_name: str
    trade_id: str
    exit_price: float
    commission: float = 0.0


class RegisterStrategyRequest(BaseModel):
    strategy_name: str


# ─── API Routes ──────────────────────────────────────────────────────────────

@app.get("/api/rankings")
def get_rankings():
    """Get ranked list of all strategies."""
    return _rankings.get_rankings()


@app.get("/api/rankings/{strategy}/equity_curve")
def get_equity_curve(strategy: str):
    """Get equity curve data for a specific strategy."""
    data = _rankings.get_equity_curve(strategy)
    if data is None:
        raise HTTPException(status_code=404, detail=f"Strategy '{strategy}' not found")
    return data


@app.get("/api/rankings/{strategy}/metrics")
def get_strategy_metrics(strategy: str):
    """Get detailed metrics for a specific strategy."""
    data = _rankings.get_strategy_metrics(strategy)
    if data is None:
        raise HTTPException(status_code=404, detail=f"Strategy '{strategy}' not found")
    return data


@app.post("/api/strategies")
def register_strategy(req: RegisterStrategyRequest):
    """Register a new strategy."""
    _rankings.register_strategy(req.strategy_name)
    return {"status": "ok", "strategy_name": req.strategy_name}


@app.post("/api/trades/open")
def open_trade(req: OpenTradeRequest):
    """Open a new trade for a strategy."""
    tracker = _rankings.get_tracker(req.strategy_name)
    if not tracker:
        raise HTTPException(status_code=404, detail=f"Strategy '{req.strategy_name}' not found")

    direction = TradeDirection.LONG if req.direction.lower() == "long" else TradeDirection.SHORT
    trade_id = tracker.open_trade(
        entry_price=req.entry_price,
        quantity=req.quantity,
        direction=direction,
    )
    return {"trade_id": trade_id, "strategy_name": req.strategy_name}


@app.post("/api/trades/close")
def close_trade(req: CloseTradeRequest):
    """Close an open trade for a strategy."""
    tracker = _rankings.get_tracker(req.strategy_name)
    if not tracker:
        raise HTTPException(status_code=404, detail=f"Strategy '{req.strategy_name}' not found")
    
    pnl = tracker.close_trade(
        trade_id=req.trade_id,
        exit_price=req.exit_price,
        commission=req.commission,
    )
    
    if pnl is None:
        raise HTTPException(status_code=404, detail=f"Trade '{req.trade_id}' not found")
    
    return {"trade_id": req.trade_id, "strategy_name": req.strategy_name, "pnl": pnl}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
