# tests/test_api.py
import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_live_strategies_returns_valid_json():
    """Test /api/strategies/live endpoint returns correct structure"""
    response = client.get("/api/strategies/live")
    assert response.status_code == 200
    data = response.json()
    assert "strategies" in data, "Missing 'strategies' key"
    assert "btc_price" in data, "Missing 'btc_price' key"
    assert "indicators" in data, "Missing 'indicators' key"
    assert isinstance(data["strategies"], list), "strategies should be a list"


def test_klines_returns_candles():
    """Test /api/klines endpoint returns candle data"""
    response = client.get("/api/klines?symbol=BTC&interval=4h&limit=100")
    assert response.status_code == 200
    candles = response.json()
    assert len(candles) > 0, "Should return at least one candle"
    assert "time" in candles[0], "Candle missing 'time' field"
    assert "close" in candles[0], "Candle missing 'close' field"
    assert "open" in candles[0], "Candle missing 'open' field"


def test_klines_respects_limit():
    """Test klines limit parameter"""
    response = client.get("/api/klines?symbol=BTC&interval=4h&limit=10")
    assert response.status_code == 200
    candles = response.json()
    assert len(candles) <= 10, f"Expected <= 10 candles, got {len(candles)}"


def test_anomalies_endpoint():
    """Test /api/dashboard/anomalies endpoint"""
    response = client.get("/api/dashboard/anomalies")
    assert response.status_code == 200
    data = response.json()
    assert "volume_anomalies" in data, "Missing 'volume_anomalies'"
    assert "ob_anomalies" in data, "Missing 'ob_anomalies'"
    assert "vol_count" in data, "Missing 'vol_count'"
    assert "ob_count" in data, "Missing 'ob_count'"


def test_health_endpoint():
    """Test /api/health endpoint"""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data


def test_live_strategies_has_required_fields():
    """Test strategy items have required fields"""
    response = client.get("/api/strategies/live")
    data = response.json()
    strategies = data.get("strategies", [])
    if len(strategies) > 0:
        s = strategies[0]
        required = ["name", "direction", "match_pct", "win_rate",
                    "profit_factor", "max_drawdown", "entry_price",
                    "take_profit", "stop_loss"]
        for field in required:
            assert field in s, f"Strategy missing field: {field}"
