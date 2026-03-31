"""
Pytest 配置和共享 fixtures
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def mock_binance_client():
    """Mock Binance client for testing"""
    from unittest.mock import AsyncMock, MagicMock
    from datetime import datetime

    mock = MagicMock()
    mock.get_symbol_price = AsyncMock(return_value=50000.0)
    mock.get_klines = AsyncMock(return_value=[
        {
            "open_time": datetime(2024, 1, 1, 0, 0, 0),
            "open": "50000",
            "high": "51000",
            "low": "49000",
            "close": "50500",
            "volume": "1000",
            "close_time": datetime(2024, 1, 1, 0, 59, 59),
        }
    ])
    mock.get_order_book = AsyncMock(return_value={
        "bids_total": 100.0,
        "asks_total": 100.0,
        "spread": 10.0,
    })
    mock._make_request = AsyncMock(return_value={})
    return mock


@pytest.fixture
def sample_klines():
    """Sample K-line data for testing"""
    return [
        {
            "open_time": 1700000000000,
            "open": "50000.00",
            "high": "50500.00",
            "low": "49500.00",
            "close": "50200.00",
            "volume": "1000.00",
            "close_time": 1700003599999,
        },
        {
            "open_time": 1700003600000,
            "open": "50200.00",
            "high": "51000.00",
            "low": "50000.00",
            "close": "50800.00",
            "volume": "1200.00",
            "close_time": 1700007199999,
        },
    ]


@pytest.fixture
def sample_signal_data():
    """Sample signal data for testing"""
    return {
        "symbol": "BTCUSDT",
        "signal": "BUY",
        "direction": "LONG",
        "price": 50000.0,
        "strategy": "MA_Cross",
        "confidence": 75.0,
        "ai_confidence": 72.5,
        "ai_rating": "★★",
        "timestamp": "2024-01-01T00:00:00",
    }


@pytest.fixture
def sample_ai_analysis():
    """Sample AI analysis result for testing"""
    return {
        "symbol": "BTCUSDT",
        "confidence_score": 72.5,
        "rating": "★★",
        "factors": {
            "technical": 75.0,
            "sentiment": 70.0,
            "social": 65.0,
            "anomaly": 80.0,
        },
        "weights": {
            "technical": "35%",
            "sentiment": "25%",
            "social": "20%",
            "anomaly": "20%",
        },
        "strategies_triggered": 2,
        "timestamp": "2024-01-01T00:00:00",
    }


@pytest.fixture
def mock_notification_manager():
    """Mock notification manager"""
    from unittest.mock import MagicMock, AsyncMock

    mock = MagicMock()
    mock.send_signal = AsyncMock(return_value=True)
    mock.send_status = AsyncMock(return_value=True)
    return mock
