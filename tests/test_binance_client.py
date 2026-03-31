# tests/test_binance_client.py
"""
Unit tests for binance_client BinanceClient class.
Note: These tests make real API calls to Binance.
"""
import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import asyncio
from binance_client import BinanceClient


@pytest.fixture
def client():
    return BinanceClient()


@pytest.mark.asyncio
async def test_get_symbol_price_btc(client):
    """BTC price should be positive and realistic"""
    price = await client.get_symbol_price("BTCUSDT")
    assert isinstance(price, (int, float)), f"Price should be numeric, got {type(price)}"
    assert price > 0, f"BTC price should be positive, got {price}"
    assert price > 1000, f"BTC price seems too low: {price}"


@pytest.mark.asyncio
async def test_get_symbol_price_eth(client):
    """ETH price should be positive and realistic"""
    price = await client.get_symbol_price("ETHUSDT")
    assert isinstance(price, (int, float))
    assert price > 0


@pytest.mark.asyncio
async def test_get_klines_returns_list(client):
    """get_klines should return a list of candles"""
    klines = await client.get_klines("BTCUSDT", "4h", limit=10)
    assert isinstance(klines, list), f"Should return list, got {type(klines)}"
    assert len(klines) > 0, "Should return at least one candle"


@pytest.mark.asyncio
async def test_get_klines_has_required_fields(client):
    """Klines should have all OHLCV fields"""
    klines = await client.get_klines("BTCUSDT", "4h", limit=5)
    required = ["open_time", "close_time", "open", "high", "low", "close", "volume"]
    for k in klines:
        for field in required:
            assert field in k, f"Kline missing field: {field}"


@pytest.mark.asyncio
async def test_get_klines_respects_limit(client):
    """get_klines should respect the limit parameter"""
    klines = await client.get_klines("BTCUSDT", "4h", limit=5)
    assert len(klines) <= 5, f"Expected <= 5 klines, got {len(klines)}"


@pytest.mark.asyncio
async def test_get_klines_high_ge_low(client):
    """High price should be >= low price for each candle"""
    klines = await client.get_klines("BTCUSDT", "4h", limit=10)
    for k in klines:
        high = float(k["high"])
        low = float(k["low"])
        assert high >= low, f"High ({high}) < Low ({low})"


@pytest.mark.asyncio
async def test_get_klines_close_positive(client):
    """All close prices should be positive"""
    klines = await client.get_klines("BTCUSDT", "4h", limit=20)
    for k in klines:
        assert float(k["close"]) > 0


@pytest.mark.asyncio
async def test_get_klines_1d_interval(client):
    """get_klines should work with 1d interval"""
    klines = await client.get_klines("BTCUSDT", "1d", limit=5)
    assert isinstance(klines, list)
    assert len(klines) > 0
