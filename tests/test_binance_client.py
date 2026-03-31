# tests/test_binance_client.py
import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import asyncio
from binance_client import get_btc_price, get_klines, get_symbol_price


def test_get_btc_price():
    """BTC price should be positive number"""
    price = get_btc_price()
    assert isinstance(price, (int, float)), f"Price should be numeric, got {type(price)}"
    assert price > 0, f"BTC price should be positive, got {price}"
    assert price > 1000, f"BTC price seems too low: {price}"


def test_get_symbol_price():
    """get_symbol_price returns price for given symbol"""
    async def _test():
        price = await get_symbol_price("ETHUSDT")
        return price
    price = asyncio.run(_test())
    assert isinstance(price, (int, float)), f"Price should be numeric, got {type(price)}"
    assert price > 0, f"ETH price should be positive, got {price}"


def test_get_klines_returns_list():
    """get_klines returns a list"""
    async def _test():
        klines = await get_klines("BTCUSDT", "4h", limit=10)
        return klines
    klines = asyncio.run(_test())
    assert isinstance(klines, list), f"Should return list, got {type(klines)}"
    assert len(klines) > 0, "Should return at least one kline"


def test_get_klines_has_required_fields():
    """Klines should have all required OHLCV fields"""
    async def _test():
        klines = await get_klines("BTCUSDT", "4h", limit=5)
        return klines
    klines = asyncio.run(_test())
    required = ["open_time", "close_time", "open", "high", "low", "close", "volume"]
    for k in klines:
        for field in required:
            assert field in k, f"Kline missing field: {field}"


def test_get_klines_respects_limit():
    """get_klines should return at most limit candles"""
    async def _test():
        klines = await get_klines("BTCUSDT", "4h", limit=5)
        return klines
    klines = asyncio.run(_test())
    assert len(klines) <= 5, f"Expected <= 5 klines, got {len(klines)}"


def test_get_klines_close_price_positive():
    """All close prices should be positive"""
    async def _test():
        klines = await get_klines("BTCUSDT", "4h", limit=20)
        return klines
    klines = asyncio.run(_test())
    for k in klines:
        assert float(k["close"]) > 0, f"Close price should be positive: {k['close']}"


def test_get_klines_high_ge_low():
    """High price should be >= low price for each candle"""
    async def _test():
        klines = await get_klines("BTCUSDT", "4h", limit=10)
        return klines
    klines = asyncio.run(_test())
    for k in klines:
        high = float(k["high"])
        low = float(k["low"])
        assert high >= low, f"High ({high}) < Low ({low})"
