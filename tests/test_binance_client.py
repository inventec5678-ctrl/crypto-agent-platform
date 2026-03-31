# tests/test_binance_client.py
import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from binance_client import (
    get_btc_price,
    get_klines,
    get_coin_price,
    get_binance_time,
)


def test_get_btc_price():
    """BTC price should be positive number"""
    price = get_btc_price()
    assert isinstance(price, (int, float)), f"Price should be numeric, got {type(price)}"
    assert price > 0, f"BTC price should be positive, got {price}"
    assert price > 1000, f"BTC price seems too low: {price}"


def test_get_coin_price():
    """get_coin_price returns price for given symbol"""
    price = get_coin_price("ETHUSDT")
    assert isinstance(price, (int, float)), f"Price should be numeric, got {type(price)}"
    assert price > 0, f"ETH price should be positive, got {price}"


def test_get_klines_returns_list():
    """get_klines returns a list"""
    klines = get_klines("BTCUSDT", "4h", limit=10)
    assert isinstance(klines, list), f"Should return list, got {type(klines)}"
    assert len(klines) > 0, "Should return at least one kline"


def test_get_klines_has_required_fields():
    """Klines should have all required OHLCV fields"""
    klines = get_klines("BTCUSDT", "4h", limit=5)
    required = ["open_time", "close_time", "open", "high", "low", "close", "volume"]
    for k in klines:
        for field in required:
            assert field in k, f"Kline missing field: {field}"


def test_get_klines_respects_limit():
    """get_klines should return at most limit candles"""
    klines = get_klines("BTCUSDT", "4h", limit=5)
    assert len(klines) <= 5, f"Expected <= 5 klines, got {len(klines)}"


def test_get_binance_time():
    """Binance server time should be a valid timestamp"""
    ts = get_binance_time()
    assert isinstance(ts, int), f"Timestamp should be int, got {type(ts)}"
    assert ts > 1e12, f"Timestamp seems too small: {ts}"


def test_get_klines_close_price_positive():
    """All close prices should be positive"""
    klines = get_klines("BTCUSDT", "4h", limit=20)
    for k in klines:
        assert float(k["close"]) > 0, f"Close price should be positive: {k['close']}"


def test_get_klines_high_ge_low():
    """High price should be >= low price for each candle"""
    klines = get_klines("BTCUSDT", "4h", limit=10)
    for k in klines:
        high = float(k["high"])
        low = float(k["low"])
        assert high >= low, f"High ({high}) < Low ({low})"
