#!/usr/bin/env python3
"""
Download 1h and 4h OHLCV data for 20 crypto symbols from Binance Futures.
Usage:
    python3 download_1h_4h.py           # download all 20 symbols × (1h, 4h)
    python3 download_1h_4h.py BTCUSDT    # download only BTCUSDT
"""

import sys
import time
import pandas as pd
import ccxt
from datetime import datetime, timezone

# ── Config ──────────────────────────────────────────────────────────────────
SYMBOLS = [
    "ADAUSDT", "APTUSDT", "ARBUSDT", "ATOMUSDT", "AVAXUSDT",
    "BNBUSDT", "BTCUSDT", "DOGEUSDT", "DOTUSDT", "ETCUSDT",
    "ETHUSDT", "LINKUSDT", "LTCUSDT", "MATICUSDT", "NEARUSDT",
    "OPUSDT", "SOLUSDT", "UNIUSDT", "XLMUSDT", "XRPUSDT",
]
TIMEFRAMES = ["1h", "4h"]
SINCE = "2020-01-01T00:00:00Z"   # earliest start
END   = "2026-04-17T23:59:59Z"   # latest end (yesterday in UTC+8)
DATA_DIR = "/Users/changrunlin/.openclaw/workspace/crypto-agent-platform/data/ohlcvutc/crypto"

# Binance rate-limit: ~1200 requests / minute for public endpoints
# We stagger with 50 ms between calls
LIMIT = 1000  # max candles per request for binance


def ccxt_symbol(sym: str) -> str:
    """Convert our naming (BTCUSDT) → ccxt format (BTC/USDT:USDT)"""
    # Strip /USDT suffix and re-add as BTC/USDT:USDT for Binance futures
    base = sym.replace("USDT", "")
    return f"{base}/USDT:USDT"


def build_ohlcv_df(ohlcv_list: list, symbol: str, timeframe: str) -> pd.DataFrame:
    """Convert ccxt ohlcv list to a typed DataFrame matching our schema."""
    if not ohlcv_list:
        return pd.DataFrame()

    df = pd.DataFrame(
        ohlcv_list,
        columns=["timestamp", "open", "high", "low", "close", "volume"],
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df.set_index("timestamp", inplace=True)
    df.index.name = "timestamp"

    # Align types
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype("float64")

    # Extra columns (not provided by ccxt — mark as 0 / empty)
    df["adj_close"]    = df["close"].astype("float64")  # no adj close for crypto
    df["dividends"]    = 0.0
    df["stock_splits"]  = 0.0
    df["market"]        = "crypto"
    df["symbol"]        = symbol
    df["currency"]      = "USDT"
    df["timeframe"]     = timeframe
    df["source"]        = "binance"
    df["fetched_at"]    = pd.Timestamp.now(timezone.utc).tz_convert("UTC")

    return df[[
        "open", "high", "low", "close", "volume",
        "adj_close", "dividends", "stock_splits",
        "market", "symbol", "currency", "timeframe", "source", "fetched_at",
    ]]


def download_timeframe(symbol: str, timeframe: str,
                       since_ms: int, end_ms: int) -> pd.DataFrame:
    """Fetch all candles for one symbol/timeframe in chunks."""
    exchange = ccxt.binance({"enableRateLimit": False})
    ccym = ccxt_symbol(symbol)

    all_rows = []
    current_since = since_ms

    while current_since < end_ms:
        try:
            ohlcv = exchange.fetch_ohlcv(ccym, timeframe, current_since, LIMIT)
            if not ohlcv:
                break
            all_rows.extend(ohlcv)
            last_ts = ohlcv[-1][0]
            # Binance is inclusive on the since side, advance by 1ms to avoid dupes
            current_since = last_ts + 1
            if last_ts >= end_ms:
                break
            # Stagger to be polite
            time.sleep(0.05)
        except Exception as e:
            print(f"  [ERROR] {symbol} {timeframe}: {e}")
            time.sleep(2)
            break

    if not all_rows:
        return pd.DataFrame()

    df = build_ohlcv_df(all_rows, symbol, timeframe)
    # Filter to our requested window
    end_dt = pd.Timestamp(end_ms, unit="ms", tz="UTC")
    df = df[df.index <= end_dt]
    return df


def save_parquet(df: pd.DataFrame, symbol: str, timeframe: str):
    if df.empty:
        print(f"  [SKIP] {symbol}_{timeframe}: no data")
        return
    path = f"{DATA_DIR}/{symbol}_{timeframe}.parquet"
    df.to_parquet(path)
    print(f"  [SAVE] {symbol}_{timeframe}  →  {len(df)} rows  "
          f"({df.index.min().date()} to {df.index.max().date()})")


def main():
    symbols = sys.argv[1:] if len(sys.argv) > 1 else SYMBOLS

    since_ms = int(pd.Timestamp(SINCE, tz="UTC").timestamp() * 1000)
    end_ms   = int(pd.Timestamp(END,   tz="UTC").timestamp() * 1000)

    print(f"Downloading 1h & 4h  ({SINCE} → {END})")
    print(f"Output: {DATA_DIR}")
    print(f"Symbols: {symbols}")
    print("─" * 60)

    for sym in symbols:
        if sym not in SYMBOLS:
            print(f"[WARN] Unknown symbol: {sym}, skipping")
            continue
        for tf in TIMEFRAMES:
            print(f"  {sym} {tf}...", end=" ", flush=True)
            df = download_timeframe(sym, tf, since_ms, end_ms)
            save_parquet(df, sym, tf)

    print("Done.")


if __name__ == "__main__":
    main()
