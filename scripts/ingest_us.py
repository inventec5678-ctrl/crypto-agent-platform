#!/usr/bin/env python3
"""
US Stock Ingestion Script — yfinance
=====================================
Fetches daily OHLCV for US stocks via yfinance.

Usage:
    python3 ingest_us.py                    # incremental: last trading day
    python3 ingest_us.py --full             # full backfill from 2010-01-01
    python3 ingest_us.py --date 2026-04-15  # specific date

Crontab (UTC+8, daily at 04:30):
    30 4 * * * cd /Users/changrunlin/.openclaw/workspace/crypto-agent-platform && python3 scripts/ingest_us.py >> logs/ingest_us.log 2>&1
"""

import sys
import argparse
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
import yfinance as yf

# ── Config ────────────────────────────────────────────────────────────────────
US_SYMBOLS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B", "LLY",
    "AVGO", "JPM", "XOM", "UNH", "V", "MA", "PG", "HD", "CVX", "MRK", "ABBV",
    "PEP", "KO", "COST", "WMT", "CRM", "BAC", "TMO", "MCD", "CSCO", "ACN",
    "ABT", "DHR", "LIN", "NKE", "ADBE", "NEE", "TXN", "PM", "UNP", "RTX",
    "NFLX", "INTC", "VZ", "AMD", "QCOM", "HON", "ORCL", "BA", "IBM", "AMGN",
    "CAT", "GE", "SBUX", "GILD", "NOW", "MDLZ",
]

TIMEFRAMES = ["1d", "1w", "1mo"]
DATA_DIR   = Path("/Users/changrunlin/.openclaw/workspace/crypto-agent-platform/data/ohlcvutc/us")
LOG_DIR    = Path("/Users/changrunlin/.openclaw/workspace/crypto-agent-platform/logs")


def setup_logging():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(LOG_DIR / "ingest_us.log"),
            logging.StreamHandler(),
        ],
    )


def build_df(raw: pd.DataFrame, symbol: str, timeframe: str) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame()

    df = raw.copy()
    df.index = df.index.tz_convert("UTC") if df.index.tz else pd.DatetimeIndex(df.index, tz="UTC")
    df.index.name = "timestamp"

    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    result = pd.DataFrame({
        "open":   df["Open"],
        "high":   df["High"],
        "low":    df["Low"],
        "close":  df["Close"],
        "volume": df["Volume"].astype("float64"),
    })

    # adj_close from yfinance (only available for 1d)
    if "Adj Close" in df.columns:
        result["adj_close"] = pd.to_numeric(df["Adj Close"], errors="coerce").fillna(result["close"])
    else:
        result["adj_close"] = result["close"]

    result["dividends"]    = df["Dividends"] if "Dividends" in df.columns else 0.0
    result["stock_splits"] = df["Stock Splits"] if "Stock Splits" in df.columns else 0.0
    result["market"]        = "us"
    result["symbol"]        = symbol
    result["currency"]      = "USD"
    result["timeframe"]     = timeframe
    result["source"]        = "yfinance"
    result["fetched_at"]    = pd.Timestamp.now(tz="UTC")

    return result[[
        "open", "high", "low", "close", "volume",
        "adj_close", "dividends", "stock_splits",
        "market", "symbol", "currency", "timeframe", "source", "fetched_at",
    ]]


def append_or_replace(df_new: pd.DataFrame, path: Path) -> int:
    if df_new.empty:
        return 0
    if not path.exists():
        df_new.to_parquet(path)
        return len(df_new)
    df_existing = pd.read_parquet(path)
    df_existing = df_existing[~df_existing.index.isin(df_new.index)]
    df_combined = pd.concat([df_existing, df_new]).sort_index()
    df_combined.to_parquet(path)
    return len(df_new)


def fetch_symbol(symbol: str, start: pd.Timestamp, end: pd.Timestamp,
                 timeframe: str) -> pd.DataFrame:
    """Map our timeframe to yfinance interval."""
    interval_map = {"1d": "1d", "1w": "1wk", "1mo": "1mo"}
    interval = interval_map.get(timeframe, "1d")
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(start=start, end=end, interval=interval, auto_adjust=False)
        return build_df(hist, symbol, timeframe)
    except Exception as e:
        logging.warning("  [ERROR] %s %s: %s", symbol, timeframe, e)
        return pd.DataFrame()


def run(mode: str, specific_date: pd.Timestamp | None):
    today = pd.Timestamp.now(tz="UTC").normalize()

    if mode == "incremental":
        # Fetch just the last completed trading day
        start = today - pd.Timedelta(days=5)
        end   = today + pd.Timedelta(days=1)
    elif mode == "full":
        start = pd.Timestamp("2010-01-01", tz="UTC")
        end   = today + pd.Timedelta(days=1)
    elif mode == "date":
        start = specific_date
        end   = specific_date + pd.Timedelta(days=1)

    logging.info("US ingestion [%s] start=%s end=%s", mode, start.date(), end.date())

    total = 0
    for sym in US_SYMBOLS:
        for tf in TIMEFRAMES:
            path = DATA_DIR / f"{sym}_{tf}.parquet"
            df = fetch_symbol(sym, start, end, tf)
            if df.empty:
                logging.info("  [SKIP] %s %s: no data", sym, tf)
                continue
            n = append_or_replace(df, path)
            logging.info("  [OK] %s %s: +%d rows", sym, tf, n)
            total += n

    logging.info("US ingestion done. +%d total rows.", total)


def main():
    parser = argparse.ArgumentParser(description="US stock OHLCV ingestion")
    parser.add_argument("--full", action="store_true", help="Full backfill from 2010")
    parser.add_argument("--date", type=str, help="Specific date YYYY-MM-DD")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    setup_logging()

    if args.full:
        mode = "full"
        specific_date = None
    elif args.date:
        specific_date = pd.Timestamp(args.date, tz="UTC")
        mode = "date"
    else:
        mode = "incremental"
        specific_date = None

    logging.info("Starting US ingestion — mode=%s", mode)
    run(mode, specific_date)


if __name__ == "__main__":
    main()
