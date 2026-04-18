#!/usr/bin/env python3
"""
Crypto Ingestion Script — Binance Futures OHLCV
================================================
Fetches daily snapshots for 20 crypto symbols × 6 timeframes.

Usage:
    python3 ingest_crypto.py             # incremental: yesterday only
    python3 ingest_crypto.py --full       # full backfill from 2020-01-01
    python3 ingest_crypto.py --date 2026-04-15   # specific date

Crontab (UTC+8, daily at 03:00):
    0 3 * * * cd /Users/changrunlin/.openclaw/workspace/crypto-agent-platform && python3 scripts/ingest_crypto.py >> logs/ingest_crypto.log 2>&1
"""

import sys
import time
import argparse
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pandas as pd
import ccxt

# ── Config ────────────────────────────────────────────────────────────────────
SYMBOLS = [
    "ADAUSDT", "APTUSDT", "ARBUSDT", "ATOMUSDT", "AVAXUSDT",
    "BNBUSDT", "BTCUSDT", "DOGEUSDT", "DOTUSDT", "ETCUSDT",
    "ETHUSDT", "LINKUSDT", "LTCUSDT", "MATICUSDT", "NEARUSDT",
    "OPUSDT", "SOLUSDT", "UNIUSDT", "XLMUSDT", "XRPUSDT",
]
# TWSE is Mon–Fri only; crypto is 24/7 so skip weekends for cleaner logs
TIMEFRAMES = ["15m", "1h", "4h", "1d", "1w", "1mo"]
DATA_DIR   = Path("/Users/changrunlin/.openclaw/workspace/crypto-agent-platform/data/ohlcvutc/crypto")
LOG_DIR    = Path("/Users/changrunlin/.openclaw/workspace/crypto-agent-platform/logs")

LIMIT = 1000  # max candles per Binance request


def setup_logging():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(LOG_DIR / "ingest_crypto.log"),
            logging.StreamHandler(),
        ],
    )


def ccxt_symbol(sym: str) -> str:
    base = sym.replace("USDT", "")
    return f"{base}/USDT:USDT"


def to_ms(ts: pd.Timestamp) -> int:
    return int(ts.timestamp() * 1000)


def existing_range(path: Path) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    """Return (first_ts, last_ts) from an existing parquet, or (None, None)."""
    if not path.exists():
        return None, None
    df = pd.read_parquet(path, columns=[])  # cheap: just index
    if df.empty:
        return None, None
    return df.index.min(), df.index.max()


def build_df(ohlcv_list: list, symbol: str, timeframe: str) -> pd.DataFrame:
    if not ohlcv_list:
        return pd.DataFrame()
    df = pd.DataFrame(ohlcv_list,
                      columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df.set_index("timestamp", inplace=True)
    df.index.name = "timestamp"
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype("float64")
    df["adj_close"]   = df["close"].astype("float64")
    df["dividends"]    = 0.0
    df["stock_splits"] = 0.0
    df["market"]       = "crypto"
    df["symbol"]       = symbol
    df["currency"]     = "USDT"
    df["timeframe"]    = timeframe
    df["source"]       = "binance"
    df["fetched_at"]   = pd.Timestamp.now(timezone.utc)
    return df[[
        "open", "high", "low", "close", "volume",
        "adj_close", "dividends", "stock_splits",
        "market", "symbol", "currency", "timeframe", "source", "fetched_at",
    ]]


def fetch_since_for_mode(symbol: str, timeframe: str, mode: str,
                          specific_date: pd.Timestamp | None) -> int:
    """
    Returns start timestamp in ms.
    incremental : yesterday 00:00 UTC → today 00:00 UTC
    full        : 2020-01-01 00:00 UTC
    date        : that date 00:00 UTC → next day 00:00 UTC
    """
    now_utc = pd.Timestamp.now(tz="UTC")
    if mode == "incremental":
        yesterday = now_utc - timedelta(days=1)
        return to_ms(yesterday.replace(hour=0, minute=0, second=0, microsecond=0))
    elif mode == "full":
        return to_ms(pd.Timestamp("2020-01-01", tz="UTC"))
    elif mode == "date" and specific_date is not None:
        start = specific_date.tz_localize("UTC")
        return to_ms(start)
    return to_ms(now_utc - timedelta(days=1))


def fetch_end_for_mode(mode: str, specific_date: pd.Timestamp | None) -> int:
    now_utc = pd.Timestamp.now(tz="UTC")
    if mode == "incremental":
        today = now_utc.replace(hour=0, minute=0, second=0, microsecond=0)
        return to_ms(today)
    elif mode == "date" and specific_date is not None:
        end = specific_date + timedelta(days=1)
        return to_ms(end.tz_localize("UTC"))
    return to_ms(now_utc)


def fetch_timeframe(symbol: str, timeframe: str,
                     since_ms: int, end_ms: int) -> pd.DataFrame:
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
            current_since = last_ts + 1
            if last_ts >= end_ms:
                break
            time.sleep(0.05)
        except Exception as e:
            logging.warning("  [%s] %s %s: %s", symbol, timeframe, e)
            time.sleep(2)
            break

    return build_df(all_rows, symbol, timeframe)


def append_or_replace(df_new: pd.DataFrame, path: Path) -> int:
    """Write df_new to parquet: append if exists and overlapping, else replace."""
    if df_new.empty:
        return 0
    if not path.exists():
        df_new.to_parquet(path)
        return len(df_new)

    df_existing = pd.read_parquet(path)
    # Remove any rows that overlap with new data
    df_existing = df_existing[~df_existing.index.isin(df_new.index)]
    df_combined = pd.concat([df_existing, df_new]).sort_index()
    df_combined.to_parquet(path)
    return len(df_new)


def run(mode: str, specific_date: pd.Timestamp | None, symbols=None):
    symbols = symbols or SYMBOLS
    since_ms = fetch_since_for_mode("", "", mode, specific_date)
    end_ms   = fetch_end_for_mode(mode, specific_date)

    mode_desc = mode
    if mode == "date":
        mode_desc = f"date={specific_date.date()}"
    logging.info("ingest_crypto [%s] since=%s end=%s", mode_desc, since_ms, end_ms)

    total_rows = 0
    for sym in symbols:
        for tf in TIMEFRAMES:
            path = DATA_DIR / f"{sym}_{tf}.parquet"
            df = fetch_timeframe(sym, tf, since_ms, end_ms)
            if df.empty:
                logging.info("  [SKIP] %s %s: no new data", sym, tf)
                continue
            n = append_or_replace(df, path)
            logging.info("  [OK] %s %s: +%d rows", sym, tf, n)
            total_rows += n

    logging.info("ingest_crypto done. +%d total rows.", total_rows)


def main():
    parser = argparse.ArgumentParser(description="Crypto OHLCV ingestion")
    parser.add_argument("--full", action="store_true", help="Full backfill from 2020-01-01")
    parser.add_argument("--date", type=str, help="Specific date YYYY-MM-DD")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be fetched")
    args = parser.parse_args()

    setup_logging()
    if args.full:
        mode = "full"
        specific_date = None
    elif args.date:
        try:
            specific_date = pd.Timestamp(args.date, tz="UTC")
        except Exception:
            print("Invalid date format. Use YYYY-MM-DD")
            sys.exit(1)
        mode = "date"
    else:
        mode = "incremental"
        specific_date = None

    logging.info("Starting crypto ingestion — mode=%s", mode)
    run(mode, specific_date)


if __name__ == "__main__":
    main()
