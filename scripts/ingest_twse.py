#!/usr/bin/env python3
"""
TWSE Ingestion Script — Taiwan Stock Exchange
==============================================
Fetches daily OHLCV for TWSE stocks via TWSE open API.

Usage:
    python3 ingest_twse.py                  # incremental: latest trading day
    python3 ingest_twse.py --full           # full backfill from 2010-01-01
    python3 ingest_twse.py --date 2026-04-15  # specific date

Crontab (UTC+8, daily at 04:00):
    0 4 * * * cd /Users/changrunlin/.openclaw/workspace/crypto-agent-platform && python3 scripts/ingest_twse.py >> logs/ingest_twse.log 2>&1
"""

import sys
import time
import json
import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path

import requests
import pandas as pd

# ── Config ────────────────────────────────────────────────────────────────────
# 59 TWSE symbols (common active stocks)
TWSE_SYMBOLS = [
    "0050", "0051", "0052", "0053", "0054", "0055", "0056", "0057", "0058", "0059",
    "1101", "1102", "1216", "1301", "1303", "1326", "1402", "1476", "1605", "1707",
    "1717", "1722", "1802", "1907", "2002", "2105", "2201", "2207", "2301", "2303",
    "2308", "2311", "2317", "2327", "2330", "2347", "2353", "2354", "2357", "2382",
    "2409", "2412", "2449", "2454", "2474", "2492", "2498", "2603", "2607", "2609",
    "2610", "2615", "2618", "2801", "2823", "2880", "2881", "2883", "2884", "2885",
    "2890", "2891", "2892", "2912", "3008", "3034", "3037", "3045", "3090", "3130",
    "3189", "3231", "3293", "3406", "3481", "3504", "3530", "3533", "3557", "3576",
    "3583", "3596", "3661", "3665", "3673", "3682", "3698", "3702", "3711", "3712",
    "4880", "4903", "4904", "4938", "4943", "4952", "4958", "4967", "4974", "4984",
    "4994", "5225", "5347", "5388", "5434", "5522", "5525", "5607", "5871", "5876",
    "5880", "5904", "6024", "6116", "6139", "6183", "6191", "6213", "6225", "6231",
    "6235", "6257", "6269", "6271", "6283", "6405", "6412", "6414", "6415", "6438",
    "6443", "6451", "6456", "6464", "6477", "6504", "6515", "6525", "6531", "6533",
    "6550", "6552", "6569", "6573", "6581", "6589", "6590", "6592", "6629", "6655",
    "6668", "6670", "6683", "6706", "6770", "8016", "8028", "8039", "8046", "8081",
    "8105", "8131", "8210", "8215", "8234", "8255", "8261", "8271", "8285", "8299",
    "8341", "8406", "8410", "8415", "8422", "8432", "8454", "8462", "8463", "8478",
    "8482", "8497", "9904", "9910", "9914", "9917", "9921", "9933", "9939", "9955",
]

TIMEFRAMES = ["1d", "1w", "1mo"]
DATA_DIR   = Path("/Users/changrunlin/.openclaw/workspace/crypto-agent-platform/data/ohlcvutc/twse")
LOG_DIR    = Path("/Users/changrunlin/.openclaw/workspace/crypto-agent-platform/logs")
TWSE_API   = "https://openapi.twse.com.tw/v1/exchangeReport/MI_5MINS_H"
BWIBBU_API = "https://www.twse.com.tw/rwd/en/aftertrading/BWIBBU"


def setup_logging():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(LOG_DIR / "ingest_twse.log"),
            logging.StreamHandler(),
        ],
    )


def is_trading_day(dt: pd.Timestamp) -> bool:
    """TWSE trading days: Mon–Fri, no major holidays (simplified)."""
    return dt.weekday() < 5  # Mon=0 … Fri=4


def fetch_mi_5mins_h(date: pd.Timestamp) -> dict:
    """Fetch intraday 5-min bars for all TWSE stocks for a given date."""
    url = f"{TWSE_API}?date={date.strftime('%Y%m%d')}"
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logging.warning("MI_5MINS_H API error for %s: %s", date.date(), e)
        return {}


def fetch_bwibbu(date: pd.Timestamp) -> dict:
    """Fetch TWSE bulk BWIBBU (P/E, dividend yield) for a given date."""
    url = f"{BWIBBU_API}?date={date.strftime('%Y%m%d')}&response=json"
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logging.warning("BWIBBU API error for %s: %s", date.date(), e)
        return {}


def convert_5min_to_daily(records: list, symbol: str, date: pd.Timestamp) -> pd.DataFrame:
    """Convert 5-min intraday bars → daily OHLCV. Returns 1-row DataFrame."""
    if not records:
        return pd.DataFrame()

    rows = []
    for r in records:
        try:
            ts_str = r.get("Time") or r.get("time")
            if ts_str is None:
                continue
            # Format: "09:00" or "09:00:00"
            h, m = map(int, ts_str.split(":")[:2])
            dt = date.replace(hour=h, minute=m, second=0, microsecond=0)
            rows.append({
                "timestamp": dt,
                "open":  float(r.get("Open", r.get("open", 0))),
                "high":  float(r.get("High", r.get("high", 0))),
                "low":   float(r.get("Low",  r.get("low",  0))),
                "close": float(r.get("Close", r.get("close", 0))),
                "volume": float(r.get("Volume", r.get("volume", 0))),
            })
        except (ValueError, TypeError, KeyError):
            continue

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).set_index("timestamp").sort_index()
    # Aggregate to daily OHLCV
    daily = pd.DataFrame({
        "open":   [df["open"].iloc[0]],
        "high":   [df["high"].max()],
        "low":    [df["low"].min()],
        "close":  [df["close"].iloc[-1]],
        "volume": [df["volume"].sum()],
    }, index=[date.normalize()])
    daily.index = daily.index.tz_localize("UTC")
    return daily


def build_df(rows_df: pd.DataFrame, symbol: str, timeframe: str) -> pd.DataFrame:
    if rows_df.empty:
        return pd.DataFrame()
    df = rows_df.copy()
    df["adj_close"]   = df["close"].astype("float64")
    df["dividends"]   = 0.0
    df["stock_splits"]= 0.0
    df["market"]      = "twse"
    df["symbol"]      = symbol
    df["currency"]    = "TWD"
    df["timeframe"]   = timeframe
    df["source"]      = "twse"
    df["fetched_at"]  = pd.Timestamp.now(tz="UTC")
    return df[[
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


def ingest_date(date: pd.Timestamp, dry_run: bool = False):
    if not is_trading_day(date):
        logging.info("Skipping %s (not a trading day)", date.date())
        return

    logging.info("Ingesting TWSE for %s", date.date())
    data = fetch_mi_5mins_h(date)

    total = 0
    for record in data:
        sym = record.get("Code", record.get("code", ""))
        if not isinstance(record, dict):
            continue
        rows_df = convert_5min_to_daily(
            record.get("data", record.get("data", [])),
            sym, date
        )
        if rows_df.empty:
            continue
        for tf in ["1d"]:
            df = build_df(rows_df, sym, tf)
            path = DATA_DIR / f"{sym}_{tf}.parquet"
            if dry_run:
                logging.info("  [DRY] %s %s: would write %d rows", sym, tf, len(df))
            else:
                n = append_or_replace(df, path)
                logging.info("  [OK] %s %s: +%d rows", sym, tf, n)
                total += n

    logging.info("TWSE %s: %d rows written.", date.date(), total)


def run(mode: str, specific_date: pd.Timestamp | None):
    if mode == "incremental":
        # TWSE market closes at 13:30 Taiwan time (05:30 UTC)
        # Safe to ingest after 04:00 UTC+8 (= 20:00 UTC previous day)
        today = pd.Timestamp.now(tz="Asia/Taipei").normalize()
        ingest_date(today - pd.Timedelta(days=1))
    elif mode == "full":
        start = pd.Timestamp("2010-01-01", tz="UTC").tz_convert("Asia/Taipei")
        end   = pd.Timestamp.now(tz="Asia/Taipei").normalize()
        current = end
        while current >= start:
            ingest_date(current)
            current -= pd.Timedelta(days=1)
    elif mode == "date":
        ingest_date(specific_date.tz_localize("Asia/Taipei"))


def main():
    parser = argparse.ArgumentParser(description="TWSE OHLCV ingestion")
    parser.add_argument("--full", action="store_true", help="Full backfill from 2010")
    parser.add_argument("--date", type=str, help="Specific date YYYY-MM-DD")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    setup_logging()

    if args.full:
        mode = "full"
        specific_date = None
    elif args.date:
        specific_date = pd.Timestamp(args.date)
        mode = "date"
    else:
        mode = "incremental"
        specific_date = None

    logging.info("Starting TWSE ingestion — mode=%s", mode)
    run(mode, specific_date)


if __name__ == "__main__":
    main()
