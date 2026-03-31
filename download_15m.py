#!/usr/bin/env python3
"""
Download BTCUSDT 15m data (2 years)
"""
import asyncio
import aiohttp
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

async def download():
    filepath = DATA_DIR / "btcusdt_15m.parquet"
    
    if filepath.exists():
        df = pd.read_parquet(filepath)
        print(f"[Cache] btcusdt_15m.parquet already exists: {len(df)} rows")
        return df
    
    print("Downloading BTCUSDT 15m data...")
    url = "https://api.binance.com/api/v3/klines"
    end_ts = int(datetime.now().timestamp() * 1000)
    start_ts = int((datetime.now() - timedelta(days=365*2)).timestamp() * 1000)
    
    all_data = []
    current_ts = start_ts
    count = 0
    
    async with aiohttp.ClientSession() as session:
        while current_ts < end_ts:
            params = {"symbol": "BTCUSDT", "interval": "15m", "startTime": current_ts, "endTime": end_ts, "limit": 1000}
            try:
                async with session.get(url, params=params) as resp:
                    if resp.status == 429:
                        print("\nRate limited, sleeping 3s...")
                        await asyncio.sleep(3)
                        continue
                    data = await resp.json()
                    if not data:
                        break
                    all_data.extend(data)
                    count += len(data)
                    current_ts = data[-1][6] + 1
                    if count % 5000 == 0:
                        print(f"\rDownloaded {count} rows...", end="", flush=True)
                    await asyncio.sleep(1.2)
            except Exception as e:
                print(f"Error: {e}, retrying...")
                await asyncio.sleep(5)
    
    print(f"\nTotal: {len(all_data)} rows")
    df = pd.DataFrame(all_data, columns=[
        "open_time","open","high","low","close","volume","close_time","quote_volume","trades","taker_buy_base","taker_buy_quote","ignore"
    ])
    for col in ["open","high","low","close","volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")
    df = df[["open_time","open","high","low","close","volume"]].copy()
    df = df.sort_values("open_time").reset_index(drop=True)
    df.to_parquet(filepath, index=False)
    print(f"[Saved] {filepath}: {len(df)} rows")
    return df

if __name__ == "__main__":
    asyncio.run(download())
