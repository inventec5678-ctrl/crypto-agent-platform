import pandas as pd
import requests
import time
import os
import json

DATA_DIR = '/Users/changrunlin/.openclaw/workspace/crypto-agent-platform/data/'

def download_binance_data(symbol, interval="1d", start_date="2017-08-17", end_date="2026-03-26"):
    """下載 Binance K 線資料"""
    start_ts = int(pd.Timestamp(start_date).timestamp() * 1000)
    end_ts = int(pd.Timestamp(end_date).timestamp() * 1000)
    
    all_klines = []
    current_ts = start_ts
    retry_count = 0
    max_retries = 5
    
    while current_ts < end_ts:
        url = "https://api.binance.com/api/v3/klines"
        chunk_end = min(current_ts + 1000 * 60 * 60 * 24 * 90, end_ts)
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": current_ts,
            "endTime": chunk_end,
            "limit": 1000
        }
        
        try:
            response = requests.get(url, params=params, timeout=30)
            
            # Check HTTP status
            if response.status_code == 429:
                # Rate limited - wait and retry
                wait_time = min(2 ** retry_count * 5, 60)
                print(f"  Rate limited, waiting {wait_time}s...")
                time.sleep(wait_time)
                retry_count += 1
                continue
            
            if response.status_code != 200:
                print(f"  HTTP {response.status_code}: {response.text[:100]}")
                time.sleep(2)
                retry_count += 1
                continue
            
            data = response.json()
            
            # Check API error
            if isinstance(data, dict) and data.get('code'):
                print(f"  API Error {data.get('code')}: {data.get('msg')}")
                if retry_count < max_retries:
                    time.sleep(2 ** retry_count)
                    retry_count += 1
                    continue
                break
            
            if not data:
                # Empty data - coin might not exist yet in this range
                # Check if this is the first query - if so, the coin might have launched later
                if current_ts == start_ts and retry_count == 0:
                    # Try to get the earliest data for this symbol
                    print(f"  No data from {start_date}, checking when {symbol} started trading...")
                    params_first = {"symbol": symbol, "interval": interval, "limit": 1}
                    resp_first = requests.get(url, params=params_first, timeout=30)
                    if resp_first.status_code == 200:
                        first_data = resp_first.json()
                        if first_data:
                            first_ts = int(first_data[0][0])
                            print(f"  {symbol} first trade: {pd.to_datetime(first_ts, unit='ms')}")
                            current_ts = first_ts
                            time.sleep(0.5)
                            continue
                break
            
            all_klines.extend(data)
            
            # Move to next chunk
            last_ts = int(data[-1][0])
            if last_ts <= current_ts:
                # No progress - break to avoid infinite loop
                print(f"  No progress, breaking")
                break
            current_ts = last_ts + 1
            retry_count = 0  # Reset on success
            time.sleep(0.3)  # Be nice to the API
            
        except requests.exceptions.Timeout:
            print(f"  Timeout, retrying...")
            retry_count += 1
            time.sleep(2 ** retry_count)
        except Exception as e:
            print(f"  Exception: {e}")
            retry_count += 1
            time.sleep(2)
            if retry_count > max_retries:
                break
    
    if not all_klines:
        return None
    
    # Convert to DataFrame
    df = pd.DataFrame(all_klines, columns=[
        'open_time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'count', 'taker_buy_base', 'taker_buy_quote', 'ignore'
    ])
    
    # Only keep needed columns
    df = df[['open_time', 'open', 'high', 'low', 'close', 'volume']]
    
    # Convert types
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Remove duplicates and sort
    df = df.drop_duplicates(subset=['open_time'])
    df = df.sort_values('open_time')
    
    return df

# 20 个币种（BTC 跳过，已存在）
coins = [
    "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT", 
    "DOTUSDT", "AVAXUSDT", "MATICUSDT", "LINKUSDT", "LTCUSDT", "UNIUSDT",
    "ATOMUSDT", "XLMUSDT", "ETCUSDT", "XMRUSDT", "FILUSDT", "APTUSDT", "ARBUSDT"
]

print("=" * 60)
print("开始下载 19 个加密货币历史 K 线资料")
print("=" * 60)

results = {}
failed = []

for i, symbol in enumerate(coins):
    filepath = os.path.join(DATA_DIR, f"{symbol.lower()}_1d.parquet")
    if os.path.exists(filepath):
        df_existing = pd.read_parquet(filepath)
        results[symbol] = len(df_existing)
        print(f"[{i+1}/19] {symbol}: 已存在 ({len(df_existing)} bars), 跳过")
        continue
    
    print(f"\n[{i+1}/19] 下载 {symbol}...")
    
    df = download_binance_data(symbol)
    
    if df is not None and len(df) > 0:
        open_time_dt = pd.to_datetime(df['open_time'], unit='ms')
        start_dt = open_time_dt.iloc[0]
        end_dt = open_time_dt.iloc[-1]
        
        filepath = os.path.join(DATA_DIR, f"{symbol.lower()}_1d.parquet")
        df.to_parquet(filepath, index=False)
        
        results[symbol] = len(df)
        print(f"  ✓ {symbol}: {len(df)} bars | {start_dt.strftime('%Y-%m-%d')} ~ {end_dt.strftime('%Y-%m-%d')}")
    else:
        failed.append(symbol)
        print(f"  ✗ {symbol}: 下载失败")
    
    time.sleep(1.5)  # Delay between coins

print("\n" + "=" * 60)
print("下载完成报告")
print("=" * 60)
print(f"\n成功下载: {len(results)} 个币种")
print(f"下载失败: {len(failed)} 个币种")

print("\n各币种数据笔数:")
total = 0
for symbol, count in sorted(results.items()):
    print(f"  {symbol}: {count:,} bars")
    total += count

print(f"\n总数据笔数: {total:,}")

if failed:
    print(f"\n失败的币种: {', '.join(failed)}")
else:
    print("\n✓ 所有币种下载成功！")