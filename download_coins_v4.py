import pandas as pd
import requests
import time
import os

DATA_DIR = '/Users/changrunlin/.openclaw/workspace/crypto-agent-platform/data/'

def get_earliest_data_time(symbol, interval="1d"):
    """取得幣種最早的交易時間"""
    url = "https://api.binance.com/api/v3/klines"
    # 查詢一個非常早的時間，使用 limit=1 應該只回傳最舊的一筆
    params = {
        "symbol": symbol,
        "interval": interval,
        "startTime": int(pd.Timestamp('2010-01-01').timestamp() * 1000),
        "limit": 1
    }
    try:
        resp = requests.get(url, params=params, timeout=30)
        if resp.status_code == 200:
            data = resp.json()
            if data:
                return int(data[0][0])
    except:
        pass
    return None

def download_binance_data(symbol, interval="1d", start_date="2017-08-17", end_date="2026-03-26"):
    """下載 Binance K 線資料"""
    start_ts = int(pd.Timestamp(start_date).timestamp() * 1000)
    end_ts = int(pd.Timestamp(end_date).timestamp() * 1000)
    
    all_klines = []
    current_ts = start_ts
    chunk_size = 1000 * 60 * 60 * 24 * 90  # 90 days in ms
    retry_count = 0
    max_retries = 5
    
    while current_ts < end_ts:
        url = "https://api.binance.com/api/v3/klines"
        chunk_end = min(current_ts + chunk_size, end_ts)
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
                wait_time = min(2 ** retry_count * 5, 60)
                print(f"  Rate limited, waiting {wait_time}s...")
                time.sleep(wait_time)
                retry_count += 1
                continue
            
            if response.status_code != 200:
                print(f"  HTTP {response.status_code}")
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
                # 如果一開始就沒有資料，嘗試找真實的起始時間
                if current_ts == start_ts:
                    earliest = get_earliest_data_time(symbol, interval)
                    if earliest and earliest > start_ts:
                        print(f"  {symbol} 起始於 {pd.to_datetime(earliest, unit='ms')}, 從那裡開始下載...")
                        current_ts = earliest
                        time.sleep(0.5)
                        continue
                break
            
            all_klines.extend(data)
            
            # Move to next chunk
            last_ts = int(data[-1][0])
            if last_ts <= current_ts:
                break
            current_ts = last_ts + 1
            retry_count = 0
            time.sleep(0.25)
            
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

# 19 個幣種（BTC 跳過，已存在）
coins = [
    "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT", 
    "DOTUSDT", "AVAXUSDT", "MATICUSDT", "LINKUSDT", "LTCUSDT", "UNIUSDT",
    "ATOMUSDT", "XLMUSDT", "ETCUSDT", "XMRUSDT", "FILUSDT", "APTUSDT", "ARBUSDT"
]

print("=" * 60)
print("開始下載 19 個加密貨幣歷史 K 線資料")
print("=" * 60)

results = {}
failed = []

for i, symbol in enumerate(coins):
    filepath = os.path.join(DATA_DIR, f"{symbol.lower()}_1d.parquet")
    if os.path.exists(filepath):
        df_existing = pd.read_parquet(filepath)
        results[symbol] = len(df_existing)
        print(f"[{i+1}/19] {symbol}: 已存在 ({len(df_existing)} bars), 跳過")
        continue
    
    print(f"\n[{i+1}/19] 下載 {symbol}...")
    
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
        print(f"  ✗ {symbol}: 下載失敗")
    
    time.sleep(1.5)

print("\n" + "=" * 60)
print("下載完成報告")
print("=" * 60)
print(f"\n成功下載: {len(results)} 個幣種")
print(f"下載失敗: {len(failed)} 個幣種")

print("\n各幣種數據筆數:")
total = 0
for symbol, count in sorted(results.items()):
    print(f"  {symbol}: {count:,} bars")
    total += count

print(f"\n總數據筆數: {total:,}")

if failed:
    print(f"\n失敗的幣種: {', '.join(failed)}")
else:
    print("\n✓ 所有幣種下載成功！")