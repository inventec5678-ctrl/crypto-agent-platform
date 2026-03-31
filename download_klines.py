import pandas as pd
import requests
import time
import os

def download_binance_data(symbol, interval, start_date="2017-08-17", end_date="2026-03-26"):
    """下載 Binance K 線資料"""
    start_ts = int(pd.Timestamp(start_date).timestamp() * 1000)
    end_ts = int(pd.Timestamp(end_date).timestamp() * 1000)
    
    all_klines = []
    current_ts = start_ts
    
    while current_ts < end_ts:
        url = "https://api.binance.com/api/v3/klines"
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": current_ts,
            "endTime": min(current_ts + 1000 * 60 * 60 * 24 * 90, end_ts),
            "limit": 1000
        }
        
        try:
            response = requests.get(url, params=params, timeout=30)
            data = response.json()
            
            if not data or isinstance(data, dict):
                break
            
            all_klines.extend(data)
            current_ts = data[-1][0] + 1
            time.sleep(0.25)
            
        except Exception as e:
            print(f"   Error: {e}")
            time.sleep(1)
    
    if not all_klines:
        return None
    
    df = pd.DataFrame(all_klines, columns=[
        'open_time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'count', 'taker_buy_base', 'taker_buy_quote', 'ignore'
    ])
    
    df = df[['open_time', 'open', 'high', 'low', 'close', 'volume']]
    
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df = df.drop_duplicates(subset=['open_time'])
    df = df.sort_values('open_time').reset_index(drop=True)
    
    return df

coins = ["ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT", 
         "DOTUSDT", "AVAXUSDT", "MATICUSDT", "LINKUSDT", "LTCUSDT", "UNIUSDT",
         "ATOMUSDT", "XLMUSDT", "ETCUSDT", "XMRUSDT", "FILUSDT", "APTUSDT", "ARBUSDT"]

timeframes = ["1d", "4h", "15m"]

base_path = "/Users/changrunlin/.openclaw/workspace/crypto-agent-platform/data/"

report = {}

for coin in coins:
    report[coin] = {}
    for tf in timeframes:
        filename = f"{base_path}{coin.lower()}_{tf}.parquet"
        
        if os.path.exists(filename):
            existing = pd.read_parquet(filename)
            report[coin][tf] = len(existing)
            print(f"✅ {coin} {tf} 已存在: {len(existing)} bars")
            continue
        
        print(f"⬇️  下載 {coin} {tf}...")
        df = download_binance_data(coin, tf)
        
        if df is not None and len(df) > 0:
            df.to_parquet(filename, index=False)
            report[coin][tf] = len(df)
            print(f"   ✅ {coin} {tf}: {len(df)} bars")
        else:
            report[coin][tf] = 0
            print(f"   ❌ {coin} {tf}: 下載失敗")

print("\n=== 下載報告 ===")
total = 0
for coin in coins:
    d = report[coin]
    bars_str = ", ".join([f"{tf}={d[tf]}" for tf in timeframes])
    print(f"{coin}: {bars_str}")
    total += sum(d.values())

print(f"\n總共: {len(coins)} 幣種 × {len(timeframes)} 時間框架 = {len(coins) * len(timeframes)} 個檔案")
print(f"總 bars: {total}")
