import pandas as pd
import requests
import time
import os

DATA_DIR = '/Users/changrunlin/.openclaw/workspace/crypto-agent-platform/data/'

def download_binance_data(symbol, interval="1d", start_date="2017-08-17", end_date="2026-03-26"):
    """下載 Binance K 線資料"""
    start_ts = int(pd.Timestamp(start_date).timestamp() * 1000)
    end_ts = int(pd.Timestamp(end_date).timestamp() * 1000)
    
    all_klines = []
    current_ts = start_ts
    
    while current_ts < end_ts:
        url = f"https://api.binance.com/api/v3/klines"
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": current_ts,
            "endTime": min(current_ts + 1000 * 60 * 60 * 24 * 90, end_ts),  # 每次最多90天
            "limit": 1000
        }
        
        try:
            response = requests.get(url, params=params, timeout=30)
            data = response.json()
            
            if not data or 'code' in data:
                print(f"  Error for {symbol}: {data}")
                break
            
            all_klines.extend(data)
            current_ts = data[-1][0] + 1  # 下一個時間點
            time.sleep(0.5)  # 避免 API 限流
        except Exception as e:
            print(f"  Exception for {symbol}: {e}")
            time.sleep(1)
    
    if not all_klines:
        return None
    
    # 轉換成 DataFrame
    df = pd.DataFrame(all_klines, columns=[
        'open_time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'count', 'taker_buy_base', 'taker_buy_quote', 'ignore'
    ])
    
    # 只保留需要的欄位
    df = df[['open_time', 'open', 'high', 'low', 'close', 'volume']]
    
    # 轉換類型
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 去除重複
    df = df.drop_duplicates(subset=['open_time'])
    df = df.sort_values('open_time')
    
    return df

# 20 個幣種（BTC 跳過，已存在）
coins = [
    "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT", 
    "DOTUSDT", "AVAXUSDT", "MATICUSDT", "LINKUSDT", "LTCUSDT", "UNIUSDT",
    "ATOMUSDT", "XLMUSDT", "ETCUSDT", "XMRUSDT", "FILUSDT", "APTUSDT", "ARBUSDT"
]

print("=" * 60)
print("開始下載 20 個加密貨幣歷史 K 線資料")
print("=" * 60)

results = {}
failed = []

for i, symbol in enumerate(coins):
    print(f"\n[{i+1}/19] 下載 {symbol}...")
    
    df = download_binance_data(symbol)
    
    if df is not None and len(df) > 0:
        # 轉換 open_time 為 datetime 方便顯示
        open_time_dt = pd.to_datetime(df['open_time'], unit='ms')
        start_dt = open_time_dt.iloc[0]
        end_dt = open_time_dt.iloc[-1]
        
        # 保存為 Parquet
        filepath = os.path.join(DATA_DIR, f"{symbol.lower()}_1d.parquet")
        df.to_parquet(filepath, index=False)
        
        results[symbol] = len(df)
        print(f"  ✓ {symbol}: {len(df)} bars | {start_dt.strftime('%Y-%m-%d')} ~ {end_dt.strftime('%Y-%m-%d')}")
    else:
        failed.append(symbol)
        print(f"  ✗ {symbol}: 下載失敗")
    
    time.sleep(1)  # 幣種間的延遲

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