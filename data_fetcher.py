# data_fetcher.py
# 統一資料下載框架

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
import os
import json as _json
import pandas as pd
import requests

OHLCVUTC_COLS = [
    'open', 'high', 'low', 'close', 'volume',
    'market', 'symbol', 'currency', 'timeframe',
    'adj_close', 'dividends', 'stock_splits', 'source', 'fetched_at'
]

@dataclass
class OHLCVUTC:
    """標準化 K 線資料格式"""
    timestamp: pd.DatetimeIndex
    open: pd.Series
    high: pd.Series
    low: pd.Series
    close: pd.Series
    volume: pd.Series
    market: str = ''
    symbol: str = ''
    currency: str = ''
    timeframe: str = ''
    adj_close: Optional[pd.Series] = None
    dividends: Optional[pd.Series] = None
    stock_splits: Optional[pd.Series] = None
    source: str = ''
    fetched_at: datetime = field(default_factory=datetime.utcnow)

    def to_dataframe(self) -> pd.DataFrame:
        data = {
            'open': self.open, 'high': self.high, 'low': self.low,
            'close': self.close, 'volume': self.volume,
            'market': self.market, 'symbol': self.symbol,
            'currency': self.currency, 'timeframe': self.timeframe,
            'adj_close': self.adj_close if self.adj_close is not None else pd.Series(dtype=float),
            'dividends': self.dividends if self.dividends is not None else pd.Series(dtype=float),
            'stock_splits': self.stock_splits if self.stock_splits is not None else pd.Series(dtype=float),
            'source': self.source,
            'fetched_at': self.fetched_at,
        }
        df = pd.DataFrame(data, index=self.timestamp)
        df.index.name = 'timestamp'
        return df


class DataFeed(ABC):
    """所有市場下載器的統一介面"""
    market: str = ''
    currency: str = ''
    base_path: str = '/Users/changrunlin/.openclaw/workspace/crypto-agent-platform/data'

    @abstractmethod
    def fetch_ohlcv(self, symbol: str, timeframe: str,
                    start: datetime, end: datetime) -> pd.DataFrame:
        """回傳標準化 ohlcvutc DataFrame（index = timestamp, UTC）"""
        pass

    def _make_path(self, symbol: str, timeframe: str) -> str:
        return f"{self.base_path}/ohlcvutc/{self.market}/{symbol}_{timeframe}.parquet"

    def download(self, symbol: str, timeframes: list[str],
                 start: datetime, end: datetime) -> list[str]:
        """下載並寫入 parquet，回傳寫入的檔案路徑列表"""
        paths = []
        for tf in timeframes:
            df = self.fetch_ohlcv(symbol, tf, start, end)
            path = self._make_path(symbol, tf)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            df.to_parquet(path)
            paths.append(path)
        return paths

    def load(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """讀取本地 parquet"""
        return pd.read_parquet(self._make_path(symbol, timeframe))


class BinanceFeed(DataFeed):
    """Binance 加密幣下載器"""
    market = 'crypto'
    currency = 'USDT'
    base_url = 'https://api.binance.com/api/v3/klines'

    TIMEFRAME_MAP = {
        '1m': '1m', '5m': '5m', '15m': '15m',
        '1h': '1h', '4h': '4h', '1d': '1d', '1w': '1w'
    }

    def fetch_ohlcv(self, symbol: str, timeframe: str,
                    start: datetime, end: datetime) -> pd.DataFrame:
        interval = self.TIMEFRAME_MAP.get(timeframe, timeframe)
        params = {
            'symbol': symbol.upper(),
            'interval': interval,
            'startTime': int(start.timestamp() * 1000),
            'endTime': int(end.timestamp() * 1000),
            'limit': 1000
        }
        resp = requests.get(self.base_url, params=params, timeout=30)
        resp.raise_for_status()
        raw = resp.json()

        records = []
        for k in raw:
            records.append({
                'timestamp': pd.to_datetime(k[0], unit='ms', utc=True),
                'open': float(k[1]),
                'high': float(k[2]),
                'low': float(k[3]),
                'close': float(k[4]),
                'volume': float(k[5]),
            })

        df = pd.DataFrame(records).set_index('timestamp')
        df.index = df.index.tz_localize(None)  # 移除時區，保持 naive UTC

        df['market'] = self.market
        df['symbol'] = symbol.upper()
        df['currency'] = self.currency
        df['timeframe'] = timeframe
        df['adj_close'] = df['close']
        df['dividends'] = 0.0
        df['stock_splits'] = 0.0
        df['source'] = 'binance'
        df['fetched_at'] = pd.Timestamp.utcnow()

        return df[OHLCVUTC_COLS]


class TWSEFeed(DataFeed):
    """TWSE 台股下載器（yfinance 備援）"""
    market = 'twse'
    currency = 'TWD'

    def fetch_ohlcv(self, symbol: str, timeframe: str,
                    start: datetime, end: datetime) -> pd.DataFrame:
        import yfinance as yf
        ticker = yf.Ticker(f"{symbol}.TW")
        df = ticker.history(start=start, end=end, interval='1d')
        if df.empty:
            return pd.DataFrame(columns=OHLCVUTC_COLS)

        if df.index.tz is not None:
            df.index = df.index.tz_convert('UTC').tz_localize(None)
        else:
            df.index = pd.to_datetime(df.index, utc=True).tz_localize(None)

        df = df.rename(columns={
            'Open': 'open', 'High': 'high', 'Low': 'low',
            'Close': 'close', 'Volume': 'volume',
            'Dividends': 'dividends', 'Stock Splits': 'stock_splits'
        })
        if 'Adj Close' in df.columns:
            df['adj_close'] = df['Adj Close']
        else:
            df['adj_close'] = df['close']

        df['market'] = self.market
        df['symbol'] = symbol
        df['currency'] = self.currency
        df['timeframe'] = timeframe
        df['source'] = 'yfinance'
        df['fetched_at'] = pd.Timestamp.utcnow()

        for col in ['Dividends', 'Stock Splits', 'Adj Close']:
            if col in df.columns:
                df.drop(col, axis=1, inplace=True)

        existing = [c for c in OHLCVUTC_COLS if c in df.columns]
        return df[existing]


class USFeed(DataFeed):
    """美股下載器（yfinance）"""
    market = 'us'
    currency = 'USD'

    def fetch_ohlcv(self, symbol: str, timeframe: str,
                    start: datetime, end: datetime) -> pd.DataFrame:
        import yfinance as yf

        interval_map = {
            '1d': '1d', '1wk': '1wk', '1mo': '1mo',
            '5m': '5m', '15m': '15m', '1h': '1h'
        }
        interval = interval_map.get(timeframe, '1d')

        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start, end=end, interval=interval)
        if df.empty:
            return pd.DataFrame(columns=OHLCVUTC_COLS)

        if df.index.tz is not None:
            df.index = df.index.tz_convert('UTC').tz_localize(None)
        else:
            df.index = pd.to_datetime(df.index, utc=True).tz_localize(None)

        df = df.rename(columns={
            'Open': 'open', 'High': 'high', 'Low': 'low',
            'Close': 'close', 'Volume': 'volume',
            'Dividends': 'dividends', 'Stock Splits': 'stock_splits'
        })
        if 'Adj Close' in df.columns:
            df['adj_close'] = df['Adj Close']
        else:
            df['adj_close'] = df['close']

        df['market'] = self.market
        df['symbol'] = symbol
        df['currency'] = self.currency
        df['timeframe'] = timeframe
        df['source'] = 'yfinance'
        df['fetched_at'] = pd.Timestamp.utcnow()

        for col in ['Dividends', 'Stock Splits', 'Adj Close']:
            if col in df.columns:
                df.drop(col, axis=1, inplace=True)

        existing = [c for c in OHLCVUTC_COLS if c in df.columns]
        return df[existing]


# ─── Registry ────────────────────────────────────────────────────────────────

REGISTRY_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'data', 'ohlcvutc', '_registry.json'
)


def update_registry():
    """Scan ohlcvutc/ directories and write _registry.json."""
    base = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'data', 'ohlcvutc'
    )
    registry = {}

    for market_dir in os.listdir(base):
        market_path = os.path.join(base, market_dir)
        if not os.path.isdir(market_path) or market_dir.startswith('_'):
            continue
        registry[market_dir] = {}
        for fname in os.listdir(market_path):
            if not fname.endswith('.parquet'):
                continue
            # filename format: SYMBOL_TIMEFRAME.parquet
            parts = fname[:-8].rsplit('_', 1)
            if len(parts) != 2:
                continue
            symbol, timeframe = parts
            registry[market_dir].setdefault(symbol, [])
            if timeframe not in registry[market_dir][symbol]:
                registry[market_dir][symbol].append(timeframe)

    os.makedirs(os.path.dirname(REGISTRY_PATH), exist_ok=True)
    with open(REGISTRY_PATH, 'w') as f:
        _json.dump(registry, f, indent=2)

    return registry


def load_ohlcv(market: str, symbol: str, timeframe: str,
               start: datetime = None, end: datetime = None) -> pd.DataFrame:
    """
    統一載入 K 線資料。

    優先從本地 ohlcvutc/ 讀取；若本地檔案不存在則自動下載。

    Args:
        market:     'crypto' | 'twse' | 'us'
        symbol:     'BTCUSDT' | '2330' | 'AAPL'
        timeframe:  '1d' | '4h' | '15m' | '1w' | ...
        start:      起始時間（可選）
        end:        結束時間（可選）

    Returns:
        pd.DataFrame with naive-UTC timestamp index and OHLCVUTC_COLS columns
    """
    feed_map = {
        'crypto': BinanceFeed(),
        'twse': TWSEFeed(),
        'us': USFeed(),
    }
    feed = feed_map.get(market)
    if not feed:
        raise ValueError(f"Unknown market: {market}")

    path = feed._make_path(symbol, timeframe)

    if os.path.exists(path):
        df = pd.read_parquet(path)
    else:
        _start = start or datetime(2020, 1, 1)
        _end   = end   or datetime.utcnow()
        df = feed.fetch_ohlcv(symbol, timeframe, _start, _end)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_parquet(path)

    # 時間範圍過濾（index 為 naive UTC，需要 strip tz from filter）
    if start:
        start_ts = pd.to_datetime(start).tz_localize('UTC').tz_convert(None)
        df = df[df.index >= start_ts]
    if end:
        end_ts = pd.to_datetime(end).tz_localize('UTC').tz_convert(None)
        df = df[df.index <= end_ts]

    return df