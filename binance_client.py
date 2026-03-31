"""幣安 API 客戶端 - WebSocket + REST API"""
import asyncio
import json
import logging
from typing import Callable, Optional
from datetime import datetime, timedelta
import httpx
from config import settings

logger = logging.getLogger(__name__)


class BinanceClient:
    """幣安 API 客戶端"""
    
    BASE_URL = "https://api.binance.com"
    WS_URL = "wss://stream.binance.com:9443/ws"
    
    # Rate Limit 設定
    RATE_LIMIT_WEIGHT = 1200  #  weight per minute for REST API
    RATE_LIMIT_SLEEP = 60     # 超過限制後等待秒數
    
    def __init__(self):
        self.api_key = settings.binance_api_key
        self.api_secret = settings.binance_api_secret
        self._price_cache: dict = {}
        self._last_request_time: Optional[datetime] = None
        self._request_count: int = 0
    
    async def _wait_for_rate_limit(self):
        """處理 Rate Limit"""
        now = datetime.now()
        if self._last_request_time:
            elapsed = (now - self._last_request_time).total_seconds()
            if elapsed < 60:
                self._request_count += 1
                if self._request_count > self.RATE_LIMIT_WEIGHT:
                    sleep_time = 60 - elapsed + 1
                    logger.warning(f"Rate limit reached, sleeping {sleep_time}s")
                    await asyncio.sleep(sleep_time)
                    self._request_count = 0
            else:
                self._request_count = 0
        self._last_request_time = now
    
    async def _make_request(self, endpoint: str, params: dict = None) -> dict:
        """發送 REST API 請求"""
        await self._wait_for_rate_limit()
        
        url = f"{self.BASE_URL}{endpoint}"
        headers = {"X-MBX-APIKEY": self.api_key} if self.api_key else {}
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(url, params=params, headers=headers)
            response.raise_for_status()
            return response.json()
    
    async def get_klines(
        self,
        symbol: str,
        interval: str = "1h",
        limit: int = 100
    ) -> list:
        """
        取得 K 線數據
        
        Args:
            symbol: 交易對，如 "BTCUSDT"
            interval: K 線周期，如 "1m", "5m", "1h", "1d"
            limit: 數量
        
        Returns:
            K 線數據列表
        """
        params = {
            "symbol": symbol.upper(),
            "interval": interval,
            "limit": limit
        }
        
        try:
            data = await self._make_request("/api/v3/klines", params)
            return self._parse_klines(data)
        except Exception as e:
            logger.error(f"Failed to get klines for {symbol}: {e}")
            return []
    
    def _parse_klines(self, raw_data: list) -> list:
        """解析 K 線數據"""
        klines = []
        for k in raw_data:
            klines.append({
                "open_time": datetime.fromtimestamp(k[0] / 1000),
                "open": float(k[1]),
                "high": float(k[2]),
                "low": float(k[3]),
                "close": float(k[4]),
                "volume": float(k[5]),
                "close_time": datetime.fromtimestamp(k[6] / 1000),
            })
        return klines
    
    async def get_symbol_price(self, symbol: str) -> Optional[float]:
        """取得當前價格"""
        try:
            data = await self._make_request("/api/v3/ticker/price", {"symbol": symbol.upper()})
            return float(data["price"])
        except Exception as e:
            logger.error(f"Failed to get price for {symbol}: {e}")
            return None
    
    async def get_multiple_prices(self, symbols: list[str]) -> dict[str, float]:
        """批量取得價格"""
        prices = {}
        for symbol in symbols:
            price = await self.get_symbol_price(symbol)
            if price:
                prices[symbol.upper()] = price
            await asyncio.sleep(0.1)  # 避免瞬間請求過多
        return prices
    
    async def get_order_book(self, symbol: str, limit: int = 100) -> dict:
        """取得訂單簿深度"""
        data = await self._make_request("/api/v3/depth", {"symbol": symbol, "limit": limit})
        bids_total = sum(float(b[1]) for b in data.get("bids", []))
        asks_total = sum(float(a[1]) for a in data.get("asks", []))
        return {
            "bids_total": bids_total,
            "asks_total": asks_total,
            "spread": float(data["asks"][0][0]) - float(data["bids"][0][0]) if data.get("asks") and data.get("bids") else 0
        }

    async def subscribe_price(
        self,
        symbols: list[str],
        callback: Callable[[str, float], None]
    ):
        """
        訂閱 WebSocket 即時價格
        
        Args:
            symbols: 交易對列表
            callback: 價格回調函數
        """
        streams = [f"{s.lower()}@ticker" for s in symbols]
        stream_param = "/".join(streams)
        ws_url = f"{self.WS_URL}/{stream_param}"
        
        logger.info(f"Subscribing to WebSocket: {stream_param}")
        
        while True:
            try:
                async with httpx.AsyncClient() as client:
                    async with client.ws_connect(ws_url) as ws:
                        logger.info("WebSocket connected")
                        async for msg in ws:
                            if msg.type == httpx.WSMsgType.TEXT:
                                data = json.loads(msg.data)
                                if "s" in data and "c" in data:
                                    symbol = data["s"]
                                    price = float(data["c"])
                                    self._price_cache[symbol] = {
                                        "price": price,
                                        "timestamp": datetime.now()
                                    }
                                    await callback(symbol, price)
                            elif msg.type == httpx.WSMsgType.ERROR:
                                logger.error(f"WebSocket error: {msg.data}")
                                break
            except Exception as e:
                logger.error(f"WebSocket connection error: {e}, reconnecting in 5s...")
                await asyncio.sleep(5)


# 全域客戶端實例
binance_client = BinanceClient()
