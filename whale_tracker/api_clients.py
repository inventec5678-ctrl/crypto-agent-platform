import time
from abc import ABC, abstractmethod
from typing import Optional


class RateLimiter:
    """簡單的 Rate Limiter，防止 API 請求過快"""

    def __init__(self, requests_per_second: float = 1.0):
        self.requests_per_second = requests_per_second
        self.min_interval = 1.0 / requests_per_second
        self.last_request_time = 0.0

    def wait(self):
        """等待直到可以發送下一個請求"""
        now = time.time()
        elapsed = now - self.last_request_time
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self.last_request_time = time.time()


class BaseBlockchainClient(ABC):
    """區塊鏈 API 客戶端基底類別"""

    def __init__(self, rate_limiter: Optional[RateLimiter] = None):
        self.rate_limiter = rate_limiter or RateLimiter()

    @abstractmethod
    def get_address_transactions(self, address: str, limit: int = 50) -> list[dict]:
        """取得地址的所有轉帳記錄"""
        pass

    @abstractmethod
    def get_transaction(self, tx_hash: str) -> dict:
        """取得單筆交易詳情"""
        pass

    @abstractmethod
    def get_recent_transactions(self, limit: int = 50) -> list[dict]:
        """取得最近的大額轉帳"""
        pass


class BlockstreamClient(BaseBlockchainClient):
    """Blockstream API 客戶端（比特幣）"""

    BASE_URL = "https://blockstream.info/api"

    def __init__(self, network: str = "btc", rate_limiter: Optional[RateLimiter] = None):
        super().__init__(rate_limiter)
        self.network = network
        if network == "btc":
            self.BASE_URL = "https://blockstream.info/api"
        elif network == "testnet":
            self.BASE_URL = "https://blockstream.info/testnet/api"

    def _get(self, endpoint: str) -> dict:
        """發送 GET 請求並處理 rate limit"""
        import urllib.request
        import json

        url = f"{self.BASE_URL}{endpoint}"
        self.rate_limiter.wait()

        try:
            with urllib.request.urlopen(url, timeout=10) as resp:
                data = resp.read().decode()
                return json.loads(data)
        except Exception as e:
            print(f"[BlockstreamClient] Error fetching {url}: {e}")
            return {}

    def get_transaction(self, tx_hash: str) -> dict:
        """取得單筆交易詳情"""
        return self._get(f"/tx/{tx_hash}")

    def get_address_transactions(self, address: str, limit: int = 50) -> list[dict]:
        """取得地址的轉帳記錄"""
        # Blockstream 的 address endpoint 可能需要信用卡，這裡用替代方案
        # 實際上 Blockstream 沒有直接的 address transactions API
        return self._get(f"/address/{address}/txs")[:limit]

    def get_recent_transactions(self, limit: int = 50) -> list[dict]:
        """取得最近區塊的所有交易"""
        # 取得最新區塊高度
        latest_block = self._get("/blocks/tip/height")
        if not latest_block:
            return []

        # 取得區塊詳情
        block_hash = self._get(f"/block-height/{latest_block}")
        if not block_hash:
            return []

        block_data = self._get(f"/block/{block_hash}/txids")
        if not isinstance(block_data, list):
            return []

        # 取得每筆交易的詳情
        transactions = []
        for txid in block_data[:limit]:
            tx = self.get_transaction(txid)
            if tx:
                transactions.append(tx)

        return transactions


class BlockchairClient(BaseBlockchainClient):
    """Blockchair API 客戶端（支援多鏈）"""

    # Blockchair 免費方案 API
    BASE_URL = "https://api.blockchair.com"

    def __init__(self, coin: str = "bitcoin", rate_limiter: Optional[RateLimiter] = None):
        super().__init__(rate_limiter)
        self.coin = coin

    def _get(self, endpoint: str) -> dict:
        """發送 GET 請求並處理 rate limit"""
        import urllib.request
        import json

        url = f"{self.BASE_URL}{endpoint}"
        self.rate_limiter.wait()

        try:
            with urllib.request.urlopen(url, timeout=15) as resp:
                data = resp.read().decode()
                result = json.loads(data)
                # Blockchair 回傳格式: {"data": {...}, "context": {...}}
                return result.get("data", {})
        except Exception as e:
            print(f"[BlockchairClient] Error fetching {url}: {e}")
            return {}

    def get_transaction(self, tx_hash: str) -> dict:
        """取得單筆交易詳數"""
        return self._get(f"/{self.coin}/transactions?q=hash:{tx_hash}")

    def get_recent_transactions(self, limit: int = 50) -> list[dict]:
        """取得最近的大額轉帳"""
        # 使用 Blockchair 的 transactions API
        # 按 fee 排序，取最新
        data = self._get(f"/{self.coin}/transactions?limit={limit}&order=time_desc")
        if isinstance(data, dict) and "transactions" in data:
            return data["transactions"]
        return []

    def get_address_transactions(self, address: str, limit: int = 50) -> list[dict]:
        """取得地址的轉帳記錄"""
        data = self._get(f"/{self.coin}/addresses/{address}/transactions?limit={limit}")
        if isinstance(data, dict):
            return data.get("transactions", data)
        return []


# Mempool.space client (另一個免費方案)
class MempoolClient(BaseBlockchainClient):
    """Mempool.space API 客戶端"""

    def __init__(self, network: str = "bitcoin", rate_limiter: Optional[RateLimiter] = None):
        super().__init__(rate_limiter)
        self.network = network
        if network == "bitcoin":
            self.BASE_URL = "https://mempool.space/api"
        elif network == "testnet":
            self.BASE_URL = "https://mempool.space/testnet/api"

    def _get(self, endpoint: str) -> dict:
        import urllib.request
        import json

        url = f"{self.BASE_URL}{endpoint}"
        self.rate_limiter.wait()

        try:
            with urllib.request.urlopen(url, timeout=10) as resp:
                return json.loads(resp.read().decode())
        except Exception as e:
            print(f"[MempoolClient] Error fetching {url}: {e}")
            return {}

    def get_transaction(self, tx_hash: str) -> dict:
        return self._get(f"/tx/{tx_hash}")

    def get_recent_transactions(self, limit: int = 50) -> list[dict]:
        # Mempool 沒有直接的 recent tx API，但可以掃描最新區塊
        return []

    def get_address_transactions(self, address: str, limit: int = 50) -> list[dict]:
        txs = self._get(f"/address/{address}/txs")
        if isinstance(txs, list):
            return txs[:limit]
        return []
