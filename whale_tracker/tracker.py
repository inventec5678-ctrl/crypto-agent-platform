import time
from datetime import datetime, timezone
from typing import Optional

from .api_clients import BlockchairClient, RateLimiter
from .exchanges import is_exchange_address, is_mining_pool_address
from .models import WhaleAlert


# BTC 鯨魚門檻（1000 BTC）
WHALE_THRESHOLD_BTC = 1000.0


class WhaleTracker:
    """鯨魚追蹤器 - 監控區塊鏈上的大額轉帳"""

    def __init__(
        self,
        threshold_btc: float = WHALE_THRESHOLD_BTC,
        api_client: Optional[BlockchairClient] = None,
    ):
        self.threshold_btc = threshold_btc
        # 預設使用 Blockchair（支援多鏈擴展）
        self.api_client = api_client or BlockchairClient(coin="bitcoin")
        self.rate_limiter = RateLimiter(requests_per_second=1.0)

    def parse_transaction(self, tx: dict) -> Optional[WhaleAlert]:
        """
        解析區塊鏈交易，判斷是否為鯨魚大額轉帳

        Args:
            tx: 交易資料 dict

        Returns:
            WhaleAlert 或 None（不是鯨魚交易）
        """
        try:
            # 從交易中提取金額
            # Blockchair 格式: inputs[].value_sat / 1e8 = BTC
            inputs = tx.get("inputs", [])
            outputs = tx.get("outputs", [])

            total_in = sum(inp.get("value", 0) for inp in inputs) / 1e8  # satoshi to BTC
            total_out = sum(out.get("value", 0) for out in outputs) / 1e8

            amount_btc = min(total_in, total_out)

            # 檢查是否超過鯨魚門檻
            if amount_btc < self.threshold_btc:
                return None

            # 找出轉入/轉出地址
            from_addresses = []
            to_addresses = []

            for inp in inputs:
                if "recipient" in inp:
                    from_addresses.append(inp["recipient"])
                elif "address" in inp:
                    from_addresses.append(inp["address"])

            for out in outputs:
                if "recipient" in out:
                    to_addresses.append(out["recipient"])
                elif "address" in out:
                    to_addresses.append(out["address"])

            # 判斷方向
            direction = self._determine_direction(from_addresses, to_addresses)

            # 取得時間戳
            timestamp = tx.get("time", "")
            if not timestamp:
                timestamp = datetime.now(timezone.utc).isoformat()

            tx_hash = tx.get("hash", tx.get("txid", ""))

            # 建立警示
            is_exchange, exchange_name = self._check_exchange_involvement(
                from_addresses, to_addresses
            )

            alert = WhaleAlert(
                amount_btc=round(amount_btc, 2),
                direction=direction,
                tx_hash=tx_hash,
                timestamp=timestamp,
                alert=is_exchange,
                from_address=",".join(from_addresses[:3]),  # 限制長度
                to_address=",".join(to_addresses[:3]),
                exchange_name=exchange_name,
                alert_message=self._generate_alert_message(amount_btc, direction, exchange_name),
            )

            return alert

        except (KeyError, ValueError, TypeError) as e:
            print(f"[WhaleTracker] Error parsing transaction: {e}")
            return None

    def _determine_direction(
        self, from_addrs: list[str], to_addrs: list[str]
    ) -> str:
        """判斷轉帳方向"""
        # 礦池地址轉出通常是 coinbase 獎勵，不是真的轉帳
        from_is_exchange, _ = self._check_exchange_involvement(from_addrs, [])
        to_is_exchange, _ = self._check_exchange_involvement([], to_addrs)

        if to_is_exchange and not from_is_exchange:
            return "to_exchange"    # 轉入交易所 → 潛在賣壓
        elif from_is_exchange and not to_is_exchange:
            return "from_exchange"  # 轉出交易所 → 潛在買盤
        elif to_is_exchange and from_is_exchange:
            return "exchange_to_exchange"
        else:
            return "unknown"

    def _check_exchange_involvement(
        self, from_addrs: list[str], to_addrs: list[str]
    ) -> tuple[bool, str]:
        """檢查交易是否涉及交易所"""
        all_addresses = from_addrs + to_addrs
        for addr in all_addresses:
            is_ex, name = is_exchange_address(addr)
            if is_ex:
                return True, name
        return False, ""

    def _generate_alert_message(
        self, amount_btc: float, direction: str, exchange_name: str
    ) -> str:
        """產生警示訊息"""
        if direction == "to_exchange":
            return f"🐋 鯨魚轉入 {exchange_name or '交易所'}，潛在賣壓！({amount_btc:,.2f} BTC)"
        elif direction == "from_exchange":
            return f"🐋 鯨魚從 {exchange_name or '交易所'} 轉出，潛在買盤！({amount_btc:,.2f} BTC)"
        elif direction == "exchange_to_exchange":
            return f"🐋 鯨魚交易所間轉帳 ({amount_btc:,.2f} BTC)"
        else:
            return f"🐋 鯨魚大額轉帳 ({amount_btc:,.2f} BTC)"

    def scan_recent(self, limit: int = 100) -> list[WhaleAlert]:
        """
        掃描最近的大額轉帳

        Args:
            limit: 最大掃描筆數

        Returns:
            WhaleAlert 列表
        """
        print(f"[WhaleTracker] Scanning recent transactions (limit={limit})...")

        try:
            transactions = self.api_client.get_recent_transactions(limit=limit)
        except Exception as e:
            print(f"[WhaleTracker] Failed to get recent transactions: {e}")
            return []

        alerts = []
        for tx in transactions:
            alert = self.parse_transaction(tx)
            if alert and alert.alert:  # 只回傳涉及交易所的鯨魚交易
                alerts.append(alert)
                print(f"[WhaleTracker] {alert.alert_message}")

        print(f"[WhaleTracker] Found {len(alerts)} whale alerts")
        return alerts

    def check_transaction(self, tx_hash: str) -> Optional[WhaleAlert]:
        """檢查特定交易是否為鯨魚交易"""
        try:
            tx = self.api_client.get_transaction(tx_hash)
            if not tx:
                return None
            return self.parse_transaction(tx)
        except Exception as e:
            print(f"[WhaleTracker] Error checking transaction {tx_hash}: {e}")
            return None

    def check_address(self, address: str, limit: int = 20) -> list[WhaleAlert]:
        """檢查地址的大額轉帳"""
        try:
            txs = self.api_client.get_address_transactions(address, limit=limit)
            alerts = []
            for tx in txs:
                alert = self.parse_transaction(tx)
                if alert:
                    alerts.append(alert)
            return alerts
        except Exception as e:
            print(f"[WhaleTracker] Error checking address {address}: {e}")
            return []
