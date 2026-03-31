from dataclasses import dataclass
from datetime import datetime
from typing import Literal


@dataclass
class WhaleAlert:
    """鯨魚大額轉帳警示"""
    amount_btc: float
    direction: Literal["to_exchange", "from_exchange"]
    tx_hash: str
    timestamp: str
    alert: bool
    # 額外援救
    type: Literal["whale_transfer"] = "whale_transfer"
    from_address: str = ""
    to_address: str = ""
    exchange_name: str = ""
    alert_message: str = ""

    def to_dict(self) -> dict:
        return {
            "type": self.type,
            "amount_btc": self.amount_btc,
            "direction": self.direction,
            "tx_hash": self.tx_hash,
            "timestamp": self.timestamp,
            "alert": self.alert,
        }
