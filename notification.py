"""Discord 通知服務"""
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Optional
import httpx
from config import settings

logger = logging.getLogger(__name__)


class NotificationManager:
    """
    Discord 通知管理器
    
    Rate Limit 控制：每檔每天 ≤3 次
    """
    
    MAX_DAILY_NOTIFICATIONS = 3
    
    def __init__(self):
        self.webhook_url = settings.discord_webhook_url
        self._daily_counts: dict[str, dict] = {}  # {symbol: {"count": int, "date": str}}
        self._lock = asyncio.Lock()
    
    async def _check_rate_limit(self, symbol: str) -> bool:
        """
        檢查是否超過 Rate Limit
        
        Returns:
            True 表示可以發送，False 表示已超限
        """
        async with self._lock:
            today = datetime.now().strftime("%Y-%m-%d")
            
            if symbol not in self._daily_counts:
                self._daily_counts[symbol] = {"count": 0, "date": today}
            
            entry = self._daily_counts[symbol]
            
            # 新的一天，重置計數
            if entry["date"] != today:
                entry["count"] = 0
                entry["date"] = today
            
            if entry["count"] >= self.MAX_DAILY_NOTIFICATIONS:
                logger.warning(f"Rate limit reached for {symbol}: {entry['count']}/{self.MAX_DAILY_NOTIFICATIONS}")
                return False
            
            entry["count"] += 1
            return True
    
    async def send_signal(
        self,
        symbol: str,
        signal_data: dict,
        ai_analysis: dict
    ) -> bool:
        """
        發送交易訊號通知
        
        Args:
            symbol: 交易對
            signal_data: 策略訊號資料
            ai_analysis: AI 分析結果
        
        Returns:
            是否發送成功
        """
        if not self.webhook_url:
            logger.warning("Discord webhook URL not configured")
            return False
        
        # 檢查 Rate Limit
        can_send = await self._check_rate_limit(symbol)
        if not can_send:
            logger.info(f"Skipping notification for {symbol} due to rate limit")
            return False
        
        # 構建 Embed 訊息
        color = 0x00ff00 if signal_data["direction"] == "LONG" else 0xff0000
        
        embed = {
            "title": f"🚨 {signal_data['signal']} - {symbol}",
            "color": color,
            "fields": [
                {
                    "name": "方向",
                    "value": signal_data["direction"],
                    "inline": True
                },
                {
                    "name": "價格",
                    "value": f"${signal_data['price']:,.2f}",
                    "inline": True
                },
                {
                    "name": "策略信心",
                    "value": f"{signal_data.get('confidence', 0):.1f}%",
                    "inline": True
                },
                {
                    "name": "AI 信心指數",
                    "value": f"{ai_analysis['confidence_score']:.1f}% {ai_analysis['rating']}",
                    "inline": False
                },
                {
                    "name": "因子分析",
                    "value": (
                        f"技術: {ai_analysis['factors']['technical']:.1f}%\n"
                        f"情緒: {ai_analysis['factors']['sentiment']:.1f}%\n"
                        f"社群: {ai_analysis['factors']['social']:.1f}%\n"
                        f"異常: {ai_analysis['factors']['anomaly']:.1f}%"
                    ),
                    "inline": True
                }
            ],
            "footer": {
                "text": f"今日通知: {self._daily_counts[symbol]['count']}/{self.MAX_DAILY_NOTIFICATIONS}"
            },
            "timestamp": datetime.now().isoformat()
        }
        
        return await self._send_webhook(embed)
    
    async def send_status(self, message: str, color: int = 0x3498db) -> bool:
        """發送狀態更新"""
        if not self.webhook_url:
            return False
        
        embed = {
            "title": "📊 系統狀態",
            "description": message,
            "color": color,
            "timestamp": datetime.now().isoformat()
        }
        
        return await self._send_webhook(embed)
    
    async def _send_webhook(self, embed: dict) -> bool:
        """發送 Webhook 請求"""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    self.webhook_url,
                    json={"embeds": [embed]},
                    headers={"Content-Type": "application/json"}
                )
                
                if response.status_code == 204:
                    logger.info("Webhook sent successfully")
                    return True
                else:
                    logger.error(f"Webhook failed: {response.status_code} - {response.text}")
                    return False
                    
        except Exception as e:
            logger.error(f"Failed to send webhook: {e}")
            return False


# 全域通知管理器實例
notification_manager = NotificationManager()
