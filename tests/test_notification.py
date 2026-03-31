"""
通知服務單元測試
"""
import pytest
import sys
import os
from unittest.mock import MagicMock, patch, AsyncMock
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestNotificationManager:
    """測試通知管理器"""

    def setup_method(self):
        """設置測試環境"""
        self.webhook_url = "https://discord.com/api/webhooks/test/webhook"

    def _make_signal_data(self):
        return {
            "symbol": "BTCUSDT",
            "signal": "BUY",
            "direction": "LONG",
            "price": 50000.0,
            "confidence": 75.0,
        }

    def _make_ai_analysis(self):
        return {
            "confidence_score": 72.5,
            "rating": "★★",
            "factors": {
                "technical": 75.0,
                "sentiment": 70.0,
                "social": 65.0,
                "anomaly": 80.0,
            },
        }

    @patch('httpx.AsyncClient')
    async def test_send_signal_success(self, mock_client_class):
        """測試發送信號通知成功"""
        mock_response = MagicMock()
        mock_response.status_code = 204

        mock_client_instance = AsyncMock()
        mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
        mock_client_instance.__aexit__ = AsyncMock(return_value=None)
        mock_client_instance.post = AsyncMock(return_value=mock_response)
        mock_client_class.return_value = mock_client_instance

        from notification import NotificationManager
        manager = NotificationManager()
        manager.webhook_url = self.webhook_url

        result = await manager.send_signal(
            self._make_signal_data()["symbol"],
            self._make_signal_data(),
            self._make_ai_analysis(),
        )

        assert result is True

    @patch('httpx.AsyncClient')
    async def test_send_signal_webhook_unavailable(self, mock_client_class):
        """測試 Webhook 不可用"""
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.text = "Not found"

        mock_client_instance = AsyncMock()
        mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
        mock_client_instance.__aexit__ = AsyncMock(return_value=None)
        mock_client_instance.post = AsyncMock(return_value=mock_response)
        mock_client_class.return_value = mock_client_instance

        from notification import NotificationManager
        manager = NotificationManager()
        manager.webhook_url = self.webhook_url

        result = await manager.send_signal(
            "BTCUSDT",
            self._make_signal_data(),
            self._make_ai_analysis(),
        )

        assert result is False

    @patch('httpx.AsyncClient')
    async def test_send_signal_rate_limited(self, mock_client_class):
        """測試 Rate Limit 攔截"""
        mock_response = MagicMock()
        mock_response.status_code = 429

        mock_client_instance = AsyncMock()
        mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
        mock_client_instance.__aexit__ = AsyncMock(return_value=None)
        mock_client_instance.post = AsyncMock(return_value=mock_response)
        mock_client_class.return_value = mock_client_instance

        from notification import NotificationManager
        manager = NotificationManager()
        manager.webhook_url = self.webhook_url

        # 前 3 次應該可以發送（MAX_DAILY_NOTIFICATIONS = 3）
        for _ in range(3):
            result = await manager.send_signal(
                "BTCUSDT",
                self._make_signal_data(),
                self._make_ai_analysis(),
            )
            assert result is True

        # 第 4 次應該被 Rate Limit 攔截
        result = await manager.send_signal(
            "BTCUSDT",
            self._make_signal_data(),
            self._make_ai_analysis(),
        )
        assert result is False

    @patch('httpx.AsyncClient')
    async def test_send_signal_no_webhook(self, mock_client_class):
        """測試無 Webhook 時不發送"""
        from notification import NotificationManager
        manager = NotificationManager()
        manager.webhook_url = ""

        result = await manager.send_signal(
            "BTCUSDT",
            self._make_signal_data(),
            self._make_ai_analysis(),
        )

        assert result is False
        mock_client_class.assert_not_called()

    @patch('httpx.AsyncClient')
    async def test_send_status(self, mock_client_class):
        """測試發送狀態更新"""
        mock_response = MagicMock()
        mock_response.status_code = 204

        mock_client_instance = AsyncMock()
        mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
        mock_client_instance.__aexit__ = AsyncMock(return_value=None)
        mock_client_instance.post = AsyncMock(return_value=mock_response)
        mock_client_class.return_value = mock_client_instance

        from notification import NotificationManager
        manager = NotificationManager()
        manager.webhook_url = self.webhook_url

        result = await manager.send_status("System is running", color=0x3498DB)

        assert result is True

    @patch('httpx.AsyncClient')
    async def test_send_webhook_error_handling(self, mock_client_class):
        """測試 Webhook 錯誤處理"""
        mock_client_instance = AsyncMock()
        mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
        mock_client_instance.__aexit__ = AsyncMock(return_value=None)
        mock_client_instance.post = AsyncMock(side_effect=Exception("Network error"))
        mock_client_class.return_value = mock_client_instance

        from notification import NotificationManager
        manager = NotificationManager()
        manager.webhook_url = self.webhook_url

        result = await manager.send_status("Test message")

        assert result is False


class TestNotificationRateLimit:
    """測試 Rate Limit 邏輯"""

    def test_daily_limit_per_symbol(self):
        """測試每天每檔幣種的限制"""
        from notification import NotificationManager
        import asyncio

        manager = NotificationManager()
        manager.webhook_url = "https://test.webhook"

        # BTC 應該有獨立計數
        for i in range(3):
            can_send = asyncio.get_event_loop().run_until_complete(
                manager._check_rate_limit("BTCUSDT")
            )
            assert can_send is True

        # 第 4 次應該被限制
        can_send = asyncio.get_event_loop().run_until_complete(
            manager._check_rate_limit("BTCUSDT")
        )
        assert can_send is False

        # ETH 應該有獨立的計數（不受 BTC 影響）
        can_send = asyncio.get_event_loop().run_until_complete(
            manager._check_rate_limit("ETHUSDT")
        )
        assert can_send is True

    def test_daily_limit_resets_new_day(self):
        """測試新的一天重置計數"""
        from notification import NotificationManager
        import asyncio

        manager = NotificationManager()
        manager.webhook_url = "https://test.webhook"

        # 模擬今天已發送 3 次
        manager._daily_counts["BTCUSDT"] = {
            "count": 3,
            "date": datetime.now().strftime("%Y-%m-%d"),
        }

        # 今天不能再發送
        can_send = asyncio.get_event_loop().run_until_complete(
            manager._check_rate_limit("BTCUSDT")
        )
        assert can_send is False

        # 模擬昨天
        manager._daily_counts["BTCUSDT"]["date"] = "2020-01-01"

        # 新的一天可以發送
        can_send = asyncio.get_event_loop().run_until_complete(
            manager._check_rate_limit("BTCUSDT")
        )
        assert can_send is True


class TestNotificationEmbedFormatting:
    """測試 Embed 格式化"""

    @patch('httpx.AsyncClient')
    async def test_buy_signal_embed_color(self, mock_client_class):
        """測試買入信號使用綠色"""
        mock_response = MagicMock()
        mock_response.status_code = 204

        captured_request = {}

        async def capture_post(*args, **kwargs):
            captured_request["json"] = kwargs.get("json", {})
            return mock_response

        mock_client_instance = AsyncMock()
        mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
        mock_client_instance.__aexit__ = AsyncMock(return_value=None)
        mock_client_instance.post = capture_post
        mock_client_class.return_value = mock_client_instance

        from notification import NotificationManager
        manager = NotificationManager()
        manager.webhook_url = self.webhook_url

        await manager.send_signal(
            "BTCUSDT",
            {"symbol": "BTCUSDT", "signal": "BUY", "direction": "LONG", "price": 50000.0, "confidence": 75.0},
            {"confidence_score": 72.5, "rating": "★★", "factors": {"technical": 75.0, "sentiment": 70.0, "social": 65.0, "anomaly": 80.0}},
        )

        assert "embeds" in captured_request["json"]
        embed = captured_request["json"]["embeds"][0]
        assert embed["color"] == 0x00FF00  # 綠色

    @patch('httpx.AsyncClient')
    async def test_sell_signal_embed_color(self, mock_client_class):
        """測試賣出信號使用紅色"""
        mock_response = MagicMock()
        mock_response.status_code = 204

        captured_request = {}

        async def capture_post(*args, **kwargs):
            captured_request["json"] = kwargs.get("json", {})
            return mock_response

        mock_client_instance = AsyncMock()
        mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
        mock_client_instance.__aexit__ = AsyncMock(return_value=None)
        mock_client_instance.post = capture_post
        mock_client_class.return_value = mock_client_instance

        from notification import NotificationManager
        manager = NotificationManager()
        manager.webhook_url = self.webhook_url

        await manager.send_signal(
            "BTCUSDT",
            {"symbol": "BTCUSDT", "signal": "SELL", "direction": "SHORT", "price": 50000.0, "confidence": 75.0},
            {"confidence_score": 72.5, "rating": "★★", "factors": {"technical": 75.0, "sentiment": 70.0, "social": 65.0, "anomaly": 80.0}},
        )

        embed = captured_request["json"]["embeds"][0]
        assert embed["color"] == 0xFF0000  # 紅色
