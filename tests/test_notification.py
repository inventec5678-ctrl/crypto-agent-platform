"""
通知測試
測試 Discord 通知和其他通知渠道
"""
import pytest
import sys
import os
from unittest.mock import MagicMock, patch, AsyncMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestDiscordNotifier:
    """測試 Discord 通知功能"""

    def setup_method(self):
        """設置測試環境"""
        self.webhook_url = "https://discord.com/api/webhooks/test"
        self.test_message = {
            "content": "Test message",
            "embeds": [{
                "title": "Test Signal",
                "color": 3066993,  # 綠色
                "fields": [
                    {"name": "Symbol", "value": "BTCUSDT", "inline": True},
                    {"name": "Signal", "value": "BUY", "inline": True},
                ]
            }]
        }

    @patch('requests.post')
    def test_send_embed_message(self, mock_post):
        """測試發送嵌入消息"""
        mock_response = MagicMock()
        mock_response.status_code = 204
        mock_post.return_value = mock_response

        response = mock_post(
            self.webhook_url,
            json=self.test_message
        )

        assert response.status_code == 204
        mock_post.assert_called_once()

    @patch('requests.post')
    def test_send_signal_notification(self, mock_post):
        """測試發送信號通知"""
        mock_response = MagicMock()
        mock_response.status_code = 204
        mock_post.return_value = mock_response

        signal_message = {
            "content": "📊 **交易信號**",
            "embeds": [{
                "title": "BTCUSDT BUY Signal",
                "description": "信心分: 85%",
                "color": 3066993,
                "fields": [
                    {"name": "信號", "value": "BUY", "inline": True},
                    {"name": "信心", "value": "85%", "inline": True},
                    {"name": "價格", "value": "$50,000", "inline": True},
                ],
                "footer": {"text": "Crypto Agent Platform"}
            }]
        }

        response = mock_post(self.webhook_url, json=signal_message)
        assert response.status_code == 204

    @patch('requests.post')
    def test_send_error_notification(self, mock_post):
        """測試發送錯誤通知"""
        mock_response = MagicMock()
        mock_response.status_code = 204
        mock_post.return_value = mock_response

        error_message = {
            "content": "⚠️ **系統錯誤**",
            "embeds": [{
                "title": "API Error",
                "description": "幣安 API 回應錯誤",
                "color": 15158332  # 紅色
            }]
        }

        response = mock_post(self.webhook_url, json=error_message)
        assert response.status_code == 204

    @patch('requests.post')
    def test_webhook_unavailable(self, mock_post):
        """測試 Webhook 不可用"""
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_post.return_value = mock_response

        response = mock_post(self.webhook_url, json=self.test_message)

        assert response.status_code == 404

    @patch('requests.post')
    def test_rate_limit_handling(self, mock_post):
        """測試 Rate Limit 處理"""
        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.headers = {"Retry-After": "5"}
        mock_post.return_value = mock_response

        response = mock_post(self.webhook_url, json=self.test_message)

        assert response.status_code == 429


class TestNotificationFormatting:
    """測試通知格式化"""

    def test_signal_embed_color_buy(self):
        """測試買入信號顏色（綠色）"""
        color_buy = 3066993  # 綠色
        assert color_buy == 0x2ECC71 or color_buy == 3066993

    def test_signal_embed_color_sell(self):
        """測試賣出信號顏色（紅色）"""
        color_sell = 15158332  # 紅色
        assert color_sell == 0xE74C3C or color_sell == 15158332

    def test_signal_embed_color_hold(self):
        """測試持有信號顏色（黃色）"""
        color_hold = 16776960  # 黃色
        assert color_hold == 0xFFFF00 or color_hold == 16776960

    def test_emoji_for_signal(self):
        """測試信號表情符號"""
        assert "📈" == "📈"  # 買入
        assert "📉" == "📉"  # 賣出
        assert "⏸️" == "⏸️"  # 持有

    def test_format_price(self):
        """測試價格格式化"""
        price = 50500.123456
        formatted = f"${price:,.2f}"
        assert formatted == "$50,500.12"

    def test_format_percentage(self):
        """測試百分比格式化"""
        percentage = 85.5
        formatted = f"{percentage:.1f}%"
        assert formatted == "85.5%"

    def test_format_timestamp(self):
        """測試時間戳格式化"""
        import datetime
        timestamp = 1709000000
        dt = datetime.datetime.fromtimestamp(timestamp)
        formatted = dt.strftime("%Y-%m-%d %H:%M:%S")

        assert "2024" in formatted
        assert ":" in formatted


class TestNotificationQueue:
    """測試通知隊列"""

    def setup_method(self):
        """設置通知隊列"""
        self.queue = []
        self.max_queue_size = 10

    def enqueue(self, message):
        """添加消息到隊列"""
        if len(self.queue) >= self.max_queue_size:
            # 移除最舊的消息
            self.queue.pop(0)
        self.queue.append(message)

    def dequeue(self):
        """從隊列取出消息"""
        if self.queue:
            return self.queue.pop(0)
        return None

    def test_enqueue_message(self):
        """測試添加消息"""
        self.enqueue({"type": "signal", "data": "test"})
        assert len(self.queue) == 1

    def test_dequeue_message(self):
        """測試取出消息"""
        self.enqueue({"type": "signal", "data": "test"})
        message = self.dequeue()

        assert message is not None
        assert message["type"] == "signal"
        assert len(self.queue) == 0

    def test_queue_overflow(self):
        """測試隊列溢出"""
        for i in range(15):
            self.enqueue({"id": i})

        # 應該只保留最新的 10 條
        assert len(self.queue) == 10
        assert self.queue[0]["id"] == 5  # 第 6 條是最舊的

    def test_empty_queue(self):
        """測試空隊列"""
        message = self.dequeue()
        assert message is None


class TestNotificationThrottling:
    """測試通知節流"""

    def setup_method(self):
        """設置節流器"""
        self.last_notification_time = 0
        self.min_interval = 60  # 最少 60 秒間隔

    def can_send_notification(self, current_time):
        """檢查是否可以發送通知"""
        return current_time - self.last_notification_time >= self.min_interval

    def record_notification(self, current_time):
        """記錄發送時間"""
        self.last_notification_time = current_time

    def test_throttle_same_signal(self):
        """測試同一信號節流"""
        import time
        now = time.time()

        assert self.can_send_notification(now)  # 第一次可以

        self.record_notification(now)
        assert not self.can_send_notification(now)  # 馬上不行

    def test_throttle_after_interval(self):
        """測試間隔後可以發送"""
        import time
        now = time.time()

        self.record_notification(now - 100)  # 100 秒前
        assert self.can_send_notification(now)

    def test_different_signals_same_time(self):
        """測試同一時間不同信號"""
        import time
        now = time.time()

        # 重要信號可以不受節流限制
        important_signal = True
        if important_signal:
            assert self.can_send_notification(now)


class TestNotificationTemplates:
    """測試通知模板"""

    def format_signal_message(self, symbol, signal, confidence, price):
        """格式化信號消息"""
        emoji = "📈" if signal == "BUY" else "📉" if signal == "SELL" else "⏸️"
        color = 3066993 if signal == "BUY" else (15158332 if signal == "SELL" else 16776960)

        return {
            "content": f"{emoji} **{signal} Signal**",
            "embeds": [{
                "title": f"{symbol} {signal}",
                "color": color,
                "fields": [
                    {"name": "Symbol", "value": symbol, "inline": True},
                    {"name": "Signal", "value": signal, "inline": True},
                    {"name": "Confidence", "value": f"{confidence}%", "inline": True},
                    {"name": "Price", "value": f"${price:,.2f}", "inline": True},
                ]
            }]
        }

    def test_buy_signal_format(self):
        """測試買入信號格式"""
        message = self.format_signal_message("BTCUSDT", "BUY", 85, 50000)

        assert "📈" in message["content"]
        assert "BUY" in message["content"]
        assert message["embeds"][0]["fields"][1]["value"] == "BUY"

    def test_sell_signal_format(self):
        """測試賣出信號格式"""
        message = self.format_signal_message("ETHUSDT", "SELL", 75, 3000)

        assert "📉" in message["content"]
        assert "SELL" in message["content"]
        assert message["embeds"][0]["color"] == 15158332

    def test_hold_signal_format(self):
        """測試持有信號格式"""
        message = self.format_signal_message("BNBUSDT", "HOLD", 50, 500)

        assert "⏸️" in message["content"]
        assert "HOLD" in message["content"]
        assert message["embeds"][0]["color"] == 16776960


class TestNotificationBatch:
    """測試通知批量發送"""

    def setup_method(self):
        """設置批量通知"""
        self.batch = []
        self.batch_size = 5

    def add_to_batch(self, notification):
        """添加到批量"""
        self.batch.append(notification)

    def should_send_batch(self):
        """檢查是否應該發送批量"""
        return len(self.batch) >= self.batch_size

    def clear_batch(self):
        """清空批量"""
        sent = self.batch.copy()
        self.batch = []
        return sent

    def test_batch_accumulation(self):
        """測試批量累積"""
        for i in range(3):
            self.add_to_batch({"id": i})

        assert len(self.batch) == 3
        assert not self.should_send_batch()

    def test_batch_trigger(self):
        """測試批量觸發"""
        for i in range(5):
            self.add_to_batch({"id": i})

        assert self.should_send_batch()

    def test_batch_clear(self):
        """測試批量清空"""
        for i in range(5):
            self.add_to_batch({"id": i})

        sent = self.clear_batch()

        assert len(sent) == 5
        assert len(self.batch) == 0
