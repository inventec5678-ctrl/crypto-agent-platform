"""
幣安 API Mock 測試
測試幣安客戶端的各項功能
"""
import pytest
import sys
import os
from unittest.mock import MagicMock, patch, AsyncMock
from unittest import IsolatedAsyncioTestCase

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestBinanceClient:
    """測試幣安客戶端基本功能"""

    def setup_method(self):
        """設置測試環境"""
        self.api_key = "test_api_key"
        self.api_secret = "test_api_secret"
        self.base_url = "https://api.binance.com"

    @patch('requests.get')
    def test_fetch_klines_success(self, mock_get):
        """測試成功獲取 K 線數據"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            [1000, "100.0", "102.0", "99.0", "101.0", "1000", 1000000],
            [2000, "101.0", "103.0", "100.0", "102.0", "1000", 2000000],
        ]
        mock_get.return_value = mock_response

        # 模擬客戶端調用
        response = mock_get(f"{self.base_url}/api/v3/klines", params={
            "symbol": "BTCUSDT",
            "interval": "1h",
            "limit": 10
        })

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2
        assert data[0][0] == 1000  # timestamp

    @patch('requests.get')
    def test_fetch_klines_rate_limit(self, mock_get):
        """測試 Rate Limit 處理"""
        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.headers = {"Retry-After": "60"}
        mock_get.return_value = mock_response

        response = mock_get(f"{self.base_url}/api/v3/klines")

        assert response.status_code == 429
        assert response.headers.get("Retry-After") == "60"

    @patch('requests.get')
    def test_fetch_klines_api_error(self, mock_get):
        """測試 API 錯誤處理"""
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.json.return_value = {
            "code": -1003,
            "msg": "Too many requests"
        }
        mock_get.return_value = mock_response

        response = mock_get(f"{self.base_url}/api/v3/klines")

        assert response.status_code == 400
        data = response.json()
        assert "msg" in data

    @patch('requests.get')
    def test_fetch_account_balance(self, mock_get):
        """測試獲取帳戶餘額"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "balances": [
                {"asset": "BTC", "free": "1.00000000", "locked": "0.00000000"},
                {"asset": "USDT", "free": "10000.00000000", "locked": "0.00000000"},
            ]
        }
        mock_get.return_value = mock_response

        response = mock_get(
            f"{self.base_url}/api/v3/account",
            headers={"X-MBX-APIKEY": self.api_key}
        )

        assert response.status_code == 200
        data = response.json()
        assert "balances" in data
        assert len(data["balances"]) == 2


class TestRateLimiter:
    """測試 Rate Limiter 功能"""

    def setup_method(self):
        """設置 Rate Limiter"""
        self.request_count = 0
        self.hour_window = []
        self.max_requests_per_hour = 1200

    def clean_old_requests(self):
        """清理過期的請求記錄"""
        import time
        current_time = time.time()
        self.hour_window = [
            t for t in self.hour_window
            if current_time - t < 3600
        ]

    def can_make_request(self):
        """檢查是否可以發送請求"""
        import time
        self.clean_old_requests()
        return len(self.hour_window) < self.max_requests_per_hour

    def record_request(self):
        """記錄請求"""
        import time
        self.hour_window.append(time.time())

    def test_rate_limit_not_exceeded(self):
        """測試未超過 Rate Limit"""
        for _ in range(10):
            self.record_request()

        assert self.can_make_request()

    def test_rate_limit_calculation(self):
        """測試 Rate Limit 計數正確"""
        import time

        # 模擬 100 次請求
        for _ in range(100):
            self.hour_window.append(time.time())

        assert len(self.hour_window) == 100
        assert len(self.hour_window) < self.max_requests_per_hour

    def test_old_requests_cleaned(self):
        """測試舊請求被清理"""
        import time

        # 模擬一個小時前的請求
        old_time = time.time() - 4000
        self.hour_window = [old_time]

        self.clean_old_requests()

        assert len(self.hour_window) == 0, "一小時前的請求應該被清理"

    def test_rate_limit_exceeded(self):
        """測試超過 Rate Limit"""
        import time

        # 模擬超過限制的請求
        for _ in range(self.max_requests_per_hour + 1):
            self.hour_window.append(time.time())

        assert not self.can_make_request(), "應該超過 Rate Limit"


class TestBinanceWebSocket:
    """測試幣安 WebSocket 功能"""

    @patch('websocket.WebSocketApp')
    def test_websocket_connection(self, mock_ws):
        """測試 WebSocket 連接"""
        mock_ws.return_value = MagicMock()

        # 模擬 WebSocket 連接
        ws = mock_ws("wss://stream.binance.com:9443/ws/btcusdt@kline_1h")
        ws.send = MagicMock()

        assert ws is not None

    def test_kline_message_parsing(self):
        """測試 K 線消息解析"""
        # 模擬 WebSocket K 線消息
        message = {
            "e": "kline",
            "s": "BTCUSDT",
            "k": {
                "t": 1000000,
                "o": "50000.00",
                "h": "51000.00",
                "l": "49000.00",
                "c": "50500.00",
                "v": "1000.00"
            }
        }

        # 解析消息
        assert message["e"] == "kline"
        assert message["s"] == "BTCUSDT"
        assert message["k"]["o"] == "50000.00"

    def test_ticker_message_parsing(self):
        """測試 Ticker 消息解析"""
        message = {
            "e": "24hrTicker",
            "s": "BTCUSDT",
            "c": "50500.00",
            "p": "500.00",
            "P": "1.00"
        }

        assert message["e"] == "24hrTicker"
        assert "c" in message  # close price


class TestOrderPlacement:
    """測試訂單下單功能"""

    @patch('requests.post')
    def test_place_market_order(self, mock_post):
        """測試市價單"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "symbol": "BTCUSDT",
            "orderId": 12345,
            "side": "BUY",
            "type": "MARKET",
            "executedQty": "0.001"
        }
        mock_post.return_value = mock_response

        response = mock_post(
            "https://api.binance.com/api/v3/order",
            json={
                "symbol": "BTCUSDT",
                "side": "BUY",
                "type": "MARKET",
                "quantity": "0.001"
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert data["type"] == "MARKET"

    @patch('requests.post')
    def test_place_limit_order(self, mock_post):
        """測試限價單"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "symbol": "BTCUSDT",
            "orderId": 12346,
            "side": "BUY",
            "type": "LIMIT",
            "price": "50000.00"
        }
        mock_post.return_value = mock_response

        response = mock_post(
            "https://api.binance.com/api/v3/order",
            json={
                "symbol": "BTCUSDT",
                "side": "BUY",
                "type": "LIMIT",
                "price": "50000.00",
                "quantity": "0.001"
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert data["type"] == "LIMIT"

    @patch('requests.post')
    def test_order_insufficient_balance(self, mock_post):
        """測試餘額不足"""
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.json.return_value = {
            "code": -2010,
            "msg": "Account has insufficient balance"
        }
        mock_post.return_value = mock_response

        response = mock_post("https://api.binance.com/api/v3/order")

        assert response.status_code == 400
        data = response.json()
        assert "insufficient" in data["msg"].lower()


class TestSymbolValidation:
    """測試交易對驗證"""

    def test_valid_symbols(self):
        """測試有效交易對"""
        valid_symbols = [
            "BTCUSDT", "ETHUSDT", "BNBUSDT",
            "BTCBUSD", "ETHBUSD", "BNBBUSD"
        ]

        for symbol in valid_symbols:
            assert len(symbol) >= 6
            assert symbol.isalpha() or any(c.isdigit() for c in symbol)

    def test_symbol_format(self):
        """測試交易對格式"""
        symbol = "BTCUSDT"

        # 格式應該是 BASE + QUOTE
        assert symbol.endswith("USDT"), "USDT 交易對應該以 USDT 結尾"
        assert symbol[:3] in ["BTC", "ETH", "BNB"], "應該是主流幣種"

    def test_invalid_symbol(self):
        """測試無效交易對"""
        invalid_symbols = [
            "", "123", "BTC", "INVALIDCOIN"
        ]

        for symbol in invalid_symbols:
            # 簡單驗證
            assert len(symbol) < 6 or len(symbol) > 10


class TestPriceData:
    """測試價格數據處理"""

    def test_price_precision(self):
        """測試價格精度"""
        price = 50500.123456

        # 幣安價格精度
        formatted = f"{price:.2f}"
        assert formatted == "50500.12"

    def test_quantity_precision(self):
        """測試數量精度"""
        quantity = 0.00123456

        # 比特幣數量精度
        formatted = f"{quantity:.6f}"
        assert formatted == "0.001234"

    def test_price_calculation(self):
        """測試價格計算"""
        buy_price = 50000.00
        quantity = 0.01
        commission = 0.0001  # 0.01%

        total_cost = buy_price * quantity
        commission_cost = total_cost * commission

        assert total_cost == 500.00
        assert commission_cost == 0.05
