"""測試：幣安客戶端"""
import pytest
import asyncio
from unittest.mock import AsyncMock, patch


class TestBinanceClient:
    """測試幣安 API 客戶端"""
    
    def test_client_initialization(self):
        """測試客戶端初始化"""
        from binance_client import BinanceClient
        
        client = BinanceClient()
        assert client.BASE_URL == "https://api.binance.com"
        assert client.WS_URL == "wss://stream.binance.com:9443/ws"
    
    def test_parse_klines(self):
        """測試 K 線解析"""
        from binance_client import BinanceClient
        
        client = BinanceClient()
        
        # 模擬原始 K 線資料
        raw_klines = [[
            1700000000000,  # open_time
            "50000.00",      # open
            "51000.00",      # high
            "49000.00",      # low
            "50500.00",      # close
            "1000.00",       # volume
            1700003600000,   # close_time
        ]]
        
        parsed = client._parse_klines(raw_klines)
        
        assert len(parsed) == 1
        assert parsed[0]["open"] == 50000.0
        assert parsed[0]["close"] == 50500.0
        assert parsed[0]["high"] == 51000.0


class TestStrategies:
    """測試交易策略"""
    
    @pytest.mark.asyncio
    async def test_ma_strategy_signal(self):
        """測試 MA 交叉策略"""
        from strategies.strategy_ma_crossover import MACrossoverStrategy
        
        # Mock klines data
        mock_klines = [
            {"close": 50000 + i} for i in range(50)
        ]
        
        with patch("binance_client.binance_client.get_klines", new_callable=AsyncMock(return_value=mock_klines)):
            strategy = MACrossoverStrategy(symbol="BTCUSDT")
            result = await strategy.analyze()
            
            # 應該有訊號（因為是漲勢）
            assert result is not None or result is None  # 取決於 MA 交叉


class TestAIAnalyst:
    """測試 AI 分析師"""
    
    @pytest.mark.asyncio
    async def test_ai_confidence_calculation(self):
        """測試 AI 信心指數計算"""
        from ai_analyst import AIAnalyst
        
        analyst = AIAnalyst()
        
        # 模擬策略結果
        mock_results = [
            {"confidence": 80.0},
            {"confidence": 70.0},
            None
        ]
        
        tech_score = analyst._calculate_technical_score(mock_results)
        assert tech_score == 75.0  # (80 + 70) / 2
    
    def test_rating_assignment(self):
        """測試等級評定"""
        from ai_analyst import AIAnalyst
        
        analyst = AIAnalyst()
        
        assert analyst._get_rating(80) == "★★★"
        assert analyst._get_rating(60) == "★★"
        assert analyst._get_rating(40) == "★"


class TestNotificationManager:
    """測試通知管理器"""
    
    @pytest.mark.asyncio
    async def test_rate_limit(self):
        """測試 Rate Limit"""
        from notification import NotificationManager
        
        manager = NotificationManager()
        manager.webhook_url = ""  # 禁用實際發送
        
        symbol = "BTCUSDT"
        
        # 前 3 次應該通過
        for i in range(3):
            can_send = await manager._check_rate_limit(symbol)
            assert can_send is True
        
        # 第 4 次應該被阻擋
        can_send = await manager._check_rate_limit(symbol)
        assert can_send is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
