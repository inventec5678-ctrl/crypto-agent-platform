"""AI 信心指數分析師"""
import logging
from typing import Optional
from datetime import datetime
from binance_client import binance_client

logger = logging.getLogger(__name__)


class AIAnalyst:
    """
    AI 信心指數分析
    
    四因子權重：
    - 技術分析：35%
    - 市場情緒：25%
    - 社群指標：20%
    - 異常偵測：20%
    """
    
    def __init__(self):
        self.name = "AI_Analyst"
    
    async def analyze(self, symbol: str, strategies_results: list[dict]) -> dict:
        """
        整合分析，輸出信心指數
        
        Args:
            symbol: 交易對
            strategies_results: 各策略的分析結果
        
        Returns:
            信心指數報告
        """
        # 計算各因子分數
        technical_score = self._calculate_technical_score(strategies_results)
        sentiment_score = await self._calculate_sentiment_score(symbol)
        social_score = await self._calculate_social_score(symbol)
        anomaly_score = await self._calculate_anomaly_score(symbol, strategies_results)
        
        # 權重計算
        final_score = (
            technical_score * 0.35 +
            sentiment_score * 0.25 +
            social_score * 0.20 +
            anomaly_score * 0.20
        )
        
        # 決定等級
        rating = self._get_rating(final_score)
        
        return {
            "symbol": symbol,
            "confidence_score": round(final_score, 2),
            "rating": rating,
            "factors": {
                "technical": round(technical_score, 2),
                "sentiment": round(sentiment_score, 2),
                "social": round(social_score, 2),
                "anomaly": round(anomaly_score, 2),
            },
            "weights": {
                "technical": "35%",
                "sentiment": "25%",
                "social": "20%",
                "anomaly": "20%",
            },
            "strategies_triggered": len([s for s in strategies_results if s]),
            "timestamp": datetime.now().isoformat(),
        }
    
    def _calculate_technical_score(self, strategies_results: list[dict]) -> float:
        """計算技術分析分數"""
        if not strategies_results:
            return 50.0  # 無策略觸發，返回中性分數
        
        scores = []
        for result in strategies_results:
            if result and "confidence" in result:
                scores.append(result["confidence"])
        
        if not scores:
            return 50.0
        
        # 平均分數
        return sum(scores) / len(scores)
    
    async def _calculate_sentiment_score(self, symbol: str) -> float:
        """
        計算市場情緒分數
        
        模擬：實際專案可串接恐懼貪婪指數 API 或其他情緒數據源
        """
        # 嘗試取得近期價格變化
        try:
            klines = await binance_client.get_klines(symbol, interval="1h", limit=24)
            if len(klines) >= 2:
                # 計算 24h 變化率
                price_change = (klines[-1]["close"] - klines[0]["close"]) / klines[0]["close"] * 100
                
                # 簡單情緒模型：漲幅 > 2% 偏貪婪，跌幅 > 2% 偏恐懼
                if price_change > 2:
                    return min(50 + price_change * 5, 100)
                elif price_change < -2:
                    return max(50 + price_change * 5, 0)
        except Exception as e:
            logger.warning(f"Failed to calculate sentiment: {e}")
        
        return 50.0  # 中性
    
    async def _calculate_social_score(self, symbol: str) -> float:
        """
        計算社群指標分數
        
        模擬：實際專案可串接 Twitter/Reddit API 取得社群討論熱度
        """
        # 這裡返回模擬分數，實際應串接社群數據 API
        return 50.0
    
    async def _calculate_anomaly_score(self, symbol: str, strategies_results: list[dict]) -> float:
        """
        計算異常偵測分數
        
        檢查：
        - 成交量異常
        - 價格波動異常
        - 多策略共振
        """
        anomaly_score = 50.0
        
        try:
            klines = await binance_client.get_klines(symbol, interval="1h", limit=50)
            
            if len(klines) >= 20:
                closes = [k["close"] for k in klines]
                volumes = [k["volume"] for k in klines]
                
                # 計算均值和標準差
                avg_price = sum(closes[-20:]) / 20
                std_price = (sum((c - avg_price) ** 2 for c in closes[-20:]) / 20) ** 0.5
                avg_volume = sum(volumes[-20:]) / 20
                
                # 最新價格偏離
                latest_price = closes[-1]
                price_deviation = abs(latest_price - avg_price) / avg_price
                
                # 最新成交量異常
                latest_volume = volumes[-1]
                volume_ratio = latest_volume / avg_volume if avg_volume > 0 else 1
                
                # 異常偵測加分
                if price_deviation > 0.02:  # 價格偏離 > 2%
                    anomaly_score += 10
                if volume_ratio > 2:  # 成交量放大 > 2倍
                    anomaly_score += 10
                
                # 多策略共振加分
                triggered_count = len([s for s in strategies_results if s])
                if triggered_count >= 2:
                    anomaly_score += 15
                
        except Exception as e:
            logger.warning(f"Failed to calculate anomaly score: {e}")
        
        return min(anomaly_score, 100)
    
    def _get_rating(self, score: float) -> str:
        """根據分數取得等級"""
        if score >= 75:
            return "★★★"
        elif score >= 50:
            return "★★"
        else:
            return "★"
