"""
AI 信心指數測試
測試 AI 分析師的信號評分邏輯
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unittest.mock import MagicMock, patch


class TestAIConfidenceScore:
    """測試 AI 信心分數計算"""

    def setup_method(self):
        """設置測試環境"""
        self.default_config = {
            "ma_weight": 0.3,
            "rsi_weight": 0.25,
            "macd_weight": 0.25,
            "volume_weight": 0.2,
        }

    def calculate_confidence(self, signals, config):
        """
        計算信心分數
        signals: dict，包含各指標信號
        config: 權重配置
        返回: 0-100 的分數
        """
        total_weight = sum(config.values())
        score = 0

        # MA 信號評分
        if signals.get("ma_bullish"):
            score += config["ma_weight"] * 100
        elif signals.get("ma_bearish"):
            score += config["ma_weight"] * 30

        # RSI 信號評分
        rsi = signals.get("rsi", 50)
        if rsi < 30:  # 超賣 = 買入信號
            score += config["rsi_weight"] * 100
        elif rsi > 70:  # 超買 = 賣出信號
            score += config["rsi_weight"] * 30
        else:
            score += config["rsi_weight"] * (100 - abs(rsi - 50) * 2)

        # MACD 信號評分
        if signals.get("macd_bullish"):
            score += config["macd_weight"] * 100
        elif signals.get("macd_bearish"):
            score += config["macd_weight"] * 30

        # Volume 信號評分
        if signals.get("volume_surge"):
            score += config["volume_weight"] * 100
        else:
            score += config["volume_weight"] * 50

        return min(100, max(0, score / total_weight * 100))

    def test_bullish_signals_high_confidence(self):
        """測試強烈買入信號應該有高信心分"""
        signals = {
            "ma_bullish": True,
            "rsi": 25,  # 超賣
            "macd_bullish": True,
            "volume_surge": True,
        }
        score = self.calculate_confidence(signals, self.default_config)

        assert score >= 85, f"強烈買入信號信心分應該 >= 85，實際: {score}"

    def test_bearish_signals_low_confidence(self):
        """測試強烈賣出信號應該有低信心分"""
        signals = {
            "ma_bearish": True,
            "rsi": 80,  # 超買
            "macd_bearish": True,
            "volume_surge": True,
        }
        score = self.calculate_confidence(signals, self.default_config)

        # 這是看跌信號，分數應該低
        assert score <= 30, f"強烈賣出信號信心分應該 <= 30，實際: {score}"

    def test_mixed_signals_moderate_confidence(self):
        """測試混合信號應該有中等信心分"""
        signals = {
            "ma_bullish": True,
            "rsi": 50,  # 中性
            "macd_bearish": False,
            "volume_surge": False,
        }
        score = self.calculate_confidence(signals, self.default_config)

        assert 40 <= score <= 70, f"混合信號信心分應該在 40-70，實際: {score}"

    def test_confidence_bounds(self):
        """測試信心分數邊界"""
        # 最低分
        signals_low = {
            "ma_bearish": True,
            "rsi": 100,
            "macd_bearish": True,
            "volume_surge": False,
        }
        score_low = self.calculate_confidence(signals_low, self.default_config)
        assert 0 <= score_low <= 100, f"分數应该在 0-100 范围，实际: {score_low}"

        # 最高分
        signals_high = {
            "ma_bullish": True,
            "rsi": 0,
            "macd_bullish": True,
            "volume_surge": True,
        }
        score_high = self.calculate_confidence(signals_high, self.default_config)
        assert 0 <= score_high <= 100, f"分數应该在 0-100 范围，实际: {score_high}"

    def test_neutral_signals(self):
        """測試中性信號"""
        signals = {
            "ma_bullish": False,
            "ma_bearish": False,
            "rsi": 50,
            "macd_bullish": False,
            "macd_bearish": False,
            "volume_surge": False,
        }
        score = self.calculate_confidence(signals, self.default_config)

        # 中性信號應該有中等分數
        assert 40 <= score <= 60, f"中性信號信心分應該在 40-60，實際: {score}"

    def test_custom_weights(self):
        """測試自定義權重"""
        custom_config = {
            "ma_weight": 0.5,
            "rsi_weight": 0.1,
            "macd_weight": 0.1,
            "volume_weight": 0.3,
        }

        signals = {
            "ma_bullish": True,
            "rsi": 50,
            "macd_bullish": False,
            "volume_surge": True,
        }

        default_score = self.calculate_confidence(signals, self.default_config)
        custom_score = self.calculate_confidence(signals, custom_config)

        # 自定義權重強調 MA 和 Volume，應該比默認高
        assert custom_score > default_score, "自定義權重應該產生不同的分數"


class TestSignalGeneration:
    """測試信號生成邏輯"""

    def generate_signal(self, confidence, threshold_buy=70, threshold_sell=30):
        """根據信心分生成交易信號"""
        if confidence >= threshold_buy:
            return "BUY"
        elif confidence <= threshold_sell:
            return "SELL"
        else:
            return "HOLD"

    def test_high_confidence_buy_signal(self):
        """測試高信心買入信號"""
        assert self.generate_signal(85) == "BUY"

    def test_low_confidence_sell_signal(self):
        """測試低信心賣出信號"""
        assert self.generate_signal(20) == "SELL"

    def test_moderate_confidence_hold_signal(self):
        """測試中等信心持有信號"""
        assert self.generate_signal(50) == "HOLD"

    def test_threshold_boundaries(self):
        """測試閾值邊界"""
        assert self.generate_signal(70) == "BUY", "等於閾值應該是 BUY"
        assert self.generate_signal(30) == "SELL", "等於閾值應該是 SELL"

    def test_custom_thresholds(self):
        """測試自定義閾值"""
        assert self.generate_signal(60, threshold_buy=80, threshold_sell=20) == "HOLD"
        assert self.generate_signal(85, threshold_buy=80, threshold_sell=20) == "BUY"


class TestAIAnalystIntegration:
    """AI 分析師整合測試"""

    def test_multi_timeframe_analysis(self):
        """測試多時間框架分析"""
        timeframe_signals = {
            "1h": {"confidence": 80, "signal": "BUY"},
            "4h": {"confidence": 65, "signal": "BUY"},
            "1d": {"confidence": 45, "signal": "HOLD"},
        }

        # 計算總體信心（加權平均）
        weights = {"1h": 0.2, "4h": 0.3, "1h": 0.5}
        total_confidence = sum(
            s["confidence"] * w
            for s, w in zip(timeframe_signals.values(), weights.values())
        ) / sum(weights.values())

        assert 50 <= total_confidence <= 80, "多時間框架分析應該給出合理的信心分"

    def test_signal_consistency(self):
        """測試信號一致性"""
        signals = ["BUY", "BUY", "HOLD", "BUY", "SELL"]

        buy_count = signals.count("BUY")
        sell_count = signals.count("SELL")

        # 大多數決策
        if buy_count > len(signals) / 2:
            final_signal = "BUY"
        elif sell_count > len(signals) / 2:
            final_signal = "SELL"
        else:
            final_signal = "HOLD"

        assert final_signal == "BUY", "多數 BUY 應該產生 BUY 信號"


class TestConfidenceHistory:
    """測試信心歷史追蹤"""

    def calculate_trend(self, history):
        """從歷史計算趨勢"""
        if len(history) < 2:
            return "STABLE"

        recent_avg = sum(history[-3:]) / min(3, len(history))
        older_avg = sum(history[:-3]) / max(1, len(history) - 3)

        diff = recent_avg - older_avg
        if diff > 10:
            return "IMPROVING"
        elif diff < -10:
            return "DECLINING"
        else:
            return "STABLE"

    def test_improving_trend(self):
        """測試上升趨勢"""
        history = [50, 55, 60, 65, 70, 75, 80]
        trend = self.calculate_trend(history)
        assert trend == "IMPROVING"

    def test_declining_trend(self):
        """測試下降趨勢"""
        history = [80, 75, 70, 65, 60, 55, 50]
        trend = self.calculate_trend(history)
        assert trend == "DECLINING"

    def test_stable_trend(self):
        """測試穩定趨勢"""
        history = [50, 52, 48, 51, 49, 50, 51]
        trend = self.calculate_trend(history)
        assert trend == "STABLE"

    def test_insufficient_history(self):
        """測試歷史數據不足"""
        history = [50]
        trend = self.calculate_trend(history)
        assert trend == "STABLE"
