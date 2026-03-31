"""
成交量突破策略 (VolumeBreakoutStrategy)

邏輯：
- 計算成交量移動平均 (volume_ma_period)
- 當成交量突破 MA * volume_multiplier 且價格趨勢符合 trend_period 方向時進場
- 使用 ATR 計算止損 (stop_loss_atr * ATR) 和止盈 (take_profit_atr * ATR)

用法：
    strategy = VolumeBreakoutStrategy(
        symbol="BTCUSDT",
        volume_ma_period=20,
        volume_multiplier=2.0,
        trend_period=5,
        stop_loss_atr=2.0,
        take_profit_atr=3.0,
    )
    signal = await strategy.analyze()
"""

from typing import Optional, Dict, Any, List
from datetime import datetime
import numpy as np

# 動態導入 binance_client（避免循環 import）
_binance_client = None

def _get_binance_client():
    global _binance_client
    if _binance_client is None:
        from binance_client import binance_client
        _binance_client = binance_client
    return _binance_client


class VolumeBreakoutStrategy:
    """
    成交量突破策略
    
    參數：
        symbol: 交易對，如 BTCUSDT
        volume_ma_period: 成交量 MA 週期（默認 20）
        volume_multiplier: 成交量倍數閾值（默認 2.0）
        trend_period: 趨勢判斷週期（默認 5）
        stop_loss_atr: ATR 倍數止損（默認 2.0）
        take_profit_atr: ATR 倍數止盈（默認 3.0）
    """
    
    def __init__(
        self,
        symbol: str = "BTCUSDT",
        volume_ma_period: int = 20,
        volume_multiplier: float = 2.0,
        trend_period: int = 5,
        stop_loss_atr: float = 2.0,
        take_profit_atr: float = 3.0,
        interval: str = "1h",
    ):
        self.symbol = symbol
        self.volume_ma_period = volume_ma_period
        self.volume_multiplier = volume_multiplier
        self.trend_period = trend_period
        self.stop_loss_atr = stop_loss_atr
        self.take_profit_atr = take_profit_atr
        self.interval = interval or "1d"
        
        # 狀態
        self._position: Optional[str] = None  # LONG / SHORT / None
        self._entry_price: float = 0
        self._entry_bar: int = 0
        self._atr: float = 0
    
    def _calculate_atr(self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray) -> float:
        """計算 ATR"""
        if len(closes) < 2:
            return 0
        trs = []
        for i in range(1, min(len(closes), 15)):
            tr = max(
                highs[-i] - lows[-i],
                abs(highs[-i] - closes[-i - 1]),
                abs(lows[-i] - closes[-i - 1])
            )
            trs.append(tr)
        return float(np.mean(trs)) if trs else 0
    
    async def analyze(self) -> Optional[Dict[str, Any]]:
        """
        分析市場並返回交易信號
        
        返回格式：
        {
            "direction": "LONG" | "SHORT" | "FLAT",
            "entry_price": float,
            "stop_loss": float,
            "take_profit": float,
            "reason": str,
            "metadata": {...}
        }
        """
        client = _get_binance_client()
        
        # 獲取足夠的 K 線數據（需要 volume_ma_period + trend_period + 緩衝）
        lookback = self.volume_ma_period + self.trend_period + 20
        klines = await client.get_klines(symbol=self.symbol, interval=self.interval, limit=lookback)
        
        if not klines or len(klines) < lookback:
            return None
        
        # 轉為 numpy arrays（最新在最後）
        closes = np.array([float(k[4]) for k in klines])
        highs = np.array([float(k[2]) for k in klines])
        lows = np.array([float(k[3]) for k in klines])
        volumes = np.array([float(k[5]) for k in klines])
        
        current_close = closes[-1]
        
        # 計算成交量 MA
        vol_ma = float(np.mean(volumes[-(self.volume_ma_period + 1):-1]))  # 排除當前 bar
        current_vol = volumes[-1]
        vol_ratio = current_vol / vol_ma if vol_ma > 0 else 1
        
        # 計算價格趨勢
        trend_start_idx = -(self.trend_period + 1)
        trend_change = (closes[-1] - closes[trend_start_idx]) / closes[trend_start_idx] * 100
        
        # 計算 ATR
        self._atr = self._calculate_atr(highs, lows, closes)
        
        # === 持倉檢查 ===
        if self._position is not None:
            pnl_pct = (current_close - self._entry_price) / self._entry_price * 100
            if self._position == "SHORT":
                pnl_pct = -pnl_pct
            
            # 止損檢查
            if pnl_pct <= -self.stop_loss_atr * self._atr / self._entry_price * 100:
                self._position = None
                return {
                    "direction": "FLAT",
                    "reason": f"止损触发 | PnL={pnl_pct:.2f}% | ATR={self._atr:.2f}",
                    "entry_price": current_close,
                    "metadata": {"pnl_pct": pnl_pct, "atr": self._atr},
                }
            
            # 止盈檢查
            if pnl_pct >= self.take_profit_atr * self._atr / self._entry_price * 100:
                self._position = None
                return {
                    "direction": "FLAT",
                    "reason": f"止盈触发 | PnL={pnl_pct:.2f}%",
                    "entry_price": current_close,
                    "metadata": {"pnl_pct": pnl_pct, "atr": self._atr},
                }
            
            # 持倉過長（50 根 bar）
            if len(klines) - self._entry_bar > 50:
                self._position = None
                return {
                    "direction": "FLAT",
                    "reason": f"持倉超時 | PnL={pnl_pct:.2f}%",
                    "entry_price": current_close,
                    "metadata": {"pnl_pct": pnl_pct, "holding_bars": len(klines) - self._entry_bar},
                }
            
            # 繼續持倉
            return None
        
        # === 進場信號 ===
        if self._position is None:
            stop_loss = None
            take_profit = None
            
            if vol_ratio >= self.volume_multiplier and trend_change > 0.5:
                # 多頭信號
                self._position = "LONG"
                self._entry_price = current_close
                self._entry_bar = len(klines)
                stop_loss = current_close - self.stop_loss_atr * self._atr
                take_profit = current_close + self.take_profit_atr * self._atr
                
                return {
                    "direction": "LONG",
                    "entry_price": current_close,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                    "reason": f"成交量突破 | Vol Ratio={vol_ratio:.2f} | Trend={trend_change:.2f}%",
                    "metadata": {
                        "vol_ratio": vol_ratio,
                        "vol_ma": vol_ma,
                        "current_vol": current_vol,
                        "trend_change": trend_change,
                        "atr": self._atr,
                    },
                }
            
            elif vol_ratio >= self.volume_multiplier and trend_change < -0.5:
                # 空頭信號
                self._position = "SHORT"
                self._entry_price = current_close
                self._entry_bar = len(klines)
                stop_loss = current_close + self.stop_loss_atr * self._atr
                take_profit = current_close - self.take_profit_atr * self._atr
                
                return {
                    "direction": "SHORT",
                    "entry_price": current_close,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                    "reason": f"成交量突破 | Vol Ratio={vol_ratio:.2f} | Trend={trend_change:.2f}%",
                    "metadata": {
                        "vol_ratio": vol_ratio,
                        "vol_ma": vol_ma,
                        "current_vol": current_vol,
                        "trend_change": trend_change,
                        "atr": self._atr,
                    },
                }
        
        return None
    
    def reset(self):
        """重置策略狀態（用於新回測）"""
        self._position = None
        self._entry_price = 0
        self._entry_bar = 0
        self._atr = 0
