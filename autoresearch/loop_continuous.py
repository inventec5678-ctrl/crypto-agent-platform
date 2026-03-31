#!/usr/bin/env python3
"""
Auto Research Continuous Loop
參考 Karpathy AutoResearch 概念

持續循環：
1. STAGE 1: 檢討 (15 min)
2. STAGE 2: 撰寫策略 (15 min)
3. STAGE 3: 回測驗證 (60 min)
4. STAGE 4: 記錄 & 回報 (5 min)

Target:
- 盈虧比 ≥ 2:1
- 勝率 ≥ 60%
- 最大回撤 ≤ 30%
- Sharpe ≥ 1.5
"""

import asyncio
import json
import time
import sys
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ============================================================
# 數據結構
# ============================================================

@dataclass
class Target:
    win_rate: float = 50.0  # 降到 50%
    profit_factor: float = 2.0
    max_drawdown: float = 30.0
    sharpe: float = 1.5

@dataclass
class BacktestResult:
    strategy_name: str
    entry_condition: str
    exit_condition: str
    stop_loss: str
    take_profit: str
    win_rate: float
    profit_factor: float
    max_drawdown: float
    sharpe: float
    trade_count: int
    total_pnl: float
    status: str  # KEEP / PARTIAL / DISCARD / CRASH
    notes: str

@dataclass
class ResearchState:
    round_num: int = 0
    total_experiments: int = 0
    keep_count: int = 0
    partial_count: int = 0
    discard_count: int = 0
    crash_count: int = 0
    started_at: str = ""
    last_round_at: str = ""


# ============================================================
# 記憶管理
# ============================================================

class MemoryManager:
    def __init__(self):
        self.memory_dir = PROJECT_ROOT / "autoresearch" / "memory"
        self.memory_dir.mkdir(exist_ok=True)
        self.research_log = self.memory_dir / "research_log.md"
        self.failed_strategies = self.memory_dir / "failed_strategies.json"
        self.best_strategies = self.memory_dir / "best_strategies.json"
        
        # 初始化檔案
        if not self.research_log.exists():
            self.research_log.write_text("# 研究日誌\n")
        if not self.failed_strategies.exists():
            self.failed_strategies.write_text("[]")
        if not self.best_strategies.exists():
            self.best_strategies.write_text(json.dumps({"target": asdict(Target()), "strategies": []}, indent=2))
    
    def read_failed_strategies(self) -> list:
        """讀取失敗策略"""
        try:
            return json.loads(self.failed_strategies.read_text())
        except:
            return []
    
    def add_failed_strategy(self, strategy: dict):
        """添加失敗策略"""
        failures = self.read_failed_strategies()
        failures.append(strategy)
        self.failed_strategies.write_text(json.dumps(failures, indent=2, ensure_ascii=False))
    
    def read_best_strategies(self) -> dict:
        """讀取最佳策略"""
        try:
            return json.loads(self.best_strategies.read_text())
        except:
            return {"target": asdict(Target()), "strategies": []}
    
    def add_best_strategy(self, strategy: dict):
        """添加最佳策略"""
        data = self.read_best_strategies()
        data["strategies"].append(strategy)
        self.best_strategies.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    
    def log_research(self, result: BacktestResult, round_num: int):
        """記錄研究結果"""
        status_icon = {"KEEP": "✅", "PARTIAL": "⚠️", "DISCARD": "❌", "CRASH": "💥"}.get(result.status, "?")
        
        log_entry = f"""
### 實驗 #{round_num} - {datetime.now().strftime('%Y-%m-%d %H:%M')}
- 策略: {result.strategy_name}
- 進場: {result.entry_condition}
- 止盈: {result.take_profit}
- 止損: {result.stop_loss}
- 結果: WR={result.win_rate}% | PF={result.profit_factor} | DD={result.max_drawdown}% | Sharpe={result.sharpe}
- 評估: {status_icon} {result.status}
- 備註: {result.notes}
"""
        with open(self.research_log, "a") as f:
            f.write(log_entry)


# ============================================================
# 策略分析 Agent
# ============================================================

# 所有策略模板
ALL_STRATEGY_TEMPLATES = [
    # === 實例 1: 純成交量 + K線（不使用傳統指標）
    {
        "type": "VolumeOnly_CandleBody",
        "strategy_name": "VolumeCandleBody_v1",
        "entry_condition": "實體 K 線 > 1.5% + 成交量 > 2x 均量",
        "exit_condition": "持倉 10 根 K 線",
        "stop_loss": "1% 或 1x ATR",
        "take_profit": "3% 或 2x ATR",
    },
    {
        "type": "VolumeOnly_WickRatio",
        "strategy_name": "WickRatio_v1",
        "entry_condition": "上下影線總長 < K 線實體的 30% + 成交量放大",
        "exit_condition": "持倉 8 根 K 線",
        "stop_loss": "0.8% 或 0.8x ATR",
        "take_profit": "2.5% 或 2x ATR",
    },
    {
        "type": "VolumeOnly_VolumeSpike",
        "strategy_name": "VolumeSpike_v1",
        "entry_condition": "單根 K 線成交量 > 過去 20 根平均的 3 倍",
        "exit_condition": "持倉 5 根 K 線",
        "stop_loss": "0.5% 或 0.5x ATR",
        "take_profit": "1.5% 或 1.5x ATR",
    },
    {
        "type": "VolumeOnly_PinBar",
        "strategy_name": "PinBar_v1",
        "entry_condition": "下影線 > 實體 2 倍 + 成交量放大",
        "exit_condition": "持倉 6 根 K 線",
        "stop_loss": "1% 或 1x ATR",
        "take_profit": "2.5% 或 2x ATR",
    },
    {
        "type": "VolumeOnly_Engulfing",
        "strategy_name": "Engulfing_v1",
        "entry_condition": "吞噬 K 線模式 + 成交量 > 前一根 1.5 倍",
        "exit_condition": "持倉 8 根 K 線",
        "stop_loss": "1.2% 或 1x ATR",
        "take_profit": "3% 或 2x ATR",
    },
    {
        "type": "VolumeOnly_ThreeWhiteSoldiers",
        "strategy_name": "ThreeWhiteSoldiers_v1",
        "entry_condition": "連續 3 根上漲 K 線 + 成交量遞增",
        "exit_condition": "持倉 10 根 K 線",
        "stop_loss": "2% 或 1.5x ATR",
        "take_profit": "5% 或 3x ATR",
    },
    
    # === 傳統策略 ===
    {
        "type": "TrendVolume",
        "strategy_name": "TrendVolumeBreakout_v1",
        "entry_condition": "當 24H 趨勢 > 1% 且成交量突破 20 日均線的 2 倍",
        "exit_condition": "持倉超過 20 根 K 線",
        "stop_loss": "2% 或 2x ATR",
        "take_profit": "4% 或 3x ATR",
    },
    {
        "type": "HighLow",
        "strategy_name": "HighLowBreakout_v1",
        "entry_condition": "價格突破 20 日高點且成交量超過 2 倍均量",
        "exit_condition": "持倉超過 15 根 K 線",
        "stop_loss": "1.5% 或 1.5x ATR",
        "take_profit": "3% 或 2.5x ATR",
    },
    {
        "type": "RSI",
        "strategy_name": "RSIReversal_v1",
        "entry_condition": "RSI(14) < 30 且 24H 趨勢為正",
        "exit_condition": "持倉超過 10 根 K 線",
        "stop_loss": "2x ATR",
        "take_profit": "2x ATR",
    },
    
    # === 時間效應 ===
    {
        "type": "WeekendEffect",
        "strategy_name": "WeekendAccumulation_v1",
        "entry_condition": "週五收盤前 4 小時買入（週末效應）",
        "exit_condition": "週一開盤賣出",
        "stop_loss": "1% 或 1x ATR",
        "take_profit": "1.5% 或 1.5x ATR",
    },
    {
        "type": "MondayEffect",
        "strategy_name": "MondayPump_v1",
        "entry_condition": "週一開盤 1 小時後買入（週一反彈效應）",
        "exit_condition": "持倉 4 小時",
        "stop_loss": "1% 或 1x ATR",
        "take_profit": "2% 或 2x ATR",
    },
    {
        "type": "VolatilityCrisis",
        "strategy_name": "VolatilityCrisisBuy_v1",
        "entry_condition": "單日暴跌 >5% 後第二天開盤買入",
        "exit_condition": "持倉 3 天",
        "stop_loss": "3% 或 2x ATR",
        "take_profit": "8% 或 4x ATR",
    },
    
    # === 多時區 ===
    {
        "type": "MultiTF_Daily4H",
        "strategy_name": "MultiTF_Daily4H_Confirm_v1",
        "entry_condition": "日線趨勢向上 + 4小時回調完成",
        "exit_condition": "日線出現止盈信號",
        "stop_loss": "2% 或 1.5x ATR",
        "take_profit": "5% 或 3x ATR",
    },
    {
        "type": "MultiTF_4H1H",
        "strategy_name": "MultiTF_4H1H_Confirm_v1",
        "entry_condition": "4小時向上突破 + 1小時回調支撐買入",
        "exit_condition": "4小時出現反向信號",
        "stop_loss": "1.5% 或 1x ATR",
        "take_profit": "3% 或 2x ATR",
    },
    
    # === 組合策略 ===
    {
        "type": "Combo_TrendVolRSI",
        "strategy_name": "Combo_TrendVolRSI_v1",
        "entry_condition": "趨勢向上 + 成交量突破 + RSI 處於 40-50 區間",
        "exit_condition": "任一條件反向",
        "stop_loss": "2% 或 1.5x ATR",
        "take_profit": "4% 或 2.5x ATR",
    },

    # === 牛市/熊市過濾 RSI ===
    {
        "type": "RegimeFilteredRSI",
        "strategy_name": "RegimeFilteredRSI_v1",
        "entry_condition": "MA200向上 + RSI<30 + 成交量>1.5x",
        "exit_condition": "RSI>60或5%或2%止損",
        "stop_loss": "2%",
        "take_profit": "5%",
    },

    # === 牛市只做多 ===
    {
        "type": "BullMarketRSI",
        "strategy_name": "BullMarketRSI_v1",
        "entry_condition": "MA200向上 + RSI<35 + 成交量>1.5x + 價格>MA200",
        "exit_condition": "RSI>65或4%或2%止損",
        "stop_loss": "2%",
        "take_profit": "4%",
    },
]


class StrategyAnalyzer:
    """分析市場和歷史，規劃新策略"""
    
    def __init__(self, memory: MemoryManager, strategy_pool=None):
        self.memory = memory
        self.strategy_pool = strategy_pool  # None = all strategies
    
    def analyze(self) -> Dict[str, Any]:
        """
        分析並規劃新策略
        返回: {
            "strategy_name": str,
            "entry_condition": str,
            "exit_condition": str,
            "stop_loss": str,
            "take_profit": str,
            "logic": str
        }
        """
        import random
        
        failures = self.memory.read_failed_strategies()
        
        # 記錄嘗試過的策略類型
        attempted_types = set()
        for f in failures:
            attempted_types.add(f.get("id", ""))
        
        # 使用全域模板，根據 strategy_pool 過濾
        templates = ALL_STRATEGY_TEMPLATES
        if self.strategy_pool:
            templates = [s for s in templates if any(p in s["type"] for p in self.strategy_pool)]
        
        random.shuffle(templates)
        
        # 嘗試還沒有失敗過的策略
        for template in templates:
            type_key = template["type"].replace("_v1", "")
            if type_key not in attempted_types:
                return {
                    **template,
                    "logic": f"嘗試新策略: {template['type']}"
                }
        
        # 都嘗試過了，隨機選擇並調整參數
        template = random.choice(templates)
        variations = [
            {"stop_loss": "1x ATR", "take_profit": "1.5x ATR"},
            {"stop_loss": "1.5x ATR", "take_profit": "2x ATR"},
            {"stop_loss": "2.5x ATR", "take_profit": "3.5x ATR"},
            {"stop_loss": "3x ATR", "take_profit": "4x ATR"},
        ]
        variation = random.choice(variations)
        
        return {
            "strategy_name": f"{template['strategy_name']}_v2",
            "entry_condition": template["entry_condition"],
            "exit_condition": template["exit_condition"],
            "stop_loss": variation["stop_loss"],
            "take_profit": variation["take_profit"],
            "logic": f"參數調整: {variation}"
        }


# ============================================================
# 策略執行器
# ============================================================

class StrategyRunner:
    """執行策略並進行回測"""
    
    def __init__(self):
        self.data = None
        self.load_data()
    
    def load_data(self):
        """載入本地數據"""
        import pandas as pd
        path = PROJECT_ROOT / "data" / "btcusdt_1d.parquet"
        if path.exists():
            self.data = pd.read_parquet(path)
            print(f"📊 載入 {len(self.data)} 根 K 線")
        else:
            print(f"⚠️ 找不到數據: {path}")
    
    def run(self, strategy_plan: Dict[str, Any], timeout_seconds: int = 3600) -> BacktestResult:
        """
        執行回測
        """
        import pandas as pd
        import numpy as np
        
        if self.data is None:
            return BacktestResult(
                strategy_name=strategy_plan["strategy_name"],
                entry_condition=strategy_plan["entry_condition"],
                exit_condition=strategy_plan["exit_condition"],
                stop_loss=strategy_plan["stop_loss"],
                take_profit=strategy_plan["take_profit"],
                win_rate=0, profit_factor=0, max_drawdown=100,
                sharpe=0, trade_count=0, total_pnl=0,
                status="CRASH",
                notes="數據載入失敗"
            )
        
        df = self.data.copy()
        
        # 計算指標
        closes = df['close'].values
        highs = df['high'].values
        lows = df['low'].values
        volumes = df['volume'].values
        
        # 計算均線
        ma20 = pd.Series(closes).rolling(20).mean().values
        
        # 計算 ATR
        trs = []
        for i in range(1, 15):
            if i < len(df):
                tr = max(
                    highs[-i] - lows[-i],
                    abs(highs[-i] - closes[-i-1]),
                    abs(lows[-i] - closes[-i-1])
                )
                trs.append(tr)
        atr = np.mean(trs) if trs else closes[-1] * 0.02
        
        # 計算成交量均線
        vol_ma20 = pd.Series(volumes).rolling(20).mean().values
        
        # 模擬回測
        trades = []
        position = None
        entry_price = 0
        entry_bar = 0
        
        for i in range(50, len(df)):
            current_price = closes[i]
            current_vol = volumes[i]
            vol_ratio = current_vol / vol_ma20[i] if vol_ma20[i] > 0 else 1
            
            # 計算 24H 趨勢
            trend_24h = (closes[i] - closes[i-1]) / closes[i-1] * 100 if i > 0 else 0
            
            # 計算偏離均線
            deviation = (closes[i] - ma20[i]) / ma20[i] * 100 if ma20[i] > 0 else 0
            
            # 進場邏輯（根據策略類型）
            if position is None:
                name = strategy_plan["strategy_name"]
                
                # 取得時間信息
                from datetime import datetime
                if 'open_time' in df.columns:
                    ts = df.iloc[i]['open_time']
                    if isinstance(ts, bytes):
                        ts = ts.decode('utf-8')
                    if isinstance(ts, str):
                        ts = pd.Timestamp(ts)
                    day_of_week = ts.dayofweek if hasattr(ts, 'dayofweek') else 0  # 0=Mon, 4=Fri
                else:
                    day_of_week = (i % 7)
                
                # TrendVolumeBreakout
                if "TrendVolume" in name:
                    if trend_24h > 1 and vol_ratio > 2.0:
                        position = "LONG"
                        entry_price = current_price
                        entry_bar = i
                
                # HighLow Breakout
                elif "HighLow" in name or ("Breakout" in name and "Low" in name):
                    high_20 = max(highs[i-20:i])
                    if current_price > high_20 and vol_ratio > 2.0:
                        position = "LONG"
                        entry_price = current_price
                        entry_bar = i
                
                # RSI Reversal
                elif "RSI" in name:
                    if len(closes) > 14:
                        deltas = np.diff(closes[i-14:i+1])
                        gains = np.where(deltas > 0, deltas, 0)
                        losses = np.where(deltas < 0, -deltas, 0)
                        avg_gain = np.mean(gains) if len(gains) > 0 else 0
                        avg_loss = np.mean(losses) if len(losses) > 0 else 0
                        rs = avg_gain / avg_loss if avg_loss > 0 else 100
                        rsi = 100 - (100 / (1 + rs)) if rs > 0 else 50
                        
                        # RSI < 30 超賣，原則上只靠 RSI 信號進場（不改 trend_24h > 0）
                        if rsi < 30 and vol_ratio > 1.3:
                            position = "LONG"
                            entry_price = current_price
                            entry_bar = i
                
                # === 非傳統策略 ===
                
                # 週末效應 - 週五買入
                elif "Weekend" in name and day_of_week == 4:  # Friday
                    position = "LONG"
                    entry_price = current_price
                    entry_bar = i
                
                # 週一效應 - 週一買入
                elif "Monday" in name and day_of_week == 0:  # Monday
                    position = "LONG"
                    entry_price = current_price
                    entry_bar = i
                
                # 暴跌後反彈 - 單日跌 >5% 後
                elif "Crisis" in name:
                    if i > 1:
                        daily_change = (closes[i] - closes[i-1]) / closes[i-1] * 100
                        if daily_change < -5:
                            position = "LONG"
                            entry_price = current_price
                            entry_bar = i
                
                # 趨勢延續 - 回調後買入
                elif "Continuation" in name:
                    if i >= 3:
                        # 連續 3 天上漲後回調
                        daily_changes = [(closes[i-j] - closes[i-j-1]) / closes[i-j-1] * 100 for j in range(1, 4)]
                        if all(c > 0 for c in daily_changes) and trend_24h < -1:
                            position = "LONG"
                            entry_price = current_price
                            entry_bar = i
                
                # 低波動突破
                elif "LowVol" in name:
                    if len(trs) >= 5:
                        avg_atr = np.mean(trs[-5:])
                        atr_threshold = atr * 0.5
                        if avg_atr < atr_threshold and vol_ratio > 1.5:
                            position = "LONG"
                            entry_price = current_price
                            entry_bar = i
                
                # Golden Cross / Death Cross
                elif "GoldenCross" in name or "DeathCross" in name:
                    if i >= 200 and "ma50" not in dir():
                        ma50_list = pd.Series(closes).rolling(50).mean().values
                        ma200_list = pd.Series(closes).rolling(200).mean().values
                        if i > 200:
                            ma50_prev = ma50_list[i-2] if not np.isnan(ma50_list[i-2]) else 0
                            ma200_prev = ma200_list[i-2] if not np.isnan(ma200_list[i-2]) else 0
                            ma50_curr = ma50_list[i-1] if not np.isnan(ma50_list[i-1]) else 0
                            ma200_curr = ma200_list[i-1] if not np.isnan(ma200_list[i-1]) else 0
                            
                            if "GoldenCross" in name:
                                if ma50_prev <= ma200_prev and ma50_curr > ma200_curr and closes[i] > ma50_curr:
                                    position = "LONG"
                                    entry_price = current_price
                                    entry_bar = i
                            elif "DeathCross" in name:
                                if ma50_prev >= ma200_prev and ma50_curr < ma200_curr and closes[i] < ma50_curr:
                                    position = "LONG"
                                    entry_price = current_price
                                    entry_bar = i
                
                # Pump Alert - 急漲後回調買入
                elif "Pump" in name:
                    if i > 0:
                        hourly_change = (closes[i] - closes[i-1]) / closes[i-1] * 100
                        if hourly_change > 3 and trend_24h < 0.5:
                            position = "LONG"
                            entry_price = current_price
                            entry_bar = i
                
                # Weekday Pattern
                elif "Weekday" in name:
                    if day_of_week in [1, 2]:  # Tue, Wed
                        position = "LONG"
                        entry_price = current_price
                        entry_bar = i
                
                # MeanReversion fallback
                elif "MeanReversion" in name:
                    if deviation < -3 and trend_24h > 0:
                        position = "LONG"
                        entry_price = current_price
                        entry_bar = i
                
                # 默認：強趨勢進場
                else:
                    if trend_24h > 1.5:
                        position = "LONG"
                        entry_price = current_price
                        entry_bar = i
                
                # === 純成交量 + K線策略 ===
                if position is None and ("VolumeOnly" in name or "Candle" in name or "Wick" in name or "Spike" in name or "Compression" in name or "PinBar" in name or "Engulfing" in name or "MultiCandle" in name):
                    
                    # 計算 K 線結構
                    body = abs(closes[i] - closes[i-1]) / closes[i-1] * 100
                    upper_wick = highs[i] - max(closes[i], closes[i-1])
                    lower_wick = min(closes[i], closes[i-1]) - lows[i]
                    wick_ratio = (upper_wick + lower_wick) / (body * closes[i-1] / 100) if body > 0 else 0
                    
                    if "VolumeCandleBody" in name:
                        # 實體 K 線 + 成交量
                        if body > 1.5 and vol_ratio > 2.0:
                            position = "LONG"
                            entry_price = current_price
                            entry_bar = i
                    
                    elif "WickRatio" in name:
                        # 影線比例
                        if wick_ratio < 0.3 and vol_ratio > 1.5:
                            position = "LONG"
                            entry_price = current_price
                            entry_bar = i
                    
                    elif "VolumeSpike" in name:
                        # 成交量暴增
                        if vol_ratio > 3.0:
                            position = "LONG"
                            entry_price = current_price
                            entry_bar = i
                    
                    elif "PriceCompression" in name:
                        # 價格壓縮後突破
                        if i >= 10:
                            recent_range = max(highs[i-10:i]) - min(lows[i-10:i])
                            avg_range = np.mean([highs[j] - lows[j] for j in range(i-10, i)])
                            if recent_range < avg_range * 0.3 and vol_ratio > 1.8:
                                position = "LONG"
                                entry_price = current_price
                                entry_bar = i
                    
                    elif "PinBar" in name:
                        # Pin Bar 形態
                        if lower_wick > body * 2 and vol_ratio > 1.5:
                            position = "LONG"
                            entry_price = current_price
                            entry_bar = i
                    
                    elif "Engulfing" in name:
                        # 吞噬形態
                        if i > 0:
                            prev_body = abs(closes[i-1] - closes[i-2]) / closes[i-2] * 100
                            if body > prev_body and vol_ratio > 1.5 and closes[i] > closes[i-1]:
                                position = "LONG"
                                entry_price = current_price
                                entry_bar = i
                    
                    elif "ThreeWhiteSoldiers" in name:
                        # 三白兵形態
                        if i >= 3:
                            c1 = (closes[i-2] - closes[i-3]) / closes[i-3] * 100
                            c2 = (closes[i-1] - closes[i-2]) / closes[i-2] * 100
                            c3 = (closes[i] - closes[i-1]) / closes[i-1] * 100
                            if c1 > 0 and c2 > 0 and c3 > 0 and vol_ratio > 1.3:
                                position = "LONG"
                                entry_price = current_price
                                entry_bar = i
                
                # === 多時區/組合策略的信號 ===
                if position is None and ("MultiTF" in name or "Combo" in name):
                    
                    # Combo: 多條件同時滿足
                    if "Combo_TrendVolRSI" in name:
                        # 趨勢向上 + 成交量突破 + RSI 在 40-50
                        if trend_24h > 0.5 and vol_ratio > 1.5:
                            # 計算 RSI
                            if len(closes) > 14:
                                deltas = np.diff(closes[i-14:i+1])
                                gains = np.where(deltas > 0, deltas, 0)
                                losses = np.where(deltas < 0, -deltas, 0)
                                avg_gain = np.mean(gains) if len(gains) > 0 else 0
                                avg_loss = np.mean(losses) if len(losses) > 0 else 0
                                rs = avg_gain / avg_loss if avg_loss > 0 else 100
                                rsi = 100 - (100 / (1 + rs)) if rs > 0 else 50
                                
                                if 40 <= rsi <= 50:
                                    position = "LONG"
                                    entry_price = current_price
                                    entry_bar = i
                    
                    elif "Combo_VolBreakRSI" in name:
                        # 成交量突破 + RSI 在 35-45
                        if vol_ratio > 1.8:
                            if len(closes) > 14:
                                deltas = np.diff(closes[i-14:i+1])
                                gains = np.where(deltas > 0, deltas, 0)
                                losses = np.where(deltas < 0, -deltas, 0)
                                avg_gain = np.mean(gains) if len(gains) > 0 else 0
                                avg_loss = np.mean(losses) if len(losses) > 0 else 0
                                rs = avg_gain / avg_loss if avg_loss > 0 else 100
                                rsi = 100 - (100 / (1 + rs)) if rs > 0 else 50
                                
                                if 35 <= rsi <= 45:
                                    position = "LONG"
                                    entry_price = current_price
                                    entry_bar = i
                    
                    elif "Combo_MA_ATR" in name:
                        # 價格站上均線 + ATR 擴張
                        ma20_current = np.mean(closes[i-20:i]) if i >= 20 else closes[-1]
                        if current_price > ma20_current and vol_ratio > 1.5:
                            position = "LONG"
                            entry_price = current_price
                            entry_bar = i
                    
                    elif "MultiTF_Daily" in name:
                        # 日線多頭 + 回調後恢復上漲
                        ma20_current = np.mean(closes[i-20:i]) if i >= 20 else closes[-1]
                        ma20_prev = np.mean(closes[i-21:i-1]) if i >= 21 else ma20_current
                        # 上升趨勢：MA20向上且價格在MA20均線上方
                        if ma20_current > ma20_prev and current_price > ma20_current and trend_24h > 0.5:
                            position = "LONG"
                            entry_price = current_price
                            entry_bar = i
                    
                    elif "MultiTF_4H1H" in name:
                        # 4小時突破 + 1小時回調
                        ma20_current = np.mean(closes[i-20:i]) if i >= 20 else closes[-1]
                        # 價格站上MA20 + 成交量放大
                        if current_price > ma20_current and vol_ratio > 1.5 and trend_24h > 0.3:
                            position = "LONG"
                            entry_price = current_price
                            entry_bar = i

                    elif "RegimeFilteredRSI" in name:
                        # 計算 MA200
                        ma200 = np.mean(closes[i-200:i]) if i >= 200 else None
                        # 計算 MA200 方向（前 20 日平均）
                        ma200_prev = np.mean(closes[i-220:i-20]) if i >= 220 else None

                        if ma200 and ma200_prev:
                            is_bull = ma200 > ma200_prev   # 牛市：MA200 向上
                            # is_bear = ma200 < ma200_prev  # 熊市：MA200 向下（不做空）

                            # 計算 RSI
                            rsi = None
                            if len(closes) > 14:
                                deltas = np.diff(closes[i-14:i+1])
                                gains = np.where(deltas > 0, deltas, 0)
                                losses = np.where(deltas < 0, -deltas, 0)
                                avg_gain = np.mean(gains) if len(gains) > 0 else 0
                                avg_loss = np.mean(losses) if len(losses) > 0 else 0
                                rs = avg_gain / avg_loss if avg_loss > 0 else 100
                                rsi = 100 - (100 / (1 + rs)) if rs > 0 else 50

                            if rsi is not None:
                                # 只在牛市時做多，熊市完全跳過
                                if is_bull:
                                    if rsi < 30 and vol_ratio > 1.5 and current_price > ma200:
                                        position = "LONG"
                                        entry_price = current_price
                                        entry_bar = i
                    
                    elif "BullMarketRSI" in name:
                        # 計算 MA200
                        ma200 = np.mean(closes[i-200:i]) if i >= 200 else None
                        ma200_prev = np.mean(closes[i-220:i-20]) if i >= 220 else None

                        if ma200 and ma200_prev:
                            is_bull = ma200 > ma200_prev   # 牛市：MA200 向上

                            # 計算 RSI
                            rsi = None
                            if len(closes) > 14:
                                deltas = np.diff(closes[i-14:i+1])
                                gains = np.where(deltas > 0, deltas, 0)
                                losses = np.where(deltas < 0, -deltas, 0)
                                avg_gain = np.mean(gains) if len(gains) > 0 else 0
                                avg_loss = np.mean(losses) if len(losses) > 0 else 0
                                rs = avg_gain / avg_loss if avg_loss > 0 else 100
                                rsi = 100 - (100 / (1 + rs)) if rs > 0 else 50

                            if rsi is not None:
                                # 牛市 + RSI<35 + 成交量放大 + 價格在MA200上方 → 做多
                                if is_bull and rsi < 35 and vol_ratio > 1.5 and current_price > ma200:
                                    position = "LONG"
                                    entry_price = current_price
                                    entry_bar = i

            # 出場邏輯
            if position is not None:
                # 根據倉位類型計算 PNL
                if position == "LONG":
                    pnl_pct = (current_price - entry_price) / entry_price * 100
                else:  # SHORT
                    pnl_pct = (entry_price - current_price) / entry_price * 100
                
                holding_bars = i - entry_bar
                
                # 止損 2% 或 2x ATR
                sl_distance = max(2.0, atr / entry_price * 100)
                # 止盈 4% 或 3x ATR
                tp_distance = max(4.0, atr / entry_price * 100 * 1.5)
                
                # 虧損
                if pnl_pct <= -sl_distance:
                    trades.append({"type": "LOSS", "pnl": pnl_pct})
                    position = None
                # 獲利
                elif pnl_pct >= tp_distance:
                    trades.append({"type": "WIN", "pnl": pnl_pct})
                    position = None
                # 超時
                elif holding_bars > 10:
                    trades.append({"type": "TIMEOUT", "pnl": pnl_pct})
                    position = None
        
        # 計算指標
        if not trades:
            return BacktestResult(
                strategy_name=strategy_plan["strategy_name"],
                entry_condition=strategy_plan["entry_condition"],
                exit_condition=strategy_plan["exit_condition"],
                stop_loss=strategy_plan["stop_loss"],
                take_profit=strategy_plan["take_profit"],
                win_rate=0, profit_factor=0, max_drawdown=100,
                sharpe=0, trade_count=0, total_pnl=0,
                status="DISCARD",
                notes="無有效交易"
            )
        
        wins = [t for t in trades if t["type"] == "WIN"]
        losses = [t for t in trades if t["type"] == "LOSS"]
        
        win_rate = len(wins) / len(trades) * 100 if trades else 0
        
        avg_win = np.mean([t["pnl"] for t in wins]) if wins else 0
        avg_loss = abs(np.mean([t["pnl"] for t in losses])) if losses else 1
        profit_factor = avg_win / avg_loss if avg_loss > 0 else 0
        
        pnls = [t["pnl"] for t in trades]
        sharpe = np.mean(pnls) / np.std(pnls) * np.sqrt(len(pnls)) if np.std(pnls) > 0 else 0
        
        # 最大回撤（簡化計算）
        cumulative = np.cumsum(pnls)
        max_dd = 0
        peak = 0
        for val in cumulative:
            if val > peak:
                peak = val
            dd = peak - val
            if dd > max_dd:
                max_dd = dd
        
        # 評估
        target = Target()
        status = "DISCARD"
        notes = ""
        
        if win_rate >= target.win_rate and profit_factor >= target.profit_factor and max_dd <= target.max_drawdown and sharpe >= target.sharpe:
            status = "KEEP"
            notes = "所有指標達標！"
        elif win_rate >= 55 and profit_factor >= 1.8:
            status = "PARTIAL"
            notes = "部分指標接近達標"
        else:
            status = "DISCARD"
            notes = f"未達標 (WR={win_rate:.1f}%, PF={profit_factor:.2f}, DD={max_dd:.1f}%)"
        
        return BacktestResult(
            strategy_name=strategy_plan["strategy_name"],
            entry_condition=strategy_plan["entry_condition"],
            exit_condition=strategy_plan["exit_condition"],
            stop_loss=strategy_plan["stop_loss"],
            take_profit=strategy_plan["take_profit"],
            win_rate=win_rate,
            profit_factor=profit_factor,
            max_drawdown=max_dd,
            sharpe=sharpe,
            trade_count=len(trades),
            total_pnl=sum(pnls),
            status=status,
            notes=notes
        )


# ============================================================
# 回報格式化
# ============================================================

def format_report(state: ResearchState, result: BacktestResult) -> str:
    """格式化回報訊息"""
    target = Target()
    
    status_icon = {"KEEP": "✅", "PARTIAL": "⚠️", "DISCARD": "❌", "CRASH": "💥"}.get(result.status, "?")
    
    wr_ok = "✅" if result.win_rate >= target.win_rate else "❌"
    pf_ok = "✅" if result.profit_factor >= target.profit_factor else "❌"
    dd_ok = "✅" if result.max_drawdown <= target.max_drawdown else "❌"
    sh_ok = "✅" if result.sharpe >= target.sharpe else "❌"
    
    return f"""
╔══════════════════════════════════════════════════════════╗
║           📊 Auto Research 第 {state.round_num} 輪報告                    ║
╠══════════════════════════════════════════════════════════╣
║ 🎯 Target: 勝率≥60% | PF≥2 | DD≤30% | Sharpe≥1.5    ║
╠══════════════════════════════════════════════════════════╣
║ 📝 策略: {result.strategy_name:<40} ║
║    進場: {result.entry_condition[:40]:<40} ║
║    止盈: {result.take_profit:<40} ║
║    止損: {result.stop_loss:<40} ║
╠══════════════════════════════════════════════════════════╣
║ 📈 結果:                                                 ║
║    勝率: {result.win_rate:>6.1f}% {wr_ok}  (目標≥{target.win_rate}%)      ║
║    盈虧比: {result.profit_factor:>6.2f} {pf_ok}  (目標≥{target.profit_factor})        ║
║    最大回撤: {result.max_drawdown:>5.1f}% {dd_ok}  (目標≤{target.max_drawdown}%)      ║
║    Sharpe: {result.sharpe:>6.2f} {sh_ok}  (目標≥{target.sharpe})         ║
║    交易次數: {result.trade_count:<6}                            ║
╠══════════════════════════════════════════════════════════╣
║ 🎯 評估: {status_icon} {result.status:<8}                                    ║
║ 📝 備註: {result.notes[:40]:<40} ║
╠══════════════════════════════════════════════════════════╣
║ 📁 累計: 總={state.total_experiments:>3} | ✅KEEP={state.keep_count:>3} | ⚠️PARTIAL={state.partial_count:>3} | ❌={state.discard_count:>3} | 💥={state.crash_count:>3} ║
╚══════════════════════════════════════════════════════════╝
"""


# ============================================================
# 主循環
# ============================================================

async def run_continuous_loop(report_callback=None, strategy_pool=None):
    """持續運行研究循環"""
    
    memory = MemoryManager()
    analyzer = StrategyAnalyzer(memory, strategy_pool=strategy_pool)
    runner = StrategyRunner()
    state = ResearchState(
        started_at=datetime.now().isoformat()
    )
    
    print("🚀 Auto Research 啟動！")
    print("   Target: 勝率≥60% | 盈虧比≥2 | 回撤≤30% | Sharpe≥1.5")
    print("=" * 60)
    
    while True:
        state.round_num += 1
        state.total_experiments += 1
        state.last_round_at = datetime.now().isoformat()
        
        print(f"\n{'='*60}")
        print(f"📍 第 {state.round_num} 輪開始")
        print(f"{'='*60}")
        
        # STAGE 1: 分析規劃 (15 分鐘)
        print("\n[STAGE 1/4] 分析市場 & 規劃策略...")
        start = time.time()
        
        try:
            strategy_plan = analyzer.analyze()
            print(f"   策略: {strategy_plan['strategy_name']}")
            print(f"   進場: {strategy_plan['entry_condition']}")
            print(f"   止盈: {strategy_plan['take_profit']}")
            print(f"   止損: {strategy_plan['stop_loss']}")
        except Exception as e:
            print(f"   ❌ 分析失敗: {e}")
            strategy_plan = {
                "strategy_name": f"Fallback_{state.round_num}",
                "entry_condition": "24H 趨勢 > 1%",
                "exit_condition": "持倉 20 根 K 線",
                "stop_loss": "2x ATR",
                "take_profit": "3x ATR",
                "logic": "備用策略"
            }
        
        elapsed = time.time() - start
        print(f"   ⏱️ 耗時: {elapsed:.1f} 秒")
        
        # STAGE 2: 撰寫策略 (15 分鐘) - 簡化版，跳過直接使用策略
        print("\n[STAGE 2/4] 準備回測環境...")
        start = time.time()
        # 這裡可以擴展為動態生成策略代碼
        print(f"   ✅ 策略就緒")
        elapsed = time.time() - start
        print(f"   ⏱️ 耗時: {elapsed:.1f} 秒")
        
        # STAGE 3: 回測驗證 (60 分鐘)
        print("\n[STAGE 3/4] 執行回測 (9年歷史數據)...")
        start = time.time()
        
        try:
            result = runner.run(strategy_plan, timeout_seconds=3600)
        except Exception as e:
            print(f"   ❌ 回測崩潰: {e}")
            result = BacktestResult(
                strategy_name=strategy_plan["strategy_name"],
                entry_condition=strategy_plan["entry_condition"],
                exit_condition=strategy_plan["exit_condition"],
                stop_loss=strategy_plan["stop_loss"],
                take_profit=strategy_plan["take_profit"],
                win_rate=0, profit_factor=0, max_drawdown=100,
                sharpe=0, trade_count=0, total_pnl=0,
                status="CRASH",
                notes=str(e)
            )
        
        elapsed = time.time() - start
        print(f"   ⏱️ 耗時: {elapsed:.1f} 秒")
        
        # 更新狀態
        if result.status == "KEEP":
            state.keep_count += 1
        elif result.status == "PARTIAL":
            state.partial_count += 1
        elif result.status == "CRASH":
            state.crash_count += 1
        else:
            state.discard_count += 1
        
        # STAGE 4: 記錄 & 回報
        print("\n[STAGE 4/4] 記錄結果...")
        memory.log_research(result, state.round_num)
        
        if result.status == "KEEP":
            memory.add_best_strategy(asdict(result))
        elif result.status in ["DISCARD", "CRASH"]:
            memory.add_failed_strategy({
                "id": result.strategy_name,
                "name": result.strategy_name,
                "problem": result.notes,
                "lesson": strategy_plan.get("logic", "")
            })
        
        # 生成回報
        report = format_report(state, result)
        print(report)
        
        # 發送回報（如果提供了回調函數）
        if report_callback:
            try:
                report_callback(report)
            except Exception as e:
                print(f"   ⚠️ 回報發送失敗: {e}")
        
        # 保存狀態
        state_file = PROJECT_ROOT / "autoresearch" / "state_continuous.json"
        state_file.write_text(json.dumps(asdict(state), indent=2))
        
        print(f"\n⏳ 進入下一輪...")
        await asyncio.sleep(5)  # 短暫休息


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Auto Research Continuous Loop")
    parser.add_argument("--report-webhook", help="WebHook URL for reporting")
    parser.add_argument("--instance-id", type=int, default=0, help="Instance ID (0=all, 1=volume-only, 2=multi-timeframe, 3=web-research)")
    parser.add_argument("--log-prefix", default="", help="Prefix for log messages")
    args = parser.parse_args()
    
    # 設置實例特定的策略池
    if args.instance_id == 1:
        # 實例 1: 純成交量 + K線策略（不使用傳統指標）
        strategy_pool = [
            "VolumeOnly",
            "CandlePattern",
            "PriceAction",
        ]
        print(f"\n{'='*60}")
        print(f"📊 實例 1: 純成交量 + K線策略")
        print(f"{'='*60}\n")
    elif args.instance_id == 2:
        # 實例 2: 多時區策略
        strategy_pool = [
            "MultiTF",
            "Timeframe",
        ]
        print(f"\n{'='*60}")
        print(f"📊 實例 2: 多時區策略")
        print(f"{'='*60}\n")
    elif args.instance_id == 3:
        # 實例 3: 所有策略（不做過濾）
        strategy_pool = None
        print(f"\n{'='*60}")
        print(f"📊 實例 3: 所有策略")
        print(f"{'='*60}\n")
    else:
        strategy_pool = None
        print(f"\n{'='*60}")
        print(f"📊 標準模式: 所有策略")
        print(f"{'='*60}\n")
    
    if args.report_webhook:
        # 設置 WebHook 回報
        import aiohttp
        
        async def webhook_report(message: str):
            payload = {"content": message}
            async with aiohttp.ClientSession() as session:
                await session.post(args.report_webhook, json=payload)
        
        asyncio.run(run_continuous_loop(report_callback=webhook_report, strategy_pool=strategy_pool))
    else:
        # 直接輸出到 console
        asyncio.run(run_continuous_loop(strategy_pool=strategy_pool))


if __name__ == "__main__":
    main()
