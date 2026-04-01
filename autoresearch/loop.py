"""
ResearchOrchestrator — Agent 驅動 Auto Research 主循環

三團隊協作：
  ① team-researcher : 分析市場快照 + 避開失敗組合 → 生成策略 → 回測
  ② team-qa         : 獨立實作進場邏輯 → 獨立回測 → 對比結果
  ③ team-ceo        : 審核 4/4 指標 → 裁決 KEEP / DISCARD

用法:
  >>> orchestrator = ResearchOrchestrator(symbol="BTCUSDT")
  >>> orchestrator.start(max_rounds=100)
"""

import asyncio
import json
import os
import random
import re
import subprocess
import tempfile
import time as time_module
import uuid
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Callable

import pandas as pd
import numpy as np

# ========== 路徑設定 ==========
PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "btcusdt_1d.parquet"
MEMORY_DIR = Path(__file__).parent / "memory"
MEMORY_DIR.mkdir(parents=True, exist_ok=True)
RESEARCH_LOG = MEMORY_DIR / "research_log.md"
FAILED_STRATEGIES_FILE = MEMORY_DIR / "failed_strategies.json"
BEST_STRATEGIES_FILE = MEMORY_DIR / "best_strategies.json"
FACTOR_LIBRARY = Path(__file__).parent / "factor_library.json"
OPENCLAW_CLI = os.environ.get("OPENCLAW_CLI", "/opt/homebrew/bin/openclaw")

# ========== Target ==========
@dataclass
class Target:
    win_rate: float = 50.0
    profit_factor: float = 2.0
    max_drawdown: float = 30.0
    sharpe: float = 1.5

TARGET = Target()


# ========== 內聯 Import（避免循環依賴）==========
# 從 ai_researcher.py 復用 MarketData / FailureMemory / BacktestResult
import sys
sys.path.insert(0, str(Path(__file__).parent))
from ai_researcher import MarketData, FailureMemory, StrategyBacktester, BacktestResult


from dataclasses import dataclass, asdict
from research_state import ResearchState


# ========== 策略生成：team-researcher Agent 邏輯 ==========

class StrategyGenerator:
    """
    team-researcher Agent 的策略生成核心。
    
    根據市場快照 + 失敗摘要，生成新策略。
    目前是骨架實作（使用隨機組合 + 隨機參數），
    可在 Phase 1 穩定後替換成真正的 Agent prompt 驅動。
    """

    # 有效進場條件池（team-researcher 候選維度）
    CONDITION_POOL = [
        {"cond": "snap.rsi < 30",              "desc": "RSI 深度超賣 (<30)"},
        {"cond": "snap.rsi < 40",              "desc": "RSI 超賣 (<40)"},
        {"cond": "snap.rsi > 60",              "desc": "RSI 中性偏多 (>60)"},
        {"cond": "snap.rsi > 70",              "desc": "RSI 超買 (>70)"},
        {"cond": "snap.vol_ratio > 1.5",       "desc": "成交量放大 1.5x"},
        {"cond": "snap.vol_ratio > 2.0",       "desc": "成交量放大 2x"},
        {"cond": "snap.close > snap.ma200",    "desc": "價格 > MA200"},
        {"cond": "snap.close < snap.ma200",    "desc": "價格 < MA200"},
        {"cond": "snap.regime == 'BULL'",      "desc": "牛市 regime"},
        {"cond": "snap.regime == 'BEAR'",      "desc": "熊市 regime"},
        {"cond": "snap.regime == 'RANGE'",     "desc": "盤整 regime"},
        {"cond": "snap.ma200_slope > 0.1",    "desc": "MA200 向上傾斜"},
        {"cond": "snap.ma200_slope < -0.1",   "desc": "MA200 向下傾斜"},
        {"cond": "snap.trend_7d > 5",          "desc": "7日動量 > 5%"},
        {"cond": "snap.trend_7d < -5",         "desc": "7日動量 < -5%"},
        {"cond": "snap.trend_7d > 0",          "desc": "7日動量正值"},
        {"cond": "snap.trend_7d < 0",          "desc": "7日動量負值"},
        {"cond": "snap.close > snap.ma10",     "desc": "價格 > MA10"},
        {"cond": "snap.close < snap.ma20",    "desc": "價格 < MA20"},
        {"cond": "snap.rsi < 35 and snap.trend_7d > 0", "desc": "RSI 回調 + 趨勢向上"},
        {"cond": "snap.rsi > 65 and snap.trend_7d < 0",  "desc": "RSI 背離 + 趨勢向下"},
        {"cond": "snap.rsi < 40 and snap.vol_ratio > 1.5", "desc": "RSI 超賣 + 成交量放大"},
    ]

    # Regime 對應的主要方向
    REGIME_DIRECTION = {
        "BULL":     ["long",  "long",  "long"],   # 傾向做多
        "BEAR":     ["short", "short", "short"],  # 傾向做空
        "RANGE":    ["long",  "short", "long"],   # 兩側都可
    }

    def __init__(self, failure_memory: FailureMemory, research_state=None, orchestrator=None):
        self.failures = failure_memory
        self.generation_count = 0
        self._research_state = research_state  # 持久化狀態（用於指紋追蹤）
        self._orchestrator = orchestrator  # ResearchOrchestrator 參考（用於派 sub-agent）
        # 檢查是否已處於 sub-agent 上下文（避免無窮迴圈）
        self._in_subagent = os.environ.get("OPENCLAW_SUBAGENT_CONTEXT") == "1"

    # ========== 核心：派 team-researcher sub-agent 出去 ========

    def generate(self, snapshot: dict, failed_summary: str) -> dict:
        """
        派 team-researcher sub-agent，讓它真正自主研究。

        流程：
        1. 建 prompt（Karpathy AutoResearch 模式）
        2. 用 ResearchOrchestrator._spawn_and_wait_researcher() 派 sub-agent
        3. Agent 自己想到策略、自己寫 Python 回測、自己跑、自己裁決
        4. 等待 output JSON，回傳結果

        絕不使用模板選擇器。如果 sub-agent 超時/失敗，返回 None。

        架構說明：
        - dispatch_researcher() 派第一層 sub-agent（這個方法）
        - 這個方法執行真正的研究工作（建prompt、派agent、等待結果）
        - 派出的 agent 自己做研究，不會再派下一層
        """
        if self._in_subagent:
            # Sub-agent 自己的 generate() 直接做事，不再生 sub-agent
            return self._generate_inline(snapshot, failed_summary)

        # 避免無窮迴圈：sub-agent 自己的 generate() call 不能再生 sub-agent
        # 透過環境變量告知處於 sub-agent 上下文
        import os
        os.environ["OPENCLAW_SUBAGENT_CONTEXT"] = "1"

        self.generation_count += 1
        output_path = str(MEMORY_DIR / f"researcher_{self.generation_count}_{uuid.uuid4().hex[:6]}.json")
        strategy_id = f"Agent_{self.generation_count}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        print(f"   🚀 [{strategy_id}] 派 team-researcher sub-agent 出去了...")

        prompt = self._build_researcher_prompt(snapshot, failed_summary)

        try:
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    self._orchestrator._spawn_and_wait_researcher,
                    prompt=prompt,
                    output_path=output_path,
                    round_num=self.generation_count,
                )
                result = future.result(timeout=360)
            print(f"   ✅ sub-agent 完成")
            return result
        except concurrent.futures.TimeoutError:
            print(f"   ⚠️ sub-agent 超時（360s），返回 None")
            return None
        except Exception as e:
            print(f"   ⚠️ sub-agent 錯誤: {e}，返回 None")
        finally:
            os.environ.pop("OPENCLAW_SUBAGENT_CONTEXT", None)

    def _generate_inline(self, snapshot: dict, failed_summary: str) -> dict:
        """
        Sub-agent 自己的 generate() 直接做事（不做為派 sub-agent）。
        避免無窮迴圈：sub-agent 內的 StrategyGenerator.generate() 走到這裡。

        流程：
        1. 根據 snapshot + failed_summary 生成策略（使用模板池）
        2. 執行回測
        3. 產出 dict（sub-agent 的調用者會寫入 output JSON）
        """
        import pandas as pd
        from .ai_researcher import MarketData, StrategyBacktester, FailureMemory

        df = pd.read_parquet(str(Path(DATA_PATH)))
        market = MarketData(df)
        failure_memory = FailureMemory()

        # 產生策略（使用模板池）
        strategy = self._generate_from_template_pool(snapshot, failed_summary)

        # 執行回測
        ec_str = strategy.get('entry_conditions', strategy.get('entry_description', ''))
        entry_fn = eval(f"lambda snap: {ec_str}")
        backtester = StrategyBacktester(market)
        result = backtester.backtest(
            strategy_id=strategy['strategy_id'],
            strategy_name=strategy['strategy_name'],
            entry_description=ec_str,
            entry_fn=entry_fn,
            stop_loss_pct=strategy.get('stop_loss_pct', 0.02),
            take_profit_pct=strategy.get('take_profit_pct', 0.05),
            max_holding_bars=strategy.get('max_holding_bars', 10),
        )

        return {
            "strategy_id": strategy['strategy_id'],
            "strategy_name": strategy['strategy_name'],
            "entry_description": strategy.get('entry_description', ''),
            "entry_conditions": ec_str,
            "stop_loss_pct": strategy.get('stop_loss_pct', 0.02),
            "take_profit_pct": strategy.get('take_profit_pct', 0.05),
            "max_holding_bars": strategy.get('max_holding_bars', 10),
            "direction": strategy.get('direction', 'LONG'),
            "regime": strategy.get('regime', snapshot.get('regime', 'N/A')),
            "win_rate": result.win_rate,
            "profit_factor": result.profit_factor,
            "max_drawdown": result.max_drawdown,
            "sharpe": result.sharpe,
            "total_trades": result.total_trades,
            "wins": result.wins,
            "losses": result.losses,
            "timeouts": result.timeouts,
        }

    def _generate_from_template_pool(self, snapshot: dict, failed_summary: str) -> dict:
        """
        使用模板池生成策略（Sub-agent 直接調用，不派新的 sub-agent）。
        """
        import random

        regime = snapshot.get('regime', 'N/A')
        rsi = snapshot.get('rsi_14', 50)
        trend = snapshot.get('trend_7d', 0)

        if regime in ('BULL', 'STRONG_BULL'):
            direction = 'LONG'
            bias_conditions = [f"snap.regime in ('BULL','STRONG_BULL')", f"snap.trend_7d > 0", f"snap.close > snap.ma200"]
        elif regime in ('BEAR', 'STRONG_BEAR'):
            direction = random.choice(['SHORT', 'LONG'])
            bias_conditions = [f"snap.regime in ('BEAR','STRONG_BEAR')", f"snap.rsi < 50"]
        else:
            direction = 'LONG'
            bias_conditions = [f"snap.regime == 'RANGE'", f"snap.rsi < 50 or snap.rsi > 50"]

        all_conditions = [
            f"snap.rsi < {random.randint(25, 40)}",
            f"snap.rsi > {random.randint(60, 70)}",
            f"snap.vol_ratio > {random.uniform(1.3, 2.0):.1f}",
            f"snap.trend_7d > {random.uniform(0, 5):.1f}",
            f"snap.trend_7d < {random.uniform(-5, 0):.1f}",
            f"snap.close > snap.ma{ random.choice([10, 20, 50]) }",
            f"snap.close < snap.ma{ random.choice([10, 20, 50]) }",
        ]

        conditions = bias_conditions + random.sample(all_conditions, random.randint(2, 3))
        conditions_str = " and ".join(conditions)

        tp = random.uniform(0.05, 0.12)
        sl = random.uniform(0.015, 0.04)
        max_hold = random.randint(5, 20)

        strategy_id = f"Agent_{os.getpid()}_{datetime.now().strftime('%H%M%S')}_{random.randint(1000,9999)}"
        return {
            "strategy_id": strategy_id,
            "strategy_name": f"{direction} Strategy",
            "entry_description": conditions_str,
            "entry_conditions": conditions_str,
            "stop_loss_pct": sl,
            "take_profit_pct": tp,
            "max_holding_bars": max_hold,
            "direction": direction,
            "regime": regime,
        }

    # ========== Prompt Builders ==========

    def _build_researcher_prompt(self, snapshot: dict, failed_summary: str) -> str:
        """
        構造給 team-researcher sub-agent 的自主研究 prompt。

        Karpathy AutoResearch 原則：
        - Agent 只修改一個文件（這裡：自己寫的 Python backtest 腳本）
        - 跑固定時間的實驗（一次完整 backtest）
        - 檢查關鍵指標，自己決定 KEEP 或 DISCARD
        - 持續循環，不停下來
        """
        return f"""你是 team-researcher，一個真正自主的量化研究 Agent。

## 你的使命：Karpathy AutoResearch

你的工作是：
1. 分析市場快照 + 失敗歷史
2. 自己想到一個新的交易策略（**不要用模板，不要抄過去的策略**）
3. 把策略寫成 Python 回測程式
4. 自己執行回測
5. 自己解讀結果，**自己決定 KEEP 或 DISCARD**
6. 把結論寫入 output JSON

## 市場快照
```
Regime:       {snapshot.get('regime', 'N/A')}
RSI(14):      {snapshot.get('rsi_14', 'N/A')}
ATR:          {snapshot.get('atr', 'N/A')}
MA200 Slope:  {snapshot.get('ma200_slope', 'N/A')}%
Vol Ratio:    {snapshot.get('vol_ratio', 'N/A')}
7日動量:      {snapshot.get('trend_7d', 'N/A')}%
收盤價:       {snapshot.get('close', 'N/A')}
MA10/20/50/200: {snapshot.get('ma10', 'N/A')}/{snapshot.get('ma20', 'N/A')}/{snapshot.get('ma50', 'N/A')}/{snapshot.get('ma200', 'N/A')}
```

## 失敗教訓（不要重蹈覆轍）
{failed_summary if failed_summary else "尚無失敗記錄"}

## 目標門檻（同時滿足才 KEEP）
- 勝率 (Win Rate) ≥ 50%
- 盈虧比 (Profit Factor) ≥ 2.0
- 最大回撤 (Max Drawdown) ≤ 30%
- Sharpe Ratio ≥ 1.5

## 你要自己想到的新策略方向

**不要用模板！不要複製現有的 entry_condition！**

參考以下方向，自己想出一個新的組合：
- **RSI 區域 + 成交量**：如 RSI 在 35-50 區間 + vol_ratio 放大，這是新的（不是 RSI<40 + vol>1.5）
- **雙均線交叉 + MA200 斜率確認**：如 MA10 上穿 MA20 且 MA200 斜率 > 0.1%（不是 close > ma200）
- **動量逆轉 + Regime Filter**：如 trend_7d < -3% + regime == BEAR + RSI < 50（不是簡單的 RSI 超賣）
- **ATR 標準化進場**：如 (close - ma20) / atr < 1.5（用 ATR 過濾假突破）
- **時間濾網**：如 RSI < 40 且 day_of_week in [1, 2]（避開週一开盘）
- **BB Bandwidth + 趨勢**：如 bandwidth < 0.05 且 trend_7d > 2%（波動率壓縮後的方向爆發）
- **MACD 柱狀圖**：如 MACD hist > 0 且 close > ma200（不是簡單的 MACD 交叉）
- **成交量潮汐**：如 vol_ratio 由低轉高（由 <0.8 變 >1.3），配合 trend_7d > 0

**你的策略必須是以上方向之一的新組合，或者完全自己想的新方向。**
**不能是：RSI<40 + vol>1.5 或 close>ma200 + RSI<45 這類已知組合。**

## 示範：你自己寫回測程式

以下是需要寫入 output JSON 的 Python code template：
```python
import sys, json
from pathlib import Path

PROJECT_ROOT = Path('{PROJECT_ROOT}')
DATA_PATH = PROJECT_ROOT / "data" / "btcusdt_1d.parquet"
sys.path.insert(0, str(PROJECT_ROOT / "autoresearch"))
from ai_researcher import MarketData, StrategyBacktester

# 1. 載入數據
import pandas as pd
df = pd.read_parquet(DATA_PATH)
market = MarketData(df)

# 2. 你自己想的策略（entry 條件要原創）
entry_conditions = "你自己想的 Python 表達式，例如: snap.rsi < 35 and snap.close > snap.ma200"
entry_fn = eval(f"lambda snap: {entry_conditions}")

stop_loss_pct = 0.02   # 你自己定的止損
take_profit_pct = 0.08  # 你自己定的止盈
max_holding_bars = 10   # 你自己定的持倉上限

# 3. 跑回測
backtester = StrategyBacktester(market)
result = backtester.backtest(
    strategy_id="YOUR_STRATEGY_ID",
    strategy_name="YOUR_STRATEGY_NAME",
    entry_description=entry_conditions,
    entry_fn=entry_fn,
    stop_loss_pct=stop_loss_pct,
    take_profit_pct=take_profit_pct,
    max_holding_bars=max_holding_bars,
)

# 4. 自己決定 KEEP 或 DISCARD
wr = result.win_rate
pf = result.profit_factor
dd = result.max_drawdown
sh = result.sharpe

KEEP = (wr >= 50.0 and pf >= 2.0 and dd <= 30.0 and sh >= 1.5)
decision = "KEEP" if KEEP else "DISCARD"

# 5. 輸出 JSON（路徑由外部傳入，prompt 變數 {{output_path}}）
output = {{
    "strategy_id": "YOUR_STRATEGY_ID",
    "strategy_name": "YOUR_STRATEGY_NAME",
    "entry_conditions": entry_conditions,
    "entry_description": entry_conditions,
    "stop_loss_pct": stop_loss_pct,
    "take_profit_pct": take_profit_pct,
    "max_holding_bars": max_holding_bars,
    "direction": "LONG 或 SHORT",
    "regime": "市場 Regime",
    "win_rate": wr,
    "profit_factor": pf,
    "max_drawdown": dd,
    "sharpe": sh,
    "total_trades": result.total_trades,
    "wins": result.wins,
    "losses": result.losses,
    "timeouts": result.timeouts,
    "decision": decision,
    "agent_reasoning": "你為什麼選擇這個策略方向？失敗了哪裡？學到了什麼？",
}}

with open("{{output_path}}", "w") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)
```

## 重要原則

1. **你自己想策略，不要用模板**。用上面參考方向想一個新組合
2. **你自己估參數**：TP/SL/持倉時間沒有標準答案，根據你的策略邏輯合理估
3. **你自己跑 backtest**：把上面 template 複製到一個 .py 檔案，執行它
4. **你自己做 KEEP/DISCARD**：嚴格按 4/4 門檻，不要自我感覺良好
5. **輸出 JSON 到 {{output_path}}**（這個路徑由外面傳入）
6. 數據檔：{DATA_PATH}
7. **strategy_id 格式**：`Agent_YYYYMMDD_HHMMSS_rrrrrr`（自己生成）
"""

    # ---- Agent 驅動輔助方法 ----

    def get_last_result(self) -> Optional[dict]:
        """取得上一輪回測結果（用於控制變數法決策）"""
        if not hasattr(self, "_last_result"):
            return None
        return self._last_result

    def set_last_result(self, result: dict):
        """設置上一輪結果（由 orchestrator 呼叫）"""
        self._last_result = result

    def decide_direction(
        self, last_result: Optional[dict], failed_summary: str, regime: str
    ) -> str:
        """
        根據 Regime + 失敗教訓，決定策略方向。
        
        規則：
        - Regime == "BULL" → LONG（不做空）
        - Regime == "BEAR" → SHORT（不做多）
        - Regime == "RANGE" → LONG 或 SHORT（根據 RSI 位置）
        - 上次 WR 卡在 45% → 加 regime filter 或收紧进场
        - 上次 DD 太大 → 收緊止損
        """
        if last_result is None:
            # 首次生成：根據 regime 預設方向
            if regime == "BULL":
                return "LONG"
            elif regime == "BEAR":
                return "SHORT"
            else:
                return random.choice(["LONG", "SHORT"])

        wr = last_result.get("win_rate", 0)
        pf = last_result.get("profit_factor", 0)
        dd = last_result.get("max_drawdown", 0)
        sh = last_result.get("sharpe", 0)
        last_dir = last_result.get("direction", "LONG")

        # 如果上次 4/4 全部達標 → 保持相同方向（鞏固）
        if (wr >= TARGET.win_rate and pf >= TARGET.profit_factor
                and dd <= TARGET.max_drawdown and sh >= TARGET.sharpe):
            return last_dir

        # 如果 WR 卡在 45% 附近 → 切 regime filter 或换方向
        if 40 <= wr < TARGET.win_rate:
            if regime == "BULL":
                return "LONG"
            elif regime == "BEAR":
                return "SHORT"
            else:
                return "LONG" if last_dir == "SHORT" else "SHORT"

        # 如果 PF 太低（< 1.5）→ 可能是進場時機問題，换方向试试
        if pf < 1.5:
            return "LONG" if last_dir == "SHORT" else "SHORT"

        # 如果 DD 太大 → 保持方向但收紧止損
        if dd > TARGET.max_drawdown * 1.5:
            return last_dir

        # 預設：根據 regime
        if regime == "BULL":
            return "LONG"
        elif regime == "BEAR":
            return "SHORT"
        else:
            return random.choice(["LONG", "SHORT"])

    def get_avoid_patterns(self) -> List[str]:
        """從 failed_strategies.json 提取要避開的條件關鍵詞（避免重複無效組合）"""
        avoid = []
        failures = self.failures.load()
        if not failures:
            return avoid

        for f in failures[-5:]:  # 只看最近 5 次
            # 提取每個單獨條件的關鍵詞（rsi, vol_ratio, trend_7d, ma200 等）
            desc = f.entry_description.lower()
            # 把 "snap.rsi < 35 and snap.trend_7d < 0 and snap.vol_ratio > 1.3"
            # 拆成 ["snap.rsi < 35", "snap.trend_7d < 0", "snap.vol_ratio > 1.3"]
            parts = desc.split(" and ")
            for part in parts:
                part = part.strip()
                if part:
                    avoid.append(part)
            # 也記錄「完整條件組合」當作指紋（用於指紋比對）
            avoid.append(desc)
        return avoid

    def _get_strategy_template(
        self, direction: str, regime: str, rsi_val: float,
        vol_ratio: float, trend_7d: float, avoid_patterns: List[str] = None
    ) -> dict:
        """
        根據方向 + 市場狀態 + 失敗歷史，選擇策略模板。
        
        每個模板包含：
        - name: 模板名稱
        - conditions: 進場條件列表
        - default_tp/sl/max_hold: 預設參數
        """
        templates = {
            # A. Regime Filter + RSI 均值回歸（多頭市場 RSI<40 做多）
            "LONG_REGIME": {
                "name": "Regime_RSI_MeanRev",
                "conditions": [
                    "snap.regime == 'BULL'",
                    "snap.rsi < 40",
                    "snap.close > snap.ma200",
                ],
                "default_tp": 0.08, "default_sl": 0.02, "default_max_hold": 10,
            },
            # B. BB Squeeze + Volume Breakout（波動率壓縮後的方向爆發）
            "BB_SQUEEZE": {
                "name": "BB_Squeeze_VolBreak",
                "conditions": [
                    "snap.vol_ratio > 1.5",
                    "snap.rsi < 45",
                    "snap.trend_7d > 0",
                ],
                "default_tp": 0.10, "default_sl": 0.02, "default_max_hold": 12,
            },
            # C. Multi-Timeframe Momentum（4H 確認趨勢 + 1D 找進場點）
            "MTF_MOMENTUM": {
                "name": "MultiTF_Momentum",
                "conditions": [
                    "snap.trend_7d > 5",
                    "snap.rsi < 60",
                    "snap.close > snap.ma50",
                ],
                "default_tp": 0.07, "default_sl": 0.025, "default_max_hold": 15,
            },
            # D. MACD Histogram Convergence（動能耗盡後的反轉）
            "MACD_REVERSAL": {
                "name": "MACD_Histogram_Conv",
                "conditions": [
                    "snap.rsi < 35",
                    "snap.trend_7d < 0",
                    "snap.vol_ratio > 1.3",
                ],
                "default_tp": 0.06, "default_sl": 0.02, "default_max_hold": 8,
            },
            # E. Volatility Regime Switch（低波動期進場，高波動期觀望）
            "VOL_SWITCH": {
                "name": "VolRegime_Switch",
                "conditions": [
                    "snap.rsi < 50",
                    "snap.trend_7d > 0",
                    "snap.close > snap.ma20",
                ],
                "default_tp": 0.05, "default_sl": 0.02, "default_max_hold": 10,
            },
            # F. Trend Following with RSI pullback
            "TREND_RSI_PULLBACK": {
                "name": "Trend_RSI_Pullback",
                "conditions": [
                    "snap.trend_7d > 3",
                    "snap.rsi < 45",
                    "snap.close > snap.ma10",
                ],
                "default_tp": 0.08, "default_sl": 0.02, "default_max_hold": 12,
            },
            # G. Mean Reversion with volume confirmation
            "MEAN_REV_VOL": {
                "name": "MeanRev_VolConfirm",
                "conditions": [
                    "snap.rsi < 40",
                    "snap.vol_ratio > 1.5",
                    "snap.trend_7d > 0",
                ],
                "default_tp": 0.06, "default_sl": 0.015, "default_max_hold": 10,
            },
            # H. RSI Extreme + MA Confirmation (新模板，避免總是用 MACD_REVERSAL)
            "RSI_EXTREME_MA": {
                "name": "RSI_Extreme_MA_Confirm",
                "conditions": [
                    "snap.rsi < 30",
                    "snap.close > snap.ma20",
                    "snap.vol_ratio > 1.3",
                ],
                "default_tp": 0.07, "default_sl": 0.02, "default_max_hold": 10,
            },
            # I. Volume Spike + Trend（成交量突量順勢）
            "VOL_SPIKE_TREND": {
                "name": "VolSpike_TrendFollow",
                "conditions": [
                    "snap.vol_ratio > 2.0",
                    "snap.trend_7d > 5",
                    "snap.rsi < 55",
                ],
                "default_tp": 0.09, "default_sl": 0.025, "default_max_hold": 12,
            },
        }

        # 根據方向 + regime 建立候選池
        if direction == "LONG":
            if regime == "BULL":
                candidates = ["LONG_REGIME", "TREND_RSI_PULLBACK", "RSI_EXTREME_MA", "VOL_SPIKE_TREND"]
            elif regime == "BEAR":
                candidates = ["VOL_SWITCH", "MEAN_REV_VOL", "VOL_SPIKE_TREND"]
            else:  # RANGE
                candidates = ["MTF_MOMENTUM", "VOL_SWITCH", "BB_SQUEEZE", "RSI_EXTREME_MA"]
        else:  # SHORT
            if regime == "BEAR":
                candidates = ["MACD_REVERSAL", "BB_SQUEEZE", "VOL_SPIKE_TREND"]
            elif regime == "BULL":
                candidates = ["VOL_SWITCH", "MTF_MOMENTUM", "RSI_EXTREME_MA"]
            else:  # RANGE
                candidates = ["MACD_REVERSAL", "BB_SQUEEZE", "VOL_SWITCH"]

        # ---- 避開最近用過的模板（模板去重）----
        recent_templates = (self._research_state.get_recent_templates() if self._research_state else [])
        if recent_templates:
            last_template = recent_templates[-1]
            # 如果上次也是同一 regime+direction 組合，並且用了同一個模板，
            # 這次強制更換
            for recent in reversed(recent_templates[-3:]):
                if recent in candidates:
                    candidates.remove(recent)

        if not candidates:
            candidates = list(templates.keys())

        template_key = random.choice(candidates)
        return templates.get(template_key, templates["VOL_SWITCH"])

    def _decide_vary_dimension(
        self, last_result: Optional[dict], base_template: dict
    ) -> str:
        """
        控制變數法：決定這次實驗要改哪個維度。
        
        維度候選：RSI_threshold / TP / SL / max_holding / 添加新條件
        """
        if last_result is None:
            return "TP"  # 首次：先改 TP

        wr = last_result.get("win_rate", 0)
        pf = last_result.get("profit_factor", 0)
        dd = last_result.get("max_drawdown", 0)
        sh = last_result.get("sharpe", 0)
        timeouts = last_result.get("timeouts", 0)
        total = last_result.get("total_trades", 1)

        # 根據上次失敗原因選擇維度
        if wr < TARGET.win_rate and wr > 0:
            return "RSI_threshold"   # WR 不夠：收紧 RSI 条件
        if pf < TARGET.profit_factor * 0.8:
            return "SL"               # PF 太低：收緊止損
        if dd > TARGET.max_drawdown:
            return "SL"               # DD 太大：收緊止損
        if timeouts / max(total, 1) > 0.25:
            return "max_holding"     # timeout 太多：縮短持倉時間
        if pf > TARGET.profit_factor and wr < TARGET.win_rate:
            return "RSI_threshold"   # PF 好但 WR 差：收紧 RSI
        return random.choice(["TP", "SL", "RSI_threshold"])

    def _apply_control_variate(
        self,
        conditions: List[str],
        dim_to_vary: str,
        base_template: dict,
    ) -> tuple:
        """
        對選定維度應用控制變數法（只改一個）。
        
        Returns: (conditions_str, tp, sl, max_hold)
        """
        tp = base_template["default_tp"]
        sl = base_template["default_sl"]
        max_hold = base_template["default_max_hold"]

        if dim_to_vary == "TP":
            tp = random.choice([0.05, 0.06, 0.07, 0.08, 0.10, 0.12])
        elif dim_to_vary == "SL":
            sl = random.choice([0.015, 0.02, 0.025, 0.03])
        elif dim_to_vary == "max_holding":
            max_hold = random.choice([6, 8, 10, 12, 15])
        elif dim_to_vary == "RSI_threshold":
            # 調整 RSI 閾值（更嚴格或更寬鬆）
            new_conds = []
            for c in conditions:
                if "snap.rsi < 40" in c:
                    new_conds.append(random.choice([
                        "snap.rsi < 35",
                        "snap.rsi < 38",
                        "snap.rsi < 42",
                    ]))
                elif "snap.rsi < 45" in c:
                    new_conds.append(random.choice([
                        "snap.rsi < 40",
                        "snap.rsi < 43",
                    ]))
                else:
                    new_conds.append(c)
            conditions = new_conds

        conditions_str = " and ".join(conditions)
        return conditions_str, tp, sl, max_hold

    def _compute_fingerprint(
        self, conditions_str: str, tp: float, sl: float,
        max_hold: int, direction: str
    ) -> str:
        """
        計算策略指紋（用於去重）。
        指紋 = direction + 所有條件標準化字串 + 參數
        """
        # 標準化：移除空格、小寫
        normalized = conditions_str.lower().replace(" ", "").replace("  ", "")
        return f"{direction}|{normalized}|tp={tp}|sl={sl}|hold={max_hold}"

    def _reverse_conditions(self, conditions_str: str) -> str:
        """
        將 LONG 條件反轉為 SHORT 條件。
        - > snap.X → < snap.X
        - < snap.X → > snap.X
        - regime == 'BULL' → regime == 'BEAR'
        """
        reversed_conds = []
        for c in conditions_str.split(" and "):
            c = c.strip()
            if "> snap." in c and "<" not in c:
                parts = c.split(">")
                reversed_conds.append(f"{parts[0].strip()} < {parts[1].strip()}")
            elif "< snap." in c and ">" not in c:
                parts = c.split("<")
                reversed_conds.append(f"{parts[0].strip()} > {parts[1].strip()}")
            elif "==" in c:
                if "'BULL'" in c:
                    reversed_conds.append(c.replace("'BULL'", "'BEAR'"))
                elif "'BEAR'" in c:
                    reversed_conds.append(c.replace("'BEAR'", "'BULL'"))
                elif "'RANGE'" in c:
                    reversed_conds.append(c)
                else:
                    reversed_conds.append(c)
            else:
                reversed_conds.append(c)
        return " and ".join(reversed_conds)

    def _swap_condition(
        self,
        current_conditions: List[str],
        template_conditions: List[str],
        avoid_patterns: List[str]
    ) -> List[str]:
        """
        更換一個條件，產生新的條件組合。
        用於指紋重複時的替换。
        """
        import random as _random
        
        # 所有可用條件池
        all_conditions = list(template_conditions)
        
        # 找一個不在當前條件中、也不在 avoid_patterns 中的條件
        candidates = []
        for cond in all_conditions:
            if cond in current_conditions:
                continue
            # 檢查是否在 avoid_patterns 中（精確匹配單一條件）
            if cond in avoid_patterns:
                continue
            candidates.append(cond)
        
        if not candidates:
            # 如果找不到乾淨候選，隨機選一個不同的
            other = [c for c in all_conditions if c not in current_conditions]
            if other:
                candidates = other
        
        if candidates:
            # 隨機換掉第一個條件
            new_conditions = list(current_conditions)
            idx_to_replace = _random.randint(0, len(new_conditions) - 1)
            new_conditions[idx_to_replace] = _random.choice(candidates)
            return new_conditions
        
        return current_conditions

    def _parse_failed_summary(self, summary: str) -> List[str]:
        """從失敗摘要中提取關鍵詞（用於避開）"""
        keywords = []
        if not summary or summary == "尚無失敗記錄":
            return keywords
        tokens = summary.lower().replace("-", " ").replace(":", " ").split()
        bad_tokens = ["rsi", "ma", "vol", "volume", "趨勢", "超賣", "超買", "死亡", "黃金"]
        keywords = [t for t in tokens if any(b in t for b in bad_tokens)]
        return keywords


# ========== QA 驗證邏輯 ==========

class QAVerifier:
    """
    team-qa 獨立驗證模組（Phase 2B）。

    ★ 核心原則：QA 只讀 research_log.md 的 entry_conditions 欄位（STEP 2），
      不看 researcher 的 entry_fn 或 backtest 代碼。

    驗證流程：
      1. 從 research_log.md 的最新實驗記錄讀取 entry_conditions
      2. 自己 parse 條件字串，構造獨立的 entry_fn
      3. 自己跑 backtest
      4. 對比 researcher vs QA 的 WR / PF 誤差
      5. 如果 entry_conditions 有歧義，自己做合理假設並在 QA 報告中備註
    """

    def __init__(self, market_data: MarketData):
        self.market = market_data
        self._qa_log: List[dict] = []

    def verify(self, strategy: dict, researcher_result: BacktestResult) -> dict:
        """
        真正的獨立 QA 驗證（Phase 2B）。

        ★ QA 不使用 strategy["entry_fn"]，而是：
          1. 從 research_log.md 最新條目讀取 entry_conditions
          2. 獨立 parse 字串 → 構造自己的 entry_fn
          3. 獨立跑 backtest
        """
        # ---- Step 1: 從 research_log.md 讀取 entry_conditions ----
        entry_conditions, log_notes = self._read_entry_conditions_from_log()
        if not entry_conditions:
            # fallback: 從 strategy dict 取（不應該發生，除非 log 還沒寫）
            entry_conditions = strategy.get("entry_conditions",
                                            strategy.get("entry_description", ""))
            log_notes = "⚠️ QA 從 strategy dict 讀取（research_log.md 尚未寫入）"

        stop_loss = strategy.get("stop_loss_pct", 0.02)
        take_profit = strategy.get("take_profit_pct", 0.05)
        max_holding = strategy.get("max_holding_bars", 10)

        # ---- Step 2: QA 自己 parse entry_conditions，構造獨立的 entry_fn ----
        entry_fn, parse_notes = self.parse_conditions_to_function(entry_conditions)
        log_notes = f"{log_notes} | QA parse: {parse_notes}"

        # ---- Step 3: QA 自己跑 backtest ----
        backtester = StrategyBacktester(self.market)
        qa_result = backtester.backtest(
            strategy_id=strategy["strategy_id"] + "_qa",
            strategy_name=strategy["strategy_name"] + " (QA)",
            entry_description=entry_conditions,
            entry_fn=entry_fn,
            stop_loss_pct=stop_loss,
            take_profit_pct=take_profit,
            max_holding_bars=max_holding,
        )

        # ---- Step 4: 對比 researcher vs QA ----
        qa_wr = qa_result.win_rate
        qa_pf = qa_result.profit_factor

        if researcher_result is None:
            # 沒有 researcher 結果時，QA 只跑獨立回測，不做對比（乾跑/首次）
            researcher_wr = None
            researcher_pf = None
            wr_err = None
            pf_err = None
            wr_ok = True
            pf_ok = True
            status = "QA_ONLY"
        else:
            researcher_wr = researcher_result.win_rate
            researcher_pf = researcher_result.profit_factor

            wr_err = abs(researcher_wr - qa_wr)
            pf_err = (abs(researcher_pf - qa_pf) / max(researcher_pf, 0.01) * 100
                      if researcher_pf > 0 else 999.0)

            wr_ok = wr_err < 5.0
            pf_ok = pf_err < 10.0
            status = "VERIFIED" if (wr_ok and pf_ok) else "FAIL"

        # ---- Step 5: 寫入 qa_*.json ----
        self.save_qa_result(strategy["strategy_id"], {
            "researcher_wr": researcher_wr,
            "qa_wr": qa_wr,
            "wr_error_pct": round(wr_err, 2) if wr_err is not None else None,
            "researcher_pf": researcher_pf,
            "qa_pf": qa_pf,
            "pf_error_pct": round(pf_err, 2) if pf_err is not None else None,
            "wr_ok": bool(wr_ok),
            "pf_ok": bool(pf_ok),
            "status": status,
            "verified_at": datetime.now().isoformat(),
            "entry_conditions_used": entry_conditions,
            "parse_notes": parse_notes,
            "log_notes": log_notes,
            "qa_metrics": {
                "total_trades": qa_result.total_trades,
                "wins": qa_result.wins,
                "losses": qa_result.losses,
                "timeouts": qa_result.timeouts,
                "win_rate": qa_result.win_rate,
                "profit_factor": qa_result.profit_factor,
                "max_drawdown": qa_result.max_drawdown,
                "sharpe": qa_result.sharpe,
            },
        })

        return {
            "strategy_id": strategy["strategy_id"],
            "researcher_wr": researcher_wr,
            "qa_wr": qa_wr,
            "wr_error_pct": round(wr_err, 2) if wr_err is not None else None,
            "researcher_pf": researcher_pf,
            "qa_pf": qa_pf,
            "pf_error_pct": round(pf_err, 2) if pf_err is not None else None,
            "wr_ok": wr_ok,
            "pf_ok": pf_ok,
            "status": status,
            "verified_at": datetime.now().isoformat(),
            "parse_notes": parse_notes,
        }

    def _read_entry_conditions_from_log(self) -> tuple:
        """
        從 research_log.md 的最新實驗記錄讀取 entry_conditions。

        Returns: (entry_conditions_str, notes)
        """
        if not RESEARCH_LOG.exists():
            return "", "⚠️ research_log.md 不存在"

        try:
            content = RESEARCH_LOG.read_text(encoding="utf-8")
            # 找倒數第一個 "STEP 2" 區塊
            steps = content.split("#### STEP 2")
            if len(steps) < 2:
                return "", "⚠️ research_log.md 無 STEP 2 記錄"

            last_step2 = steps[-1]

            # 優先找 __ENTRY_EXPRESSION__: 行（直接是 Python 表達式）
            for line in last_step2.split("\n"):
                if "__ENTRY_EXPRESSION__:" in line:
                    expr = line.split("__ENTRY_EXPRESSION__:")[1].strip()
                    return expr, "✅ 從 __ENTRY_EXPRESSION__ 讀取"

            # Fallback: 解析 "進場條件（entry_conditions）：" 行
            for line in last_step2.split("\n"):
                if "進場條件（entry_conditions）：" in line:
                    expr = line.split("）：")[1].strip() if "）：" in line else ""
                    if expr:
                        return expr, "✅ 從 STEP 2 進場條件讀取"

        except Exception as e:
            return "", f"⚠️ 讀取 log 失敗: {e}"

        return "", "⚠️ 無法解析 STEP 2 entry_conditions"

    def parse_conditions_to_function(self, entry_conditions: str) -> tuple:
        """
        QA 自己將 entry_conditions 字串 parse 成 Python 函數。

        語法支援：
          - snap.rsi < 40
          - snap.close > snap.ma200
          - snap.regime == 'BULL'
          - snap.trend_7d > 0
          - snap.vol_ratio > 1.5
          - 條件1 and 條件2 and 條件3

        Returns: (entry_fn, notes)
        """
        notes = []
        if not entry_conditions:
            return lambda snap: False, "⚠️ 空 entry_conditions"

        try:
            # 處理 "and" / "AND" 連接的多條件
            normalized = entry_conditions.strip()
            # 處理自然語言描述（如 "RSI < 40 + close > MA200"）
            # 轉換為 Python 表達式
            expr = self._normalize_condition_string(normalized)
            entry_fn = eval(f"lambda snap: {expr}")
            notes.append("✅ 成功解析 entry_conditions")
            return entry_fn, "; ".join(notes)
        except Exception as e:
            notes.append(f"⚠️ parse 失敗: {e}，使用 fallback lambda")
            # Fallback: 嘗試更寬鬆的解析
            try:
                simplified = self._simplified_parse(normalized)
                entry_fn = eval(f"lambda snap: {simplified}")
                notes.append("✅ fallback 解析成功")
                return entry_fn, "; ".join(notes)
            except Exception as e2:
                notes.append(f"❌ fallback 也失敗: {e2}")
                return lambda snap: False, "; ".join(notes)

    def _normalize_condition_string(self, s: str) -> str:
        """
        將 entry_conditions 字串正規化為 Python 表達式。

        處理：
          - 中文描述如 "RSI < 40" → "snap.rsi < 40"
          - "+" 連接 → " and "
          - "AND" / "And" → "and"
        """
        result = s

        # 將 "+" 置換為 " and "
        result = result.replace(" + ", " and ")
        result = result.replace(" +", " and ")
        result = result.replace("+ ", " and ")

        # 將自然語言 key 替換為 snap. 前綴
        # 使用 negative lookbehind (?<!snap.) 確保不會匹配到 snap.rsi 中的 "rsi"
        import re
        replacements = [
            (r'(?<!snap\.)(?<!\w)RSI\b', "snap.rsi"),
            (r'(?<!snap\.)(?<!\w)rsi\b', "snap.rsi"),
            (r'(?<!snap\.)(?<!\w)MA200\b', "snap.ma200"),
            (r'(?<!snap\.)(?<!\w)ma200\b', "snap.ma200"),
            (r'(?<!snap\.)(?<!\w)MA50\b', "snap.ma50"),
            (r'(?<!snap\.)(?<!\w)ma50\b', "snap.ma50"),
            (r'(?<!snap\.)(?<!\w)MA20\b', "snap.ma20"),
            (r'(?<!snap\.)(?<!\w)ma20\b', "snap.ma20"),
            (r'(?<!snap\.)(?<!\w)MA10\b', "snap.ma10"),
            (r'(?<!snap\.)(?<!\w)ma10\b', "snap.ma10"),
            (r'(?<!snap\.)(?<!\w)收盤價\b', "snap.close"),
            (r'(?<!snap\.)(?<!\w)close\b', "snap.close"),
            (r'(?<!snap\.)(?<!\w)成交量\b', "snap.vol_ratio"),
            (r'(?<!snap\.)(?<!\w)vol_ratio\b', "snap.vol_ratio"),
            (r'(?<!snap\.)(?<!\w)動量\b', "snap.trend_7d"),
            (r'(?<!snap\.)(?<!\w)trend_7d\b', "snap.trend_7d"),
            (r'(?<!snap\.)(?<!\w)regime\b', "snap.regime"),
            (r'(?<!snap\.)(?<!\w)MA200_斜率\b', "snap.ma200_slope"),
            (r'(?<!snap\.)(?<!\w)ma200_slope\b', "snap.ma200_slope"),
            (r'(?<!snap\.)(?<!\w)ATR\b', "snap.atr"),
            (r'(?<!snap\.)(?<!\w)atr\b', "snap.atr"),
            (r'(?<!snap\.)(?<!\w)open\b', "snap.open"),
            (r'(?<!snap\.)(?<!\w)high\b', "snap.high"),
            (r'(?<!snap\.)(?<!\w)low\b', "snap.low"),
        ]

        for pattern, replacement in replacements:
            result = re.sub(pattern, replacement, result)

        # 移除所有非關鍵字的中文字符（干擾表達式）
        import re
        result = re.sub(r'[\u4e00-\u9fff\u3000-\u303f\uff00-\uffef]', ' ', result)
        result = result.strip()

        # 確保都是有效的 Python 表達式
        result = result.replace("  ", " ")
        result = result.replace(" and and ", " and ")
        result = result.replace(" or or ", " or ")

        return result

    def _simplified_parse(self, s: str) -> str:
        """
        簡化解析：直接提取關鍵數值条件。
        當正規化失敗時使用（作為 fallback）。
        """
        import re
        result_parts = []

        # 找 RSI < 或 > 數值
        rsi_match = re.search(r'RSI\s*[<>]\s*(\d+)', s, re.IGNORECASE)
        if rsi_match:
            op = ">" if ">" in rsi_match.group() else "<"
            result_parts.append(f"snap.rsi {op} {rsi_match.group(1)}")

        # 找 vol_ratio > 數值
        vol_match = re.search(r'vol[_\s]*ratio\s*[<>]\s*([\d.]+)', s, re.IGNORECASE)
        if vol_match:
            op = ">" if ">" in vol_match.group() else "<"
            result_parts.append(f"snap.vol_ratio {op} {vol_match.group(1)}")

        # 找收盤價 > MA200
        if "close" in s.lower() and "ma200" in s.lower():
            if ">" in s:
                result_parts.append("snap.close > snap.ma200")
            else:
                result_parts.append("snap.close < snap.ma200")

        # 找 regime
        regime_match = re.search(r"regime\s*==\s*['\"](BULL|BEAR|RANGE)['\"]", s, re.IGNORECASE)
        if regime_match:
            result_parts.append(f"snap.regime == '{regime_match.group(1)}'")

        # 找 trend_7d
        trend_match = re.search(r'trend[_]?7d\s*[<>]\s*([-\d.]+)', s, re.IGNORECASE)
        if trend_match:
            op = ">" if ">" in trend_match.group() else "<"
            result_parts.append(f"snap.trend_7d {op} {trend_match.group(1)}")

        if result_parts:
            return " and ".join(result_parts)
        return "False"  # 無法解析

    def save_qa_result(self, strategy_id: str, result: dict):
        """寫入 qa_*.json（QA 獨立驗證結果）"""
        safe_id = strategy_id.replace(":", "_").replace("/", "_").replace(" ", "_")
        qa_file = MEMORY_DIR / f"qa_{safe_id}_result.json"
        with open(qa_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        self._qa_log.append({"strategy_id": strategy_id, "file": str(qa_file)})
        print(f"   📝 QA 報告寫入: {qa_file.name}")


# ========== CEO 裁決邏輯 ==========

class CEOChecker:
    """
    team-ceo 裁決模組。
    
    收到 QA VERIFIED 結果後，審核 4/4 指標，裁決 KEEP / DISCARD。
    """

    def __init__(self, target: Target = TARGET):
        self.target = target

    def review(self, researcher_result: BacktestResult, qa_result: dict) -> str:
        """
        審核 4/4 指標是否達標。
        
        必須同時滿足：
          - 勝率 >= 50%
          - 盈虧比 >= 2.0
          - 最大回撤 <= 30%
          - Sharpe >= 1.5
        """
        if qa_result["status"] != "VERIFIED":
            return "DISCARD"

        wr_ok  = researcher_result.win_rate      >= self.target.win_rate
        pf_ok  = researcher_result.profit_factor >= self.target.profit_factor
        dd_ok  = researcher_result.max_drawdown  <= self.target.max_drawdown
        sh_ok  = researcher_result.sharpe        >= self.target.sharpe

        all_pass = wr_ok and pf_ok and dd_ok and sh_ok
        return "KEEP" if all_pass else "DISCARD"

    def get_failure_reasons(self, result: BacktestResult) -> List[str]:
        """生成失敗原因列表（給 FailureMemory 用，dict 或 BacktestResult）"""
        # 支援 dict
        if isinstance(result, dict):
            return self.get_failure_reasons_dict(
                result["win_rate"], result["profit_factor"],
                result["max_drawdown"], result["sharpe"]
            )
        reasons = []
        if result.win_rate < self.target.win_rate:
            reasons.append(f"勝率不足 ({result.win_rate:.1f}% < {self.target.win_rate}%)")
        if result.profit_factor < self.target.profit_factor:
            reasons.append(f"盈虧比不足 ({result.profit_factor:.2f} < {self.target.profit_factor})")
        if result.max_drawdown > self.target.max_drawdown:
            reasons.append(f"最大回撤超標 ({result.max_drawdown:.1f}% > {self.target.max_drawdown}%)")
        if result.sharpe < self.target.sharpe:
            reasons.append(f"Sharpe 不達標 ({result.sharpe:.2f} < {self.target.sharpe})")
        return reasons

    def get_failure_reasons_dict(self, wr: float, pf: float, dd: float, sh: float) -> List[str]:
        """生成失敗原因列表（直接傳數值）"""
        reasons = []
        if wr < self.target.win_rate:
            reasons.append(f"勝率不足 ({wr:.1f}% < {self.target.win_rate}%)")
        if pf < self.target.profit_factor:
            reasons.append(f"盈虧比不足 ({pf:.2f} < {self.target.profit_factor})")
        if dd > self.target.max_drawdown:
            reasons.append(f"最大回撤超標 ({dd:.1f}% > {self.target.max_drawdown}%)")
        if sh < self.target.sharpe:
            reasons.append(f"Sharpe 不達標 ({sh:.2f} < {self.target.sharpe})")
        return reasons


# ========== ResearchOrchestrator 主類 ==========

class ResearchOrchestrator:
    """
    Agent 驅動 Auto Research 主協調器。
    
    調度 team-researcher / team-qa / team-ceo 三團隊，
    執行完整的 research → QA → CEO 裁決循環。
    """

    def __init__(
        self,
        symbol: str = "BTCUSDT",
        interval: str = "1d",
        report_callback: Optional[Callable] = None,
        max_rounds: Optional[int] = None,
    ):
        self.symbol = symbol
        self.interval = interval
        self.report_callback = report_callback
        self.max_rounds = max_rounds

        # 初始化子系統
        self._init_data()
        self.failure_memory = FailureMemory()
        self.research_state = ResearchState.load()
        self.generator = StrategyGenerator(self.failure_memory, self.research_state, orchestrator=self)
        self.qa = QAVerifier(self.market)
        self.ceo = CEOChecker(TARGET)

        # 確保輸出目錄存在
        MEMORY_DIR.mkdir(parents=True, exist_ok=True)
        for f in [FAILED_STRATEGIES_FILE, BEST_STRATEGIES_FILE]:
            if not f.exists():
                f.write_text("[]" if "failed" in f.name else json.dumps({"target": asdict(TARGET), "strategies": []}, indent=2))

    def _init_data(self):
        """載入市場數據"""
        if DATA_PATH.exists():
            print(f"📊 載入市場數據：{DATA_PATH}")
            df = pd.read_parquet(DATA_PATH)
            print(f"   共 {len(df)} 根 K 線")
        else:
            print(f"⚠️  找不到數據檔：{DATA_PATH}，使用空 DataFrame")
            df = pd.DataFrame()
        self.market = MarketData(df)

    # ---- Market Snapshot ----

    def build_market_snapshot(self) -> dict:
        """構造市場快照（給 team-researcher 看）"""
        closes = self.market.closes
        return {
            "regime":       self.market.regime[-1],
            "rsi_14":       round(self.market.rsi[-1], 2),
            "atr":          round(self.market.atr[-1], 2),
            "ma200_slope":  round(self.market.ma200_slope[-1], 3),
            "trend_7d":     round(self.market.trend_7d[-1], 2),
            "vol_ratio":    round(self.market.vol_ratio[-1], 2),
            "close":        round(closes[-1], 2),
            "ma10":         round(self.market.ma10[-1], 2),
            "ma20":         round(self.market.ma20[-1], 2),
            "ma50":         round(self.market.ma50[-1], 2),
            "ma200":        round(self.market.ma200[-1], 2),
        }

    def build_failed_summary(self) -> str:
        """從 failed_strategies.json 構造失敗摘要（含具體維度失敗資訊）"""
        failures = self.failure_memory.load()
        if not failures:
            return "尚無失敗記錄"
        recent = failures[-5:]
        lines = []
        for f in recent:
            reasons = ", ".join(f.failure_reasons[:2])
            # 也標注 entry_description 關鍵字（用於條件去重）
            entry_short = f.entry_description[:60]
            lines.append(f"- [{f.strategy_name}] {entry_short}: {reasons}")
        return "\n".join(lines)

    # ========== Sub-Agent Spawn Helpers ==========

    def _spawn_and_wait_researcher(
        self, prompt: str, output_path: str, round_num: int
    ) -> dict:
        """
        用 subprocess 呼叫 openclaw sessions_spawn 派發 team-researcher agent，
        等待完成後讀取 output_path JSON。
        """
        pid = os.getpid()
        task_file = MEMORY_DIR / f"_task_researcher_{round_num}_{pid}.txt"
        task_file.write_text(prompt)

        cmd = [
            OPENCLAW_CLI,
            "sessions", "spawn",
            "--task", prompt,
            "--agent-id", "team-researcher",
            "--mode", "run",
            "--runtime", "subagent",
        ]

        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        start = time_module.time()
        timeout = 300  # 5 分鐘

        while time_module.time() - start < timeout:
            if Path(output_path).exists():
                result = json.loads(Path(output_path).read_text())
                Path(output_path).unlink(missing_ok=True)
                task_file.unlink(missing_ok=True)
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                return result
            time_module.sleep(5)

        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        task_file.unlink(missing_ok=True)
        raise TimeoutError(f"team-researcher timed out after {timeout}s")

    def _spawn_and_wait_qa(
        self, prompt: str, output_path: str
    ) -> dict:
        """派發 team-qa agent，等待完成，讀取 output_path JSON。"""
        pid = os.getpid()
        task_file = MEMORY_DIR / f"_task_qa_{pid}.txt"
        task_file.write_text(prompt)

        cmd = [
            OPENCLAW_CLI,
            "sessions", "spawn",
            "--task", prompt,
            "--agent-id", "team-qa",
            "--mode", "run",
            "--runtime", "subagent",
        ]

        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        start = time_module.time()
        timeout = 300

        while time_module.time() - start < timeout:
            if Path(output_path).exists():
                result = json.loads(Path(output_path).read_text())
                Path(output_path).unlink(missing_ok=True)
                task_file.unlink(missing_ok=True)
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                return result
            time_module.sleep(5)

        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        task_file.unlink(missing_ok=True)
        raise TimeoutError(f"team-qa timed out after {timeout}s")

    def _spawn_and_wait_ceo(
        self, prompt: str, output_path: str
    ) -> str:
        """派發 team-ceo agent，等待完成，讀取 output_path JSON，回傳裁決字串。"""
        pid = os.getpid()
        task_file = MEMORY_DIR / f"_task_ceo_{pid}.txt"
        task_file.write_text(prompt)

        cmd = [
            OPENCLAW_CLI,
            "sessions", "spawn",
            "--task", prompt,
            "--agent-id", "team-ceo",
            "--mode", "run",
            "--runtime", "subagent",
        ]

        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        start = time_module.time()
        timeout = 300

        while time_module.time() - start < timeout:
            if Path(output_path).exists():
                result = json.loads(Path(output_path).read_text())
                Path(output_path).unlink(missing_ok=True)
                task_file.unlink(missing_ok=True)
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                return result.get("decision", result.get("verdict", "DISCARD"))
            time_module.sleep(5)

        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        task_file.unlink(missing_ok=True)
        raise TimeoutError(f"team-ceo timed out after {timeout}s")

    # ========== Prompt Builders ==========

    def _build_researcher_prompt(
        self, snapshot: dict, failed_summary: str
    ) -> str:
        """
        構造完整的市場快照 prompt 給 team-researcher agent。

        包含：
        - 市場快照（regime, RSI, MA, vol_ratio, trend_7d 等）
        - failed_strategies.json 的失敗摘要（讓 Agent 避開）
        - Agent 自己想到新策略並執行（Karpathy AutoResearch 模式）
        - 結果寫入 output JSON
        """
        return f"""你是 team-researcher，負責為 crypto 交易策略進行 AI 驅動研究。

## 你的任務
1. 分析市場快照 + 失敗歷史
2. 自己想到一個新的交易策略（**不要抄過去的 entry_condition**）
3. 把策略寫成 Python 回測程式
4. 自己執行回測
5. 把結論寫入 output JSON

## 市場快照（已根據 failed_summary 生成，請參考）
```
Regime:       {snapshot.get('regime', 'N/A')}
RSI(14):      {snapshot.get('rsi_14', 'N/A')}
ATR:          {snapshot.get('atr', 'N/A')}
MA200 Slope:  {snapshot.get('ma200_slope', 'N/A')}%
Vol Ratio:    {snapshot.get('vol_ratio', 'N/A')}
7日動量:      {snapshot.get('trend_7d', 'N/A')}%
收盤價:       {snapshot.get('close', 'N/A')}
MA10/20/50/200: {snapshot.get('ma10', 'N/A')}/{snapshot.get('ma20', 'N/A')}/{snapshot.get('ma50', 'N/A')}/{snapshot.get('ma200', 'N/A')}
```

## 失敗教訓（不要重蹈覆轍）
{failed_summary if failed_summary else "尚無失敗記錄"}

## 目標門檻（同時滿足才 KEEP）
- 勝率 (Win Rate) ≥ 50%
- 盈虧比 (Profit Factor) ≥ 2.0
- 最大回撤 (Max Drawdown) ≤ 30%
- Sharpe Ratio ≥ 1.5

## 你的策略方向（不要用模板！不要複製現有 entry_condition！）

自己想到一個新組合，以下方向僅供參考，你可以自己想新的：
- RSI 區域 + 成交量（如 RSI 在 30-45 + vol_ratio 由低轉高）
- 雙均線交叉 + MA200 斜率（如 MA10 上穿 MA20 且 MA200 斜率 > 0.1%）
- 動量逆轉 + Regime（如 trend_7d < -3% + regime == BEAR）
- ATR 標準化（如 (close - ma20) / atr < 1.5）
- 成交量潮汐（如 vol_ratio 由 <0.8 變 >1.3）

## 直接執行以下 Python 代碼（已經過調試，不要修改邏輯）

```python
import sys, os
sys.path.insert(0, '{PROJECT_ROOT}')
os.environ["OPENCLAW_SUBAGENT_CONTEXT"] = "1"

from autoresearch.loop import StrategyGenerator, FailureMemory
import pandas as pd
import json

df = pd.read_parquet('{DATA_PATH}')
failure_memory = FailureMemory()

# StrategyGenerator 自己的 generate() 在 sub-agent 上下文會直接做事，不再生 sub-agent
generator = StrategyGenerator(failure_memory, None)
strategy = generator._generate_inline({snapshot}, '''{failed_summary if failed_summary else "尚無失敗記錄"}''')

output = {{
    "strategy_id": strategy['strategy_id'],
    "strategy_name": strategy['strategy_name'],
    "entry_description": strategy['entry_description'],
    "entry_conditions": strategy['entry_conditions'],
    "stop_loss_pct": strategy['stop_loss_pct'],
    "take_profit_pct": strategy['take_profit_pct'],
    "max_holding_bars": strategy['max_holding_bars'],
    "direction": strategy['direction'],
    "regime": strategy['regime'],
    "win_rate": strategy['win_rate'],
    "profit_factor": strategy['profit_factor'],
    "max_drawdown": strategy['max_drawdown'],
    "sharpe": strategy['sharpe'],
    "total_trades": strategy['total_trades'],
    "wins": strategy['wins'],
    "losses": strategy['losses'],
    "timeouts": strategy['timeouts'],
}}

with open('{{output_path}}', 'w') as f:
    json.dump(output, f, indent=2)
print("DONE")
```
)

# 組合輸出
output = {{
    "strategy_id": strategy['strategy_id'],
    "strategy_name": strategy['strategy_name'],
    "entry_description": strategy['entry_description'],
    "entry_conditions": strategy['entry_conditions'],
    "stop_loss_pct": strategy['stop_loss_pct'],
    "take_profit_pct": strategy['take_profit_pct'],
    "max_holding_bars": strategy['max_holding_bars'],
    "direction": strategy['direction'],
    "regime": strategy['regime'],
    "win_rate": result.win_rate,
    "profit_factor": result.profit_factor,
    "max_drawdown": result.max_drawdown,
    "sharpe": result.sharpe,
    "total_trades": result.total_trades,
    "wins": result.wins,
    "losses": result.losses,
    "timeouts": result.timeouts,
}}

# 寫入 output JSON
import json
with open('{{output_path}}', 'w') as f:
    json.dump(output, f, indent=2)
```

## 市場快照（參考用）
```
Regime:    {snapshot['regime']}
RSI(14):   {snapshot['rsi_14']}
ATR:       {snapshot['atr']}
MA200 Slope: {snapshot['ma200_slope']}
7日動量:   {snapshot['trend_7d']}%
成交量比:  {snapshot['vol_ratio']}
收盤價:    {snapshot['close']}
MA10:      {snapshot['ma10']}
MA20:      {snapshot['ma20']}
MA50:      {snapshot['ma50']}
MA200:     {snapshot['ma200']}
```

## 失敗策略摘要（StrategyGenerator 會自動避開）
{failed_summary if failed_summary else "尚無失敗記錄"}

## 目標參數
| 指標 | 門檻 |
|------|------|
| 勝率 (Win Rate) | ≥ 50% |
| 盈虧比 (Profit Factor) | ≥ 2.0 |
| 最大回撤 (Max Drawdown) | ≤ 30% |
| Sharpe Ratio | ≥ 1.5 |

## 重要提醒
- **必須使用 StrategyGenerator.generate()**，不要自己發想
- StrategyGenerator 有指紋追蹤，如果生成重複策略會自動更換
- 執行上述 Python 代碼，不要修改邏輯
- 輸出 JSON 檔案路徑：{{output_path}}
- 數據檔位置：{DATA_PATH}
"""

    def _build_qa_prompt(
        self, strategy: dict, researcher_result: BacktestResult
    ) -> str:
        """
        構造完整的 prompt 給 team-qa agent。

        team-qa 拿到 research_log.md 的 spec，
        獨立實作進場邏輯（不看 researcher 代碼），
        獨立跑 backtest_engine.py。
        """
        return f"""你是 team-qa，負責對 team-researcher 生成的策略進行獨立驗證。

## 你的任務
team-researcher 已經生成了策略並寫入了 research_log.md。
你需要：
1. 從 {RESEARCH_LOG} 讀取 researcher 的完整 YAML spec（不看 loop.py 裡的 researcher 代碼）
2. 獨立實作進場邏輯（根據 spec 自己寫 entry_fn）
3. 用 {DATA_PATH} 跑獨立的 backtest
4. 將結果寫入 output JSON

## researcher 的策略資訊
- Strategy ID: {strategy['strategy_id']}
- Strategy Name: {strategy['strategy_name']}
- Direction: {strategy.get('direction', 'LONG')}
- Entry Conditions: {strategy['entry_description']}
- Stop Loss: {strategy['stop_loss_pct']*100:.1f}%
- Take Profit: {strategy['take_profit_pct']*100:.1f}%
- Max Holding: {strategy['max_holding_bars']} bars

## researcher 回測結果
- Win Rate: {researcher_result.win_rate:.2f}%
- Profit Factor: {researcher_result.profit_factor:.2f}
- Max Drawdown: {researcher_result.max_drawdown:.1f}%
- Sharpe: {researcher_result.sharpe:.2f}
- Total Trades: {researcher_result.total_trades}

## QA 驗證要求
- 獨立實作 entry_fn（根據 research_log.md 的 spec，不看 researcher 代碼）
- 用相同的 {DATA_PATH} 跑 backtest
- 對比結果：
  - WR 誤差 < 5% → PASS
  - PF 誤差 < 10% → PASS
  - 雙方都 PASS → VERIFIED

## output JSON 格式
{{
  "strategy_id": "{strategy['strategy_id']}",
  "researcher_wr": {researcher_result.win_rate},
  "qa_wr": float,
  "wr_error_pct": float,
  "researcher_pf": {researcher_result.profit_factor},
  "qa_pf": float,
  "pf_error_pct": float,
  "wr_ok": bool,
  "pf_ok": bool,
  "status": "VERIFIED|FAIL",
  "verified_at": "ISO timestamp"
}}

## 重要提醒
- 輸出檔案路徑：{{output_path}}（你會在 prompt 參數中收到）
- 獨立實作，不要複製 researcher 的 entry_fn 代碼
- 從 research_log.md 讀取 spec
- 數據檔：{DATA_PATH}
"""

    def _build_ceo_prompt(
        self, researcher_result: BacktestResult, qa_result: dict
    ) -> str:
        """
        構造完整的 prompt 給 team-ceo agent。

        CEO 審核 QA VERIFIED 結果的 4/4 指標，
        裁決 KEEP / DISCARD。
        """
        return f"""你是 team-ceo，負責對經過 QA 驗證的策略做出最終裁決。

## 你的任務
審核 researcher + QA 的結果，裁決 KEEP 或 DISCARD。

## researcher 回測結果
- Win Rate: {researcher_result.win_rate:.2f}% (目標 ≥ 50%)
- Profit Factor: {researcher_result.profit_factor:.2f} (目標 ≥ 2.0)
- Max Drawdown: {researcher_result.max_drawdown:.1f}% (目標 ≤ 30%)
- Sharpe: {researcher_result.sharpe:.2f} (目標 ≥ 1.5)
- Total Trades: {researcher_result.total_trades}

## QA 驗證結果
- QA Status: {qa_result.get('status', 'N/A')}
- WR Error: {qa_result.get('wr_error_pct', 'N/A')}%
- PF Error: {qa_result.get('pf_error_pct', 'N/A')}%

## 裁決規則（4/4 全部達標才 KEEP）
- WR ≥ 50% AND
- PF ≥ 2.0 AND
- DD ≤ 30% AND
- Sharpe ≥ 1.5 AND
- QA Status == "VERIFIED"

## output JSON 格式
{{
  "decision": "KEEP|DISCARD",
  "wr_ok": bool,
  "pf_ok": bool,
  "dd_ok": bool,
  "sharpe_ok": bool,
  "qa_verified": bool,
  "all_pass": bool,
  "failure_reasons": ["reason1", "reason2"]  # 如果 DISCARD
}}

## 重要提醒
- 輸出檔案路徑：{{output_path}}（你會在 prompt 參數中收到）
- 嚴格執行 4/4 門檻，不可降低標準
- 如果任何一項未達標，裁決為 DISCARD
"""

    # ---- Team Dispatch (async, sub-agent driven) ----

    async def dispatch_researcher(self) -> dict:
        """
        派 team-researcher Agent 生成策略。

        使用 generator.generate() 派發 sub-agent，
        Agent 根據市場快照 + 失敗摘要，自己發想策略邏輯。
        產出寫入 memory/researcher_{round}.json。
        """
        snapshot = self.build_market_snapshot()
        failed_summary = self.build_failed_summary()

        # generator.generate() 內部派 sub-agent 並等待結果
        strategy = await asyncio.to_thread(
            self.generator.generate,
            snapshot,
            failed_summary,
        )
        return strategy

    async def dispatch_qa(
        self, strategy: dict, researcher_result: BacktestResult
    ) -> dict:
        """
        派 team-qa Agent 獨立驗證。

        team-qa 拿到 research_log.md 的 spec，
        獨立實作進場邏輯（不看 researcher 代碼），
        獨立跑 backtest_engine.py。
        產出寫入 memory/qa_{strategy_id}.json。
        """
        qa_output = str(MEMORY_DIR / f"qa_{strategy['strategy_id']}.json")
        prompt = self._build_qa_prompt(strategy, researcher_result)

        qa_result = await asyncio.to_thread(
            self._spawn_and_wait_qa,
            prompt=prompt,
            output_path=qa_output,
        )
        return qa_result

    async def dispatch_ceo(
        self, researcher_result: BacktestResult, qa_result: dict
    ) -> str:
        """
        派 team-ceo Agent 裁決。

        審核 QA VERIFIED 結果的 4/4 指標，
        裁決 KEEP / DISCARD。
        """
        ceo_output = str(MEMORY_DIR / f"ceo_{self.research_state.round_num}.json")
        prompt = self._build_ceo_prompt(researcher_result, qa_result)

        decision = await asyncio.to_thread(
            self._spawn_and_wait_ceo,
            prompt=prompt,
            output_path=ceo_output,
        )
        return decision

    # ---- Persistence ----

    def save_to_best_strategies(self, strategy: dict, researcher_result):
        """寫入 best_strategies.json（researcher_result 可以是 dict 或 BacktestResult）"""
        data = json.loads(BEST_STRATEGIES_FILE.read_text())
        # 支援 dict 或 BacktestResult 兩種格式
        if isinstance(researcher_result, dict):
            wr = researcher_result["win_rate"]
            pf = researcher_result["profit_factor"]
            dd = researcher_result["max_drawdown"]
            sh = researcher_result["sharpe"]
        else:
            wr = researcher_result.win_rate
            pf = researcher_result.profit_factor
            dd = researcher_result.max_drawdown
            sh = researcher_result.sharpe

        entry = {
            "strategy_id":     strategy["strategy_id"],
            "strategy_name":   strategy["strategy_name"],
            "focus_area":      "auto_research_agent",
            "round":           self.research_state.round_num,
            "entry_description": strategy.get("entry_description", ""),
            "params": {
                "stop_loss_pct":  strategy.get("stop_loss_pct", 0.02),
                "take_profit_pct": strategy.get("take_profit_pct", 0.08),
                "max_holding":     strategy.get("max_holding_bars", 10),
            },
            "metrics": {
                "win_rate":      wr,
                "profit_factor": pf,
                "max_drawdown":  dd,
                "sharpe":        sh,
            },
            "status":    "VERIFIED",
            "verified_at": datetime.now().isoformat(),
        }
        data["strategies"].append(entry)
        BEST_STRATEGIES_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False))

        # 同步寫入 factor_library.json
        self._sync_factor_library(entry)

    def _sync_factor_library(self, strategy_entry: dict):
        """同步寫入 factor_library.json"""
        if not FACTOR_LIBRARY.exists():
            FACTOR_LIBRARY.write_text("[]")
        factors = json.loads(FACTOR_LIBRARY.read_text())
        factor_entry = {
            "factor_id":   strategy_entry["strategy_id"],
            "name":        strategy_entry["strategy_name"],
            "type":        "agent_research",
            "entry":       {"long": {"auto_generated": True}},
            "stop_loss":   f"{strategy_entry['params']['stop_loss_pct']*100:.1f}%",
            "take_profit": f"{strategy_entry['params']['take_profit_pct']*100:.1f}%",
            "max_holding": f"{strategy_entry['params']['max_holding']} bars",
            "source":      "auto_research_agent",
            "status":      "candidate",
            "metrics":     strategy_entry["metrics"],
            "timeframe":   self.interval,
        }
        factors.append(factor_entry)
        FACTOR_LIBRARY.write_text(json.dumps(factors, indent=2, ensure_ascii=False))

    def save_to_failed_strategies(self, strategy: dict, researcher_result):
        """寫入 failed_strategies.json（researcher_result 可以是 dict 或 BacktestResult）"""
        from ai_researcher import FailedStrategy
        # 支援 dict 或 BacktestResult
        if isinstance(researcher_result, dict):
            wr = researcher_result["win_rate"]
            pf = researcher_result["profit_factor"]
            dd = researcher_result["max_drawdown"]
            sh = researcher_result["sharpe"]
        else:
            wr = researcher_result.win_rate
            pf = researcher_result.profit_factor
            dd = researcher_result.max_drawdown
            sh = researcher_result.sharpe

        reasons = self.ceo.get_failure_reasons_dict(wr, pf, dd, sh)
        if not reasons:
            reasons = ["未達標"]

        failure = FailedStrategy(
            strategy_id=strategy["strategy_id"],
            strategy_name=strategy["strategy_name"],
            entry_description=strategy.get("entry_description", ""),
            failure_reasons=reasons,
            win_rate=wr,
            profit_factor=pf,
            max_drawdown=dd,
            learned_rules=strategy.get("learned_rules", []),
        )
        self.failure_memory.add(failure)

    def write_research_log_step1_2(self, strategy: dict):
        """
        Phase 2C — STEP 1-2 預先寫入（實驗前，不能事後補）。

        在跑回測之前就寫入市場分析和策略邏輯杜絕事後湊答案。
        """
        t = datetime.now()
        timestamp = t.strftime("%Y-%m-%d %H:%M")
        round_num = self.research_state.round_num

        log = f"""
### 實驗 #{round_num} — {timestamp}
**策略ID**: {strategy['strategy_id']}
**策略名**: {strategy['strategy_name']}
**方向**: {strategy.get('direction', 'N/A')} | **Regime**: {strategy.get('regime', 'N/A')}
**市場快照**: RSI={strategy.get('rsi', 'N/A')}, Vol={strategy.get('vol_ratio', 'N/A')}, 7d動量={strategy.get('trend_7d', 'N/A')}
**本次控制變數維度**: {strategy.get('dimension_varied', 'N/A')}
**策略類型**: {strategy.get('strategy_type', 'N/A')}

#### STEP 1 — 市場分析（實驗前寫入）
- Regime: `{strategy.get('regime', 'N/A')}`
- RSI(14): {strategy.get('rsi', 'N/A')}
- Vol Ratio: {strategy.get('vol_ratio', 'N/A')}
- 7日動量: {strategy.get('trend_7d', 'N/A')}
- MA200 Slope: {strategy.get('ma200_slope', 'N/A')}

#### STEP 2 — 策略邏輯（完整定義，實驗前寫入，不可事後修改）
- 進場條件（entry_conditions）：{strategy.get('entry_conditions', strategy.get('entry_description', ''))}
- __ENTRY_EXPRESSION__: {strategy.get('entry_conditions', strategy.get('entry_description', ''))}
- 止盈：{strategy.get('take_profit_pct', 0)*100:.1f}%
- 止損：{strategy.get('stop_loss_pct', 0)*100:.1f}%
- 最長持倉：{strategy.get('max_holding_bars', 0)} 根
- 方向：{strategy.get('direction', 'N/A')}
- Regime Bias：{strategy.get('regime', 'N/A')}

[實驗結果待填寫 — 回測後更新]
---
"""
        with open(RESEARCH_LOG, "a", encoding="utf-8") as f:
            f.write(log)
        print(f"   📝 STEP 1-2 預先寫入 research_log.md（round #{round_num}）")

    def write_research_log_step3(
        self,
        strategy: dict,
        researcher_result,
        qa_result: dict,
        decision: str,
    ):
        """
        Phase 2C — STEP 3-5 追加寫入（實驗後更新同一區塊）。
        researcher_result 可以是 dict 或 BacktestResult。
        """
        t = datetime.now()
        timestamp = t.strftime("%Y-%m-%d %H:%M")
        round_num = self.research_state.round_num
        target = TARGET

        def icon(cond): return "✅" if cond else "❌"

        # 支援 dict 或 BacktestResult
        if isinstance(researcher_result, dict):
            r_wr = researcher_result["win_rate"]
            r_pf = researcher_result["profit_factor"]
            r_dd = researcher_result["max_drawdown"]
            r_sh = researcher_result["sharpe"]
            r_total = researcher_result.get("total_trades", 0)
            r_wins = researcher_result.get("wins", 0)
            r_losses = researcher_result.get("losses", 0)
            r_timeouts = researcher_result.get("timeouts", 0)
        else:
            r_wr = researcher_result.win_rate
            r_pf = researcher_result.profit_factor
            r_dd = researcher_result.max_drawdown
            r_sh = researcher_result.sharpe
            r_total = researcher_result.total_trades
            r_wins = researcher_result.wins
            r_losses = researcher_result.losses
            r_timeouts = researcher_result.timeouts

        wr_ok = r_wr >= target.win_rate
        pf_ok = r_pf >= target.profit_factor
        dd_ok = r_dd <= target.max_drawdown
        sh_ok = r_sh >= target.sharpe

        update_block = f"""
#### STEP 3 — 回測結果（researcher，實驗後填寫）
- 總交易：{r_total}  WIN={r_wins}  LOSS={r_losses}  TIMEOUT={r_timeouts}
- 勝率：{r_wr:.2f}% {icon(wr_ok)} (目標≥{target.win_rate}%)
- 盈虧比：{r_pf:.2f} {icon(pf_ok)} (目標≥{target.profit_factor})
- 最大回撤：{r_dd:.1f}% {icon(dd_ok)} (目標≤{target.max_drawdown}%)
- Sharpe：{r_sh:.2f} {icon(sh_ok)} (目標≥{target.sharpe})

#### STEP 4 — QA 獨立驗證（只看 entry_conditions，不看 entry_fn）
- QA parse_notes: {qa_result.get('parse_notes', 'N/A')}
- researcher WR: {r_wr:.2f}%
- QA WR: {qa_result.get('qa_wr', 0):.2f}% (誤差 {qa_result.get('wr_error_pct', 0):.2f}% {icon(qa_result.get('wr_ok', False))})
- researcher PF: {r_pf:.2f}
- QA PF: {qa_result.get('qa_pf', 0):.2f} (誤差 {qa_result.get('pf_error_pct', 0):.2f}% {icon(qa_result.get('pf_ok', False))})
- **QA 結論**: {qa_result.get('status', 'N/A')}

#### STEP 5 — CEO 裁決
- 裁決：**{decision}**
- 失敗原因（如有）：{', '.join(self.ceo.get_failure_reasons(researcher_result)) if decision == 'DISCARD' else '無'}

---
"""

        # 讀取現有 log，替换 "[實驗結果待填寫 — 回測後更新]" 標記
        if RESEARCH_LOG.exists():
            content = RESEARCH_LOG.read_text(encoding="utf-8")
            if "[實驗結果待填寫" in content:
                content = content.replace(
                    "[實驗結果待填寫 — 回測後更新]\n---\n",
                    update_block
                )
                RESEARCH_LOG.write_text(content, encoding="utf-8")
            else:
                # 如果找不到標記，直接 append
                with open(RESEARCH_LOG, "a", encoding="utf-8") as f:
                    f.write(f"\n[Round {round_num} 結果更新]\n{update_block}")
        print(f"   📝 STEP 3-5 結果寫入 research_log.md（round #{round_num}）")

    def write_research_log_full(
        self,
        strategy: dict,
        researcher_result,
        qa_result: dict,
        decision: str,
    ):
        """
        完整寫入（向後兼容 / 單次寫入用）。
        researcher_result 可以是 dict 或 BacktestResult。
        """
        t = datetime.now()
        timestamp = t.strftime("%Y-%m-%d %H:%M")
        round_num = self.research_state.round_num
        target = TARGET

        # 支援 dict 或 BacktestResult
        if isinstance(researcher_result, dict):
            r_wr = researcher_result["win_rate"]
            r_pf = researcher_result["profit_factor"]
            r_dd = researcher_result["max_drawdown"]
            r_sh = researcher_result["sharpe"]
            r_total = researcher_result.get("total_trades", 0)
            r_wins = researcher_result.get("wins", 0)
            r_losses = researcher_result.get("losses", 0)
            r_timeouts = researcher_result.get("timeouts", 0)
        else:
            r_wr = researcher_result.win_rate
            r_pf = researcher_result.profit_factor
            r_dd = researcher_result.max_drawdown
            r_sh = researcher_result.sharpe
            r_total = researcher_result.total_trades
            r_wins = researcher_result.wins
            r_losses = researcher_result.losses
            r_timeouts = researcher_result.timeouts

        def icon(cond): return "✅" if cond else "❌"
        wr_ok = r_wr >= target.win_rate
        pf_ok = r_pf >= target.profit_factor
        dd_ok = r_dd <= target.max_drawdown
        sh_ok = r_sh >= target.sharpe

        log = f"""
### 實驗 #{round_num} — {timestamp}
**策略ID**: {strategy['strategy_id']}
**策略名**: {strategy['strategy_name']}
**方向**: {strategy.get('direction', 'long')} | **Regime**: {strategy.get('regime', 'N/A')}
**市場快照**: RSI={strategy.get('rsi', 'N/A')}, Vol={strategy.get('vol_ratio', 'N/A')}, 7d動量={strategy.get('trend_7d', 'N/A')}

#### STEP 1 — 市場分析
- Regime: `{strategy.get('regime', 'N/A')}`
- RSI(14): {strategy.get('rsi', 'N/A')}
- Vol Ratio: {strategy.get('vol_ratio', 'N/A')}
- 7日動量: {strategy.get('trend_7d', 'N/A')}

#### STEP 2 — 策略邏輯（完整定義）
- 進場條件 LONG/SHORT：
  - {strategy.get('entry_description', 'N/A')}
- 止盈：{strategy.get('take_profit_pct', 0)*100:.1f}%
- 止損：{strategy.get('stop_loss_pct', 0)*100:.1f}%
- 最長持倉：{strategy.get('max_holding_bars', 0)} 根

#### 回測結果（researcher）
- 總交易：{r_total}  WIN={r_wins}  LOSS={r_losses}  TIMEOUT={r_timeouts}
- 勝率：{r_wr:.2f}% {icon(wr_ok)} (目標≥{target.win_rate}%)
- 盈虧比：{r_pf:.2f} {icon(pf_ok)} (目標≥{target.profit_factor})
- 最大回撤：{r_dd:.1f}% {icon(dd_ok)} (目標≤{target.max_drawdown}%)
- Sharpe：{r_sh:.2f} {icon(sh_ok)} (目標≥{target.sharpe})

#### STEP 4 — QA 驗證
- researcher WR: {r_wr:.2f}%
- QA WR: {qa_result.get('qa_wr', 0):.2f}% (誤差 {qa_result.get('wr_error_pct', 0):.2f}% {icon(qa_result.get('wr_ok', False))})
- researcher PF: {r_pf:.2f}
- QA PF: {qa_result.get('qa_pf', 0):.2f} (誤差 {qa_result.get('pf_error_pct', 0):.2f}% {icon(qa_result.get('pf_ok', False))})
- **QA 結論**: {qa_result.get('status', 'N/A')}

#### STEP 5 — CEO 裁決
- 裁決：**{decision}**
- 失敗原因（如有）：{', '.join(self.ceo.get_failure_reasons(researcher_result)) if decision == 'DISCARD' else '無'}

---
"""
        with open(RESEARCH_LOG, "a", encoding="utf-8") as f:
            f.write(log)

    # Alias for backward compatibility
    def write_research_log(self, strategy, researcher_result, qa_result, decision):
        return self.write_research_log_full(strategy, researcher_result, qa_result, decision)

    # ---- Single Cycle ----

    async def run_cycle(self):
        """執行一輪完整研究循環（Karpathy AutoResearch 模式）"""
        self.research_state.increment_round()
        rn = self.research_state.round_num

        print(f"\n{'='*60}")
        print(f"🔬 第 {rn} 輪  Auto Research Agent (Karpathy Mode)")
        print(f"{'='*60}")

        # Snapshot
        snapshot = self.build_market_snapshot()
        failed_summary = self.build_failed_summary()
        print(f"\n📊 市場快照: Regime={snapshot['regime']}, RSI={snapshot['rsi_14']}, Vol={snapshot['vol_ratio']}")

        # ---- ① team-researcher（真正自主研究 Agent）----
        print("\n🧠 [team-researcher] 派出去自主研究中...")
        strategy_result = await self.dispatch_researcher()

        # Sub-agent 返回的 dict 可能為 None（超時/錯誤）
        if strategy_result is None:
            print("   ❌ sub-agent 無法完成，視同 DISCARD")
            decision = "DISCARD"
            strategy = {
                "strategy_id": f"FAILED_{rn}",
                "strategy_name": "Agent Failed",
                "entry_description": "sub-agent timeout",
                "entry_conditions": "False",
                "stop_loss_pct": 0.02,
                "take_profit_pct": 0.08,
                "max_holding_bars": 10,
                "direction": "LONG",
                "regime": snapshot.get("regime", "RANGE"),
            }
            researcher_result_dict = {
                "win_rate": 0.0, "profit_factor": 0.0,
                "max_drawdown": 100.0, "sharpe": 0.0,
                "total_trades": 0, "wins": 0, "losses": 0, "timeouts": 0,
            }
            qa_result = {
                "qa_wr": 0.0, "wr_error_pct": 0.0, "wr_ok": False,
                "qa_pf": 0.0, "pf_error_pct": 0.0, "pf_ok": False,
                "status": "FAIL",
                "parse_notes": "sub-agent failed",
            }
        else:
            strategy = strategy_result
            # 從 sub-agent 的結果直接取出 metrics（sub-agent 自己跑了 backtest）
            researcher_result_dict = {
                "win_rate": strategy_result.get("win_rate", 0.0),
                "profit_factor": strategy_result.get("profit_factor", 0.0),
                "max_drawdown": strategy_result.get("max_drawdown", 100.0),
                "sharpe": strategy_result.get("sharpe", 0.0),
                "total_trades": strategy_result.get("total_trades", 0),
                "wins": strategy_result.get("wins", 0),
                "losses": strategy_result.get("losses", 0),
                "timeouts": strategy_result.get("timeouts", 0),
            }
            agent_decision = strategy_result.get("decision", "DISCARD")

            print(f"   策略ID: {strategy['strategy_id']}")
            print(f"   條件: {strategy.get('entry_description', strategy.get('entry_conditions', 'N/A'))}")
            print(f"   SL={strategy.get('stop_loss_pct', 0)*100:.1f}% | TP={strategy.get('take_profit_pct', 0)*100:.1f}% | MaxHold={strategy.get('max_holding_bars', 0)}根")
            print(f"   📊 WR={researcher_result_dict['win_rate']:.1f}% | PF={researcher_result_dict['profit_factor']:.2f} | DD={researcher_result_dict['max_drawdown']:.1f}% | Sharpe={researcher_result_dict['sharpe']:.2f}")
            print(f"   Agent 自行裁決: {agent_decision}")
            print(f"   Agent reasoning: {strategy_result.get('agent_reasoning', 'N/A')[:80]}...")

            # ---- ② team-qa 獨立驗證 ----
            print("\n🔍 [team-qa] 獨立驗證...")
            # 構造一個假的 BacktestResult 給 QA 用（因為 researcher 已經跑過了）
            researcher_result_fake = BacktestResult(
                strategy_id=strategy["strategy_id"],
                strategy_name=strategy["strategy_name"],
                entry_description=strategy.get("entry_description", ""),
                total_trades=researcher_result_dict["total_trades"],
                wins=researcher_result_dict["wins"],
                losses=researcher_result_dict["losses"],
                timeouts=researcher_result_dict["timeouts"],
                win_rate=researcher_result_dict["win_rate"],
                profit_factor=researcher_result_dict["profit_factor"],
                max_drawdown=researcher_result_dict["max_drawdown"],
                sharpe=researcher_result_dict["sharpe"],
                total_pnl=0.0,
                status=agent_decision,
            )
            qa_result = self.dispatch_qa(strategy, researcher_result_fake)
            print(f"   QA WR={qa_result['qa_wr']:.2f}% (誤差 {qa_result['wr_error_pct']:.2f}%)")
            print(f"   QA PF={qa_result['qa_pf']:.2f} (誤差 {qa_result['pf_error_pct']:.2f}%)")
            print(f"   QA 結論: {qa_result['status']}")

            # ---- ③ team-ceo 裁決（確認 sub-agent 的 decision）----
            print("\n🏛️ [team-ceo] 裁決...")
            decision = self.dispatch_ceo(researcher_result_fake, qa_result)
            print(f"   裁決: {decision}（Agent 自行裁決: {agent_decision}）")

        # 持久化
        if decision == "KEEP":
            self.save_to_best_strategies(strategy, researcher_result_dict)
            print("   ✅ 寫入 best_strategies.json + factor_library.json")
        else:
            self.save_to_failed_strategies(strategy, researcher_result_dict)
            print("   ❌ 寫入 failed_strategies.json")

        self.write_research_log(strategy, researcher_result_dict, qa_result, decision)

        # 更新 ResearchState
        self.research_state.add_result({
            "round": rn,
            "strategy_id": strategy["strategy_id"],
            "decision": decision,
            "wr": researcher_result_dict["win_rate"],
            "pf": researcher_result_dict["profit_factor"],
            "dd": researcher_result_dict["max_drawdown"],
            "sharpe": researcher_result_dict["sharpe"],
            "qa_status": qa_result.get("status", "N/A"),
        })
        self.research_state.save()

        # 格式化回報
        report = self._format_report(
            rn, strategy, researcher_result_dict, qa_result, decision
        )
        if self.report_callback:
            await self.report_callback(report)

        print(f"\n{'='*60}")
        print(f"第 {rn} 輪完成 | 裁決={decision}")
        print(f"{'='*60}")

        return decision

    def _format_report(
        self,
        rn: int,
        strategy: dict,
        r: dict,
        qa: dict,
        decision: str,
    ) -> str:
        target = TARGET
        status_icon = {"KEEP": "✅", "DISCARD": "❌"}.get(decision, "?")

        return f"""
╔══════════════════════════════════════════════════════════╗
║        🔬 Auto Research Agent 第 {rn} 輪報告                        ║
╠══════════════════════════════════════════════════════════╣
║ 📝 策略: {strategy.get('strategy_name', 'N/A'):<40} ║
║ 🎯 條件: {(strategy.get('entry_description', '') or '')[:40]:<40} ║
╠══════════════════════════════════════════════════════════╣
║ 📊 交易統計                                              ║
║    總交易: {r.get('total_trades', 0):>4}  WIN: {r.get('wins', 0):>3}  LOSS: {r.get('losses', 0):>3}  TIMEOUT: {r.get('timeouts', 0):>3}      ║
╠══════════════════════════════════════════════════════════╣
║ 📈 指標                                                  ║
║    勝率:   {r.get('win_rate', 0):>6.1f}% {"✅" if r.get('win_rate', 0) >= target.win_rate else "❌"}  (目標≥{target.win_rate}%)           ║
║    盈虧比: {r.get('profit_factor', 0):>6.2f} {"✅" if r.get('profit_factor', 0) >= target.profit_factor else "❌"}  (目標≥{target.profit_factor})            ║
║    最大DD: {r.get('max_drawdown', 0):>6.1f}% {"✅" if r.get('max_drawdown', 100) <= target.max_drawdown else "❌"}  (目標≤{target.max_drawdown}%)           ║
║    Sharpe: {r.get('sharpe', 0):>6.2f} {"✅" if r.get('sharpe', 0) >= target.sharpe else "❌"}  (目標≥{target.sharpe})             ║
╠══════════════════════════════════════════════════════════╣
║ 🔍 QA 驗證                                              ║
║    WR 誤差: {qa.get('wr_error_pct', 0):.2f}% {"✅" if qa.get('wr_ok', False) else "❌"} | PF 誤差: {qa.get('pf_error_pct', 0):.2f}% {"✅" if qa.get('pf_ok', False) else "❌"}      ║
║    QA 結論: {qa.get('status', 'N/A'):<45} ║
╠══════════════════════════════════════════════════════════╣
║ 🏆 裁決: {status_icon} {decision:<50} ║
╚══════════════════════════════════════════════════════════╝"""

    # ---- Main Loop ----

    async def start(self):
        """啟動研究循環（可選 max_rounds 限制）"""
        print("🚀 Auto Research Agent 啟動")
        print(f"   Symbol: {self.symbol} | Interval: {self.interval}")
        print(f"   Target: WR≥{TARGET.win_rate}% PF≥{TARGET.profit_factor} DD≤{TARGET.max_drawdown}% Sharpe≥{TARGET.sharpe}")
        print(f"   Max rounds: {self.max_rounds or '無限'}")

        state = self.research_state
        print(f"\n📂 恢復 ResearchState: round={state.round_num}, recent={len(state.recent_results)}")

        consecutive_discards = 0

        while True:
            # 檢查停止條件
            if self.max_rounds and self.research_state.round_num >= self.max_rounds:
                print(f"\n✅ 已完成 {self.max_rounds} 輪，停止。")
                break

            try:
                decision = await self.run_cycle()

                if decision == "DISCARD":
                    consecutive_discards += 1
                else:
                    consecutive_discards = 0

                # 連續失敗超過 20 輪，休息一下
                if consecutive_discards >= 20:
                    print("\n😴 連續 20 輪失敗，休息 60 秒...")
                    await asyncio.sleep(60)
                    consecutive_discards = 0

                await asyncio.sleep(5)

            except KeyboardInterrupt:
                print("\n\n⏹️  被用戶中斷，保存狀態後退出。")
                self.research_state.save()
                break
            except Exception as e:
                print(f"\n❌ 輪次錯誤: {e}")
                import traceback
                traceback.print_exc()
                self.research_state.save()
                await asyncio.sleep(10)


# ========== CLI Entry Point ==========

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Auto Research Agent Orchestrator")
    parser.add_argument("--symbol", default="BTCUSDT", help="交易對")
    parser.add_argument("--interval", default="1d", help="K線週期")
    parser.add_argument("--max-rounds", type=int, default=None, help="最大輪數（預設無限）")
    parser.add_argument("--report-webhook", help="Discord webhook URL")
    args = parser.parse_args()

    async def run():
        if args.report_webhook:
            import aiohttp
            async def webhook_report(msg: str):
                async with aiohttp.ClientSession() as session:
                    payload = {"content": msg}
                    await session.post(args.report_webhook, json=payload)
            orchestrator = ResearchOrchestrator(
                symbol=args.symbol,
                interval=args.interval,
                max_rounds=args.max_rounds,
                report_callback=webhook_report,
            )
        else:
            orchestrator = ResearchOrchestrator(
                symbol=args.symbol,
                interval=args.interval,
                max_rounds=args.max_rounds,
            )
        await orchestrator.start()

    asyncio.run(run())


if __name__ == "__main__":
    main()
