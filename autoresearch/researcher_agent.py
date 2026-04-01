"""
researcher_agent.py — 完全自由研究模式

Researcher Agent 可以自由使用任何方法：
- RSI, MACD, Bollinger Bands, EMA, SMA, ATR, ADX, KD, CCI... 統統可以用
- 自己發明新指標也可以
- 只用價格型態也可以
- 想怎麼組合都行

目標：找出有效策略，Win Rate ≥ 50%、Profit Factor ≥ 2.0、Max Drawdown ≤ 30%、Sharpe ≥ 1.5
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import random
import json
import inspect

from backtest.backtest_engine import BacktestEngine, BaseStrategy, PositionSide

DATA_PATH = PROJECT_ROOT / "data" / "btcusdt_1d.parquet"
MEMORY_DIR = PROJECT_ROOT / "autoresearch" / "memory"
RESULTS_TSV = MEMORY_DIR / "results.tsv"
FAILED_JSON = MEMORY_DIR / "failed_strategies.json"
RESEARCH_LOG = MEMORY_DIR / "research_log.md"

MEMORY_DIR.mkdir(parents=True, exist_ok=True)

# ========== 目標門檻 ==========
WIN_RATE_THRESH = 50.0
PROFIT_FACTOR_THRESH = 2.0
MAX_DD_THRESH = 30.0
SHARPE_THRESH = 1.5


# ========== 工具函數 ==========

def _init_results_tsv():
    if not RESULTS_TSV.exists():
        RESULTS_TSV.write_text("commit\twin_rate\tprofit_factor\tmax_dd\tsharpe\tstatus\tdescription\n")


def _record_result(commit: str, wr: float, pf: float, dd: float, sharpe: float, status: str, desc: str):
    _init_results_tsv()
    content = RESULTS_TSV.read_text()
    line = f"{commit}\t{wr:.4f}\t{pf:.4f}\t{dd:.2f}\t{sharpe:.4f}\t{status}\t{desc}\n"
    RESULTS_TSV.write_text(content + line)


def _load_failed():
    if FAILED_JSON.exists():
        try:
            return json.loads(FAILED_JSON.read_text())
        except:
            return []
    return []


def _save_failed(failed_list):
    FAILED_JSON.write_text(json.dumps(failed_list, indent=2, ensure_ascii=False))


def _generate_commit_id(desc: str) -> str:
    import hashlib
    return hashlib.md5(desc.encode()).hexdigest()[:7]


# ========== 自由策略生成器 ==========
# Researcher 可以自由選擇任何方法：已知指標、自己發明、或兩者混用

def agent_invent_strategy(df: pd.DataFrame, market_data: dict, round_num: int) -> tuple:
    """
    Researcher Agent 完全自由發明 / 選擇策略。

    輸入：
    - df: OHLCV DataFrame（open, high, low, close, volume）
    - market_data: {"BTCUSDT": df}
    - round_num: 第幾輪

    輸出：
    - strategy: 繼承 BaseStrategy 的類別實例
    - strategy_name: str（策略名）
    - entry_logic: str（進場條件文字描述）
    - params: dict（策略參數）

    研究員完全自由：想用 RSI/MACD/Bollinger 就用，想自己發明就發明。
    """

    closes = df['close'].values
    volumes = df['volume'].values
    highs = df['high'].values
    lows = df['low'].values
    opens = df['open'].values

    # 提供基本統計資訊（研究員可以自由使用或忽略）
    recent_closes = closes[-60:]
    recent_volumes = volumes[-60:]

    stats = {
        "n_bars": len(df),
        "price_range_60d": float(recent_closes.max() - recent_closes.min()),
        "price_range_60d_pct": float((recent_closes.max() - recent_closes.min()) / recent_closes.mean()),
        "vol_avg_60d": float(recent_volumes.mean()),
        "vol_std_60d": float(recent_volumes.std()),
        "price_avg_60d": float(recent_closes.mean()),
        "price_std_60d": float(recent_closes.std()),
        "last_close": float(closes[-1]),
        "last_volume": float(volumes[-1]),
        "close_5bar_chg": float((closes[-1] - closes[-5]) / closes[-5] if len(closes) >= 5 else 0),
        "vol_last_vs_5avg": float(volumes[-1] / recent_volumes[-5:].mean()) if recent_volumes[-5:].mean() > 0 else 1.0,
        "candle_directions": [int(closes[-i] > closes[-i-1]) for i in range(1, 4)],
    }

    # 隨機選擇一種策略方向（8種可選，研究員可以完全自由替換）
    strategy_type = random.choice([
        "rsi_reversal",
        "macd_momentum",
        "bollinger_breakout",
        "volume_surge",
        "price_deviation",
        "sequential_direction",
        "momentum_burst",
        "custom_hybrid",
    ])

    lookback = random.choice([3, 5, 7, 9, 12, 14, 20, 25])
    threshold_multiplier = round(random.uniform(1.1, 3.0), 1)

    name_prefix = random.choice([
        "Alpha", "Nova", "Flux", "Pulse", "Wave", "Arc", "Edge",
        "Spark", "Drift", "Phase", "Echo", "Ridge", "Helix", "Prism",
        "RSI", "MACD", "BB", "ATR", "ADX"  # 現在允許用指標名當前綴
    ])
    name_suffix = random.choice([
        "Break", "Flow", "Surge", "Fold", "Shift", "Trap", "Hook",
        "Swipe", "Crush", "Crack", "Leap", "Drop", "Swing", "Tilt",
        "Cross", "Zone", "Edge", "Pulse"
    ])
    strategy_name = f"{name_prefix}{name_suffix}{lookback}"

    # ============================================================
    # 策略邏輯（研究員完全自由選擇 / 發明）
    # ============================================================

    # ---- 先檢查是否能用 talib（有則用，沒有則跳過）----
    _has_talib = False
    try:
        import talib
        _has_talib = True
    except ImportError:
        pass

    if strategy_type == "rsi_reversal" and _has_talib:
        """使用 RSI：低於 threshold 超賣 → 進場"""
        rsi_period = random.choice([7, 14, 21])
        rsi_threshold = random.choice([25, 30, 35])

        class AgentStrategy(BaseStrategy):
            def __init__(self):
                self.rsi_period = rsi_period
                self.rsi_threshold = rsi_threshold
                self.p = {"rsi_period": rsi_period, "rsi_threshold": rsi_threshold}

            def generate_signal(self, market_data):
                df = market_data["BTCUSDT"]
                rsi = talib.RSI(df['close'].values, timeperiod=self.rsi_period)
                if rsi[-1] < self.rsi_threshold:
                    return PositionSide.LONG
                return PositionSide.FLAT

        entry_logic = (
            f"- rsi = RSI(closes, period={rsi_period})\n"
            f"- rsi[-1] < {rsi_threshold}（超賣）\n"
            f"- → 進場"
        )
        strategy = AgentStrategy()

    elif strategy_type == "macd_momentum" and _has_talib:
        """使用 MACD：MACD 線上穿訊號線 → 進場"""
        fast, slow, signal = random.choice([
            (12, 26, 9), (8, 17, 9), (10, 21, 9), (6, 13, 4)
        ])

        class AgentStrategy(BaseStrategy):
            def __init__(self):
                self.fast, self.slow, self.signal = fast, slow, signal
                self.p = {"fast": fast, "slow": slow, "signal": signal}

            def generate_signal(self, market_data):
                df = market_data["BTCUSDT"]
                macd, sig, hist = talib.MACD(df['close'].values,
                                               fastperiod=self.fast,
                                               slowperiod=self.slow,
                                               signalperiod=self.signal)
                if macd[-1] > sig[-1] and macd[-2] <= sig[-2]:
                    return PositionSide.LONG
                return PositionSide.FLAT

        entry_logic = (
            f"- MACD(fast={fast}, slow={slow}, signal={signal})\n"
            f"- MACD line 上穿 signal line（金叉）\n"
            f"- → 進場"
        )
        strategy = AgentStrategy()

    elif strategy_type == "bollinger_breakout" and _has_talib:
        """使用 Bollinger Bands：價格突破上軌 → 進場"""
        bb_period = random.choice([10, 20, 25])
        bb_nbdevup = random.choice([1.5, 2.0, 2.5])

        class AgentStrategy(BaseStrategy):
            def __init__(self):
                self.bb_period = bb_period
                self.bb_nbdevup = bb_nbdevup
                self.p = {"bb_period": bb_period, "bb_nbdevup": bb_nbdevup}

            def generate_signal(self, market_data):
                df = market_data["BTCUSDT"]
                upper, mid, lower = talib.BBANDS(df['close'].values,
                                                  timeperiod=self.bb_period,
                                                  nbdevup=self.bb_nbdevup,
                                                  nbdevdn=self.bb_nbdevup)
                if df['close'].values[-1] > upper[-1]:
                    return PositionSide.LONG
                return PositionSide.FLAT

        entry_logic = (
            f"- BBANDS(period={bb_period}, nbdev={bb_nbdevup})\n"
            f"- close > upper_band（突破上軌）\n"
            f"- → 進場"
        )
        strategy = AgentStrategy()

    elif strategy_type == "sequential_direction":
        """
        自定義邏輯：連續N根K線同方向 + 成交量確認
        """
        n = lookback
        t = threshold_multiplier

        class AgentStrategy(BaseStrategy):
            def __init__(self):
                self.n = n
                self.t = t
                self.p = {"lookback": n, "threshold": t}

            def generate_signal(self, market_data):
                df = market_data["BTCUSDT"]
                c = df['close'].values
                v = df['volume'].values

                if len(c) < self.n + 2:
                    return PositionSide.FLAT

                directions = [1 if c[-i] > c[-i-1] else -1 for i in range(1, self.n+1)]
                same_direction = len(set(directions)) == 1
                net_change = (c[-1] - c[-self.n]) / c[-self.n]
                vol_avg = v[-self.n:].mean()
                vol_confirm = v[-1] > vol_avg * self.t

                if same_direction and net_change > 0 and vol_confirm:
                    return PositionSide.LONG

                return PositionSide.FLAT

        entry_logic = (
            f"- 連續{lookback}根K線同方向\n"
            f"- 淨變化 > 0\n"
            f"- 成交量 > 均量 × {threshold_multiplier}\n"
            f"- → 進場"
        )
        strategy = AgentStrategy()

    elif strategy_type == "volume_surge":
        """
        自定義邏輯：成交量突放 + 價格上漲
        """
        n = lookback
        t = threshold_multiplier

        class AgentStrategy(BaseStrategy):
            def __init__(self):
                self.n = n
                self.t = t
                self.p = {"vol_lookback": n, "vol_mult": t}

            def generate_signal(self, market_data):
                df = market_data["BTCUSDT"]
                c = df['close'].values
                v = df['volume'].values

                if len(c) < self.n + 1:
                    return PositionSide.FLAT

                vol_avg = v[-self.n:].mean()
                vol_ratio = v[-1] / vol_avg if vol_avg > 0 else 1.0
                price_up = c[-1] > c[-2]

                if vol_ratio > self.t and price_up:
                    return PositionSide.LONG

                return PositionSide.FLAT

        entry_logic = (
            f"- vol_ratio = 成交量 / 均量\n"
            f"- vol_ratio > {threshold_multiplier} + 價格上漲\n"
            f"- → 進場"
        )
        strategy = AgentStrategy()

    elif strategy_type == "price_deviation":
        """
        自定義邏輯：價格低於均值的某個偏離比例
        """
        n = lookback
        dev_thresh = round(random.uniform(0.015, 0.06), 3)

        class AgentStrategy(BaseStrategy):
            def __init__(self):
                self.n = n
                self.dev_thresh = dev_thresh
                self.p = {"dev_lookback": n, "deviation_pct": dev_thresh}

            def generate_signal(self, market_data):
                df = market_data["BTCUSDT"]
                c = df['close'].values

                if len(c) < self.n + 1:
                    return PositionSide.FLAT

                avg = c[-self.n:].mean()
                deviation = (c[-1] - avg) / avg

                if deviation < -self.dev_thresh:
                    return PositionSide.LONG

                return PositionSide.FLAT

        entry_logic = (
            f"- avg = mean(closes[-{lookback}:])\n"
            f"- deviation < -{dev_thresh}（低於均值 {dev_thresh*100}%）\n"
            f"- → 進場"
        )
        strategy = AgentStrategy()

    elif strategy_type == "momentum_burst":
        """
        自定義邏輯：動量突然爆發
        """
        n = lookback
        mom_thresh = round(random.uniform(0.01, 0.05), 2)

        class AgentStrategy(BaseStrategy):
            def __init__(self):
                self.n = n
                self.mom_thresh = mom_thresh
                self.p = {"momentum_lookback": n, "momentum_thresh": mom_thresh}

            def generate_signal(self, market_data):
                df = market_data["BTCUSDT"]
                c = df['close'].values
                v = df['volume'].values

                if len(c) < self.n + 1:
                    return PositionSide.FLAT

                momentum = (c[-1] - c[-self.n]) / c[-self.n]
                vol_avg = v[-self.n:].mean()
                vol_ratio = v[-1] / vol_avg if vol_avg > 0 else 1.0

                if momentum > self.mom_thresh and vol_ratio > 1.1:
                    return PositionSide.LONG

                return PositionSide.FLAT

        entry_logic = (
            f"- momentum = (close - close[-{lookback}]) / close[-{lookback}]\n"
            f"- momentum > {mom_thresh} + vol_ratio > 1.1\n"
            f"- → 進場"
        )
        strategy = AgentStrategy()

    else:  # custom_hybrid
        """
        完全自定義混合策略
        """
        n = lookback
        t = threshold_multiplier

        class AgentStrategy(BaseStrategy):
            def __init__(self):
                self.n = n
                self.t = t
                self.p = {"lookback": n, "threshold": t}

            def generate_signal(self, market_data):
                df = market_data["BTCUSDT"]
                c = df['close'].values
                v = df['volume'].values
                h = df['high'].values
                l = df['low'].values

                if len(c) < self.n + 2:
                    return PositionSide.FLAT

                # 混合邏輯：價格動能 + 成交量 + 波動率
                momentum = (c[-1] - c[-self.n]) / c[-self.n]
                vol_avg = v[-self.n:].mean()
                vol_ratio = v[-1] / vol_avg if vol_avg > 0 else 1.0
                vol_std = v[-self.n:].std()
                vol_z = (v[-1] - vol_avg) / (vol_std + 1e-9)

                # 自定義進場條件
                if momentum > 0.02 and vol_ratio > self.t and vol_z > 1.0:
                    return PositionSide.LONG

                return PositionSide.FLAT

        entry_logic = (
            f"- momentum > 0.02 + vol_ratio > {threshold_multiplier} + vol_z > 1.0\n"
            f"- → 進場（混合策略）"
        )
        strategy = AgentStrategy()

    params = strategy.p
    return strategy, strategy_name, entry_logic, params


# ========== 研究迴圈 ==========

def run_research_loop(rounds: int = 3):
    """
    完全自由研究循環

    每輪：
    1. 讀取 OHLCV 數據
    2. Researcher 自己發明 / 選擇策略（完全自由）
    3. 跑 BacktestEngine
    4. 記錄到 results.tsv + research_log.md
    """

    df = pd.read_parquet(str(DATA_PATH))
    market_data = {"BTCUSDT": df}

    print(f"\n{'='*60}")
    print(f"  研究 Agent 啟動（完全自由模式）")
    print(f"  數據: {len(df)} bars")
    print(f"  Columns: {list(df.columns)}")
    print(f"  目標: WR≥{WIN_RATE_THRESH}% | PF≥{PROFIT_FACTOR_THRESH} | DD≤{MAX_DD_THRESH}% | Sharpe≥{SHARPE_THRESH}")
    print(f"{'='*60}")

    failed = _load_failed()
    _init_results_tsv()

    for round_num in range(1, rounds + 1):
        print(f"\n--- Round {round_num} ---")

        strategy, strategy_name, entry_logic, params = agent_invent_strategy(df, market_data, round_num)

        print(f"  策略名: {strategy_name}")
        print(f"  參數: {params}")
        print(f"  進場邏輯:\n    " + "\n    ".join(entry_logic.split("\n")))

        # ========== 跑回測 ==========
        engine = BacktestEngine()

        import asyncio
        async def run_backtest():
            engine.load_dataframe("BTCUSDT", df)
            engine.set_strategy(strategy)
            engine.stop_loss = 0.02
            engine.take_profit = 0.05
            return await engine.run()

        result = asyncio.run(run_backtest())

        wr = result.Win_Rate
        pf = result.Profit_Factor
        dd = result.Max_Drawdown_Pct
        sharpe = result.Sharpe_Ratio
        trades = result.Total_Trades

        print(f"  WR={wr:.1f}% PF={pf:.2f} DD={dd:.1f}% Sharpe={sharpe:.2f} Trades={trades}")

        # ========== 評估 ==========
        all_pass = (wr >= WIN_RATE_THRESH and pf >= PROFIT_FACTOR_THRESH
                    and dd <= MAX_DD_THRESH and sharpe >= SHARPE_THRESH)
        status = "keep" if all_pass else "discard"

        print(f"  Decision: {status.upper()}")

        desc = f"{strategy_name}: {params}"

        # ========== 寫入 research_log.md ==========
        strategy_code = inspect.getsource(strategy.__class__)
        exit_logic = "- SL = 2%\n- TP = 5%\n- 最大持倉 = 10 根 K 線"

        spec = f"""
## Strategy Spec: {strategy_name}

**Generated at**: {datetime.now().isoformat()}
**Round**: {round_num}

### 進場條件
{entry_logic}

### 出場條件
{exit_logic}

### 參數
{json.dumps(params, indent=2)}

### Python 代碼
```python
{strategy_code}
```

### Backtest 結果
- Win Rate: {wr:.1f}%
- Profit Factor: {pf:.2f}
- Max Drawdown: {dd:.1f}%
- Sharpe Ratio: {sharpe:.2f}
- Total Trades: {trades}

---
"""
        RESEARCH_LOG.write_text(RESEARCH_LOG.read_text() + spec)

        # ========== 寫入 results.tsv ==========
        commit_id = _generate_commit_id(desc)
        _record_result(commit_id, wr, pf, dd, sharpe, status, desc)

        if status == "discard":
            failed.append({
                "round": round_num,
                "strategy": strategy_name,
                "params": params,
                "metrics": {"wr": wr, "pf": pf, "dd": dd, "sharpe": sharpe},
                "reason": f"wr={wr:.1f}% pf={pf:.2f} dd={dd:.1f} sharpe={sharpe:.2f}"
            })
            _save_failed(failed)

        engine.close()

    print(f"\n{'='*60}")
    print(f"✅ {rounds} 輪研究完成")
    print(f"📄 results.tsv: {RESULTS_TSV}")
    print(f"📄 research_log.md: {RESEARCH_LOG}")
    print(f"   總策略數: {rounds}")

    if RESULTS_TSV.exists():
        print(f"\n📊 results.tsv 內容:")
        print(RESULTS_TSV.read_text())


if __name__ == "__main__":
    run_research_loop(rounds=3)
