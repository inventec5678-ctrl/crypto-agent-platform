"""
researcher_agent.py — Raw OHLCV 自主研究

研究 Agent 只能看到：
- open, high, low, close, volume

研究 Agent 自己：
- 構造任何指標
- 定義進場/出場邏輯
- 用 BacktestEngine 跑回測
- 把結果寫入 results.tsv

廢除了：
- MarketSnapshot 類（不再使用）
- ai_researcher.py 的 StrategyBacktester.backtest(entry_fn) 模式
- 所有 pre-computed 指標
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

from backtest.backtest_engine import BacktestEngine, BaseStrategy, PositionSide

DATA_PATH = PROJECT_ROOT / "data" / "btcusdt_1d.parquet"
MEMORY_DIR = PROJECT_ROOT / "autoresearch" / "memory"
RESULTS_TSV = MEMORY_DIR / "results.tsv"
FAILED_JSON = MEMORY_DIR / "failed_strategies.json"
RESEARCH_LOG = MEMORY_DIR / "research_log.md"

# 確保目錄存在
MEMORY_DIR.mkdir(parents=True, exist_ok=True)


# ========== 目標門檻 ==========
WIN_RATE_THRESH = 50.0
PROFIT_FACTOR_THRESH = 2.0
MAX_DD_THRESH = 30.0
SHARPE_THRESH = 1.5


def _init_results_tsv():
    """初始化 results.tsv"""
    if not RESULTS_TSV.exists():
        RESULTS_TSV.write_text("commit\twin_rate\tprofit_factor\tmax_dd\tsharpe\tstatus\tdescription\n")


def _record_result(commit: str, wr: float, pf: float, dd: float, sharpe: float, status: str, desc: str):
    """寫入 results.tsv"""
    _init_results_tsv()
    content = RESULTS_TSV.read_text()
    line = f"{commit}\t{wr:.4f}\t{pf:.4f}\t{dd:.2f}\t{sharpe:.4f}\t{status}\t{desc}\n"
    RESULTS_TSV.write_text(content + line)


def _load_failed():
    """載入失敗策略"""
    if FAILED_JSON.exists():
        try:
            return json.loads(FAILED_JSON.read_text())
        except:
            return []
    return []


def _save_failed(failed_list):
    """保存失敗策略"""
    FAILED_JSON.write_text(json.dumps(failed_list, indent=2, ensure_ascii=False))


def _build_desc(strategy: BaseStrategy) -> str:
    """從 strategy 物件建立精確描述"""
    p = strategy.__dict__
    name = strategy.__class__.__name__
    if name == "VolBreakoutStrategy":
        return (f"vol_breakout: vol_ratio>{p['vol_mult']}(vol_period={p['vol_period']}), "
                f"price_change>{p['price_filter']}, TP=5%, SL=2%")
    elif name == "MABreakoutStrategy":
        return (f"ma_breakout: ma_period={p['ma_period']}, price_thresh={p['price_thresh']}, "
                f"vol_thresh={p['vol_thresh']}, TP=5%, SL=2%")
    elif name == "RSIReversalStrategy":
        return (f"rsi_reversal: rsi_period={p['rsi_period']}, rsi_oversold={p['rsi_oversold']}, "
                f"vol_thresh={p['vol_thresh']}, TP=5%, SL=2%")
    elif name == "MomentumStrategy":
        return (f"momentum: lookback={p['lookback']}, momentum_thresh={p['momentum_thresh']}, "
                f"vol_ratio={p['vol_ratio']}, TP=5%, SL=2%")
    return f"{name}: {p}"


def _get_entry_logic(strategy: BaseStrategy) -> str:
    """從 strategy 類別提取進場條件描述"""
    name = strategy.__class__.__name__
    p = strategy.__dict__
    if name == "VolBreakoutStrategy":
        return (f"- vol_ratio = volumes[-1] / (sum(volumes[-{p['vol_period']}:]) / {p['vol_period']})\n"
                f"- vol_ratio > {p['vol_mult']}\n"
                f"- price_change = (closes[-1] - closes[-{p['vol_period']}]) / closes[-{p['vol_period']}]\n"
                f"- price_change > {p['price_filter']}")
    elif name == "MABreakoutStrategy":
        return (f"- ma = mean(closes[-{p['ma_period']}:])\n"
                f"- price_change = (closes[-1] - closes[-{p['ma_period']}]) / closes[-{p['ma_period']}]\n"
                f"- closes[-1] > ma\n"
                f"- price_change > {p['price_thresh']}\n"
                f"- vol_ratio > {p['vol_thresh']}")
    elif name == "RSIReversalStrategy":
        return (f"- rsi(period={p['rsi_period']}) < {p['rsi_oversold']}\n"
                f"- vol_ratio > {p['vol_thresh']}")
    elif name == "MomentumStrategy":
        return (f"- momentum = (closes[-1] - closes[-{p['lookback']}]) / closes[-{p['lookback']}]\n"
                f"- momentum > {p['momentum_thresh']}\n"
                f"- vol_ratio > {p['vol_ratio']}")
    return "See Python code"


def _get_exit_logic(strategy: BaseStrategy) -> str:
    """從 strategy 類別提取出场條件描述"""
    return "- SL = 2%\n- TP = 5%\n- 最大持倉 = 10 根 K 線"


def write_strategy_spec(strategy_name: str, strategy_class_code: str, params: dict,
                        entry_logic: str, exit_logic: str, backtest_result):
    """寫入策略規格書到 research_log.md"""
    spec = f"""
## Strategy Spec: {strategy_name}

**Generated at**: {datetime.now().isoformat()}

### 進場條件
{entry_logic}

### 出場條件
{exit_logic}

### 參數
{json.dumps(params, indent=2)}

### Python 代碼
```python
{strategy_class_code}
```

### Backtest 結果
- Win Rate: {backtest_result.Win_Rate:.1f}%
- Profit Factor: {backtest_result.Profit_Factor:.2f}
- Max Drawdown: {backtest_result.Max_Drawdown_Pct:.1f}%
- Sharpe Ratio: {backtest_result.Sharpe_Ratio:.2f}
- Total Trades: {backtest_result.Total_Trades}

---
"""
    RESEARCH_LOG.write_text(RESEARCH_LOG.read_text() + spec)


def _generate_commit_id(strategy_desc: str) -> str:
    """生成簡短的 commit ID"""
    import hashlib
    return hashlib.md5(strategy_desc.encode()).hexdigest()[:7]


# ========== 研究 Agent 示範策略工廠 ==========
# 這些是 Agent 自己"想到"的策略（實際上是隨機組合）
# 研究 Agent 應該自己想出有意義的策略，這裡只是示範

class StrategyFactory:
    """策略工廠 — 示範如何自己發明策略"""

    @staticmethod
    def random_strategy(round_num: int) -> BaseStrategy:
        """
        隨機生成一個策略（這裡是示範）
        實際上 Agent 應該自己想出有意義的策略邏輯
        """
        # 隨機選擇策略類型
        strategy_type = random.choice(["ma_breakout", "rsi_reversal", "vol_breakout", "momentum"])

        if strategy_type == "ma_breakout":
            return MABreakoutStrategy(
                ma_period=random.choice([10, 20, 30]),
                price_thresh=random.uniform(0.01, 0.05),
                vol_thresh=random.uniform(1.2, 2.0)
            )
        elif strategy_type == "rsi_reversal":
            return RSIReversalStrategy(
                rsi_period=random.choice([7, 14, 21]),
                rsi_oversold=random.choice([25, 30, 35]),
                vol_thresh=random.uniform(1.0, 1.5)
            )
        elif strategy_type == "vol_breakout":
            return VolBreakoutStrategy(
                vol_period=random.choice([10, 20]),
                vol_mult=random.uniform(1.5, 2.5),
                price_filter=random.uniform(0.01, 0.03)
            )
        else:  # momentum
            return MomentumStrategy(
                lookback=random.choice([5, 10, 20]),
                momentum_thresh=random.uniform(0.02, 0.05),
                vol_ratio=random.uniform(1.0, 1.5)
            )

    @staticmethod
    def get_strategy_description(strategy: BaseStrategy) -> str:
        """取得策略描述"""
        return strategy.__class__.__name__


# ========== 策略範例（Agent 自己發明的策略）============

class MABreakoutStrategy(BaseStrategy):
    """
    均線突破策略
    Agent 邏輯：價格站上均線 + 成交量放大 = 做多
    """
    def __init__(self, ma_period: int = 20, price_thresh: float = 0.03, vol_thresh: float = 1.5):
        self.ma_period = ma_period
        self.price_thresh = price_thresh  # 價格變化閾值
        self.vol_thresh = vol_thresh        # 成交量倍數閾值

    def generate_signal(self, market_data):
        df = market_data["BTCUSDT"]
        closes = df['close'].values
        volumes = df['volume'].values

        if len(closes) < self.ma_period + 2:
            return PositionSide.FLAT

        # Agent 自己計算指標
        ma = np.mean(closes[-self.ma_period:])
        vol_avg = np.mean(volumes[-self.ma_period:])
        price_change = (closes[-1] - closes[-self.ma_period]) / closes[-self.ma_period]
        vol_ratio = volumes[-1] / vol_avg if vol_avg > 0 else 1.0

        # Agent 自己定義進場邏輯
        if closes[-1] > ma and price_change > self.price_thresh and vol_ratio > self.vol_thresh:
            return PositionSide.LONG

        return PositionSide.FLAT


class RSIReversalStrategy(BaseStrategy):
    """
    RSI 反轉策略
    Agent 邏輯：RSI 超賣 + 成交量放大 = 做多
    """
    def __init__(self, rsi_period: int = 14, rsi_oversold: float = 30.0, vol_thresh: float = 1.2):
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.vol_thresh = vol_thresh

    def _compute_rsi(self, prices: np.ndarray, period: int) -> float:
        """自己計算 RSI"""
        if len(prices) < period + 1:
            return 50.0
        deltas = np.diff(prices[-period-1:])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def generate_signal(self, market_data):
        df = market_data["BTCUSDT"]
        closes = df['close'].values
        volumes = df['volume'].values

        if len(closes) < self.rsi_period + 2:
            return PositionSide.FLAT

        # Agent 自己計算 RSI
        rsi = self._compute_rsi(closes, self.rsi_period)

        # 自己計算成交量比率
        vol_avg = np.mean(volumes[-20:])
        vol_ratio = volumes[-1] / vol_avg if vol_avg > 0 else 1.0

        # Agent 自己定義進場：RSI 超賣 + 成交量放大
        if rsi < self.rsi_oversold and vol_ratio > self.vol_thresh:
            return PositionSide.LONG

        return PositionSide.FLAT


class VolBreakoutStrategy(BaseStrategy):
    """
    成交量突破策略
    Agent 邏輯：成交量突破均線 + 價格上漲 = 做多
    """
    def __init__(self, vol_period: int = 20, vol_mult: float = 2.0, price_filter: float = 0.02):
        self.vol_period = vol_period
        self.vol_mult = vol_mult
        self.price_filter = price_filter

    def generate_signal(self, market_data):
        df = market_data["BTCUSDT"]
        closes = df['close'].values
        volumes = df['volume'].values

        if len(closes) < self.vol_period + 2:
            return PositionSide.FLAT

        # Agent 自己計算
        vol_avg = np.mean(volumes[-self.vol_period:])
        vol_now = volumes[-1]

        # 價格變化
        price_change = (closes[-1] - closes[-self.vol_period]) / closes[-self.vol_period]

        # Agent 自己定義進場：成交量突破 + 價格上漲
        if vol_now > vol_avg * self.vol_mult and price_change > self.price_filter:
            return PositionSide.LONG

        return PositionSide.FLAT


class MomentumStrategy(BaseStrategy):
    """
    動量策略
    Agent 邏輯：短期動量強勁 + 成交量確認 = 做多
    """
    def __init__(self, lookback: int = 10, momentum_thresh: float = 0.03, vol_ratio: float = 1.2):
        self.lookback = lookback
        self.momentum_thresh = momentum_thresh
        self.vol_ratio = vol_ratio

    def generate_signal(self, market_data):
        df = market_data["BTCUSDT"]
        closes = df['close'].values
        volumes = df['volume'].values

        if len(closes) < self.lookback + 2:
            return PositionSide.FLAT

        # Agent 自己計算動量
        momentum = (closes[-1] - closes[-self.lookback]) / closes[-self.lookback]

        # 成交量確認
        vol_avg = np.mean(volumes[-self.lookback:])
        vol_ratio = volumes[-1] / vol_avg if vol_avg > 0 else 1.0

        # Agent 自己定義進場：動量強勁 + 成交量確認
        if momentum > self.momentum_thresh and vol_ratio > self.vol_ratio:
            return PositionSide.LONG

        return PositionSide.FLAT


# ========== 研究迴圈 ==========

def run_research_loop(rounds: int = 3):
    """
    研究循環（Karpathy AutoResearch 模式）

    每輪：
    1. 讀 raw OHLCV（只有 open, high, low, close, volume）
    2. Agent 自己發明策略
    3. 跑 BacktestEngine
    4. 記錄到 results.tsv
    """
    # 讀 raw OHLCV
    df = pd.read_parquet(str(DATA_PATH))

    # 只把 raw OHLCV 傳給 Agent（不經過任何預處理）
    market_data = {"BTCUSDT": df}

    print(f"\n{'='*60}")
    print(f"  研究 Agent 啟動（Raw OHLCV Only）")
    print(f"  數據: {len(df)} bars")
    print(f"  Columns: {list(df.columns)}")
    print(f"  目標: WR≥{WIN_RATE_THRESH}% | PF≥{PROFIT_FACTOR_THRESH} | DD≤{MAX_DD_THRESH}% | Sharpe≥{SHARPE_THRESH}")
    print(f"{'='*60}")

    failed = _load_failed()

    for round_num in range(1, rounds + 1):
        print(f"\n--- Round {round_num} ---")

        # ========== Agent 自己想到的策略 ==========
        # 這裡隨機選擇一個策略類型作為示範
        # 實際上 Agent 應該自己想出有意義的策略

        strategy = StrategyFactory.random_strategy(round_num)
        strategy_name = StrategyFactory.get_strategy_description(strategy)

        print(f"  策略: {strategy_name}")
        print(f"  參數: {strategy.__dict__}")

        # ========== 用 BacktestEngine 跑回測 ==========
        engine = BacktestEngine()

        # 封裝非同步
        import asyncio
        async def run_backtest():
            engine.load_dataframe("BTCUSDT", df)
            engine.set_strategy(strategy)
            engine.stop_loss = 0.02
            engine.take_profit = 0.05
            return await engine.run()

        result = asyncio.run(run_backtest())

        # 提取結果
        wr = result.Win_Rate
        pf = result.Profit_Factor
        dd = result.Max_Drawdown_Pct
        sharpe = result.Sharpe_Ratio
        trades = result.Total_Trades

        print(f"  WR={wr:.1f}% PF={pf:.2f} DD={dd:.1f}% Sharpe={sharpe:.2f} Trades={trades}")

        # ========== 評估決策 ==========
        all_pass = (wr >= WIN_RATE_THRESH and pf >= PROFIT_FACTOR_THRESH
                    and dd <= MAX_DD_THRESH and sharpe >= SHARPE_THRESH)
        status = "keep" if all_pass else "discard"

        print(f"  Decision: {status.upper()}")

        # 描述（精確參數）
        desc = _build_desc(strategy)

        # ========== 寫入 Strategy Spec 到 research_log.md ==========
        import inspect
        strategy_code = inspect.getsource(strategy.__class__)
        params = strategy.__dict__
        write_strategy_spec(
            strategy_name=strategy_name,
            strategy_class_code=strategy_code,
            params=params,
            entry_logic=_get_entry_logic(strategy),
            exit_logic=_get_exit_logic(strategy),
            backtest_result=result
        )

        # ========== 寫入 results.tsv ==========
        commit_id = _generate_commit_id(desc)
        _record_result(commit_id, wr, pf, dd, sharpe, status, desc)

        # ========== 如果失敗，記錄到 failed_strategies ==========
        if status == "discard":
            failed.append({
                "round": round_num,
                "strategy": strategy_name,
                "params": strategy.__dict__,
                "metrics": {"wr": wr, "pf": pf, "dd": dd, "sharpe": sharpe},
                "reason": f"wr={wr:.1f}% pf={pf:.2f} dd={dd:.1f} sharpe={sharpe:.2f}"
            })
            _save_failed(failed)

        # 關閉引擎
        engine.close()

    print(f"\n{'='*60}")
    print(f"✅ {rounds} 輪研究完成")
    print(f"📄 results.tsv: {RESULTS_TSV}")
    print(f"   總策略數: {rounds}")

    # 顯示 results.tsv 內容
    if RESULTS_TSV.exists():
        print(f"\n📊 results.tsv 內容:")
        print(RESULTS_TSV.read_text())


if __name__ == "__main__":
    run_research_loop(rounds=3)
