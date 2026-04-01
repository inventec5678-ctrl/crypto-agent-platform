"""
researcher_agent.py — Karpathy AutoResearch 模式的自主研究 Agent

每次循環：
1. 分析市場數據
2. 想一個新策略
3. 寫入 experiment_strategy.py
4. 跑 backtest_engine.py
5. 根據結果記錄到 results.tsv
6. 繼續下一輪
"""

import sys
import json
import random
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.absolute()
DATA_PATH = PROJECT_ROOT / "data" / "btcusdt_1d.parquet"
AUTORESEARCH_DIR = PROJECT_ROOT / "autoresearch"
MEMORY_DIR = AUTORESEARCH_DIR / "memory"
EXPERIMENT_STRATEGY = AUTORESEARCH_DIR / "experiment_strategy.py"
RESULTS_TSV = MEMORY_DIR / "results.tsv"
FAILED_JSON = MEMORY_DIR / "failed_strategies.json"
RESEARCH_LOG = MEMORY_DIR / "research_log.md"


def load_market_snapshot():
    """載入市場快照"""
    import pandas as pd
    from autoresearch.ai_researcher import MarketData

    df = pd.read_parquet(str(DATA_PATH))
    md = MarketData(df)
    snap = md.get_snapshot(len(df) - 1)

    return {
        "regime": str(snap.regime),
        "rsi_14": float(snap.rsi),
        "atr": float(snap.atr),
        "ma200_slope": float(snap.ma200_slope),
        "trend_7d": float(snap.trend_7d),
        "vol_ratio": float(snap.vol_ratio),
        "close": float(snap.close),
        "ma10": float(snap.ma10),
        "ma20": float(snap.ma20),
        "ma50": float(snap.ma50),
        "ma200": float(snap.ma200),
    }


def load_failed_strategies():
    if FAILED_JSON.exists():
        return json.loads(FAILED_JSON.read_text())
    return []


def save_failed_strategy(strategy):
    """記錄失敗策略"""
    failed = load_failed_strategies()
    failed.append(strategy)
    FAILED_JSON.write_text(json.dumps(failed, indent=2, ensure_ascii=False))


def write_strategy_to_file(code: str):
    """寫入 experiment_strategy.py"""
    EXPERIMENT_STRATEGY.write_text(code)


def run_backtest():
    """跑 backtest，返回結果"""
    sys.path.insert(0, str(PROJECT_ROOT))
    from autoresearch.ai_researcher import StrategyBacktester, MarketData
    import pandas as pd

    # 讀取 experiment_strategy.py 的內容
    code = EXPERIMENT_STRATEGY.read_text()

    # 解析策略參數
    rsi_match = re.search(r'rsi\s*<\s*(\d+)', code)
    vol_match = re.search(r'vol_ratio\s*>\s*([\d.]+)', code)
    trend_match = re.search(r'trend_7d\s*>\s*([-\d.]+)', code)
    sl_match = re.search(r'STOP_LOSS\s*=\s*([\d.]+)', code)
    tp_match = re.search(r'TAKE_PROFIT\s*=\s*([\d.]+)', code)
    hold_match = re.search(r'MAX_HOLDING_BARS\s*=\s*(\d+)', code)

    # 構造 entry_conditions 字串
    conditions = []
    if rsi_match:
        conditions.append(f"snap.rsi < {rsi_match.group(1)}")
    if vol_match:
        conditions.append(f"snap.vol_ratio > {vol_match.group(1)}")
    if trend_match:
        conditions.append(f"snap.trend_7d > {trend_match.group(1)}")

    entry_conditions = " and ".join(conditions) if conditions else "snap.rsi < 50"

    entry_fn = eval(f"lambda snap: {entry_conditions}")

    df = pd.read_parquet(str(DATA_PATH))
    md = MarketData(df)
    backtester = StrategyBacktester(md)

    result = backtester.backtest(
        strategy_id=f"exp_{len(load_failed_strategies())+1}",
        strategy_name="Experiment",
        entry_description=entry_conditions,
        entry_fn=entry_fn,
        stop_loss_pct=float(sl_match.group(1)) if sl_match else 0.02,
        take_profit_pct=float(tp_match.group(1)) if tp_match else 0.05,
        max_holding_bars=int(hold_match.group(1)) if hold_match else 10,
    )

    return {
        "win_rate": result.win_rate,
        "profit_factor": result.profit_factor,
        "max_drawdown": result.max_drawdown,
        "sharpe": result.sharpe,
        "total_trades": result.total_trades,
        "wins": result.wins,
        "losses": result.losses,
    }


def record_result(strategy_id: str, wr: float, pf: float, dd: float, sharpe: float, status: str, description: str):
    """寫入 results.tsv"""
    import hashlib
    commit = hashlib.md5(strategy_id.encode()).hexdigest()[:7]

    line = f"{commit}\t{wr:.6f}\t{0.0:.1f}\t{status}\t{description}\n"

    if RESULTS_TSV.exists():
        content = RESULTS_TSV.read_text()
        if not content.strip():
            content = "commit\tval_bpb\tmemory_gb\tstatus\tdescription\n"
    else:
        content = "commit\tval_bpb\tmemory_gb\tstatus\tdescription\n"

    RESULTS_TSV.write_text(content + line)


def run_research_loop(rounds: int = 3):
    """執行 research 循環"""
    snapshot = load_market_snapshot()
    failed = load_failed_strategies()

    print(f"\n{'='*60}")
    print(f"  team-researcher 啟動")
    print(f"  Market: {snapshot['regime']} | RSI={snapshot['rsi_14']:.1f} | Close={snapshot['close']}")
    print(f"  測試輪數: {rounds}")
    print(f"{'='*60}")

    # 失敗過的關鍵詞組合（用於避開）
    avoid_keywords = set()
    for f in failed:
        avoid_keywords.update(f.get("avoid_patterns", []))

    for round_num in range(1, rounds + 1):
        print(f"\n--- Round {round_num}/{rounds} ---")

        # === 根據 snapshot 和 avoid_keywords 想一個新策略 ===
        # regime-aware 隨機策略生成
        regime = snapshot["regime"]

        if regime in ("BULL", "STRONG_BULL"):
            direction = "LONG"
            rsi_thresh = random.randint(30, 45)
            vol_thresh = random.uniform(1.2, 1.8)
            trend_thresh = random.uniform(0, 3)
        elif regime in ("BEAR", "STRONG_BEAR"):
            direction = random.choice(["LONG", "SHORT"])
            rsi_thresh = random.randint(55, 70) if direction == "SHORT" else random.randint(30, 45)
            vol_thresh = random.uniform(1.2, 1.8)
            trend_thresh = random.uniform(-3, 0) if direction == "SHORT" else random.uniform(0, 3)
        else:
            direction = "LONG"
            rsi_thresh = random.randint(30, 70)
            vol_thresh = random.uniform(0.8, 1.5)
            trend_thresh = 0

        tp = random.uniform(0.06, 0.12)
        sl = random.uniform(0.015, 0.03)
        max_hold = random.randint(5, 20)

        entry_code = f"snap.rsi < {rsi_thresh} and snap.vol_ratio > {vol_thresh:.1f} and snap.trend_7d > {trend_thresh:.1f}"

        strategy_code = f'''
"""
experiment_strategy.py — Round {round_num}
Market: {snapshot['regime']} | RSI={snapshot['rsi_14']:.1f}
"""
from dataclasses import dataclass

def get_entry(snap) -> bool:
    return ({entry_code})

STOP_LOSS = {sl:.4f}
TAKE_PROFIT = {tp:.4f}
MAX_HOLDING_BARS = {max_hold}
DIRECTION = "{direction}"
'''

        print(f"  策略: {entry_code}")
        print(f"  TP={tp:.2%} SL={sl:.2%} HOLD={max_hold}")

        # 寫入策略文件
        write_strategy_to_file(strategy_code)

        # 跑 backtest
        print(f"  執行回測...")
        result = run_backtest()

        wr = result["win_rate"]
        pf = result["profit_factor"]
        dd = result["max_drawdown"]
        sharpe = result["sharpe"]

        print(f"  結果: WR={wr:.1f}% PF={pf:.2f} DD={dd:.1f}% Sharpe={sharpe:.2f}")

        # 評估
        all_pass = (wr >= 50 and pf >= 2.0 and dd <= 30 and sharpe >= 1.5)
        status = "keep" if all_pass else "discard"

        print(f"  裁決: {status.upper()}")

        # 記錄
        desc = f"round{round_num}: {entry_code[:50]}... TP={tp:.2%} SL={sl:.2%}"
        record_result(f"round_{round_num}", wr, pf, dd, sharpe, status, desc)

        if status == "discard":
            save_failed_strategy({
                "round": round_num,
                "entry_conditions": entry_code,
                "avoid_patterns": [entry_code.split(" and ")[0].strip()],
                "wr": wr,
                "pf": pf,
                "reason": f"wr={wr:.1f}% < 50%" if wr < 50 else f"pf={pf:.2f} < 2.0"
            })

    print(f"\n✅ {rounds} 輪研究完成")
    print(f"📄 results.tsv 已更新")


if __name__ == "__main__":
    run_research_loop(rounds=3)
