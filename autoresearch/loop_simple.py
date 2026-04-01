"""
loop_simple.py — 極簡 Orchestrator（Karpathy AutoResearch 模式）

每次循環：
1. 派 team-researcher（讓它自主研究，結果寫入 results.tsv）
2. 派 team-qa（獨立驗證）
3. 派 team-ceo（裁決寫庫）

team-researcher 會一直研究直到被告知停止（NEVER STOP 模式）。
"""

import os
import sys
import json
import time
import subprocess
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent.absolute()
DATA_PATH = PROJECT_ROOT / "data" / "btcusdt_1d.parquet"
AUTORESEARCH_DIR = PROJECT_ROOT / "autoresearch"
MEMORY_DIR = AUTORESEARCH_DIR / "memory"
PROMPTS_DIR = AUTORESEARCH_DIR / "prompts"
EXPERIMENT_STRATEGY = AUTORESEARCH_DIR / "experiment_strategy.py"
RESULTS_TSV = MEMORY_DIR / "results.tsv"
RESEARCH_LOG = MEMORY_DIR / "research_log.md"

PROMPT_RESEARCHER = (PROMPTS_DIR / "team-researcher_prompt.md").read_text()
PROMPT_QA = (PROMPTS_DIR / "team-qa_prompt.md").read_text()
PROMPT_CEO = (PROMPTS_DIR / "team-ceo_prompt.md").read_text()


def build_market_snapshot() -> dict:
    """從市場數據構造 snapshot"""
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


def spawn_researcher(rounds: int = 3):
    """
    直接用 Python subprocess 執行 researcher_agent.py（3 輪後自動停止）

    openclaw sessions spawn 不存在，改用：
    python3 -c "from autoresearch.researcher_agent import run_research_loop; run_research_loop(rounds)"
    """
    print("[loop_simple] 派發 team-researcher subprocess...")

    task_script = f"""
import sys
sys.path.insert(0, '{PROJECT_ROOT}')
from autoresearch.researcher_agent import run_research_loop
run_research_loop(rounds={rounds})
"""

    try:
        result = subprocess.run(
            [sys.executable, "-c", task_script],
            capture_output=False,
            text=True,
            cwd=str(PROJECT_ROOT),
            timeout=600,
        )
        return {"status": "done", "returncode": result}
    except subprocess.TimeoutExpired:
        return {"status": "timeout"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def spawn_openclaw_agent(agent_id: str, prompt: str, timeout: int = 300) -> dict:
    """
    派一個 OpenClaw subagent（單輪對話模式）

    注意：openclaw sessions spawn 不存在。
    openclaw agent --local -m "..." 只能跑單輪，無法維持 persistent session。
    因此 researcher 任務改用 spawn_researcher() 直接執行 Python logic。
    team-qa / team-ceo 可用 openclaw agent --local -m "..." 跑單輪。
    """
    if agent_id == "researcher":
        # researcher 用 Python subprocess（可控制輪數）
        return spawn_researcher(rounds=3)

    # team-qa / team-ceo 用 openclaw agent --local
    task_file = MEMORY_DIR / f"task_{agent_id}_{int(time.time())}.txt"
    task_file.write_text(prompt)

    try:
        result = subprocess.run(
            [
                "openclaw", "agent", "--local",
                "--agent", f"team-{agent_id}",
                "-m", prompt,
                "--timeout", str(timeout),
            ],
            capture_output=True,
            text=True,
            timeout=timeout + 30,
            cwd=str(PROJECT_ROOT),
        )
        return {"status": "done", "stdout": result.stdout, "stderr": result.stderr}
    except subprocess.TimeoutExpired:
        return {"status": "timeout"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def run_cycle():
    """執行一輪完整循環"""
    print(f"\n{'='*60}")
    print(f"  Auto Research Cycle — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")

    # Phase 1: team-researcher（Python subprocess，3 輪後停止）
    print("\n[Phase 1] 派 team-researcher...")
    snapshot = build_market_snapshot()
    researcher_prompt = PROMPT_RESEARCHER + f"\n\n## Market Snapshot\n{json.dumps(snapshot, indent=2)}"

    # 直接執行 researcher_agent.py（3 輪）
    result = spawn_researcher(rounds=3)
    print(f"team-researcher 完成: {result}")

    # Phase 2: team-qa（可選，用 openclaw agent --local 跑單輪 QA）
    # Phase 3: team-ceo（可選，用 openclaw agent --local 跑單輪裁決）

    print("\n✅ Auto Research Cycle 完成")
    print(f"📄 results.tsv 已更新 — 請查閱 {RESULTS_TSV}")


if __name__ == "__main__":
    print("Karpathy AutoResearch Mode — 啟動")
    print("team-researcher 將執行 3 輪研究後停止")
    print("中斷：Ctrl+C\n")

    run_cycle()
