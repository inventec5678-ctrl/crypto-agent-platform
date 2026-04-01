"""
openclaw_wrapper - 封裝 OpenClaw sessions_spawn 呼叫

用於 ResearchOrchestrator 派發 team-researcher / team-qa / team-ceo sub-agent。
"""

import subprocess
import time
import json
import os
from pathlib import Path
from typing import Dict, Any

OPENCLAW_CLI = os.environ.get("OPENCLAW_CLI", "/opt/homebrew/bin/openclaw")


def spawn_agent(
    task: str,
    agent_id: str,
    output_path: str,
    timeout: int = 300,
) -> Dict[str, Any]:
    """
    派發 OpenClaw sub-agent，等待完成，讀取 output_path。

    Args:
        task: 給 agent 的 task 描述（完整 prompt）
        agent_id: agent ID (如 "team-researcher", "team-qa", "team-ceo")
        output_path: agent 完成後寫入的 JSON 檔路徑
        timeout: 秒

    Returns:
        dict: 讀取 output_path 的內容

    Raises:
        TimeoutError: agent 超時
        RuntimeError: openclaw 執行失敗
    """
    output_path = Path(output_path)
    pid = os.getpid()

    # 1. 寫入 task 到臨時檔（方便 Debug）
    task_file = output_path.parent / f"_task_{agent_id}_{pid}.txt"
    task_file.write_text(task)

    # 2. 用 openclaw sessions_spawn 派發
    cmd = [
        OPENCLAW_CLI,
        "sessions",
        "spawn",
        "--task", task,
        "--agent-id", agent_id,
        "--mode", "run",
        "--runtime", "subagent",
    ]

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    try:
        # 3. 等待 output_path 出現（每 5 秒輪詢一次）
        start = time.time()
        while time.time() - start < timeout:
            if output_path.exists():
                result = json.loads(output_path.read_text())
                output_path.unlink(missing_ok=True)
                task_file.unlink(missing_ok=True)
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
                return result
            time.sleep(5)

        # Timeout
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        task_file.unlink(missing_ok=True)
        raise TimeoutError(f"Agent {agent_id} timed out after {timeout}s")

    except Exception as e:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        task_file.unlink(missing_ok=True)
        raise RuntimeError(f"Agent {agent_id} failed: {e}")


# ---- 便捷封裝 ----

def spawn_researcher_agent(
    task: str,
    output_path: str,
    timeout: int = 300,
) -> Dict[str, Any]:
    """派發 team-researcher agent"""
    return spawn_agent(task=task, agent_id="team-researcher", output_path=output_path, timeout=timeout)


def spawn_qa_agent(
    task: str,
    output_path: str,
    timeout: int = 300,
) -> Dict[str, Any]:
    """派發 team-qa agent"""
    return spawn_agent(task=task, agent_id="team-qa", output_path=output_path, timeout=timeout)


def spawn_ceo_agent(
    task: str,
    output_path: str,
    timeout: int = 300,
) -> Dict[str, Any]:
    """派發 team-ceo agent"""
    return spawn_agent(task=task, agent_id="team-ceo", output_path=output_path, timeout=timeout)
