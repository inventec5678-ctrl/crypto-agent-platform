"""
ResearchState — Auto Research 持久化狀態

用於 ResearchOrchestrator 的跨輪次狀態管理。
"""

import json
import os
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional


MEMORY_DIR = Path(__file__).parent / "memory"
STATE_FILE = MEMORY_DIR / "research_state.json"


@dataclass
class ResearchState:
    round_num: int = 0
    current_focus: Optional[str] = None
    recent_results: List[dict] = field(default_factory=list)
    dimension_scores: Dict[str, float] = field(default_factory=dict)
    # 追蹤每輪嘗試過的維度（避免重複調整同一維度）
    dimension_history: List[dict] = field(default_factory=list)
    # 追蹤已嘗試過的完整策略指紋（避免重複生成相同策略）
    seen_fingerprints: List[str] = field(default_factory=list)
    # 追蹤最近使用過的模板名稱（避免重複選擇同一模板）
    recent_templates: List[str] = field(default_factory=list)
    last_result: Optional[dict] = None
    started_at: Optional[str] = None
    last_run_at: Optional[str] = None

    def save(self):
        """寫入磁盤"""
        MEMORY_DIR.mkdir(parents=True, exist_ok=True)
        data = asdict(self)
        STATE_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False))

    @classmethod
    def load(cls) -> "ResearchState":
        """從磁盤載入，無則回傳乾淨狀態"""
        if STATE_FILE.exists():
            try:
                data = json.loads(STATE_FILE.read_text())
                return cls(**data)
            except Exception:
                pass
        return cls(started_at=datetime.now().isoformat())

    def add_result(self, result: dict):
        """追加結果到 recent_results（保留最近 20 輪）"""
        self.recent_results.append(result)
        if len(self.recent_results) > 20:
            self.recent_results = self.recent_results[-20:]
        self.last_result = result
        self.last_run_at = datetime.now().isoformat()
        
        # 記錄維度歷史（用於維度追蹤）
        if "dimension_varied" in result:
            self.dimension_history.append({
                "round": result.get("round"),
                "dimension": result.get("dimension_varied"),
                "wr": result.get("wr", 0),
                "pf": result.get("pf", 0),
                "dd": result.get("dd", 0),
                "strategy_id": result.get("strategy_id"),
            })
            if len(self.dimension_history) > 10:
                self.dimension_history = self.dimension_history[-10:]

    def increment_round(self) -> int:
        self.round_num += 1
        return self.round_num

    def add_fingerprint(self, fp: str):
        """記錄已嘗試過的策略指紋"""
        if fp not in self.seen_fingerprints:
            self.seen_fingerprints.append(fp)
            if len(self.seen_fingerprints) > 100:
                self.seen_fingerprints = self.seen_fingerprints[-100:]

    def add_template(self, template_name: str):
        """記錄最近使用過的模板"""
        if template_name not in self.recent_templates:
            self.recent_templates.append(template_name)
            if len(self.recent_templates) > 10:
                self.recent_templates = self.recent_templates[-10:]

    def get_fingerprints(self) -> List[str]:
        return self.seen_fingerprints.copy()

    def get_recent_templates(self) -> List[str]:
        return self.recent_templates.copy()
