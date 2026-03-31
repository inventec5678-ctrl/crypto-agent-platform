"""
Autoresearch 持久化層

將實驗結果儲存到 JSON 文件
"""

import json
import os
import shutil
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime
from filelock import FileLock
import copy

from .models import ExperimentResult, BestParams, AutoresearchState


class Persistence:
    """
    實驗結果持久化
    
    目錄結構:
    autoresearch/
      state.json              # 全局狀態
      experiments/            # 所有實驗記錄
        exp_001.json
        exp_002.json
        ...
      best/                    # 每個策略的最佳參數
        MACrossoverStrategy.json
        RSIReversalStrategy.json
      archive/                  # 歸檔（可選）
    
    文件鎖確保並發安全
    """

    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)
        self.experiments_dir = self.base_dir / "experiments"
        self.best_dir = self.base_dir / "best"
        self.state_file = self.base_dir / "state.json"
        self.lock_file = self.base_dir / ".lock"
        
        self._ensure_dirs()
    
    def _ensure_dirs(self):
        """確保目錄存在"""
        self.experiments_dir.mkdir(parents=True, exist_ok=True)
        self.best_dir.mkdir(parents=True, exist_ok=True)
    
    def _read_state(self) -> AutoresearchState:
        """讀取全局狀態"""
        if not self.state_file.exists():
            return AutoresearchState()
        
        with open(self.state_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return AutoresearchState.from_dict(data)
    
    def _write_state(self, state: AutoresearchState):
        """寫入全局狀態"""
        with FileLock(str(self.lock_file), timeout=10):
            with open(self.state_file, "w", encoding="utf-8") as f:
                json.dump(state.to_dict(), f, ensure_ascii=False, indent=2)
    
    def save_experiment(self, result: ExperimentResult) -> str:
        """
        保存實驗結果
        
        Returns:
            實驗文件路徑
        """
        filepath = self.experiments_dir / f"exp_{result.experiment_id}.json"
        
        with FileLock(str(self.lock_file), timeout=10):
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)
        
        # 更新狀態
        state = self._read_state()
        state.total_experiments += 1
        state.last_experiment_at = datetime.now().isoformat()
        
        if result.status == "keep":
            state.successful_experiments += 1
        elif result.status == "discard":
            state.discarded_experiments += 1
        elif result.status == "crash":
            state.crashed_experiments += 1
        
        # 更新 recent_results (保留最近 100 個)
        state.recent_results.append(result.experiment_id)
        if len(state.recent_results) > 100:
            state.recent_results = state.recent_results[-100:]
        
        # 更新最佳參數
        strategy_name = result.strategy_name
        if result.status == "keep":
            if strategy_name not in state.best_by_strategy or \
               result.sharpe > state.best_by_strategy[strategy_name].metrics.get("sharpe_ratio", -999):
                state.best_by_strategy[strategy_name] = BestParams(
                    strategy_name=strategy_name,
                    params=result.params,
                    metrics=result.metrics,
                    experiment_id=result.experiment_id,
                    updated_at=datetime.now().isoformat()
                )
        
        self._write_state(state)
        return str(filepath)
    
    def get_best_params(self, strategy_name: str) -> Optional[BestParams]:
        """獲取策略的最佳參數"""
        state = self._read_state()
        return state.best_by_strategy.get(strategy_name)
    
    def get_all_experiments(self, limit: int = 100, strategy_name: Optional[str] = None) -> List[ExperimentResult]:
        """獲取所有實驗結果"""
        files = sorted(self.experiments_dir.glob("exp_*.json"), reverse=True)
        
        results = []
        for f in files[:limit]:
            try:
                with open(f, "r", encoding="utf-8") as fp:
                    data = json.load(fp)
                    exp = ExperimentResult.from_dict(data)
                    if strategy_name is None or exp.strategy_name == strategy_name:
                        results.append(exp)
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
        
        return results
    
    def get_recent_experiments(self, n: int = 10, strategy_name: Optional[str] = None) -> List[ExperimentResult]:
        """獲取最近的 N 個實驗"""
        return self.get_all_experiments(limit=n, strategy_name=strategy_name)
    
    def is_duplicate_params(self, strategy_name: str, params: Dict[str, Any], lookback: int = 20) -> bool:
        """
        檢查參數組合是否最近已跑過
        
        Args:
            strategy_name: 策略名稱
            params: 要檢查的參數
            lookback: 回看最近多少個實驗
            
        Returns:
            True 如果找到相似參數
        """
        recent = self.get_recent_experiments(n=lookback, strategy_name=strategy_name)
        
        for exp in recent:
            if exp.params == params:
                return True
        
        return False
    
    def get_state(self) -> AutoresearchState:
        """獲取全局狀態"""
        return self._read_state()
    
    def get_best_by_strategy(self) -> Dict[str, BestParams]:
        """獲取所有策略的最佳參數"""
        state = self._read_state()
        return state.best_by_strategy
    
    def archive_failed(self, min_age_hours: int = 24) -> int:
        """
        歸檔失敗的舊實驗（節省空間）
        
        Args:
            min_age_hours: 至少多少小時前的失敗實驗
            
        Returns:
            歸檔數量
        """
        archive_dir = self.base_dir / "archive"
        archive_dir.mkdir(exist_ok=True)
        
        now = datetime.now()
        count = 0
        
        for f in self.experiments_dir.glob("exp_*.json"):
            try:
                with open(f, "r", encoding="utf-8") as fp:
                    data = json.load(fp)
                    exp = ExperimentResult.from_dict(data)
                
                # 只歸檔 discard 或 crash 的
                if exp.status not in ("discard", "crash"):
                    continue
                
                # 檢查年齡
                ts = datetime.fromisoformat(exp.timestamp)
                age_hours = (now - ts).total_seconds() / 3600
                
                if age_hours >= min_age_hours:
                    shutil.move(str(f), str(archive_dir / f.name))
                    count += 1
            except Exception:
                continue
        
        return count
    
    def prune_experiments(self, keep_best_n: int = 50, strategy_name: Optional[str] = None) -> int:
        """
        清理舊實驗，保留每個策略最好的 N 個
        
        Returns:
            刪除數量
        """
        deleted = 0
        
        # 按策略分組
        by_strategy: Dict[str, List[ExperimentResult]] = {}
        for exp in self.get_all_experiments(limit=10000):
            if strategy_name and exp.strategy_name != strategy_name:
                continue
            if exp.status == "keep":
                by_strategy.setdefault(exp.strategy_name, []).append(exp)
        
        # 每個策略保留最好的 N 個
        all_to_delete = []
        for strat, exps in by_strategy.items():
            sorted_exps = sorted(exps, key=lambda e: e.sharpe, reverse=True)
            all_to_delete.extend(sorted_exps[keep_best_n:])
        
        for exp in all_to_delete:
            f = self.experiments_dir / f"exp_{exp.experiment_id}.json"
            if f.exists():
                f.unlink()
                deleted += 1
        
        return deleted
    
    def generate_report(self) -> str:
        """生成實驗報告"""
        state = self._read_state()
        best_params = state.best_by_strategy
        
        lines = []
        lines.append("=" * 65)
        lines.append("              Autoresearch 實驗報告")
        lines.append("=" * 65)
        lines.append(f"總實驗數:    {state.total_experiments}")
        lines.append(f"成功 (keep): {state.successful_experiments}")
        lines.append(f"放棄 (discard): {state.discarded_experiments}")
        lines.append(f"崩潰 (crash): {state.crashed_experiments}")
        lines.append("")
        
        if state.started_at:
            lines.append(f"開始時間: {state.started_at}")
        if state.last_experiment_at:
            lines.append(f"最後實驗: {state.last_experiment_at}")
        
        lines.append("")
        lines.append("【各策略最佳參數】")
        
        for strat, best in best_params.items():
            lines.append(f"\n  📈 {strat}")
            lines.append(f"     夏普值: {best.metrics.get('sharpe_ratio', 0):.4f}")
            lines.append(f"     勝率:   {best.metrics.get('win_rate', 0):.2f}%")
            lines.append(f"     回撤:   {best.metrics.get('max_drawdown_pct', 0):.2f}%")
            lines.append(f"     收益:   {best.metrics.get('total_return_pct', 0):.2f}%")
            lines.append(f"     參數:   {best.params}")
            lines.append(f"     ExpID: {best.experiment_id}")
        
        if not best_params:
            lines.append("  (尚無成功實驗)")
        
        lines.append("")
        lines.append("=" * 65)
        
        return "\n".join(lines)
