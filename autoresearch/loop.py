"""
Autoresearch 24/7 自動循環控制器

核心循環：
1. 讀取當前策略參數
2. 產生實驗變體
3. 自動跑回測
4. 評估績效
5. 保留成功，放棄失敗
6. 持續迭代
"""

import asyncio
import time
import signal
import sys
import threading
from pathlib import Path
from typing import Dict, Any, Optional, List, Type, Callable
from dataclasses import dataclass
from datetime import datetime
import random

from .experiment_engine import ExperimentEngine
from .experiment_strategies import MutationStrategy, GridSearchStrategy, BayesianOptimizer, EnsembleStrategy
from .persistence import Persistence
from .models import StrategyParamSpec, AutoresearchState


@dataclass
class LoopConfig:
    """循環配置"""
    max_experiments: Optional[int] = None    # 最大實驗次數（None=無限）
    max_hours: Optional[float] = None        # 最大運行小時（None=無限）
    max_consecutive_failures: int = 10       # 連續失敗後休息
    pause_between_experiments: float = 5.0   # 實驗間隔（秒）
    pause_on_failure: float = 30.0           # 失敗後額外暫停（秒）
    strategy_weights: Optional[Dict[str, float]] = None  # 策略抽樣權重
    enable_bayesian: bool = True              # 是否啟用貝葉斯優化
    keep_best_params_from_persistence: bool = True  # 啟動時從歷史讀取最佳參數


class AutoresearchLoop:
    """
    24/7 Autoresearch 循環
    
    用法:
    >>> loop = AutoresearchLoop(
    ...     base_dir="~/.openclaw/workspace/crypto-agent-platform/autoresearch",
    ...     strategies={"MACrossoverStrategy": MACrossoverStrategy}
    ... )
    >>> loop.start()  # 永不停止
    >>> # 或
    >>> loop.start(max_experiments=100)  # 跑 100 次後停止
    """

    def __init__(
        self,
        base_dir: str,
        strategies: Dict[str, Type],
        param_specs: Dict[str, Dict[str, StrategyParamSpec]],
        config: Optional[LoopConfig] = None,
        symbol: str = "BTCUSDT",
        interval: str = "1h",
        start_date: str = "2024-01-01",
        end_date: str = "2024-12-31",
        data_fetcher=None  # 可注入本地數據
    ):
        self.base_dir = Path(base_dir).expanduser()
        self.data_fetcher = data_fetcher
        self.strategies = strategies
        self.param_specs = param_specs  # {strategy_name: {param_name: spec}}
        self.config = config or LoopConfig()
        self.symbol = symbol
        self.interval = interval
        self.start_date = start_date
        self.end_date = end_date
        
        # 初始化持久化
        self.persistence = Persistence(str(self.base_dir))
        
        # 初始化探索策略
        self.explorers: Dict[str, EnsembleStrategy] = {}
        for name, specs in param_specs.items():
            self.explorers[name] = EnsembleStrategy(specs)
        
        # 初始化實驗引擎
        self.engine = ExperimentEngine(
            persistence=self.persistence,
            symbol=symbol,
            interval=interval,
            start_date=start_date,
            end_date=end_date,
            data_fetcher=data_fetcher
        )
        
        # 狀態
        self._running = False
        self._stop_requested = False
        self._paused = False
        self._state: Optional[AutoresearchState] = None
        
        # 計數器
        self.experiment_count = 0
        self.consecutive_failures = 0
        self.consecutive_successes = 0
        
        # 當前策略
        self.current_strategy_name: Optional[str] = None
        self.current_params: Optional[Dict[str, Any]] = None
        
        # 回調
        self.on_experiment_complete: Optional[Callable] = None
        self.on_new_best: Optional[Callable] = None
        self.on_crash: Optional[Callable] = None
        self.on_loop_end: Optional[Callable] = None
        
        # 信號處理
        self._setup_signal_handlers()
        
        # 初始化狀態
        self._init_state()

    def _setup_signal_handlers(self):
        """設定信號處理"""
        def signal_handler(signum, frame):
            sig_name = signal.Signals(signum).name
            print(f"\n📡 收到信號 {sig_name}，準備停止...")
            self._stop_requested = True
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    def _init_state(self):
        """初始化/恢復狀態"""
        state = self.persistence.get_state()
        
        if state.started_at is None:
            state.started_at = datetime.now().isoformat()
        
        # 恢復最佳參數到探索器
        if self.config.keep_best_params_from_persistence:
            for strat_name, explorer in self.explorers.items():
                best = self.persistence.get_best_params(strat_name)
                if best:
                    explorer.best_params = best.params
                    explorer.best_sharpe = best.metrics.get("sharpe_ratio", -999)
                    print(f"  📂 恢復 {strat_name} 最佳參數: Sharpe={explorer.best_sharpe:.3f}")
        
        self._state = state
        self.experiment_count = state.total_experiments
        print(f"\nAutoresearch 已初始化:")
        print(f"  策略數: {len(self.strategies)}")
        print(f"  總實驗數: {self.experiment_count}")
        print(f"  最佳策略: {len(state.best_by_strategy)}")

    def _select_strategy(self) -> str:
        """選擇下一個策略（帶權重隨機）"""
        weights = self.config.strategy_weights or {}
        
        # 根據歷史表現動態調整權重
        adjusted_weights = {}
        for name in self.strategies:
            base_weight = weights.get(name, 1.0)
            best = self.persistence.get_best_params(name)
            if best:
                sharpe = best.metrics.get("sharpe_ratio", 0)
                adjusted_weights[name] = base_weight * (1 + sharpe * 0.5)
            else:
                adjusted_weights[name] = base_weight * 0.5  # 未探索的策略優先
        
        total = sum(adjusted_weights.values())
        roll = random.random() * total
        
        cumulative = 0
        for name, w in adjusted_weights.items():
            cumulative += w
            if roll <= cumulative:
                return name
        
        return random.choice(list(self.strategies.keys()))

    def _should_stop(self) -> bool:
        """檢查是否應該停止"""
        if self._stop_requested:
            return True
        
        if self.config.max_experiments and self.experiment_count >= self.config.max_experiments:
            print(f"\n🏁 達到最大實驗次數 ({self.config.max_experiments})")
            return True
        
        if self.config.max_hours:
            elapsed = (datetime.now() - datetime.fromisoformat(self._state.started_at)).total_seconds() / 3600
            if elapsed >= self.config.max_hours:
                print(f"\n🏁 達到最大運行時間 ({self.config.max_hours}h)")
                return True
        
        return False

    async def _run_single_iteration(self):
        """執行單次迭代"""
        # ── 1. 選擇策略 ──
        strategy_name = self._select_strategy()
        self.current_strategy_name = strategy_name
        strategy_class = self.strategies[strategy_name]
        explorer = self.explorers[strategy_name]
        
        # ── 2. 產生實驗參數 ──
        params = explorer.suggest()
        self.current_params = params
        
        # 避免完全重複
        attempts = 0
        while self.persistence.is_duplicate_params(strategy_name, params) and attempts < 10:
            params = explorer.mutation.random_explore()
            attempts += 1
        
        # ── 3. 執行實驗 ──
        description = f"{strategy_name} #{self.experiment_count + 1}"
        
        result = await self.engine.run_single(
            strategy_class=strategy_class,
            params=params,
            description=description,
            progress_callback=lambda msg: print(f"  {msg}")
        )
        
        self.experiment_count += 1
        
        # ── 4. 更新探索器 ──
        explorer.record(result)
        if result.status == "keep":
            explorer.update_best(result.params, result.sharpe)
        
        # ── 5. 更新計數器 ──
        if result.status == "keep":
            self.consecutive_failures = 0
            self.consecutive_successes += 1
        else:
            self.consecutive_failures += 1
            self.consecutive_successes = 0
        
        # ── 6. 回調 ──
        if self.on_experiment_complete:
            try:
                self.on_experiment_complete(result)
            except Exception as e:
                print(f"  ⚠️ on_experiment_complete 回調失敗: {e}")
        
        if result.status == "keep":
            best = self.persistence.get_best_params(strategy_name)
            if best and best.experiment_id == result.experiment_id:
                if self.on_new_best:
                    try:
                        self.on_new_best(result, best)
                    except Exception as e:
                        print(f"  ⚠️ on_new_best 回調失敗: {e}")
        
        if result.status == "crash" and self.on_crash:
            try:
                self.on_crash(result)
            except Exception as e:
                print(f"  ⚠️ on_crash 回調失敗: {e}")
        
        # ── 7. 連續失敗處理 ──
        if self.consecutive_failures >= self.config.max_consecutive_failures:
            print(f"\n😴 連續 {self.consecutive_failures} 次失敗，休息 {self.config.pause_on_failure}s...")
            await asyncio.sleep(self.config.pause_on_failure)
            self.consecutive_failures = 0
        
        return result

    async def start_async(self, max_experiments: Optional[int] = None, max_hours: Optional[float] = None):
        """
        啟動異步循環
        
        Args:
            max_experiments: 最大實驗次數（可選）
            max_hours: 最大運行小時（可選）
        """
        self._running = True
        self._stop_requested = False
        
        # 覆寫配置
        if max_experiments is not None:
            self.config.max_experiments = max_experiments
        if max_hours is not None:
            self.config.max_hours = max_hours
        
        print(f"\n{'='*65}")
        print(f"  🚀 Autoresearch 24/7 循環啟動")
        print(f"{'='*65}")
        print(f"  策略: {list(self.strategies.keys())}")
        print(f"  Symbol: {self.symbol} | Interval: {self.interval}")
        print(f"  回測期間: {self.start_date} ~ {self.end_date}")
        print(f"  最大實驗: {self.config.max_experiments or '∞'}")
        print(f"  最大時長: {self.config.max_hours or '∞'}h")
        print(f"{'='*65}")
        print()
        
        iteration = 0
        start_ts = time.time()
        
        while not self._should_stop():
            iteration += 1
            iter_start = time.time()
            
            print(f"\n{'─'*65}")
            print(f"  迭代 #{iteration} | 總實驗 #{self.experiment_count + 1}")
            print(f"  策略: {self.current_strategy_name or '?'}")
            print(f"  連續成功: {self.consecutive_successes} | 連續失敗: {self.consecutive_failures}")
            
            try:
                result = await self._run_single_iteration()
                
                iter_time = time.time() - iter_start
                print(f"\n  迭代耗時: {iter_time:.1f}s")
                print(f"  實驗狀態: {result.status}")
                
                # 印出狀態摘要
                state = self.persistence.get_state()
                print(f"  累計: {state.total_experiments} 實驗 | "
                      f"{state.successful_experiments} 成功 | "
                      f"{state.discarded_experiments} 放棄 | "
                      f"{state.crashed_experiments} 崩潰")
                
            except Exception as e:
                print(f"\n  💥 迭代異常: {e}")
                import traceback
                traceback.print_exc()
            
            # 實驗間隔
            if not self._should_stop() and self.config.pause_between_experiments > 0:
                await asyncio.sleep(self.config.pause_between_experiments)
        
        total_time = time.time() - start_ts
        self._running = False
        
        print(f"\n{'='*65}")
        print(f"  🛑 Autoresearch 循環結束")
        print(f"  總實驗: {self.experiment_count}")
        print(f"  總耗時: {total_time / 3600:.2f}h")
        print(f"{'='*65}")
        print()
        print(self.persistence.generate_report())
        
        if self.on_loop_end:
            self.on_loop_end(self.experiment_count, total_time)

    def start(
        self,
        max_experiments: Optional[int] = None,
        max_hours: Optional[float] = None,
        background: bool = False
    ):
        """
        啟動循環（同步入口）
        
        Args:
            max_experiments: 最大實驗次數（可選）
            max_hours: 最大運行小時（可選）
            background: 是否在後台運行
        """
        if background:
            thread = threading.Thread(
                target=lambda: asyncio.run(
                    self.start_async(max_experiments=max_experiments, max_hours=max_hours)
                ),
                daemon=True,
                name="autoresearch-loop"
            )
            thread.start()
            print(f"Autoresearch 後台線程已啟動 (PID: {thread.native_id})")
            return thread
        else:
            asyncio.run(self.start_async(max_experiments=max_experiments, max_hours=max_hours))

    def stop(self):
        """請求停止"""
        self._stop_requested = True

    def pause(self):
        """暫停"""
        self._paused = True

    def resume(self):
        """繼續"""
        self._paused = False

    def status(self) -> str:
        """獲取狀態摘要"""
        state = self.persistence.get_state()
        parts = [
            f"Running: {self._running}",
            f"Experiments: {self.experiment_count}/{self.config.max_experiments or '∞'}",
            f"Consecutive Success: {self.consecutive_successes}",
            f"Consecutive Failures: {self.consecutive_failures}",
            f"Best Strategies: {len(state.best_by_strategy)}",
        ]
        return " | ".join(parts)
