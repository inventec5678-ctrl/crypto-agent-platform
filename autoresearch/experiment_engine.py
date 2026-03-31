"""
Autoresearch 實驗引擎

核心：執行單次實驗（參數 → 回測 → 評估 → 記錄）
"""

import asyncio
import time
import traceback
import sys
import os
from pathlib import Path
from typing import Dict, Any, Optional, Callable, TYPE_CHECKING
from datetime import datetime

# 確保專案根目錄在 Python 路徑
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backtest.backtest_engine import BacktestEngine, PositionSide
from .models import ExperimentResult


class BacktestStrategyWrapper:
    """
    將 live 策略（analyze）適配到 BacktestEngine（generate_signal）
    
    BacktestEngine 會調用 strategy.generate_signal(market_data)
    我們攔截這個調用，内部驅動 live 策略的 analyze()
    """
    
    def __init__(self, live_strategy, strategy_class):
        self.live_strategy = live_strategy
        self.strategy_class = strategy_class
        self._last_direction = None
    
    async def _call_analyze(self, market_data):
        """
        從 market_data 構造 klines 格式，調用 live 策略的 analyze()
        
        market_data: {symbol: DataFrame} from BacktestEngine
        """
        symbol = list(market_data.keys())[0]
        df = market_data[symbol]
        
        if len(df) < 5:
            return None
        
        # 構造 klines 格式（同 binance_client.get_klines 返回格式）
        klines = []
        for _, row in df.iterrows():
            klines.append({
                "open_time": int(row["open_time"].timestamp() * 1000) if hasattr(row["open_time"], "timestamp") else 0,
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row["volume"]) if "volume" in row else 0,
                "close_time": 0,
                "quote_volume": 0,
                "trades": 0,
                "taker_buy_base": 0,
                "taker_buy_quote": 0,
                "ignore": 0,
            })
        
        # 動態注入 klines 到策略實例（避免修改策略類）
        # 策略的 analyze() 內部會調用 binance_client.get_klines，
        # 我們 patch 掉這個調用
        from unittest.mock import patch
        import binance_client
        
        original_get = binance_client.binance_client.get_klines
        
        async def fake_get_klines(*args, **kwargs):
            return klines
        
        try:
            with patch.object(binance_client.binance_client, 'get_klines', fake_get_klines):
                result = await self.live_strategy.analyze()
            return result
        except Exception:
            return None
    
    def generate_signal(self, market_data: Dict[str, Any]) -> PositionSide:
        """
        BacktestEngine 同步調用的接口
        由於 analyze 是 async，我們用執行緒包裝
        """
        import concurrent.futures
        
        def run_in_thread():
            """在獨立執行緒中執行異步程式碼"""
            try:
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                try:
                    return new_loop.run_until_complete(self._call_analyze(market_data))
                finally:
                    new_loop.close()
            except Exception:
                return None
        
        try:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                signal_data = executor.submit(run_in_thread).result(timeout=30)
        except Exception:
            return PositionSide.FLAT
        
        if signal_data is None:
            # None means "hold/no action" - don't override the backtest engine's
            # position tracking. Return FLAT so the engine keeps its current position.
            return PositionSide.FLAT

        direction = signal_data.get("direction", "")

        if direction == "LONG":
            return PositionSide.LONG
        elif direction == "SHORT":
            return PositionSide.SHORT
        elif direction in ("CLOSE_LONG", "CLOSE_SHORT", "FLAT"):
            return PositionSide.FLAT

        return PositionSide.FLAT


class ExperimentEngine:
    """
    實驗引擎
    
    職責：
    1. 接收策略 + 參數
    2. 執行回測
    3. 評估績效
    4. 判定 keep/discard/crash
    5. 記錄到持久化層
    """

    def __init__(
        self,
        persistence,
        symbol: str = "BTCUSDT",
        interval: str = "1h",
        start_date: str = "2024-01-01",
        end_date: str = "2024-12-31",
        initial_capital: float = 10000.0,
        data_fetcher=None  # 可注入自定義數據獲取器
    ):
        self.persistence = persistence
        self.symbol = symbol
        self.interval = interval
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        self.data_fetcher = data_fetcher
        
        # 評估閾值
        self.min_sharpe = -2.0          # 低於此值直接放棄
        self.min_trades = 3             # 交易次數太少不準確
        self.crash_on_exception = True

    async def run_single(
        self,
        strategy_class,
        params: Dict[str, Any],
        experiment_id: Optional[str] = None,
        description: str = "",
        progress_callback: Optional[Callable] = None
    ) -> ExperimentResult:
        """
        執行單次實驗
        
        Args:
            strategy_class: 策略類（可調用構造函數）
            params: 策略參數
            experiment_id: 實驗 ID（可選）
            description: 實驗描述
            
        Returns:
            ExperimentResult
        """
        strategy_name = strategy_class.__name__
        timestamp = datetime.now().isoformat()
        start_time = time.time()
        
        # 生成 experiment_id
        if experiment_id is None:
            import hashlib, json
            content = f"{strategy_name}:{json.dumps(params, sort_keys=True)}:{timestamp}"
            experiment_id = hashlib.sha256(content.encode()).hexdigest()[:12]
        
        if progress_callback:
            progress_callback(f"開始實驗 {experiment_id}: {params}")
        
        try:
            # ── 1. 建立策略 ──
            live_strategy = strategy_class(**params)
            
            # ── 2. 適配到 BacktestEngine 接口 ──
            strategy = BacktestStrategyWrapper(live_strategy, strategy_class)
            
            # ── 3. 建立回測引擎 ──
            config_path = PROJECT_ROOT / "backtest" / "backtest_config.yaml"
            engine = BacktestEngine(str(config_path) if config_path.exists() else None)
            engine.initial_capital = self.initial_capital
            engine.max_position_size = 0.2
            engine.set_strategy(strategy)
            
            # ── 3. 載入數據 ──
            if self.data_fetcher:
                # 使用注入的數據（支持預先載入的數據）
                for symbol, df in self.data_fetcher.items():
                    engine.load_dataframe(symbol, df)
            else:
                await engine.load_data(
                    self.symbol,
                    self.interval,
                    self.start_date,
                    self.end_date
                )
            
            # ── 4. 執行回測 ──
            if progress_callback:
                progress_callback(f"執行回測中...")
            
            result = await engine.run()
            
            # ── 5. 提取指標 ──
            metrics = {
                "sharpe_ratio": result.Sharpe_Ratio,
                "max_drawdown_pct": result.Max_Drawdown_Pct,
                "win_rate": result.Win_Rate,
                "total_return_pct": result.total_return_pct,
                "profit_factor": result.Profit_Factor,
                "total_trades": result.Total_Trades,
                "volatility": getattr(result, 'Sharpe_Ratio', 0) or 0,
                "sortino_ratio": getattr(result, 'Sharpe_Ratio', 0) or 0,
            }
            
            # ── 6. 評估狀態 ──
            duration = time.time() - start_time
            status = self._evaluate_status(metrics, result.Total_Trades)
            
            if status == "keep":
                if description:
                    description = f"✅ {description}"
                else:
                    description = f"Sharpe={metrics['sharpe_ratio']:.3f}, WinRate={metrics['win_rate']:.1f}%"
            elif status == "discard":
                reason = self._get_discard_reason(metrics, result.Total_Trades)
                description = f"❌ 放棄: {reason}"
            else:
                description = f"💥 崩潰"
            
            experiment_result = ExperimentResult(
                experiment_id=experiment_id,
                commit_hash=self._get_commit_hash(),
                strategy_name=strategy_name,
                params=params,
                metrics=metrics,
                status=status,
                description=description,
                timestamp=timestamp,
                duration_seconds=round(duration, 2),
                backtest_period={
                    "start": self.start_date,
                    "end": self.end_date
                }
            )
            
            # ── 7. 持久化 ──
            self.persistence.save_experiment(experiment_result)
            
            if progress_callback:
                progress_callback(
                    f"實驗完成: {experiment_id} | "
                    f"Sharpe={metrics['sharpe_ratio']:.3f} | "
                    f"Status={status} | "
                    f"耗時={duration:.1f}s"
                )
            
            engine.close()
            
            return experiment_result
            
        except Exception as e:
            duration = time.time() - start_time
            tb = traceback.format_exc()
            
            experiment_result = ExperimentResult(
                experiment_id=experiment_id,
                commit_hash=self._get_commit_hash(),
                strategy_name=strategy_name,
                params=params,
                metrics={},
                status="crash",
                description=f"💥 崩潰: {type(e).__name__}: {str(e)}",
                timestamp=timestamp,
                duration_seconds=round(duration, 2),
                backtest_period={
                    "start": self.start_date,
                    "end": self.end_date
                }
            )
            
            self.persistence.save_experiment(experiment_result)
            
            if progress_callback:
                progress_callback(f"💥 實驗崩潰: {experiment_id} - {e}")
            
            return experiment_result

    def _evaluate_status(self, metrics: Dict[str, float], total_trades: int) -> str:
        """評估實驗狀態"""
        sharpe = metrics.get("sharpe_ratio", -999)
        max_dd = metrics.get("max_drawdown_pct", 100)
        win_rate = metrics.get("win_rate", 0)
        total_return = metrics.get("total_return_pct", -100)
        
        # 崩潰檢測：只有在交易次數為0或明顯資料問題時才崩潰
        if total_trades == 0:
            return "crash"
        
        # 交易次數太少
        if total_trades < self.min_trades:
            return "discard"
        
        # 硬性門檻
        if sharpe < self.min_sharpe:
            return "discard"
        if max_dd > 95:
            return "discard"
        
        # Keep 的條件
        keep_conditions = 0
        if sharpe > 0.5:
            keep_conditions += 1
        if max_dd < 30:
            keep_conditions += 1
        if total_return > 0:
            keep_conditions += 1
        if win_rate > 45:
            keep_conditions += 1
        
        if keep_conditions >= 2:
            return "keep"
        elif keep_conditions == 1:
            # 邊緣案例，看總體表現
            score = sharpe * 0.4 + (100 - max_dd) * 0.01 + total_return * 0.01 + win_rate * 0.01
            if score > 10:
                return "keep"
        
        return "discard"

    def _get_discard_reason(self, metrics: Dict[str, float], total_trades: int) -> str:
        """獲取放棄原因"""
        sharpe = metrics.get("sharpe_ratio", -999)
        max_dd = metrics.get("max_drawdown_pct", 100)
        
        if total_trades < self.min_trades:
            return f"交易次數不足 ({total_trades})"
        if sharpe < self.min_sharpe:
            return f"夏普值過低 ({sharpe:.3f})"
        if max_dd > 95:
            return f"回撤過大 ({max_dd:.1f}%)"
        
        return "綜合指標未達標準"

    def _get_commit_hash(self) -> str:
        """獲取當前 commit hash"""
        try:
            import subprocess
            result = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass
        return "unknown"


class ExperimentEngineSync:
    """實驗引擎同步包裝（用於非 async 環境）"""

    def __init__(self, **kwargs):
        self.async_engine = ExperimentEngine(**kwargs)

    def run_single(self, strategy_class, params: Dict[str, Any], **kwargs):
        return asyncio.run(
            self.async_engine.run_single(strategy_class, params, **kwargs)
        )
