#!/usr/bin/env python3
"""
Autoresearch CLI - 命令行入口

用法:
  python cli.py run [--experiments N] [--hours H] [--strategy NAME]
  python cli.py status
  python cli.py report
  python cli.py best [strategy_name]
  python cli.py history [strategy_name]
  python cli.py prune [--keep N]
"""

import argparse
import asyncio
import sys
import os
from pathlib import Path

# 確保專案路徑
PROJECT_ROOT = Path(__file__).parent.parent.parent
print(f"DEBUG __file__={__file__} PROJECT_ROOT={PROJECT_ROOT} sys.path[0]={sys.path[0]}")
sys.path.insert(0, str(PROJECT_ROOT))

from autoresearch.loop import AutoresearchLoop, LoopConfig
from autoresearch.persistence import Persistence
from autoresearch.registry import ALL_SPECS, get_specs, get_default_params


# 策略適配器：將 live 策略（analyze）轉換為 backtest 引擎介面（generate_signal）
import asyncio
from backtest.backtest_engine import BaseStrategy, PositionSide
from typing import Dict


class LiveStrategyAdapter(BaseStrategy):
    """
    將 live 策略（使用 analyze() 方法）適配到 backtest 引擎
    
    原理：
    - 從市場數據構造模擬 klines
    - 調用策略的 analyze() 獲取信號
    - 轉換為 PositionSide
    """
    
    def __init__(self, live_strategy):
        self.live_strategy = live_strategy
        self._last_signal = PositionSide.FLAT
    
    def _df_to_klines(self, df) -> list:
        """將 DataFrame 轉為 binance_client 格式的 klines"""
        rows = []
        for _, row in df.iterrows():
            rows.append({
                "open_time": row.get("open_time", 0),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row["volume"]),
                "close_time": 0,
                "quote_volume": 0,
                "trades": 0,
                "taker_buy_base": 0,
                "taker_buy_quote": 0,
                "ignore": 0,
            })
        return rows
    
    def generate_signal(self, market_data: Dict[str, any]) -> PositionSide:
        """同步介面：從市場數據生成信號"""
        try:
            # 取第一個 symbol 的數據
            symbol = list(market_data.keys())[0]
            df = market_data[symbol]
            
            if len(df) < 10:
                return PositionSide.FLAT
            
            # 轉為 klines 格式
            klines = self._df_to_klines(df)
            
            # 同步調用（策略的 analyze 是 async）
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                signal_data = loop.run_until_complete(
                    self.live_strategy.analyze_for_backtest(klines)
                )
            finally:
                loop.close()
            
            if signal_data is None:
                return PositionSide.FLAT
            
            direction = signal_data.get("direction", "")
            
            if direction == "LONG":
                return PositionSide.LONG
            elif direction in ("CLOSE_LONG", "CLOSE_SHORT", "FLAT"):
                return PositionSide.FLAT
            elif direction == "SHORT":
                return PositionSide.SHORT
            
            return PositionSide.FLAT
            
        except Exception:
            return PositionSide.FLAT


def load_strategies():
    """動態載入所有策略"""
    strategies = {}
    
    # MACrossoverStrategy
    try:
        from strategies.strategy_ma_crossover import MACrossoverStrategy
        strategies["MACrossoverStrategy"] = MACrossoverStrategy
    except ImportError as e:
        print(f"⚠️ 無法載入 MACrossoverStrategy: {e}")
    
    # RSIReversalStrategy  
    try:
        from strategies.strategy_rsi import RSIReversalStrategy
        strategies["RSIReversalStrategy"] = RSIReversalStrategy
    except ImportError as e:
        print(f"⚠️ 無法載入 RSIReversalStrategy: {e}")
    
    # BBStrategy
    try:
        from strategies.strategy_bb import BBBreakoutStrategy
        strategies["BBBreakoutStrategy"] = BBBreakoutStrategy
    except ImportError as e:
        print(f"⚠️ 無法載入 BBBreakoutStrategy: {e}")
    
    return strategies


def cmd_run(args):
    """運行 Autoresearch 循環"""
    strategies = load_strategies()
    
    if not strategies:
        print("❌ 沒有可用的策略")
        sys.exit(1)
    
    # 過濾策略
    if args.strategy:
        strategies = {k: v for k, v in strategies.items() if k == args.strategy}
        if not strategies:
            print(f"❌ 未知策略: {args.strategy}")
            sys.exit(1)
    
    # 構建參數空間
    param_specs = {}
    for name in strategies:
        specs = get_specs(name)
        if specs:
            param_specs[name] = specs
        else:
            print(f"⚠️  {name} 沒有定義參數空間，跳過")
    
    if not param_specs:
        print("❌ 沒有有效的策略參數空間")
        sys.exit(1)
    
    # 配置
    config = LoopConfig(
        max_experiments=args.experiments,
        max_hours=args.hours,
        pause_between_experiments=args.pause,
        enable_bayesian=not args.no_bayesian,
        keep_best_params_from_persistence=True,
    )
    
    # Autoresearch 目錄
    base_dir = PROJECT_ROOT / "autoresearch"
    base_dir.mkdir(exist_ok=True)
    
    # 創建循環
    loop = AutoresearchLoop(
        base_dir=str(base_dir),
        strategies=strategies,
        param_specs=param_specs,
        config=config,
        symbol=args.symbol,
        interval=args.interval,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    
    # 啟動
    loop.start(
        max_experiments=args.experiments,
        max_hours=args.hours,
        background=args.background
    )


def cmd_status(args):
    """顯示狀態"""
    base_dir = PROJECT_ROOT / "autoresearch"
    persistence = Persistence(str(base_dir))
    state = persistence.get_state()
    
    print(f"""
╔══════════════════════════════════════════════╗
║         Autoresearch 狀態                    ║
╠══════════════════════════════════════════════╣
║  總實驗數:    {state.total_experiments}
║  成功 (keep): {state.successful_experiments}
║  放棄 (discard): {state.discarded_experiments}
║  崩潰 (crash): {state.crashed_experiments}
║  已探索策略:  {len(state.best_by_strategy)}
╠══════════════════════════════════════════════╣""")
    
    for strat, best in state.best_by_strategy.items():
        sharpe = best.metrics.get("sharpe_ratio", 0)
        print(f"║  📈 {strat:<35}")
        print(f"║     Sharpe={sharpe:.3f} | WinRate={best.metrics.get('win_rate', 0):.1f}% "
              f"| DD={best.metrics.get('max_drawdown_pct', 0):.1f}%")
    
    print(f"""╠══════════════════════════════════════════════╣
║  開始時間:   {state.started_at or 'N/A'}
║  最後實驗:   {state.last_experiment_at or 'N/A'}
╚══════════════════════════════════════════════╝""")


def cmd_report(args):
    """生成完整報告"""
    base_dir = PROJECT_ROOT / "autoresearch"
    persistence = Persistence(str(base_dir))
    print(persistence.generate_report())


def cmd_best(args):
    """顯示最佳參數"""
    base_dir = PROJECT_ROOT / "autoresearch"
    persistence = Persistence(str(base_dir))
    
    if args.strategy_name:
        best = persistence.get_best_params(args.strategy_name)
        if best:
            print(f"\n🏆 {args.strategy_name} 最佳參數:")
            print(f"   夏普值: {best.metrics.get('sharpe_ratio', 0):.4f}")
            print(f"   勝率:   {best.metrics.get('win_rate', 0):.2f}%")
            print(f"   回撤:   {best.metrics.get('max_drawdown_pct', 0):.2f}%")
            print(f"   收益:   {best.metrics.get('total_return_pct', 0):.2f}%")
            print(f"   參數:   {best.params}")
            print(f"   ExpID:  {best.experiment_id}")
            print(f"   更新:   {best.updated_at}")
        else:
            print(f"❌ 沒有 {args.strategy_name} 的歷史數據")
    else:
        # 所有最佳
        for strat, best in persistence.get_best_by_strategy().items():
            print(f"\n🏆 {strat}:")
            print(f"   Sharpe={best.metrics.get('sharpe_ratio', 0):.3f}")
            print(f"   參數: {best.params}")


def cmd_history(args):
    """顯示實驗歷史"""
    base_dir = PROJECT_ROOT / "autoresearch"
    persistence = Persistence(str(base_dir))
    
    experiments = persistence.get_recent_experiments(
        n=args.limit,
        strategy_name=args.strategy_name
    )
    
    if not experiments:
        print("❌ 沒有實驗記錄")
        return
    
    print(f"\n最近 {len(experiments)} 個實驗:")
    print(f"{'ExpID':<14} {'策略':<25} {'Sharpe':>8} {'WinRate':>8} {'DD%':>7} {'Status':<10} {'Time':<8}")
    print("-" * 90)
    
    for exp in experiments:
        status_icon = {"keep": "✅", "discard": "❌", "crash": "💥"}.get(exp.status, "?")
        print(
            f"{exp.experiment_id:<14} "
            f"{exp.strategy_name:<25} "
            f"{exp.sharpe:>8.3f} "
            f"{exp.win_rate:>7.1f}% "
            f"{exp.max_drawdown:>6.1f}% "
            f"{status_icon} "
            f"{exp.timestamp[11:19]}"
        )


def cmd_prune(args):
    """清理舊實驗"""
    base_dir = PROJECT_ROOT / "autoresearch"
    persistence = Persistence(str(base_dir))
    
    deleted = persistence.prune_experiments(
        keep_best_n=args.keep,
        strategy_name=args.strategy_name
    )
    
    archived = persistence.archive_failed(min_age_hours=args.archive_hours)
    
    print(f"✅ 清理完成: 刪除 {deleted} 個實驗, 歸檔 {archived} 個失敗實驗")


def cmd_single(args):
    """單次實驗"""
    from autoresearch.experiment_engine import ExperimentEngine
    
    strategies = load_strategies()
    
    if args.strategy not in strategies:
        print(f"❌ 未知策略: {args.strategy}")
        sys.exit(1)
    
    # 解析參數
    params = {}
    for p in (args.params or []):
        if "=" not in p:
            continue
        k, v = p.split("=", 1)
        try:
            params[k] = int(v)
        except ValueError:
            try:
                params[k] = float(v)
            except ValueError:
                params[k] = v
    
    if not params:
        # 使用默認
        params = get_default_params(args.strategy)
        print(f"使用默認參數: {params}")
    
    base_dir = PROJECT_ROOT / "autoresearch"
    persistence = Persistence(str(base_dir))
    
    engine = ExperimentEngine(
        persistence=persistence,
        symbol=args.symbol,
        interval=args.interval,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    
    result = asyncio.run(
        engine.run_single(
            strategy_class=strategies[args.strategy],
            params=params,
            description=args.description or f"CLI single experiment"
        )
    )
    
    print(f"\n實驗完成: {result.experiment_id}")
    print(f"狀態: {result.status}")
    print(f"Sharpe: {result.sharpe:.4f}")
    print(f"WinRate: {result.win_rate:.2f}%")
    print(f"MaxDD: {result.max_drawdown:.2f}%")
    print(f"描述: {result.description}")


def main():
    parser = argparse.ArgumentParser(
        description="Autoresearch 24/7 自動策略實驗系統",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="子命令")
    
    # run
    p_run = subparsers.add_parser("run", help="啟動 Autoresearch 循環")
    p_run.add_argument("--experiments", "-n", type=int, default=None, help="最大實驗次數")
    p_run.add_argument("--hours", "-H", type=float, default=None, help="最大運行小時")
    p_run.add_argument("--strategy", "-s", type=str, default=None, help="只運行指定策略")
    p_run.add_argument("--symbol", default="BTCUSDT", help="交易對 (默認: BTCUSDT)")
    p_run.add_argument("--interval", "-i", default="1h", help="K線周期 (默認: 1h)")
    p_run.add_argument("--start-date", default="2024-01-01", help="回測開始日期")
    p_run.add_argument("--end-date", default="2024-12-31", help="回測結束日期")
    p_run.add_argument("--pause", type=float, default=5.0, help="實驗間隔秒數")
    p_run.add_argument("--no-bayesian", action="store_true", help="禁用貝葉斯優化")
    p_run.add_argument("--background", "-b", action="store_true", help="後台運行")
    p_run.set_defaults(func=cmd_run)
    
    # status
    p_status = subparsers.add_parser("status", help="顯示 Autoresearch 狀態")
    p_status.set_defaults(func=cmd_status)
    
    # report
    p_report = subparsers.add_parser("report", help="生成完整實驗報告")
    p_report.set_defaults(func=cmd_report)
    
    # best
    p_best = subparsers.add_parser("best", help="顯示最佳參數")
    p_best.add_argument("strategy_name", nargs="?", help="策略名稱")
    p_best.set_defaults(func=cmd_best)
    
    # history
    p_history = subparsers.add_parser("history", help="顯示實驗歷史")
    p_history.add_argument("--strategy", dest="strategy_name", default=None)
    p_history.add_argument("--limit", "-n", type=int, default=20)
    p_history.set_defaults(func=cmd_history)
    
    # prune
    p_prune = subparsers.add_parser("prune", help="清理舊實驗")
    p_prune.add_argument("--keep", type=int, default=50, help="每策略保留最佳 N 個")
    p_prune.add_argument("--strategy", dest="strategy_name", default=None)
    p_prune.add_argument("--archive-hours", type=int, default=24, help="歸檔多少小時前的失敗實驗")
    p_prune.set_defaults(func=cmd_prune)
    
    # single
    p_single = subparsers.add_parser("single", help="運行單次實驗")
    p_single.add_argument("strategy", help="策略名稱")
    p_single.add_argument("--params", "-p", action="append", help="參數如 fast_period=10")
    p_single.add_argument("--description", "-d", help="實驗描述")
    p_single.add_argument("--symbol", default="BTCUSDT")
    p_single.add_argument("--interval", default="1h")
    p_single.add_argument("--start-date", default="2024-01-01")
    p_single.add_argument("--end-date", default="2024-12-31")
    p_single.set_defaults(func=cmd_single)
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    
    args.func(args)


if __name__ == "__main__":
    main()
