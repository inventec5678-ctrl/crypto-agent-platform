"""
Autoresearch 整合介面
位置: crypto-agent-platform/backtest/autoresearch_interface.py

預留介面，讓策略 Agent 可以自動呼叫回測
"""

import os
import json
import yaml
from typing import Dict, List, Optional, Any, Type
from datetime import datetime
import asyncio

from .backtest_engine import (
    BacktestEngine, 
    BaseStrategy, 
    BacktestResult,
    PositionSide
)
from .performance_metrics import PerformanceMetrics
from .optimization import GridSearchOptimizer, WalkForwardAnalyzer, OptimizationResult


class StrategyRequest:
    """策略請求資料結構"""
    
    def __init__(
        self,
        strategy_class: Type[BaseStrategy],
        strategy_params: Dict[str, Any],
        symbol: str,
        interval: str = "1h",
        start_date: str = "2024-01-01",
        end_date: str = "2024-12-31",
        initial_capital: float = 10000.0,
        risk_params: Optional[Dict[str, Any]] = None
    ):
        self.strategy_class = strategy_class
        self.strategy_params = strategy_params
        self.symbol = symbol
        self.interval = interval
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        self.risk_params = risk_params or {}


class OptimizationRequest:
    """優化請求資料結構"""
    
    def __init__(
        self,
        strategy_class: Type[BaseStrategy],
        param_space: Dict[str, List[Any]],
        symbol: str,
        interval: str = "1h",
        start_date: str = "2024-01-01",
        end_date: str = "2024-12-31",
        method: str = "grid_search",
        metric: str = "sharpe_ratio",
        n_windows: int = 5
    ):
        self.strategy_class = strategy_class
        self.param_space = param_space
        self.symbol = symbol
        self.interval = interval
        self.start_date = start_date
        self.end_date = end_date
        self.method = method
        self.metric = metric
        self.n_windows = n_windows


class AutoResearchClient:
    """
    Autoresearch 整合客戶端
    
    供策略 Agent 呼叫回測的標準介面
    
    使用範例:
    ```python
    from backtest import AutoResearchClient, SimpleMovingAverageCrossover
    
    client = AutoResearchClient()
    
    # 單次回測
    result = await client.run_backtest(
        strategy_class=SimpleMovingAverageCrossover,
        params={'fast_period': 10, 'slow_period': 30},
        symbol='BTCUSDT'
    )
    
    # 參數優化
    opt_result = await client.optimize_strategy(
        strategy_class=SimpleMovingAverageCrossover,
        param_space={
            'fast_period': [5, 10, 15, 20],
            'slow_period': [20, 30, 50]
        },
        symbol='BTCUSDT'
    )
    ```
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化客戶端
        
        Args:
            config_path: 配置文件路徑
        """
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), "backtest_config.yaml")
        
        self.config = self._load_config(config_path)
        self.engine: Optional[BacktestEngine] = None
        self.last_result: Optional[BacktestResult] = None
    
    def _load_config(self, config_path: str) -> Dict:
        """載入配置"""
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        return {}
    
    async def initialize(self, request: StrategyRequest) -> BacktestEngine:
        """
        初始化回測引擎
        
        Args:
            request: 策略請求
            
        Returns:
            BacktestEngine: 設定好的引擎實例
        """
        self.engine = BacktestEngine()
        
        # 載入數據
        await self.engine.load_data(
            symbol=request.symbol,
            interval=request.interval,
            start_date=request.start_date,
            end_date=request.end_date
        )
        
        # 設定策略
        strategy = request.strategy_class(**request.strategy_params)
        self.engine.set_strategy(strategy)
        
        # 設定風險參數
        if request.risk_params:
            if 'max_position_size' in request.risk_params:
                self.engine.max_position_size = request.risk_params['max_position_size']
            if 'stop_loss' in request.risk_params:
                self.engine.stop_loss = request.risk_params['stop_loss']
            if 'take_profit' in request.risk_params:
                self.engine.take_profit = request.risk_params['take_profit']
        
        # 覆寫初始資金
        if request.initial_capital:
            self.engine.initial_capital = request.initial_capital
            self.engine.cash = request.initial_capital
            self.engine.equity = request.initial_capital
        
        return self.engine
    
    async def run_backtest(
        self,
        strategy_class: Type[BaseStrategy],
        params: Dict[str, Any],
        symbol: str,
        interval: str = "1h",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        **kwargs
    ) -> BacktestResult:
        """
        執行單次回測
        
        Args:
            strategy_class: 策略類別
            params: 策略參數
            symbol: 交易對
            interval: K線周期
            start_date: 開始日期
            end_date: 結束日期
            **kwargs: 其他參數（如 initial_capital, risk_params）
            
        Returns:
            BacktestResult: 回測結果
        """
        request = StrategyRequest(
            strategy_class=strategy_class,
            strategy_params=params,
            symbol=symbol,
            interval=interval,
            start_date=start_date or self.config.get('backtest', {}).get('start_date', '2024-01-01'),
            end_date=end_date or self.config.get('backtest', {}).get('end_date', '2024-12-31'),
            initial_capital=kwargs.get('initial_capital', self.config.get('backtest', {}).get('initial_capital', 10000)),
            risk_params=kwargs.get('risk_params')
        )
        
        engine = await self.initialize(request)
        result = await engine.run()
        
        self.last_result = result
        engine.close()
        
        return result
    
    async def optimize_strategy(
        self,
        strategy_class: Type[BaseStrategy],
        param_space: Dict[str, List[Any]],
        symbol: str,
        interval: str = "1h",
        method: str = "grid_search",
        metric: str = "sharpe_ratio",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        **kwargs
    ) -> OptimizationResult:
        """
        執行策略參數優化
        
        Args:
            strategy_class: 策略類別
            param_space: 參數空間
            symbol: 交易對
            interval: K線周期
            method: 優化方法 ("grid_search" | "walk_forward")
            metric: 優化目標指標
            start_date: 開始日期
            end_date: 結束日期
            
        Returns:
            OptimizationResult: 優化結果
        """
        # 載入數據
        fetcher = self.engine.data_fetcher if self.engine else None
        
        from .backtest_engine import BinanceDataFetcher
        if fetcher is None:
            fetcher = BinanceDataFetcher()
        
        start = start_date or self.config.get('backtest', {}).get('start_date', '2024-01-01')
        end = end_date or self.config.get('backtest', {}).get('end_date', '2024-12-31')
        
        print(f"📥 正在下載優化所需數據...")
        data = await fetcher.fetch_all_klines(symbol, interval, start, end)
        
        if data.empty:
            raise ValueError(f"無法獲取 {symbol} 的數據")
        
        df = data.sort_values('open_time').reset_index(drop=True)
        market_data = {symbol: df}
        
        if method == "grid_search":
            optimizer = GridSearchOptimizer(
                strategy_class=strategy_class,
                data=market_data,
                param_space=param_space,
                metric=metric,
                n_jobs=kwargs.get('n_jobs', 1)
            )
            result = optimizer.optimize()
            return result
        
        elif method == "walk_forward":
            analyzer = WalkForwardAnalyzer(
                strategy_class=strategy_class,
                data=market_data,
                param_space=param_space,
                metric=metric,
                n_windows=kwargs.get('n_windows', 5)
            )
            return analyzer.analyze()
        
        else:
            raise ValueError(f"未知優化方法: {method}")
    
    async def compare_strategies(
        self,
        strategies: List[Tuple[Type[BaseStrategy], Dict[str, Any]]],
        symbol: str,
        interval: str = "1h",
        **kwargs
    ) -> List[BacktestResult]:
        """
        比較多個策略的表現
        
        Args:
            strategies: [(策略類別, 參數), ...]
            symbol: 交易對
            interval: K線周期
            
        Returns:
            List[BacktestResult]: 各策略的回測結果
        """
        results = []
        
        for strategy_class, params in strategies:
            print(f"\n📊 測試策略: {strategy_class.__name__}")
            result = await self.run_backtest(
                strategy_class=strategy_class,
                params=params,
                symbol=symbol,
                interval=interval,
                **kwargs
            )
            results.append(result)
        
        # 排序
        results.sort(key=lambda x: x.Sharpe_Ratio, reverse=True)
        
        print("\n" + "=" * 60)
        print("           策略比較結果")
        print("=" * 60)
        for i, result in enumerate(results):
            print(f"{i+1}. {result.Total_Trades} 筆交易 | "
                  f"報酬: {result.total_return_pct:.2f}% | "
                  f"夏普: {result.Sharpe_Ratio:.3f} | "
                  f"勝率: {result.Win_Rate:.2f}%")
        print("=" * 60)
        
        return results
    
    def export_result(
        self,
        result: BacktestResult,
        format: str = "json",
        filepath: Optional[str] = None
    ) -> str:
        """
        匯出回測結果
        
        Args:
            result: 回測結果
            format: 匯出格式 ("json" | "yaml")
            filepath: 檔案路徑（預設自動生成）
            
        Returns:
            str: 匯出的資料
        """
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"backtest_result_{timestamp}.{format}"
        
        # 轉換為字典
        data = {
            'initial_capital': result.initial_capital,
            'final_equity': result.final_equity,
            'total_return': result.total_return,
            'total_return_pct': result.total_return_pct,
            'sharpe_ratio': result.Sharpe_Ratio,
            'max_drawdown': result.Max_Drawdown,
            'max_drawdown_pct': result.Max_Drawdown_Pct,
            'win_rate': result.Win_Rate,
            'profit_factor': result.Profit_Factor,
            'total_trades': result.Total_Trades,
            'winning_trades': result.Winning_Trades,
            'losing_trades': result.Losing_Trades,
            'avg_win': result.Avg_Win,
            'avg_loss': result.Avg_Loss,
            'best_trade': result.Best_Trade,
            'worst_trade': result.Worst_Trade,
            'trades': [
                {
                    'entry_time': t.entry_time.isoformat(),
                    'exit_time': t.exit_time.isoformat(),
                    'symbol': t.symbol,
                    'side': t.side.value,
                    'entry_price': t.entry_price,
                    'exit_price': t.exit_price,
                    'quantity': t.quantity,
                    'pnl': t.pnl,
                    'pnl_pct': t.pnl_pct,
                    'commission': t.commission,
                    'duration': t.duration
                }
                for t in result.trades
            ],
            'equity_curve': [
                {
                    'timestamp': p.timestamp.isoformat(),
                    'equity': p.equity,
                    'drawdown': p.drawdown,
                    'position_value': p.position_value
                }
                for p in result.equity_curve
            ]
        }
        
        if format == "json":
            output = json.dumps(data, indent=2, ensure_ascii=False)
        else:
            output = yaml.dump(data, allow_unicode=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(output)
        
        print(f"✅ 結果已匯出至: {filepath}")
        
        return output
    
    def close(self):
        """關閉客戶端"""
        if self.engine:
            self.engine.close()


# 同步版本的客戶端（包裝異步版本）
class SyncAutoResearchClient:
    """同步版本的 Autoresearch 客戶端"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.async_client = AutoResearchClient(config_path)
    
    def run_backtest(self, **kwargs) -> BacktestResult:
        """同步執行單次回測"""
        return asyncio.run(self.async_client.run_backtest(**kwargs))
    
    def optimize_strategy(self, **kwargs) -> OptimizationResult:
        """同步執行策略優化"""
        return asyncio.run(self.async_client.optimize_strategy(**kwargs))
    
    def compare_strategies(self, **kwargs) -> List[BacktestResult]:
        """同步比較策略"""
        return asyncio.run(self.async_client.compare_strategies(**kwargs))
    
    def close(self):
        """關閉客戶端"""
        self.async_client.close()


# ============ Agent 便捷函數 ============

def create_backtest_task(
    agent_id: str,
    strategy_class: Type[BaseStrategy],
    params: Dict[str, Any],
    symbol: str,
    **kwargs
) -> Dict[str, Any]:
    """
    建立回測任務（供 Agent 调用）
    
    這是一個工廠函數，創建可序列化的任務描述
    然後交由執行器運行
    """
    return {
        'task_id': f"backtest_{agent_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
        'type': 'backtest',
        'strategy_class': strategy_class.__name__,
        'params': params,
        'symbol': symbol,
        'created_at': datetime.now().isoformat(),
        'config': kwargs
    }


# ============ REST API 預留介面 ============

class BacktestAPIServer:
    """
    回測 API 伺服器（預留介面）
    
    當需要提供 HTTP API 時啟用
    """
    
    def __init__(self, client: AutoResearchClient, host: str = "0.0.0.0", port: int = 8080):
        self.client = client
        self.host = host
        self.port = port
        self.app = None
    
    async def start(self):
        """啟動 API 伺服器"""
        from aiohttp import web
        
        self.app = web.Application()
        
        # 路由定義
        self.app.router.add_post('/api/backtest', self.handle_backtest)
        self.app.router.add_post('/api/optimize', self.handle_optimize)
        self.app.router.add_get('/api/result/{task_id}', self.handle_get_result)
        
        runner = web.AppRunner(self.app)
        await runner.setup()
        site = web.TCPSite(runner, self.host, self.port)
        await site.start()
        
        print(f"🌐 回測 API 伺服器已啟動: http://{self.host}:{self.port}")
    
    async def handle_backtest(self, request: web.Request) -> web.Response:
        """處理回測請求"""
        data = await request.json()
        
        # 解析請求
        # TODO: 實現完整的請求解析
        
        result = await self.client.run_backtest(**data)
        
        return web.json_response({
            'status': 'success',
            'result': self._result_to_dict(result)
        })
    
    async def handle_optimize(self, request: web.Request) -> web.Response:
        """處理優化請求"""
        data = await request.json()
        
        result = await self.client.optimize_strategy(**data)
        
        return web.json_response({
            'status': 'success',
            'result': {
                'best_params': result.best_params,
                'best_metrics': result.best_metrics,
                'overfitting_score': result.overfitting_score
            }
        })
    
    async def handle_get_result(self, request: web.Request) -> web.Response:
        """取得歷史結果"""
        task_id = request.match_info['task_id']
        
        # TODO: 實現結果查詢
        
        return web.json_response({
            'status': 'not_implemented',
            'task_id': task_id
        })
    
    def _result_to_dict(self, result: BacktestResult) -> Dict:
        """轉換結果為字典"""
        return {
            'total_return_pct': result.total_return_pct,
            'sharpe_ratio': result.Sharpe_Ratio,
            'max_drawdown_pct': result.Max_Drawdown_Pct,
            'win_rate': result.Win_Rate,
            'total_trades': result.Total_Trades
        }


if __name__ == "__main__":
    # 測試範例
    from backtest_engine import SimpleMovingAverageCrossover
    
    client = SyncAutoResearchClient()
    
    # 同步執行回測
    print("執行測試回測...")
    
    # 注意: 需要網路連接才能下載數據
    # result = client.run_backtest(
    #     strategy_class=SimpleMovingAverageCrossover,
    #     params={'fast_period': 10, 'slow_period': 30},
    #     symbol='BTCUSDT',
    #     interval='1h'
    # )
    # 
    # print(f"夏普值: {result.Sharpe_Ratio}")
    # print(f"總報酬: {result.total_return_pct}%")
    
    print("客戶端初始化成功!")
