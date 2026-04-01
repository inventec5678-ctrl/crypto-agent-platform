"""
回測引擎核心模組
位置: crypto-agent-platform/backtest/backtest_engine.py

功能:
- 歷史數據載入（從幣安 API 取得）
- 策略信號產生
- 部位計算（PnL）
- 資金管理
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import yaml
import asyncio
import aiohttp
import json
import os


class PositionSide(Enum):
    """倉位方向"""
    LONG = "LONG"
    SHORT = "SHORT"
    FLAT = "FLAT"


class OrderType(Enum):
    """訂單類型"""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_LOSS = "STOP_LOSS"
    TAKE_PROFIT = "TAKE_PROFIT"


@dataclass
class Order:
    """訂單資料結構"""
    timestamp: datetime
    symbol: str
    side: PositionSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    filled_price: Optional[float] = None
    fee: float = 0.0
    slippage: float = 0.0
    order_id: str = ""


@dataclass
class Position:
    """部位資料結構"""
    symbol: str
    side: PositionSide
    quantity: float
    entry_price: float
    current_price: float
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    entry_time: Optional[datetime] = None


@dataclass
class Trade:
    """交易記錄"""
    entry_time: datetime
    exit_time: datetime
    symbol: str
    side: PositionSide
    entry_price: float
    exit_price: float
    quantity: float
    pnl: float
    pnl_pct: float
    commission: float
    duration: int  # 秒


@dataclass
class EquityPoint:
    """權益曲線資料點"""
    timestamp: datetime
    equity: float
    drawdown: float
    position_value: float


@dataclass
class BacktestResult:
    """回測結果"""
    initial_capital: float
    final_equity: float
    total_return: float
    total_return_pct: float
    trades: List[Trade]
    equity_curve: List[EquityPoint]
    Sharpe_Ratio: float
    Max_Drawdown: float
    Max_Drawdown_Pct: float
    Win_Rate: float
    Profit_Factor: float
    Total_Trades: int
    Winning_Trades: int
    Losing_Trades: int
    Avg_Win: float
    Avg_Loss: float
    Best_Trade: float
    Worst_Trade: float
    Avg_Trade_Duration: float


class BinanceDataFetcher:
    """幣安歷史數據獲取器"""
    
    BASE_URL = "https://api.binance.com"
    
    def __init__(self):
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def _get_session(self) -> aiohttp.ClientSession:
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()
        return self.session
    
    async def fetch_klines(
        self,
        symbol: str,
        interval: str,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        limit: int = 1000
    ) -> pd.DataFrame:
        """
        獲取K線數據
        
        Args:
            symbol: 交易對，如 'BTCUSDT'
            interval: K線周期，如 '1h', '4h', '1d'
            start_time: 開始時間戳(毫秒)
            end_time: 結束時間戳(毫秒)
            limit: 每次最大獲取數量
        """
        endpoint = f"{self.BASE_URL}/api/v3/klines"
        params = {
            "symbol": symbol.upper(),
            "interval": interval,
            "limit": limit
        }
        if start_time:
            params["startTime"] = start_time
        if end_time:
            params["endTime"] = end_time
        
        session = await self._get_session()
        async with session.get(endpoint, params=params) as response:
            if response.status != 200:
                raise Exception(f"Binance API error: {response.status}")
            data = await response.json()
        
        # 轉換為DataFrame
        df = pd.DataFrame(data, columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades", "taker_buy_base",
            "taker_buy_quote", "ignore"
        ])
        
        # 類型轉換
        numeric_cols = ["open", "high", "low", "close", "volume", "quote_volume"]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
        df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")
        
        return df[["open_time", "close_time", "open", "high", "low", "close", "volume"]]
    
    async def fetch_all_klines(
        self,
        symbol: str,
        interval: str,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """獲取完整歷史數據（自動分頁）"""
        start_ts = int(pd.Timestamp(start_date).timestamp() * 1000)
        end_ts = int(pd.Timestamp(end_date).timestamp() * 1000)
        
        all_data = []
        current_ts = start_ts
        
        while current_ts < end_ts:
            df = await self.fetch_klines(
                symbol=symbol,
                interval=interval,
                start_time=current_ts,
                end_time=end_ts
            )
            
            if df.empty:
                break
            
            # 確保 close_time 存在
            if 'close_time' not in df.columns:
                if 'open_time' in df.columns:
                    df['close_time'] = df['open_time'].shift(-1)
                    df.loc[df.index[-1], 'close_time'] = df['open_time'].iloc[-1] + pd.Timedelta(days=1)
                else:
                    break
            
            all_data.append(df)
            
            # 下一個起始點
            current_ts = int(df["close_time"].iloc[-1].timestamp() * 1000) + 1
            
            # 避免請求過快
            await asyncio.sleep(0.2)
        
        if all_data:
            return pd.concat(all_data, ignore_index=True)
        return pd.DataFrame()
    
    async def close_async(self):
        """關閉會話（異步）"""
        if self.session and not self.session.closed:
            await self.session.close()
    
    def close(self):
        """關閉會話"""
        if self.session and not self.session.closed:
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # 創建新執行緒關閉
                    import concurrent.futures
                    def close_session():
                        new_loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(new_loop)
                        try:
                            new_loop.run_until_complete(self.session.close())
                        finally:
                            new_loop.close()
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        executor.submit(close_session)
                else:
                    loop.run_until_complete(self.session.close())
            except Exception:
                pass


class BacktestEngine:
    """
    回測引擎核心類
    
    用法:
    >>> engine = BacktestEngine(config)
    >>> engine.load_data("BTCUSDT", "1h", "2024-01-01", "2024-12-31")
    >>> engine.set_strategy(my_strategy)
    >>> result = engine.run()
    """
    
    def __init__(self, config_path: str = None):
        """初始化回測引擎"""
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)['backtest']
        else:
            # 預設配置
            self.config = {
                'initial_capital': 10000.0,
                'maker_fee': 0.0002,
                'taker_fee': 0.0004,
                'slippage': 0.0001,
                'start_date': '2024-01-01',
                'end_date': '2024-12-31'
            }
        
        # 引擎狀態
        self.data: Dict[str, pd.DataFrame] = {}
        self.strategy = None
        self.position: Optional[Position] = None
        self.orders: List[Order] = []
        self.trades: List[Trade] = []
        self.equity_curve: List[EquityPoint] = []
        
        # 帳戶狀態
        self.initial_capital = self.config['initial_capital']
        self.cash = self.initial_capital
        self.equity = self.initial_capital
        
        # 風險參數
        self.max_position_size = 0.2
        self.stop_loss = None
        self.take_profit = None
        
        # 數據獲取器
        self.data_fetcher = BinanceDataFetcher()
    
    async def load_data(
        self,
        symbol: str,
        interval: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ):
        """載入歷史數據"""
        start = start_date or self.config.get('start_date', '2024-01-01')
        end = end_date or self.config.get('end_date', '2024-12-31')
        
        print(f"📥 正在下載 {symbol} {interval} 歷史數據...")
        df = await self.data_fetcher.fetch_all_klines(symbol, interval, start, end)
        
        if df.empty:
            raise ValueError(f"無法獲取 {symbol} 的歷史數據")
        
        # 確保資料按時間排序
        df = df.sort_values('open_time').reset_index(drop=True)
        self.data[symbol] = df
        print(f"✅ 已載入 {len(df)} 根K線, 時間範圍: {df['open_time'].min()} ~ {df['open_time'].max()}")
    
    def load_dataframe(self, symbol: str, df: pd.DataFrame):
        """直接載入DataFrame（用於本地數據）"""
        df = df.copy()
        # 確保有 close_time 欄位
        if 'close_time' not in df.columns:
            if 'open_time' in df.columns:
                # 從 open_time 推算 close_time（假设 1d 周期）
                df['close_time'] = df['open_time'].shift(-1)
                df.loc[df.index[-1], 'close_time'] = df['open_time'].iloc[-1] + pd.Timedelta(days=1)
        df = df.sort_values('open_time').reset_index(drop=True)
        self.data[symbol] = df
        print(f"✅ 已載入 {len(df)} 根K線")
    
    def set_strategy(self, strategy):
        """設定交易策略"""
        self.strategy = strategy
        print(f"✅ 策略已設定: {strategy.__class__.__name__}")
    
    def _calculate_position_value(self) -> float:
        """計算持倉的未實現損益"""
        if not self.position or self.position.side == PositionSide.FLAT:
            return 0.0
        
        if self.position.side == PositionSide.LONG:
            return (self.position.current_price - self.position.entry_price) * self.position.quantity
        else:  # SHORT
            return (self.position.entry_price - self.position.current_price) * self.position.quantity
    
    def _calculate_equity(self) -> float:
        """計算總權益 = 初始資金 + 已實現盈虧 + 未實現盈虧"""
        realized_pnl = sum(t.pnl for t in self.trades)
        unrealized_pnl = self._calculate_position_value()
        return self.initial_capital + realized_pnl + unrealized_pnl
    
    def _calculate_drawdown(self) -> Tuple[float, float]:
        """計算當前回撤"""
        peak = max(p.equity for p in self.equity_curve) if self.equity_curve else self.initial_capital
        drawdown = self.equity - peak
        drawdown_pct = (drawdown / peak) * 100 if peak > 0 else 0
        return drawdown, drawdown_pct
    
    def _execute_order(self, order: Order, market_data: Dict[str, pd.DataFrame]) -> Order:
        """執行訂單"""
        symbol_data = market_data[order.symbol]
        current_price = symbol_data['close'].iloc[-1]
        
        order.filled_price = current_price * (1 + order.slippage if order.side == PositionSide.LONG else 1 - order.slippage)
        order.filled_price = order.filled_price * (1 + self.config.get('taker_fee', 0.0004))
        
        # 計算倉位
        order_cost = order.quantity * order.filled_price
        
        if order.side == PositionSide.LONG:
            # 買入
            if self.cash >= order_cost:
                self.cash -= order_cost
                self.position = Position(
                    symbol=order.symbol,
                    side=PositionSide.LONG,
                    quantity=order.quantity,
                    entry_price=order.filled_price,
                    current_price=order.filled_price,
                    unrealized_pnl=0.0,
                    realized_pnl=0.0,
                    entry_time=order.timestamp
                )
        elif order.side == PositionSide.SHORT:
            # 賣空（假設有足夠保證金）
            self.position = Position(
                symbol=order.symbol,
                side=PositionSide.SHORT,
                quantity=order.quantity,
                entry_price=order.filled_price,
                current_price=order.filled_price,
                unrealized_pnl=0.0,
                realized_pnl=0.0,
                entry_time=order.timestamp
            )
            self.cash += order_cost  # 收到賣空資金
        
        elif order.side == PositionSide.FLAT:
            # 平倉
            if self.position:
                self._close_position(order.timestamp, order.filled_price)
        
        return order
    
    def _close_position(self, exit_time: datetime, exit_price: float):
        """平倉"""
        if not self.position:
            return
        
        if self.position.side == PositionSide.LONG:
            pnl = (exit_price - self.position.entry_price) * self.position.quantity
            self.cash += self.position.quantity * exit_price
        else:  # SHORT
            pnl = (self.position.entry_price - exit_price) * self.position.quantity
            self.cash += self.position.quantity * exit_price
        
        # 扣除交易費用
        commission = exit_price * self.position.quantity * self.config.get('taker_fee', 0.0004)
        self.cash -= commission
        
        # 記錄交易
        trade = Trade(
            entry_time=self.position.entry_time,
            exit_time=exit_time,
            symbol=self.position.symbol,
            side=self.position.side,
            entry_price=self.position.entry_price,
            exit_price=exit_price,
            quantity=self.position.quantity,
            pnl=pnl - commission,
            pnl_pct=(pnl - commission) / (self.position.entry_price * self.position.quantity) * 100,
            commission=commission,
            duration=int((exit_time - self.position.entry_time).total_seconds())
        )
        self.trades.append(trade)
        
        self.position = None
    
    def _update_equity(self, timestamp: datetime, market_data: Dict[str, pd.DataFrame]):
        """更新權益曲線"""
        if self.position:
            self.position.current_price = market_data[self.position.symbol]['close'].iloc[-1]
            if self.position.side == PositionSide.LONG:
                self.position.unrealized_pnl = (
                    self.position.current_price - self.position.entry_price
                ) * self.position.quantity
            else:
                self.position.unrealized_pnl = (
                    self.position.entry_price - self.position.current_price
                ) * self.position.quantity
        
        self.equity = self._calculate_equity()
        drawdown, drawdown_pct = self._calculate_drawdown()
        
        self.equity_curve.append(EquityPoint(
            timestamp=timestamp,
            equity=self.equity,
            drawdown=drawdown,
            position_value=self._calculate_position_value()
        ))
    
    async def run(self) -> BacktestResult:
        """
        執行回測
        
        Returns:
            BacktestResult: 回測結果物件
        """
        if not self.data:
            raise ValueError("請先載入數據")
        if not self.strategy:
            raise ValueError("請先設定策略")
        
        print("🚀 開始回測...")
        
        # 合并所有symbol的數據
        all_timestamps = set()
        for symbol, df in self.data.items():
            all_timestamps.update(df['open_time'].tolist())
        
        timestamps = sorted(list(all_timestamps))
        
        # 遍歷每個時間點
        for i, ts in enumerate(timestamps):
            # 準備策略輸入
            market_data = {}
            for symbol, df in self.data.items():
                # 獲取截至當前的歷史數據
                mask = df['open_time'] <= ts
                if mask.any():
                    market_data[symbol] = df[mask].copy()
            
            if not market_data:
                continue
            
            # 執行策略，獲取信號
            signal = self.strategy.generate_signal(market_data)
            
            # 更新倉位狀態
            if self.position:
                self.position.current_price = market_data[self.position.symbol]['close'].iloc[-1]
                
                # 止損/止盈檢查
                pnl_pct = (
                    (self.position.current_price - self.position.entry_price) / self.position.entry_price
                    if self.position.side == PositionSide.LONG else
                    (self.position.entry_price - self.position.current_price) / self.position.entry_price
                )
                
                if self.stop_loss and pnl_pct <= -self.stop_loss:
                    signal = PositionSide.FLAT  # 觸發止損
                elif self.take_profit and pnl_pct >= self.take_profit:
                    signal = PositionSide.FLAT  # 觸發止盈
            
            # 執行信號
            if signal != PositionSide.FLAT and not self.position:
                # 開倉
                symbol = list(market_data.keys())[0]
                max_qty = (self.equity * self.max_position_size) / market_data[symbol]['close'].iloc[-1]
                
                order = Order(
                    timestamp=ts,
                    symbol=symbol,
                    side=signal,
                    order_type=OrderType.MARKET,
                    quantity=max_qty,
                    slippage=self.config.get('slippage', 0.0001)
                )
                self._execute_order(order, market_data)
            
            elif signal == PositionSide.FLAT and self.position:
                # 平倉
                order = Order(
                    timestamp=ts,
                    symbol=self.position.symbol,
                    side=PositionSide.FLAT,
                    order_type=OrderType.MARKET,
                    quantity=self.position.quantity,
                    slippage=self.config.get('slippage', 0.0001)
                )
                self._execute_order(order, market_data)
            
            # 更新權益曲線
            self._update_equity(ts, market_data)
        
        # 計算績效指標
        result = self._calculate_performance()
        
        print(f"✅ 回測完成!")
        print(f"   初始資金: ${self.initial_capital:,.2f}")
        print(f"   最終權益: ${result.final_equity:,.2f}")
        print(f"   總收益: {result.total_return_pct:.2f}%")
        print(f"   夏普值: {result.Sharpe_Ratio:.3f}")
        print(f"   最大回撤: {result.Max_Drawdown_Pct:.2f}%")
        print(f"   勝率: {result.Win_Rate:.2f}%")
        
        return result
    
    def _calculate_performance(self) -> BacktestResult:
        """計算績效指標"""
        from .performance_metrics import PerformanceMetrics
        
        if not self.trades:
            # 沒有交易，回傳基本結果
            return BacktestResult(
                initial_capital=self.initial_capital,
                final_equity=self.equity,
                total_return=self.equity - self.initial_capital,
                total_return_pct=(self.equity - self.initial_capital) / self.initial_capital * 100,
                trades=[],
                equity_curve=self.equity_curve,
                Sharpe_Ratio=0.0,
                Max_Drawdown=0.0,
                Max_Drawdown_Pct=0.0,
                Win_Rate=0.0,
                Profit_Factor=0.0,
                Total_Trades=0,
                Winning_Trades=0,
                Losing_Trades=0,
                Avg_Win=0.0,
                Avg_Loss=0.0,
                Best_Trade=0.0,
                Worst_Trade=0.0,
                Avg_Trade_Duration=0.0
            )
        
        metrics = PerformanceMetrics(self.trades, self.equity_curve, self.initial_capital)
        
        winning_trades = [t for t in self.trades if t.pnl > 0]
        losing_trades = [t for t in self.trades if t.pnl <= 0]
        
        return BacktestResult(
            initial_capital=self.initial_capital,
            final_equity=self.equity,
            total_return=self.equity - self.initial_capital,
            total_return_pct=(self.equity - self.initial_capital) / self.initial_capital * 100,
            trades=self.trades,
            equity_curve=self.equity_curve,
            Sharpe_Ratio=metrics.sharpe_ratio(),
            Max_Drawdown=metrics.max_drawdown(),
            Max_Drawdown_Pct=metrics.max_drawdown_pct(),
            Win_Rate=metrics.win_rate(),
            Profit_Factor=metrics.profit_factor(),
            Total_Trades=len(self.trades),
            Winning_Trades=len(winning_trades),
            Losing_Trades=len(losing_trades),
            Avg_Win=np.mean([t.pnl for t in winning_trades]) if winning_trades else 0.0,
            Avg_Loss=np.mean([t.pnl for t in losing_trades]) if losing_trades else 0.0,
            Best_Trade=max(t.pnl for t in self.trades) if self.trades else 0.0,
            Worst_Trade=min(t.pnl for t in self.trades) if self.trades else 0.0,
            Avg_Trade_Duration=np.mean([t.duration for t in self.trades]) if self.trades else 0.0
        )
    
    def close(self):
        """關閉引擎"""
        self.data_fetcher.close()


# ============ 策略範例 ============

class BaseStrategy:
    """策略基類"""
    
    def generate_signal(self, market_data: Dict[str, pd.DataFrame]) -> PositionSide:
        """
        產生交易信號
        
        Args:
            market_data: 市場數據字典 {symbol: DataFrame}
            
        Returns:
            PositionSide: 交易信號
        """
        raise NotImplementedError("策略必須實作 generate_signal 方法")


class SimpleMovingAverageCrossover(BaseStrategy):
    """均線交叉策略"""
    
    def __init__(self, fast_period: int = 10, slow_period: int = 30):
        self.fast_period = fast_period
        self.slow_period = slow_period
    
    def generate_signal(self, market_data: Dict[str, pd.DataFrame]) -> PositionSide:
        df = list(market_data.values())[0]  # 取第一個symbol
        
        if len(df) < self.slow_period:
            return PositionSide.FLAT
        
        # 計算均線
        fast_ma = df['close'].rolling(window=self.fast_period).mean()
        slow_ma = df['close'].rolling(window=self.slow_period).mean()
        
        # 黃金交叉 vs 死亡交叉
        if fast_ma.iloc[-1] > slow_ma.iloc[-1] and fast_ma.iloc[-2] <= slow_ma.iloc[-2]:
            return PositionSide.LONG
        elif fast_ma.iloc[-1] < slow_ma.iloc[-1] and fast_ma.iloc[-2] >= slow_ma.iloc[-2]:
            return PositionSide.SHORT
        
        return PositionSide.FLAT


# ============ Autoresearch 整合介面 ============

class AutoResearchInterface:
    """
    Autoresearch 整合介面
    預留讓策略 Agent 可以自動呼叫回測
    """
    
    def __init__(self, backtest_dir: str = None):
        self.backtest_dir = backtest_dir or os.path.dirname(__file__)
        self.config_path = os.path.join(self.backtest_dir, "backtest_config.yaml")
        self.engine = BacktestEngine(self.config_path)
    
    async def run_backtest(
        self,
        strategy: BaseStrategy,
        symbol: str,
        interval: str,
        start_date: str,
        end_date: str,
        **kwargs
    ) -> BacktestResult:
        """
        執行回測的標準介面
        
        Args:
            strategy: 交易策略實例
            symbol: 交易對
            interval: K線周期
            start_date: 開始日期
            end_date: 結束日期
            
        Returns:
            BacktestResult: 回測結果
        """
        # 載入數據
        await self.engine.load_data(symbol, interval, start_date, end_date)
        
        # 設定策略
        self.engine.set_strategy(strategy)
        
        # 設定風險參數
        if 'max_position_size' in kwargs:
            self.engine.max_position_size = kwargs['max_position_size']
        if 'stop_loss' in kwargs:
            self.engine.stop_loss = kwargs['stop_loss']
        if 'take_profit' in kwargs:
            self.engine.take_profit = kwargs['take_profit']
        
        # 執行回測
        result = await self.engine.run()
        
        return result
    
    def get_optimization_space(self, strategy_class) -> Dict[str, List]:
        """
        取得策略的優化參數空間
        用於 Grid Search
        """
        # 可根據策略類型動態定義
        return {
            'fast_period': [5, 10, 15, 20],
            'slow_period': [20, 30, 50, 100]
        }


# 便捷函數
async def quick_backtest(strategy: BaseStrategy, symbol: str = "BTCUSDT", 
                         interval: str = "1h", period: str = "30d"):
    """快速回測"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=int(period.replace('d','')))
    
    interface = AutoResearchInterface()
    return await interface.run_backtest(
        strategy=strategy,
        symbol=symbol,
        interval=interval,
        start_date=start_date.strftime('%Y-%m-%d'),
        end_date=end_date.strftime('%Y-%m-%d')
    )


if __name__ == "__main__":
    # 測試範例
    async def test():
        engine = BacktestEngine()
        
        # 使用均線交叉策略
        strategy = SimpleMovingAverageCrossover(fast_period=10, slow_period=30)
        engine.set_strategy(strategy)
        engine.max_position_size = 0.2
        engine.stop_loss = 0.02
        engine.take_profit = 0.05
        
        try:
            result = await engine.run()
            print(f"\n回測結果:")
            print(f"  總收益: {result.total_return_pct:.2f}%")
            print(f"  夏普值: {result.Sharpe_Ratio:.3f}")
            print(f"  最大回撤: {result.Max_Drawdown_Pct:.2f}%")
        finally:
            engine.close()
    
    # asyncio.run(test())
