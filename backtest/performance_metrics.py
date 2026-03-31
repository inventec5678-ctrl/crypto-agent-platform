"""
績效指標計算模組
位置: crypto-agent-platform/backtest/performance_metrics.py

功能:
- 計算夏普值（Sharpe Ratio）
- 計算最大回撤（Max Drawdown）
- 計算勝率（Win Rate）
- 計算盈虧比（Profit Factor）
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass


@dataclass
class PerformanceMetrics:
    """
    績效指標計算類
    
    Attributes:
        trades: 交易記錄列表
        equity_curve: 權益曲線數據
        initial_capital: 初始資金
    """
    trades: List
    equity_curve: List
    initial_capital: float
    
    def __post_init__(self):
        """初始化後處理"""
        self.returns = self._calculate_returns()
        self.daily_returns = self._calculate_daily_returns()
    
    def _calculate_returns(self) -> np.ndarray:
        """計算收益率序列"""
        if not self.equity_curve or len(self.equity_curve) < 2:
            return np.array([])
        
        equities = [p.equity for p in self.equity_curve]
        returns = np.diff(equities) / equities[:-1]
        return returns
    
    def _calculate_daily_returns(self) -> pd.Series:
        """計算日收益率"""
        if not self.equity_curve:
            return pd.Series()
        
        df = pd.DataFrame([
            {'timestamp': p.timestamp, 'equity': p.equity} 
            for p in self.equity_curve
        ])
        
        df['date'] = df['timestamp'].dt.date
        daily = df.groupby('date')['equity'].last()
        daily_returns = daily.pct_change().dropna()
        
        return daily_returns
    
    def sharpe_ratio(self, risk_free_rate: float = 0.0, periods_per_year: int = 8760) -> float:
        """
        計算夏普值 (Sharpe Ratio)
        
        夏普值 = (平均報酬率 - 無風險利率) / 報酬率標準差
        
        Args:
            risk_free_rate: 無風險利率（年化）
            periods_per_year: 每年期數
                - 1小時K線: 8760
                - 日K線: 365
                - 15分鐘K線: 35040
        
        Returns:
            float: 夏普值
        """
        if len(self.returns) < 2:
            return 0.0
        
        excess_returns = self.returns - (risk_free_rate / periods_per_year)
        
        if np.std(excess_returns) == 0:
            return 0.0
        
        sharpe = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(periods_per_year)
        
        return float(sharpe)
    
    def sortino_ratio(self, risk_free_rate: float = 0.0, periods_per_year: int = 8760) -> float:
        """
        計算索提諾值 (Sortino Ratio)
        
        索提諾值 = (平均報酬率 - 無風險利率) / 下行標準差
        
        只考慮負收益的波動性
        """
        if len(self.returns) < 2:
            return 0.0
        
        excess_returns = self.returns - (risk_free_rate / periods_per_year)
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0 or np.std(downside_returns) == 0:
            return 0.0
        
        sortino = np.mean(excess_returns) / np.std(downside_returns) * np.sqrt(periods_per_year)
        
        return float(sortino)
    
    def max_drawdown(self) -> float:
        """
        計算最大回撤 (Max Drawdown)
        
        Returns:
            float: 最大回撤金額
        """
        if not self.equity_curve:
            return 0.0
        
        peak = self.initial_capital
        max_dd = 0.0
        
        for point in self.equity_curve:
            if point.equity > peak:
                peak = point.equity
            drawdown = peak - point.equity
            if drawdown > max_dd:
                max_dd = drawdown
        
        return float(max_dd)
    
    def max_drawdown_pct(self) -> float:
        """
        計算最大回撤百分比 (Max Drawdown %)
        
        Returns:
            float: 最大回撤百分比
        """
        if not self.equity_curve:
            return 0.0
        
        peak = self.initial_capital
        max_dd_pct = 0.0
        
        for point in self.equity_curve:
            if point.equity > peak:
                peak = point.equity
            drawdown_pct = (peak - point.equity) / peak * 100 if peak > 0 else 0
            if drawdown_pct > max_dd_pct:
                max_dd_pct = drawdown_pct
        
        return float(max_dd_pct)
    
    def max_drawdown_duration(self) -> int:
        """
        計算最大回撤持續時間（天）
        
        Returns:
            int: 最大回撤持續天數
        """
        if not self.equity_curve:
            return 0
        
        peak = self.initial_capital
        peak_time = self.equity_curve[0].timestamp if self.equity_curve else None
        max_duration = 0
        current_duration = 0
        
        for point in self.equity_curve:
            if point.equity >= peak:
                peak = point.equity
                peak_time = point.timestamp
                current_duration = 0
            else:
                current_duration = (point.timestamp - peak_time).days
            
            if current_duration > max_duration:
                max_duration = current_duration
        
        return max_duration
    
    def win_rate(self) -> float:
        """
        計算勝率 (Win Rate)
        
        Returns:
            float: 勝率百分比
        """
        if not self.trades:
            return 0.0
        
        winning_trades = sum(1 for t in self.trades if t.pnl > 0)
        total_trades = len(self.trades)
        
        return float(winning_trades / total_trades * 100) if total_trades > 0 else 0.0
    
    def profit_factor(self) -> float:
        """
        計算盈虧比 (Profit Factor)
        
        Profit Factor = 總獲利 / 總虧損
        
        Returns:
            float: 盈虧比
        """
        if not self.trades:
            return 0.0
        
        gross_profit = sum(t.pnl for t in self.trades if t.pnl > 0)
        gross_loss = abs(sum(t.pnl for t in self.trades if t.pnl < 0))
        
        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0.0
        
        return float(gross_profit / gross_loss)
    
    def expectancy(self) -> float:
        """
        計算期望值 (Expectancy)
        
        期望值 = (勝率 × 平均獲利) - (敗率 × 平均虧損)
        
        Returns:
            float: 每筆交易期望收益
        """
        if not self.trades:
            return 0.0
        
        winning_trades = [t for t in self.trades if t.pnl > 0]
        losing_trades = [t for t in self.trades if t.pnl <= 0]
        
        win_rate = len(winning_trades) / len(self.trades)
        avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
        avg_loss = abs(np.mean([t.pnl for t in losing_trades])) if losing_trades else 0
        
        return win_rate * avg_win - (1 - win_rate) * avg_loss
    
    def expectancy_ratio(self) -> float:
        """
        計算期望值比率 (Expectancy Ratio)
        
        期望值比率 = 期望值 / 初始資金
        """
        return self.expectancy() / self.initial_capital * 100
    
    def recovery_factor(self) -> float:
        """
        計算恢復因子 (Recovery Factor)
        
        Recovery Factor = 總淨利 / 最大回撤
        """
        total_net_profit = sum(t.pnl for t in self.trades)
        max_dd = self.max_drawdown()
        
        if max_dd == 0:
            return float('inf') if total_net_profit > 0 else 0.0
        
        return float(total_net_profit / max_dd)
    
    def payoff_ratio(self) -> float:
        """
        計算獲利虧損比 (Payoff Ratio)
        
        Payoff Ratio = 平均獲利 / 平均虧損
        """
        winning_trades = [t for t in self.trades if t.pnl > 0]
        losing_trades = [t for t in self.trades if t.pnl <= 0]
        
        avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
        avg_loss = abs(np.mean([t.pnl for t in losing_trades])) if losing_trades else 0
        
        if avg_loss == 0:
            return float('inf') if avg_win > 0 else 0.0
        
        return float(avg_win / avg_loss)
    
    def calmar_ratio(self) -> float:
        """
        卡爾馬比率 (Calmar Ratio)
        
        Calmar Ratio = 年化報酬率 / 最大回撤
        """
        annual_return = self.annual_return()
        max_dd = self.max_drawdown()
        
        if max_dd == 0:
            return 0.0
        
        return float(annual_return / max_dd)
    
    def annual_return(self) -> float:
        """
        計算年化報酬率
        """
        if not self.equity_curve or len(self.equity_curve) < 2:
            return 0.0
        
        start_time = self.equity_curve[0].timestamp
        end_time = self.equity_curve[-1].timestamp
        
        days = (end_time - start_time).days
        if days == 0:
            return 0.0
        
        total_return = (self.equity_curve[-1].equity - self.initial_capital) / self.initial_capital
        
        # 年化
        annual = (1 + total_return) ** (365 / days) - 1
        
        return float(annual * 100)
    
    def volatility(self, periods_per_year: int = 8760) -> float:
        """
        計算波動率 (Volatility)
        
        Returns:
            float: 年化波動率
        """
        if len(self.returns) < 2:
            return 0.0
        
        return float(np.std(self.returns) * np.sqrt(periods_per_year) * 100)
    
    def downside_deviation(self) -> float:
        """
        計算下行偏差 (Downside Deviation)
        """
        if len(self.returns) < 2:
            return 0.0
        
        downside_returns = self.returns[self.returns < 0]
        
        if len(downside_returns) == 0:
            return 0.0
        
        return float(np.std(downside_returns) * 100)
    
    def skewness(self) -> float:
        """
        計算收益分布偏度 (Skewness)
        """
        if len(self.returns) < 3:
            return 0.0
        
        return float(pd.Series(self.returns).skew())
    
    def kurtosis(self) -> float:
        """
        計算收益分布峰度 (Kurtosis)
        """
        if len(self.returns) < 4:
            return 0.0
        
        return float(pd.Series(self.returns).kurtosis())
    
    def var(self, confidence: float = 0.95) -> float:
        """
        計算 Value at Risk (VaR)
        
        Args:
            confidence: 信心水準（如 0.95 = 95%）
        
        Returns:
            float: VaR 值
        """
        if len(self.returns) < 2:
            return 0.0
        
        return float(np.percentile(self.returns, (1 - confidence) * 100))
    
    def cvar(self, confidence: float = 0.95) -> float:
        """
        計算 Conditional VaR (CVaR) / Expected Shortfall
        
        Returns:
            float: CVaR 值
        """
        if len(self.returns) < 2:
            return 0.0
        
        var = self.var(confidence)
        return float(np.mean(self.returns[self.returns <= var]))
    
    def get_all_metrics(self) -> Dict[str, float]:
        """
        取得所有績效指標
        
        Returns:
            Dict[str, float]: 指標字典
        """
        return {
            'Total Return (%)': self.total_return(),
            'Annual Return (%)': self.annual_return(),
            'Sharpe Ratio': self.sharpe_ratio(),
            'Sortino Ratio': self.sortino_ratio(),
            'Max Drawdown ($)': self.max_drawdown(),
            'Max Drawdown (%)': self.max_drawdown_pct(),
            'Max Drawdown Duration (days)': self.max_drawdown_duration(),
            'Win Rate (%)': self.win_rate(),
            'Profit Factor': self.profit_factor(),
            'Expectancy ($)': self.expectancy(),
            'Expectancy Ratio (%)': self.expectancy_ratio(),
            'Payoff Ratio': self.payoff_ratio(),
            'Recovery Factor': self.recovery_factor(),
            'Calmar Ratio': self.calmar_ratio(),
            'Volatility (%)': self.volatility(),
            'Downside Deviation (%)': self.downside_deviation(),
            'Skewness': self.skewness(),
            'Kurtosis': self.kurtosis(),
            f'VaR ({{95%}})': self.var(0.95),
            f'CVaR ({{95%}})': self.cvar(0.95),
        }
    
    def total_return(self) -> float:
        """計算總報酬率"""
        if not self.trades:
            final_equity = self.equity_curve[-1].equity if self.equity_curve else self.initial_capital
        else:
            final_equity = sum(t.pnl for t in self.trades) + self.initial_capital
        
        return float((final_equity - self.initial_capital) / self.initial_capital * 100)
    
    def generate_report(self) -> str:
        """
        生成績效報告
        
        Returns:
            str: 格式化的報告文字
        """
        metrics = self.get_all_metrics()
        
        report = []
        report.append("=" * 60)
        report.append("              績效指標報告")
        report.append("=" * 60)
        report.append("")
        
        report.append("【收益指標】")
        report.append(f"  總報酬率:        {metrics['Total Return (%)']:.2f}%")
        report.append(f"  年化報酬率:      {metrics['Annual Return (%)']:.2f}%")
        report.append("")
        
        report.append("【風險指標】")
        report.append(f"  夏普值:          {metrics['Sharpe Ratio']:.3f}")
        report.append(f"  索提諾值:        {metrics['Sortino Ratio']:.3f}")
        report.append(f"  卡爾馬比率:      {metrics['Calmar Ratio']:.3f}")
        report.append(f"  最大回撤:        ${metrics['Max Drawdown ($)']:.2f} ({metrics['Max Drawdown (%)']:.2f}%)")
        report.append(f"  回撤持續:        {metrics['Max Drawdown Duration (days)']:.0f} 天")
        report.append(f"  波動率:          {metrics['Volatility (%)']:.2f}%")
        report.append("")
        
        report.append("【交易指標】")
        report.append(f"  勝率:            {metrics['Win Rate (%)']:.2f}%")
        report.append(f"  盈虧比:          {metrics['Profit Factor']:.3f}")
        report.append(f"  獲利虧損比:      {metrics['Payoff Ratio']:.3f}")
        report.append(f"  期望值:          ${metrics['Expectancy ($)']:.2f}")
        report.append(f"  恢復因子:        {metrics['Recovery Factor']:.3f}")
        report.append("")
        
        report.append("【分布指標】")
        report.append(f"  偏度:            {metrics['Skewness']:.3f}")
        report.append(f"  峰度:            {metrics['Kurtosis']:.3f}")
        report.append(f"  VaR (95%):       {metrics['VaR (95%)']:.4f}")
        report.append(f"  CVaR (95%):      {metrics['CVaR (95%)']:.4f}")
        report.append("")
        report.append("=" * 60)
        
        return "\n".join(report)


# 便捷函數
def quick_metrics(trades: List, equity_curve: List, initial_capital: float) -> Dict[str, float]:
    """快速計算績效指標"""
    metrics = PerformanceMetrics(trades, equity_curve, initial_capital)
    return metrics.get_all_metrics()


if __name__ == "__main__":
    # 測試範例
    from dataclasses import dataclass
    from datetime import datetime, timedelta
    import random
    
    @dataclass
    class MockTrade:
        entry_time: datetime
        exit_time: datetime
        pnl: float
    
    @dataclass
    class MockEquity:
        timestamp: datetime
        equity: float
    
    # 模擬數據
    trades = []
    equity = 10000
    for i in range(50):
        trades.append(MockTrade(
            entry_time=datetime.now() - timedelta(days=i*2),
            exit_time=datetime.now() - timedelta(days=i*2-1),
            pnl=random.uniform(-100, 200)
        ))
        equity += trades[-1].pnl
    
    equity_curve = [
        MockEquity(timestamp=datetime.now() - timedelta(days=i), equity=10000 + i*50)
        for i in range(100)
    ]
    
    # 計算指標
    metrics = PerformanceMetrics(trades, equity_curve, 10000)
    print(metrics.generate_report())
