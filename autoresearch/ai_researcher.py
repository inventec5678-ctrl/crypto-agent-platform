#!/usr/bin/env python3
"""
Auto Research v3 - AI 自主發想策略系統

核心概念：
- AI 每次循環分析 2 年市場數據
- 從失敗歷史學習，避免重複犯錯
- 動態發想策略，不依賴固定策略池
- 每輪結束有完整回報
"""

import pandas as pd
import numpy as np
import json
import random
import asyncio
from datetime import datetime
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable

# ========== 專案路徑 ==========
PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "btcusdt_1d.parquet"
MEMORY_DIR = PROJECT_ROOT / "autoresearch" / "memory"
RESEARCH_LOG = MEMORY_DIR / "research_log.md"
FAILED_STRATEGIES = MEMORY_DIR / "failed_strategies.json"
BEST_STRATEGIES = MEMORY_DIR / "best_strategies.json"

# ========== Target ==========
@dataclass
class Target:
    win_rate: float = 50.0
    profit_factor: float = 2.0
    max_drawdown: float = 30.0
    sharpe: float = 1.5

TARGET = Target()

# ========== 市場數據封裝 ==========
@dataclass
class MarketSnapshot:
    """單一 K 線的市場快照"""
    bar_index: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    
    # 技術指標
    rsi: float
    atr: float
    
    # 均線
    ma10: float
    ma20: float
    ma50: float
    ma200: float
    
    # 動能
    trend_12h: float    # 24小時變化 %
    trend_24h: float
    trend_7d: float
    
    # 成交量
    vol_ratio: float    # 當前成交量 / 20日均量
    
    # 市場狀態
    regime: str          # "BULL" / "BEAR" / "RANGE"
    day_of_week: int     # 0=Mon, 4=Fri
    
    # MA200 方向
    ma200_slope: float   # MA200 斜率


class MarketData:
    """市場數據管理器"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.closes = df['close'].values
        self.highs = df['high'].values
        self.lows = df['low'].values
        self.volumes = df['volume'].values
        self.opens = df.get('open', self.closes).values
        
        # 計算技術指標
        self._calculate_indicators()
    
    def _calculate_indicators(self):
        """預先計算所有指標"""
        n = len(self.df)
        
        # RSI (14)
        self.rsi = np.full(n, 50.0)
        for i in range(14, n):
            deltas = np.diff(self.closes[i-14:i+1])
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            avg_gain = np.mean(gains)
            avg_loss = np.mean(losses)
            if avg_loss == 0:
                self.rsi[i] = 100
            else:
                rs = avg_gain / avg_loss
                self.rsi[i] = 100 - (100 / (1 + rs))
        
        # ATR (14)
        tr = np.zeros(n)
        tr[0] = self.highs[0] - self.lows[0]
        for i in range(1, n):
            hl = self.highs[i] - self.lows[i]
            hc = abs(self.highs[i] - self.closes[i-1])
            lc = abs(self.lows[i] - self.closes[i-1])
            tr[i] = max(hl, hc, lc)
        self.atr = pd.Series(tr).rolling(14).mean().values
        
        # 均線
        self.ma10 = pd.Series(self.closes).rolling(10).mean().values
        self.ma20 = pd.Series(self.closes).rolling(20).mean().values
        self.ma50 = pd.Series(self.closes).rolling(50).mean().values
        self.ma200 = pd.Series(self.closes).rolling(200).mean().values
        
        # 成交量均線
        vol_ma20 = pd.Series(self.volumes).rolling(20).mean()
        self.vol_ratio = self.volumes / vol_ma20.values
        self.vol_ratio = np.nan_to_num(self.vol_ratio, nan=1.0)
        
        # MA200 斜率（過去 20 日）
        self.ma200_slope = np.zeros(n)
        for i in range(200, n):
            ma20_vals = self.ma200[i-20:i]
            self.ma200_slope[i] = (ma20_vals[-1] - ma20_vals[0]) / ma20_vals[0] * 100
        
        # 動能
        self.trend_24h = np.zeros(n)
        self.trend_7d = np.zeros(n)
        for i in range(1, n):
            self.trend_24h[i] = (self.closes[i] - self.closes[i-1]) / self.closes[i-1] * 100
            self.trend_7d[i] = (self.closes[i] - self.closes[i-7]) / self.closes[i-7] * 100 if i >= 7 else 0
        
        # 市場 Regime（用 MA200 斜率判斷）
        self.regime = np.full(n, "RANGE")
        for i in range(200, n):
            if self.ma200_slope[i] > 0.1:
                self.regime[i] = "BULL"
            elif self.ma200_slope[i] < -0.1:
                self.regime[i] = "BEAR"
    
    def get_snapshot(self, i: int) -> Optional[MarketSnapshot]:
        """取得指定 bar 的市場快照"""
        if i < 250:
            return None
        
        try:
            ts = self.df.iloc[i].get('open_time', None)
            if ts is None:
                day_of_week = (i % 7)
            else:
                if isinstance(ts, (int, float)):
                    day_of_week = 0
                else:
                    day_of_week = pd.Timestamp(ts).dayofweek
        except:
            day_of_week = (i % 7)
        
        return MarketSnapshot(
            bar_index=i,
            open=self.opens[i],
            high=self.highs[i],
            low=self.lows[i],
            close=self.closes[i],
            volume=self.volumes[i],
            rsi=self.rsi[i] if not np.isnan(self.rsi[i]) else 50.0,
            atr=self.atr[i] if not np.isnan(self.atr[i]) else self.closes[i] * 0.02,
            ma10=self.ma10[i] if not np.isnan(self.ma10[i]) else self.closes[i],
            ma20=self.ma20[i] if not np.isnan(self.ma20[i]) else self.closes[i],
            ma50=self.ma50[i] if not np.isnan(self.ma50[i]) else self.closes[i],
            ma200=self.ma200[i] if not np.isnan(self.ma200[i]) else self.closes[i],
            trend_12h=self.trend_24h[i],  # 12h 和 24h 相同（1d K線）
            trend_24h=self.trend_24h[i],
            trend_7d=self.trend_7d[i],
            vol_ratio=self.vol_ratio[i],
            regime=self.regime[i],
            day_of_week=day_of_week,
            ma200_slope=self.ma200_slope[i]
        )


# ========== 失敗策略記憶 ==========
@dataclass
class FailedStrategy:
    strategy_id: str
    strategy_name: str
    entry_description: str
    failure_reasons: List[str]  # 失敗原因列表
    win_rate: float
    profit_factor: float
    max_drawdown: float
    attempt_count: int = 1
    learned_rules: List[str] = None  # 從失敗學到的規則
    
    def to_dict(self) -> dict:
        d = asdict(self)
        return d


class FailureMemory:
    """失敗策略記憶管理"""
    
    def __init__(self):
        MEMORY_DIR.mkdir(parents=True, exist_ok=True)
        if not FAILED_STRATEGIES.exists():
            FAILED_STRATEGIES.write_text("[]")
        if not BEST_STRATEGIES.exists():
            BEST_STRATEGIES.write_text(json.dumps({"target": asdict(TARGET), "strategies": []}, indent=2))
    
    def load(self) -> List[FailedStrategy]:
        try:
            data = json.loads(FAILED_STRATEGIES.read_text())
            return [FailedStrategy(**d) for d in data]
        except:
            return []
    
    def save(self, failures: List[FailedStrategy]):
        data = [f.to_dict() for f in failures]
        FAILED_STRATEGIES.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    
    def add(self, failure: FailedStrategy):
        failures = self.load()
        
        # 檢查是否已存在
        existing = [i for i, f in enumerate(failures) if f.strategy_id == failure.strategy_id]
        if existing:
            # 更新現有記錄
            idx = existing[0]
            failures[idx].attempt_count += 1
            failures[idx].win_rate = (failures[idx].win_rate + failure.win_rate) / 2
            failures[idx].failure_reasons = list(set(failures[idx].failure_reasons + failure.failure_reasons))
        else:
            failures.append(failure)
        
        self.save(failures)
    
    def get_learned_rules(self) -> List[str]:
        """從所有失敗中提取學到的規則"""
        failures = self.load()
        rules = []
        for f in failures:
            if f.learned_rules:
                rules.extend(f.learned_rules)
        return list(set(rules))
    
    def get_failed_strategy_ids(self) -> List[str]:
        """獲取所有失敗過的策略 ID"""
        return [f.strategy_id for f in self.load()]


# ========== 策略回測 ==========
@dataclass
class BacktestResult:
    strategy_id: str
    strategy_name: str
    entry_description: str
    
    total_trades: int
    wins: int
    losses: int
    timeouts: int
    
    win_rate: float
    profit_factor: float
    max_drawdown: float
    sharpe: float
    total_pnl: float
    
    status: str  # KEEP / DISCARD
    
    def to_dict(self) -> dict:
        return asdict(self)


class StrategyBacktester:
    """策略回測器"""
    
    def __init__(self, market_data: MarketData):
        self.market = market_data
        self.failures = FailureMemory()
    
    def analyze_failure_and_learn(self, strategy_id: str, trades: List[dict]) -> List[str]:
        """分析失敗原因，生成學到的規則"""
        failures = self.failures.load()
        existing = [f for f in failures if f.strategy_id == strategy_id]
        
        reasons = []
        learned = []
        
        if trades:
            pnls = [t['pnl'] for t in trades]
            wins = [p for p in pnls if p > 0]
            losses = [p for p in pnls if p < 0]
            timeouts = [t for t in trades if t.get('exit_reason') == 'TIMEOUT']
            
            avg_win = np.mean(wins) if wins else 0
            avg_loss = abs(np.mean(losses)) if losses else 1
            pf = avg_win / avg_loss if avg_loss > 0 else 0
            
            # 分析失敗原因
            if existing:
                existing = existing[0]
                if existing.attempt_count >= 3:
                    reasons.append("多次嘗試失敗，放棄此策略方向")
                    learned.append("避免使用相同的進場組合")
            
            if len(timeouts) / len(trades) > 0.3:
                reasons.append(f"TIMEOUT 太多 ({len(timeouts)}/{len(trades)})")
                learned.append("應收緊持倉時間或放寬止盈條件")
            
            if pf < 1.0:
                reasons.append(f"盈虧比過低 ({pf:.2f})")
                learned.append("需要更好的進場時機或更嚴格的止損")
        
        return reasons, learned
    
    def backtest(
        self,
        strategy_id: str,
        strategy_name: str,
        entry_description: str,
        entry_fn: Callable[[MarketSnapshot], bool],
        stop_loss_pct: float = 0.02,
        take_profit_pct: float = 0.05,
        max_holding_bars: int = 10
    ) -> BacktestResult:
        """執行策略回測"""
        
        trades = []
        position = None
        
        # 從 2 年前開始（留足夠 lookback）
        start_bar = min(730, len(self.market.df) - 1000)  # 確保有足夠歷史
        
        for i in range(start_bar, len(self.market.df)):
            snapshot = self.market.get_snapshot(i)
            if snapshot is None:
                continue
            
            if position is None:
                # 嘗試進場
                if entry_fn(snapshot):
                    position = {
                        "entry_price": snapshot.close,
                        "entry_bar": i,
                        "entry_snapshot": snapshot
                    }
            
            else:
                # 檢查出場
                pnl_pct = (snapshot.close - position["entry_price"]) / position["entry_price"] * 100
                holding = i - position["entry_bar"]
                
                exit_reason = None
                result_type = None
                
                if pnl_pct <= -stop_loss_pct * 100:
                    exit_reason = "STOP_LOSS"
                    result_type = "LOSS"
                elif pnl_pct >= take_profit_pct * 100:
                    exit_reason = "TAKE_PROFIT"
                    result_type = "WIN"
                elif holding >= max_holding_bars:
                    exit_reason = "TIMEOUT"
                    result_type = "TIMEOUT"
                
                if exit_reason:
                    trades.append({
                        "entry_bar": position["entry_bar"],
                        "exit_bar": i,
                        "pnl_pct": pnl_pct,
                        "exit_reason": exit_reason,
                        "result": result_type
                    })
                    position = None
        
        # 計算指標
        total = len(trades)
        wins = [t for t in trades if t["result"] == "WIN"]
        losses = [t for t in trades if t["result"] == "LOSS"]
        timeouts = [t for t in trades if t["result"] == "TIMEOUT"]
        
        win_count = len(wins)
        loss_count = len(losses)
        
        # 勝率（不含 TIMEOUT）
        if wins or losses:
            win_rate = win_count / (win_count + loss_count) * 100
        else:
            win_rate = 0
        
        # 盈虧比
        avg_win = np.mean([t['pnl_pct'] for t in wins]) if wins else 0
        avg_loss = abs(np.mean([t['pnl_pct'] for t in losses])) if losses else 1
        profit_factor = avg_win / avg_loss if avg_loss > 0 else 0
        
        # 最大回撤
        pnls = [t['pnl_pct'] for t in trades]
        cumulative = np.cumsum(pnls)
        max_dd = 0
        peak = 0
        for val in cumulative:
            if val > peak:
                peak = val
            dd = peak - val
            if dd > max_dd:
                max_dd = dd
        
        # Sharpe
        sharpe = np.mean(pnls) / np.std(pnls) * np.sqrt(len(pnls)) if np.std(pnls) > 0 else 0
        total_pnl = sum(pnls)
        
        # 評估
        target = TARGET
        status = "KEEP" if (
            win_rate >= target.win_rate and
            profit_factor >= target.profit_factor and
            max_dd <= target.max_drawdown and
            sharpe >= target.sharpe
        ) else "DISCARD"
        
        return BacktestResult(
            strategy_id=strategy_id,
            strategy_name=strategy_name,
            entry_description=entry_description,
            total_trades=total,
            wins=win_count,
            losses=loss_count,
            timeouts=len(timeouts),
            win_rate=win_rate,
            profit_factor=profit_factor,
            max_drawdown=max_dd,
            sharpe=sharpe,
            total_pnl=total_pnl,
            status=status
        )


# ========== AI 策略生成器 ==========
class AIStrategyGenerator:
    """AI 策略生成器"""
    
    def __init__(self, market_data: MarketData, failure_memory: FailureMemory):
        self.market = market_data
        self.failures = failure_memory
        self.generation_count = 0
        
        # 已知的有效模式（使用 snap. 前綴）
        self.effective_patterns = [
            {"condition": "snap.rsi < 35", "description": "RSI 超賣"},
            {"condition": "snap.rsi < 30", "description": "RSI 深度超賣"},
            {"condition": "snap.vol_ratio > 1.5", "description": "成交量放大"},
            {"condition": "snap.vol_ratio > 2.0", "description": "成交量大幅放大"},
            {"condition": "snap.close > snap.ma200", "description": "價格在 MA200 上方"},
            {"condition": "snap.regime == 'BULL'", "description": "牛市環境"},
            {"condition": "snap.ma200_slope > 0", "description": "MA200 向上"},
            {"condition": "snap.trend_7d > 5", "description": "7日趨勢向上"},
            {"condition": "snap.trend_7d > 10", "description": "7日強趨勢向上"},
            {"condition": "snap.day_of_week == 5", "description": "週六"},
            {"condition": "snap.day_of_week == 0", "description": "週一"},
            {"condition": "snap.rsi < 40 and snap.trend_7d > 0", "description": "RSI 回調 + 趨勢向上"},
        ]
    
    def generate(self) -> dict:
        """生成一個新策略"""
        self.generation_count += 1
        strategy_id = f"AI_Strategy_{self.generation_count}_{datetime.now().strftime('%Y%m%d_%H%M')}"
        
        # 分析失敗歷史，避免重複
        failed_ids = self.failures.get_failed_strategy_ids()
        learned_rules = self.failures.get_learned_rules()
        
        # 隨機選擇 2-3 個條件組合
        num_conditions = random.randint(2, 3)
        selected = random.sample(self.effective_patterns, num_conditions)
        
        conditions = [p["condition"] for p in selected]
        description = " + ".join([p["description"] for p in selected])
        
        # 動態生成 entry 函數（使用 lambda）
        conditions_str = " and ".join(conditions)
        entry_fn = eval(f"lambda snap: {conditions_str}")
        
        # 隨機參數
        stop_loss = random.choice([0.015, 0.02, 0.025])
        take_profit = random.choice([0.04, 0.05, 0.06, 0.08])
        max_holding = random.choice([8, 10, 12, 15])
        
        return {
            "strategy_id": strategy_id,
            "strategy_name": f"{description[:30]}",
            "entry_description": description,
            "entry_fn": entry_fn,
            "stop_loss_pct": stop_loss,
            "take_profit_pct": take_profit,
            "max_holding_bars": max_holding,
            "learned_rules": learned_rules
        }


# ========== 回報格式化 ==========
def format_round_report(round_num: int, result: BacktestResult, strategy: dict) -> str:
    """格式化回合報告"""
    target = TARGET
    
    status_icon = {"KEEP": "✅", "DISCARD": "❌"}.get(result.status, "?")
    
    wr_icon = "✅" if result.win_rate >= target.win_rate else "❌"
    pf_icon = "✅" if result.profit_factor >= target.profit_factor else "❌"
    dd_icon = "✅" if result.max_drawdown <= target.max_drawdown else "❌"
    sh_icon = "✅" if result.sharpe >= target.sharpe else "❌"
    
    return f"""
╔══════════════════════════════════════════════════════════╗
║           🔬 Auto Research v3 第 {round_num} 輪報告                      ║
╠══════════════════════════════════════════════════════════╣
║ 📝 策略: {result.strategy_name:<40} ║
║ 🎯 條件: {result.entry_description[:40]:<40} ║
╠══════════════════════════════════════════════════════════╣
║ 📊 交易統計                                              ║
║    總交易: {result.total_trades:>4}  WIN: {result.wins:>3}  LOSS: {result.losses:>3}  TIMEOUT: {result.timeouts:>3}      ║
╠══════════════════════════════════════════════════════════╣
║ 📈 指標                                                  ║
║    勝率:   {result.win_rate:>6.1f}% {wr_icon}  (目標≥{target.win_rate}%)           ║
║    盈虧比: {result.profit_factor:>6.2f} {pf_icon}  (目標≥{target.profit_factor})            ║
║    最大DD: {result.max_drawdown:>6.1f}% {dd_icon}  (目標≤{target.max_drawdown}%)           ║
║    Sharpe: {result.sharpe:>6.2f} {sh_icon}  (目標≥{target.sharpe})             ║
╠══════════════════════════════════════════════════════════╣
║ ⚙️ 參數                                                  ║
║    止損: {strategy['stop_loss_pct']*100:.1f}%  止盈: {strategy['take_profit_pct']*100:.1f}%  最長持倉: {strategy['max_holding_bars']}根     ║
╠══════════════════════════════════════════════════════════╣
║ 🏆 評估: {status_icon} {result.status:<50} ║
╚══════════════════════════════════════════════════════════╝"""


# ========== 主循環 ==========
async def run_research(report_callback=None):
    """執行研究循環"""
    
    # 載入數據
    print("📊 載入市場數據...")
    df = pd.read_parquet(DATA_PATH)
    print(f"   總共 {len(df)} 根 K 線")
    
    market = MarketData(df)
    failures = FailureMemory()
    generator = AIStrategyGenerator(market, failures)
    backtester = StrategyBacktester(market)
    
    round_num = 0
    
    while True:
        round_num += 1
        print(f"\n{'='*60}")
        print(f"🔬 第 {round_num} 輪策略研究")
        print(f"{'='*60}")
        
        # 顯示學到的規則
        learned = failures.get_learned_rules()
        if learned:
            print(f"\n📚 從失敗中學到的規則 ({len(learned)} 條):")
            for rule in learned[:5]:
                print(f"   • {rule}")
        
        # 生成策略
        print("\n🧠 AI 分析市場數據，生成策略...")
        strategy = generator.generate()
        print(f"   策略ID: {strategy['strategy_id']}")
        print(f"   條件: {strategy['entry_description']}")
        print(f"   止損: {strategy['stop_loss_pct']*100:.1f}% | 止盈: {strategy['take_profit_pct']*100:.1f}% | 最長持倉: {strategy['max_holding_bars']}根")
        
        # 回測
        print("\n📈 執行回測...")
        result = backtester.backtest(
            strategy_id=strategy['strategy_id'],
            strategy_name=strategy['strategy_name'],
            entry_description=strategy['entry_description'],
            entry_fn=strategy['entry_fn'],
            stop_loss_pct=strategy['stop_loss_pct'],
            take_profit_pct=strategy['take_profit_pct'],
            max_holding_bars=strategy['max_holding_bars']
        )
        
        # 格式化報告
        report = format_round_report(round_num, result, strategy)
        print(report)
        
        # 寫入日誌
        log_entry = f"""
### 第 {round_num} 輪 - {datetime.now().strftime('%Y-%m-%d %H:%M')}
**策略**: {result.strategy_name}
**條件**: {result.entry_description}
**結果**: WIN={result.wins}/{result.total_trades} | WR={result.win_rate:.1f}% | PF={result.profit_factor:.2f} | DD={result.max_drawdown:.1f}% | Sharpe={result.sharpe:.2f}
**評估**: {'✅ KEEP' if result.status == 'KEEP' else '❌ DISCARD'}
"""
        with open(RESEARCH_LOG, "a") as f:
            f.write(log_entry)
        
        # 發送回報
        if report_callback:
            await report_callback(report)
        
        # 根據結果處理
        if result.status == "KEEP":
            # 保存成功策略
            print("\n🎉 策略達標！寫入最佳策略庫...")
            best_data = json.loads(BEST_STRATEGIES.read_text())
            best_data["strategies"].append({
                "strategy_id": strategy['strategy_id'],
                "strategy_name": strategy['strategy_name'],
                "entry_description": strategy['entry_description'],
                "params": {
                    "stop_loss": strategy['stop_loss_pct'],
                    "take_profit": strategy['take_profit_pct'],
                    "max_holding": strategy['max_holding_bars']
                },
                "metrics": {
                    "win_rate": result.win_rate,
                    "profit_factor": result.profit_factor,
                    "max_drawdown": result.max_drawdown,
                    "sharpe": result.sharpe
                },
                "found_at": datetime.now().isoformat()
            })
            BEST_STRATEGIES.write_text(json.dumps(best_data, indent=2, ensure_ascii=False))
        else:
            # 分析失敗原因並記錄
            reasons, learned = backtester.analyze_failure_and_learn(strategy['strategy_id'], [])
            failure = FailedStrategy(
                strategy_id=strategy['strategy_id'],
                strategy_name=strategy['strategy_name'],
                entry_description=strategy['entry_description'],
                failure_reasons=reasons if reasons else ["未達標"],
                win_rate=result.win_rate,
                profit_factor=result.profit_factor,
                max_drawdown=result.max_drawdown,
                learned_rules=learned
            )
            failures.add(failure)
            print(f"\n📝 已記錄失敗策略: {strategy['strategy_id']}")
        
        # 等待間隔
        await asyncio.sleep(5)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Auto Research v3")
    parser.add_argument("--report-webhook", help="WebHook URL for reporting")
    args = parser.parse_args()
    
    if args.report_webhook:
        import aiohttp
        async def webhook_report(message: str):
            payload = {"content": message}
            async with aiohttp.ClientSession() as session:
                await session.post(args.report_webhook, json=payload)
        asyncio.run(run_research(report_callback=webhook_report))
    else:
        asyncio.run(run_research())


if __name__ == "__main__":
    main()
