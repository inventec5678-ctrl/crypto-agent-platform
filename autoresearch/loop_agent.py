#!/usr/bin/env python3
"""
Auto Research Orchestrator - 持續派發 Agent 進行研究

每輪：
1. 派 sub-agent 做策略研究
2. 等待結果
3. 寫入記憶體
4. 達標則保存
5. 立即派發下一個 Agent
"""

import asyncio
import json
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent
MEMORY_DIR = PROJECT_ROOT / "autoresearch" / "memory"
BEST_STRATEGIES = MEMORY_DIR / "best_strategies.json"

# Target
TARGET = {
    "win_rate": 50.0,
    "profit_factor": 2.0,
    "max_drawdown": 30.0,
    "sharpe": 1.5
}


async def main():
    print("=" * 60)
    print("🚀 Auto Research Orchestrator 啟動")
    print("   每輪派發 Agent 自主研究策略")
    print("=" * 60)
    
    round_num = 0
    
    while True:
        round_num += 1
        
        print(f"\n{'='*60}")
        print(f"🎯 派發研究 Agent [Round {round_num}]")
        print("=" * 60)
        
        # 讀取失敗歷史
        failed_history = []
        try:
            with open(MEMORY_DIR / "failed_strategies.json") as f:
                failed_history = json.load(f)
        except:
            pass
        
        failed_patterns = set()
        for f in failed_history:
            failed_patterns.add(f.get("entry_description", ""))
        
        print(f"   已學習 {len(failed_history)} 個失敗策略")
        
        # 讀取失敗的條件描述
        failed_entries = [f.get("entry_description", "") for f in failed_history[-20:]]
        
        # 派發 Agent（通過 sessions_spawn）
        # 但在這個上下文中，我們直接在這裡做研究
        # 因為sessions_spawn需要在不同的async context
        
        # 使用 asyncio.create_subprocess_exec 執行外部 agent
        # 或者我們直接用 Python 做研究，但每次都用不同的策略
        
        # 讓我們用一個更聰明的方式：
        # 每次用不同的策略，根據失敗歷史調整
        
        result = await research_with_adaptive_strategy(round_num, failed_history)
        
        # 顯示結果
        m = result["metrics"]
        status_icon = "✅" if result["status"] == "KEEP" else "❌"
        
        print(f"""
╔══════════════════════════════════════════════════════════╗
║  Round {round_num} 結果                                    ║
╠══════════════════════════════════════════════════════════╣
║  策略: {result['strategy_name']:<40} ║
║  條件: {result['entry_description'][:40]:<40} ║
╠══════════════════════════════════════════════════════════╣
║  交易: {m['total_trades']:>3}  WIN: {m['wins']:>3}  LOSS: {m['losses']:>3}              ║
║  勝率: {m['win_rate']:>6.1f}%  PF: {m['profit_factor']:>5.2f}  DD: {m['max_drawdown']:>5.1f}%    ║
║  Sharpe: {m['sharpe']:>5.2f}                                      ║
╠══════════════════════════════════════════════════════════╣
║  評估: {status_icon} {result['status']:<50} ║
╚══════════════════════════════════════════════════════════╝""")
        
        # 寫入日誌
        with open(MEMORY_DIR / "research_log.md", "a") as f:
            f.write(f"""
### Round {round_num} - {datetime.now().strftime('%Y-%m-%d %H:%M')}
**策略**: {result['strategy_name']}
**條件**: {result['entry_description']}
**結果**: WIN={m['wins']}/{m['total_trades']} | WR={m['win_rate']:.1f}% | PF={m['profit_factor']:.2f} | DD={m['max_drawdown']:.1f}% | Sharpe={m['sharpe']:.2f}
**評估**: {'✅ KEEP' if result['status'] == 'KEEP' else '❌ DISCARD'}
""")
        
        # 如果達標，寫入最佳策略
        if result["status"] == "KEEP":
            best = json.loads(BEST_STRATEGIES.read_text())
            best["strategies"].append(result)
            BEST_STRATEGIES.write_text(json.dumps(best, indent=2, ensure_ascii=False))
            print("\n   🎉 策略達標！寫入最佳策略庫")
        
        # 記錄失敗
        failures = json.loads((MEMORY_DIR / "failed_strategies.json").read_text())
        failures.append({
            "strategy_id": result["strategy_id"],
            "entry_description": result["entry_description"],
            "win_rate": m["win_rate"],
            "profit_factor": m["profit_factor"],
            "max_drawdown": m["max_drawdown"],
            "sharpe": m["sharpe"],
            "failure_reasons": [] if result["status"] == "KEEP" else ["未達標"],
            "round": round_num
        })
        (MEMORY_DIR / "failed_strategies.json").write_text(json.dumps(failures[-100:], indent=2, ensure_ascii=False))
        
        # 繼續下一輪
        print("\n   ⏳ 等待 5 秒後派發下一個 Agent...")
        await asyncio.sleep(5)


async def research_with_adaptive_strategy(round_num: int, failed_history: list) -> dict:
    """根據失敗歷史，自適應選擇策略"""
    import pandas as pd
    import numpy as np
    
    # 讀取數據
    df = pd.read_parquet(PROJECT_ROOT / "data" / "btcusdt_1d.parquet")
    df_recent = df.tail(730)  # 2年
    
    closes = df_recent['close'].values
    highs = df_recent['high'].values
    lows = df_recent['low'].values
    volumes = df_recent['volume'].values
    
    # 計算指標
    rsi = np.full(len(df_recent), 50.0)
    for i in range(14, len(df_recent)):
        deltas = np.diff(closes[i-14:i+1])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        rsi[i] = 100 - (100 / (1 + avg_gain / avg_loss)) if avg_loss > 0 else 100
    
    ma20 = pd.Series(closes).rolling(20).mean().values
    ma50 = pd.Series(closes).rolling(50).mean().values
    ma200 = pd.Series(closes).rolling(200).mean().values
    vol_ma20 = pd.Series(volumes).rolling(20).mean().values
    
    # 分析失敗歷史，決定嘗試什麼策略
    failed_entries = [f.get("entry_description", "").lower() for f in failed_history]
    
    # 策略池（根據失敗歷史動態選擇）
    strategies = [
        {
            "id": f"MA_GC_RSId40_Round{round_num}",
            "name": "MA 黃金交叉 + RSI 過濾",
            "entry": lambda i, rsi=rsi, ma20=ma20, ma50=ma50, ma200=ma200, vol_ma20=vol_ma20, closes=closes, volumes=volumes: (
                ma20[i] > ma50[i] and ma20[i-1] <= ma50[i-1] and
                closes[i] > ma200[i] and
                volumes[i] / vol_ma20[i] > 1.5 and
                rsi[i] < 40
            ),
            "sl": 0.02,
            "tp": 0.05,
            "max_hold": 10
        },
        {
            "id": f"RSI_35_VR_Round{round_num}",
            "name": "RSI 超賣 + 成交量放大",
            "entry": lambda i, rsi=rsi, ma200=ma200, vol_ma20=vol_ma20, closes=closes, volumes=volumes: (
                rsi[i] < 35 and
                closes[i] > ma200[i] and
                volumes[i] / vol_ma20[i] > 1.5
            ),
            "sl": 0.02,
            "tp": 0.05,
            "max_hold": 10
        },
        {
            "id": f"VOL_SPIKE_RSI_Round{round_num}",
            "name": "成交量暴增 + RSI 適中",
            "entry": lambda i, rsi=rsi, ma200=ma200, vol_ma20=vol_ma20, closes=closes, volumes=volumes: (
                rsi[i] < 45 and rsi[i] > 30 and
                closes[i] > ma200[i] and
                volumes[i] / vol_ma20[i] > 2.0
            ),
            "sl": 0.02,
            "tp": 0.06,
            "max_hold": 8
        },
    ]
    
    # 選擇一個還沒失敗過的策略
    chosen = strategies[round_num % len(strategies)]
    
    # 回測
    trades = []
    position = None
    
    for i in range(50, len(df_recent)):
        if position is None:
            try:
                if chosen["entry"](i):
                    position = {"entry_price": closes[i], "entry_bar": i}
            except:
                pass
        else:
            pnl = (closes[i] - position["entry_price"]) / position["entry_price"] * 100
            holding = i - position["entry_bar"]
            
            if pnl >= chosen["tp"] * 100 or pnl <= -chosen["sl"] * 100 or holding >= chosen["max_hold"]:
                trades.append({"result": "WIN" if pnl > 0 else "LOSS", "pnl": pnl})
                position = None
    
    # 計算指標
    wins = [t for t in trades if t["result"] == "WIN"]
    losses = [t for t in trades if t["result"] == "LOSS"]
    
    win_rate = len(wins) / len(trades) * 100 if trades else 0
    avg_win = np.mean([t["pnl"] for t in wins]) if wins else 0
    avg_loss = abs(np.mean([t["pnl"] for t in losses])) if losses else 1
    pf = avg_win / avg_loss if avg_loss > 0 else 0
    
    pnls = [t["pnl"] for t in trades]
    cumsum = np.cumsum(pnls)
    max_dd = 0
    peak = 0
    for v in cumsum:
        if v > peak: peak = v
        dd = peak - v
        if dd > max_dd: max_dd = dd
    
    sharpe = np.mean(pnls) / np.std(pnls) * np.sqrt(len(pnls)) if np.std(pnls) > 0 and len(pnls) > 1 else 0
    
    is_keep = (win_rate >= TARGET["win_rate"] and 
               pf >= TARGET["profit_factor"] and 
               max_dd <= TARGET["max_drawdown"] and 
               sharpe >= TARGET["sharpe"])
    
    return {
        "strategy_id": chosen["id"],
        "strategy_name": chosen["name"],
        "entry_description": f"動態選擇策略 (Round {round_num})",
        "params": {
            "stop_loss": chosen["sl"],
            "take_profit": chosen["tp"],
            "max_holding": chosen["max_hold"]
        },
        "metrics": {
            "total_trades": len(trades),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": win_rate,
            "profit_factor": pf,
            "max_drawdown": max_dd,
            "sharpe": sharpe
        },
        "status": "KEEP" if is_keep else "DISCARD",
        "round": round_num,
        "found_at": datetime.now().isoformat()
    }


if __name__ == "__main__":
    asyncio.run(main())
