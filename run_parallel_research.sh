#!/bin/bash
# 並行 Auto Research 腳本
# 啟動多個實例，同時搜索不同策略

cd ~/.openclaw/workspace/crypto-agent-platform

# 實例 1: 趨勢策略
echo "啟動實例 1: 趨勢策略..."
nohup python3 -c "
import sys
sys.path.insert(0, '.')
from autoresearch.loop_continuous import StrategyAnalyzer, MemoryManager

# 只保留趨勢相關策略
memory = MemoryManager()
with open('autoresearch/memory/strategy_pool_1.txt', 'w') as f:
    f.write('TrendVolume\nHighLow\nTrendContinuation\nGoldenCross\nMeanReversion\n')

exec(open('autoresearch/loop_continuous.py').read())
" > /tmp/autorun_p1.log 2>&1 &
PID1=$!
echo "實例 1 PID: $PID1"

# 實例 2: 波動率策略
echo "啟動實例 2: 波動率策略..."
nohup python3 -c "
import sys
sys.path.insert(0, '.')
from autoresearch.loop_continuous import StrategyAnalyzer, MemoryManager

# 只保留波動率相關策略
memory = MemoryManager()
with open('autoresearch/memory/strategy_pool_2.txt', 'w') as f:
    f.write('Volatility\nLowVol\nPumpAlert\nCrisis\n')

exec(open('autoresearch/loop_continuous.py').read())
" > /tmp/autorun_p2.log 2>&1 &
PID2=$!
echo "實例 2 PID: $PID2"

# 實例 3: 時間效應策略
echo "啟動實例 3: 時間效應..."
nohup python3 -c "
import sys
sys.path.insert(0, '.')
from autoresearch.loop_continuous import StrategyAnalyzer, MemoryManager

# 只保留時間效應策略
memory = MemoryManager()
with open('autoresearch/memory/strategy_pool_3.txt', 'w') as f:
    f.write('Weekend\nMonday\nWeekday\n')

exec(open('autoresearch/loop_continuous.py').read())
" > /tmp/autorun_p3.log 2>&1 &
PID3=$!
echo "實例 3 PID: $PID3"

echo ""
echo "==================================="
echo "並行研究已啟動！"
echo "實例 1 PID: $PID1 (趨勢策略)"
echo "實例 2 PID: $PID2 (波動率策略)"
echo "實例 3 PID: $PID3 (時間效應)"
echo "==================================="
echo ""
echo "查看日誌:"
echo "  tail -f /tmp/autorun_p1.log"
echo "  tail -f /tmp/autorun_p2.log"
echo "  tail -f /tmp/autorun_p3.log"
