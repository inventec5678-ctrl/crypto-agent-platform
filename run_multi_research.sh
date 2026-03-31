#!/bin/bash
# 並行 Auto Research - 簡單版
# 每個實例獨立運行，共用同一個 research_log.md

cd ~/.openclaw/workspace/crypto-agent-platform

# Kill existing instances first
pkill -f "loop_continuous" 2>/dev/null
sleep 2

# 實例 1: 標準策略（不帶特殊參數）
echo "🚀 啟動實例 1: 標準策略..."
nohup python3 autoresearch/loop_continuous.py --instance-id 1 --log-prefix "[Instance-1]" > /tmp/autorun_1.log 2>&1 &
echo "   PID: $! (查看: tail -f /tmp/autorun_1.log)"

sleep 2

# 實例 2: 高勝率模式（只保留嚴格進場條件）
echo "🚀 啟動實例 2: 高勝率模式..."
nohup python3 autoresearch/loop_continuous.py --instance-id 2 --log-prefix "[Instance-2]" > /tmp/autorun_2.log 2>&1 &
echo "   PID: $! (查看: tail -f /tmp/autorun_2.log)"

sleep 2

# 實例 3: 多時區模式
echo "🚀 啟動實例 3: 多時區模式..."
nohup python3 autoresearch/loop_continuous.py --instance-id 3 --log-prefix "[Instance-3]" > /tmp/autorun_3.log 2>&1 &
echo "   PID: $! (查看: tail -f /tmp/autorun_3.log)"

echo ""
echo "========================================="
echo "✅ 3 個並行研究實例已啟動！"
echo "========================================="
echo ""
echo "查看所有實例狀態:"
echo "  ps aux | grep loop_continuous | grep -v grep"
echo ""
echo "查看特定實例:"
echo "  tail -f /tmp/autorun_1.log"
echo "  tail -f /tmp/autorun_2.log"  
echo "  tail -f /tmp/autorun_3.log"
