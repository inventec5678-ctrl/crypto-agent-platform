#!/bin/bash
# 持續成交量策略研究腳本
# 參考 Karpathy AutoResearch 概念

cd ~/.openclaw/workspace/crypto-agent-platform

LOG_FILE="autoresearch/research_loop.log"
RESULTS_FILE="autoresearch/research_results.tsv"

# 初始化結果文件
if [ ! -f "$RESULTS_FILE" ]; then
    echo -e "timestamp\tbest_win_rate\tbest_sharpe\tbest_params\tstatus" > "$RESULTS_FILE"
fi

echo "$(date): 🚀 開始持續成交量策略研究" | tee -a "$LOG_FILE"

# 持續迴圈
iteration=0
while true; do
    iteration=$((iteration + 1))
    echo "$(date): === 第 $iteration 次研究 ===" | tee -a "$LOG_FILE"
    
    # 運行研究腳本
    python3 simple_volume_research.py >> "$LOG_FILE" 2>&1
    
    # 檢查是否找到好策略
    # 這裡可以添加邏輯來解析結果並更新 factor_library.json
    
    echo "$(date): ✅ 研究完成，冷卻 30 分鐘..." | tee -a "$LOG_FILE"
    
    # 冷卻 30 分鐘（可以調整）
    sleep 1800
done
