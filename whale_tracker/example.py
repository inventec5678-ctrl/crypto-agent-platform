#!/usr/bin/env python3
"""
鯨魚追蹤器 - 使用範例

使用 Blockchair API 監控比特幣區塊鏈上的大額轉帳
"""

from whale_tracker import WhaleTracker
from whale_tracker.api_clients import BlockchairClient, MempoolClient, RateLimiter


def main():
    print("=" * 60)
    print("🐋 Whale Tracker - 比特幣鯨魚監控")
    print("=" * 60)

    # 初始化追蹤器（使用 Blockchair）
    tracker = WhaleTracker(threshold_btc=1000.0)

    # 方法1：掃描最近的鯨魚交易
    print("\n📡 掃描最近的大額轉帳...")
    alerts = tracker.scan_recent(limit=50)

    if alerts:
        print(f"\n找到 {len(alerts)} 筆鯨魚警示：\n")
        for alert in alerts:
            print(f"  {alert.alert_message}")
            print(f"    TX: {alert.tx_hash}")
            print(f"    時間: {alert.timestamp}")
            print()
    else:
        print("未發現大額鯨魚轉帳")

    # 方法2：檢查特定交易
    # 替换为实际交易哈希进行测试
    # example_tx = "abc123..."
    # result = tracker.check_transaction(example_tx)
    # if result:
    #     print(f"交易 {example_tx}: {result.alert_message}")


if __name__ == "__main__":
    main()
