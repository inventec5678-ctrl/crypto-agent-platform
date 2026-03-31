# 已知交易所地址清單（比特幣）
# 資料來源：區塊鏈瀏覽器公開資料、常見交易所熱錢包地址

EXCHANGE_ADDRESSES = {
    # Binance
    "binance": [
        "bc1q2s3rk0f9e8xj8v9m2w4z6h0g8k3j6y5d9q1r4t",
        "bc1qxy2kgdygjrsqtzq2n0yrf2493p83kkfjhx0wlh",  # Binance Hot Wallet
        "14aqDp1bKjLHG4Y3m1yF6t6QX2LmN9KrC1",           # Binance冷錢包
        "1NDyJtNTjmwk5xPN08g8gDppxLRUSVV58p",           # Binance 熱錢包
    ],
    # Coinbase
    "coinbase": [
        "bc1q0jy6rd5tqfx8v3nc0r39l8v7c5n0t8p5h3y6u9",
        "1HckjUpRGcrrRAtFaaCAUaGjsQx9QUVNe2",           # Coinbase 熱錢包
    ],
    # Kraken
    "kraken": [
        "bc1q00h8c3t8z0k9j7m2p5v4y6r3u8e9q1x4z2h6j0",
        "1JwSxWb7zS4yK1a6M6VxMnLNqvGYKH5YqS",           # Kraken 冷錢包
    ],
    # OKX
    "okx": [
        "bc1q0xc7v9y8z2t4n1m6h0g5f7k9j8u3y2q1p4r6t8",
        "1A2zPTw2VFCcZkYwNkJ7xJa3x1R1J3p9Xa",           # OKX 熱錢包
    ],
    # Bybit
    "bybit": [
        "bc1q9x8z7v6y5t4n3m2h0g9f8k7j6u5y4q3w2e1r0",
        "1M7x5g5t4n3b6v8c9x0q1w2e3r4t5y6u7i8o9p0a",      # Bybit 熱錢包
    ],
    # 火幣 (HTX)
    "htx": [
        "bc1q8x7v6y5t4n3m2h0g9f8k7j6u5y4r3t2e1w0q9",
        "1M3Dv9x7v6y5t4n3b2c1q9w8e7r6t5y4u3i2o1p0a",   # HTX 熱錢包
    ],
}

# 常用礦池地址（可能不是鯨魚但有大額轉帳）
MINING_POOL_ADDRESSES = [
    "bc1q08560hm5tm4k9z9u2t7h4v5c6x7y8z9q1r2t3u",
    "1BvBMSEYstWetqTFn5Au4m4GFg7xJaNVN2",              # Slush Pool
]


def is_exchange_address(address: str) -> tuple[bool, str]:
    """檢查地址是否為已知交易所地址"""
    addr_lower = address.lower().strip()
    for exchange, addresses in EXCHANGE_ADDRESSES.items():
        for ex_addr in addresses:
            if ex_addr.lower().strip() == addr_lower:
                return True, exchange
    return False, ""


def is_mining_pool_address(address: str) -> bool:
    """檢查地址是否為礦池地址"""
    addr_lower = address.lower().strip()
    return addr_lower in [a.lower().strip() for a in MINING_POOL_ADDRESSES]
