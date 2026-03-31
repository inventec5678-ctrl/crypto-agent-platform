"""Mock data for offline testing of Social Sentiment Module."""

from datetime import datetime
from typing import List, Dict, Any

from .twitter_client import Tweet
from .reddit_client import RedditPost
from .fear_greed import FearGreedData


# Mock Fear & Greed Data
MOCK_FEAR_GREED_HISTORY: List[FearGreedData] = [
    FearGreedData(value=72, value_classification="Greed", timestamp=datetime.now(), time_until_update=3600),
    FearGreedData(value=68, value_classification="Greed", timestamp=datetime.now(), time_until_update=3600),
    FearGreedData(value=65, value_classification="Greed", timestamp=datetime.now(), time_until_update=3600),
    FearGreedData(value=71, value_classification="Greed", timestamp=datetime.now(), time_until_update=3600),
    FearGreedData(value=74, value_classification="Greed", timestamp=datetime.now(), time_until_update=3600),
]

MOCK_FEAR_GREED_CURRENT = FearGreedData(
    value=68,
    value_classification="Greed",
    timestamp=datetime.now(),
    time_until_update=3600,
)

# Mock Twitter Data by Symbol
MOCK_TWEETS: Dict[str, List[Tweet]] = {
    "BTC": [
        Tweet(
            id="btc_1",
            text="🚀 $BTC to the moon! Bullish on Bitcoin, this is just the beginning! WAGMI!",
            author="crypto_bull",
            created_at=datetime.now(),
            likes=250,
            retweets=67,
            replies=23,
        ),
        Tweet(
            id="btc_2",
            text="$BTC whale alert - large wallet accumulating 10,000 BTC",
            author="whale_watcher",
            created_at=datetime.now(),
            likes=189,
            retweets=45,
            replies=12,
        ),
        Tweet(
            id="btc_3",
            text="Just bought more $BTC, this dip is a gift! Diamond hands! 💎🙌",
            author="btc_hodler",
            created_at=datetime.now(),
            likes=156,
            retweets=34,
            replies=8,
        ),
        Tweet(
            id="btc_4",
            text="$BTC looking strong, breakout imminent. Keys ready!",
            author="trading_alerts",
            created_at=datetime.now(),
            likes=123,
            retweets=28,
            replies=5,
        ),
        Tweet(
            id="btc_5",
            text="Bitcoin adoption accelerating - major partnership announced",
            author="crypto_news",
            created_at=datetime.now(),
            likes=345,
            retweets=89,
            replies=34,
        ),
    ],
    "ETH": [
        Tweet(
            id="eth_1",
            text="🔥 $ETH upgrade incoming! Gas fees about to drop significantly",
            author="eth_maxi",
            created_at=datetime.now(),
            likes=234,
            retweets=56,
            replies=18,
        ),
        Tweet(
            id="eth_2",
            text="$ETH defi TVL at all-time highs! Bullish case stronger than ever",
            author="defi_analyst",
            created_at=datetime.now(),
            likes=178,
            retweets=45,
            replies=12,
        ),
        Tweet(
            id="eth_3",
            text="$ETH looking for breakout above $4000, technical analysis suggests🚀",
            author="chart_master",
            created_at=datetime.now(),
            likes=145,
            retweets=38,
            replies=9,
        ),
    ],
    "SOL": [
        Tweet(
            id="sol_1",
            text="🐸 $SOL pumping! Solana ecosystem growing rapidly",
            author="solana_gang",
            created_at=datetime.now(),
            likes=198,
            retweets=52,
            replies=15,
        ),
        Tweet(
            id="sol_2",
            text="$SOL NFT volume surge! Solana becoming the NFT chain",
            author="nft_tracker",
            created_at=datetime.now(),
            likes=167,
            retweets=41,
            replies=11,
        ),
        Tweet(
            id="sol_3",
            text="$SOL partnerships announced, institutional interest growing",
            author="solana_news",
            created_at=datetime.now(),
            likes=234,
            retweets=67,
            replies=22,
        ),
    ],
}

# Mock Reddit Data by Symbol
MOCK_REDDIT_POSTS: Dict[str, List[RedditPost]] = {
    "BTC": [
        RedditPost(
            id="btc_r1",
            title="Bitcoin looking incredibly bullish! 🚀",
            selftext="Just accumulated more BTC, this dip is a gift. WAGMI!",
            author="crypto_hodler",
            subreddit="CryptoCurrency",
            created_utc=datetime.now(),
            score=1250,
            num_comments=234,
            url="https://reddit.com/r/CryptoCurrency/...",
        ),
        RedditPost(
            id="btc_r2",
            title="$BTC whale alert - large wallet accumulating",
            selftext="Someone just bought 10,000 BTC. This is huge.",
            author="whale_alerts",
            subreddit="Bitcoin",
            created_utc=datetime.now(),
            score=890,
            num_comments=156,
            url="https://reddit.com/r/Bitcoin/...",
        ),
        RedditPost(
            id="btc_r3",
            title="Bitcoin partnership with major tech company announced",
            selftext="Big news! Bitcoin adoption is accelerating.",
            author="crypto_news_daily",
            subreddit="Bitcoin",
            created_utc=datetime.now(),
            score=2100,
            num_comments=445,
            url="https://reddit.com/r/Bitcoin/...",
        ),
    ],
    "ETH": [
        RedditPost(
            id="eth_r1",
            title="$ETH upgrade details revealed - huge improvements",
            selftext="Ethereum is about to get so much better. Staking rewards increasing!",
            author="eth_daily",
            subreddit="ethereum",
            created_utc=datetime.now(),
            score=980,
            num_comments=178,
            url="https://reddit.com/r/ethereum/...",
        ),
        RedditPost(
            id="eth_r2",
            title="ETH defi ecosystem hitting new highs",
            selftext="TVL and usage up significantly. This is bullish.",
            author="defi_watch",
            subreddit="CryptoCurrency",
            created_utc=datetime.now(),
            score=756,
            num_comments=123,
            url="https://reddit.com/r/CryptoCurrency/...",
        ),
    ],
    "SOL": [
        RedditPost(
            id="sol_r1",
            title="Solana ecosystem growing fast - check these numbers",
            selftext="TVL, wallets, and transactions all up. Solana is winning.",
            author="solana_fan",
            subreddit="SOLMarkets",
            created_utc=datetime.now(),
            score=654,
            num_comments=89,
            url="https://reddit.com/r/SOLMarkets/...",
        ),
        RedditPost(
            id="sol_r2",
            title="$SOL looking ready for next leg up",
            selftext="Technical analysis shows strong support. Target $200+",
            author="trading_pro",
            subreddit="SOLMarkets",
            created_utc=datetime.now(),
            score=534,
            num_comments=67,
            url="https://reddit.com/r/SOLMarkets/...",
        ),
    ],
}


def get_mock_sentiment_result(symbol: str) -> Dict[str, Any]:
    """
    Get a complete mock sentiment result for testing.

    Args:
        symbol: Crypto symbol

    Returns:
        Dictionary matching the expected output format
    """
    import random

    # Generate somewhat realistic mock values
    base_sentiment = 60 + random.randint(-15, 25)
    fear_greed = 50 + random.randint(-15, 25)
    twitter_volume = 10000 + random.randint(0, 10000)

    return {
        "symbol": symbol.upper(),
        "sentiment_score": base_sentiment,
        "fear_greed_index": fear_greed,
        "twitter_volume": twitter_volume,
        "reddit_sentiment": round((base_sentiment - 50) / 50, 4),  # Convert to -1 to 1
        "twitter_sentiment": round((base_sentiment - 50) / 50 * 0.9, 4),
        "timestamp": datetime.now().isoformat(),
        "discussion_volume": twitter_volume + random.randint(100, 500),
        "twitter_posts_analyzed": 50,
        "reddit_posts_analyzed": 25,
        "data_sources": ["twitter", "reddit", "fear_greed"],
    }


# Export all mock data
__all__ = [
    "MOCK_FEAR_GREED_CURRENT",
    "MOCK_FEAR_GREED_HISTORY",
    "MOCK_TWEETS",
    "MOCK_REDDIT_POSTS",
    "get_mock_sentiment_result",
]
