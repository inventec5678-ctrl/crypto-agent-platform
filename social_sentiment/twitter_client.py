"""Twitter/X Sentiment Client using Nitter (free) or Twitter API v2."""

import asyncio
import aiohttp
import time
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from datetime import datetime

from .sentiment_analyzer import get_sentiment_analyzer, SentimentResult


@dataclass
class Tweet:
    """Represents a tweet."""
    id: str
    text: str
    author: str
    created_at: datetime
    likes: int
    retweets: int
    replies: int


@dataclass
class TwitterSearchResult:
    """Twitter search result with sentiment."""
    symbol: str
    tweets: List[Tweet]
    sentiment_results: List[SentimentResult]
    avg_sentiment: float
    volume: int
    raw_sentiment_score: float  # -1 to 1


class RateLimiter:
    """Simple rate limiter for API requests."""

    def __init__(self, requests_per_minute: int = 60):
        self.requests_per_minute = requests_per_minute
        self.min_interval = 60.0 / requests_per_minute
        self.last_request = 0.0

    async def acquire(self):
        """Wait if necessary to respect rate limits."""
        now = time.time()
        elapsed = now - self.last_request
        if elapsed < self.min_interval:
            await asyncio.sleep(self.min_interval - elapsed)
        self.last_request = time.time()


class TwitterClient:
    """
    Twitter sentiment client supporting:
    - Nitter (free, no API key)
    - Twitter API v2 (requires bearer token)
    """

    def __init__(
        self,
        bearer_token: Optional[str] = None,
        use_nitter: bool = True,
        nitter_instances: Optional[List[str]] = None,
        requests_per_minute: int = 60,
    ):
        self.bearer_token = bearer_token
        self.use_nitter = use_nitter
        self.nitter_instances = nitter_instances or [
            "nitter.net",
            "nitter.privacydev.net",
        ]
        self.current_nitter_index = 0
        self.rate_limiter = RateLimiter(requests_per_minute)
        self.sentiment_analyzer = get_sentiment_analyzer()
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    def _get_nitter_base_url(self) -> str:
        """Get current Nitter instance URL."""
        return f"https://{self.nitter_instances[self.current_nitter_index]}"

    async def _rotate_nitter_instance(self):
        """Rotate to next Nitter instance."""
        self.current_nitter_index = (
            self.current_nitter_index + 1
        ) % len(self.nitter_instances)

    async def search_tweets_nitter(
        self, query: str, limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Search tweets using Nitter RSS/API.

        Note: Nitter's API is limited, this is a best-effort approach.
        """
        await self.rate_limiter.acquire()

        base_url = self._get_nitter_base_url()

        # Nitter doesn't have a direct search API, use RSS as fallback
        # For more complete implementation, consider using a third-party service
        try:
            session = await self._get_session()
            url = f"{base_url}/i/search"

            async with session.get(
                url,
                params={"q": query, "f": "tweets", "size": limit},
                timeout=aiohttp.ClientTimeout(total=10),
                headers={"User-Agent": "Mozilla/5.0"},
            ) as response:
                if response.status == 200:
                    # Parse HTML for tweets (simplified)
                    text = await response.text()
                    return self._parse_nitter_html(text, limit)
                else:
                    await self._rotate_nitter_instance()
                    return []
        except Exception:
            await self._rotate_nitter_instance()
            return []

    def _parse_nitter_html(self, html: str, limit: int) -> List[Dict[str, Any]]:
        """Parse Nitter HTML to extract tweets (simplified)."""
        import re

        tweets = []
        # Simple regex to find tweet content (Nitter structure varies)
        # This is a basic implementation
        tweet_pattern = re.compile(r'<div class="tweet-content[^>]*>([^<]+)</div>')

        matches = tweet_pattern.findall(html)
        for i, match in enumerate(matches[:limit]):
            tweets.append(
                {
                    "id": f"nitter_{int(time.time())}_{i}",
                    "text": match.strip(),
                    "author": "unknown",
                    "created_at": datetime.now().isoformat(),
                    "likes": 0,
                    "retweets": 0,
                    "replies": 0,
                }
            )

        return tweets

    async def search_tweets_api_v2(
        self, query: str, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Search tweets using Twitter API v2.

        Requires bearer token.
        """
        if not self.bearer_token:
            raise ValueError("Bearer token required for Twitter API v2")

        await self.rate_limiter.acquire()

        session = await self._get_session()
        url = "https://api.twitter.com/2/tweets/search/recent"

        headers = {"Authorization": f"Bearer {self.bearer_token}"}
        params = {"query": query, "max_results": min(limit, 100)}

        try:
            async with session.get(
                url, headers=headers, params=params, timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._parse_twitter_api_response(data)
                elif response.status == 429:
                    # Rate limited, wait and retry
                    await asyncio.sleep(60)
                    return await self.search_tweets_api_v2(query, limit)
                else:
                    return []
        except Exception:
            return []

    def _parse_twitter_api_response(self, data: Dict) -> List[Dict[str, Any]]:
        """Parse Twitter API v2 response."""
        tweets = []
        includes = data.get("includes", {})
        users = {u["id"]: u["username"] for u in includes.get("users", [])}

        for tweet in data.get("data", []):
            author_id = tweet.get("author_id", "")
            tweets.append(
                {
                    "id": tweet["id"],
                    "text": tweet["text"],
                    "author": users.get(author_id, "unknown"),
                    "created_at": tweet.get("created_at", ""),
                    "likes": tweet.get("public_metrics", {}).get("like_count", 0),
                    "retweets": tweet.get("public_metrics", {}).get("retweet_count", 0),
                    "replies": tweet.get("public_metrics", {}).get("reply_count", 0),
                }
            )

        return tweets

    async def get_crypto_sentiment(
        self, symbol: str, limit: int = 50
    ) -> TwitterSearchResult:
        """
        Get sentiment for a specific cryptocurrency symbol.

        Args:
            symbol: Crypto symbol (e.g., "BTC", "ETH")
            limit: Maximum number of tweets to fetch

        Returns:
            TwitterSearchResult with sentiment analysis
        """
        # Build search query for crypto
        query = f"${symbol} OR {symbol} crypto OR {symbol} bitcoin"

        # Fetch tweets
        if self.use_nitter:
            raw_tweets = await self.search_tweets_nitter(query, limit)
        else:
            raw_tweets = await self.search_tweets_api_v2(query, limit)

        # Parse into Tweet objects
        tweets = []
        for t in raw_tweets:
            tweets.append(
                Tweet(
                    id=t["id"],
                    text=t["text"],
                    author=t["author"],
                    created_at=datetime.fromisoformat(t["created_at"].replace("Z", "+00:00"))
                    if t["created_at"]
                    else datetime.now(),
                    likes=t["likes"],
                    retweets=t["retweets"],
                    replies=t["replies"],
                )
            )

        # Analyze sentiment
        texts = [t.text for t in tweets]
        sentiment_results = self.sentiment_analyzer.analyze_batch(texts)

        # Aggregate
        avg_sentiment, volume = self.sentiment_analyzer.aggregate_sentiment(
            sentiment_results
        )

        return TwitterSearchResult(
            symbol=symbol,
            tweets=tweets,
            sentiment_results=sentiment_results,
            avg_sentiment=avg_sentiment,
            volume=volume,
            raw_sentiment_score=avg_sentiment,  # Already -1 to 1
        )

    async def close(self):
        """Close the HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()


# Mock data for offline testing
MOCK_TWEETS = [
    Tweet(
        id="1",
        text="🚀 $BTC to the moon! Bullish on Bitcoin, this is just the beginning!",
        author="crypto_bull",
        created_at=datetime.now(),
        likes=150,
        retweets=45,
        replies=12,
    ),
    Tweet(
        id="2",
        text="$BTC looking strong, accumulation phase continues",
        author="hodler123",
        created_at=datetime.now(),
        likes=89,
        retweets=23,
        replies=8,
    ),
    Tweet(
        id="3",
        text="Bearish vibes... $BTC might dump hard",
        author="bear_trader",
        created_at=datetime.now(),
        likes=34,
        retweets=12,
        replies=5,
    ),
    Tweet(
        id="4",
        text="$BTC upgrade announced! Big things coming 🚀",
        author="crypto_news",
        created_at=datetime.now(),
        likes=234,
        retweets=67,
        replies=23,
    ),
    Tweet(
        id="5",
        text="$BTC whale activity detected, someone is accumulating",
        author="whale_watcher",
        created_at=datetime.now(),
        likes=156,
        retweets=45,
        replies=15,
    ),
]


class MockTwitterClient(TwitterClient):
    """Mock Twitter client for offline testing."""

    async def search_tweets_nitter(
        self, query: str, limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Return mock tweets."""
        return [
            {
                "id": t.id,
                "text": t.text,
                "author": t.author,
                "created_at": t.created_at.isoformat(),
                "likes": t.likes,
                "retweets": t.retweets,
                "replies": t.replies,
            }
            for t in MOCK_TWEETS[:limit]
        ]

    async def search_tweets_api_v2(
        self, query: str, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Return mock tweets."""
        return await self.search_tweets_nitter(query, limit)
