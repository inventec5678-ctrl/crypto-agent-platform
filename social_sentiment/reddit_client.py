"""Reddit Sentiment Client using Reddit API (free OAuth)."""

import asyncio
import aiohttp
import base64
import time
import json
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from datetime import datetime

from .sentiment_analyzer import get_sentiment_analyzer, SentimentResult


@dataclass
class RedditPost:
    """Represents a Reddit post."""
    id: str
    title: str
    selftext: str
    author: str
    subreddit: str
    created_utc: datetime
    score: int
    num_comments: int
    url: str


@dataclass
class RedditSearchResult:
    """Reddit search result with sentiment."""
    symbol: str
    posts: List[RedditPost]
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


class RedditClient:
    """
    Reddit sentiment client using OAuth authentication.

    Reddit API is free but requires OAuth for write access.
    Read access works without authentication (limited rate).
    """

    def __init__(
        self,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        user_agent: str = "CryptoSentimentBot/1.0",
        subreddits: Optional[List[str]] = None,
        requests_per_minute: int = 60,
    ):
        self.client_id = client_id
        self.client_secret = client_secret
        self.username = username
        self.password = password
        self.user_agent = user_agent
        self.subreddits = subreddits or [
            "CryptoCurrency",
            "Bitcoin",
            "ethereum",
            "SOLMarkets",
        ]
        self.rate_limiter = RateLimiter(requests_per_minute)
        self.sentiment_analyzer = get_sentiment_analyzer()
        self._session: Optional[aiohttp.ClientSession] = None
        self._access_token: Optional[str] = None
        self._token_expiry: float = 0

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def _get_access_token(self) -> Optional[str]:
        """Get OAuth access token for Reddit API."""
        # Check if we have a valid token
        if self._access_token and time.time() < self._token_expiry:
            return self._access_token

        if not self.client_id or not self.client_secret:
            # No OAuth credentials, use unauthenticated access
            return None

        await self.rate_limiter.acquire()

        session = await self._get_session()

        # Basic auth header
        auth = base64.b64encode(
            f"{self.client_id}:{self.client_secret}".encode()
        ).decode()

        try:
            async with session.post(
                "https://www.reddit.com/api/v1/access_token",
                headers={
                    "Authorization": f"Basic {auth}",
                    "User-Agent": self.user_agent,
                },
                data={
                    "grant_type": "password",
                    "username": self.username,
                    "password": self.password,
                },
                timeout=aiohttp.ClientTimeout(total=10),
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    self._access_token = data.get("access_token")
                    self._token_expiry = time.time() + data.get("expires_in", 3600)
                    return self._access_token
                else:
                    return None
        except Exception:
            return None

    async def search_posts(
        self, query: str, subreddit: Optional[str] = None, limit: int = 25
    ) -> List[Dict[str, Any]]:
        """
        Search Reddit posts.

        Args:
            query: Search query
            subreddit: Specific subreddit (None for all)
            limit: Maximum number of posts

        Returns:
            List of post data dictionaries
        """
        await self.rate_limiter.acquire()

        session = await self._get_session()

        # Build URL
        if subreddit:
            url = f"https://www.reddit.com/r/{subreddit}/search.json"
        else:
            url = "https://www.reddit.com/search.json"

        params = {"q": query, "limit": min(limit, 100), "sort": "relevance"}

        headers = {"User-Agent": self.user_agent}
        if self._access_token:
            headers["Authorization"] = f"Bearer {self._access_token}"

        try:
            async with session.get(
                url,
                params=params,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=15),
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._parse_search_response(data)
                elif response.status == 429:
                    # Rate limited
                    await asyncio.sleep(60)
                    return []
                else:
                    return []
        except Exception:
            return []

    def _parse_search_response(self, data: Dict) -> List[Dict[str, Any]]:
        """Parse Reddit search API response."""
        posts = []
        children = data.get("data", {}).get("children", [])

        for child in children:
            post_data = child.get("data", {})
            posts.append(
                {
                    "id": post_data.get("id", ""),
                    "title": post_data.get("title", ""),
                    "selftext": post_data.get("selftext", ""),
                    "author": post_data.get("author", ""),
                    "subreddit": post_data.get("subreddit", ""),
                    "created_utc": datetime.fromtimestamp(
                        post_data.get("created_utc", 0)
                    ).isoformat(),
                    "score": post_data.get("score", 0),
                    "num_comments": post_data.get("num_comments", 0),
                    "url": post_data.get("url", ""),
                }
            )

        return posts

    async def get_subreddit_posts(
        self, subreddit: str, limit: int = 25, sort: str = "hot"
    ) -> List[Dict[str, Any]]:
        """
        Get posts from a specific subreddit.

        Args:
            subreddit: Subreddit name (without r/)
            limit: Maximum number of posts
            sort: Sort method (hot, new, top, rising)

        Returns:
            List of post data dictionaries
        """
        await self.rate_limiter.acquire()

        session = await self._get_session()
        url = f"https://www.reddit.com/r/{subreddit}/{sort}.json"

        params = {"limit": min(limit, 100)}

        headers = {"User-Agent": self.user_agent}
        if self._access_token:
            headers["Authorization"] = f"Bearer {self._access_token}"

        try:
            async with session.get(
                url,
                params=params,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=15),
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._parse_search_response(data)
                else:
                    return []
        except Exception:
            return []

    async def get_crypto_sentiment(
        self, symbol: str, limit: int = 50
    ) -> RedditSearchResult:
        """
        Get sentiment for a specific cryptocurrency from Reddit.

        Args:
            symbol: Crypto symbol (e.g., "BTC", "ETH")
            limit: Maximum number of posts to analyze

        Returns:
            RedditSearchResult with sentiment analysis
        """
        # Search across all crypto subreddits
        all_posts = []

        # Search in each subreddit
        for subreddit in self.subreddits:
            posts = await self.search_posts(
                query=f"{symbol} OR ${symbol}",
                subreddit=subreddit,
                limit=limit // len(self.subreddits) + 10,
            )
            all_posts.extend(posts)

        # Also get hot posts from main crypto subs (increased from 10 to 30)
        for subreddit in ["CryptoCurrency", "Bitcoin", "ethereum", "SOLMarkets"]:
            posts = await self.get_subreddit_posts(subreddit, limit=30)
            # Filter to relevant posts
            symbol_lower = symbol.lower()
            filtered = [
                p for p in posts
                if symbol_lower in p["title"].lower() or symbol_lower in p.get("selftext", "").lower()
            ]
            all_posts.extend(filtered)

        # Remove duplicates by ID
        seen_ids = set()
        unique_posts = []
        for p in all_posts:
            if p["id"] not in seen_ids:
                seen_ids.add(p["id"])
                unique_posts.append(p)

        # Take top posts by score
        unique_posts.sort(key=lambda x: x["score"], reverse=True)
        unique_posts = unique_posts[:limit]

        # Parse into RedditPost objects
        posts = []
        for p in unique_posts:
            posts.append(
                RedditPost(
                    id=p["id"],
                    title=p["title"],
                    selftext=p["selftext"],
                    author=p["author"],
                    subreddit=p["subreddit"],
                    created_utc=datetime.fromisoformat(p["created_utc"].replace("Z", "+00:00"))
                    if p["created_utc"]
                    else datetime.now(),
                    score=p["score"],
                    num_comments=p["num_comments"],
                    url=p["url"],
                )
            )

        # Analyze sentiment (combine title and selftext)
        texts = [f"{p.title} {p.selftext}" for p in posts]
        sentiment_results = self.sentiment_analyzer.analyze_batch(texts)

        # Aggregate
        avg_sentiment, volume = self.sentiment_analyzer.aggregate_sentiment(
            sentiment_results
        )

        return RedditSearchResult(
            symbol=symbol,
            posts=posts,
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
MOCK_POSTS = [
    RedditPost(
        id="p1",
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
        id="p2",
        title="$BTC whale alert - large wallet accumulating",
        selftext="Someone just bought 10,000 BTC. This is huge for the market.",
        author="whale_alerts",
        subreddit="Bitcoin",
        created_utc=datetime.now(),
        score=890,
        num_comments=156,
        url="https://reddit.com/r/Bitcoin/...",
    ),
    RedditPost(
        id="p3",
        title="Should I be worried about the correction?",
        selftext="BTC dropped 5% today, feeling nervous. Anyone else?",
        author="new_crypto_investor",
        subreddit="CryptoCurrency",
        created_utc=datetime.now(),
        score=456,
        num_comments=89,
        url="https://reddit.com/r/CryptoCurrency/...",
    ),
    RedditPost(
        id="p4",
        title="Bitcoin partnership announced with major tech company",
        selftext="Big news! Bitcoin adoption is accelerating. This is just the beginning.",
        author="crypto_news_daily",
        subreddit="Bitcoin",
        created_utc=datetime.now(),
        score=2100,
        num_comments=445,
        url="https://reddit.com/r/Bitcoin/...",
    ),
    RedditPost(
        id="p5",
        title="BTC price analysis - breakout imminent",
        selftext="Technical analysis suggests BTC is about to breakout. Keys ready!",
        author="trading_pro",
        subreddit="CryptoCurrency",
        created_utc=datetime.now(),
        score=678,
        num_comments=123,
        url="https://reddit.com/r/CryptoCurrency/...",
    ),
]


class MockRedditClient(RedditClient):
    """Mock Reddit client for offline testing."""

    async def search_posts(
        self, query: str, subreddit: Optional[str] = None, limit: int = 25
    ) -> List[Dict[str, Any]]:
        """Return mock posts."""
        return [
            {
                "id": p.id,
                "title": p.title,
                "selftext": p.selftext,
                "author": p.author,
                "subreddit": p.subreddit,
                "created_utc": p.created_utc.isoformat(),
                "score": p.score,
                "num_comments": p.num_comments,
                "url": p.url,
            }
            for p in MOCK_POSTS[:limit]
        ]

    async def get_subreddit_posts(
        self, subreddit: str, limit: int = 25, sort: str = "hot"
    ) -> List[Dict[str, Any]]:
        """Return mock posts."""
        return await self.search_posts(query="", subreddit=subreddit, limit=limit)
