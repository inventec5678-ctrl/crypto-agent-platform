"""Main Sentiment Aggregator - combines all sources into final score."""

import asyncio
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List
from datetime import datetime

from .config import SentimentConfig, load_config_from_env
from .twitter_client import TwitterClient, MockTwitterClient
from .reddit_client import RedditClient, MockRedditClient
from .fear_greed import FearGreedClient, MockFearGreedClient
from .sentiment_analyzer import SentimentAnalyzer


@dataclass
class SentimentResult:
    """Final aggregated sentiment result."""

    symbol: str
    sentiment_score: float  # 0-100
    fear_greed_index: int  # 0-100
    twitter_volume: int
    reddit_sentiment: float  # -1 to 1
    twitter_sentiment: float  # -1 to 1
    timestamp: str

    # Additional metadata
    discussion_volume: int = 0
    twitter_posts_analyzed: int = 0
    reddit_posts_analyzed: int = 0
    data_sources: List[str] = None
    
    # Articles from Reddit
    articles: List[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    def __post_init__(self):
        if self.data_sources is None:
            self.data_sources = []
        if self.articles is None:
            self.articles = []
        # Ensure timestamp is ISO format string
        if isinstance(self.timestamp, datetime):
            self.timestamp = self.timestamp.isoformat()


class SentimentAggregator:
    """
    Main aggregator that combines sentiment from multiple sources.

    Final Score = Volume Score (50%) + Text Sentiment Score (50%)

    Volume Score: Normalized discussion volume (0-100)
    Text Sentiment Score: Combined Twitter + Reddit sentiment (-1 to 1) mapped to (0-100)
    """

    def __init__(self, config: Optional[SentimentConfig] = None):
        self.config = config or load_config_from_env()
        self.sentiment_analyzer = SentimentAnalyzer()

        # Initialize clients
        if self.config.mock_mode:
            self.twitter_client = MockTwitterClient()
            self.reddit_client = MockRedditClient()
            self.fear_greed_client = MockFearGreedClient()
        else:
            twitter_cfg = self.config.twitter
            self.twitter_client = TwitterClient(
                bearer_token=twitter_cfg.bearer_token,
                use_nitter=twitter_cfg.use_nitter,
                nitter_instances=twitter_cfg.nitter_instances,
                requests_per_minute=twitter_cfg.requests_per_minute,
            )

            reddit_cfg = self.config.reddit
            self.reddit_client = RedditClient(
                client_id=reddit_cfg.client_id,
                client_secret=reddit_cfg.client_secret,
                username=reddit_cfg.username,
                password=reddit_cfg.password,
                user_agent=reddit_cfg.user_agent,
                subreddits=reddit_cfg.crypto_subreddits,
                requests_per_minute=reddit_cfg.requests_per_minute,
            )

            fear_greed_cfg = self.config.fear_greed
            self.fear_greed_client = FearGreedClient(
                api_url=fear_greed_cfg.api_url,
            )

    async def get_sentiment(
        self,
        symbol: str,
        twitter_limit: int = 50,
        reddit_limit: int = 50,
    ) -> SentimentResult:
        """
        Get aggregated sentiment for a cryptocurrency symbol.

        Args:
            symbol: Crypto symbol (e.g., "BTC", "ETH")
            twitter_limit: Max tweets to analyze
            reddit_limit: Max Reddit posts to analyze

        Returns:
            SentimentResult with aggregated sentiment
        """
        # Fetch all data concurrently
        data_sources = []

        twitter_task = asyncio.create_task(
            self._fetch_twitter_sentiment(symbol, twitter_limit)
        )
        reddit_task = asyncio.create_task(
            self._fetch_reddit_sentiment(symbol, reddit_limit)
        )
        fear_greed_task = asyncio.create_task(
            self._fetch_fear_greed_index()
        )

        # Gather results
        twitter_result, reddit_result, fear_greed_value = await asyncio.gather(
            twitter_task, reddit_task, fear_greed_task, return_exceptions=True
        )

        # Process Twitter
        twitter_sentiment = 0.0
        twitter_volume = 0
        twitter_count = 0
        if not isinstance(twitter_result, Exception) and twitter_result:
            twitter_sentiment = twitter_result.raw_sentiment_score
            twitter_volume = twitter_result.volume
            twitter_count = len(twitter_result.tweets)
            data_sources.append("twitter")

        # Process Reddit
        reddit_sentiment = 0.0
        reddit_count = 0
        reddit_posts = []
        if not isinstance(reddit_result, Exception) and reddit_result:
            reddit_sentiment = reddit_result.raw_sentiment_score
            reddit_count = len(reddit_result.posts)
            reddit_posts = reddit_result.posts
            data_sources.append("reddit")

        # Process Fear & Greed
        fear_greed_index = 50
        if not isinstance(fear_greed_value, Exception) and fear_greed_value:
            fear_greed_index = fear_greed_value.value
            data_sources.append("fear_greed")

        # Calculate final sentiment score
        sentiment_score = self._calculate_final_score(
            twitter_sentiment=twitter_sentiment,
            reddit_sentiment=reddit_sentiment,
            fear_greed_index=fear_greed_index,
            twitter_volume=twitter_volume,
        )

        # Total discussion volume
        discussion_volume = twitter_volume + reddit_count

        # Convert Reddit posts to articles format with per-post sentiment
        articles = []
        for post in reddit_posts[:30]:  # Max 30 articles
            # Analyze each post's title for individual sentiment
            post_analysis = self.sentiment_analyzer.analyze(post.title)
            # Normalize from -1..1 to 0..1 range for display
            post_sentiment = (post_analysis.score + 1) / 2
            
            articles.append({
                "title": post.title[:200] if post.title else "",  # Truncate long titles
                "source": f"r/{post.subreddit}",
                "sentiment": round(post_sentiment, 2),
                "url": post.url,
                "score": post.score,
                "num_comments": post.num_comments,
                "author": post.author,
                "created_utc": post.created_utc.isoformat() if hasattr(post, 'created_utc') and post.created_utc else "",
            })

        return SentimentResult(
            symbol=symbol.upper(),
            sentiment_score=round(sentiment_score, 2),
            fear_greed_index=fear_greed_index,
            twitter_volume=twitter_volume,
            reddit_sentiment=round(reddit_sentiment, 4),
            twitter_sentiment=round(twitter_sentiment, 4),
            timestamp=datetime.now().isoformat(),
            discussion_volume=discussion_volume,
            twitter_posts_analyzed=twitter_count,
            reddit_posts_analyzed=reddit_count,
            data_sources=data_sources,
            articles=articles,
        )

    async def _fetch_twitter_sentiment(
        self, symbol: str, limit: int
    ):
        """Fetch Twitter sentiment."""
        try:
            return await self.twitter_client.get_crypto_sentiment(symbol, limit)
        except Exception as e:
            return e

    async def _fetch_reddit_sentiment(
        self, symbol: str, limit: int
    ):
        """Fetch Reddit sentiment."""
        try:
            return await self.reddit_client.get_crypto_sentiment(symbol, limit)
        except Exception as e:
            return e

    async def _fetch_fear_greed_index(self):
        """Fetch Fear & Greed index."""
        try:
            return await self.fear_greed_client.get_current()
        except Exception as e:
            return e

    def _calculate_final_score(
        self,
        twitter_sentiment: float,
        reddit_sentiment: float,
        fear_greed_index: int,
        twitter_volume: int,
    ) -> float:
        """
        Calculate final sentiment score.

        Formula:
        Volume Score (50%) + Text Sentiment Score (50%)

        Volume Score: Log-normalized volume mapped to 0-100
        Text Sentiment Score: Combined sentiment mapped to 0-100
        """
        import math

        # Normalize sentiment from -1..1 to 0..100
        def sentiment_to_score(s: float) -> float:
            return (s + 1) / 2 * 100  # -1 -> 0, 0 -> 50, 1 -> 100

        # Normalize Fear & Greed to 0-100 (already in this range)
        fear_greed_score = float(fear_greed_index)

        # Volume score (log scale for better distribution)
        if twitter_volume > 0:
            volume_score = min(100, math.log10(twitter_volume + 1) / 4 * 100)
        else:
            volume_score = 50  # Neutral if no volume

        # Text sentiment score (average of Twitter and Reddit sentiment)
        text_scores = []
        if twitter_sentiment != 0:
            text_scores.append(sentiment_to_score(twitter_sentiment))
        if reddit_sentiment != 0:
            text_scores.append(sentiment_to_score(reddit_sentiment))

        if text_scores:
            text_sentiment_score = sum(text_scores) / len(text_scores)
        else:
            text_sentiment_score = 50  # Neutral

        # Weighted combination
        volume_weight = self.config.volume_weight  # 0.5
        text_weight = self.config.text_weight  # 0.5

        final_score = (
            volume_score * volume_weight + text_sentiment_score * text_weight
        )

        # Clamp to 0-100
        return max(0, min(100, final_score))

    async def get_batch_sentiment(
        self, symbols: List[str]
    ) -> Dict[str, SentimentResult]:
        """
        Get sentiment for multiple symbols concurrently.

        Args:
            symbols: List of crypto symbols

        Returns:
            Dictionary mapping symbol to SentimentResult
        """
        tasks = [self.get_sentiment(symbol) for symbol in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        output = {}
        for symbol, result in zip(symbols, results):
            if isinstance(result, Exception):
                # Create error result
                output[symbol.upper()] = SentimentResult(
                    symbol=symbol.upper(),
                    sentiment_score=50.0,
                    fear_greed_index=50,
                    twitter_volume=0,
                    reddit_sentiment=0.0,
                    twitter_sentiment=0.0,
                    timestamp=datetime.now().isoformat(),
                    data_sources=["error"],
                )
            else:
                output[symbol.upper()] = result

        return output

    async def close(self):
        """Close all client sessions."""
        await asyncio.gather(
            self.twitter_client.close(),
            self.reddit_client.close(),
            self.fear_greed_client.close(),
            return_exceptions=True,
        )

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()


# Convenience function
async def get_sentiment(
    symbol: str,
    config: Optional[SentimentConfig] = None,
    mock: bool = False,
) -> Dict[str, Any]:
    """
    Quick function to get sentiment for a symbol.

    Args:
        symbol: Crypto symbol
        config: Optional config
        mock: Force mock mode

    Returns:
        Dictionary with sentiment data
    """
    if mock:
        cfg = config or SentimentConfig(mock_mode=True)
        cfg.mock_mode = True

    aggregator = SentimentAggregator(config)
    try:
        result = await aggregator.get_sentiment(symbol)
        return result.to_dict()
    finally:
        await aggregator.close()
