"""Configuration for Social Sentiment Module."""

import os
from dataclasses import dataclass
from typing import List


@dataclass
class TwitterConfig:
    """Twitter API configuration."""

    # Use Nitter (free) or official Twitter API v2
    use_nitter: bool = True
    nitter_instances: List[str] = None

    # Official Twitter API v2 (if use_nitter=False)
    bearer_token: str = None
    api_key: str = None
    api_secret: str = None
    access_token: str = None
    access_secret: str = None

    # Rate limiting
    requests_per_minute: int = 60
    rate_limit_delay: float = 1.0  # seconds between requests

    def __post_init__(self):
        if self.nitter_instances is None:
            self.nitter_instances = [
                "nitter.net",
                "nitter.privacydev.net",
                "nitter.poast.org",
            ]


@dataclass
class RedditConfig:
    """Reddit API configuration."""

    client_id: str = None
    client_secret: str = None
    username: str = None
    password: str = None
    user_agent: str = "CryptoSentimentBot/1.0"

    # Subreddits to monitor
    crypto_subreddits: List[str] = None

    # Rate limiting
    requests_per_minute: int = 60
    rate_limit_delay: float = 1.0

    def __post_init__(self):
        if self.crypto_subreddits is None:
            self.crypto_subreddits = [
                "CryptoCurrency",
                "Bitcoin",
                "ethereum",
                "SOLMarkets",
                "Cardico",
            ]


@dataclass
class FearGreedConfig:
    """Fear & Greed API configuration."""

    # Alternative.me API is free, no key required
    api_url: str = "https://api.alternative.me/fng/"
    update_interval: int = 3600  # seconds, index updates every 8 hours

    # Cache
    cache_ttl: int = 3600  # seconds


@dataclass
class SentimentConfig:
    """Main sentiment configuration."""

    twitter: TwitterConfig = None
    reddit: RedditConfig = None
    fear_greed: FearGreedConfig = None

    # Weights for final score
    volume_weight: float = 0.5
    text_weight: float = 0.5

    # Mock mode for offline testing
    mock_mode: bool = False

    def __post_init__(self):
        if self.twitter is None:
            self.twitter = TwitterConfig()
        if self.reddit is None:
            self.reddit = RedditConfig()
        if self.fear_greed is None:
            self.fear_greed = FearGreedConfig()


# Environment variable overrides
def load_config_from_env() -> SentimentConfig:
    """Load configuration from environment variables."""

    twitter_config = TwitterConfig(
        use_nitter=os.getenv("TWITTER_USE_NITTER", "true").lower() == "true",
        bearer_token=os.getenv("TWITTER_BEARER_TOKEN"),
        api_key=os.getenv("TWITTER_API_KEY"),
        api_secret=os.getenv("TWITTER_API_SECRET"),
        access_token=os.getenv("TWITTER_ACCESS_TOKEN"),
        access_secret=os.getenv("TWITTER_ACCESS_SECRET"),
    )

    reddit_config = RedditConfig(
        client_id=os.getenv("REDDIT_CLIENT_ID"),
        client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
        username=os.getenv("REDDIT_USERNAME"),
        password=os.getenv("REDDIT_PASSWORD"),
        user_agent=os.getenv("REDDIT_USER_AGENT", "CryptoSentimentBot/1.0"),
    )

    fear_greed_config = FearGreedConfig(
        api_url=os.getenv("FEAR_GREED_API_URL", "https://api.alternative.me/fng/"),
    )

    return SentimentConfig(
        twitter=twitter_config,
        reddit=reddit_config,
        fear_greed=fear_greed_config,
        mock_mode=os.getenv("SENTIMENT_MOCK_MODE", "false").lower() == "true",
    )


# Default config
default_config = load_config_from_env()
