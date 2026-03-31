"""Social Sentiment Analysis Module for Crypto Agent Platform."""

from .aggregator import SentimentAggregator
from .twitter_client import TwitterClient
from .reddit_client import RedditClient
from .fear_greed import FearGreedClient
from .sentiment_analyzer import SentimentAnalyzer

__all__ = [
    "SentimentAggregator",
    "TwitterClient",
    "RedditClient",
    "FearGreedClient",
    "SentimentAnalyzer",
]
__version__ = "1.0.0"
