"""Simple NLP Sentiment Analyzer for Crypto Content."""

import re
from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class SentimentResult:
    """Sentiment analysis result."""

    score: float  # -1.0 to 1.0
    positive_count: int
    negative_count: int
    neutral_count: int
    total_count: int
    confidence: float  # 0.0 to 1.0


class SentimentAnalyzer:
    """
    Simple lexicon-based sentiment analyzer optimized for crypto content.
    
    Uses a curated lexicon of crypto-related terms with sentiment weights.
    """

    # Positive words with weights (0.1 to 1.0)
    POSITIVE_WORDS: Dict[str, float] = {
        # Strong positive
        "moon": 0.9,
        "bullish": 0.85,
        "pump": 0.7,
        " breakout": 0.7,
        "surge": 0.7,
        "soar": 0.75,
        "rally": 0.7,
        "gain": 0.6,
        "profit": 0.7,
        "win": 0.7,
        "hodl": 0.6,
        "hold": 0.4,
        "long": 0.5,
        "buy": 0.5,
        "accumulate": 0.6,
        "up": 0.4,
        "higher": 0.5,
        "growth": 0.6,
        "adoption": 0.6,
        "upgrade": 0.6,
        "partnership": 0.6,
        " listing": 0.6,
        "approval": 0.7,
        "bull": 0.7,
        " ATH": 0.8,
        "all-time high": 0.8,
        "green": 0.5,
        "rocket": 0.8,
        " diamond hands": 0.6,
        "to the moon": 0.9,
        "WAGMI": 0.8,
        "NGMI": -0.3,
        "bull run": 0.8,
        "halving": 0.6,
        "s2f": 0.5,
        "deflationary": 0.5,
        "burn": 0.5,
        "supply": 0.4,
    }

    # Negative words with weights (-0.1 to -1.0)
    NEGATIVE_WORDS: Dict[str, float] = {
        # Strong negative
        "crash": -0.85,
        "dump": -0.75,
        "bearish": -0.8,
        "scam": -0.9,
        "rug": -0.85,
        "rugpull": -0.95,
        "hack": -0.8,
        "sell": -0.4,
        "down": -0.4,
        "loss": -0.7,
        "fear": -0.6,
        "panic": -0.8,
        "fud": -0.5,
        "drop": -0.5,
        "fall": -0.5,
        "bear": -0.7,
        "short": -0.4,
        "liquidation": -0.8,
        "liquidation": -0.8,
        "bankrupt": -0.9,
        "fraud": -0.9,
        "investigation": -0.7,
        "ban": -0.7,
        "regulation": -0.5,
        "ban": -0.6,
        "red": -0.4,
        "blood": -0.6,
        "rekt": -0.7,
        "paper hands": -0.5,
        "correction": -0.4,
        "consolidation": -0.2,
        "volatile": -0.3,
        " whale": -0.3,
        "manipulation": -0.6,
        "ponzi": -0.9,
        "shitcoin": -0.7,
    }

    # Neutral modifiers
    NEUTRAL_WORDS = {"crypto", "bitcoin", "ethereum", "blockchain", "token", "coin", "trading"}

    # Emoji sentiment mappings
    EMOJI_SENTIMENT: Dict[str, float] = {
        "🚀": 0.9,  # rocket
        "📈": 0.7,  # chart up
        "💎": 0.6,  # diamond
        "🙌": 0.5,  # handshake
        "❤️": 0.4,  # heart
        "🔥": 0.6,  # fire
        "💰": 0.5,  # money bag
        "✅": 0.3,  # check
        "⭐": 0.4,  # star
        "📉": -0.7,  # chart down
        "💀": -0.7,  # skull
        "😭": -0.6,  # crying
        "❌": -0.4,  # cross
        "⚠️": -0.5,  # warning
        "🧻": -0.6,  # paper hands
        "🥱": -0.3,  # boring
    }

    def __init__(self):
        """Initialize the sentiment analyzer."""
        # Compile regex patterns for efficiency
        self._word_pattern = re.compile(r"\b\w+\b")
        self._emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F1E0-\U0001F1FF"  # flags
            "\U00002702-\U000027B0"  # dingbats
            "\U000024C2-\U0001F251"  # enclosed characters
            "]+",
            flags=re.UNICODE,
        )

    def analyze(self, text: str) -> SentimentResult:
        """
        Analyze sentiment of a single text.

        Args:
            text: Input text to analyze

        Returns:
            SentimentResult with score and metadata
        """
        if not text or not isinstance(text, str):
            return SentimentResult(
                score=0.0,
                positive_count=0,
                negative_count=0,
                neutral_count=0,
                total_count=0,
                confidence=0.0,
            )

        # Clean and lowercase
        text_lower = text.lower()
        words = self._word_pattern.findall(text_lower)

        # Find emoji
        emojis = self._emoji_pattern.findall(text)

        # Calculate word sentiment
        word_scores = []
        positive_count = 0
        negative_count = 0
        neutral_count = 0

        for word in words:
            # Skip neutral words
            if word in self.NEUTRAL_WORDS:
                continue

            # Check positive words
            for pos_word, weight in self.POSITIVE_WORDS.items():
                if pos_word in text_lower:
                    word_scores.append(weight)
                    positive_count += 1
                    break
            else:
                # Check negative words
                for neg_word, weight in self.NEGATIVE_WORDS.items():
                    if neg_word in text_lower:
                        word_scores.append(weight)
                        negative_count += 1
                        break

        # Calculate emoji sentiment
        emoji_scores = [self.EMOJI_SENTIMENT.get(emoji, 0) for emoji in emojis]

        # Combine all scores
        all_scores = word_scores + emoji_scores
        total_count = len(all_scores)

        if total_count == 0:
            return SentimentResult(
                score=0.0,
                positive_count=0,
                negative_count=0,
                neutral_count=1,
                total_count=1,
                confidence=0.0,
            )

        # Weighted average
        score = sum(all_scores) / total_count

        # Confidence based on signal strength
        signal_strength = sum(1 for s in all_scores if abs(s) > 0.3) / total_count
        confidence = min(1.0, signal_strength + 0.1)

        neutral_count = total_count - positive_count - negative_count

        return SentimentResult(
            score=round(score, 4),
            positive_count=positive_count,
            negative_count=negative_count,
            neutral_count=neutral_count,
            total_count=total_count,
            confidence=round(confidence, 4),
        )

    def analyze_batch(self, texts: List[str]) -> List[SentimentResult]:
        """
        Analyze sentiment of multiple texts.

        Args:
            texts: List of input texts

        Returns:
            List of SentimentResult
        """
        return [self.analyze(text) for text in texts]

    def aggregate_sentiment(self, results: List[SentimentResult]) -> Tuple[float, int]:
        """
        Aggregate sentiment from multiple results.

        Args:
            results: List of SentimentResult

        Returns:
            Tuple of (average_score, total_volume)
        """
        if not results:
            return 0.0, 0

        total_score = sum(r.score * r.confidence for r in results)
        total_weight = sum(r.confidence for r in results)

        if total_weight == 0:
            avg_score = 0.0
        else:
            avg_score = total_score / total_weight

        total_volume = sum(r.total_count for r in results)

        return round(avg_score, 4), total_volume


# Singleton instance
_sentiment_analyzer = None


def get_sentiment_analyzer() -> SentimentAnalyzer:
    """Get singleton sentiment analyzer instance."""
    global _sentiment_analyzer
    if _sentiment_analyzer is None:
        _sentiment_analyzer = SentimentAnalyzer()
    return _sentiment_analyzer
