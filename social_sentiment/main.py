#!/usr/bin/env python3
"""Main entry point for Social Sentiment Module.

Usage:
    python main.py --symbol BTC
    python main.py --symbols BTC ETH SOL
    python main.py --mock
"""

import asyncio
import argparse
import json
import sys
from datetime import datetime
from typing import List, Optional

from .aggregator import SentimentAggregator
from .config import SentimentConfig, load_config_from_env


async def main(
    symbols: List[str],
    config: Optional[SentimentConfig] = None,
    verbose: bool = False,
) -> dict:
    """
    Main function to fetch sentiment for symbols.

    Args:
        symbols: List of crypto symbols
        config: Optional config
        verbose: Print verbose output

    Returns:
        Dictionary with results
    """
    aggregator = SentimentAggregator(config)

    try:
        if len(symbols) == 1:
            # Single symbol
            result = await aggregator.get_sentiment(symbols[0])
            output = result.to_dict()

            if verbose:
                print(f"\n{'='*60}")
                print(f"Sentiment Analysis for {symbols[0]}")
                print(f"{'='*60}")
                print(f"  Sentiment Score:      {output['sentiment_score']}/100")
                print(f"  Fear & Greed Index:  {output['fear_greed_index']}/100")
                print(f"  Twitter Volume:      {output['twitter_volume']}")
                print(f"  Twitter Sentiment:   {output['twitter_sentiment']}")
                print(f"  Reddit Sentiment:    {output['reddit_sentiment']}")
                print(f"  Total Discussion:    {output['discussion_volume']}")
                print(f"  Data Sources:        {', '.join(output['data_sources'])}")
                print(f"  Timestamp:           {output['timestamp']}")
                print(f"{'='*60}\n")

        else:
            # Multiple symbols
            results = await aggregator.get_batch_sentiment(symbols)
            output = {symbol: result.to_dict() for symbol, result in results.items()}

            if verbose:
                print(f"\n{'='*60}")
                print(f"Batch Sentiment Analysis")
                print(f"{'='*60}")
                for symbol in symbols:
                    r = output[symbol.upper()]
                    print(f"\n{symbol}:")
                    print(f"  Score: {r['sentiment_score']} | F&G: {r['fear_greed_index']} | Vol: {r['twitter_volume']}")
                print(f"\n{'='*60}\n")

        return output

    finally:
        await aggregator.close()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Crypto Social Sentiment Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --symbol BTC
  python main.py --symbols BTC ETH SOL
  python main.py --mock --symbol BTC
  python main.py --verbose --symbols BTC ETH

Environment Variables:
  TWITTER_BEARER_TOKEN    Twitter API v2 bearer token
  TWITTER_USE_NITTER      Use Nitter instead of official API (default: true)
  REDDIT_CLIENT_ID        Reddit OAuth client ID
  REDDIT_CLIENT_SECRET    Reddit OAuth client secret
  SENTIMENT_MOCK_MODE     Enable mock mode for testing
        """,
    )

    parser.add_argument(
        "--symbol",
        dest="symbol",
        help="Single crypto symbol (e.g., BTC, ETH)",
    )

    parser.add_argument(
        "--symbols",
        dest="symbols",
        nargs="+",
        help="Multiple crypto symbols",
    )

    parser.add_argument(
        "--mock",
        dest="mock",
        action="store_true",
        help="Use mock data for offline testing",
    )

    parser.add_argument(
        "--verbose",
        dest="verbose",
        action="store_true",
        help="Print detailed output",
    )

    parser.add_argument(
        "--json",
        dest="json_output",
        action="store_true",
        help="Output as JSON",
    )

    return parser.parse_args()


async def async_main():
    """Async main entry point."""
    args = parse_args()

    # Determine symbols
    if args.symbol:
        symbols = [args.symbol.upper()]
    elif args.symbols:
        symbols = [s.upper() for s in args.symbols]
    else:
        # Default to BTC
        symbols = ["BTC"]

    # Build config
    config = load_config_from_env()
    if args.mock:
        config.mock_mode = True

    # Run
    result = await main(symbols, config, args.verbose)

    # JSON output
    if args.json_output:
        print(json.dumps(result, indent=2, default=str))

    return result


def main_entry():
    """Entry point for script execution."""
    try:
        asyncio.run(async_main())
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main_entry()


# Example usage as module:
async def example_usage():
    """Example of using the module programmatically."""
    from social_sentiment import SentimentAggregator

    # With mock mode
    config = SentimentConfig(mock_mode=True)
    aggregator = SentimentAggregator(config)

    # Single symbol
    result = await aggregator.get_sentiment("BTC")
    print(f"BTC Sentiment: {result.sentiment_score}")

    # Multiple symbols
    batch = await aggregator.get_batch_sentiment(["BTC", "ETH", "SOL"])
    for symbol, data in batch.items():
        print(f"{symbol}: {data.sentiment_score}")

    await aggregator.close()


async def example_api_format():
    """Show the expected API output format."""
    return {
        "symbol": "BTC",
        "sentiment_score": 68.5,  # 0-100
        "fear_greed_index": 55,  # 0-100
        "twitter_volume": 15000,  # discussion volume
        "reddit_sentiment": 0.65,  # -1 to 1
        "twitter_sentiment": 0.72,  # -1 to 1
        "timestamp": "2026-03-26T13:11:00",
        "discussion_volume": 15500,
        "twitter_posts_analyzed": 50,
        "reddit_posts_analyzed": 25,
        "data_sources": ["twitter", "reddit", "fear_greed"],
    }
