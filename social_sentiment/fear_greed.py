"""Fear & Greed Index Client using Alternative.me API (free)."""

import asyncio
import aiohttp
import time
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from datetime import datetime


@dataclass
class FearGreedData:
    """Fear & Greed index data."""
    value: int  # 0-100
    value_classification: str  # "Fear", "Greed", "Neutral", "Extreme Fear", "Extreme Greed"
    timestamp: datetime
    time_until_update: int  # seconds until next update


class FearGreedClient:
    """
    Client for Alternative.me Fear & Greed API.

    API is completely free, no authentication required.
    Updates every 8 hours.
    """

    def __init__(self, api_url: str = "https://api.alternative.me/fng/"):
        self.api_url = api_url
        self._session: Optional[aiohttp.ClientSession] = None
        self._cache: Optional[FearGreedData] = None
        self._cache_time: float = 0
        self._cache_ttl: int = 3600  # Cache for 1 hour

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def get_current(self, use_cache: bool = True) -> FearGreedData:
        """
        Get current Fear & Greed index.

        Args:
            use_cache: Use cached data if available

        Returns:
            FearGreedData with current index
        """
        # Check cache
        if use_cache and self._cache and (time.time() - self._cache_time) < self._cache_ttl:
            return self._cache

        session = await self._get_session()

        try:
            async with session.get(
                self.api_url, timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    fear_greed_data = self._parse_response(data)

                    # Update cache
                    self._cache = fear_greed_data
                    self._cache_time = time.time()

                    return fear_greed_data
                else:
                    # Return cached data on error (if available)
                    if self._cache:
                        return self._cache
                    raise Exception(f"Failed to fetch Fear & Greed: {response.status}")
        except Exception as e:
            # Return cached data on error
            if self._cache:
                return self._cache
            raise e

    def _parse_response(self, data: Dict[str, Any]) -> FearGreedData:
        """Parse Alternative.me API response."""
        # API returns data[0] for current, data[1:] for historical
        if not data.get("data"):
            raise Exception("Invalid API response: no data")

        current = data["data"][0]

        return FearGreedData(
            value=int(current.get("value", 50)),
            value_classification=current.get("value_classification", "Neutral"),
            timestamp=datetime.fromtimestamp(int(current.get("timestamp", 0))),
            time_until_update=int(current.get("time_until_update", 0)),
        )

    async def get_historical(
        self, limit: int = 10
    ) -> List[FearGreedData]:
        """
        Get historical Fear & Greed data.

        Args:
            limit: Number of historical data points (max ~100)

        Returns:
            List of FearGreedData sorted by most recent first
        """
        session = await self._get_session()
        url = f"{self.api_url}?limit={min(limit, 100)}"

        try:
            async with session.get(
                url, timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return [
                        FearGreedData(
                            value=int(item.get("value", 50)),
                            value_classification=item.get("value_classification", "Neutral"),
                            timestamp=datetime.fromtimestamp(int(item.get("timestamp", 0))),
                            time_until_update=int(item.get("time_until_update", 0)),
                        )
                        for item in data.get("data", [])
                    ]
                else:
                    return []
        except Exception:
            return []

    def get_classification(self, value: int) -> str:
        """
        Get classification string for a value.

        Args:
            value: Fear & Greed value (0-100)

        Returns:
            Classification string
        """
        if value <= 10:
            return "Extreme Fear"
        elif value <= 25:
            return "Fear"
        elif value <= 45:
            return "Greed"
        elif value <= 55:
            return "Neutral"
        elif value <= 75:
            return "Greed"
        else:
            return "Extreme Greed"

    async def close(self):
        """Close the HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()


# Mock data for offline testing
MOCK_FEAR_GREED_DATA = FearGreedData(
    value=65,
    value_classification="Greed",
    timestamp=datetime.now(),
    time_until_update=3600,
)


class MockFearGreedClient(FearGreedClient):
    """Mock Fear & Greed client for offline testing."""

    async def get_current(self, use_cache: bool = True) -> FearGreedData:
        """Return mock Fear & Greed data."""
        return MOCK_FEAR_GREED_DATA

    async def get_historical(self, limit: int = 10) -> List[FearGreedData]:
        """Return mock historical Fear & Greed data."""
        base_value = MOCK_FEAR_GREED_DATA.value
        results = []
        for i in range(limit):
            value = max(0, min(100, base_value - (i * 5) + (i % 3) * 10))
            results.append(
                FearGreedData(
                    value=int(value),
                    value_classification=self.get_classification(int(value)),
                    timestamp=datetime.now(),
                    time_until_update=0,
                )
            )
        return results
