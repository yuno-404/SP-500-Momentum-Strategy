"""
Strategy Engine - Momentum calculation and portfolio selection
"""

import pandas as pd
import numpy as np
from config import MOMENTUM_4W, MOMENTUM_13W, MOMENTUM_26W


class MomentumStrategy:
    """
    Sector Momentum Rotation Strategy
    - Calculate momentum using 4W/13W/26W returns
    - Select top N stocks per sector
    - Weight by sector ETF market cap
    """

    def __init__(
        self,
        prices,
        ticker_sector,
        sector_weights_ts,
        top_n=3,
        point_in_time_universe=None,
    ):
        """
        Args:
            prices: DataFrame of stock prices
            ticker_sector: dict mapping ticker -> sector name
            sector_weights_ts: DataFrame of sector weights over time
            top_n: Number of stocks to select per sector
        """
        self.prices = prices
        self.ticker_sector = ticker_sector
        self.sector_weights_ts = sector_weights_ts
        self.top_n = top_n
        self.point_in_time_universe = point_in_time_universe

    def _get_ticker_sector_for_date(self, date):
        """Return ticker->sector map for given date (PIT if available)."""
        if not self.point_in_time_universe:
            return self.ticker_sector

        pit_dates = [d for d in self.point_in_time_universe.keys() if d <= date]
        if not pit_dates:
            return self.ticker_sector

        return self.point_in_time_universe[max(pit_dates)]

    def calculate_momentum(self):
        """
        Calculate momentum score as average of 4W/13W/26W returns
        Identifies stocks with sustained uptrend
        """
        ret_4w = self.prices.pct_change(MOMENTUM_4W, fill_method=None)
        ret_13w = self.prices.pct_change(MOMENTUM_13W, fill_method=None)
        ret_26w = self.prices.pct_change(MOMENTUM_26W, fill_method=None)

        # Equal weight average
        momentum = (ret_4w + ret_13w + ret_26w) / 3

        return momentum

    def select_portfolio(self, date, momentum):
        """
        Select portfolio for given date

        Returns:
            dict: {ticker: {'weight': float, 'sector': str, 'momentum': float}}
        """
        scores_at_date = momentum.loc[date]

        ticker_sector_for_date = self._get_ticker_sector_for_date(date)

        # Group all active tickers by sector (do not drop sectors with NaN scores).
        sector_to_tickers = {}
        for ticker, sector in ticker_sector_for_date.items():
            if ticker in scores_at_date.index and sector:
                sector_to_tickers.setdefault(sector, []).append(ticker)

        # Get sector weights for this date.
        sector_weights = self._get_sector_weights(date)
        if not sector_weights:
            # Fallback to equal sector weights across known sectors.
            unique_sectors = sorted(set(sector_to_tickers.keys()))
            if unique_sectors:
                equal_w = 1.0 / len(unique_sectors)
                sector_weights = {s: equal_w for s in unique_sectors}

        portfolio = {}
        missing_sectors = []

        # Select top N from each target sector.
        for sector, sector_weight in sector_weights.items():
            if sector_weight <= 0:
                continue

            sector_tickers = sector_to_tickers.get(sector, [])
            if not sector_tickers:
                missing_sectors.append(sector)
                continue

            sector_scores = (
                scores_at_date[sector_tickers].dropna().sort_values(ascending=False)
            )
            if len(sector_scores) == 0:
                missing_sectors.append(sector)
                continue

            top_stocks = sector_scores.head(self.top_n)
            stock_weight = sector_weight / len(top_stocks)

            for ticker, momentum_score in top_stocks.items():
                portfolio[ticker] = {
                    "weight": stock_weight,
                    "sector": sector,
                    "momentum": momentum_score,
                }

        # Normalize only for tiny floating errors when no sector is missing.
        total = sum(p["weight"] for p in portfolio.values())
        if total > 0 and not missing_sectors and abs(total - 1.0) < 1e-6:
            for ticker in portfolio:
                portfolio[ticker]["weight"] /= total

        return portfolio

    def _get_sector_weights(self, date):
        """Get sector weights for given date"""
        # Find closest previous date with weights
        available_dates = self.sector_weights_ts.index[
            self.sector_weights_ts.index <= date
        ]

        if len(available_dates) == 0:
            # No weights available, use equal weight
            return {}

        weights_series = self.sector_weights_ts.loc[available_dates[-1]]
        return weights_series.to_dict()


if __name__ == "__main__":
    # Test with dummy data
    import pandas as pd
    import numpy as np

    print("=" * 60)
    print("Strategy Engine Test")
    print("=" * 60)

    # Create dummy data
    dates = pd.date_range("2020-01-01", periods=200, freq="D")
    tickers = ["AAPL", "GOOGL", "MSFT", "JPM", "BAC", "XOM", "CVX"]

    np.random.seed(42)
    prices = pd.DataFrame(
        np.random.randn(200, 7).cumsum(axis=0) + 100, index=dates, columns=tickers
    )

    ticker_sector = {
        "AAPL": "Information Technology",
        "GOOGL": "Communication Services",
        "MSFT": "Information Technology",
        "JPM": "Financials",
        "BAC": "Financials",
        "XOM": "Energy",
        "CVX": "Energy",
    }

    # Dummy sector weights
    sector_weights = pd.DataFrame(
        {
            "Information Technology": [0.30] * 7,
            "Communication Services": [0.10] * 7,
            "Financials": [0.15] * 7,
            "Energy": [0.10] * 7,
        },
        index=pd.date_range("2020-01-31", periods=7, freq="M"),
    )

    # Test strategy
    strategy = MomentumStrategy(prices, ticker_sector, sector_weights, top_n=2)

    momentum = strategy.calculate_momentum()
    print("\n[OK] Momentum calculated")

    test_date = dates[150]
    portfolio = strategy.select_portfolio(test_date, momentum)

    print(f"\n[OK] Portfolio selected for {test_date.date()}:")
    for ticker, info in portfolio.items():
        print(f"  {ticker}: {info['weight'] * 100:.2f}% ({info['sector']})")

    total_weight = sum(p["weight"] for p in portfolio.values())
    print(f"\nTotal weight: {total_weight * 100:.2f}%")
