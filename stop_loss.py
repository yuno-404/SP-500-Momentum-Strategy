"""
Stop Loss Manager - Risk management with multiple stop loss strategies
"""

import pandas as pd
import numpy as np
from datetime import datetime


class StopLossManager:
    """
    Manage stop loss rules for the portfolio

    Supports:
    1. Individual stock stop loss (from purchase price)
    2. Trailing stop loss (from peak price)
    3. Portfolio-level stop loss
    """

    def __init__(
        self,
        stock_stop_loss=-0.12,  # -12% from purchase
        trailing_stop=-0.15,  # -15% from peak
        portfolio_stop_loss=-0.10,  # -10% portfolio drawdown
        portfolio_halt=-0.15,
    ):  # -15% full exit
        """
        Args:
            stock_stop_loss: Sell if stock drops X% from purchase price
            trailing_stop: Sell if stock drops X% from its peak
            portfolio_stop_loss: Reduce positions if portfolio drops X%
            portfolio_halt: Exit all positions if portfolio drops X%
        """
        self.stock_stop_loss = stock_stop_loss
        self.trailing_stop = trailing_stop
        self.portfolio_stop_loss = portfolio_stop_loss
        self.portfolio_halt = portfolio_halt

        # Track purchase prices and peaks
        self.purchase_prices = {}  # {ticker: price}
        self.peak_prices = {}  # {ticker: peak_price}
        self.portfolio_peak = None

    def _get_price_row(self, prices, date):
        """Get nearest available trading-day price row at or before date."""
        if date in prices.index:
            return prices.loc[date], date

        available_dates = prices.index[prices.index <= date]
        if len(available_dates) == 0:
            return None, None

        actual_date = available_dates[-1]
        return prices.loc[actual_date], actual_date

    def update_purchase_prices(self, portfolio, prices, date):
        """Record purchase prices for new positions"""
        current_prices, _ = self._get_price_row(prices, date)

        if current_prices is None:
            return

        for ticker in portfolio.keys():
            if ticker not in self.purchase_prices and ticker in current_prices.index:
                self.purchase_prices[ticker] = current_prices[ticker]
                self.peak_prices[ticker] = current_prices[ticker]

    def update_peaks(self, portfolio, prices, date):
        """Update peak prices for trailing stop"""
        current_prices, _ = self._get_price_row(prices, date)

        if current_prices is None:
            return

        for ticker in portfolio.keys():
            if ticker in current_prices.index:
                current_price = current_prices[ticker]

                # Update peak if new high
                if ticker in self.peak_prices:
                    self.peak_prices[ticker] = max(
                        self.peak_prices[ticker], current_price
                    )
                else:
                    self.peak_prices[ticker] = current_price

    def check_stock_stops(self, portfolio, prices, date):
        """
        Check if any stocks hit stop loss

        Returns:
            list: Tickers to sell
        """
        to_sell = []
        current_prices, _ = self._get_price_row(prices, date)

        if current_prices is None:
            return to_sell

        for ticker in portfolio.keys():
            if ticker not in current_prices.index:
                continue

            current_price = current_prices[ticker]

            # Rule 1: Stop loss from purchase price
            if ticker in self.purchase_prices:
                purchase_price = self.purchase_prices[ticker]
                pct_change = (current_price - purchase_price) / purchase_price

                if pct_change <= self.stock_stop_loss:
                    to_sell.append(
                        {
                            "ticker": ticker,
                            "reason": "PURCHASE_STOP",
                            "loss": pct_change,
                            "purchase": purchase_price,
                            "current": current_price,
                        }
                    )
                    continue

            # Rule 2: Trailing stop from peak
            if ticker in self.peak_prices:
                peak_price = self.peak_prices[ticker]
                drawdown = (current_price - peak_price) / peak_price

                if drawdown <= self.trailing_stop:
                    to_sell.append(
                        {
                            "ticker": ticker,
                            "reason": "TRAILING_STOP",
                            "drawdown": drawdown,
                            "peak": peak_price,
                            "current": current_price,
                        }
                    )

        return to_sell

    def check_portfolio_stop(self, portfolio_value):
        """
        Check portfolio-level stop loss

        Returns:
            str: 'REDUCE' (reduce 50%), 'HALT' (exit all), or None
        """
        # Update portfolio peak
        if self.portfolio_peak is None:
            self.portfolio_peak = portfolio_value
        else:
            self.portfolio_peak = max(self.portfolio_peak, portfolio_value)

        # Calculate drawdown
        drawdown = (portfolio_value - self.portfolio_peak) / self.portfolio_peak

        # Check thresholds
        if drawdown <= self.portfolio_halt:
            return "HALT"
        elif drawdown <= self.portfolio_stop_loss:
            return "REDUCE"

        return None

    def remove_sold_stocks(self, sold_tickers):
        """Clean up tracking for sold stocks"""
        for item in sold_tickers:
            ticker = item["ticker"]
            if ticker in self.purchase_prices:
                del self.purchase_prices[ticker]
            if ticker in self.peak_prices:
                del self.peak_prices[ticker]

    def get_replacement_stocks(
        self, sold_stocks, sector_momentum, ticker_sector, current_portfolio, top_n=3
    ):
        """
        Find replacement stocks from same sector

        Args:
            sold_stocks: List of sold stock info
            sector_momentum: Current momentum scores
            ticker_sector: Ticker to sector mapping
            current_portfolio: Current holdings
            top_n: How many top stocks to consider per sector

        Returns:
            dict: {ticker: momentum_score} of replacement stocks
        """
        replacements = {}

        for item in sold_stocks:
            ticker = item["ticker"]
            sector = ticker_sector.get(ticker)

            if not sector:
                continue

            # Get all stocks in same sector
            sector_stocks = [t for t, s in ticker_sector.items() if s == sector]

            # Get momentum scores
            sector_scores = sector_momentum[sector_stocks].sort_values(ascending=False)

            # Find first stock not in current portfolio
            for candidate, score in sector_scores.items():
                if candidate not in current_portfolio and candidate not in replacements:
                    replacements[candidate] = score
                    break

        return replacements

    def get_stats(self):
        """Get stop loss statistics"""
        return {
            "tracked_stocks": len(self.purchase_prices),
            "portfolio_peak": self.portfolio_peak,
            "stock_stop_loss": f"{self.stock_stop_loss * 100:.1f}%",
            "trailing_stop": f"{self.trailing_stop * 100:.1f}%",
            "portfolio_stop": f"{self.portfolio_stop_loss * 100:.1f}%",
            "portfolio_halt": f"{self.portfolio_halt * 100:.1f}%",
        }


if __name__ == "__main__":
    # Test stop loss manager
    print("=" * 60)
    print("Stop Loss Manager Test")
    print("=" * 60)

    # Create manager
    sl = StopLossManager(
        stock_stop_loss=-0.12,
        trailing_stop=-0.15,
        portfolio_stop_loss=-0.10,
        portfolio_halt=-0.15,
    )

    # Simulate portfolio
    dates = pd.date_range("2020-01-01", periods=30, freq="D")
    prices = pd.DataFrame(
        {
            "AAPL": [100, 102, 105, 103, 98, 95, 87, 85, 90, 92]
            + [95] * 20,  # Triggers stop
            "MSFT": [200, 205, 210, 215, 220, 225, 230, 235, 240, 245]
            + [250] * 20,  # Good
            "NVDA": [300, 310, 320, 315, 310, 305, 300, 295, 290, 285]
            + [280] * 20,  # Trailing
        },
        index=dates,
    )

    portfolio = {"AAPL": {}, "MSFT": {}, "NVDA": {}}

    # Day 1: Purchase
    sl.update_purchase_prices(portfolio, prices, dates[0])
    print(f"\nDay 1 - Purchase prices: {sl.purchase_prices}")

    # Day 7: Check stops (AAPL should trigger)
    sl.update_peaks(portfolio, prices, dates[6])
    to_sell = sl.check_stock_stops(portfolio, prices, dates[6])
    print(f"\nDay 7 - Stocks to sell:")
    for item in to_sell:
        print(
            f"  {item['ticker']}: {item['reason']} ({item.get('loss', item.get('drawdown')) * 100:.1f}%)"
        )

    # Portfolio stop
    portfolio_action = sl.check_portfolio_stop(28000)  # Down from 30000
    print(f"\nPortfolio action: {portfolio_action}")

    print("\n" + "=" * 60)
