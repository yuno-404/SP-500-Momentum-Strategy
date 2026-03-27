import math

from strategy import MomentumStrategy


class SignalService:
    """Generate strategy signals payloads."""

    def __init__(self, market_data_service, top_n):
        self.market_data_service = market_data_service
        self.top_n = top_n

    def get_latest_portfolio(self, start_date):
        ticker_sector, sector_weights, stock_prices = (
            self.market_data_service.load_strategy_data(start_date)
        )
        strategy = MomentumStrategy(
            stock_prices, ticker_sector, sector_weights, top_n=self.top_n
        )
        momentum = strategy.calculate_momentum()
        latest_date = momentum.index[-1]
        portfolio = strategy.select_portfolio(latest_date, momentum)
        return latest_date, portfolio, stock_prices

    def to_signal_rows(self, portfolio, stock_prices):
        rows = []
        for ticker, info in portfolio.items():
            price = self.market_data_service.get_last_price_from_history(
                stock_prices, ticker
            )
            momentum = info["momentum"]
            rows.append(
                {
                    "ticker": ticker,
                    "sector": info["sector"],
                    "weight": round(info["weight"], 6),
                    "momentum": round(momentum * 100, 2)
                    if not (math.isnan(momentum) or math.isinf(momentum))
                    else 0,
                    "price": round(price, 2) if price else None,
                }
            )
        rows.sort(key=lambda x: x["sector"])
        return rows
