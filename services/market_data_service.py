from datetime import datetime, timedelta

from data_manager import DataManager
from config import REQUIRE_LOCAL_SECTOR_AUM_FOR_BACKTEST


class MarketDataService:
    """Centralized data access for strategy and pricing."""

    def __init__(self, use_cache: bool = True):
        self.dm = DataManager(use_cache=use_cache)

    def default_start_date(self, days_back: int = 365) -> str:
        return (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")

    def load_strategy_data(self, start_date: str):
        ticker_sector = self.dm.get_sp500_components()
        sector_weights = self.dm.download_sector_etf_weights(start_date)
        stock_prices = self.dm.download_stock_prices(ticker_sector.keys(), start_date)
        return ticker_sector, sector_weights, stock_prices

    def load_backtest_data(
        self,
        start_date: str,
        use_point_in_time_universe: bool = True,
        require_local_sector_aum: bool = REQUIRE_LOCAL_SECTOR_AUM_FOR_BACKTEST,
    ):
        ticker_sector = self.dm.get_sp500_components()
        sector_weights = self.dm.download_sector_etf_weights(
            start_date,
            require_local_csv=require_local_sector_aum,
        )

        pit_universe = None
        if use_point_in_time_universe:
            end_date = datetime.now().strftime("%Y-%m-%d")
            pit_universe = self.dm.get_point_in_time_universe_by_month(
                start_date, end_date=end_date
            )
            pit_tickers = sorted(
                {t for month_map in pit_universe.values() for t in month_map.keys()}
            )
            tickers = pit_tickers if pit_tickers else list(ticker_sector.keys())
        else:
            tickers = list(ticker_sector.keys())

        stock_prices = self.dm.download_stock_prices(tickers, start_date)
        benchmark = self.dm.download_benchmark(start_date)
        return ticker_sector, sector_weights, stock_prices, benchmark, pit_universe

    def get_live_prices(self, tickers):
        import yfinance as yf

        prices = {}
        for ticker in sorted(set(tickers)):
            try:
                hist = yf.Ticker(ticker).history(period="1d")
                prices[ticker] = (
                    float(hist["Close"].iloc[-1]) if len(hist) > 0 else None
                )
            except Exception:
                prices[ticker] = None
        return prices

    @staticmethod
    def get_last_price_from_history(stock_prices, ticker):
        if ticker not in stock_prices.columns:
            return None
        series = stock_prices[ticker]
        if hasattr(series, "dropna"):
            series = series.dropna()
        if len(series) == 0:
            return None
        return float(series.iloc[-1])
