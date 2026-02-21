import math
from datetime import datetime, timedelta

from analyzer import PerformanceAnalyzer
from backtester import Backtester
from config import START_CAPITAL, REQUIRE_LOCAL_SECTOR_AUM_FOR_BACKTEST
from stop_loss import StopLossManager
from strategy import MomentumStrategy


class BacktestService:
    """Run backtests and format API responses."""

    def __init__(self, market_data_service, default_top_n):
        self.market_data_service = market_data_service
        self.default_top_n = default_top_n

    def run(self, data):
        years_back = data.get("years_back", 10)
        top_n = data.get("top_n", self.default_top_n)

        use_stop_loss = data.get("use_stop_loss", False)
        stock_stop_loss = data.get("stock_stop_loss", -0.12)
        trailing_stop = data.get("trailing_stop", -0.15)
        portfolio_stop_loss = data.get("portfolio_stop_loss", -0.10)
        portfolio_halt = data.get("portfolio_halt", -0.15)
        check_frequency = data.get("check_frequency", "weekly")
        execution_lag_months = data.get("execution_lag_months", 1)
        transaction_cost = data.get("transaction_cost", None)
        slippage_cost = data.get("slippage_cost", None)
        use_point_in_time_universe = data.get("use_point_in_time_universe", True)
        require_local_sector_aum = data.get(
            "require_local_sector_aum", REQUIRE_LOCAL_SECTOR_AUM_FOR_BACKTEST
        )

        start_date = (datetime.now() - timedelta(days=365 * years_back)).strftime(
            "%Y-%m-%d"
        )

        ticker_sector, sector_weights, stock_prices, benchmark, pit_universe = (
            self.market_data_service.load_backtest_data(
                start_date,
                use_point_in_time_universe=use_point_in_time_universe,
                require_local_sector_aum=require_local_sector_aum,
            )
        )

        strategy = MomentumStrategy(
            stock_prices,
            ticker_sector,
            sector_weights,
            top_n=top_n,
            point_in_time_universe=pit_universe if use_point_in_time_universe else None,
        )

        if use_stop_loss:
            stop_loss_manager = StopLossManager(
                stock_stop_loss=stock_stop_loss,
                trailing_stop=trailing_stop,
                portfolio_stop_loss=portfolio_stop_loss,
                portfolio_halt=portfolio_halt,
            )
            backtester = Backtester(
                strategy,
                benchmark,
                stop_loss_manager,
                check_frequency,
                signal_lag_months=execution_lag_months,
                transaction_cost=transaction_cost,
                slippage_cost=slippage_cost,
            )
        else:
            backtester = Backtester(
                strategy,
                benchmark,
                signal_lag_months=execution_lag_months,
                transaction_cost=transaction_cost,
                slippage_cost=slippage_cost,
            )

        results = backtester.run()
        analyzer = PerformanceAnalyzer(results)
        metrics = analyzer.calculate_metrics()
        portfolio_df = analyzer.get_portfolio_dataframe()

        response = {
            "status": "success",
            "parameters": {
                "years_back": years_back,
                "top_n": top_n,
                "start_date": start_date,
                "end_date": datetime.now().strftime("%Y-%m-%d"),
                "actual_start_date": results["dates"][0].strftime("%Y-%m-%d")
                if len(results["dates"]) > 0
                else start_date,
                "actual_end_date": results["dates"][-1].strftime("%Y-%m-%d")
                if len(results["dates"]) > 0
                else datetime.now().strftime("%Y-%m-%d"),
                "total_months": len(results["dates"]),
                "start_capital": START_CAPITAL,
                "execution_lag_months": execution_lag_months,
                "use_point_in_time_universe": use_point_in_time_universe,
                "require_local_sector_aum": require_local_sector_aum,
            },
            "metrics": self._serialize_metrics(metrics),
            "stop_loss_stats": self._stop_loss_stats(results)
            if use_stop_loss
            else None,
            "portfolio": portfolio_df.to_dict("records"),
            "performance": {
                "dates": [d.strftime("%Y-%m-%d") for d in results["dates"][7:]],
                "portfolio_values": [float(v) for v in results["portfolio_values"][7:]],
                "spy_values": [
                    float(v) for v in results["benchmark_values"].get("SPY", [])[7:]
                ]
                if isinstance(results["benchmark_values"], dict)
                else [],
                "spmo_values": [
                    float(v) for v in results["benchmark_values"].get("SPMO", [])[7:]
                ]
                if isinstance(results["benchmark_values"], dict)
                else [],
                "spx_values": [
                    float(v) for v in results["benchmark_values"].get("SPX", [])[7:]
                ]
                if isinstance(results["benchmark_values"], dict)
                else [],
                "benchmark_values": [
                    float(v)
                    for v in (
                        results["benchmark_values"].get("SPY", [])[7:]
                        if isinstance(results["benchmark_values"], dict)
                        else results["benchmark_values"][7:]
                    )
                ],
            },
            "sector_allocation": portfolio_df.groupby("Sector").size().to_dict()
            if len(portfolio_df) > 0 and "Sector" in portfolio_df.columns
            else {},
            "debug": {
                "initial_capital": START_CAPITAL,
                "final_portfolio_value": float(results["portfolio_values"][-1])
                if len(results["portfolio_values"]) > 0
                else 0,
                "final_spy_value": float(
                    results["benchmark_values"].get("SPY", [0])[-1]
                )
                if isinstance(results["benchmark_values"], dict)
                and len(results["benchmark_values"].get("SPY", [])) > 0
                else 0,
                "initial_spy_value": float(
                    results["benchmark_values"].get("SPY", [START_CAPITAL])[0]
                )
                if isinstance(results["benchmark_values"], dict)
                else START_CAPITAL,
                "spy_first_6_months": [
                    float(v) for v in results["benchmark_values"].get("SPY", [])[:6]
                ]
                if isinstance(results["benchmark_values"], dict)
                else [],
            },
        }
        return response

    @staticmethod
    def _serialize_metrics(metrics):
        out = {}
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                if math.isnan(value) or math.isinf(value):
                    out[key] = None
                else:
                    out[key] = float(value)
            elif hasattr(value, "strftime"):
                out[key] = value.strftime("%Y-%m-%d")
            else:
                out[key] = value
        return out

    @staticmethod
    def _stop_loss_stats(results):
        events = results.get("stop_loss_events", [])
        return {
            "total_events": len(events),
            "stock_stops": len(
                [e for e in events if e["reason"] in ["PURCHASE_STOP", "TRAILING_STOP"]]
            ),
            "portfolio_actions": len(
                [
                    e
                    for e in events
                    if e["reason"] in ["PORTFOLIO_REDUCE", "PORTFOLIO_HALT"]
                ]
            ),
        }
