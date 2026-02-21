"""
Parameter sensitivity sweep for robustness checks.

Runs multiple backtests across:
- top_n
- transaction cost
- slippage cost

Outputs: sensitivity_report.csv
"""

import argparse
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from backtester import Backtester
from config import (
    TOP_N_PER_SECTOR,
    TRANSACTION_COST,
    SLIPPAGE_COST,
    REQUIRE_LOCAL_SECTOR_AUM_FOR_BACKTEST,
)
from data_manager import DataManager
from strategy import MomentumStrategy


def cagr(values, years):
    if len(values) < 2 or years <= 0 or values[0] <= 0:
        return 0.0
    return (values[-1] / values[0]) ** (1.0 / years) - 1.0


def run_sweep(years=10):
    start_date = (datetime.now() - timedelta(days=365 * years)).strftime("%Y-%m-%d")
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    dm = DataManager(use_cache=True)
    ticker_sector = dm.get_sp500_components()
    pit_universe = dm.get_point_in_time_universe_by_month(start_date)
    pit_tickers = sorted(
        {t for month_map in pit_universe.values() for t in month_map.keys()}
    )
    sector_weights = dm.download_sector_etf_weights(
        start_date,
        require_local_csv=REQUIRE_LOCAL_SECTOR_AUM_FOR_BACKTEST,
    )
    stock_prices = dm.download_stock_prices(
        pit_tickers if pit_tickers else ticker_sector.keys(), start_date
    )
    benchmark = dm.download_benchmark(start_date)

    top_ns = sorted(set([1, TOP_N_PER_SECTOR, 2, 3]))
    tcosts = sorted(set([TRANSACTION_COST, 0.0005, 0.0015]))
    scosts = sorted(set([SLIPPAGE_COST, 0.0000, 0.0010]))

    rows = []
    for top_n in top_ns:
        strategy = MomentumStrategy(
            stock_prices,
            ticker_sector,
            sector_weights,
            top_n=top_n,
            point_in_time_universe=pit_universe,
        )

        for tc in tcosts:
            for sc in scosts:
                bt = Backtester(
                    strategy,
                    benchmark,
                    signal_lag_months=1,
                    transaction_cost=tc,
                    slippage_cost=sc,
                )
                results = bt.run()

                values = np.array(results["portfolio_values"][7:], dtype=float)
                dates = results["dates"][7:]
                if len(values) < 2:
                    continue

                years_actual = (dates[-1] - dates[0]).days / 365.25
                total_ret = values[-1] / values[0] - 1.0
                rows.append(
                    {
                        "top_n": top_n,
                        "transaction_cost": tc,
                        "slippage_cost": sc,
                        "total_return": float(total_ret),
                        "cagr": float(cagr(values, years_actual)),
                        "num_trades": len(results.get("trades", [])),
                    }
                )

    df = pd.DataFrame(rows)
    if len(df) == 0:
        print("No valid sensitivity results.")
        return

    df = df.sort_values("cagr", ascending=False)
    output_path = reports_dir / "sensitivity_report.csv"
    df.to_csv(str(output_path), index=False)

    print("=" * 72)
    print("SENSITIVITY SUMMARY")
    print("=" * 72)
    print(f"Runs: {len(df)}")
    print(f"Best CAGR: {df.iloc[0]['cagr'] * 100:.2f}%")
    print(f"Worst CAGR: {df.iloc[-1]['cagr'] * 100:.2f}%")
    print(f"Report: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backtest parameter sensitivity sweep")
    parser.add_argument("--years", type=int, default=10)
    args = parser.parse_args()
    run_sweep(years=args.years)
