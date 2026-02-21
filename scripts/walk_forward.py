"""
Walk-forward robustness report.

This is an out-of-sample style stability check:
- Use rolling windows (train months + test months)
- Run full backtest in each window
- Report test-window performance only

Usage:
    python walk_forward.py
    python walk_forward.py --years 12 --train-months 36 --test-months 12
"""

import argparse
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from data_manager import DataManager
from strategy import MomentumStrategy
from backtester import Backtester
from config import TOP_N_PER_SECTOR, REQUIRE_LOCAL_SECTOR_AUM_FOR_BACKTEST


def run_walk_forward(years=12, train_months=36, test_months=12):
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
    prices = dm.download_stock_prices(
        pit_tickers if pit_tickers else ticker_sector.keys(), start_date
    )
    benchmark = dm.download_benchmark(start_date)

    monthly_idx = prices.resample("ME").last().index
    warmup_months = 7
    window_months = warmup_months + train_months + test_months

    rows = []
    for start_i in range(0, len(monthly_idx) - window_months + 1, test_months):
        end_i = start_i + window_months - 1
        start_m = monthly_idx[start_i]
        end_m = monthly_idx[end_i]

        mask = (prices.index >= start_m) & (prices.index <= end_m)
        p_slice = prices.loc[mask]
        b_slice = benchmark.loc[
            (benchmark.index >= start_m) & (benchmark.index <= end_m)
        ]

        if len(p_slice) < 180 or len(b_slice) < 180:
            continue

        # Align sector weights to window
        sw_slice = sector_weights.loc[
            (sector_weights.index >= start_m) & (sector_weights.index <= end_m)
        ]
        if len(sw_slice) == 0:
            continue

        strategy = MomentumStrategy(
            p_slice,
            ticker_sector,
            sw_slice,
            top_n=TOP_N_PER_SECTOR,
            point_in_time_universe={
                d: m for d, m in pit_universe.items() if (d >= start_m and d <= end_m)
            },
        )
        bt = Backtester(strategy, b_slice)
        result = bt.run()

        returns_df = result.get("returns_df", pd.DataFrame())
        if len(returns_df) < test_months:
            continue

        test_df = returns_df.tail(test_months)
        cum = float(np.prod(1.0 + test_df["net_return"].values) - 1.0)
        years_test = test_months / 12.0
        cagr = float((1.0 + cum) ** (1.0 / years_test) - 1.0) if years_test > 0 else 0.0

        rows.append(
            {
                "window_start": start_m.strftime("%Y-%m"),
                "window_end": end_m.strftime("%Y-%m"),
                "test_months": test_months,
                "oos_return": cum,
                "oos_cagr": cagr,
            }
        )

    if not rows:
        print("No valid walk-forward windows produced.")
        return

    df = pd.DataFrame(rows)
    output_path = reports_dir / "walk_forward_report.csv"
    df.to_csv(str(output_path), index=False)

    print("=" * 72)
    print("WALK-FORWARD SUMMARY")
    print("=" * 72)
    print(f"Windows: {len(df)}")
    print(f"Avg OOS Return: {df['oos_return'].mean() * 100:.2f}%")
    print(f"Median OOS Return: {df['oos_return'].median() * 100:.2f}%")
    print(f"Negative OOS windows: {(df['oos_return'] < 0).sum()} / {len(df)}")
    print(f"Report: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Walk-forward robustness check")
    parser.add_argument("--years", type=int, default=12)
    parser.add_argument("--train-months", type=int, default=36)
    parser.add_argument("--test-months", type=int, default=12)
    args = parser.parse_args()
    run_walk_forward(args.years, args.train_months, args.test_months)
