"""Build local monthly sector AUM CSV for offline backtests.

Usage:
    python build_sector_aum_csv.py
    python build_sector_aum_csv.py --years 10
    python build_sector_aum_csv.py --start-date 2015-01-01 --no-proxy-fallback
"""

import argparse
from datetime import datetime, timedelta

from config import SECTOR_AUM_CSV
from data_manager import DataManager


def _default_start_date(years):
    return (datetime.now() - timedelta(days=365 * years)).strftime("%Y-%m-%d")


def main():
    parser = argparse.ArgumentParser(description="Build sector_aum_monthly.csv")
    parser.add_argument("--years", type=int, default=10)
    parser.add_argument("--start-date", type=str, default=None)
    parser.add_argument("--output", type=str, default=SECTOR_AUM_CSV)
    parser.add_argument(
        "--no-proxy-fallback",
        action="store_true",
        help="Fail if shares*price is unavailable (do not fallback to proxy AUM).",
    )
    args = parser.parse_args()

    start_date = args.start_date or _default_start_date(args.years)
    dm = DataManager(use_cache=False)

    result = dm.export_sector_aum_csv_snapshot(
        start_date=start_date,
        output_path=args.output,
        allow_proxy_fallback=not args.no_proxy_fallback,
    )

    print("=" * 72)
    print("SECTOR AUM CSV BUILD COMPLETE")
    print("=" * 72)
    print(f"Output: {result['path']}")
    print(f"Rows:   {result['rows']}")
    print(f"Source: {result['source']}")
    if result["source"] != "shares_x_price":
        print(
            "[WARN] Source is proxy_current_aum_scaled (uses current AUM anchor). "
            "For strict anti-lookahead backtests, replace with true historical monthly AUM."
        )


if __name__ == "__main__":
    main()
