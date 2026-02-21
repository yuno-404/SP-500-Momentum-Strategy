"""CLI helpers for Financial Modeling Prep adapter."""

import argparse
import json
from datetime import datetime, timedelta

from config import SECTOR_ETFS
from fmp_adapter import FMPAdapter, FMPAPIError, FMPPlanLimitError


def _default_start(years_back: int = 5) -> str:
    return (datetime.now() - timedelta(days=365 * years_back)).strftime("%Y-%m-%d")


def cmd_quote(args):
    adapter = FMPAdapter()
    row = adapter.quote(args.symbol)
    print(json.dumps(row, ensure_ascii=True, indent=2))


def cmd_search(args):
    adapter = FMPAdapter()
    rows = adapter.search_name(args.query, limit=args.limit)
    print(json.dumps(rows, ensure_ascii=True, indent=2))


def cmd_export_sector_aum(args):
    adapter = FMPAdapter()
    start_date = args.start_date or _default_start(args.years_back)
    try:
        out = adapter.export_sector_proxy_aum_csv(
            output_path=args.output,
            sector_etfs=SECTOR_ETFS,
            start_date=start_date,
            end_date=args.end_date,
        )
    except FMPPlanLimitError as exc:
        print("[WARN] Export skipped due to plan limit.")
        print(str(exc))
        return

    print(f"[OK] Exported {len(out)} rows -> {args.output}")
    print("[WARN] This is proxy AUM (current assets scaled by historical prices).")


def main():
    parser = argparse.ArgumentParser(description="FMP adapter utilities")
    sub = parser.add_subparsers(dest="command", required=True)

    p_quote = sub.add_parser("quote", help="Fetch latest quote")
    p_quote.add_argument("--symbol", required=True)
    p_quote.set_defaults(func=cmd_quote)

    p_search = sub.add_parser("search", help="Search company names")
    p_search.add_argument("--query", required=True)
    p_search.add_argument("--limit", type=int, default=10)
    p_search.set_defaults(func=cmd_search)

    p_export = sub.add_parser(
        "export-sector-aum", help="Build and export proxy monthly sector AUM CSV"
    )
    p_export.add_argument("--output", default="data/sector_aum_monthly.csv")
    p_export.add_argument("--start-date", default=None)
    p_export.add_argument("--end-date", default=None)
    p_export.add_argument("--years-back", type=int, default=5)
    p_export.set_defaults(func=cmd_export_sector_aum)

    args = parser.parse_args()

    try:
        args.func(args)
    except FMPAPIError as exc:
        print(f"[ERROR] {exc}")
    except Exception as exc:
        print(f"[ERROR] Unexpected failure: {exc}")


if __name__ == "__main__":
    main()
