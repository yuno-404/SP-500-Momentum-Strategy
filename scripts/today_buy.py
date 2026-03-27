"""
TODAY'S BUY SIGNALS
===================
Runs the momentum strategy on latest data and outputs exactly what to buy.

Usage:
    python today_buy.py                  # Default $30,000
    python today_buy.py --capital 50000  # Custom capital
"""

import argparse
import warnings
import math
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

from data_manager import DataManager
from strategy import MomentumStrategy
from config import TOP_N_PER_SECTOR, SECTOR_ETFS


def generate_signals(capital=30000):
    print("=" * 65)
    print(f"  TODAY'S BUY LIST  |  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  Capital: ${capital:,.0f}  |  Top {TOP_N_PER_SECTOR} per sector")
    print("=" * 65)

    # ── Load Data (need ~7 months of history for momentum) ───────────
    start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")

    dm = DataManager()
    ticker_sector = dm.get_sp500_components()
    sector_weights = dm.download_sector_etf_weights(start_date)
    stock_prices = dm.download_stock_prices(ticker_sector.keys(), start_date)

    # ── Calculate Momentum ───────────────────────────────────────────
    strategy = MomentumStrategy(
        prices=stock_prices,
        ticker_sector=ticker_sector,
        sector_weights_ts=sector_weights,
        top_n=TOP_N_PER_SECTOR,
    )
    momentum = strategy.calculate_momentum()

    # Use the latest available date
    latest_date = momentum.index[-1]
    portfolio = strategy.select_portfolio(latest_date, momentum)

    if not portfolio:
        print("\nNo signals generated. Check data availability.")
        return

    # ── Build output table ───────────────────────────────────────────
    rows = []
    for ticker, info in portfolio.items():
        w = info["weight"]
        mom = info["momentum"]
        dollar = capital * w
        # Get latest price
        price = (
            stock_prices[ticker].dropna().iloc[-1]
            if ticker in stock_prices.columns
            else None
        )
        shares = int(dollar / price) if price and price > 0 else 0
        actual_dollar = shares * price if price else 0

        rows.append(
            {
                "sector": info["sector"],
                "ticker": ticker,
                "weight": w,
                "momentum": mom,
                "price": price,
                "shares": shares,
                "dollar": actual_dollar,
            }
        )

    rows.sort(key=lambda r: (r["sector"], -r["weight"]))

    # ── Print ────────────────────────────────────────────────────────
    sep = (
        "+"
        + "-" * 25
        + "+"
        + "-" * 8
        + "+"
        + "-" * 9
        + "+"
        + "-" * 10
        + "+"
        + "-" * 10
        + "+"
        + "-" * 8
        + "+"
        + "-" * 12
        + "+"
    )

    print(f"\n{sep}")
    print(
        f"| {'Sector':<23} | {'Ticker':<6} | {'Weight':>7} | {'Momentum':>8} | {'Price':>8} | {'Shares':>6} | {'$ Amount':>10} |"
    )
    print(sep.replace("-", "="))

    current_sector = None
    sector_totals = {}
    total_invested = 0
    total_shares_count = 0

    for r in rows:
        if r["sector"] != current_sector:
            if current_sector is not None:
                print(sep)
            current_sector = r["sector"]

        mom_str = (
            f"{r['momentum'] * 100:.1f}%"
            if not (math.isnan(r["momentum"]) or math.isinf(r["momentum"]))
            else "N/A"
        )
        price_str = f"${r['price']:.2f}" if r["price"] else "N/A"

        print(
            f"| {r['sector']:<23} | {r['ticker']:<6} | {r['weight'] * 100:>6.2f}% | {mom_str:>8} | {price_str:>8} | {r['shares']:>6} | ${r['dollar']:>9,.0f} |"
        )

        s = r["sector"]
        sector_totals[s] = sector_totals.get(s, 0) + r["dollar"]
        total_invested += r["dollar"]
        total_shares_count += r["shares"]

    print(sep)

    # ── Summary ──────────────────────────────────────────────────────
    cash_remaining = capital - total_invested
    print(f"\n{'SUMMARY':=^65}")
    print(f"  Total stocks:     {len(rows)}")
    print(f"  Total sectors:    {len(sector_totals)}")
    print(f"  Total invested:   ${total_invested:>12,.0f}")
    print(
        f"  Cash remaining:   ${cash_remaining:>12,.0f}  ({cash_remaining / capital * 100:.1f}%)"
    )
    print(f"  Signal date:      {latest_date.strftime('%Y-%m-%d')}")
    print("=" * 65)

    # ── Sector breakdown ─────────────────────────────────────────────
    print(f"\n{'SECTOR BREAKDOWN':=^65}")
    for sector in sorted(sector_totals.keys()):
        amt = sector_totals[sector]
        pct = amt / capital * 100
        bar = "#" * int(pct / 2)
        print(f"  {sector:<28} ${amt:>8,.0f}  ({pct:>5.1f}%) {bar}")
    print("=" * 65)

    return portfolio


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Today's buy signals")
    parser.add_argument(
        "--capital", type=float, default=30000, help="Total capital (default: $30,000)"
    )
    args = parser.parse_args()
    generate_signals(capital=args.capital)
