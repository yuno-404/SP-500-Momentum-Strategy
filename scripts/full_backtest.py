"""
Complete Backtest with Full Benchmark Comparison
=================================================
Runs the S&P 500 Sector Momentum Strategy and produces a comprehensive
comparison table against SPY, SPMO, and SPX (S&P 500 Index).

Metrics: Sharpe Ratio, MDD, Sortino, Calmar, Annualized Return, Volatility,
         Win Rate, Best/Worst Month, and more.

Usage:
    python full_backtest.py              # Default 10 years
    python full_backtest.py --years 5    # Custom period
"""

import argparse
import warnings
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for saving
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

warnings.filterwarnings("ignore")

from data_manager import DataManager
from strategy import MomentumStrategy
from backtester import Backtester
from stop_loss import StopLossManager
from config import (
    TOP_N_PER_SECTOR,
    START_CAPITAL,
    REQUIRE_LOCAL_SECTOR_AUM_FOR_BACKTEST,
)


# ── Helper Functions ─────────────────────────────────────────────────────────


def compute_monthly_returns(values):
    """Convert portfolio value series to monthly return series."""
    arr = np.array(values, dtype=float)
    return np.diff(arr) / arr[:-1]


def annualized_return(values, years):
    """CAGR from value series."""
    if values[0] <= 0 or years <= 0:
        return 0.0
    return (values[-1] / values[0]) ** (1 / years) - 1


def annualized_volatility(monthly_returns):
    """Annualized volatility from monthly returns."""
    return np.std(monthly_returns, ddof=1) * np.sqrt(12)


def sharpe_ratio(monthly_returns, risk_free_annual=0.02):
    """Sharpe ratio (annualized)."""
    excess = np.mean(monthly_returns) * 12 - risk_free_annual
    vol = np.std(monthly_returns, ddof=1) * np.sqrt(12)
    return excess / vol if vol > 0 else 0.0


def sortino_ratio(monthly_returns, risk_free_annual=0.02):
    """Sortino ratio (annualized, downside deviation)."""
    excess = np.mean(monthly_returns) * 12 - risk_free_annual
    downside = monthly_returns[monthly_returns < 0]
    down_vol = np.std(downside, ddof=1) * np.sqrt(12) if len(downside) > 1 else 0
    return excess / down_vol if down_vol > 0 else 0.0


def max_drawdown(values):
    """Maximum drawdown (negative number)."""
    arr = np.array(values, dtype=float)
    peak = np.maximum.accumulate(arr)
    dd = (arr - peak) / peak
    return np.min(dd)


def max_drawdown_duration(values, dates):
    """Duration of worst drawdown in months."""
    arr = np.array(values, dtype=float)
    peak = np.maximum.accumulate(arr)
    dd = (arr - peak) / peak

    worst_idx = np.argmin(dd)
    # Find when peak was set
    peak_val = peak[worst_idx]
    peak_idx = np.where(arr == peak_val)[0][0]

    # Find recovery (if any)
    recovery_idx = len(arr) - 1  # default: never recovered
    for j in range(worst_idx, len(arr)):
        if arr[j] >= peak_val:
            recovery_idx = j
            break

    return recovery_idx - peak_idx


def calmar_ratio(values, years):
    """Calmar ratio = annualized return / |max drawdown|."""
    cagr = annualized_return(values, years)
    mdd = abs(max_drawdown(values))
    return cagr / mdd if mdd > 0 else 0.0


def win_rate(monthly_returns):
    """Fraction of positive months."""
    if len(monthly_returns) == 0:
        return 0.0
    return np.sum(monthly_returns > 0) / len(monthly_returns)


def best_month(monthly_returns):
    return np.max(monthly_returns) if len(monthly_returns) > 0 else 0.0


def worst_month(monthly_returns):
    return np.min(monthly_returns) if len(monthly_returns) > 0 else 0.0


def avg_positive_month(monthly_returns):
    pos = monthly_returns[monthly_returns > 0]
    return np.mean(pos) if len(pos) > 0 else 0.0


def avg_negative_month(monthly_returns):
    neg = monthly_returns[monthly_returns < 0]
    return np.mean(neg) if len(neg) > 0 else 0.0


# ── Main Pipeline ────────────────────────────────────────────────────────────


def run_full_backtest(
    years_back=10,
    use_stop_loss=False,
    save_plots=True,
    use_point_in_time_universe=True,
    require_local_sector_aum=REQUIRE_LOCAL_SECTOR_AUM_FOR_BACKTEST,
):
    """
    Run complete backtest and produce full comparison.

    Returns:
        dict with keys: metrics_table (DataFrame), strategy_results, figures
    """
    print("=" * 72)
    print("   S&P 500 SECTOR MOMENTUM STRATEGY - FULL BACKTEST REPORT")
    print("=" * 72)

    start_date = (datetime.now() - timedelta(days=365 * years_back)).strftime(
        "%Y-%m-%d"
    )
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Data ─────────────────────────────────────────────────────
    print("\n[1/5] Loading data ...")
    dm = DataManager()
    ticker_sector = dm.get_sp500_components()
    sector_weights = dm.download_sector_etf_weights(
        start_date,
        require_local_csv=require_local_sector_aum,
    )
    pit_universe = None
    if use_point_in_time_universe:
        pit_universe = dm.get_point_in_time_universe_by_month(start_date)
        pit_tickers = sorted(
            {t for month_map in pit_universe.values() for t in month_map.keys()}
        )
        tickers = pit_tickers if pit_tickers else list(ticker_sector.keys())
    else:
        tickers = list(ticker_sector.keys())

    stock_prices = dm.download_stock_prices(tickers, start_date)
    benchmark = dm.download_benchmark(start_date)

    # ── Step 2: Strategy ─────────────────────────────────────────────────
    print("\n[2/5] Initializing strategy ...")
    strategy = MomentumStrategy(
        prices=stock_prices,
        ticker_sector=ticker_sector,
        sector_weights_ts=sector_weights,
        top_n=TOP_N_PER_SECTOR,
        point_in_time_universe=pit_universe if use_point_in_time_universe else None,
    )

    # ── Step 3: Backtest ─────────────────────────────────────────────────
    print("\n[3/5] Running backtest ...")
    stop_loss_mgr = None
    if use_stop_loss:
        stop_loss_mgr = StopLossManager()

    backtester = Backtester(strategy, benchmark, stop_loss_manager=stop_loss_mgr)
    results = backtester.run()

    # ── Step 4: Compute Metrics ──────────────────────────────────────────
    print("\n[4/5] Computing performance metrics ...")

    # Strategy values (skip 7-month warm-up)
    strat_values = np.array(results["portfolio_values"][7:], dtype=float)
    dates = results["dates"][7:]

    # Benchmark values
    bench_dict = results["benchmark_values"]
    spy_values = np.array(bench_dict["SPY"][7:], dtype=float)
    spmo_values = np.array(
        bench_dict.get("SPMO", [START_CAPITAL] * len(strat_values))[7:], dtype=float
    )
    spx_values = np.array(
        bench_dict.get("SPX", [START_CAPITAL] * len(strat_values))[7:], dtype=float
    )

    # Compute years from actual investable date range
    actual_years = (dates[-1] - dates[0]).days / 365.25
    if actual_years <= 0:
        actual_years = len(strat_values) / 12

    # Monthly returns for each series
    strat_rets = compute_monthly_returns(strat_values)
    spy_rets = compute_monthly_returns(spy_values)
    spmo_rets = compute_monthly_returns(spmo_values)
    spx_rets = compute_monthly_returns(spx_values)

    def build_row(name, values, rets, years):
        base_value = float(values[0]) if len(values) > 0 else float(START_CAPITAL)
        if base_value <= 0:
            base_value = float(START_CAPITAL)
        growth = float(values[-1]) / base_value if len(values) > 0 else 1.0

        return {
            "Total Return": f"{(growth - 1) * 100:.2f}%",
            "Annualized Return (CAGR)": f"{annualized_return(values, years) * 100:.2f}%",
            "Annualized Volatility": f"{annualized_volatility(rets) * 100:.2f}%",
            "Sharpe Ratio": f"{sharpe_ratio(rets):.3f}",
            "Sortino Ratio": f"{sortino_ratio(rets):.3f}",
            "Calmar Ratio": f"{calmar_ratio(values, years):.3f}",
            "Max Drawdown (MDD)": f"{max_drawdown(values) * 100:.2f}%",
            "MDD Duration (months)": f"{max_drawdown_duration(values, dates)}",
            "Win Rate (monthly)": f"{win_rate(rets) * 100:.1f}%",
            "Best Month": f"{best_month(rets) * 100:.2f}%",
            "Worst Month": f"{worst_month(rets) * 100:.2f}%",
            "Avg Positive Month": f"{avg_positive_month(rets) * 100:.2f}%",
            "Avg Negative Month": f"{avg_negative_month(rets) * 100:.2f}%",
            "Final Value ($30K start)": f"${START_CAPITAL * growth:,.0f}",
        }

    rows = {
        "Strategy": build_row("Strategy", strat_values, strat_rets, actual_years),
        "SPY (S&P 500 ETF)": build_row("SPY", spy_values, spy_rets, actual_years),
        "SPMO (Momentum ETF)": build_row("SPMO", spmo_values, spmo_rets, actual_years),
        "SPX (S&P 500 Index)": build_row("SPX", spx_values, spx_rets, actual_years),
    }

    metrics_df = pd.DataFrame(rows).T
    metrics_df.index.name = "Portfolio"

    # ── Step 5: Display & Save ───────────────────────────────────────────
    print("\n[5/5] Generating reports ...\n")

    # Column headers
    col_names = ["Strategy", "SPY", "SPMO", "SPX"]
    cw = 14  # column width
    label_w = 26  # label column width
    sep_line = "+" + "-" * (label_w + 2) + ("+" + "-" * (cw + 2)) * 4 + "+"

    def table_row(label, vals):
        """Format a table row with fixed-width columns."""
        cells = "".join(f"| {v:>{cw}} " for v in vals)
        return f"| {label:<{label_w}} {cells}|"

    print(sep_line)
    print(table_row("Metric", col_names))
    print(sep_line.replace("-", "="))

    # Iterate metrics in order
    metric_keys = list(metrics_df.columns)
    for key in metric_keys:
        vals = [metrics_df.loc[idx, key] for idx in metrics_df.index]
        print(table_row(key, vals))
    print(sep_line)

    # ── Excess Return vs Benchmarks ──────────────────────────────────────
    strat_cagr = annualized_return(strat_values, actual_years)
    spy_cagr = annualized_return(spy_values, actual_years)
    spmo_cagr = annualized_return(spmo_values, actual_years)
    spx_cagr = annualized_return(spx_values, actual_years)

    print("")
    print(sep_line)
    print(table_row("EXCESS vs Benchmark", ["", "SPY", "SPMO", "SPX"]))
    print(sep_line.replace("-", "="))
    print(
        table_row(
            "Annualized Alpha",
            [
                "",
                f"{(strat_cagr - spy_cagr) * 100:+.2f}%",
                f"{(strat_cagr - spmo_cagr) * 100:+.2f}%",
                f"{(strat_cagr - spx_cagr) * 100:+.2f}%",
            ],
        )
    )
    print(
        table_row(
            "Total Excess Return",
            [
                "",
                f"{((strat_values[-1] - spy_values[-1]) / spy_values[0]) * 100:+.2f}%",
                f"{((strat_values[-1] - spmo_values[-1]) / spmo_values[0]) * 100:+.2f}%",
                f"{((strat_values[-1] - spx_values[-1]) / spx_values[0]) * 100:+.2f}%",
            ],
        )
    )
    print(sep_line)

    # ── Strategy-specific info ───────────────────────────────────────────
    if "trades" in results and len(results["trades"]) > 0:
        print(f"\nTotal Trades:          {len(results['trades'])}")
    if (
        "returns_df" in results
        and len(results["returns_df"]) > 0
        and "cost" in results["returns_df"].columns
    ):
        total_costs = results["returns_df"]["cost"].sum()
        print(f"Total Transaction Cost: {total_costs * 100:.4f}% of portfolio")
    if "stop_loss_events" in results and len(results["stop_loss_events"]) > 0:
        print(f"Stop Loss Events:      {len(results['stop_loss_events'])}")

    # ── Current Holdings ─────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("CURRENT PORTFOLIO HOLDINGS")
    print("=" * 72)
    final = results.get("final_portfolio", {})
    sectors = {}
    for ticker, info in sorted(final.items(), key=lambda x: x[1]["sector"]):
        s = info["sector"]
        if s not in sectors:
            sectors[s] = []
        sectors[s].append((ticker, info["weight"], info.get("momentum", 0)))

    for sector in sorted(sectors.keys()):
        total_w = sum(w for _, w, _ in sectors[sector])
        print(f"\n  {sector} ({total_w * 100:.1f}%)")
        for ticker, weight, mom in sectors[sector]:
            mom_str = (
                f"{mom * 100:.1f}%" if not (np.isnan(mom) or np.isinf(mom)) else "N/A"
            )
            print(f"    {ticker:<8}  Weight: {weight * 100:.2f}%   Momentum: {mom_str}")

    # ── Charts ───────────────────────────────────────────────────────────
    if save_plots:
        fig = _generate_comparison_charts(
            dates,
            strat_values,
            spy_values,
            spmo_values,
            spx_values,
            strat_rets,
            spy_rets,
            spmo_rets,
            spx_rets,
            results,
            actual_years,
        )
        output_path = reports_dir / "backtest_report.png"
        fig.savefig(str(output_path), dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"\nChart saved to: {output_path}")

    # Save metrics to CSV
    metrics_path = reports_dir / "backtest_metrics.csv"
    metrics_df.to_csv(str(metrics_path))
    print(f"Metrics saved to: {metrics_path}")

    print("\n" + "=" * 72)
    print("   BACKTEST COMPLETE")
    print("=" * 72)

    return {
        "metrics_table": metrics_df,
        "results": results,
        "values": {
            "strategy": strat_values,
            "spy": spy_values,
            "spmo": spmo_values,
            "spx": spx_values,
        },
        "returns": {
            "strategy": strat_rets,
            "spy": spy_rets,
            "spmo": spmo_rets,
            "spx": spx_rets,
        },
        "dates": dates,
        "years": actual_years,
    }


def _generate_comparison_charts(
    dates, strat_v, spy_v, spmo_v, spx_v, strat_r, spy_r, spmo_r, spx_r, results, years
):
    """Generate a 2x3 dashboard figure."""
    fig, axes = plt.subplots(2, 3, figsize=(22, 12))
    fig.suptitle(
        "S&P 500 Sector Momentum Strategy - Full Backtest Report",
        fontsize=18,
        fontweight="bold",
        y=0.98,
    )

    colors = {
        "strategy": "#2E86DE",
        "spy": "#E74C3C",
        "spmo": "#F39C12",
        "spx": "#8E44AD",
    }

    # ── 1. Cumulative Performance (Normalized to $1) ─────────────────────
    ax = axes[0, 0]
    norm = lambda v: np.array(v) / v[0]
    ax.plot(
        dates, norm(strat_v), label="Strategy", color=colors["strategy"], linewidth=2
    )
    ax.plot(
        dates,
        norm(spy_v),
        label="SPY",
        color=colors["spy"],
        linewidth=1.5,
        linestyle="--",
    )
    ax.plot(
        dates,
        norm(spmo_v),
        label="SPMO",
        color=colors["spmo"],
        linewidth=1.5,
        linestyle="-.",
    )
    ax.plot(
        dates,
        norm(spx_v),
        label="SPX",
        color=colors["spx"],
        linewidth=1.5,
        linestyle=":",
    )
    ax.set_title("Cumulative Performance (Growth of $1)", fontweight="bold")
    ax.set_ylabel("Growth")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("$%.1f"))

    # ── 2. Drawdown Comparison ───────────────────────────────────────────
    ax = axes[0, 1]

    def dd_series(values):
        peak = np.maximum.accumulate(values)
        return (values - peak) / peak * 100

    ax.fill_between(
        dates,
        dd_series(strat_v),
        0,
        alpha=0.3,
        color=colors["strategy"],
        label="Strategy",
    )
    ax.plot(
        dates,
        dd_series(spy_v),
        color=colors["spy"],
        linewidth=1,
        linestyle="--",
        label="SPY",
    )
    ax.plot(
        dates,
        dd_series(spmo_v),
        color=colors["spmo"],
        linewidth=1,
        linestyle="-.",
        label="SPMO",
    )
    ax.set_title("Drawdown (%)", fontweight="bold")
    ax.set_ylabel("Drawdown %")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # ── 3. Rolling 12-Month Return ───────────────────────────────────────
    ax = axes[0, 2]

    def rolling_12m(rets):
        if len(rets) < 12:
            return np.array([])
        r12 = []
        for i in range(12, len(rets) + 1):
            r12.append(np.prod(1 + rets[i - 12 : i]) - 1)
        return np.array(r12)

    r12_strat = rolling_12m(strat_r)
    r12_spy = rolling_12m(spy_r)
    r12_dates = dates[12 : 12 + len(r12_strat)]
    if len(r12_strat) > 0 and len(r12_dates) == len(r12_strat):
        ax.plot(
            r12_dates,
            r12_strat * 100,
            color=colors["strategy"],
            label="Strategy",
            linewidth=1.5,
        )
        ax.plot(
            r12_dates,
            r12_spy * 100,
            color=colors["spy"],
            label="SPY",
            linewidth=1,
            linestyle="--",
        )
        ax.axhline(0, color="gray", linewidth=0.8)
    ax.set_title("Rolling 12-Month Return (%)", fontweight="bold")
    ax.set_ylabel("Return %")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # ── 4. Monthly Returns Histogram ─────────────────────────────────────
    ax = axes[1, 0]
    ax.hist(
        strat_r * 100,
        bins=30,
        alpha=0.6,
        color=colors["strategy"],
        label="Strategy",
        edgecolor="white",
    )
    ax.hist(
        spy_r * 100,
        bins=30,
        alpha=0.4,
        color=colors["spy"],
        label="SPY",
        edgecolor="white",
    )
    ax.axvline(
        np.mean(strat_r) * 100, color=colors["strategy"], linestyle="--", linewidth=2
    )
    ax.axvline(np.mean(spy_r) * 100, color=colors["spy"], linestyle="--", linewidth=2)
    ax.set_title("Monthly Returns Distribution", fontweight="bold")
    ax.set_xlabel("Monthly Return (%)")
    ax.set_ylabel("Frequency")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis="y")

    # ── 5. Monthly Returns Bar (Strategy) ────────────────────────────────
    ax = axes[1, 1]
    bar_colors = ["#27AE60" if r > 0 else "#E74C3C" for r in strat_r]
    ax.bar(range(len(strat_r)), strat_r * 100, color=bar_colors, alpha=0.7, width=1.0)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Strategy Monthly Returns", fontweight="bold")
    ax.set_ylabel("Return (%)")
    ax.set_xlabel("Month")
    ax.grid(alpha=0.3, axis="y")

    # ── 6. Sector Allocation Pie Chart ───────────────────────────────────
    ax = axes[1, 2]
    final = results.get("final_portfolio", {})
    if final:
        sector_weights = {}
        for ticker, info in final.items():
            s = info["sector"]
            sector_weights[s] = sector_weights.get(s, 0) + info["weight"]
        labels = list(sector_weights.keys())
        sizes = list(sector_weights.values())
        colors_pie = plt.cm.Set3(np.linspace(0, 1, len(labels)))
        wedges, texts, autotexts = ax.pie(
            sizes,
            labels=None,
            autopct="%1.1f%%",
            colors=colors_pie,
            startangle=90,
            textprops={"fontsize": 8},
        )
        ax.legend(
            wedges,
            labels,
            title="Sectors",
            loc="center left",
            bbox_to_anchor=(1, 0, 0.5, 1),
            fontsize=7,
        )
    ax.set_title("Current Sector Allocation", fontweight="bold")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


# ── CLI Entry ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="S&P 500 Sector Momentum Backtest")
    parser.add_argument(
        "--years", type=int, default=10, help="Backtest period in years (default: 10)"
    )
    parser.add_argument(
        "--stop-loss", action="store_true", help="Enable stop loss management"
    )
    parser.add_argument("--no-plots", action="store_true", help="Skip chart generation")
    parser.add_argument(
        "--no-pit",
        action="store_true",
        help="Disable point-in-time universe (not recommended)",
    )
    parser.add_argument(
        "--allow-proxy-sector-aum",
        action="store_true",
        help="Allow fallback sector AUM proxies when local CSV is missing",
    )
    args = parser.parse_args()

    output = run_full_backtest(
        years_back=args.years,
        use_stop_loss=args.stop_loss,
        save_plots=not args.no_plots,
        use_point_in_time_universe=not args.no_pit,
        require_local_sector_aum=not args.allow_proxy_sector_aum,
    )
