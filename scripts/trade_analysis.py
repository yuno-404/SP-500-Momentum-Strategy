"""
Trade-Level Strategy Analysis (No Compounding)
===============================================
Evaluates each monthly rebalance as an independent trade using fixed capital.
No compounding — every period uses the same capital base to isolate strategy quality.

Metrics:
  1. Mathematical Expectancy
  2. Information Coefficient (IC) & ICIR
  3. Entry / Exit / Total Efficiency
  4. MFE / MAE per trade
  5. Per-trade P&L breakdown

Usage:
    python scripts/trade_analysis.py              # Default 10 years
    python scripts/trade_analysis.py --years 20
"""

import argparse
import warnings
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

# Add parent to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_manager import DataManager
from strategy import MomentumStrategy
from config import (
    TOP_N_PER_SECTOR,
    START_CAPITAL,
    TRANSACTION_COST,
    SLIPPAGE_COST,
    REQUIRE_LOCAL_SECTOR_AUM_FOR_BACKTEST,
)


def run_trade_analysis(years_back=10):
    print("=" * 80)
    print("   TRADE-LEVEL STRATEGY ANALYSIS (FIXED CAPITAL, NO COMPOUNDING)")
    print("=" * 80)

    start_date = (datetime.now() - timedelta(days=365 * years_back)).strftime("%Y-%m-%d")

    # ── Load Data ─────────────────────────────────────────────────────────
    print("\n[1/4] Loading data ...")
    dm = DataManager()
    ticker_sector = dm.get_sp500_components()
    sector_weights = dm.download_sector_etf_weights(
        start_date, require_local_csv=REQUIRE_LOCAL_SECTOR_AUM_FOR_BACKTEST
    )
    pit_universe = dm.get_point_in_time_universe_by_month(start_date)
    pit_tickers = sorted({t for m in pit_universe.values() for t in m.keys()})
    tickers = pit_tickers if pit_tickers else list(ticker_sector.keys())

    stock_prices = dm.download_stock_prices(tickers, start_date)

    # ── Strategy Setup ────────────────────────────────────────────────────
    print("[2/4] Initializing strategy ...")
    strategy = MomentumStrategy(
        prices=stock_prices,
        ticker_sector=ticker_sector,
        sector_weights_ts=sector_weights,
        top_n=TOP_N_PER_SECTOR,
        point_in_time_universe=pit_universe,
    )

    momentum = strategy.calculate_momentum()
    monthly_prices = strategy.prices.resample("ME").last()
    monthly_momentum = momentum.resample("ME").last()
    daily_prices = strategy.prices  # for intra-month high/low

    # ── Run Trade-by-Trade Analysis ───────────────────────────────────────
    print("[3/4] Analyzing trades ...")

    trades = []  # list of per-stock trade records
    monthly_records = []  # list of per-month portfolio-level records
    ic_values = []  # IC per month

    signal_lag = 1
    capital = START_CAPITAL

    for i in range(8, len(monthly_momentum.index)):
        date = monthly_momentum.index[i]
        prev_date = monthly_momentum.index[i - 1]

        # Signal from lagged month
        signal_idx = i - signal_lag
        if signal_idx < 7:
            continue
        signal_date = monthly_momentum.index[signal_idx]
        portfolio = strategy.select_portfolio(signal_date, monthly_momentum)

        if not portfolio:
            continue

        # Total portfolio weight (may be < 1.0 if sectors missing)
        total_weight = sum(info["weight"] for info in portfolio.values())

        # Check capital sufficiency: capital must cover all positions
        # Each stock gets capital * weight allocation
        if total_weight <= 0:
            continue

        month_predicted = []  # predicted returns (momentum scores)
        month_actual = []  # actual returns
        month_gross_pnl = 0.0
        month_trades = []

        for ticker, info in portfolio.items():
            if ticker not in monthly_prices.columns:
                continue

            entry_price = monthly_prices.iloc[i - 1].get(ticker)
            exit_price = monthly_prices.iloc[i].get(ticker)

            if pd.isna(entry_price) or pd.isna(exit_price) or entry_price <= 0:
                continue

            weight = info["weight"]
            allocated_capital = capital * weight
            shares = int(allocated_capital / entry_price)
            if shares <= 0:
                continue

            actual_invested = shares * entry_price
            stock_return = (exit_price - entry_price) / entry_price
            pnl = shares * (exit_price - entry_price)

            # Intra-month high/low for MFE/MAE
            mask = (daily_prices.index > prev_date) & (daily_prices.index <= date)
            if ticker in daily_prices.columns:
                intra = daily_prices.loc[mask, ticker].dropna()
            else:
                intra = pd.Series(dtype=float)

            if len(intra) > 0:
                highest = intra.max()
                lowest = intra.min()
            else:
                highest = max(entry_price, exit_price)
                lowest = min(entry_price, exit_price)

            # MFE / MAE (per share, then as %)
            mfe_pct = (highest - entry_price) / entry_price  # max favorable
            mae_pct = (lowest - entry_price) / entry_price   # max adverse (negative)
            mfe_dollar = shares * (highest - entry_price)
            mae_dollar = shares * (lowest - entry_price)

            # Efficiency (long only)
            price_range = highest - lowest
            if price_range > 0:
                entry_eff = (highest - entry_price) / price_range
                exit_eff = (exit_price - lowest) / price_range
                total_eff = (exit_price - entry_price) / price_range
            else:
                entry_eff = exit_eff = total_eff = 0.0

            # Transaction cost
            cost = actual_invested * (TRANSACTION_COST + SLIPPAGE_COST) * 2  # round trip
            net_pnl = pnl - cost

            trade = {
                "date": date,
                "ticker": ticker,
                "sector": info["sector"],
                "weight": weight,
                "momentum_score": info["momentum"],
                "shares": shares,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "highest": highest,
                "lowest": lowest,
                "return_pct": stock_return,
                "pnl_gross": pnl,
                "cost": cost,
                "pnl_net": net_pnl,
                "mfe_pct": mfe_pct,
                "mae_pct": mae_pct,
                "mfe_dollar": mfe_dollar,
                "mae_dollar": mae_dollar,
                "entry_efficiency": entry_eff,
                "exit_efficiency": exit_eff,
                "total_efficiency": total_eff,
            }
            trades.append(trade)
            month_trades.append(trade)
            month_gross_pnl += net_pnl

            month_predicted.append(info["momentum"])
            month_actual.append(stock_return)

        # IC for this month (Spearman rank correlation)
        if len(month_predicted) >= 3:
            ic, _ = stats.spearmanr(month_predicted, month_actual)
            if not np.isnan(ic):
                ic_values.append(ic)

        month_net_return = month_gross_pnl / capital if capital > 0 else 0
        monthly_records.append({
            "date": date,
            "num_stocks": len(month_trades),
            "net_pnl": month_gross_pnl,
            "net_return": month_net_return,
        })

    # ── Compute Aggregate Metrics ─────────────────────────────────────────
    print("[4/4] Computing metrics ...\n")

    trades_df = pd.DataFrame(trades)
    if len(trades_df) == 0:
        print("[ERROR] No trades generated.")
        return

    # Separate wins and losses
    wins = trades_df[trades_df["pnl_net"] > 0]
    losses = trades_df[trades_df["pnl_net"] <= 0]

    win_rate = len(wins) / len(trades_df)
    loss_rate = 1 - win_rate
    avg_win = wins["pnl_net"].mean() if len(wins) > 0 else 0
    avg_loss = abs(losses["pnl_net"].mean()) if len(losses) > 0 else 0
    expectancy = (win_rate * avg_win) - (loss_rate * avg_loss)

    avg_win_pct = wins["return_pct"].mean() * 100 if len(wins) > 0 else 0
    avg_loss_pct = losses["return_pct"].mean() * 100 if len(losses) > 0 else 0

    # IC / ICIR
    ic_array = np.array(ic_values)
    mean_ic = np.mean(ic_array) if len(ic_array) > 0 else 0
    std_ic = np.std(ic_array, ddof=1) if len(ic_array) > 1 else 0
    icir = mean_ic / std_ic if std_ic > 0 else 0

    # Efficiency averages
    avg_entry_eff = trades_df["entry_efficiency"].mean()
    avg_exit_eff = trades_df["exit_efficiency"].mean()
    avg_total_eff = trades_df["total_efficiency"].mean()

    # MFE/MAE averages
    avg_mfe = trades_df["mfe_pct"].mean() * 100
    avg_mae = trades_df["mae_pct"].mean() * 100

    # Profit factor
    total_wins = wins["pnl_net"].sum() if len(wins) > 0 else 0
    total_losses = abs(losses["pnl_net"].sum()) if len(losses) > 0 else 0
    profit_factor = total_wins / total_losses if total_losses > 0 else float("inf")

    # Monthly P&L
    monthly_df = pd.DataFrame(monthly_records)
    total_net_pnl = monthly_df["net_pnl"].sum()
    avg_monthly_pnl = monthly_df["net_pnl"].mean()

    # ══════════════════════════════════════════════════════════════════════
    # DISPLAY
    # ══════════════════════════════════════════════════════════════════════

    w = 80
    sep = "=" * w
    sep2 = "-" * w

    # ── Section 1: Core Strategy Metrics ──────────────────────────────────
    print(sep)
    print("  1. MATHEMATICAL EXPECTANCY & CORE METRICS")
    print(sep)
    print(f"  Fixed Capital per Period:     ${capital:,.0f}")
    print(f"  Total Trades (stock-months):  {len(trades_df)}")
    print(f"  Total Months Traded:          {len(monthly_df)}")
    print(f"  Avg Stocks per Month:         {trades_df.groupby('date').size().mean():.1f}")
    print(sep2)
    print(f"  Win Rate:                     {win_rate * 100:.1f}%")
    print(f"  Loss Rate:                    {loss_rate * 100:.1f}%")
    print(f"  Avg Win (per stock-trade):    ${avg_win:,.2f}  ({avg_win_pct:+.2f}%)")
    print(f"  Avg Loss (per stock-trade):   ${avg_loss:,.2f}  ({avg_loss_pct:+.2f}%)")
    print(sep2)
    print(f"  >>> EXPECTANCY per trade:     ${expectancy:,.2f}")
    verdict = "VIABLE" if expectancy > 0 else "NOT VIABLE"
    print(f"  >>> Verdict:                  {verdict}")
    print(sep2)
    print(f"  Profit Factor:                {profit_factor:.2f}")
    print(f"  Total Net P&L:                ${total_net_pnl:,.2f}")
    print(f"  Avg Monthly Net P&L:          ${avg_monthly_pnl:,.2f}")
    print(f"  Total Costs:                  ${trades_df['cost'].sum():,.2f}")

    # ── Section 2: IC / ICIR ──────────────────────────────────────────────
    print(f"\n{sep}")
    print("  2. INFORMATION COEFFICIENT (IC) & ICIR")
    print(sep)
    print(f"  IC observations (months):     {len(ic_array)}")
    print(f"  Mean IC:                      {mean_ic:.4f}")
    print(f"  Std IC:                        {std_ic:.4f}")
    print(f"  ICIR (IC / Std):              {icir:.4f}")
    print(sep2)

    ic_pass = abs(mean_ic) > 0.05
    icir_pass = icir > 0.5
    print(f"  |IC| > 0.05?                  {'YES' if ic_pass else 'NO'}  (|IC| = {abs(mean_ic):.4f})")
    print(f"  ICIR > 0.5?                   {'YES' if icir_pass else 'NO'}  (ICIR = {icir:.4f})")
    if ic_pass and icir_pass:
        print("  >>> Alpha factor: SIGNIFICANT & STABLE")
    elif ic_pass:
        print("  >>> Alpha factor: Significant but UNSTABLE")
    else:
        print("  >>> Alpha factor: WEAK predictive power")

    # ── Section 3: Trade Efficiency ───────────────────────────────────────
    print(f"\n{sep}")
    print("  3. ENTRY / EXIT / TOTAL TRADE EFFICIENCY")
    print(sep)
    print(f"  Avg Entry Efficiency:         {avg_entry_eff * 100:.1f}%")
    print(f"  Avg Exit Efficiency:          {avg_exit_eff * 100:.1f}%")
    print(f"  Avg Total Efficiency:         {avg_total_eff * 100:.1f}%")
    print(sep2)
    print("  (100% = perfect timing, 0% = worst timing)")
    print("  Entry: how close to the low you entered")
    print("  Exit:  how close to the high you exited")
    print("  Total: how much of the range you captured")

    # ── Section 4: MFE / MAE ──────────────────────────────────────────────
    print(f"\n{sep}")
    print("  4. MFE / MAE ANALYSIS")
    print(sep)
    print(f"  Avg MFE (max unrealized gain):  {avg_mfe:+.2f}%")
    print(f"  Avg MAE (max unrealized loss):  {avg_mae:+.2f}%")
    print(sep2)

    # MFE vs actual return analysis: how much profit did we leave on the table?
    trades_df["mfe_captured"] = np.where(
        trades_df["mfe_pct"] > 0,
        trades_df["return_pct"] / trades_df["mfe_pct"],
        0
    )
    avg_mfe_captured = trades_df.loc[trades_df["mfe_pct"] > 0, "mfe_captured"].mean()
    print(f"  Avg MFE Captured:               {avg_mfe_captured * 100:.1f}%  (of max possible gain)")

    # Winning trades MAE analysis
    if len(wins) > 0:
        avg_win_mae = wins["mae_pct"].mean() * 100
        print(f"  Avg MAE on winning trades:      {avg_win_mae:+.2f}%  (drawdown before profit)")

    # Losing trades MFE analysis
    if len(losses) > 0:
        losses_with_mfe = losses[losses["mfe_pct"] > 0.01]
        if len(losses_with_mfe) > 0:
            pct_lost_had_profit = len(losses_with_mfe) / len(losses) * 100
            avg_lost_mfe = losses_with_mfe["mfe_pct"].mean() * 100
            print(f"  Losing trades that were once profitable: {pct_lost_had_profit:.0f}%  (avg MFE: {avg_lost_mfe:+.2f}%)")

    # ── Section 5: Per-Trade Detail (last 20) ─────────────────────────────
    print(f"\n{sep}")
    print("  5. RECENT TRADES (Last 20 Months)")
    print(sep)

    recent_dates = sorted(trades_df["date"].unique())[-20:]
    recent = trades_df[trades_df["date"].isin(recent_dates)].sort_values(
        ["date", "sector"]
    )

    header = (
        f"  {'Date':<12} {'Ticker':<7} {'Sector':<22} "
        f"{'Entry':>8} {'Exit':>8} {'Ret%':>7} "
        f"{'P&L':>10} {'MFE%':>6} {'MAE%':>7} {'Eff%':>5}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))

    prev_date = None
    for _, t in recent.iterrows():
        d = t["date"].strftime("%Y-%m")
        if prev_date and d != prev_date:
            # Month subtotal
            month_trades_sub = recent[recent["date"] == prev_date_raw]
            month_pnl = month_trades_sub["pnl_net"].sum()
            print(f"  {'':>12} {'':>7} {'MONTH TOTAL':>22} "
                  f"{'':>8} {'':>8} {'':>7} "
                  f"{'${:,.0f}'.format(month_pnl):>10}")
            print()

        print(
            f"  {d:<12} {t['ticker']:<7} {t['sector']:<22} "
            f"${t['entry_price']:>7.2f} ${t['exit_price']:>7.2f} "
            f"{t['return_pct'] * 100:>+6.1f}% "
            f"{'${:,.0f}'.format(t['pnl_net']):>10} "
            f"{t['mfe_pct'] * 100:>+5.1f} {t['mae_pct'] * 100:>+6.1f} "
            f"{t['total_efficiency'] * 100:>4.0f}%"
        )
        prev_date = d
        prev_date_raw = t["date"]

    # Final month subtotal
    if prev_date:
        month_trades_sub = recent[recent["date"] == prev_date_raw]
        month_pnl = month_trades_sub["pnl_net"].sum()
        print(f"  {'':>12} {'':>7} {'MONTH TOTAL':>22} "
              f"{'':>8} {'':>8} {'':>7} "
              f"{'${:,.0f}'.format(month_pnl):>10}")

    # ── Section 6: Monthly P&L Summary ────────────────────────────────────
    print(f"\n{sep}")
    print("  6. MONTHLY P&L SUMMARY")
    print(sep)

    monthly_pnls = monthly_df["net_pnl"].values
    pos_months = monthly_pnls[monthly_pnls > 0]
    neg_months = monthly_pnls[monthly_pnls <= 0]

    print(f"  Profitable Months:    {len(pos_months)} / {len(monthly_pnls)}  ({len(pos_months)/len(monthly_pnls)*100:.0f}%)")
    print(f"  Avg Profitable Month: ${pos_months.mean():,.0f}" if len(pos_months) > 0 else "")
    print(f"  Avg Losing Month:     ${neg_months.mean():,.0f}" if len(neg_months) > 0 else "")
    print(f"  Best Month:           ${monthly_pnls.max():,.0f}")
    print(f"  Worst Month:          ${monthly_pnls.min():,.0f}")
    print(f"  Median Month:         ${np.median(monthly_pnls):,.0f}")
    print(f"  Std Dev Monthly P&L:  ${np.std(monthly_pnls):,.0f}")

    # ── Section 7: Sector Breakdown ───────────────────────────────────────
    print(f"\n{sep}")
    print("  7. SECTOR PERFORMANCE BREAKDOWN")
    print(sep)

    sector_header = f"  {'Sector':<25} {'Trades':>6} {'WinR%':>6} {'AvgRet%':>8} {'TotalP&L':>12} {'AvgMFE%':>8} {'AvgMAE%':>8}"
    print(sector_header)
    print("  " + "-" * (len(sector_header) - 2))

    for sector, grp in trades_df.groupby("sector"):
        s_wins = grp[grp["pnl_net"] > 0]
        s_wr = len(s_wins) / len(grp) * 100
        s_avg_ret = grp["return_pct"].mean() * 100
        s_total_pnl = grp["pnl_net"].sum()
        s_mfe = grp["mfe_pct"].mean() * 100
        s_mae = grp["mae_pct"].mean() * 100
        print(
            f"  {sector:<25} {len(grp):>6} {s_wr:>5.0f}% {s_avg_ret:>+7.2f}% "
            f"{'${:,.0f}'.format(s_total_pnl):>12} {s_mfe:>+7.2f}% {s_mae:>+7.2f}%"
        )

    # ── Section 8: Benchmark Comparison (Monthly Return % — No Compounding) ──
    print(f"\n{sep}")
    print("  8. BENCHMARK COMPARISON (MONTHLY RETURN RATE %)")
    print(sep)
    print("  Pure return-rate comparison: no compounding, no dollar amounts.")
    print("  Strategy monthly return = portfolio net P&L / $30K.")
    print("  Benchmark monthly return = (close - prev_close) / prev_close.")
    print("  Alpha = Strategy% - Benchmark%  per month.")
    print(sep2)

    actual_years = len(monthly_df) / 12

    # ── Download benchmark prices ──
    print("  Loading benchmark data (SPY, SPMO, SPX) ...")
    benchmark_prices = dm.download_benchmark(start_date)
    bench_monthly_prices = benchmark_prices.resample("ME").last()

    # Strategy monthly return rates (already computed, no compounding)
    strat_monthly_rets = monthly_df["net_return"].values  # e.g. 0.025 = +2.5%
    strat_dates = list(monthly_df["date"])

    # ── Compute benchmark monthly return rates ──
    bench_rets = {}  # name -> np.array of monthly return rates
    for col in ["SPY", "SPMO", "SPX"]:
        if col not in bench_monthly_prices.columns:
            continue
        bm = bench_monthly_prices[col].dropna()
        bm_r = []
        for idx in range(len(strat_dates)):
            d = strat_dates[idx]
            mask = bm.index <= d + pd.Timedelta(days=5)
            if mask.sum() < 2:
                bm_r.append(0.0)
                continue
            bm_filtered = bm[mask]
            cur_price = bm_filtered.iloc[-1]
            prev_price = bm_filtered.iloc[-2]
            if prev_price > 0:
                bm_r.append((cur_price - prev_price) / prev_price)
            else:
                bm_r.append(0.0)
        bench_rets[col] = np.array(bm_r)

    # ── Alpha per month ──
    alpha_vs = {}
    for name, br in bench_rets.items():
        alpha_vs[name] = strat_monthly_rets - br

    # ── Metrics on monthly return rates ──
    def _sharpe_r(rets, rf_annual=0.02):
        rf_m = rf_annual / 12
        excess = rets - rf_m
        s = np.std(excess, ddof=1)
        return (np.mean(excess) / s) * np.sqrt(12) if s > 0 else 0.0

    def _sortino_r(rets, rf_annual=0.02):
        rf_m = rf_annual / 12
        excess = rets - rf_m
        down = excess[excess < 0]
        dv = np.std(down, ddof=1) if len(down) > 1 else 0
        return (np.mean(excess) / dv) * np.sqrt(12) if dv > 0 else 0.0

    def _mdd_from_returns(rets):
        """MDD from a return series (non-compounding cumulative)."""
        cum = np.cumsum(rets)
        peak = np.maximum.accumulate(cum)
        # Use cumulative return as proxy for wealth growth
        # MDD = max peak-to-trough in cumulative return space
        dd = cum - peak  # this is in return units
        return np.min(dd)  # most negative value

    def _mdd_pct_from_returns(rets):
        """MDD as % of wealth, assuming wealth = 1 + cumsum(rets)."""
        cum = np.cumsum(rets)
        wealth = 1.0 + cum  # start at $1 notional
        peak = np.maximum.accumulate(wealth)
        dd = (wealth - peak) / peak
        return np.min(dd)

    def _mdd_duration_from_returns(rets):
        cum = np.cumsum(rets)
        wealth = 1.0 + cum
        peak = np.maximum.accumulate(wealth)
        dd = (wealth - peak) / peak
        worst_idx = np.argmin(dd)
        peak_val = peak[worst_idx]
        candidates = np.where(wealth[:worst_idx + 1] == peak_val)[0]
        peak_idx = candidates[0] if len(candidates) > 0 else 0
        recovery_idx = len(wealth) - 1
        for j in range(worst_idx, len(wealth)):
            if wealth[j] >= peak_val:
                recovery_idx = j
                break
        return recovery_idx - peak_idx

    def _build_ret_metrics(name, rets):
        pos = rets[rets > 0]
        neg = rets[rets <= 0]
        return {
            "name": name,
            "mean_monthly": np.mean(rets) * 100,
            "median_monthly": np.median(rets) * 100,
            "annualized": np.mean(rets) * 12 * 100,
            "volatility": np.std(rets, ddof=1) * np.sqrt(12) * 100,
            "sharpe": _sharpe_r(rets),
            "sortino": _sortino_r(rets),
            "mdd": _mdd_pct_from_returns(rets) * 100,
            "mdd_duration": _mdd_duration_from_returns(rets),
            "win_rate": (np.sum(rets > 0) / len(rets)) * 100,
            "best_month": np.max(rets) * 100,
            "worst_month": np.min(rets) * 100,
            "avg_win": np.mean(pos) * 100 if len(pos) > 0 else 0,
            "avg_loss": np.mean(neg) * 100 if len(neg) > 0 else 0,
            "cumulative": np.sum(rets) * 100,
            "rets": rets,
        }

    strat_rm = _build_ret_metrics("Strategy", strat_monthly_rets)
    bench_rm = {}
    for name, br in bench_rets.items():
        bench_rm[name] = _build_ret_metrics(name, br)

    # ── Display comparison table ──
    col_w = 16
    label_w = 28
    table_sep = "  " + "+" + "-" * (label_w + 2) + ("+" + "-" * (col_w + 2)) * 4 + "+"

    def _trow(label, vals):
        cells = "".join(f"| {v:>{col_w}} " for v in vals)
        return f"  | {label:<{label_w}} {cells}|"

    col_names = ["Strategy", "SPY", "SPMO", "SPX"]

    print(table_sep)
    print(_trow("Metric", col_names))
    print(table_sep.replace("-", "="))

    def _bv(name, key, fmt=".2f"):
        return f"{bench_rm.get(name, {}).get(key, 0):{fmt}}"

    rows_table = [
        ("Cumulative Return", f"{strat_rm['cumulative']:.2f}%",
         lambda n: f"{_bv(n, 'cumulative')}%"),
        ("Mean Monthly Return", f"{strat_rm['mean_monthly']:.2f}%",
         lambda n: f"{_bv(n, 'mean_monthly')}%"),
        ("Median Monthly Return", f"{strat_rm['median_monthly']:.2f}%",
         lambda n: f"{_bv(n, 'median_monthly')}%"),
        ("Annualized Return", f"{strat_rm['annualized']:.2f}%",
         lambda n: f"{_bv(n, 'annualized')}%"),
        ("Ann. Volatility", f"{strat_rm['volatility']:.2f}%",
         lambda n: f"{_bv(n, 'volatility')}%"),
        ("Sharpe Ratio", f"{strat_rm['sharpe']:.3f}",
         lambda n: f"{_bv(n, 'sharpe', '.3f')}"),
        ("Sortino Ratio", f"{strat_rm['sortino']:.3f}",
         lambda n: f"{_bv(n, 'sortino', '.3f')}"),
        ("Max Drawdown", f"{strat_rm['mdd']:.2f}%",
         lambda n: f"{_bv(n, 'mdd')}%"),
        ("MDD Duration (months)", f"{strat_rm['mdd_duration']}",
         lambda n: f"{bench_rm.get(n, {}).get('mdd_duration', 0)}"),
        ("Win Rate (monthly)", f"{strat_rm['win_rate']:.1f}%",
         lambda n: f"{_bv(n, 'win_rate', '.1f')}%"),
        ("Best Month", f"{strat_rm['best_month']:.2f}%",
         lambda n: f"{_bv(n, 'best_month')}%"),
        ("Worst Month", f"{strat_rm['worst_month']:.2f}%",
         lambda n: f"{_bv(n, 'worst_month')}%"),
        ("Avg Winning Month", f"{strat_rm['avg_win']:.2f}%",
         lambda n: f"{_bv(n, 'avg_win')}%"),
        ("Avg Losing Month", f"{strat_rm['avg_loss']:.2f}%",
         lambda n: f"{_bv(n, 'avg_loss')}%"),
    ]

    for label, strat_val, bench_fn in rows_table:
        print(_trow(label, [strat_val, bench_fn("SPY"), bench_fn("SPMO"), bench_fn("SPX")]))
    print(table_sep)

    # ── Alpha table ──
    print()
    print(table_sep)
    print(_trow("ALPHA (Strategy - BM)", ["", "vs SPY", "vs SPMO", "vs SPX"]))
    print(table_sep.replace("-", "="))

    alpha_mean_row = [""]
    alpha_cum_row = [""]
    alpha_wr_row = [""]  # % of months strategy beats benchmark
    for name in ["SPY", "SPMO", "SPX"]:
        if name in alpha_vs:
            a = alpha_vs[name]
            alpha_mean_row.append(f"{np.mean(a) * 100:+.2f}%/mo")
            alpha_cum_row.append(f"{np.sum(a) * 100:+.2f}%")
            alpha_wr_row.append(f"{np.sum(a > 0) / len(a) * 100:.1f}%")
        else:
            alpha_mean_row.append("N/A")
            alpha_cum_row.append("N/A")
            alpha_wr_row.append("N/A")

    print(_trow("Mean Monthly Alpha", alpha_mean_row))
    print(_trow("Cumulative Alpha", alpha_cum_row))
    print(_trow("Months Beating BM", alpha_wr_row))
    print(table_sep)

    # ── Win/Lose verdict ──
    print()
    for name in ["SPY", "SPMO", "SPX"]:
        if name not in bench_rm:
            continue
        diff_ann = strat_rm["annualized"] - bench_rm[name]["annualized"]
        if diff_ann > 1.0:  # >1% annualized advantage
            verdict_bm = "WIN"
        elif diff_ann > -1.0:
            verdict_bm = "DRAW"
        else:
            verdict_bm = "LOSE"
        beat_pct = np.sum(alpha_vs[name] > 0) / len(alpha_vs[name]) * 100
        print(f"  Strategy vs {name:>4}: {verdict_bm}  "
              f"(alpha: {diff_ann:+.2f}%/yr, beats {beat_pct:.0f}% of months)")

    # ── Generate Charts ──────────────────────────────────────────────────
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    from pathlib import Path

    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle(
        "S&P 500 Sector Momentum — Monthly Return Rate Comparison (No Compounding)\n"
        f"{len(monthly_df)} months | Pure % comparison | Alpha = Strategy% - Benchmark%",
        fontsize=15, fontweight="bold", y=0.98,
    )

    colors = {"Strategy": "#2E86DE", "SPY": "#E74C3C", "SPMO": "#F39C12", "SPX": "#8E44AD"}
    dates_plot = strat_dates

    # ── Chart 1: Cumulative Return % ──
    ax = axes[0, 0]
    strat_cum_ret = np.cumsum(strat_monthly_rets) * 100
    ax.plot(dates_plot, strat_cum_ret, label="Strategy",
            color=colors["Strategy"], linewidth=2.5)
    for name in ["SPY", "SPMO", "SPX"]:
        if name in bench_rets:
            bm_cum = np.cumsum(bench_rets[name]) * 100
            ax.plot(dates_plot, bm_cum, label=name,
                    color=colors[name], linewidth=1.5, linestyle="--")
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.set_title("Cumulative Return % (Sum of Monthly Returns)", fontweight="bold")
    ax.set_ylabel("Cumulative Return (%)")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    # ── Chart 2: Drawdown % ──
    ax = axes[0, 1]
    def _dd_from_rets(rets):
        cum = np.cumsum(rets)
        wealth = 1.0 + cum
        peak = np.maximum.accumulate(wealth)
        return (wealth - peak) / peak * 100

    ax.fill_between(dates_plot, _dd_from_rets(strat_monthly_rets), 0,
                     alpha=0.3, color=colors["Strategy"], label="Strategy")
    for name in ["SPY", "SPMO", "SPX"]:
        if name in bench_rets:
            ax.plot(dates_plot, _dd_from_rets(bench_rets[name]),
                    color=colors[name], linewidth=1, linestyle="--", label=name)
    ax.set_title("Drawdown (%)", fontweight="bold")
    ax.set_ylabel("Drawdown %")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    # ── Chart 3: Monthly Alpha vs SPY ──
    ax = axes[1, 0]
    if "SPY" in alpha_vs:
        alpha_monthly = alpha_vs["SPY"] * 100
        bar_colors = ["#27AE60" if v > 0 else "#E74C3C" for v in alpha_monthly]
        ax.bar(dates_plot, alpha_monthly, color=bar_colors, alpha=0.7, width=25)
        ax.axhline(0, color="black", linewidth=1)

        # Add moving average
        ma = pd.Series(alpha_monthly).rolling(12).mean()
        ax.plot(dates_plot, ma, color="#2C3E50", linewidth=2, label="12M MA")

        beat_pct = np.sum(alpha_monthly > 0) / len(alpha_monthly) * 100
        ax.set_title(f"Monthly Alpha vs SPY (Strategy wins {beat_pct:.0f}% of months)",
                     fontweight="bold")
        ax.set_ylabel("Alpha (%)")
        ax.legend(fontsize=10)
    ax.grid(alpha=0.3, axis="y")

    # ── Chart 4: Rolling 12-Month Cumulative Alpha ──
    ax = axes[1, 1]
    if "SPY" in alpha_vs:
        rolling_window = 12
        rolling_alpha = pd.Series(alpha_vs["SPY"] * 100).rolling(rolling_window).sum()
        valid_idx = ~rolling_alpha.isna()
        valid_dates = [dates_plot[i] for i in range(len(dates_plot)) if valid_idx.iloc[i]]
        valid_alpha = rolling_alpha[valid_idx].values

        bar_colors = ["#27AE60" if v > 0 else "#E74C3C" for v in valid_alpha]
        ax.bar(valid_dates, valid_alpha, color=bar_colors, alpha=0.7, width=25)
        ax.axhline(0, color="black", linewidth=1)
        ax.set_title("Rolling 12-Month Alpha vs SPY", fontweight="bold")
        ax.set_ylabel("12M Cumulative Alpha (%)")

        win_12m = np.sum(valid_alpha > 0)
        total_12m = len(valid_alpha)
        ax.text(0.02, 0.95,
                f"Positive 12M alpha: {win_12m}/{total_12m} ({win_12m/total_12m*100:.0f}%)",
                transform=ax.transAxes, fontsize=9, verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    ax.grid(alpha=0.3, axis="y")

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    chart_path = reports_dir / "trade_analysis_benchmark.png"
    fig.savefig(str(chart_path), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\n  Chart saved: {chart_path}")

    # ── Final Verdict ─────────────────────────────────────────────────────
    print(f"\n{sep}")
    print("  FINAL ASSESSMENT")
    print(sep)

    strat_sharpe = strat_rm["sharpe"]
    strat_mdd = strat_rm["mdd"]

    checks = {
        "Expectancy > 0": expectancy > 0,
        "|IC| > 0.05": abs(mean_ic) > 0.05,
        "ICIR > 0.5": icir > 0.5,
        "Profit Factor > 1.5": profit_factor > 1.5,
        "Win Rate > 45%": win_rate > 0.45,
        "Total Efficiency > 30%": avg_total_eff > 0.3,
        "Sharpe > 0.5": strat_sharpe > 0.5,
        "MDD > -50%": strat_mdd > -50.0,
        "Beat SPY (ann. ret)": strat_rm["annualized"] > bench_rm.get("SPY", {}).get("annualized", 0),
        "Beat SPMO (ann. ret)": strat_rm["annualized"] > bench_rm.get("SPMO", {}).get("annualized", 0),
    }

    passed = sum(checks.values())
    total = len(checks)

    for name, ok in checks.items():
        mark = "PASS" if ok else "FAIL"
        print(f"  [{mark}] {name}")

    print(sep2)
    if passed == total:
        print(f"  RESULT: {passed}/{total} checks passed — STRATEGY IS STRONG")
    elif passed >= 7:
        print(f"  RESULT: {passed}/{total} checks passed — STRATEGY HAS POTENTIAL")
    elif passed >= 4:
        print(f"  RESULT: {passed}/{total} checks passed — STRATEGY NEEDS IMPROVEMENT")
    else:
        print(f"  RESULT: {passed}/{total} checks passed — STRATEGY IS WEAK")

    print(sep)

    return {
        "trades_df": trades_df,
        "monthly_df": monthly_df,
        "expectancy": expectancy,
        "ic_mean": mean_ic,
        "icir": icir,
        "profit_factor": profit_factor,
        "strat_metrics": strat_rm,
        "bench_metrics": bench_rm,
        "actual_years": actual_years,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trade-Level Strategy Analysis")
    parser.add_argument("--years", type=int, default=10, help="Backtest period (default: 10)")
    args = parser.parse_args()

    run_trade_analysis(years_back=args.years)
