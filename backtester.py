"""
Backtester - Execute strategy and track performance
"""

import pandas as pd
import numpy as np
from config import START_CAPITAL, TRANSACTION_COST, SLIPPAGE_COST


class Backtester:
    """Run backtest and record all transactions"""

    def __init__(
        self,
        strategy,
        benchmark_prices,
        stop_loss_manager=None,
        check_frequency="weekly",
        signal_lag_months=1,
        transaction_cost=None,
        slippage_cost=None,
    ):
        """
        Args:
            strategy: MomentumStrategy instance
            benchmark_prices: DataFrame of benchmark prices (columns: SPY, SPMO, SPX)
            stop_loss_manager: StopLossManager instance (optional)
            check_frequency: 'daily', 'weekly', or 'monthly'
            signal_lag_months: Rebalance signal lag in months (1 = T+1 monthly proxy)
            transaction_cost: Optional override of transaction cost
            slippage_cost: Optional override of slippage cost
        """
        self.strategy = strategy
        self.benchmark_prices = benchmark_prices
        self.start_capital = START_CAPITAL
        self.transaction_cost = (
            TRANSACTION_COST if transaction_cost is None else float(transaction_cost)
        )
        self.slippage_cost = (
            SLIPPAGE_COST if slippage_cost is None else float(slippage_cost)
        )
        self.stop_loss = stop_loss_manager
        self.check_frequency = check_frequency
        self.signal_lag_months = max(0, int(signal_lag_months))

        # Results storage
        self.results = {
            "dates": [],
            "portfolio_values": [],
            "benchmark_values": {},  # Dict not list
            "monthly_returns": [],
            "trades": [],
            "holdings": [],
            "stop_loss_events": [],
        }

    def run(self):
        """Execute full backtest"""
        print("\n" + "=" * 60)
        print("[INFO] Starting Backtest")
        print("=" * 60)

        # Calculate momentum
        momentum = self.strategy.calculate_momentum()

        # Resample to monthly
        monthly_prices = self.strategy.prices.resample("ME").last()
        monthly_momentum = momentum.resample("ME").last()
        benchmark_monthly = self.benchmark_prices.resample("ME").last()

        print(f"\n[INFO] Benchmark Data Check:")
        print(f"   Type: {type(benchmark_monthly)}")
        if isinstance(benchmark_monthly, pd.DataFrame):
            print(f"   Columns: {benchmark_monthly.columns.tolist()}")
            print(f"   Shape: {benchmark_monthly.shape}")
            print(f"   First SPY value: {benchmark_monthly['SPY'].iloc[0]:.2f}")
            print(
                f"   NaN count: SPY={benchmark_monthly['SPY'].isna().sum()}, SPMO={benchmark_monthly['SPMO'].isna().sum()}, SPX={benchmark_monthly['SPX'].isna().sum()}"
            )

        # Initialize
        current_portfolio = {}
        portfolio_value = self.start_capital

        # Initialize benchmark values (dict for multiple benchmarks)
        # Track cumulative returns starting from 1.0
        benchmark_cumulative_returns = {}
        if isinstance(benchmark_monthly, pd.DataFrame):
            for col in benchmark_monthly.columns:
                benchmark_cumulative_returns[col] = 1.0  # Start at 1.0 (100%)
        else:
            benchmark_cumulative_returns["SPY"] = 1.0

        # Store in results (will convert to dollar values)
        for bench_name in benchmark_cumulative_returns.keys():
            self.results["benchmark_values"][bench_name] = []

        # Monthly iteration
        for i, date in enumerate(monthly_momentum.index):
            # Update benchmarks (every month, including first 6)
            if i > 0:
                if isinstance(benchmark_monthly, pd.DataFrame):
                    for bench_name in benchmark_monthly.columns:
                        current_price = benchmark_monthly[bench_name].iloc[i]
                        prev_price = benchmark_monthly[bench_name].iloc[i - 1]

                        # Debug first update
                        if i == 1 and bench_name == "SPY":
                            print(
                                f"\n[DEBUG] First benchmark update (i=1, {bench_name}):"
                            )
                            print(f"   Current price: {current_price}")
                            print(f"   Previous price: {prev_price}")
                            print(f"   notna(current): {pd.notna(current_price)}")
                            print(f"   notna(prev): {pd.notna(prev_price)}")
                            print(
                                f"   prev > 0: {prev_price > 0 if pd.notna(prev_price) else 'NaN'}"
                            )

                        # Skip if data not available (NaN)
                        if (
                            pd.notna(current_price)
                            and pd.notna(prev_price)
                            and prev_price > 0
                        ):
                            bench_return = (current_price - prev_price) / prev_price
                            benchmark_cumulative_returns[bench_name] *= 1 + bench_return

                            # Debug first few updates
                            if i <= 3 and bench_name == "SPY":
                                print(
                                    f"   Month {i} {bench_name}: return={bench_return * 100:+.2f}%, cumulative={benchmark_cumulative_returns[bench_name]:.4f}"
                                )
                        else:
                            if i <= 10 and bench_name == "SPY":
                                print(
                                    f"   [WARN] Month {i} {bench_name}: Skipped (NaN or invalid)"
                                )
                else:
                    bench_return = (
                        benchmark_monthly.iloc[i] - benchmark_monthly.iloc[i - 1]
                    ) / benchmark_monthly.iloc[i - 1]
                    benchmark_cumulative_returns["SPY"] *= 1 + bench_return

            # Skip first 7 months for strategy (ensure 26-week momentum data)
            # 7 months ≈ 210 days > 182 days (26 weeks) needed for momentum
            if i < 7:
                self.results["portfolio_values"].append(portfolio_value)
                for bench_name in benchmark_cumulative_returns.keys():
                    benchmark_value = (
                        self.start_capital * benchmark_cumulative_returns[bench_name]
                    )
                    self.results["benchmark_values"][bench_name].append(benchmark_value)
                self.results["dates"].append(date)
                continue

            # === STOP LOSS CHECK (before monthly rebalance) ===
            if self.stop_loss and current_portfolio:
                # Get check dates based on frequency
                check_dates = self._get_check_dates(
                    monthly_momentum.index[i - 1] if i > 0 else date, date
                )

                for check_date in check_dates:
                    # Check individual stock stop losses
                    self.stop_loss.update_peaks(
                        current_portfolio, self.strategy.prices, check_date
                    )
                    to_sell = self.stop_loss.check_stock_stops(
                        current_portfolio, self.strategy.prices, check_date
                    )

                    if to_sell:
                        print(
                            f"\n[WARN] Stop Loss Triggered on {check_date.strftime('%Y-%m-%d')}:"
                        )

                        for item in to_sell:
                            print(
                                f"    {item['ticker']}: {item['reason']} ({item.get('loss', item.get('drawdown')) * 100:.1f}%)"
                            )

                            # Calculate realized loss for this stock
                            ticker = item["ticker"]
                            if ticker in current_portfolio:
                                stock_weight = current_portfolio[ticker]["weight"]
                                stock_value = portfolio_value * stock_weight

                                # Get current price
                                if ticker in self.strategy.prices.columns:
                                    # Find nearest trading day
                                    nearest_date = self.strategy.prices.index[
                                        self.strategy.prices.index <= check_date
                                    ][-1]
                                    current_price = self.strategy.prices.loc[
                                        nearest_date, ticker
                                    ]

                                    if (
                                        pd.notna(current_price)
                                        and ticker in self.stop_loss.purchase_prices
                                    ):
                                        purchase_price = self.stop_loss.purchase_prices[
                                            ticker
                                        ]
                                        if (
                                            pd.notna(purchase_price)
                                            and purchase_price > 0
                                        ):
                                            # Calculate actual loss
                                            realized_return = (
                                                current_price - purchase_price
                                            ) / purchase_price
                                            realized_pnl = stock_value * realized_return
                                            portfolio_value += realized_pnl

                                            print(
                                                f"        Realized P&L: ${realized_pnl:,.0f} ({realized_return * 100:.2f}%)"
                                            )

                                # Remove from portfolio
                                del current_portfolio[ticker]

                            # Record stop loss event
                            self.results["stop_loss_events"].append(
                                {
                                    "date": check_date,
                                    "ticker": item["ticker"],
                                    "reason": item["reason"],
                                    "loss_pct": item.get("loss", item.get("drawdown")),
                                }
                            )

                        # Find replacement stocks (simplified - will rebalance next month)
                        self.stop_loss.remove_sold_stocks(to_sell)

                    # Check portfolio-level stop loss
                    portfolio_action = self.stop_loss.check_portfolio_stop(
                        portfolio_value
                    )

                    if portfolio_action == "HALT":
                        print(
                            f"\n[ALERT] PORTFOLIO HALT: Exiting all positions on {check_date.strftime('%Y-%m-%d')}"
                        )
                        self.results["stop_loss_events"].append(
                            {
                                "date": check_date,
                                "ticker": "PORTFOLIO",
                                "reason": "PORTFOLIO_HALT",
                                "loss_pct": (
                                    portfolio_value - self.stop_loss.portfolio_peak
                                )
                                / self.stop_loss.portfolio_peak,
                            }
                        )
                        current_portfolio = {}
                        break

                    elif portfolio_action == "REDUCE":
                        print(
                            f"\n[WARN] PORTFOLIO REDUCE: Cutting 50% on {check_date.strftime('%Y-%m-%d')}"
                        )
                        self.results["stop_loss_events"].append(
                            {
                                "date": check_date,
                                "ticker": "PORTFOLIO",
                                "reason": "PORTFOLIO_REDUCE",
                                "loss_pct": (
                                    portfolio_value - self.stop_loss.portfolio_peak
                                )
                                / self.stop_loss.portfolio_peak,
                            }
                        )
                        # Reduce all positions by 50%
                        for ticker in current_portfolio:
                            current_portfolio[ticker]["weight"] *= 0.5

            # === MONTHLY REBALANCE ===
            # Select target using lagged signal (T+1 monthly proxy by default).
            signal_idx = i - self.signal_lag_months
            if signal_idx >= 7:
                signal_date = monthly_momentum.index[signal_idx]
                new_portfolio = self.strategy.select_portfolio(
                    signal_date, monthly_momentum
                )
            else:
                new_portfolio = {}

            # Update purchase prices for new positions
            if self.stop_loss:
                self.stop_loss.update_purchase_prices(
                    new_portfolio, self.strategy.prices, date
                )

            # Calculate returns if we had positions
            if current_portfolio:
                gross_return = self._calculate_returns(
                    current_portfolio, monthly_prices, i
                )
                turnover = self._calculate_turnover(current_portfolio, new_portfolio)
                cost = turnover * (self.transaction_cost + self.slippage_cost)
                net_return = gross_return - cost

                portfolio_value *= 1 + net_return

                self.results["monthly_returns"].append(
                    {
                        "date": date,
                        "gross_return": gross_return,
                        "cost": cost,
                        "net_return": net_return,
                        "portfolio_value": portfolio_value,
                    }
                )

            # Record trades
            self._record_trades(current_portfolio, new_portfolio, date)

            # Update holdings
            current_portfolio = new_portfolio
            self.results["holdings"].append(
                {
                    "date": date,
                    "num_stocks": len(current_portfolio),
                    "num_sectors": len(
                        set(p["sector"] for p in current_portfolio.values())
                    ),
                }
            )

            self.results["portfolio_values"].append(portfolio_value)
            for bench_name in benchmark_cumulative_returns.keys():
                benchmark_value = (
                    self.start_capital * benchmark_cumulative_returns[bench_name]
                )
                self.results["benchmark_values"][bench_name].append(benchmark_value)
            self.results["dates"].append(date)

            # Progress update
            if i % 12 == 0:
                spy_cumulative = benchmark_cumulative_returns.get("SPY", 1.0)
                spy_val = self.start_capital * spy_cumulative
                print(
                    f"  {date.strftime('%Y-%m')}: Portfolio ${portfolio_value:,.0f} | SPY ${spy_val:,.0f} ({(spy_cumulative - 1) * 100:+.1f}%)"
                )

        print("\n[OK] Backtest Complete")

        # Convert to DataFrames
        self.results["returns_df"] = pd.DataFrame(self.results["monthly_returns"])
        self.results["trades_df"] = pd.DataFrame(self.results["trades"])
        self.results["holdings_df"] = pd.DataFrame(self.results["holdings"])
        self.results["final_portfolio"] = current_portfolio

        return self.results

    def _get_check_dates(self, start_date, end_date):
        """Get dates to check stop loss based on frequency"""
        if self.check_frequency == "daily":
            # Check every trading day
            mask = (self.strategy.prices.index > start_date) & (
                self.strategy.prices.index <= end_date
            )
            return self.strategy.prices.index[mask]

        elif self.check_frequency == "weekly":
            # Check using actual weekly trading closes (avoid weekend timestamps)
            weekly_last_trading_days = (
                self.strategy.prices.groupby(
                    self.strategy.prices.index.to_period("W-FRI")
                )
                .tail(1)
                .index
            )
            mask = (weekly_last_trading_days > start_date) & (
                weekly_last_trading_days <= end_date
            )
            return weekly_last_trading_days[mask]

        else:  # monthly
            # Check only at month end
            return [end_date]

    def _calculate_returns(self, portfolio, prices, idx):
        """Calculate portfolio return for the month"""
        total_return = 0

        for ticker, info in portfolio.items():
            if ticker not in prices.columns:
                continue

            prev_price = prices.iloc[idx - 1][ticker]
            curr_price = prices.iloc[idx][ticker]

            if pd.notna(prev_price) and pd.notna(curr_price) and prev_price > 0:
                stock_return = (curr_price - prev_price) / prev_price
                total_return += stock_return * info["weight"]

        return total_return

    def _calculate_turnover(self, old_portfolio, new_portfolio):
        """
        Calculate turnover as sum of absolute weight changes / 2.
        This correctly captures both full sells/buys AND weight adjustments.
        A complete 100% portfolio replacement = turnover of 1.0.
        """
        all_tickers = set(old_portfolio.keys()) | set(new_portfolio.keys())
        total_change = 0.0
        for ticker in all_tickers:
            old_w = old_portfolio.get(ticker, {}).get("weight", 0.0)
            new_w = new_portfolio.get(ticker, {}).get("weight", 0.0)
            total_change += abs(new_w - old_w)
        return total_change / 2.0  # one-way turnover

    def _record_trades(self, old_portfolio, new_portfolio, date):
        """Record buy/sell transactions"""
        old_tickers = set(old_portfolio.keys())
        new_tickers = set(new_portfolio.keys())

        # Sells
        for ticker in old_tickers - new_tickers:
            self.results["trades"].append(
                {
                    "date": date,
                    "ticker": ticker,
                    "action": "SELL",
                    "sector": old_portfolio[ticker]["sector"],
                }
            )

        # Buys
        for ticker in new_tickers - old_tickers:
            self.results["trades"].append(
                {
                    "date": date,
                    "ticker": ticker,
                    "action": "BUY",
                    "sector": new_portfolio[ticker]["sector"],
                }
            )

    def get_summary(self):
        """Get summary statistics"""
        pv = np.array(self.results["portfolio_values"][7:])

        # Handle dict benchmark_values
        if isinstance(self.results["benchmark_values"], dict):
            bv = np.array(self.results["benchmark_values"]["SPY"][7:])
        else:
            bv = np.array(self.results["benchmark_values"][7:])

        return {
            "final_portfolio_value": pv[-1],
            "final_benchmark_value": bv[-1],
            "total_return": (pv[-1] - self.start_capital) / self.start_capital,
            "benchmark_return": (bv[-1] - self.start_capital) / self.start_capital,
            "excess_return": (pv[-1] - bv[-1]) / self.start_capital,
            "num_trades": len(self.results["trades"]),
        }


if __name__ == "__main__":
    print("Backtester module - use with strategy and data manager")
