"""
Performance Analyzer - Calculate metrics and generate reports
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from config import START_CAPITAL


class PerformanceAnalyzer:
    """Analyze backtest results and generate visualizations"""

    def __init__(self, results):
        """
        Args:
            results: dict from Backtester.run()
        """
        self.results = results
        self.start_capital = START_CAPITAL

    def calculate_metrics(self):
        """Calculate all performance metrics"""
        # Skip first 7 months (warm-up period)
        pv = np.array(self.results["portfolio_values"][7:])

        # Get benchmark values (support both old single benchmark and new multiple benchmarks)
        if isinstance(self.results["benchmark_values"], dict):
            bv_spy = np.array(self.results["benchmark_values"]["SPY"][7:])
            bv_spmo = np.array(
                self.results["benchmark_values"].get(
                    "SPMO", [self.start_capital] * len(pv)
                )[7:]
            )
            bv_spx = np.array(
                self.results["benchmark_values"].get(
                    "SPX", [self.start_capital] * len(pv)
                )[7:]
            )
        else:
            # Old format compatibility
            bv_spy = np.array(self.results["benchmark_values"][7:])
            bv_spmo = bv_spy
            bv_spx = bv_spy

        # Guard: short or empty backtests should not crash metric calculation.
        if len(pv) == 0:
            fallback_date = (
                self.results["dates"][0]
                if len(self.results.get("dates", [])) > 0
                else pd.Timestamp.now()
            )
            final_portfolio = self.results.get("final_portfolio", {})
            return {
                "total_return": 0.0,
                "spy_return": 0.0,
                "spmo_return": 0.0,
                "spx_return": 0.0,
                "annualized_return": 0.0,
                "spy_annualized": 0.0,
                "spmo_annualized": 0.0,
                "spx_annualized": 0.0,
                "benchmark_return": 0.0,
                "benchmark_annualized": 0.0,
                "volatility": 0.0,
                "max_drawdown": 0.0,
                "max_drawdown_date": fallback_date,
                "sharpe_ratio": 0.0,
                "sortino_ratio": 0.0,
                "calmar_ratio": 0.0,
                "win_rate": 0.0,
                "total_trades": len(self.results.get("trades", [])),
                "avg_monthly_turnover": 0.0,
                "total_costs": 0.0,
                "final_num_stocks": len(final_portfolio),
                "final_num_sectors": len(
                    set(
                        p.get("sector")
                        for p in final_portfolio.values()
                        if "sector" in p
                    )
                ),
            }

        # Get returns safely
        if (
            "returns_df" in self.results
            and len(self.results["returns_df"]) > 0
            and "net_return" in self.results["returns_df"].columns
        ):
            returns = self.results["returns_df"]["net_return"].values
        else:
            # Calculate returns from portfolio values if returns_df is missing
            returns = np.diff(pv) / pv[:-1] if len(pv) > 1 else np.array([0])

        # Calculate actual years from investment period (more accurate)
        # Use actual months of returns, not calendar time
        if "returns_df" in self.results and len(self.results["returns_df"]) > 0:
            actual_months = len(self.results["returns_df"])
            years = actual_months / 12
        elif len(self.results["dates"]) > 6:
            # Fallback: use dates but account for warm-up period
            start_date = self.results["dates"][6]  # First trading date after warm-up
            end_date = self.results["dates"][-1]  # Last trading date
            years = (end_date - start_date).days / 365.25
        else:
            years = len(pv) / 12 if len(pv) > 0 else 1

        metrics = {
            # Returns
            "total_return": (pv[-1] - self.start_capital) / self.start_capital,
            "spy_return": (bv_spy[-1] - self.start_capital) / self.start_capital,
            "spmo_return": (bv_spmo[-1] - self.start_capital) / self.start_capital,
            "spx_return": (bv_spx[-1] - self.start_capital) / self.start_capital,
            "annualized_return": (pv[-1] / self.start_capital) ** (1 / years) - 1,
            "spy_annualized": (bv_spy[-1] / self.start_capital) ** (1 / years) - 1,
            "spmo_annualized": (bv_spmo[-1] / self.start_capital) ** (1 / years) - 1,
            "spx_annualized": (bv_spx[-1] / self.start_capital) ** (1 / years) - 1,
            # Legacy names for compatibility
            "benchmark_return": (bv_spy[-1] - self.start_capital) / self.start_capital,
            "benchmark_annualized": (bv_spy[-1] / self.start_capital) ** (1 / years)
            - 1,
            # Risk
            "volatility": np.std(returns) * np.sqrt(12) if len(returns) > 0 else 0,
            "max_drawdown": self._max_drawdown(pv),
            "max_drawdown_date": self._max_drawdown_date(pv),
            # Risk-adjusted
            "sharpe_ratio": self._sharpe_ratio(returns) if len(returns) > 0 else 0,
            "sortino_ratio": self._sortino_ratio(returns) if len(returns) > 0 else 0,
            "calmar_ratio": self._calmar_ratio(pv, years) if len(pv) > 0 else 0,
            # Trading
            "win_rate": len(returns[returns > 0]) / len(returns)
            if len(returns) > 0
            else 0,
            "total_trades": len(self.results["trades"]),
            "avg_monthly_turnover": len(self.results["trades"]) / len(pv)
            if len(pv) > 0
            else 0,
            "total_costs": self.results["returns_df"]["cost"].sum()
            if "returns_df" in self.results
            and len(self.results["returns_df"]) > 0
            and "cost" in self.results["returns_df"].columns
            else 0,
            # Holdings
            "final_num_stocks": len(self.results["final_portfolio"]),
            "final_num_sectors": len(
                set(p["sector"] for p in self.results["final_portfolio"].values())
            ),
        }

        return metrics

    def _max_drawdown(self, values):
        """Calculate maximum drawdown"""
        if len(values) == 0:
            return 0
        peak = np.maximum.accumulate(values)
        dd = (values - peak) / peak
        return np.min(dd)

    def _max_drawdown_date(self, values):
        """Find date of maximum drawdown"""
        if len(values) == 0:
            return (
                self.results["dates"][0]
                if len(self.results["dates"]) > 0
                else pd.Timestamp.now()
            )
        peak = np.maximum.accumulate(values)
        dd = (values - peak) / peak
        idx = np.argmin(dd)
        return self.results["dates"][min(idx + 7, len(self.results["dates"]) - 1)]

    def _sharpe_ratio(self, returns, risk_free=0.02):
        """Calculate Sharpe ratio"""
        if len(returns) == 0:
            return 0
        excess = np.mean(returns) * 12 - risk_free
        vol = np.std(returns) * np.sqrt(12)
        return excess / vol if vol > 0 else 0

    def _sortino_ratio(self, returns, risk_free=0.02):
        """Calculate Sortino ratio (downside deviation)"""
        if len(returns) == 0:
            return 0
        excess = np.mean(returns) * 12 - risk_free
        downside = returns[returns < 0]
        down_vol = np.std(downside) * np.sqrt(12) if len(downside) > 0 else 0
        return excess / down_vol if down_vol > 0 else 0

    def _calmar_ratio(self, values, years):
        """Calculate Calmar ratio"""
        if len(values) == 0 or years == 0:
            return 0
        annual_ret = (values[-1] / values[0]) ** (1 / years) - 1 if values[0] > 0 else 0
        max_dd = abs(self._max_drawdown(values))
        return annual_ret / max_dd if max_dd > 0 else 0

    def print_summary(self, metrics):
        """Print formatted performance summary"""
        print("\n" + "=" * 70)
        print("[INFO] PERFORMANCE SUMMARY")
        print("=" * 70)

        # Returns
        print("\n【Returns】")
        print(f"  Strategy Total:        {metrics['total_return'] * 100:>8.2f}%")
        print(f"  Benchmark Total:       {metrics['benchmark_return'] * 100:>8.2f}%")
        print(
            f"  Excess Return:         {(metrics['total_return'] - metrics['benchmark_return']) * 100:>8.2f}%"
        )
        print(f"  Strategy Annualized:   {metrics['annualized_return'] * 100:>8.2f}%")
        print(
            f"  Benchmark Annualized:  {metrics['benchmark_annualized'] * 100:>8.2f}%"
        )
        print(f"  SPMO Annualized:       {metrics['spmo_annualized'] * 100:>8.2f}%")
        print(f"  SPX Annualized:        {metrics['spx_annualized'] * 100:>8.2f}%")

        # Risk
        print("\n【Risk】")
        print(f"  Volatility:            {metrics['volatility'] * 100:>8.2f}%")
        print(f"  Max Drawdown:          {metrics['max_drawdown'] * 100:>8.2f}%")
        print(
            f"  Drawdown Date:         {metrics['max_drawdown_date'].strftime('%Y-%m')}"
        )

        # Risk-adjusted
        print("\n【Risk-Adjusted Returns】")
        print(f"  Sharpe Ratio:          {metrics['sharpe_ratio']:>8.3f}")
        print(f"  Sortino Ratio:         {metrics['sortino_ratio']:>8.3f}")
        print(f"  Calmar Ratio:          {metrics['calmar_ratio']:>8.3f}")

        # Trading
        print("\n【Trading】")
        print(f"  Win Rate:              {metrics['win_rate'] * 100:>8.1f}%")
        print(f"  Total Trades:          {metrics['total_trades']:>8.0f}")
        print(f"  Avg Monthly Turnover:  {metrics['avg_monthly_turnover']:>8.1f}")
        print(f"  Total Costs:           {metrics['total_costs'] * 100:>8.2f}%")

        print("=" * 70)

        return metrics

    def plot_performance(self, save_path=None):
        """
        Generate performance visualization
        Returns: matplotlib figure (for web display)
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle(
            "S&P 500 Sector Momentum Strategy - Performance Dashboard",
            fontsize=16,
            fontweight="bold",
        )

        dates = self.results["dates"][7:]
        pv = self.results["portfolio_values"][7:]

        # Handle both dict and list benchmark_values
        if isinstance(self.results["benchmark_values"], dict):
            bv_spy = self.results["benchmark_values"]["SPY"][7:]
            bv_spmo = self.results["benchmark_values"].get(
                "SPMO", [self.start_capital] * len(pv)
            )[7:]
            bv_spx = self.results["benchmark_values"].get(
                "SPX", [self.start_capital] * len(pv)
            )[7:]
        else:
            bv_spy = self.results["benchmark_values"][7:]
            bv_spmo = bv_spy
            bv_spx = bv_spy

        # 1. Cumulative Performance
        ax = axes[0, 0]
        ax.plot(dates, pv, label="Strategy", linewidth=2, color="#2E86DE")
        ax.plot(
            dates, bv_spy, label="SPY", linewidth=2, color="#EE5A6F", linestyle="--"
        )
        ax.plot(
            dates, bv_spmo, label="SPMO", linewidth=2, color="#F39C12", linestyle="-."
        )
        ax.plot(dates, bv_spx, label="SPX", linewidth=2, color="#8E44AD", linestyle=":")
        ax.fill_between(
            dates,
            pv,
            bv_spy,
            where=np.array(pv) >= np.array(bv_spy),
            alpha=0.15,
            color="green",
        )
        ax.fill_between(
            dates,
            pv,
            bv_spy,
            where=np.array(pv) < np.array(bv_spy),
            alpha=0.15,
            color="red",
        )
        ax.set_title("Cumulative Performance", fontweight="bold")
        ax.set_ylabel("Portfolio Value ($)")
        ax.legend()
        ax.grid(alpha=0.3)
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, p: f"${x / 1000:.0f}K")
        )

        # 2. Drawdown
        ax = axes[0, 1]
        peak = np.maximum.accumulate(pv)
        dd = (np.array(pv) - peak) / peak * 100
        ax.fill_between(dates, dd, 0, color="#FF6348", alpha=0.5)
        ax.plot(dates, dd, color="#C23616", linewidth=2)
        ax.set_title("Drawdown", fontweight="bold")
        ax.set_ylabel("Drawdown (%)")
        ax.grid(alpha=0.3)

        # 3. Monthly Returns
        ax = axes[1, 0]
        returns = self.results["returns_df"]["net_return"].values * 100
        colors = ["#27AE60" if r > 0 else "#E74C3C" for r in returns]
        ax.bar(range(len(returns)), returns, color=colors, alpha=0.7)
        ax.axhline(0, color="black", linewidth=1)
        ax.set_title("Monthly Returns", fontweight="bold")
        ax.set_ylabel("Return (%)")
        ax.set_xlabel("Month")
        ax.grid(alpha=0.3, axis="y")

        # 4. Sector Allocation
        ax = axes[1, 1]
        sectors = pd.Series(
            [p["sector"] for p in self.results["final_portfolio"].values()]
        )
        sector_dist = sectors.value_counts()
        colors_pie = plt.cm.Set3(np.linspace(0, 1, len(sector_dist)))

        wedges, texts, autotexts = ax.pie(
            sector_dist.values,
            labels=None,
            autopct="%1.1f%%",
            colors=colors_pie,
            startangle=90,
            textprops={"fontsize": 9},
        )
        ax.legend(
            wedges,
            sector_dist.index,
            title="Sectors",
            loc="center left",
            bbox_to_anchor=(1, 0, 0.5, 1),
            fontsize=8,
        )
        ax.set_title("Current Sector Allocation", fontweight="bold")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"[INFO] Chart saved to {save_path}")

        return fig

    def get_portfolio_dataframe(self):
        """Get current portfolio as DataFrame (for web display)"""
        import math

        portfolio_data = []

        for ticker, info in self.results["final_portfolio"].items():
            # Handle NaN values
            weight = info.get("weight", 0)
            momentum = info.get("momentum", 0)

            weight_str = (
                f"{weight * 100:.2f}%"
                if not (math.isnan(weight) or math.isinf(weight))
                else "0.00%"
            )
            momentum_str = (
                f"{momentum * 100:.2f}%"
                if not (math.isnan(momentum) or math.isinf(momentum))
                else "0.00%"
            )

            portfolio_data.append(
                {
                    "Ticker": ticker,
                    "Sector": info.get("sector", "Unknown"),
                    "Weight": weight_str,
                    "Momentum": momentum_str,
                }
            )

        df = (
            pd.DataFrame(portfolio_data).sort_values("Sector")
            if portfolio_data
            else pd.DataFrame()
        )
        return df

    def export_results(self, output_dir="/mnt/user-data/outputs"):
        """
        Export all results to CSV files (for web integration)
        """
        import os

        os.makedirs(output_dir, exist_ok=True)

        # 1. Performance time series
        perf_data = {
            "Date": self.results["dates"][7:],
            "Portfolio_Value": self.results["portfolio_values"][7:],
        }
        if isinstance(self.results["benchmark_values"], dict):
            for bench_name, bench_vals in self.results["benchmark_values"].items():
                perf_data[f"{bench_name}_Value"] = bench_vals[7:]
        else:
            perf_data["Benchmark_Value"] = self.results["benchmark_values"][7:]
        perf_df = pd.DataFrame(perf_data)
        perf_df.to_csv(f"{output_dir}/performance.csv", index=False)

        # 2. Monthly returns
        self.results["returns_df"].to_csv(
            f"{output_dir}/monthly_returns.csv", index=False
        )

        # 3. Trades
        self.results["trades_df"].to_csv(f"{output_dir}/trades.csv", index=False)

        # 4. Current portfolio
        portfolio_df = self.get_portfolio_dataframe()
        portfolio_df.to_csv(f"{output_dir}/current_portfolio.csv", index=False)

        print(f"\n[OK] Results exported to {output_dir}/")
        return output_dir


if __name__ == "__main__":
    print("Performance Analyzer module - use after backtesting")
