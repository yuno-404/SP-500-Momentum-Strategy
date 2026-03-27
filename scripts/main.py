"""
Main Entry Point - Run complete backtest pipeline
"""

from datetime import datetime, timedelta
import warnings

warnings.filterwarnings("ignore")

from data_manager import DataManager
from strategy import MomentumStrategy
from backtester import Backtester
from analyzer import PerformanceAnalyzer
from config import TOP_N_PER_SECTOR, REQUIRE_LOCAL_SECTOR_AUM_FOR_BACKTEST


def run_backtest(years_back=10, show_plots=True, export_data=True):
    """
    Run complete backtest pipeline

    Args:
        years_back: Number of years to backtest
        show_plots: Whether to display plots
        export_data: Whether to export results to CSV

    Returns:
        dict: Complete results including metrics and portfolio
    """
    print("=" * 70)
    print("S&P 500 SECTOR MOMENTUM STRATEGY - BACKTEST SYSTEM")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Backtest Period: {years_back} years")
    print(f"  Stocks per Sector: {TOP_N_PER_SECTOR}")
    print(f"  Weighting: Dynamic Sector ETF Market Cap")
    print("\n" + "=" * 70)

    # Calculate start date
    start_date = (datetime.now() - timedelta(days=365 * years_back)).strftime(
        "%Y-%m-%d"
    )

    # Step 1: Load Data
    print("\n【Step 1/5】 Loading Data")
    dm = DataManager()

    ticker_sector = dm.get_sp500_components()
    sector_weights = dm.download_sector_etf_weights(
        start_date,
        require_local_csv=REQUIRE_LOCAL_SECTOR_AUM_FOR_BACKTEST,
    )
    stock_prices = dm.download_stock_prices(ticker_sector.keys(), start_date)
    benchmark = dm.download_benchmark(start_date)

    # Step 2: Initialize Strategy
    print("\n【Step 2/5】 Initializing Strategy")
    strategy = MomentumStrategy(
        prices=stock_prices,
        ticker_sector=ticker_sector,
        sector_weights_ts=sector_weights,
        top_n=TOP_N_PER_SECTOR,
    )
    print("[OK] Strategy ready")

    # Step 3: Run Backtest
    print("\n【Step 3/5】 Running Backtest")
    backtester = Backtester(strategy, benchmark)
    results = backtester.run()

    # Step 4: Analyze Performance
    print("\n【Step 4/5】 Analyzing Performance")
    analyzer = PerformanceAnalyzer(results)
    metrics = analyzer.calculate_metrics()
    analyzer.print_summary(metrics)

    # Step 5: Generate Outputs
    print("\n【Step 5/5】 Generating Outputs")

    if show_plots:
        print("\n[INFO] Generating charts...")
        fig = analyzer.plot_performance()
        import matplotlib.pyplot as plt

        plt.show()

    if export_data:
        export_path = analyzer.export_results()

    # Print final portfolio
    print("\n" + "=" * 70)
    print("[INFO] CURRENT PORTFOLIO")
    print("=" * 70)

    portfolio_df = analyzer.get_portfolio_dataframe()
    print(portfolio_df.to_string(index=False))

    # Sector summary
    print("\n" + "-" * 70)
    print("SECTOR ALLOCATION")
    print("-" * 70)

    sector_summary = portfolio_df.groupby("Sector").size()
    for sector, count in sector_summary.items():
        total_weight = sum(
            p["weight"]
            for t, p in results["final_portfolio"].items()
            if p["sector"] == sector
        )
        print(f"{sector:<35} {count} stocks ({total_weight * 100:.1f}%)")

    print("\n" + "=" * 70)
    print("[OK] BACKTEST COMPLETE")
    print("=" * 70)

    return {
        "results": results,
        "metrics": metrics,
        "portfolio_df": portfolio_df,
        "analyzer": analyzer,
    }


def main():
    """Interactive main function"""
    print("\n" + "=" * 70)
    print("🎯 BACKTEST PARAMETERS")
    print("=" * 70)

    # Get user input
    years = input("\nBacktest period (years, default=10): ").strip()
    years = int(years) if years else 10

    confirm = input(f"\nRun backtest for {years} years? (Y/n): ").strip().lower()
    if confirm == "n":
        print("[CANCELLED]")
        return

    # Run backtest
    output = run_backtest(years_back=years, show_plots=True, export_data=True)

    return output


if __name__ == "__main__":
    output = main()
