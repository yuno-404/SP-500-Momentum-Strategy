"""
Unit Tests for S&P 500 Sector Momentum Strategy
Run: python test_all.py
"""

import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Test results tracker
test_results = []


def test(name):
    """Decorator for test functions"""

    def decorator(func):
        def wrapper():
            try:
                func()
                test_results.append((name, "PASS", None))
                print(f"[OK] {name}")
            except Exception as e:
                test_results.append((name, "FAIL", str(e)))
                print(f"[FAIL] {name}: {e}")

        return wrapper

    return decorator


print("=" * 70)
print("UNIT TESTS - S&P 500 Sector Momentum Strategy")
print("=" * 70)


# ==========================================
# Test 1: Config Module
# ==========================================
print("\n【Test 1】 Config Module")


@test("1.1 Import config")
def test_config_import():
    from config import START_CAPITAL, SECTOR_ETFS, TOP_N_PER_SECTOR

    assert START_CAPITAL == 30000
    assert len(SECTOR_ETFS) == 11
    assert TOP_N_PER_SECTOR == 1


test_config_import()


@test("1.2 Sector ETF tickers valid")
def test_sector_etfs():
    from config import SECTOR_ETFS

    expected_tickers = [
        "XLC",
        "XLY",
        "XLP",
        "XLE",
        "XLF",
        "XLV",
        "XLI",
        "XLK",
        "XLB",
        "XLRE",
        "XLU",
    ]

    actual_tickers = list(SECTOR_ETFS.values())
    assert set(actual_tickers) == set(expected_tickers)


test_sector_etfs()


# ==========================================
# Test 2: Data Manager
# ==========================================
print("\n【Test 2】 Data Manager")


@test("2.1 Import DataManager")
def test_datamanager_import():
    from data_manager import DataManager

    dm = DataManager()
    assert dm is not None


test_datamanager_import()


@test("2.2 Get S&P 500 components")
def test_get_components():
    from data_manager import DataManager

    dm = DataManager()
    ticker_sector = dm.get_sp500_components()

    assert len(ticker_sector) > 400  # Should have ~500 stocks
    assert "AAPL" in ticker_sector or "MSFT" in ticker_sector
    assert isinstance(ticker_sector, dict)


test_get_components()


@test("2.3 Download stock prices (sample)")
def test_download_prices():
    from data_manager import DataManager

    dm = DataManager()

    # Test with just 5 stocks
    test_tickers = ["AAPL", "MSFT", "GOOGL", "JPM", "XOM"]
    start_date = "2023-01-01"

    prices = dm.download_stock_prices(test_tickers, start_date)

    assert isinstance(prices, pd.DataFrame)
    assert len(prices) > 200  # Should have ~250 trading days per year
    assert prices.shape[1] <= 5  # Should have up to 5 stocks


test_download_prices()


@test("2.4 Download benchmark")
def test_download_benchmark():
    from data_manager import DataManager

    dm = DataManager()

    start_date = "2023-01-01"
    benchmark = dm.download_benchmark(start_date)

    assert isinstance(benchmark, pd.DataFrame)
    assert set(["SPY", "SPMO", "SPX"]).issubset(set(benchmark.columns))
    assert len(benchmark) > 200


test_download_benchmark()


# ==========================================
# Test 3: Strategy
# ==========================================
print("\n【Test 3】 Strategy Engine")


@test("3.1 Import MomentumStrategy")
def test_strategy_import():
    from strategy import MomentumStrategy

    assert MomentumStrategy is not None


test_strategy_import()


@test("3.2 Calculate momentum")
def test_calculate_momentum():
    from strategy import MomentumStrategy

    # Create dummy data
    dates = pd.date_range("2020-01-01", periods=200, freq="D")
    prices = pd.DataFrame(
        np.random.randn(200, 5).cumsum(axis=0) + 100,
        index=dates,
        columns=["A", "B", "C", "D", "E"],
    )

    ticker_sector = {t: "Tech" for t in prices.columns}
    sector_weights = pd.DataFrame({"Tech": [1.0]}, index=[dates[-1]])

    strategy = MomentumStrategy(prices, ticker_sector, sector_weights, top_n=2)
    momentum = strategy.calculate_momentum()

    assert isinstance(momentum, pd.DataFrame)
    assert momentum.shape == prices.shape


test_calculate_momentum()


@test("3.3 Select portfolio")
def test_select_portfolio():
    from strategy import MomentumStrategy

    # Create dummy data with multiple sectors
    dates = pd.date_range("2020-01-01", periods=200, freq="D")
    tickers = ["AAPL", "MSFT", "GOOGL", "JPM", "BAC", "XOM", "CVX"]

    prices = pd.DataFrame(
        np.random.randn(200, 7).cumsum(axis=0) + 100, index=dates, columns=tickers
    )

    ticker_sector = {
        "AAPL": "Tech",
        "MSFT": "Tech",
        "GOOGL": "Tech",
        "JPM": "Financials",
        "BAC": "Financials",
        "XOM": "Energy",
        "CVX": "Energy",
    }

    sector_weights = pd.DataFrame(
        {"Tech": [0.5], "Financials": [0.3], "Energy": [0.2]}, index=[dates[-1]]
    )

    strategy = MomentumStrategy(prices, ticker_sector, sector_weights, top_n=2)
    momentum = strategy.calculate_momentum()

    portfolio = strategy.select_portfolio(dates[-1], momentum)

    assert isinstance(portfolio, dict)
    assert len(portfolio) > 0

    # Check total weight = 1.0
    total_weight = sum(p["weight"] for p in portfolio.values())
    assert abs(total_weight - 1.0) < 0.001


test_select_portfolio()


# ==========================================
# Test 4: Backtester
# ==========================================
print("\n【Test 4】 Backtester")


@test("4.1 Import Backtester")
def test_backtester_import():
    from backtester import Backtester

    assert Backtester is not None


test_backtester_import()


@test("4.2 Run backtest (minimal)")
def test_run_backtest():
    from strategy import MomentumStrategy
    from backtester import Backtester

    # Create minimal test data
    dates = pd.date_range("2020-01-01", periods=250, freq="D")
    tickers = ["A", "B", "C", "D", "E", "F"]

    prices = pd.DataFrame(
        np.random.randn(250, 6).cumsum(axis=0) + 100, index=dates, columns=tickers
    )

    ticker_sector = {
        "A": "Tech",
        "B": "Tech",
        "C": "Finance",
        "D": "Finance",
        "E": "Energy",
        "F": "Energy",
    }

    sector_weights = pd.DataFrame(
        {"Tech": [0.4] * 12, "Finance": [0.4] * 12, "Energy": [0.2] * 12},
        index=pd.date_range("2020-01-31", periods=12, freq="ME"),
    )

    benchmark = pd.Series(np.random.randn(250).cumsum() + 100, index=dates)

    strategy = MomentumStrategy(prices, ticker_sector, sector_weights, top_n=1)
    backtester = Backtester(strategy, benchmark)

    results = backtester.run()

    assert "dates" in results
    assert "portfolio_values" in results
    assert "returns_df" in results
    assert len(results["portfolio_values"]) > 0


test_run_backtest()


@test("4.3 Turnover calculation")
def test_turnover():
    from backtester import Backtester

    old_portfolio = {
        "A": {"weight": 1 / 3},
        "B": {"weight": 1 / 3},
        "C": {"weight": 1 / 3},
    }
    new_portfolio = {
        "A": {"weight": 1 / 3},
        "D": {"weight": 1 / 3},
        "E": {"weight": 1 / 3},
    }

    # Mock backtester
    class MockStrategy:
        pass

    bt = Backtester(MockStrategy(), pd.Series())
    turnover = bt._calculate_turnover(old_portfolio, new_portfolio)

    # One-way turnover should be 2/3 for replacing B and C by D and E.
    assert abs(turnover - (2 / 3)) < 1e-9


test_turnover()


# ==========================================
# Test 5: Analyzer
# ==========================================
print("\n【Test 5】 Performance Analyzer")


@test("5.1 Import PerformanceAnalyzer")
def test_analyzer_import():
    from analyzer import PerformanceAnalyzer

    assert PerformanceAnalyzer is not None


test_analyzer_import()


@test("5.2 Calculate metrics")
def test_calculate_metrics():
    from analyzer import PerformanceAnalyzer

    # Create mock results
    dates = pd.date_range("2020-01-31", periods=24, freq="ME")

    results = {
        "dates": dates,
        "portfolio_values": [30000 * (1.01**i) for i in range(30)],
        "benchmark_values": [30000 * (1.005**i) for i in range(30)],
        "returns_df": pd.DataFrame(
            {
                "date": dates,
                "net_return": np.random.randn(24) * 0.02,
                "cost": [0.001] * 24,
            }
        ),
        "trades": [],
        "final_portfolio": {
            "AAPL": {"sector": "Tech", "weight": 0.1},
            "MSFT": {"sector": "Tech", "weight": 0.1},
        },
    }

    analyzer = PerformanceAnalyzer(results)
    metrics = analyzer.calculate_metrics()

    assert "total_return" in metrics
    assert "sharpe_ratio" in metrics
    assert "max_drawdown" in metrics
    assert isinstance(metrics["total_return"], float)


test_calculate_metrics()


@test("5.3 Get portfolio dataframe")
def test_portfolio_df():
    from analyzer import PerformanceAnalyzer

    results = {
        "dates": pd.date_range("2020-01-31", periods=12, freq="ME"),
        "portfolio_values": [30000] * 18,
        "benchmark_values": [30000] * 18,
        "returns_df": pd.DataFrame({"net_return": [0.01] * 12}),
        "trades": [],
        "final_portfolio": {
            "AAPL": {"sector": "Tech", "weight": 0.5, "momentum": 0.1},
            "JPM": {"sector": "Finance", "weight": 0.5, "momentum": 0.08},
        },
    }

    analyzer = PerformanceAnalyzer(results)
    df = analyzer.get_portfolio_dataframe()

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert "Ticker" in df.columns
    assert "Sector" in df.columns


test_portfolio_df()


# ==========================================
# Test 6: Integration Test
# ==========================================
print("\n【Test 6】 Integration Tests")


@test("6.1 End-to-end pipeline (mock data)")
def test_end_to_end():
    from strategy import MomentumStrategy
    from backtester import Backtester
    from analyzer import PerformanceAnalyzer

    # Create complete mock data
    dates = pd.date_range("2020-01-01", periods=500, freq="D")
    tickers = ["T1", "T2", "T3", "T4", "T5", "T6"]

    prices = pd.DataFrame(
        np.random.randn(500, 6).cumsum(axis=0) + 100, index=dates, columns=tickers
    )

    ticker_sector = {"T1": "A", "T2": "A", "T3": "B", "T4": "B", "T5": "C", "T6": "C"}

    sector_weights = pd.DataFrame(
        {"A": [0.33] * 18, "B": [0.33] * 18, "C": [0.34] * 18},
        index=pd.date_range("2020-01-31", periods=18, freq="ME"),
    )

    benchmark = pd.Series(np.random.randn(500).cumsum() + 100, index=dates)

    # Run pipeline
    strategy = MomentumStrategy(prices, ticker_sector, sector_weights, top_n=1)
    backtester = Backtester(strategy, benchmark)
    results = backtester.run()

    analyzer = PerformanceAnalyzer(results)
    metrics = analyzer.calculate_metrics()

    # Validate
    assert results["final_portfolio"] is not None
    assert metrics["total_return"] is not None
    assert metrics["sharpe_ratio"] is not None


test_end_to_end()


@test("6.2 Portfolio weight normalization")
def test_weight_normalization():
    from strategy import MomentumStrategy

    dates = pd.date_range("2020-01-01", periods=200, freq="D")
    tickers = ["A", "B", "C", "D"]

    prices = pd.DataFrame(
        np.random.randn(200, 4).cumsum(axis=0) + 100, index=dates, columns=tickers
    )

    ticker_sector = {"A": "S1", "B": "S1", "C": "S2", "D": "S2"}
    sector_weights = pd.DataFrame({"S1": [0.6], "S2": [0.4]}, index=[dates[-1]])

    strategy = MomentumStrategy(prices, ticker_sector, sector_weights, top_n=1)
    momentum = strategy.calculate_momentum()
    portfolio = strategy.select_portfolio(dates[-1], momentum)

    total = sum(p["weight"] for p in portfolio.values())
    assert abs(total - 1.0) < 0.0001  # Should equal 1.0


test_weight_normalization()


# ==========================================
# Print Summary
# ==========================================
print("\n" + "=" * 70)
print("TEST SUMMARY")
print("=" * 70)

passed = sum(1 for _, status, _ in test_results if status == "PASS")
failed = sum(1 for _, status, _ in test_results if status == "FAIL")

print(f"\nTotal Tests: {len(test_results)}")
print(f"Passed: {passed} [OK]")
print(f"Failed: {failed} [FAIL]")

if failed > 0:
    print("\n" + "=" * 70)
    print("FAILED TESTS:")
    print("=" * 70)
    for name, status, error in test_results:
        if status == "FAIL":
            print(f"\n{name}:")
            print(f"  Error: {error}")

print("\n" + "=" * 70)

if failed == 0:
    print("ALL TESTS PASSED")
else:
    print(f"[WARN] {failed} TEST(S) FAILED - PLEASE FIX")

print("=" * 70)

# Exit code
sys.exit(0 if failed == 0 else 1)
