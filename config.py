# S&P 500 Sector Momentum Strategy Configuration

# Backtest Parameters
START_CAPITAL = 30000
TRANSACTION_COST = 0.001  # 0.1%
SLIPPAGE_COST = 0.0005  # 0.05% execution slippage (one-way)
TOP_N_PER_SECTOR = 1  # 1 stock per sector = 11 stocks total

# Momentum Calculation (in trading days)
MOMENTUM_4W = 21  # 4 weeks
MOMENTUM_13W = 63  # 13 weeks
MOMENTUM_26W = 126  # 26 weeks

# Sector ETF Tickers (for weight reference)
SECTOR_ETFS = {
    "Communication Services": "XLC",
    "Consumer Discretionary": "XLY",
    "Consumer Staples": "XLP",
    "Energy": "XLE",
    "Financials": "XLF",
    "Health Care": "XLV",
    "Industrials": "XLI",
    "Information Technology": "XLK",
    "Materials": "XLB",
    "Real Estate": "XLRE",
    "Utilities": "XLU",
}

# Data Sources
SP500_WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
BENCHMARK_TICKER = "SPY"

# Optional real monthly AUM source (CSV). If file exists, DataManager will use it.
# Expected columns: date, and one column per sector name in SECTOR_ETFS keys.
SECTOR_AUM_CSV = "data/sector_aum_monthly.csv"

# Backtest safety: require local monthly sector AUM CSV for backtest workflows.
# This avoids silently falling back to runtime proxy construction.
REQUIRE_LOCAL_SECTOR_AUM_FOR_BACKTEST = True
