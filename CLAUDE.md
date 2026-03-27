# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

S&P 500 Sector Momentum Strategy — a research backtesting framework + live portfolio management system. Python 3.10+, Flask API, single-page frontend.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run tests
python tests/test_all.py      # Full suite (17 tests)
python tests/test_core.py     # Core logic regression

# Backtest (research)
python scripts/full_backtest.py --years 20
python scripts/walk_forward.py --years 20 --train-months 36 --test-months 12
python scripts/sensitivity_backtest.py --years 20
python scripts/trade_analysis.py --years 20   # Per-trade metrics, no compounding

# Build required data
python scripts/build_sector_aum_csv.py --years 20

# Live trading signals
python scripts/today_buy.py --capital 50000

# Start API server (http://localhost:5000)
python api.py
```

Root-level `full_backtest.py`, `walk_forward.py`, etc. are wrappers that delegate to `scripts/`.

## Architecture

**Layered design**: Core strategy (framework-agnostic) → Services (business logic) → API (Flask routes).

### Core Layer
- `strategy.py` — Momentum calculation (3-factor average of 4W/13W/26W returns), stock selection (top N per sector), sector-weighted portfolio construction
- `backtester.py` — Monthly rebalancing simulation with transaction costs, slippage, signal lag, and optional stop-loss
- `analyzer.py` — Performance metrics (Sharpe, Sortino, Calmar, MDD) and visualization
- `stop_loss.py` — Individual stock stops (-12%), trailing stops (-15%), portfolio-level REDUCE/HALT

### Services Layer (`services/`)
- `signal_service.py` — Generate current month's buy signals
- `portfolio_service.py` — Holdings tracking, rebalance planning, P&L
- `backtest_service.py` — Orchestrates backtest for API consumption
- `market_data_service.py` — Data loading with live price enrichment
- `portfolio_repository.py` — JSON persistence (`portfolio.json`)

### Infrastructure
- `data_manager.py` — Yahoo Finance prices, Wikipedia S&P 500 list, sector AUM weights, benchmark data (SPY/SPMO/SPX). All cached as pickle files with 24h TTL via `cache_manager.py`
- `config.py` — Global parameters: `START_CAPITAL=30000`, `TOP_N_PER_SECTOR=1`, momentum lookbacks, sector ETF tickers, cost assumptions
- `fmp_adapter.py` — Optional Financial Modeling Prep API (requires `FMP_API_KEY` in `.env`)

## Key Design Decisions

- **Sector AUM weights** from `data/sector_aum_monthly.csv` drive portfolio allocation. `REQUIRE_LOCAL_SECTOR_AUM_FOR_BACKTEST=True` prevents silent fallback to proxies.
- **Point-in-time universe** adjusts S&P 500 membership retroactively to mitigate survivorship bias. Toggle with `--no-pit`.
- **Signal lag** (`signal_lag_months=1`): uses previous month's momentum signal for current month's trades (T+1 proxy).
- **7-month warm-up**: first 7 months are skipped to ensure 26-week momentum data is available.

## API Endpoints

- `POST /api/backtest` — Run backtest with custom parameters
- `GET /api/portfolio/signals` — Current buy signals
- `GET /api/portfolio/holdings` — Holdings + P&L
- `GET /api/portfolio/rebalance` — Target vs current comparison
- `POST /api/portfolio/confirm-rebalance` — Execute monthly rebalance
