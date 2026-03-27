# Architecture

This project is organized by responsibility so strategy logic, persistence, and HTTP handling can evolve independently.

## High-Level Layers

- `core strategy`: signal generation, backtest engine, metrics.
- `application services`: orchestration for API workflows.
- `interfaces`: Flask HTTP endpoints + static web UI.
- `infrastructure`: JSON repository and file cache.

## Directory Map

```text
.
├── api.py                         # Flask app factory + route registration
├── index.html                     # Frontend UI
├── services/
│   ├── backtest_service.py        # Backtest orchestration + API response shaping
│   ├── market_data_service.py     # DataManager + live price access
│   ├── portfolio_repository.py    # portfolio.json persistence boundary
│   ├── portfolio_service.py       # Holdings, rebalance planning, trade application
│   └── signal_service.py          # Strategy signal generation for live workflows
├── strategy.py                    # Momentum factor + portfolio construction
├── backtester.py                  # Simulation engine
├── analyzer.py                    # Performance metrics and charts
├── data_manager.py                # Historical market and sector data ingestion
├── cache_manager.py               # Cache file lifecycle
└── docs/
    ├── ARCHITECTURE.md
    └── MAINTENANCE.md
```

## Request Flow (Live Rebalance)

1. `api.py` route receives `/api/portfolio/rebalance`.
2. `SignalService` computes latest target portfolio from historical data.
3. `MarketDataService` loads live prices and historical fallback prices.
4. `PortfolioService.build_rebalance_plan()` compares current shares vs target shares.
5. API returns `sells`, `buys`, `holds`, `skipped_no_price`, and `skipped_insufficient_budget`.

## Request Flow (Backtest)

1. `api.py` route receives `/api/backtest` with parameters.
2. `BacktestService` loads data via `MarketDataService`.
3. Strategy + backtester execute.
4. Analyzer computes metrics.
5. Service serializes all values to JSON-safe payload for frontend.

## Design Decisions

- Keep strategy implementation (`strategy.py`, `backtester.py`) framework-agnostic.
- Keep Flask routes thin; business logic lives in `services/`.
- Keep persistence behind repository (`PortfolioRepository`) to ease DB migration later.
- Rebalance is deterministic and integer-share based; residual remains as cash.
