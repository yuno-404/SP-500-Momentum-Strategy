# Maintenance Guide

## Development Workflow

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run core regression tests:

```bash
PYTHONIOENCODING=utf-8 python test_core.py
```

3. Run API locally:

```bash
python api.py
```

4. Run robustness checks:

```bash
python walk_forward.py
python sensitivity_backtest.py
```

## Where to Change What

- **Modify factor logic**: `strategy.py`
- **Modify transaction simulation**: `backtester.py`
- **Modify execution timing (T+1 lag)**: `backtester.py` (`signal_lag_months`)
- **Modify API response contracts**: `services/backtest_service.py`, `services/portfolio_service.py`
- **Modify persistence format**: `services/portfolio_repository.py`
- **Modify data sources**: `services/market_data_service.py`, `data_manager.py`

## Rebalance Policy (Current)

- Uses total account value (`cash + market value`) as target base.
- Converts target allocation to integer shares with floor division.
- If allocation is not enough for 1 share, the allocation remains as cash.
- If live price is unavailable, use last historical close fallback.
- If still no price, ticker is skipped and returned in `skipped_no_price`.

## Known Operational Considerations

- Yahoo Finance can fail intermittently; fallback logic avoids hard failures but can skip some tickers.
- JSON portfolio storage is simple and local; concurrent writes are not supported.
- `portfolio.json` is stateful; back it up before major manual testing.

## Suggested Next Refactors

- Add schema validation for all API request bodies.
- Add unit tests for `services/portfolio_service.py` and `services/backtest_service.py`.
- Replace JSON repository with SQLite/PostgreSQL abstraction while keeping service interfaces unchanged.
