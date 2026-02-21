# Priority Fixes (High -> Low)

This document records the highest-impact fixes applied first, and why they matter.

## 1) Portfolio Construction Consistency (HIGH)

### What was fixed
- `strategy.py` now builds candidates by sector first, then ranks within each target sector.
- Missing sectors no longer get silently replaced by overweighting other sectors.
- Normalization is only applied for tiny floating-point error cases.

### Why needed
- Original flow started from `dropna()` scores and only kept sectors with valid momentum rows.
- That can accidentally drift strategy intent from sector-balanced rotation to concentrated bets.

### If not fixed
- Sector exposure becomes unstable and inconsistent with the original strategy design.
- Performance can look better/worse by accident due to unintended concentration.

## 2) Backtest Annualization Accuracy (HIGH)

### What was fixed
- `full_backtest.py` now computes `actual_years` from `dates[0]` to `dates[-1]` in the post-warmup slice.

### Why needed
- Previous implementation effectively offset start date twice after warm-up.

### If not fixed
- CAGR can be materially biased (typically overstated if years are underestimated).
- Comparison against SPY/SPMO/SPX becomes unreliable.

## 3) Execution Realism: Slippage Cost (HIGH)

### What was fixed
- Added `SLIPPAGE_COST` in `config.py`.
- Backtest transaction cost now uses `turnover * (TRANSACTION_COST + SLIPPAGE_COST)`.

### Why needed
- Real fills are rarely achieved exactly at close/mid prices.

### If not fixed
- Backtest edge is overstated and can fail when moving to live trading.

## 4) Rebalance Transparency for Cash Residuals (HIGH)

### What was fixed
- Rebalance flow already keeps cash when allocation cannot buy one whole share.
- API/UI now report skipped reasons (`no_price` vs `insufficient_budget`).

### Why needed
- With integer shares, cash residual is normal and must be visible.

### If not fixed
- Users may assume strategy is broken when positions are intentionally not opened.

## 5) Execution Timing: T+1 Monthly Proxy (HIGH)

### What was fixed
- `Backtester` now supports `signal_lag_months` (default `1`).
- API exposes `execution_lag_months` in `/api/backtest`.

### Why needed
- Signal generation and execution should not happen at the same bar in a way that risks optimistic assumptions.

### If not fixed
- Strategy can appear slightly better than a realistically executable process.
- Live-vs-backtest drift increases.

## 6) Robustness Validation Tools (HIGH)

### What was fixed
- Added `walk_forward.py` for OOS rolling-window checks.
- Added `sensitivity_backtest.py` for parameter and cost sweeps.

### Why needed
- Single-period backtests are vulnerable to regime luck.

### If not fixed
- You may deploy a strategy that only worked in one historical slice or one parameter point.

## 7) Point-in-Time Universe Backtest (HIGH)

### What was fixed
- Added month-end point-in-time universe reconstruction from S&P 500 changes table.
- Backtest and CLI tools now default to PIT universe.

### Why needed
- Using only today's constituents for the full history introduces survivorship bias.

### If not fixed
- Historical performance is typically overstated.
- Strategy can appear stronger than what was actually investable at the time.

### Current implementation notes
- PIT is reconstructed from Wikipedia changes and current constituents.
- When a removed ticker has no direct sector metadata, the implementation uses event-level proxy sector assignment.
- This is a strong improvement over static-universe backtests, but not equivalent to licensed institutional PIT datasets.

## Can we get true monthly AUM?

Short answer: **usually not from free public APIs in a clean, survivable way**.

- `yfinance` often exposes current `totalAssets`, but not a robust full monthly history for all ETFs.
- Issuer sites may provide fund facts, but coverage, format, and scraping stability vary.
- Professional datasets (Bloomberg, FactSet, Morningstar Direct, Refinitiv) are the reliable source.

Current implementation uses a practical proxy:
- anchor with latest AUM,
- evolve by ETF monthly price ratios,
- normalize cross-sector weights each month.

Optional upgrade now supported:
- If you provide `data/sector_aum_monthly.csv`, `DataManager` will use those monthly AUM values directly.
- This is the preferred path if you have licensed/proprietary monthly AUM data.
