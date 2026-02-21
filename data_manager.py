"""
Data Manager - Handle all data download and sector weight calculation
"""

import yfinance as yf
import pandas as pd
import requests
from io import StringIO
from datetime import datetime
from pathlib import Path
from config import (
    SP500_WIKI_URL,
    SECTOR_ETFS,
    BENCHMARK_TICKER,
    SECTOR_AUM_CSV,
)
from cache_manager import CacheManager


class DataManager:
    """Manages all data operations"""

    def __init__(self, use_cache=True):
        self.ticker_sector = None
        self.sector_weights_ts = None
        self.use_cache = use_cache
        self.cache = CacheManager() if use_cache else None

    def get_sp500_components(self):
        """Get S&P 500 tickers and their sectors"""

        # Try cache first
        if self.use_cache:
            cached = self.cache.load_cache("components")
            if cached:
                self.ticker_sector = cached
                print(f"[OK] Loaded {len(self.ticker_sector)} stocks from cache")
                return self.ticker_sector

        print("[INFO] Fetching S&P 500 components from web...")
        headers = {"User-Agent": "Mozilla/5.0"}

        df = pd.read_html(StringIO(requests.get(SP500_WIKI_URL, headers=headers).text))[
            0
        ]
        df["Symbol"] = df["Symbol"].str.replace(".", "-", regex=False)

        self.ticker_sector = df.set_index("Symbol")["GICS Sector"].to_dict()

        # Save to cache
        if self.use_cache:
            self.cache.save_cache(self.ticker_sector, "components")

        print(
            f"[OK] Got {len(self.ticker_sector)} stocks across {len(set(self.ticker_sector.values()))} sectors"
        )

        return self.ticker_sector

    @staticmethod
    def _normalize_ticker(value):
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return None
        ticker = str(value).strip().upper().replace(".", "-")
        return ticker if ticker else None

    def _download_sp500_changes(self):
        """Download S&P 500 changes table from Wikipedia."""
        headers = {"User-Agent": "Mozilla/5.0"}
        html = requests.get(SP500_WIKI_URL, headers=headers, timeout=30).text
        tables = pd.read_html(StringIO(html))

        changes_table = None
        for table in tables:
            cols = [str(c).lower() for c in table.columns]
            if any("date" in c for c in cols) and any(
                ("added" in c or "removed" in c) for c in cols
            ):
                if changes_table is None or len(table) > len(changes_table):
                    changes_table = table

        if changes_table is None:
            return pd.DataFrame(columns=["date", "added", "removed"])

        df = changes_table.copy()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [
                "_".join([str(x) for x in col if str(x) != "nan"]).strip("_")
                for col in df.columns
            ]

        col_map = {str(c).lower(): c for c in df.columns}
        date_col = next((col_map[k] for k in col_map if "date" in k), None)
        added_col = next(
            (
                col_map[k]
                for k in col_map
                if "added" in k and ("ticker" in k or "symbol" in k)
            ),
            None,
        )
        removed_col = next(
            (
                col_map[k]
                for k in col_map
                if "removed" in k and ("ticker" in k or "symbol" in k)
            ),
            None,
        )

        if added_col is None:
            added_col = next((col_map[k] for k in col_map if "added" in k), None)
        if removed_col is None:
            removed_col = next((col_map[k] for k in col_map if "removed" in k), None)

        if date_col is None:
            return pd.DataFrame(columns=["date", "added", "removed"])

        out = pd.DataFrame(
            {
                "date": pd.to_datetime(df[date_col], errors="coerce"),
                "added": df[added_col].map(self._normalize_ticker)
                if added_col is not None
                else None,
                "removed": df[removed_col].map(self._normalize_ticker)
                if removed_col is not None
                else None,
            }
        )
        out = out.dropna(subset=["date"]).sort_values("date")
        return out

    def get_point_in_time_universe_by_month(self, start_date, end_date=None):
        """Approximate point-in-time S&P 500 membership at month end."""
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")

        current = self.get_sp500_components().copy()
        try:
            changes = self._download_sp500_changes()
        except Exception as e:
            print(
                f"[WARN] Failed to fetch S&P 500 changes ({e}); using static universe"
            )
            changes = pd.DataFrame(columns=["date", "added", "removed"])

        month_ends = pd.date_range(start=start_date, end=end_date, freq="ME")
        if len(month_ends) == 0:
            return {}

        if len(changes) == 0:
            return {d: current.copy() for d in month_ends}

        changes_desc = changes.sort_values("date", ascending=False).reset_index(
            drop=True
        )

        active = set(current.keys())
        sector_map = current.copy()
        by_month = {}
        idx = 0

        for dt in sorted(month_ends, reverse=True):
            while idx < len(changes_desc) and changes_desc.loc[idx, "date"] > dt:
                added = changes_desc.loc[idx, "added"]
                removed = changes_desc.loc[idx, "removed"]

                if added:
                    active.discard(added)

                if removed:
                    active.add(removed)
                    if removed not in sector_map:
                        proxy_sector = sector_map.get(added)
                        if proxy_sector:
                            sector_map[removed] = proxy_sector

                idx += 1

            month_map = {t: sector_map[t] for t in active if t in sector_map}
            by_month[dt] = month_map

        return {d: by_month[d] for d in sorted(by_month.keys())}

    def download_stock_prices(self, tickers, start_date):
        """Download historical stock prices"""

        # Try cache first
        if self.use_cache:
            cached = self.cache.load_cache("stock_prices", start_date=start_date)
            if cached is not None:
                print(f"[OK] Loaded prices from cache ({len(cached.columns)} stocks)")
                return cached

        print(f"[INFO] Downloading {len(tickers)} stock prices from Yahoo Finance...")

        raw_data = yf.download(
            list(tickers), start=start_date, progress=True, threads=True
        )

        if "Adj Close" in raw_data.columns:
            prices = raw_data["Adj Close"]
        elif "Close" in raw_data.columns:
            prices = raw_data["Close"]
        else:
            prices = raw_data.xs("Close", axis=1, level=0)

        prices = prices.dropna(axis=1, how="all")

        # Save to cache
        if self.use_cache:
            self.cache.save_cache(prices, "stock_prices", start_date=start_date)

        print(
            f"[OK] Downloaded {len(prices.columns)} stocks from {prices.index[0].date()} to {prices.index[-1].date()}"
        )

        return prices

    def download_sector_etf_weights(self, start_date, require_local_csv=False):
        """
        Download Sector ETF prices and calculate dynamic weights
        This represents the true S&P 500 sector allocation over time.
        Uses totalAssets (AUM) as primary metric, falls back to sharesOutstanding * price.
        """
        # Try cache first
        if self.use_cache and not require_local_csv:
            cached = self.cache.load_cache("sector_weights", start_date=start_date)
            if cached is not None:
                self.sector_weights_ts = cached
                print(
                    f"[OK] Loaded sector weights from cache ({len(self.sector_weights_ts)} months)"
                )
                return self.sector_weights_ts

        # Optional: use true monthly AUM CSV if available
        true_aum = self._load_true_monthly_aum(start_date)
        if true_aum is not None:
            self.sector_weights_ts = true_aum
            if self.use_cache and not require_local_csv:
                self.cache.save_cache(
                    self.sector_weights_ts, "sector_weights", start_date=start_date
                )
            print(
                f"[OK] Sector weights loaded from monthly AUM CSV ({len(self.sector_weights_ts)} months)"
            )
            return self.sector_weights_ts

        if require_local_csv:
            raise ValueError(
                "Local sector AUM CSV is required but unavailable/invalid. "
                f"Please provide {SECTOR_AUM_CSV}."
            )

        # Optional: build monthly AUM from historical shares outstanding * close.
        # This avoids using a single "today" AUM anchor as the primary source.
        shares_aum = self._build_monthly_aum_from_yf_shares(start_date)
        if shares_aum is not None:
            self.sector_weights_ts = shares_aum
            if self.use_cache:
                self.cache.save_cache(
                    self.sector_weights_ts, "sector_weights", start_date=start_date
                )
            print(
                f"[OK] Sector weights built from yfinance shares*price ({len(self.sector_weights_ts)} months)"
            )
            return self.sector_weights_ts

        print("[INFO] Downloading Sector ETF data for weight calculation...")

        etf_tickers = list(SECTOR_ETFS.values())
        etf_data = yf.download(etf_tickers, start=start_date, progress=False)

        if "Adj Close" in etf_data.columns:
            etf_prices = etf_data["Adj Close"]
        else:
            etf_prices = etf_data["Close"]

        # Get AUM or market cap for each ETF
        print("[INFO] Calculating sector weights from ETF data...")
        etf_aum = {}
        for sector, ticker in SECTOR_ETFS.items():
            try:
                etf = yf.Ticker(ticker)
                info = etf.info
                # Prefer totalAssets (AUM) — works for ALL ETFs including XLC, XLRE
                total_assets = info.get("totalAssets", None)
                shares = info.get("sharesOutstanding", None)

                if total_assets and total_assets > 0:
                    etf_aum[sector] = total_assets
                    print(f"  [OK] {ticker} ({sector}): AUM ${total_assets / 1e9:.1f}B")
                elif shares and ticker in etf_prices.columns:
                    # Fallback: shares * latest price
                    latest_price = etf_prices[ticker].dropna().iloc[-1]
                    etf_aum[sector] = shares * latest_price
                    print(
                        f"  [OK] {ticker} ({sector}): MCap ${shares * latest_price / 1e9:.1f}B (shares fallback)"
                    )
                else:
                    print(f"  [WARN] {ticker} ({sector}): No data available")
            except Exception as e:
                print(f"  [WARN] {ticker} error: {e}")

        # Build dynamic monthly weights.
        # We anchor each ETF with current AUM, then evolve it by monthly ETF price changes:
        # proxy_aum_t = base_aum * (price_t / price_0)
        monthly_prices = etf_prices.resample("ME").last()
        dynamic_aum = {}

        for sector, ticker in SECTOR_ETFS.items():
            base_aum = etf_aum.get(sector)
            if base_aum is None or ticker not in monthly_prices.columns:
                continue

            series = monthly_prices[ticker].dropna()
            if len(series) == 0:
                continue

            base_price = series.iloc[0]
            if pd.isna(base_price) or base_price <= 0:
                continue

            dynamic_aum[sector] = base_aum * (monthly_prices[ticker] / base_price)

        if not dynamic_aum:
            raise ValueError(
                "No valid sector ETF AUM/price data to compute sector weights"
            )

        dynamic_aum_df = pd.DataFrame(dynamic_aum).ffill().bfill()
        row_sums = dynamic_aum_df.sum(axis=1)
        self.sector_weights_ts = dynamic_aum_df.div(row_sums, axis=0)

        # Save to cache
        if self.use_cache:
            self.cache.save_cache(
                self.sector_weights_ts, "sector_weights", start_date=start_date
            )

        latest_weights = self.sector_weights_ts.iloc[-1].dropna().to_dict()
        print(
            f"\n[OK] Sector weights calculated for {len(self.sector_weights_ts)} months, {len(latest_weights)} sectors"
        )
        print("\n[INFO] Sector weights:")
        for sector, weight in sorted(latest_weights.items(), key=lambda x: -x[1]):
            print(f"  {sector:<30} {weight * 100:>6.2f}%")

        return self.sector_weights_ts

    def export_sector_aum_csv_snapshot(
        self,
        start_date,
        output_path=SECTOR_AUM_CSV,
        allow_proxy_fallback=True,
    ):
        """Build and export monthly sector AUM table for offline backtests.

        Priority:
        1) yfinance shares*close (preferred, no future AUM anchor)
        2) proxy current AUM + price scaling (optional fallback)
        """
        shares_weights = self._build_monthly_aum_from_yf_shares(start_date)
        if shares_weights is not None and len(shares_weights) > 0:
            out = shares_weights.copy()
            out.index.name = "date"
            out = out.reset_index()
            out.to_csv(output_path, index=False)
            return {
                "rows": len(out),
                "source": "shares_x_price",
                "path": output_path,
            }

        if not allow_proxy_fallback:
            raise ValueError(
                "Unable to build sector AUM from shares*price and proxy fallback is disabled."
            )

        # Build proxy AUM table from current AUM anchor and monthly prices
        etf_tickers = list(SECTOR_ETFS.values())
        etf_data = yf.download(etf_tickers, start=start_date, progress=False)
        if "Adj Close" in etf_data.columns:
            etf_prices = etf_data["Adj Close"]
        else:
            etf_prices = etf_data["Close"]

        monthly_prices = etf_prices.resample("ME").last()
        proxy_aum = {}

        for sector, ticker in SECTOR_ETFS.items():
            try:
                info = yf.Ticker(ticker).info
            except Exception:
                info = {}
            base_aum = info.get("totalAssets", None)

            if not base_aum or ticker not in monthly_prices.columns:
                continue

            series = monthly_prices[ticker].dropna()
            if len(series) == 0:
                continue

            base_price = series.iloc[0]
            if pd.isna(base_price) or base_price <= 0:
                continue

            proxy_aum[sector] = base_aum * (monthly_prices[ticker] / base_price)

        if not proxy_aum:
            raise ValueError("Failed to build sector AUM snapshot from all methods")

        out = pd.DataFrame(proxy_aum).dropna(how="all")
        out.index.name = "date"
        out = out.reset_index()
        out.to_csv(output_path, index=False)
        return {
            "rows": len(out),
            "source": "proxy_current_aum_scaled",
            "path": output_path,
        }

    def _build_monthly_aum_from_yf_shares(self, start_date):
        """Build sector weights via monthly (shares outstanding * close) per ETF.

        Notes:
        - yfinance shares history can be sparse/missing by ticker/date.
        - We interpolate then ffill/bfill on monthly grid for robustness.
        - If insufficient data, return None so caller can fallback.
        """
        end_date = datetime.now().strftime("%Y-%m-%d")
        etf_tickers = list(SECTOR_ETFS.values())

        try:
            etf_data = yf.download(etf_tickers, start=start_date, progress=False)
            if "Adj Close" in etf_data.columns:
                daily_close = etf_data["Adj Close"]
            else:
                daily_close = etf_data["Close"]
        except Exception as e:
            print(f"[WARN] Failed to download ETF prices for shares-based AUM ({e})")
            return None

        if daily_close is None or len(daily_close) == 0:
            return None

        monthly_close = daily_close.resample("ME").last()
        if len(monthly_close) == 0:
            return None

        monthly_aum = {}
        for sector, ticker in SECTOR_ETFS.items():
            if ticker not in monthly_close.columns:
                continue

            try:
                shares = yf.Ticker(ticker).get_shares_full(
                    start=start_date, end=end_date
                )
            except Exception as e:
                print(f"  [WARN] {ticker} shares history error: {e}")
                continue

            if shares is None or len(shares) == 0:
                print(f"  [WARN] {ticker}: no historical sharesOutstanding data")
                continue

            shares = pd.Series(shares, copy=False).astype(float)
            shares = shares[~shares.index.duplicated(keep="last")]

            # Normalize timezone for alignment with monthly_close index
            try:
                if getattr(shares.index, "tz", None) is not None:
                    shares.index = shares.index.tz_localize(None)
            except Exception:
                pass

            shares = shares.sort_index()
            shares_m = shares.resample("ME").last().reindex(monthly_close.index)
            shares_m = shares_m.interpolate(method="time", limit_direction="both")
            shares_m = shares_m.ffill().bfill()

            close_m = monthly_close[ticker].astype(float)
            close_m = close_m.ffill().bfill()

            if shares_m.notna().sum() == 0 or close_m.notna().sum() == 0:
                continue

            aum_series = shares_m * close_m
            if aum_series.notna().sum() == 0:
                continue

            monthly_aum[sector] = aum_series

        if not monthly_aum:
            return None

        aum_df = pd.DataFrame(monthly_aum).dropna(how="all")
        if len(aum_df) == 0:
            return None

        row_sums = aum_df.sum(axis=1)
        valid = row_sums > 0
        weights = aum_df.loc[valid].div(row_sums.loc[valid], axis=0)

        if len(weights) == 0:
            return None

        covered = len(weights.columns)
        print(
            f"[INFO] Shares-based sector AUM ready ({covered}/{len(SECTOR_ETFS)} sectors covered)"
        )
        return weights

    def _load_true_monthly_aum(self, start_date):
        """Load real monthly AUM from local CSV if provided."""
        csv_path = Path(SECTOR_AUM_CSV)
        if not csv_path.exists():
            return None

        try:
            df = pd.read_csv(csv_path)
            if "date" not in df.columns:
                print("[WARN] AUM CSV missing 'date' column, fallback to proxy weights")
                return None

            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date").sort_index()
            df = df[df.index >= pd.to_datetime(start_date)]

            expected_cols = list(SECTOR_ETFS.keys())
            available_cols = [c for c in expected_cols if c in df.columns]
            if len(available_cols) == 0:
                print(
                    "[WARN] AUM CSV has no matching sector columns, fallback to proxy"
                )
                return None

            aum = df[available_cols].copy()
            aum = aum.resample("ME").last().ffill().dropna(how="all")
            row_sums = aum.sum(axis=1)
            valid = row_sums > 0
            weights = aum.loc[valid].div(row_sums.loc[valid], axis=0)

            if len(weights) == 0:
                print(
                    "[WARN] AUM CSV has no valid rows after cleaning, fallback to proxy"
                )
                return None

            print(f"[INFO] Using true monthly AUM from: {csv_path}")
            return weights
        except Exception as e:
            print(f"[WARN] Failed to parse AUM CSV ({e}), fallback to proxy")
            return None

    def download_benchmark(self, start_date):
        """Download benchmarks (SPY, SPMO, S&P 500 Index)"""

        # Try cache first
        if self.use_cache:
            cached = self.cache.load_cache("benchmark", start_date=start_date)
            if cached is not None:
                print(f"[OK] Loaded benchmarks from cache")
                return cached

        print(f"[INFO] Downloading benchmarks (SPY, SPMO, ^GSPC)...")

        # Download all benchmarks
        benchmarks = yf.download(
            ["SPY", "SPMO", "^GSPC"], start=start_date, progress=False
        )

        if "Adj Close" in benchmarks.columns:
            prices = benchmarks["Adj Close"].copy()
        else:
            prices = benchmarks["Close"].copy()

        # Rename columns for clarity
        if isinstance(prices, pd.DataFrame):
            # Handle MultiIndex columns
            if isinstance(prices.columns, pd.MultiIndex):
                prices.columns = prices.columns.get_level_values(-1)

            # Standardize column names
            column_mapping = {
                "SPY": "SPY",
                "SPMO": "SPMO",
                "^GSPC": "SPX",
                "GSPC": "SPX",
            }
            prices = prices.rename(columns=column_mapping)

            # Ensure we have the expected columns
            expected_cols = ["SPY", "SPMO", "SPX"]
            prices = prices[expected_cols]

            # Forward fill NaN values (use previous day's price)
            prices = prices.ffill()

            # If still NaN at start, backfill
            prices = prices.bfill()

            print(f"[OK] Benchmark columns: {prices.columns.tolist()}")
            print(f"[OK] Date range: {prices.index[0]} to {prices.index[-1]}")
            print(f"[OK] Total trading days: {len(prices)}")

        # Save to cache
        if self.use_cache:
            self.cache.save_cache(prices, "benchmark", start_date=start_date)

        print(f"[OK] Benchmarks data ready")
        return prices

    def get_sector_weight_at_date(self, date):
        """Get sector weights for a specific date"""
        if self.sector_weights_ts is None:
            raise ValueError(
                "Sector weights not loaded. Call download_sector_etf_weights() first."
            )

        # Find closest previous date
        available_dates = self.sector_weights_ts.index[
            self.sector_weights_ts.index <= date
        ]

        if len(available_dates) == 0:
            return None

        return self.sector_weights_ts.loc[available_dates[-1]]


if __name__ == "__main__":
    # Test
    dm = DataManager()

    # Test 1: Get components
    ticker_sector = dm.get_sp500_components()
    print(f"\nTest: Got {len(ticker_sector)} tickers")

    # Test 2: Download sector weights
    sector_weights = dm.download_sector_etf_weights("2020-01-01")
    print(f"\nTest: Sector weights shape: {sector_weights.shape}")

    # Test 3: Get weight at specific date
    test_date = pd.Timestamp("2023-12-31")
    weights = dm.get_sector_weight_at_date(test_date)
    print(f"\nTest: Weights at {test_date.date()}:")
    print(weights)
