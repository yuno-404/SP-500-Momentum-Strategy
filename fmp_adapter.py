"""Financial Modeling Prep adapter and helpers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, Optional

import pandas as pd
import requests

from env_config import get_fmp_api_key


class FMPAPIError(RuntimeError):
    """Base error for FMP adapter."""


class FMPAuthError(FMPAPIError):
    """Invalid or missing API key."""


class FMPPlanLimitError(FMPAPIError):
    """Endpoint is not available for current subscription."""


@dataclass
class FMPAdapter:
    api_key: Optional[str] = None
    base_url: str = "https://financialmodelingprep.com/stable"
    timeout: int = 20

    def __post_init__(self):
        if not self.api_key:
            self.api_key = get_fmp_api_key(required=True)
        self.base_url = self.base_url.rstrip("/")

    def _request_json(self, endpoint: str, params: Optional[dict] = None):
        request_params = dict(params or {})
        request_params["apikey"] = self.api_key

        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        resp = requests.get(url, params=request_params, timeout=self.timeout)

        if resp.status_code in (401, 403):
            raise FMPAuthError(
                "FMP authentication failed. Check FMP_API_KEY in .env or environment."
            )

        if resp.status_code == 402:
            message = resp.text.strip()[:300]
            raise FMPPlanLimitError(
                f"FMP endpoint '{endpoint}' is not available for your plan: {message}"
            )

        if resp.status_code >= 400:
            raise FMPAPIError(
                f"FMP request failed ({resp.status_code}) for '{endpoint}': {resp.text[:300]}"
            )

        try:
            return resp.json()
        except ValueError as exc:
            raise FMPAPIError(
                f"FMP response is not valid JSON for '{endpoint}'"
            ) from exc

    def search_name(self, query: str, limit: int = 10):
        data = self._request_json("search-name", {"query": query})
        if not isinstance(data, list):
            return []
        return data[: max(0, int(limit))]

    def quote(self, symbol: str) -> dict:
        data = self._request_json("quote", {"symbol": symbol})
        if isinstance(data, list) and data:
            return data[0]
        return {}

    def profile(self, symbol: str) -> dict:
        data = self._request_json("profile", {"symbol": symbol})
        if isinstance(data, list) and data:
            return data[0]
        return {}

    def historical_price_eod_full(
        self,
        symbol: str,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
    ) -> pd.DataFrame:
        params = {"symbol": symbol}
        if from_date:
            params["from"] = from_date
        if to_date:
            params["to"] = to_date

        data = self._request_json("historical-price-eod/full", params)
        if not isinstance(data, list) or len(data) == 0:
            return pd.DataFrame(columns=["close"])

        df = pd.DataFrame(data)
        if "date" not in df.columns:
            return pd.DataFrame(columns=["close"])

        close_col = "close" if "close" in df.columns else None
        if close_col is None:
            return pd.DataFrame(columns=["close"])

        out = df[["date", close_col]].copy()
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
        out = out.dropna(subset=["date"]).sort_values("date")
        out = out.set_index("date")
        out.columns = ["close"]
        return out

    def _current_etf_assets(self, symbol: str) -> Optional[float]:
        info = self.profile(symbol)
        for key in ("totalAssets", "aum", "netAssets", "marketCap"):
            value = info.get(key)
            if value is None:
                continue
            try:
                num = float(value)
            except (TypeError, ValueError):
                continue
            if num > 0:
                return num
        return None

    def build_sector_proxy_aum_monthly(
        self,
        sector_etfs: Dict[str, str],
        start_date: str,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")

        month_index = pd.date_range(start=start_date, end=end_date, freq="ME")
        if len(month_index) == 0:
            return pd.DataFrame()

        data = {}
        for sector, ticker in sector_etfs.items():
            base_assets = self._current_etf_assets(ticker)
            if base_assets is None:
                continue

            prices = self.historical_price_eod_full(
                ticker,
                from_date=start_date,
                to_date=end_date,
            )
            if len(prices) == 0:
                continue

            monthly_close = prices["close"].resample("ME").last()
            monthly_close = monthly_close.reindex(month_index).ffill().bfill()
            valid = monthly_close.dropna()
            if len(valid) == 0:
                continue

            base_price = float(valid.iloc[0])
            if base_price <= 0:
                continue

            data[sector] = base_assets * (monthly_close / base_price)

        if not data:
            return pd.DataFrame()

        return pd.DataFrame(data).dropna(how="all")

    def export_sector_proxy_aum_csv(
        self,
        output_path: str,
        sector_etfs: Dict[str, str],
        start_date: str,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        df = self.build_sector_proxy_aum_monthly(
            sector_etfs=sector_etfs,
            start_date=start_date,
            end_date=end_date,
        )
        if len(df) == 0:
            raise FMPAPIError(
                "No sector AUM rows generated. Check plan access and endpoint data availability."
            )

        out = df.copy()
        out.index.name = "date"
        out = out.reset_index()
        out.to_csv(output_path, index=False)
        return out


def first_nonempty_name(records: Iterable[dict]) -> str:
    for row in records:
        name = str(row.get("name", "")).strip()
        if name:
            return name
    return ""
