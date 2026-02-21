import math
from datetime import datetime


class PortfolioService:
    """Portfolio state operations and rebalance planning."""

    def __init__(self, repo):
        self.repo = repo

    def get_holdings_with_pnl(self, live_prices):
        state = self.repo.load()
        if not state["holdings"]:
            return {
                "status": "empty",
                "holdings": [],
                "summary": None,
            }

        total_cost = 0.0
        total_market = 0.0
        total_pnl = 0.0
        enriched = []
        for h in state["holdings"]:
            price_now = live_prices.get(h["ticker"])
            cost_basis = h["shares"] * h["buy_price"]
            market_value = h["shares"] * price_now if price_now else cost_basis
            pnl = market_value - cost_basis
            pnl_pct = pnl / cost_basis if cost_basis > 0 else 0
            total_cost += cost_basis
            total_market += market_value
            total_pnl += pnl
            enriched.append(
                {
                    **h,
                    "price_now": round(price_now, 2) if price_now else None,
                    "cost_basis": round(cost_basis, 2),
                    "market_value": round(market_value, 2),
                    "pnl": round(pnl, 2),
                    "pnl_pct": round(pnl_pct * 100, 2),
                }
            )

        return {
            "status": "ok",
            "holdings": enriched,
            "cash": state.get("cash", 0),
            "summary": {
                "total_cost": round(total_cost, 2),
                "total_market": round(total_market, 2),
                "total_pnl": round(total_pnl, 2),
                "total_pnl_pct": round(total_pnl / total_cost * 100, 2)
                if total_cost > 0
                else 0,
                "total_value": round(total_market + state.get("cash", 0), 2),
                "initial_capital": state.get("initial_capital", 0),
            },
        }

    def initialize_buy(self, capital, orders):
        # Reset portfolio for initial buy to avoid duplicated bootstrap trades.
        state = self.repo.empty_state()
        state["initial_capital"] = capital
        state["start_date"] = datetime.now().strftime("%Y-%m-%d")

        total_spent = 0.0
        for order in orders:
            shares = int(order["shares"])
            price = float(order["price"])
            if shares <= 0 or price <= 0:
                continue
            cost = shares * price
            total_spent += cost
            state["holdings"].append(
                {
                    "ticker": order["ticker"],
                    "sector": order["sector"],
                    "shares": shares,
                    "buy_price": price,
                    "buy_date": datetime.now().strftime("%Y-%m-%d"),
                    "weight": order.get("weight", 0),
                }
            )
            state["history"].append(
                {
                    "date": datetime.now().strftime("%Y-%m-%d"),
                    "action": "BUY",
                    "ticker": order["ticker"],
                    "sector": order["sector"],
                    "shares": shares,
                    "price": price,
                    "value": round(cost, 2),
                    "pnl": 0,
                    "pnl_pct": 0,
                }
            )

        state["cash"] = round(capital - total_spent, 2)
        self.repo.save(state)
        return {
            "status": "ok",
            "message": f"Bought {len(state['holdings'])} stocks, cash remaining: ${state['cash']:,.2f}",
        }

    def build_rebalance_plan(self, new_portfolio, live_prices, fallback_prices=None):
        state = self.repo.load()
        current = {h["ticker"]: h for h in state["holdings"]}
        fallback_prices = fallback_prices or {}

        def effective_price(ticker, holding=None):
            px = live_prices.get(ticker)
            if px and px > 0:
                return px
            fb = fallback_prices.get(ticker)
            if fb and fb > 0:
                return fb
            if holding and holding.get("buy_price", 0) > 0:
                return holding["buy_price"]
            return None

        total_value = state.get("cash", 0)
        for h in state["holdings"]:
            px = effective_price(h["ticker"], h)
            total_value += h["shares"] * px if px else 0

        target_shares = {}
        skipped_no_price = []
        skipped_insufficient_budget = []

        for ticker, info in new_portfolio.items():
            px = effective_price(ticker, current.get(ticker))
            if px is None or px <= 0:
                skipped_no_price.append(ticker)
                continue

            target_alloc = info["weight"] * total_value
            shares = int(target_alloc / px)
            if shares == 0 and target_alloc > 0:
                skipped_insufficient_budget.append(
                    {
                        "ticker": ticker,
                        "sector": info["sector"],
                        "target_alloc": round(target_alloc, 2),
                        "price": round(px, 2),
                    }
                )
            target_shares[ticker] = shares

        sells = []
        buys = []
        holds = []

        all_tickers = set(current.keys()) | set(target_shares.keys())
        for ticker in all_tickers:
            h = current.get(ticker)
            current_shares = h["shares"] if h else 0
            target = target_shares.get(ticker, 0)
            px = effective_price(ticker, h)
            if px is None or px <= 0:
                continue

            if target < current_shares:
                sell_shares = current_shares - target
                base_price = h["buy_price"] if h else px
                pnl = (px - base_price) * sell_shares
                pnl_pct = (
                    ((px - base_price) / base_price * 100) if base_price > 0 else 0
                )
                sells.append(
                    {
                        "ticker": ticker,
                        "sector": h["sector"]
                        if h
                        else new_portfolio.get(ticker, {}).get("sector", ""),
                        "shares": sell_shares,
                        "buy_price": base_price,
                        "price_now": round(px, 2),
                        "pnl": round(pnl, 2),
                        "pnl_pct": round(pnl_pct, 2),
                        "sell_value": round(sell_shares * px, 2),
                    }
                )
            elif target > current_shares:
                info = new_portfolio[ticker]
                buy_shares = target - current_shares
                momentum = info["momentum"]
                buys.append(
                    {
                        "ticker": ticker,
                        "sector": info["sector"],
                        "weight": round(info["weight"] * 100, 2),
                        "momentum": round(momentum * 100, 2)
                        if not (math.isnan(momentum) or math.isinf(momentum))
                        else 0,
                        "price_now": round(px, 2),
                        "shares": buy_shares,
                        "cost": round(buy_shares * px, 2),
                    }
                )
            elif h:
                pnl = (px - h["buy_price"]) * current_shares
                pnl_pct = (
                    ((px - h["buy_price"]) / h["buy_price"] * 100)
                    if h["buy_price"] > 0
                    else 0
                )
                holds.append(
                    {
                        "ticker": ticker,
                        "sector": h["sector"],
                        "shares": current_shares,
                        "buy_price": h["buy_price"],
                        "price_now": round(px, 2),
                        "pnl": round(pnl, 2),
                        "pnl_pct": round(pnl_pct, 2),
                    }
                )

        sell_proceeds = sum(x["sell_value"] for x in sells)
        available_cash = state.get("cash", 0) + sell_proceeds

        return {
            "status": "ok",
            "sells": sells,
            "buys": buys,
            "holds": holds,
            "available_cash": round(available_cash, 2),
            "portfolio_value": round(total_value, 2),
            "skipped_no_price": skipped_no_price,
            "skipped_insufficient_budget": skipped_insufficient_budget,
        }

    def apply_rebalance(self, sells, buys):
        state = self.repo.load()
        cash = state.get("cash", 0)
        skipped_sells = []
        skipped_buys = []
        skipped_insufficient_cash = []

        for sell in sells:
            ticker = sell["ticker"]
            price = sell["price"]
            shares = sell["shares"]
            if price is None or price <= 0 or shares is None or shares <= 0:
                skipped_sells.append(ticker)
                continue

            for i, h in enumerate(state["holdings"]):
                if h["ticker"] != ticker:
                    continue
                shares = min(int(shares), int(h["shares"]))
                if shares <= 0:
                    break
                value = shares * price
                pnl = (price - h["buy_price"]) * shares
                pnl_pct = (
                    ((price - h["buy_price"]) / h["buy_price"] * 100)
                    if h["buy_price"] > 0
                    else 0
                )
                state["history"].append(
                    {
                        "date": datetime.now().strftime("%Y-%m-%d"),
                        "action": "SELL",
                        "ticker": ticker,
                        "sector": h["sector"],
                        "shares": shares,
                        "price": price,
                        "value": round(value, 2),
                        "buy_price": h["buy_price"],
                        "pnl": round(pnl, 2),
                        "pnl_pct": round(pnl_pct, 2),
                    }
                )
                cash += value
                remaining = h["shares"] - shares
                if remaining <= 0:
                    state["holdings"].pop(i)
                else:
                    state["holdings"][i]["shares"] = remaining
                break

        for buy in buys:
            ticker = buy["ticker"]
            price = buy["price"]
            shares = buy["shares"]
            if price is None or price <= 0 or shares is None or shares <= 0:
                skipped_buys.append(ticker)
                continue

            shares = int(shares)
            cost = shares * price
            if cost > cash:
                skipped_insufficient_cash.append(ticker)
                continue
            cash -= cost

            existing = next(
                (h for h in state["holdings"] if h["ticker"] == ticker), None
            )
            if existing:
                old = existing["shares"]
                new_total = old + shares
                if new_total > 0:
                    existing["buy_price"] = round(
                        (existing["buy_price"] * old + price * shares) / new_total, 6
                    )
                existing["shares"] = new_total
                existing["buy_date"] = datetime.now().strftime("%Y-%m-%d")
                existing["sector"] = buy.get("sector", existing.get("sector", ""))
                existing["weight"] = buy.get("weight", existing.get("weight", 0))
            else:
                state["holdings"].append(
                    {
                        "ticker": ticker,
                        "sector": buy.get("sector", ""),
                        "shares": shares,
                        "buy_price": price,
                        "buy_date": datetime.now().strftime("%Y-%m-%d"),
                        "weight": buy.get("weight", 0),
                    }
                )

            state["history"].append(
                {
                    "date": datetime.now().strftime("%Y-%m-%d"),
                    "action": "BUY",
                    "ticker": ticker,
                    "sector": buy.get("sector", ""),
                    "shares": shares,
                    "price": price,
                    "value": round(cost, 2),
                    "pnl": 0,
                    "pnl_pct": 0,
                }
            )

        state["cash"] = round(cash, 2)
        self.repo.save(state)

        msg = f"Rebalance done. Cash: ${cash:,.2f}"
        if skipped_sells or skipped_buys:
            msg += f" (Skipped invalid prices: sells={len(skipped_sells)}, buys={len(skipped_buys)})"
        if skipped_insufficient_cash:
            msg += f" (Skipped for insufficient cash: {len(skipped_insufficient_cash)})"

        return {
            "status": "ok",
            "message": msg,
        }

    def get_history(self):
        state = self.repo.load()
        return {
            "status": "ok",
            "history": state.get("history", []),
            "initial_capital": state.get("initial_capital", 0),
            "start_date": state.get("start_date"),
        }
