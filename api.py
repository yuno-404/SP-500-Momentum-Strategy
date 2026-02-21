"""
Flask API Backend for S&P 500 Momentum Strategy.
Run: python api.py
Access: http://localhost:5000
"""

import os
import traceback

from flask import Flask, jsonify, request, send_file
from flask_cors import CORS

from cache_manager import CacheManager
from config import TOP_N_PER_SECTOR
from services.backtest_service import BacktestService
from services.market_data_service import MarketDataService
from services.portfolio_repository import PortfolioRepository
from services.portfolio_service import PortfolioService
from services.signal_service import SignalService


def create_app():
    app = Flask(__name__)
    CORS(app)

    root_dir = os.path.dirname(__file__)
    portfolio_file = os.path.join(root_dir, "portfolio.json")
    cache_manager = CacheManager()

    market_service = MarketDataService(use_cache=True)
    signal_service = SignalService(market_service, TOP_N_PER_SECTOR)
    portfolio_repo = PortfolioRepository(portfolio_file)
    portfolio_service = PortfolioService(portfolio_repo)
    backtest_service = BacktestService(market_service, TOP_N_PER_SECTOR)

    @app.route("/api/portfolio/holdings", methods=["GET"])
    def get_holdings():
        """Get current holdings with live prices and unrealized P&L."""
        try:
            state = portfolio_repo.load()
            tickers = [h["ticker"] for h in state["holdings"]]
            live_prices = market_service.get_live_prices(tickers)
            return jsonify(portfolio_service.get_holdings_with_pnl(live_prices))
        except Exception as e:
            return jsonify(
                {
                    "status": "error",
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                }
            ), 500

    @app.route("/api/portfolio/buy", methods=["POST"])
    def confirm_buy():
        """Confirm initial buy orders and initialize local portfolio."""
        try:
            data = request.get_json() or {}
            capital = float(data["capital"])
            orders = data.get("orders", [])
            return jsonify(portfolio_service.initialize_buy(capital, orders))
        except Exception as e:
            return jsonify({"status": "error", "error": str(e)}), 500

    @app.route("/api/portfolio/signals", methods=["GET"])
    def get_signals():
        """Get latest strategy signals."""
        try:
            start_date = market_service.default_start_date(days_back=365)
            latest_date, portfolio, stock_prices = signal_service.get_latest_portfolio(
                start_date
            )
            signals = signal_service.to_signal_rows(portfolio, stock_prices)
            return jsonify(
                {
                    "status": "ok",
                    "signals": signals,
                    "date": latest_date.strftime("%Y-%m-%d"),
                }
            )
        except Exception as e:
            return jsonify(
                {
                    "status": "error",
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                }
            ), 500

    @app.route("/api/portfolio/rebalance", methods=["GET"])
    def get_rebalance():
        """Compare current holdings vs strategy targets and build rebalance plan."""
        try:
            start_date = market_service.default_start_date(days_back=365)
            latest_date, new_portfolio, stock_prices = (
                signal_service.get_latest_portfolio(start_date)
            )

            state = portfolio_repo.load()
            all_tickers = set(h["ticker"] for h in state["holdings"]) | set(
                new_portfolio.keys()
            )
            live_prices = market_service.get_live_prices(all_tickers)

            fallback_prices = {}
            for t in all_tickers:
                fallback_prices[t] = market_service.get_last_price_from_history(
                    stock_prices, t
                )

            plan = portfolio_service.build_rebalance_plan(
                new_portfolio, live_prices, fallback_prices=fallback_prices
            )
            plan["date"] = latest_date.strftime("%Y-%m-%d")
            return jsonify(plan)
        except Exception as e:
            return jsonify(
                {
                    "status": "error",
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                }
            ), 500

    @app.route("/api/portfolio/confirm-rebalance", methods=["POST"])
    def confirm_rebalance():
        """Apply sells/buys and persist updated portfolio state."""
        try:
            data = request.get_json() or {}
            sells = data.get("sells", [])
            buys = data.get("buys", [])
            return jsonify(portfolio_service.apply_rebalance(sells, buys))
        except Exception as e:
            return jsonify({"status": "error", "error": str(e)}), 500

    @app.route("/api/portfolio/history", methods=["GET"])
    def get_history():
        """Get full trade history."""
        try:
            return jsonify(portfolio_service.get_history())
        except Exception as e:
            return jsonify({"status": "error", "error": str(e)}), 500

    @app.route("/api/portfolio/reset", methods=["POST"])
    def reset_portfolio():
        """Reset local portfolio file."""
        try:
            portfolio_repo.reset()
            return jsonify({"status": "ok", "message": "Portfolio reset"})
        except Exception as e:
            return jsonify({"status": "error", "error": str(e)}), 500

    @app.route("/api/health", methods=["GET"])
    def health_check():
        return jsonify({"status": "healthy"})

    @app.route("/api/cache/info", methods=["GET"])
    def cache_info():
        try:
            info = cache_manager.get_cache_info()
            return jsonify({"cache": info, "total_files": len(info)})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/api/cache/clear", methods=["POST"])
    def clear_cache():
        try:
            cache_manager.clear_cache()
            return jsonify({"message": "Cache cleared successfully"})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/api/backtest", methods=["POST"])
    def run_backtest():
        """Run backtest with optional stop-loss config."""
        try:
            data = request.get_json() or {}
            return jsonify(backtest_service.run(data))
        except Exception as e:
            return jsonify(
                {
                    "status": "error",
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                }
            ), 500

    @app.route("/api/portfolio/current", methods=["GET"])
    def get_current_portfolio():
        return jsonify({"message": "Use /api/backtest to generate portfolio"})

    @app.route("/api/stock-prices", methods=["GET"])
    def get_stock_prices():
        try:
            tickers_str = request.args.get("tickers", "")
            if not tickers_str:
                return jsonify(
                    {"status": "error", "message": "No tickers provided"}
                ), 400

            tickers = [t.strip() for t in tickers_str.split(",") if t.strip()]
            prices = market_service.get_live_prices(tickers)
            return jsonify({"status": "success", "prices": prices})
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500

    @app.route("/")
    def home():
        """Serve static frontend."""
        html_path = os.path.join(root_dir, "index.html")
        if os.path.exists(html_path):
            return send_file(html_path)
        return jsonify(
            {
                "error": "index.html not found",
                "message": "Please ensure index.html is in the same directory as api.py",
            }
        ), 404

    return app


app = create_app()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("S&P 500 Momentum Strategy API Server")
    print("=" * 60)
    print("Starting server on http://localhost:5000")
    print("=" * 60 + "\n")
    app.run(host="0.0.0.0", port=5000, debug=True)
