"""Backward-compatible wrapper for scripts.full_backtest."""

import runpy


if __name__ == "__main__":
    runpy.run_module("scripts.full_backtest", run_name="__main__")
