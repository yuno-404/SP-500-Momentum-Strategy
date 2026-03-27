"""Backward-compatible wrapper for scripts.sensitivity_backtest."""

import runpy


if __name__ == "__main__":
    runpy.run_module("scripts.sensitivity_backtest", run_name="__main__")
