"""Backward-compatible wrapper for scripts.today_buy."""

import runpy


if __name__ == "__main__":
    runpy.run_module("scripts.today_buy", run_name="__main__")
