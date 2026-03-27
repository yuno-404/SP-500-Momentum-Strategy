"""Backward-compatible wrapper for scripts.fmp_tools."""

import runpy


if __name__ == "__main__":
    runpy.run_module("scripts.fmp_tools", run_name="__main__")
