"""Backward-compatible wrapper for scripts.build_sector_aum_csv."""

import runpy


if __name__ == "__main__":
    runpy.run_module("scripts.build_sector_aum_csv", run_name="__main__")
