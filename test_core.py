"""Backward-compatible wrapper for tests.test_core."""

import runpy


if __name__ == "__main__":
    runpy.run_module("tests.test_core", run_name="__main__")
