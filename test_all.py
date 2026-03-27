"""Backward-compatible wrapper for tests.test_all."""

import runpy


if __name__ == "__main__":
    runpy.run_module("tests.test_all", run_name="__main__")
