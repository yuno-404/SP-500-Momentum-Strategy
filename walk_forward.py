"""Backward-compatible wrapper for scripts.walk_forward."""

import runpy


if __name__ == "__main__":
    runpy.run_module("scripts.walk_forward", run_name="__main__")
