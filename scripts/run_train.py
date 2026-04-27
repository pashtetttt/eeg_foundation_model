#!/usr/bin/env python3
"""Run default training workflow via package wrapper."""

from _bootstrap import *  # noqa: F401,F403
from eeg_thesis.cli import run_train


if __name__ == "__main__":
    run_train()
