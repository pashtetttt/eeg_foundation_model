#!/usr/bin/env python3
"""Run default feature-selection workflow via package wrapper."""

from _bootstrap import *  # noqa: F401,F403
from eeg_thesis.cli import run_feature_selection


if __name__ == "__main__":
    run_feature_selection()
