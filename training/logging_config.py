"""
Centralized logging configuration for the Pearls AQI Predictor project.

Ensures consistent log formatting and level across all entry-point scripts.
Call `setup_logging()` at the very beginning of each main script.
"""
from __future__ import annotations

import logging
import sys


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logger for console output with a standard format."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)-7s] [%(name)-25s] %(message)s",
        stream=sys.stdout,
    )