"""
AI Trading System

A comprehensive AI-powered trading system for achieving financial freedom
through automated investment strategies across multiple markets.
"""

__version__ = "1.0.0"
__author__ = "AI Trading Team"
__description__ = "AI-powered 24/7 automated trading system"

from .utils.config import get_config
from .utils.logger import get_logger
from .utils.exceptions import TradingSystemException

__all__ = [
    "get_config",
    "get_logger", 
    "TradingSystemException"
]