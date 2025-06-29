"""
Data Collectors Module for AI Trading System

This module provides unified API interfaces for multiple trading exchanges
with automatic failover, rate limiting, and real-time data streaming.
"""

from .base_api import TradingAPI
from .alpaca_api import AlpacaAPI
from .upbit_api import UpbitAPI
from .kis_api import KisAPI
from .api_manager import APIManager, api_manager
from .rate_limiter import RateLimiter, ExchangeRateLimiter, rate_limiter, rate_limit
from .models import (
    # Enums
    OrderSide, OrderType, OrderStatus, TimeFrame, AssetType, ExchangeType,
    
    # Data Models
    MarketData, TickData, Order, OrderRequest, Position, Balance,
    AccountInfo, Trade, Symbol, WebSocketMessage, APIResponse
)

__all__ = [
    # Base classes
    "TradingAPI",
    
    # Exchange implementations
    "AlpacaAPI",
    "UpbitAPI", 
    "KisAPI",
    
    # API Manager
    "APIManager",
    "api_manager",
    
    # Rate limiting
    "RateLimiter",
    "ExchangeRateLimiter",
    "rate_limiter",
    "rate_limit",
    
    # Enums
    "OrderSide",
    "OrderType", 
    "OrderStatus",
    "TimeFrame",
    "AssetType",
    "ExchangeType",
    
    # Data models
    "MarketData",
    "TickData",
    "Order",
    "OrderRequest",
    "Position",
    "Balance",
    "AccountInfo",
    "Trade",
    "Symbol",
    "WebSocketMessage",
    "APIResponse"
]

# Version info
__version__ = "1.0.0"
__description__ = "Multi-exchange trading API with unified interface"