"""
Data models for unified trading API interface
"""

from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Union

from pydantic import BaseModel, Field, validator


class OrderSide(str, Enum):
    """Order side enumeration"""
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    """Order type enumeration"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderStatus(str, Enum):
    """Order status enumeration"""
    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class TimeFrame(str, Enum):
    """Time frame enumeration"""
    MIN_1 = "1m"
    MIN_5 = "5m"
    MIN_15 = "15m"
    MIN_30 = "30m"
    HOUR_1 = "1h"
    HOUR_4 = "4h"
    DAY_1 = "1d"
    WEEK_1 = "1w"
    MONTH_1 = "1M"


class AssetType(str, Enum):
    """Asset type enumeration"""
    STOCK = "stock"
    CRYPTO = "crypto"
    ETF = "etf"
    OPTION = "option"
    FUTURE = "future"


class ExchangeType(str, Enum):
    """Exchange type enumeration"""
    ALPACA = "alpaca"
    BINANCE = "binance"
    KIWOOM = "kiwoom"
    UPBIT = "upbit"
    KIS = "kis"


class MarketData(BaseModel):
    """Unified market data model"""
    
    symbol: str = Field(..., description="Trading symbol")
    exchange: ExchangeType = Field(..., description="Exchange identifier")
    timestamp: datetime = Field(..., description="Data timestamp")
    
    # OHLCV data
    open: Decimal = Field(..., description="Open price")
    high: Decimal = Field(..., description="High price")
    low: Decimal = Field(..., description="Low price")
    close: Decimal = Field(..., description="Close price")
    volume: Decimal = Field(..., description="Volume")
    
    # Additional fields
    vwap: Optional[Decimal] = Field(None, description="Volume weighted average price")
    trade_count: Optional[int] = Field(None, description="Number of trades")
    timeframe: TimeFrame = Field(default=TimeFrame.MIN_1, description="Data timeframe")
    
    @validator('timestamp')
    def validate_timestamp(cls, v):
        if v > datetime.now():
            raise ValueError('Timestamp cannot be in the future')
        return v


class TickData(BaseModel):
    """Real-time tick data model"""
    
    symbol: str = Field(..., description="Trading symbol")
    exchange: ExchangeType = Field(..., description="Exchange identifier")
    timestamp: datetime = Field(..., description="Tick timestamp")
    
    # Tick data
    price: Decimal = Field(..., description="Last trade price")
    size: Decimal = Field(..., description="Trade size")
    bid: Optional[Decimal] = Field(None, description="Best bid price")
    ask: Optional[Decimal] = Field(None, description="Best ask price")
    bid_size: Optional[Decimal] = Field(None, description="Best bid size")
    ask_size: Optional[Decimal] = Field(None, description="Best ask size")


class OrderRequest(BaseModel):
    """Order request model"""
    
    symbol: str = Field(..., description="Trading symbol")
    side: OrderSide = Field(..., description="Order side")
    type: OrderType = Field(..., description="Order type")
    quantity: Decimal = Field(..., description="Order quantity")
    price: Optional[Decimal] = Field(None, description="Order price (for limit orders)")
    stop_price: Optional[Decimal] = Field(None, description="Stop price (for stop orders)")
    time_in_force: str = Field(default="GTC", description="Time in force")
    client_order_id: Optional[str] = Field(None, description="Client order ID")
    
    @validator('quantity')
    def validate_quantity(cls, v):
        if v <= 0:
            raise ValueError('Quantity must be positive')
        return v
    
    @validator('price')
    def validate_price(cls, v):
        if v is not None and v <= 0:
            raise ValueError('Price must be positive')
        return v


class Order(BaseModel):
    """Order model"""
    
    id: str = Field(..., description="Order ID")
    client_order_id: Optional[str] = Field(None, description="Client order ID")
    symbol: str = Field(..., description="Trading symbol")
    exchange: ExchangeType = Field(..., description="Exchange identifier")
    
    # Order details
    side: OrderSide = Field(..., description="Order side")
    type: OrderType = Field(..., description="Order type")
    status: OrderStatus = Field(..., description="Order status")
    
    # Quantities and prices
    quantity: Decimal = Field(..., description="Order quantity")
    filled_quantity: Decimal = Field(default=Decimal('0'), description="Filled quantity")
    remaining_quantity: Decimal = Field(..., description="Remaining quantity")
    
    price: Optional[Decimal] = Field(None, description="Order price")
    avg_fill_price: Optional[Decimal] = Field(None, description="Average fill price")
    stop_price: Optional[Decimal] = Field(None, description="Stop price")
    
    # Timestamps
    created_at: datetime = Field(..., description="Order creation time")
    updated_at: Optional[datetime] = Field(None, description="Last update time")
    filled_at: Optional[datetime] = Field(None, description="Fill time")
    
    # Additional fields
    time_in_force: str = Field(default="GTC", description="Time in force")
    commission: Optional[Decimal] = Field(None, description="Commission paid")
    
    @property
    def is_filled(self) -> bool:
        return self.status == OrderStatus.FILLED
    
    @property
    def fill_percentage(self) -> float:
        return float(self.filled_quantity / self.quantity * 100)


class Position(BaseModel):
    """Position model"""
    
    symbol: str = Field(..., description="Trading symbol")
    exchange: ExchangeType = Field(..., description="Exchange identifier")
    asset_type: AssetType = Field(..., description="Asset type")
    
    # Position details
    quantity: Decimal = Field(..., description="Position quantity (+ for long, - for short)")
    avg_price: Decimal = Field(..., description="Average entry price")
    market_value: Decimal = Field(..., description="Current market value")
    unrealized_pnl: Decimal = Field(..., description="Unrealized P&L")
    realized_pnl: Decimal = Field(default=Decimal('0'), description="Realized P&L")
    
    # Timestamps
    opened_at: datetime = Field(..., description="Position open time")
    updated_at: datetime = Field(..., description="Last update time")
    
    @property
    def is_long(self) -> bool:
        return self.quantity > 0
    
    @property
    def is_short(self) -> bool:
        return self.quantity < 0
    
    @property
    def pnl_percentage(self) -> float:
        if self.avg_price == 0:
            return 0.0
        return float(self.unrealized_pnl / (abs(self.quantity) * self.avg_price) * 100)


class Balance(BaseModel):
    """Account balance model"""
    
    exchange: ExchangeType = Field(..., description="Exchange identifier")
    currency: str = Field(..., description="Currency code")
    
    # Balance details
    total: Decimal = Field(..., description="Total balance")
    available: Decimal = Field(..., description="Available balance")
    locked: Decimal = Field(default=Decimal('0'), description="Locked balance")
    
    # USD equivalent
    usd_value: Optional[Decimal] = Field(None, description="USD equivalent value")
    
    updated_at: datetime = Field(..., description="Last update time")


class AccountInfo(BaseModel):
    """Account information model"""
    
    exchange: ExchangeType = Field(..., description="Exchange identifier")
    account_id: str = Field(..., description="Account ID")
    
    # Account status
    is_active: bool = Field(..., description="Account active status")
    trading_enabled: bool = Field(..., description="Trading enabled status")
    
    # Balances
    balances: List[Balance] = Field(default_factory=list, description="Account balances")
    
    # Risk information
    buying_power: Optional[Decimal] = Field(None, description="Available buying power")
    margin_used: Optional[Decimal] = Field(None, description="Used margin")
    margin_available: Optional[Decimal] = Field(None, description="Available margin")
    
    updated_at: datetime = Field(..., description="Last update time")


class Trade(BaseModel):
    """Trade execution model"""
    
    id: str = Field(..., description="Trade ID")
    order_id: str = Field(..., description="Related order ID")
    symbol: str = Field(..., description="Trading symbol")
    exchange: ExchangeType = Field(..., description="Exchange identifier")
    
    # Trade details
    side: OrderSide = Field(..., description="Trade side")
    quantity: Decimal = Field(..., description="Trade quantity")
    price: Decimal = Field(..., description="Trade price")
    commission: Decimal = Field(default=Decimal('0'), description="Commission paid")
    
    timestamp: datetime = Field(..., description="Trade timestamp")


class Symbol(BaseModel):
    """Trading symbol information"""
    
    symbol: str = Field(..., description="Symbol code")
    name: str = Field(..., description="Symbol name")
    exchange: ExchangeType = Field(..., description="Exchange identifier")
    asset_type: AssetType = Field(..., description="Asset type")
    
    # Trading specifications
    min_quantity: Decimal = Field(..., description="Minimum order quantity")
    max_quantity: Optional[Decimal] = Field(None, description="Maximum order quantity")
    quantity_increment: Decimal = Field(..., description="Quantity increment")
    
    min_price: Optional[Decimal] = Field(None, description="Minimum price")
    max_price: Optional[Decimal] = Field(None, description="Maximum price")
    price_increment: Decimal = Field(..., description="Price increment (tick size)")
    
    # Status
    is_tradable: bool = Field(default=True, description="Symbol is tradable")
    is_marginable: bool = Field(default=False, description="Symbol is marginable")
    
    # Market hours
    market_open: Optional[str] = Field(None, description="Market open time")
    market_close: Optional[str] = Field(None, description="Market close time")
    timezone: str = Field(default="UTC", description="Market timezone")


class WebSocketMessage(BaseModel):
    """WebSocket message model"""
    
    channel: str = Field(..., description="Message channel")
    exchange: ExchangeType = Field(..., description="Exchange identifier")
    timestamp: datetime = Field(..., description="Message timestamp")
    data: Dict = Field(..., description="Message data")
    
    # Message metadata
    message_type: str = Field(..., description="Message type")
    symbol: Optional[str] = Field(None, description="Symbol (if applicable)")


class APIResponse(BaseModel):
    """Generic API response model"""
    
    success: bool = Field(..., description="Request success status")
    data: Optional[Union[Dict, List]] = Field(None, description="Response data")
    error: Optional[str] = Field(None, description="Error message")
    error_code: Optional[str] = Field(None, description="Error code")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")
    
    # Rate limiting info
    rate_limit_remaining: Optional[int] = Field(None, description="Remaining rate limit")
    rate_limit_reset: Optional[datetime] = Field(None, description="Rate limit reset time")