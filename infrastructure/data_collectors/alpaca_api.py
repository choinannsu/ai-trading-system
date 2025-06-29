"""
Alpaca API implementation for US stocks
"""

import json
import hmac
import hashlib
import base64
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Any
from urllib.parse import urlencode

import aiohttp
import websockets

from .base_api import TradingAPI
from .models import (
    MarketData, TickData, Order, OrderRequest, Position, Balance, 
    AccountInfo, Trade, Symbol, ExchangeType, WebSocketMessage,
    OrderSide, OrderType, OrderStatus, AssetType, TimeFrame
)
from .rate_limiter import rate_limit
from utils.logger import get_logger
from utils.exceptions import APIError, TradingError, ValidationError

logger = get_logger(__name__)


class AlpacaAPI(TradingAPI):
    """Alpaca API implementation"""
    
    def __init__(self, api_key: str, secret_key: str, environment: str = "paper"):
        super().__init__(api_key, secret_key, environment)
        self.session: Optional[aiohttp.ClientSession] = None
        
    @property
    def exchange_type(self) -> ExchangeType:
        return ExchangeType.ALPACA
    
    @property
    def base_url(self) -> str:
        if self.environment == "live":
            return "https://api.alpaca.markets"
        return "https://paper-api.alpaca.markets"
    
    @property
    def data_url(self) -> str:
        return "https://data.alpaca.markets"
    
    @property
    def websocket_url(self) -> str:
        if self.environment == "live":
            return "wss://stream.data.alpaca.markets/v2/iex"
        return "wss://stream.data.alpaca.markets/v2/iex"
    
    def _get_headers(self) -> Dict[str, str]:
        """Get request headers"""
        return {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.secret_key,
            "Content-Type": "application/json"
        }
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session"""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession(headers=self._get_headers())
        return self.session
    
    async def _make_request(self, method: str, endpoint: str, 
                          base_url: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """Make HTTP request with rate limiting"""
        await self.rate_limiter.acquire(self.exchange_type.value, endpoint)
        
        url = f"{base_url or self.base_url}/v2/{endpoint}"
        session = await self._get_session()
        
        try:
            async with session.request(method, url, **kwargs) as response:
                data = await response.json()
                
                if response.status >= 400:
                    error_message = data.get('message', f'HTTP {response.status}')
                    raise APIError(
                        f"Alpaca API error: {error_message}",
                        api_name="alpaca",
                        status_code=response.status,
                        response=str(data)
                    )
                
                return data
                
        except aiohttp.ClientError as e:
            raise APIError(f"HTTP request failed: {str(e)}", api_name="alpaca")
    
    # Market Data Methods
    @rate_limit("alpaca", "bars")
    async def get_market_data(self, symbol: str, timeframe: str = "1m", 
                            limit: int = 100) -> List[MarketData]:
        """Get historical market data"""
        try:
            # Convert timeframe to Alpaca format
            alpaca_timeframe = self._convert_timeframe(timeframe)
            
            # Get date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)  # Default to 30 days
            
            params = {
                "symbols": symbol,
                "timeframe": alpaca_timeframe,
                "start": start_date.isoformat(),
                "end": end_date.isoformat(),
                "limit": limit,
                "adjustment": "raw"
            }
            
            data = await self._make_request(
                "GET", 
                f"stocks/{symbol}/bars",
                base_url=self.data_url,
                params=params
            )
            
            bars = data.get("bars", {}).get(symbol, [])
            
            market_data = []
            for bar in bars:
                market_data.append(MarketData(
                    symbol=symbol,
                    exchange=self.exchange_type,
                    timestamp=datetime.fromisoformat(bar["t"].replace("Z", "+00:00")),
                    open=Decimal(str(bar["o"])),
                    high=Decimal(str(bar["h"])),
                    low=Decimal(str(bar["l"])),
                    close=Decimal(str(bar["c"])),
                    volume=Decimal(str(bar["v"])),
                    vwap=Decimal(str(bar.get("vw", 0))),
                    trade_count=bar.get("n", 0),
                    timeframe=TimeFrame(timeframe)
                ))
            
            return market_data
            
        except Exception as e:
            logger.error(f"Error getting market data for {symbol}: {str(e)}")
            raise APIError(f"Failed to get market data: {str(e)}", api_name="alpaca")
    
    @rate_limit("alpaca", "latest_trade")
    async def get_current_price(self, symbol: str) -> TickData:
        """Get current price for symbol"""
        try:
            data = await self._make_request(
                "GET",
                f"stocks/{symbol}/trades/latest",
                base_url=self.data_url
            )
            
            trade = data.get("trade", {})
            
            return TickData(
                symbol=symbol,
                exchange=self.exchange_type,
                timestamp=datetime.fromisoformat(trade["t"].replace("Z", "+00:00")),
                price=Decimal(str(trade["p"])),
                size=Decimal(str(trade["s"]))
            )
            
        except Exception as e:
            logger.error(f"Error getting current price for {symbol}: {str(e)}")
            raise APIError(f"Failed to get current price: {str(e)}", api_name="alpaca")
    
    @rate_limit("alpaca", "assets")
    async def get_symbols(self) -> List[Symbol]:
        """Get available trading symbols"""
        try:
            data = await self._make_request("GET", "assets")
            
            symbols = []
            for asset in data:
                if asset["tradable"] and asset["status"] == "active":
                    symbols.append(Symbol(
                        symbol=asset["symbol"],
                        name=asset["name"],
                        exchange=self.exchange_type,
                        asset_type=AssetType.STOCK,
                        min_quantity=Decimal("1"),
                        quantity_increment=Decimal("1"),
                        price_increment=Decimal("0.01"),
                        is_tradable=asset["tradable"],
                        is_marginable=asset.get("marginable", False),
                        market_open="09:30",
                        market_close="16:00",
                        timezone="America/New_York"
                    ))
            
            return symbols
            
        except Exception as e:
            logger.error(f"Error getting symbols: {str(e)}")
            raise APIError(f"Failed to get symbols: {str(e)}", api_name="alpaca")
    
    # Trading Methods
    @rate_limit("alpaca", "orders")
    async def place_order(self, order_request: OrderRequest) -> Order:
        """Place a trading order"""
        try:
            # Convert to Alpaca order format
            alpaca_order = {
                "symbol": order_request.symbol,
                "side": order_request.side.value,
                "type": self._convert_order_type(order_request.type),
                "qty": str(order_request.quantity),
                "time_in_force": order_request.time_in_force
            }
            
            if order_request.price:
                alpaca_order["limit_price"] = str(order_request.price)
            
            if order_request.stop_price:
                alpaca_order["stop_price"] = str(order_request.stop_price)
            
            if order_request.client_order_id:
                alpaca_order["client_order_id"] = order_request.client_order_id
            
            data = await self._make_request("POST", "orders", json=alpaca_order)
            
            return self._parse_order(data)
            
        except Exception as e:
            logger.error(f"Error placing order: {str(e)}")
            raise TradingError(f"Failed to place order: {str(e)}", 
                             symbol=order_request.symbol)
    
    @rate_limit("alpaca", "orders")
    async def cancel_order(self, order_id: str) -> Order:
        """Cancel an existing order"""
        try:
            data = await self._make_request("DELETE", f"orders/{order_id}")
            return self._parse_order(data)
            
        except Exception as e:
            logger.error(f"Error cancelling order {order_id}: {str(e)}")
            raise TradingError(f"Failed to cancel order: {str(e)}")
    
    @rate_limit("alpaca", "orders")
    async def get_order(self, order_id: str) -> Order:
        """Get order details"""
        try:
            data = await self._make_request("GET", f"orders/{order_id}")
            return self._parse_order(data)
            
        except Exception as e:
            logger.error(f"Error getting order {order_id}: {str(e)}")
            raise APIError(f"Failed to get order: {str(e)}", api_name="alpaca")
    
    @rate_limit("alpaca", "orders")
    async def get_orders(self, symbol: Optional[str] = None, 
                        status: Optional[str] = None) -> List[Order]:
        """Get orders list"""
        try:
            params = {}
            if symbol:
                params["symbols"] = symbol
            if status:
                params["status"] = status
            
            data = await self._make_request("GET", "orders", params=params)
            
            return [self._parse_order(order_data) for order_data in data]
            
        except Exception as e:
            logger.error(f"Error getting orders: {str(e)}")
            raise APIError(f"Failed to get orders: {str(e)}", api_name="alpaca")
    
    # Account Methods
    @rate_limit("alpaca", "account")
    async def get_balance(self) -> List[Balance]:
        """Get account balance"""
        try:
            data = await self._make_request("GET", "account")
            
            return [Balance(
                exchange=self.exchange_type,
                currency="USD",
                total=Decimal(data["equity"]),
                available=Decimal(data["buying_power"]),
                locked=Decimal(data["equity"]) - Decimal(data["buying_power"]),
                usd_value=Decimal(data["equity"]),
                updated_at=datetime.now()
            )]
            
        except Exception as e:
            logger.error(f"Error getting balance: {str(e)}")
            raise APIError(f"Failed to get balance: {str(e)}", api_name="alpaca")
    
    @rate_limit("alpaca", "positions")
    async def get_positions(self) -> List[Position]:
        """Get current positions"""
        try:
            data = await self._make_request("GET", "positions")
            
            positions = []
            for pos_data in data:
                positions.append(Position(
                    symbol=pos_data["symbol"],
                    exchange=self.exchange_type,
                    asset_type=AssetType.STOCK,
                    quantity=Decimal(pos_data["qty"]),
                    avg_price=Decimal(pos_data["avg_entry_price"]),
                    market_value=Decimal(pos_data["market_value"]),
                    unrealized_pnl=Decimal(pos_data["unrealized_pl"]),
                    realized_pnl=Decimal(pos_data.get("realized_pl", "0")),
                    opened_at=datetime.now(),  # Alpaca doesn't provide this
                    updated_at=datetime.now()
                ))
            
            return positions
            
        except Exception as e:
            logger.error(f"Error getting positions: {str(e)}")
            raise APIError(f"Failed to get positions: {str(e)}", api_name="alpaca")
    
    @rate_limit("alpaca", "account")
    async def get_account_info(self) -> AccountInfo:
        """Get account information"""
        try:
            data = await self._make_request("GET", "account")
            
            balances = await self.get_balance()
            
            return AccountInfo(
                exchange=self.exchange_type,
                account_id=data["id"],
                is_active=data["status"] == "ACTIVE",
                trading_enabled=data["trading_blocked"] == False,
                balances=balances,
                buying_power=Decimal(data["buying_power"]),
                updated_at=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error getting account info: {str(e)}")
            raise APIError(f"Failed to get account info: {str(e)}", api_name="alpaca")
    
    @rate_limit("alpaca", "orders")
    async def get_trades(self, symbol: Optional[str] = None, 
                        limit: int = 100) -> List[Trade]:
        """Get trade history"""
        try:
            # Get filled orders (Alpaca doesn't have separate trades endpoint)
            params = {"status": "filled", "limit": limit}
            if symbol:
                params["symbols"] = symbol
            
            orders_data = await self._make_request("GET", "orders", params=params)
            
            trades = []
            for order_data in orders_data:
                if order_data["status"] == "filled":
                    trades.append(Trade(
                        id=order_data["id"],
                        order_id=order_data["id"],
                        symbol=order_data["symbol"],
                        exchange=self.exchange_type,
                        side=OrderSide(order_data["side"]),
                        quantity=Decimal(order_data["filled_qty"]),
                        price=Decimal(order_data["filled_avg_price"] or "0"),
                        timestamp=datetime.fromisoformat(
                            order_data["filled_at"].replace("Z", "+00:00")
                        ) if order_data["filled_at"] else datetime.now()
                    ))
            
            return trades
            
        except Exception as e:
            logger.error(f"Error getting trades: {str(e)}")
            raise APIError(f"Failed to get trades: {str(e)}", api_name="alpaca")
    
    # WebSocket Methods
    async def _authenticate_websocket(self) -> Dict[str, Any]:
        """Create WebSocket authentication message"""
        return {
            "action": "auth",
            "key": self.api_key,
            "secret": self.secret_key
        }
    
    async def _create_subscription_message(self, channel: str, 
                                         symbol: str) -> Dict[str, Any]:
        """Create subscription message for WebSocket"""
        return {
            "action": "subscribe",
            "trades": [symbol] if channel == "trade" else [],
            "quotes": [symbol] if channel == "quote" else [],
            "bars": [symbol] if channel == "bar" else []
        }
    
    async def _parse_websocket_message(self, message: Dict[str, Any]) -> Optional[WebSocketMessage]:
        """Parse incoming WebSocket message"""
        try:
            for msg in message:
                msg_type = msg.get("T")
                
                if msg_type == "t":  # Trade
                    return WebSocketMessage(
                        channel="trade",
                        exchange=self.exchange_type,
                        timestamp=datetime.fromisoformat(msg["t"].replace("Z", "+00:00")),
                        message_type="tick_data",
                        symbol=msg["S"],
                        data={
                            "symbol": msg["S"],
                            "exchange": self.exchange_type,
                            "timestamp": datetime.fromisoformat(msg["t"].replace("Z", "+00:00")),
                            "price": Decimal(str(msg["p"])),
                            "size": Decimal(str(msg["s"]))
                        }
                    )
                
                elif msg_type == "q":  # Quote
                    return WebSocketMessage(
                        channel="quote",
                        exchange=self.exchange_type,
                        timestamp=datetime.fromisoformat(msg["t"].replace("Z", "+00:00")),
                        message_type="tick_data",
                        symbol=msg["S"],
                        data={
                            "symbol": msg["S"],
                            "exchange": self.exchange_type,
                            "timestamp": datetime.fromisoformat(msg["t"].replace("Z", "+00:00")),
                            "price": Decimal(str(msg["ap"])),  # Ask price
                            "size": Decimal(str(msg["as"])),   # Ask size
                            "bid": Decimal(str(msg["bp"])),    # Bid price
                            "ask": Decimal(str(msg["ap"])),    # Ask price
                            "bid_size": Decimal(str(msg["bs"])), # Bid size
                            "ask_size": Decimal(str(msg["as"]))  # Ask size
                        }
                    )
                
                elif msg_type == "b":  # Bar
                    return WebSocketMessage(
                        channel="bar",
                        exchange=self.exchange_type,
                        timestamp=datetime.fromisoformat(msg["t"].replace("Z", "+00:00")),
                        message_type="market_data",
                        symbol=msg["S"],
                        data={
                            "symbol": msg["S"],
                            "exchange": self.exchange_type,
                            "timestamp": datetime.fromisoformat(msg["t"].replace("Z", "+00:00")),
                            "open": Decimal(str(msg["o"])),
                            "high": Decimal(str(msg["h"])),
                            "low": Decimal(str(msg["l"])),
                            "close": Decimal(str(msg["c"])),
                            "volume": Decimal(str(msg["v"])),
                            "vwap": Decimal(str(msg.get("vw", 0))),
                            "trade_count": msg.get("n", 0),
                            "timeframe": TimeFrame.MIN_1
                        }
                    )
            
            return None
            
        except Exception as e:
            logger.error(f"Error parsing WebSocket message: {str(e)}")
            return None
    
    # Helper Methods
    def _convert_timeframe(self, timeframe: str) -> str:
        """Convert timeframe to Alpaca format"""
        mapping = {
            "1m": "1Min",
            "5m": "5Min",
            "15m": "15Min",
            "30m": "30Min",
            "1h": "1Hour",
            "4h": "4Hour",
            "1d": "1Day",
            "1w": "1Week",
            "1M": "1Month"
        }
        return mapping.get(timeframe, "1Min")
    
    def _convert_order_type(self, order_type: OrderType) -> str:
        """Convert order type to Alpaca format"""
        mapping = {
            OrderType.MARKET: "market",
            OrderType.LIMIT: "limit",
            OrderType.STOP: "stop",
            OrderType.STOP_LIMIT: "stop_limit"
        }
        return mapping.get(order_type, "market")
    
    def _parse_order(self, order_data: Dict[str, Any]) -> Order:
        """Parse Alpaca order data to Order model"""
        return Order(
            id=order_data["id"],
            client_order_id=order_data.get("client_order_id"),
            symbol=order_data["symbol"],
            exchange=self.exchange_type,
            side=OrderSide(order_data["side"]),
            type=self._parse_order_type(order_data["type"]),
            status=self._parse_order_status(order_data["status"]),
            quantity=Decimal(order_data["qty"]),
            filled_quantity=Decimal(order_data["filled_qty"]),
            remaining_quantity=Decimal(order_data["qty"]) - Decimal(order_data["filled_qty"]),
            price=Decimal(order_data["limit_price"]) if order_data.get("limit_price") else None,
            avg_fill_price=Decimal(order_data["filled_avg_price"]) if order_data.get("filled_avg_price") else None,
            stop_price=Decimal(order_data["stop_price"]) if order_data.get("stop_price") else None,
            created_at=datetime.fromisoformat(order_data["created_at"].replace("Z", "+00:00")),
            updated_at=datetime.fromisoformat(order_data["updated_at"].replace("Z", "+00:00")) if order_data.get("updated_at") else None,
            filled_at=datetime.fromisoformat(order_data["filled_at"].replace("Z", "+00:00")) if order_data.get("filled_at") else None,
            time_in_force=order_data["time_in_force"]
        )
    
    def _parse_order_type(self, alpaca_type: str) -> OrderType:
        """Parse Alpaca order type"""
        mapping = {
            "market": OrderType.MARKET,
            "limit": OrderType.LIMIT,
            "stop": OrderType.STOP,
            "stop_limit": OrderType.STOP_LIMIT
        }
        return mapping.get(alpaca_type, OrderType.MARKET)
    
    def _parse_order_status(self, alpaca_status: str) -> OrderStatus:
        """Parse Alpaca order status"""
        mapping = {
            "new": OrderStatus.PENDING,
            "partially_filled": OrderStatus.PARTIALLY_FILLED,
            "filled": OrderStatus.FILLED,
            "done_for_day": OrderStatus.CANCELLED,
            "canceled": OrderStatus.CANCELLED,
            "expired": OrderStatus.CANCELLED,
            "replaced": OrderStatus.CANCELLED,
            "pending_cancel": OrderStatus.PENDING,
            "pending_replace": OrderStatus.PENDING,
            "accepted": OrderStatus.PENDING,
            "pending_new": OrderStatus.PENDING,
            "accepted_for_bidding": OrderStatus.PENDING,
            "stopped": OrderStatus.CANCELLED,
            "rejected": OrderStatus.REJECTED,
            "suspended": OrderStatus.CANCELLED,
            "calculated": OrderStatus.PENDING
        }
        return mapping.get(alpaca_status, OrderStatus.PENDING)
    
    async def close(self) -> None:
        """Close HTTP session"""
        if self.session and not self.session.closed:
            await self.session.close()
        await self.disconnect()