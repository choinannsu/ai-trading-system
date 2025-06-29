"""
Upbit API implementation for Korean cryptocurrency
"""

import json
import hmac
import hashlib
import uuid
import time
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Any
from urllib.parse import urlencode, urlparse

import aiohttp
import websockets
import jwt

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


class UpbitAPI(TradingAPI):
    """Upbit API implementation for Korean cryptocurrency"""
    
    def __init__(self, api_key: str, secret_key: str, environment: str = "live"):
        super().__init__(api_key, secret_key, environment)
        self.session: Optional[aiohttp.ClientSession] = None
        
    @property
    def exchange_type(self) -> ExchangeType:
        return ExchangeType.UPBIT
    
    @property
    def base_url(self) -> str:
        return "https://api.upbit.com"
    
    @property
    def websocket_url(self) -> str:
        return "wss://api.upbit.com/websocket/v1"
    
    def _create_jwt_token(self, query_params: Optional[Dict] = None) -> str:
        """Create JWT token for authentication"""
        payload = {
            'access_key': self.api_key,
            'nonce': str(uuid.uuid4())
        }
        
        if query_params:
            query_string = urlencode(query_params)
            m = hashlib.sha512()
            m.update(query_string.encode())
            query_hash = m.hexdigest()
            
            payload['query_hash'] = query_hash
            payload['query_hash_alg'] = 'SHA512'
        
        return jwt.encode(payload, self.secret_key, algorithm='HS256')
    
    def _get_headers(self, query_params: Optional[Dict] = None) -> Dict[str, str]:
        """Get request headers with JWT authentication"""
        headers = {
            "Content-Type": "application/json"
        }
        
        if self.api_key and self.secret_key:
            jwt_token = self._create_jwt_token(query_params)
            headers["Authorization"] = f"Bearer {jwt_token}"
        
        return headers
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session"""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()
        return self.session
    
    async def _make_request(self, method: str, endpoint: str, 
                          params: Optional[Dict] = None, **kwargs) -> Dict[str, Any]:
        """Make HTTP request with rate limiting"""
        await self.rate_limiter.acquire(self.exchange_type.value, endpoint)
        
        url = f"{self.base_url}/v1/{endpoint}"
        
        # For authenticated requests, add JWT token
        headers = self._get_headers(params if method == "GET" else None)
        kwargs.setdefault('headers', {}).update(headers)
        
        session = await self._get_session()
        
        try:
            if method == "GET" and params:
                async with session.get(url, params=params, **kwargs) as response:
                    return await self._handle_response(response)
            else:
                async with session.request(method, url, **kwargs) as response:
                    return await self._handle_response(response)
                    
        except aiohttp.ClientError as e:
            raise APIError(f"HTTP request failed: {str(e)}", api_name="upbit")
    
    async def _handle_response(self, response: aiohttp.ClientResponse) -> Dict[str, Any]:
        """Handle HTTP response"""
        data = await response.json()
        
        if response.status >= 400:
            error_message = data.get('error', {}).get('message', f'HTTP {response.status}')
            error_name = data.get('error', {}).get('name', 'UNKNOWN_ERROR')
            
            raise APIError(
                f"Upbit API error: {error_message}",
                api_name="upbit",
                status_code=response.status,
                response=str(data),
                error_code=error_name
            )
        
        return data
    
    # Market Data Methods
    @rate_limit("upbit", "candles")
    async def get_market_data(self, symbol: str, timeframe: str = "1m", 
                            limit: int = 100) -> List[MarketData]:
        """Get historical market data"""
        try:
            upbit_symbol = self._format_symbol(symbol)
            upbit_timeframe = self._convert_timeframe(timeframe)
            
            endpoint = f"candles/{upbit_timeframe}"
            params = {
                "market": upbit_symbol,
                "count": min(limit, 200)  # Upbit max is 200
            }
            
            data = await self._make_request("GET", endpoint, params=params)
            
            market_data = []
            for candle in data:
                market_data.append(MarketData(
                    symbol=symbol,
                    exchange=self.exchange_type,
                    timestamp=datetime.fromisoformat(candle["candle_date_time_kst"].replace("T", " ")),
                    open=Decimal(str(candle["opening_price"])),
                    high=Decimal(str(candle["high_price"])),
                    low=Decimal(str(candle["low_price"])),
                    close=Decimal(str(candle["trade_price"])),
                    volume=Decimal(str(candle["candle_acc_trade_volume"])),
                    timeframe=TimeFrame(timeframe)
                ))
            
            # Upbit returns data in descending order, reverse to get ascending
            return list(reversed(market_data))
            
        except Exception as e:
            logger.error(f"Error getting market data for {symbol}: {str(e)}")
            raise APIError(f"Failed to get market data: {str(e)}", api_name="upbit")
    
    @rate_limit("upbit", "ticker")
    async def get_current_price(self, symbol: str) -> TickData:
        """Get current price for symbol"""
        try:
            upbit_symbol = self._format_symbol(symbol)
            
            params = {"markets": upbit_symbol}
            data = await self._make_request("GET", "ticker", params=params)
            
            if not data or len(data) == 0:
                raise APIError(f"No data returned for {symbol}")
            
            ticker = data[0]
            
            return TickData(
                symbol=symbol,
                exchange=self.exchange_type,
                timestamp=datetime.fromtimestamp(ticker["timestamp"] / 1000),
                price=Decimal(str(ticker["trade_price"])),
                size=Decimal(str(ticker["trade_volume"])),
                bid=Decimal(str(ticker.get("highest_52_week_price", 0))),  # Approximate
                ask=Decimal(str(ticker.get("lowest_52_week_price", 0)))    # Approximate
            )
            
        except Exception as e:
            logger.error(f"Error getting current price for {symbol}: {str(e)}")
            raise APIError(f"Failed to get current price: {str(e)}", api_name="upbit")
    
    @rate_limit("upbit", "market/all")
    async def get_symbols(self) -> List[Symbol]:
        """Get available trading symbols"""
        try:
            data = await self._make_request("GET", "market/all")
            
            symbols = []
            for market_info in data:
                if market_info["warning"] is None:  # Skip warned markets
                    base_currency, quote_currency = market_info["market"].split("-")
                    
                    symbols.append(Symbol(
                        symbol=market_info["market"],
                        name=f"{quote_currency}/{base_currency}",
                        exchange=self.exchange_type,
                        asset_type=AssetType.CRYPTO,
                        min_quantity=Decimal("0.00000001"),  # Standard crypto precision
                        quantity_increment=Decimal("0.00000001"),
                        price_increment=Decimal("1") if base_currency == "KRW" else Decimal("0.00000001"),
                        is_tradable=True,
                        timezone="Asia/Seoul"
                    ))
            
            return symbols
            
        except Exception as e:
            logger.error(f"Error getting symbols: {str(e)}")
            raise APIError(f"Failed to get symbols: {str(e)}", api_name="upbit")
    
    # Trading Methods
    @rate_limit("upbit", "orders")
    async def place_order(self, order_request: OrderRequest) -> Order:
        """Place a trading order"""
        try:
            upbit_symbol = self._format_symbol(order_request.symbol)
            
            order_data = {
                "market": upbit_symbol,
                "side": order_request.side.value,
                "ord_type": self._convert_order_type(order_request.type)
            }
            
            if order_request.type == OrderType.MARKET:
                if order_request.side == OrderSide.BUY:
                    # Market buy order requires price (KRW amount)
                    order_data["price"] = str(order_request.quantity * (order_request.price or 1))
                else:
                    # Market sell order requires volume
                    order_data["volume"] = str(order_request.quantity)
            else:
                # Limit order
                order_data["volume"] = str(order_request.quantity)
                order_data["price"] = str(order_request.price)
            
            if order_request.client_order_id:
                order_data["identifier"] = order_request.client_order_id
            
            data = await self._make_request("POST", "orders", json=order_data)
            
            return self._parse_order(data)
            
        except Exception as e:
            logger.error(f"Error placing order: {str(e)}")
            raise TradingError(f"Failed to place order: {str(e)}", 
                             symbol=order_request.symbol)
    
    @rate_limit("upbit", "order")
    async def cancel_order(self, order_id: str) -> Order:
        """Cancel an existing order"""
        try:
            params = {"uuid": order_id}
            data = await self._make_request("DELETE", "order", params=params)
            return self._parse_order(data)
            
        except Exception as e:
            logger.error(f"Error cancelling order {order_id}: {str(e)}")
            raise TradingError(f"Failed to cancel order: {str(e)}")
    
    @rate_limit("upbit", "order")
    async def get_order(self, order_id: str) -> Order:
        """Get order details"""
        try:
            params = {"uuid": order_id}
            data = await self._make_request("GET", "order", params=params)
            return self._parse_order(data)
            
        except Exception as e:
            logger.error(f"Error getting order {order_id}: {str(e)}")
            raise APIError(f"Failed to get order: {str(e)}", api_name="upbit")
    
    @rate_limit("upbit", "orders")
    async def get_orders(self, symbol: Optional[str] = None, 
                        status: Optional[str] = None) -> List[Order]:
        """Get orders list"""
        try:
            params = {}
            if symbol:
                params["market"] = self._format_symbol(symbol)
            if status:
                params["state"] = self._convert_order_status_to_upbit(status)
            
            data = await self._make_request("GET", "orders", params=params)
            
            return [self._parse_order(order_data) for order_data in data]
            
        except Exception as e:
            logger.error(f"Error getting orders: {str(e)}")
            raise APIError(f"Failed to get orders: {str(e)}", api_name="upbit")
    
    # Account Methods
    @rate_limit("upbit", "accounts")
    async def get_balance(self) -> List[Balance]:
        """Get account balance"""
        try:
            data = await self._make_request("GET", "accounts")
            
            balances = []
            for account_data in data:
                total = Decimal(account_data["balance"]) + Decimal(account_data["locked"])
                
                if total > 0:  # Only include non-zero balances
                    balances.append(Balance(
                        exchange=self.exchange_type,
                        currency=account_data["currency"],
                        total=total,
                        available=Decimal(account_data["balance"]),
                        locked=Decimal(account_data["locked"]),
                        updated_at=datetime.now()
                    ))
            
            return balances
            
        except Exception as e:
            logger.error(f"Error getting balance: {str(e)}")
            raise APIError(f"Failed to get balance: {str(e)}", api_name="upbit")
    
    @rate_limit("upbit", "accounts")
    async def get_positions(self) -> List[Position]:
        """Get current positions (crypto holdings)"""
        try:
            balances = await self.get_balance()
            
            positions = []
            for balance in balances:
                if balance.currency != "KRW" and balance.total > 0:
                    # For crypto, positions are just holdings
                    positions.append(Position(
                        symbol=f"KRW-{balance.currency}",  # Upbit format
                        exchange=self.exchange_type,
                        asset_type=AssetType.CRYPTO,
                        quantity=balance.total,
                        avg_price=Decimal("0"),  # Not provided by Upbit
                        market_value=balance.usd_value or Decimal("0"),
                        unrealized_pnl=Decimal("0"),  # Not calculated
                        opened_at=datetime.now(),
                        updated_at=balance.updated_at
                    ))
            
            return positions
            
        except Exception as e:
            logger.error(f"Error getting positions: {str(e)}")
            raise APIError(f"Failed to get positions: {str(e)}", api_name="upbit")
    
    @rate_limit("upbit", "accounts")
    async def get_account_info(self) -> AccountInfo:
        """Get account information"""
        try:
            balances = await self.get_balance()
            
            # Calculate total KRW value
            krw_balance = next((b for b in balances if b.currency == "KRW"), None)
            total_krw = krw_balance.total if krw_balance else Decimal("0")
            
            return AccountInfo(
                exchange=self.exchange_type,
                account_id="upbit_account",  # Upbit doesn't provide account ID
                is_active=True,
                trading_enabled=True,
                balances=balances,
                buying_power=krw_balance.available if krw_balance else Decimal("0"),
                updated_at=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error getting account info: {str(e)}")
            raise APIError(f"Failed to get account info: {str(e)}", api_name="upbit")
    
    @rate_limit("upbit", "orders")
    async def get_trades(self, symbol: Optional[str] = None, 
                        limit: int = 100) -> List[Trade]:
        """Get trade history"""
        try:
            params = {"state": "done", "limit": min(limit, 100)}
            if symbol:
                params["market"] = self._format_symbol(symbol)
            
            data = await self._make_request("GET", "orders", params=params)
            
            trades = []
            for order_data in data:
                if order_data["state"] == "done":  # Completed orders
                    trades.append(Trade(
                        id=order_data["uuid"],
                        order_id=order_data["uuid"],
                        symbol=order_data["market"],
                        exchange=self.exchange_type,
                        side=OrderSide(order_data["side"]),
                        quantity=Decimal(order_data["executed_volume"]),
                        price=Decimal(order_data["avg_price"]) if order_data.get("avg_price") else Decimal("0"),
                        commission=Decimal(order_data.get("paid_fee", "0")),
                        timestamp=datetime.fromisoformat(order_data["created_at"].replace("Z", "+00:00"))
                    ))
            
            return trades
            
        except Exception as e:
            logger.error(f"Error getting trades: {str(e)}")
            raise APIError(f"Failed to get trades: {str(e)}", api_name="upbit")
    
    # WebSocket Methods
    async def _authenticate_websocket(self) -> Dict[str, Any]:
        """Create WebSocket authentication message"""
        # Upbit WebSocket doesn't require authentication for public data
        return None
    
    async def _create_subscription_message(self, channel: str, 
                                         symbol: str) -> Dict[str, Any]:
        """Create subscription message for WebSocket"""
        upbit_symbol = self._format_symbol(symbol)
        
        if channel == "ticker":
            return [{
                "ticket": str(uuid.uuid4()),
                "type": "ticker",
                "codes": [upbit_symbol]
            }]
        elif channel == "trade":
            return [{
                "ticket": str(uuid.uuid4()),
                "type": "trade",
                "codes": [upbit_symbol]
            }]
        elif channel == "orderbook":
            return [{
                "ticket": str(uuid.uuid4()),
                "type": "orderbook",
                "codes": [upbit_symbol]
            }]
        
        return [{"ticket": str(uuid.uuid4())}]
    
    async def _parse_websocket_message(self, message: Dict[str, Any]) -> Optional[WebSocketMessage]:
        """Parse incoming WebSocket message"""
        try:
            msg_type = message.get("type", "")
            
            if msg_type == "ticker":
                return WebSocketMessage(
                    channel="ticker",
                    exchange=self.exchange_type,
                    timestamp=datetime.fromtimestamp(message["timestamp"] / 1000),
                    message_type="tick_data",
                    symbol=message["code"],
                    data={
                        "symbol": message["code"],
                        "exchange": self.exchange_type,
                        "timestamp": datetime.fromtimestamp(message["timestamp"] / 1000),
                        "price": Decimal(str(message["trade_price"])),
                        "size": Decimal(str(message["trade_volume"])),
                        "bid": Decimal(str(message.get("highest_52_week_price", 0))),
                        "ask": Decimal(str(message.get("lowest_52_week_price", 0)))
                    }
                )
            
            elif msg_type == "trade":
                return WebSocketMessage(
                    channel="trade",
                    exchange=self.exchange_type,
                    timestamp=datetime.fromtimestamp(message["timestamp"] / 1000),
                    message_type="tick_data",
                    symbol=message["code"],
                    data={
                        "symbol": message["code"],
                        "exchange": self.exchange_type,
                        "timestamp": datetime.fromtimestamp(message["timestamp"] / 1000),
                        "price": Decimal(str(message["trade_price"])),
                        "size": Decimal(str(message["trade_volume"]))
                    }
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error parsing WebSocket message: {str(e)}")
            return None
    
    # Helper Methods
    def _format_symbol(self, symbol: str) -> str:
        """Format symbol for Upbit API (KRW-SYMBOL format)"""
        if "-" in symbol:
            return symbol.upper()  # Already in correct format
        
        # Assume KRW market if not specified
        return f"KRW-{symbol.upper()}"
    
    def _convert_timeframe(self, timeframe: str) -> str:
        """Convert timeframe to Upbit format"""
        mapping = {
            "1m": "minutes/1",
            "5m": "minutes/5",
            "15m": "minutes/15",
            "30m": "minutes/30",
            "1h": "minutes/60",
            "4h": "minutes/240",
            "1d": "days",
            "1w": "weeks",
            "1M": "months"
        }
        return mapping.get(timeframe, "minutes/1")
    
    def _convert_order_type(self, order_type: OrderType) -> str:
        """Convert order type to Upbit format"""
        mapping = {
            OrderType.MARKET: "market",
            OrderType.LIMIT: "limit"
        }
        return mapping.get(order_type, "limit")
    
    def _convert_order_status_to_upbit(self, status: str) -> str:
        """Convert order status to Upbit format"""
        mapping = {
            "pending": "wait",
            "filled": "done",
            "cancelled": "cancel"
        }
        return mapping.get(status, "wait")
    
    def _parse_order(self, order_data: Dict[str, Any]) -> Order:
        """Parse Upbit order data to Order model"""
        return Order(
            id=order_data["uuid"],
            client_order_id=order_data.get("identifier"),
            symbol=order_data["market"],
            exchange=self.exchange_type,
            side=OrderSide(order_data["side"]),
            type=self._parse_order_type(order_data["ord_type"]),
            status=self._parse_order_status(order_data["state"]),
            quantity=Decimal(order_data.get("volume", order_data.get("price", "0"))),
            filled_quantity=Decimal(order_data.get("executed_volume", "0")),
            remaining_quantity=Decimal(order_data.get("remaining_volume", "0")),
            price=Decimal(order_data["price"]) if order_data.get("price") else None,
            avg_fill_price=Decimal(order_data["avg_price"]) if order_data.get("avg_price") else None,
            created_at=datetime.fromisoformat(order_data["created_at"].replace("Z", "+00:00")),
            time_in_force="GTC"
        )
    
    def _parse_order_type(self, upbit_type: str) -> OrderType:
        """Parse Upbit order type"""
        mapping = {
            "market": OrderType.MARKET,
            "limit": OrderType.LIMIT
        }
        return mapping.get(upbit_type, OrderType.LIMIT)
    
    def _parse_order_status(self, upbit_status: str) -> OrderStatus:
        """Parse Upbit order status"""
        mapping = {
            "wait": OrderStatus.PENDING,
            "done": OrderStatus.FILLED,
            "cancel": OrderStatus.CANCELLED
        }
        return mapping.get(upbit_status, OrderStatus.PENDING)
    
    async def close(self) -> None:
        """Close HTTP session"""
        if self.session and not self.session.closed:
            await self.session.close()
        await self.disconnect()