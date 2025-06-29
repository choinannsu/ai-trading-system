"""
Base abstract class for trading APIs
"""

import asyncio
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Optional, Callable, Any

try:
    import websockets
    from websockets.exceptions import ConnectionClosed, InvalidStatusCode
except ImportError:
    websockets = None
    ConnectionClosed = Exception
    InvalidStatusCode = Exception

from .models import (
    MarketData, TickData, Order, OrderRequest, Position, Balance, 
    AccountInfo, Trade, Symbol, ExchangeType, WebSocketMessage
)
from .rate_limiter import rate_limiter
from utils.logger import get_logger
from utils.exceptions import APIError, TradingError, DataCollectionError

logger = get_logger(__name__)


class TradingAPI(ABC):
    """Abstract base class for trading APIs"""
    
    def __init__(self, api_key: str, secret_key: str, environment: str = "sandbox"):
        self.api_key = api_key
        self.secret_key = secret_key
        self.environment = environment
        self.is_connected = False
        self.websocket = None
        self.websocket_subscriptions = set()
        self.message_handlers: Dict[str, Callable] = {}
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 5
        self.reconnect_delay = 5  # seconds
        
        # Event callbacks
        self.on_market_data: Optional[Callable[[MarketData], None]] = None
        self.on_tick_data: Optional[Callable[[TickData], None]] = None
        self.on_order_update: Optional[Callable[[Order], None]] = None
        self.on_trade: Optional[Callable[[Trade], None]] = None
        self.on_error: Optional[Callable[[Exception], None]] = None
        
    @property
    @abstractmethod
    def exchange_type(self) -> ExchangeType:
        """Get exchange type"""
        pass
    
    @property
    @abstractmethod
    def base_url(self) -> str:
        """Get base API URL"""
        pass
    
    @property
    @abstractmethod
    def websocket_url(self) -> str:
        """Get WebSocket URL"""
        pass
    
    # Market Data Methods
    @abstractmethod
    async def get_market_data(self, symbol: str, timeframe: str = "1m", 
                            limit: int = 100) -> List[MarketData]:
        """Get historical market data"""
        pass
    
    @abstractmethod
    async def get_current_price(self, symbol: str) -> TickData:
        """Get current price for symbol"""
        pass
    
    @abstractmethod
    async def get_symbols(self) -> List[Symbol]:
        """Get available trading symbols"""
        pass
    
    # Trading Methods
    @abstractmethod
    async def place_order(self, order_request: OrderRequest) -> Order:
        """Place a trading order"""
        pass
    
    @abstractmethod
    async def cancel_order(self, order_id: str) -> Order:
        """Cancel an existing order"""
        pass
    
    @abstractmethod
    async def get_order(self, order_id: str) -> Order:
        """Get order details"""
        pass
    
    @abstractmethod
    async def get_orders(self, symbol: Optional[str] = None, 
                        status: Optional[str] = None) -> List[Order]:
        """Get orders list"""
        pass
    
    # Account Methods
    @abstractmethod
    async def get_balance(self) -> List[Balance]:
        """Get account balance"""
        pass
    
    @abstractmethod
    async def get_positions(self) -> List[Position]:
        """Get current positions"""
        pass
    
    @abstractmethod
    async def get_account_info(self) -> AccountInfo:
        """Get account information"""
        pass
    
    @abstractmethod
    async def get_trades(self, symbol: Optional[str] = None, 
                        limit: int = 100) -> List[Trade]:
        """Get trade history"""
        pass
    
    # WebSocket Methods
    @abstractmethod
    async def _authenticate_websocket(self) -> Dict[str, Any]:
        """Create WebSocket authentication message"""
        pass
    
    @abstractmethod
    async def _create_subscription_message(self, channel: str, 
                                         symbol: str) -> Dict[str, Any]:
        """Create subscription message for WebSocket"""
        pass
    
    @abstractmethod
    async def _parse_websocket_message(self, message: Dict[str, Any]) -> Optional[WebSocketMessage]:
        """Parse incoming WebSocket message"""
        pass
    
    # Connection Management
    async def connect(self) -> None:
        """Connect to the exchange"""
        try:
            logger.info(f"Connecting to {self.exchange_type.value} exchange")
            await self._connect_websocket()
            self.is_connected = True
            self.reconnect_attempts = 0
            logger.info(f"Successfully connected to {self.exchange_type.value}")
        except Exception as e:
            logger.error(f"Failed to connect to {self.exchange_type.value}: {str(e)}")
            raise APIError(f"Connection failed: {str(e)}", api_name=self.exchange_type.value)
    
    async def disconnect(self) -> None:
        """Disconnect from the exchange"""
        try:
            self.is_connected = False
            if self.websocket:
                await self.websocket.close()
                self.websocket = None
            logger.info(f"Disconnected from {self.exchange_type.value}")
        except Exception as e:
            logger.error(f"Error during disconnect: {str(e)}")
    
    async def _connect_websocket(self) -> None:
        """Connect to WebSocket"""
        try:
            self.websocket = await websockets.connect(
                self.websocket_url,
                ping_interval=20,
                ping_timeout=10
            )
            
            # Authenticate if required
            auth_message = await self._authenticate_websocket()
            if auth_message:
                await self.websocket.send(str(auth_message))
            
            # Start message handler
            asyncio.create_task(self._websocket_message_handler())
            
        except Exception as e:
            raise APIError(f"WebSocket connection failed: {str(e)}", 
                         api_name=self.exchange_type.value)
    
    async def _websocket_message_handler(self) -> None:
        """Handle incoming WebSocket messages"""
        try:
            async for message in self.websocket:
                try:
                    import json
                    data = json.loads(message)
                    parsed_message = await self._parse_websocket_message(data)
                    
                    if parsed_message:
                        await self._handle_parsed_message(parsed_message)
                        
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse WebSocket message: {str(e)}")
                except Exception as e:
                    logger.error(f"Error handling WebSocket message: {str(e)}")
                    if self.on_error:
                        self.on_error(e)
        
        except ConnectionClosed:
            logger.warning(f"WebSocket connection closed for {self.exchange_type.value}")
            await self._handle_disconnect()
        except Exception as e:
            logger.error(f"WebSocket error: {str(e)}")
            await self._handle_disconnect()
    
    async def _handle_parsed_message(self, message: WebSocketMessage) -> None:
        """Handle parsed WebSocket message"""
        try:
            if message.message_type == "market_data" and self.on_market_data:
                market_data = MarketData(**message.data)
                self.on_market_data(market_data)
            
            elif message.message_type == "tick_data" and self.on_tick_data:
                tick_data = TickData(**message.data)
                self.on_tick_data(tick_data)
            
            elif message.message_type == "order_update" and self.on_order_update:
                order = Order(**message.data)
                self.on_order_update(order)
            
            elif message.message_type == "trade" and self.on_trade:
                trade = Trade(**message.data)
                self.on_trade(trade)
                
        except Exception as e:
            logger.error(f"Error handling parsed message: {str(e)}")
            if self.on_error:
                self.on_error(e)
    
    async def _handle_disconnect(self) -> None:
        """Handle WebSocket disconnection"""
        self.is_connected = False
        
        if self.reconnect_attempts < self.max_reconnect_attempts:
            self.reconnect_attempts += 1
            wait_time = self.reconnect_delay * (2 ** (self.reconnect_attempts - 1))
            
            logger.info(f"Attempting to reconnect to {self.exchange_type.value} "
                       f"(attempt {self.reconnect_attempts}/{self.max_reconnect_attempts}) "
                       f"in {wait_time} seconds")
            
            await asyncio.sleep(wait_time)
            
            try:
                await self.connect()
                # Re-subscribe to channels
                await self._resubscribe_channels()
            except Exception as e:
                logger.error(f"Reconnection failed: {str(e)}")
                if self.reconnect_attempts >= self.max_reconnect_attempts:
                    logger.error(f"Max reconnection attempts reached for {self.exchange_type.value}")
                    if self.on_error:
                        self.on_error(APIError("Max reconnection attempts reached"))
        else:
            logger.error(f"Max reconnection attempts reached for {self.exchange_type.value}")
            if self.on_error:
                self.on_error(APIError("Max reconnection attempts reached"))
    
    async def _resubscribe_channels(self) -> None:
        """Re-subscribe to WebSocket channels after reconnection"""
        for subscription in self.websocket_subscriptions.copy():
            try:
                channel, symbol = subscription.split(":", 1)
                await self.subscribe_market_data(symbol, channel)
            except Exception as e:
                logger.error(f"Failed to resubscribe to {subscription}: {str(e)}")
    
    # Subscription Methods
    async def subscribe_market_data(self, symbol: str, channel: str = "trade") -> None:
        """Subscribe to market data WebSocket channel"""
        if not self.is_connected:
            raise APIError("Not connected to exchange", api_name=self.exchange_type.value)
        
        try:
            subscription_key = f"{channel}:{symbol}"
            
            if subscription_key not in self.websocket_subscriptions:
                message = await self._create_subscription_message(channel, symbol)
                await self.websocket.send(str(message))
                self.websocket_subscriptions.add(subscription_key)
                logger.info(f"Subscribed to {channel} for {symbol} on {self.exchange_type.value}")
            
        except Exception as e:
            raise APIError(f"Subscription failed: {str(e)}", 
                         api_name=self.exchange_type.value)
    
    async def unsubscribe_market_data(self, symbol: str, channel: str = "trade") -> None:
        """Unsubscribe from market data WebSocket channel"""
        subscription_key = f"{channel}:{symbol}"
        
        if subscription_key in self.websocket_subscriptions:
            # Create unsubscription message (implementation depends on exchange)
            # Most exchanges use similar format with "unsubscribe" action
            try:
                message = await self._create_subscription_message(channel, symbol)
                message["method"] = "UNSUBSCRIBE"  # Common pattern
                await self.websocket.send(str(message))
                self.websocket_subscriptions.remove(subscription_key)
                logger.info(f"Unsubscribed from {channel} for {symbol} on {self.exchange_type.value}")
            except Exception as e:
                logger.error(f"Unsubscription failed: {str(e)}")
    
    # Health Check
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check"""
        try:
            # Test API connection
            account_info = await self.get_account_info()
            
            # Check WebSocket connection
            websocket_status = "connected" if self.is_connected else "disconnected"
            
            # Get rate limit status
            rate_limit_status = rate_limiter.get_status(self.exchange_type.value)
            
            return {
                "exchange": self.exchange_type.value,
                "api_connection": "healthy",
                "websocket_connection": websocket_status,
                "account_status": "active" if account_info.is_active else "inactive",
                "rate_limits": rate_limit_status,
                "reconnect_attempts": self.reconnect_attempts,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "exchange": self.exchange_type.value,
                "api_connection": "unhealthy",
                "websocket_connection": "disconnected" if not self.is_connected else "unknown",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    # Rate Limited Request Helper
    async def _make_request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """Make rate-limited HTTP request"""
        await rate_limiter.acquire(self.exchange_type.value, endpoint)
        
        # Implementation would depend on HTTP client (aiohttp, httpx, etc.)
        # This is a placeholder for the actual HTTP request logic
        raise NotImplementedError("Subclasses must implement _make_request method")
    
    # Utility Methods
    def set_callbacks(self, **callbacks) -> None:
        """Set event callbacks"""
        for event, callback in callbacks.items():
            if hasattr(self, f"on_{event}"):
                setattr(self, f"on_{event}", callback)
            else:
                logger.warning(f"Unknown callback event: {event}")
    
    def get_connection_status(self) -> Dict[str, Any]:
        """Get connection status"""
        return {
            "exchange": self.exchange_type.value,
            "is_connected": self.is_connected,
            "websocket_connected": self.websocket is not None and not self.websocket.closed,
            "subscriptions": list(self.websocket_subscriptions),
            "reconnect_attempts": self.reconnect_attempts
        }