"""
API Manager for multi-exchange trading system
"""

import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Set, Callable, Any, Union
from collections import defaultdict
from dataclasses import dataclass

from .base_api import TradingAPI
from .alpaca_api import AlpacaAPI
from .upbit_api import UpbitAPI
from .kis_api import KisAPI
from .models import (
    MarketData, TickData, Order, OrderRequest, Position, Balance, 
    AccountInfo, Trade, Symbol, ExchangeType, WebSocketMessage,
    OrderSide, OrderType, OrderStatus, AssetType
)
from utils.config import get_config
from utils.logger import get_logger, log_trade, log_performance, log_system_health
from utils.exceptions import APIError, TradingError, SystemHealthError

logger = get_logger(__name__)


@dataclass
class ExchangeStatus:
    """Exchange connection status"""
    exchange: ExchangeType
    is_connected: bool
    last_heartbeat: datetime
    error_count: int
    last_error: Optional[str] = None
    health_score: float = 1.0  # 0.0 to 1.0


@dataclass
class UnifiedQuote:
    """Unified quote across exchanges"""
    symbol: str
    exchanges: Dict[ExchangeType, TickData]
    best_bid: Optional[TickData] = None
    best_ask: Optional[TickData] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        self._calculate_best_prices()
    
    def _calculate_best_prices(self):
        """Calculate best bid/ask across exchanges"""
        best_bid_price = Decimal('0')
        best_ask_price = Decimal('999999999')
        
        for tick_data in self.exchanges.values():
            if tick_data.bid and tick_data.bid > best_bid_price:
                best_bid_price = tick_data.bid
                self.best_bid = tick_data
            
            if tick_data.ask and tick_data.ask < best_ask_price:
                best_ask_price = tick_data.ask
                self.best_ask = tick_data


class APIManager:
    """Multi-exchange API manager with failover and load balancing"""
    
    def __init__(self):
        self.exchanges: Dict[ExchangeType, TradingAPI] = {}
        self.exchange_status: Dict[ExchangeType, ExchangeStatus] = {}
        self.enabled_exchanges: Set[ExchangeType] = set()
        self.primary_exchanges: Dict[AssetType, ExchangeType] = {}
        
        # Data caching and aggregation
        self.latest_quotes: Dict[str, UnifiedQuote] = {}
        self.price_cache: Dict[str, Dict[ExchangeType, TickData]] = defaultdict(dict)
        self.market_data_cache: Dict[str, List[MarketData]] = {}
        
        # Event callbacks
        self.on_market_data: Optional[Callable[[MarketData], None]] = None
        self.on_tick_data: Optional[Callable[[TickData], None]] = None
        self.on_order_update: Optional[Callable[[Order], None]] = None
        self.on_trade: Optional[Callable[[Trade], None]] = None
        self.on_exchange_error: Optional[Callable[[ExchangeType, Exception], None]] = None
        
        # Health monitoring
        self.health_check_interval = 30  # seconds
        self.max_error_count = 5
        self.failover_cooldown = 300  # 5 minutes
        
        self._initialize_exchanges()
        
    def _initialize_exchanges(self) -> None:
        """Initialize exchange connections based on configuration"""
        config = get_config()
        
        # Initialize Alpaca for US stocks
        if config.api.alpaca_api_key and config.api.alpaca_secret_key:
            self.exchanges[ExchangeType.ALPACA] = AlpacaAPI(
                config.api.alpaca_api_key,
                config.api.alpaca_secret_key,
                config.environment
            )
            self.enabled_exchanges.add(ExchangeType.ALPACA)
            self.primary_exchanges[AssetType.STOCK] = ExchangeType.ALPACA
            self.primary_exchanges[AssetType.ETF] = ExchangeType.ALPACA
        
        # Initialize Upbit for crypto
        if config.api.upbit_access_key and config.api.upbit_secret_key:
            self.exchanges[ExchangeType.UPBIT] = UpbitAPI(
                config.api.upbit_access_key,
                config.api.upbit_secret_key,
                config.environment
            )
            self.enabled_exchanges.add(ExchangeType.UPBIT)
            self.primary_exchanges[AssetType.CRYPTO] = ExchangeType.UPBIT
        
        # Initialize KIS for Korean stocks
        if config.api.kis_api_key and config.api.kis_secret_key:
            self.exchanges[ExchangeType.KIS] = KisAPI(
                config.api.kis_api_key,
                config.api.kis_secret_key,
                getattr(config.api, 'kis_account_no', ''),
                config.environment
            )
            self.enabled_exchanges.add(ExchangeType.KIS)
            # Set KIS as primary for Korean stocks
            if AssetType.STOCK not in self.primary_exchanges:
                self.primary_exchanges[AssetType.STOCK] = ExchangeType.KIS
        
        # Initialize exchange status
        for exchange_type in self.enabled_exchanges:
            self.exchange_status[exchange_type] = ExchangeStatus(
                exchange=exchange_type,
                is_connected=False,
                last_heartbeat=datetime.now(),
                error_count=0
            )
        
        logger.info(f"Initialized {len(self.enabled_exchanges)} exchanges: {self.enabled_exchanges}")
    
    async def connect_all(self) -> Dict[ExchangeType, bool]:
        """Connect to all enabled exchanges"""
        connection_results = {}
        
        for exchange_type in self.enabled_exchanges:
            try:
                api = self.exchanges[exchange_type]
                
                # Set up event callbacks
                api.set_callbacks(
                    market_data=self._handle_market_data,
                    tick_data=self._handle_tick_data,
                    order_update=self._handle_order_update,
                    trade=self._handle_trade,
                    error=lambda e, ex=exchange_type: self._handle_exchange_error(ex, e)
                )
                
                await api.connect()
                self.exchange_status[exchange_type].is_connected = True
                self.exchange_status[exchange_type].last_heartbeat = datetime.now()
                connection_results[exchange_type] = True
                
                logger.info(f"Successfully connected to {exchange_type.value}")
                
            except Exception as e:
                logger.error(f"Failed to connect to {exchange_type.value}: {str(e)}")
                self.exchange_status[exchange_type].is_connected = False
                self.exchange_status[exchange_type].error_count += 1
                self.exchange_status[exchange_type].last_error = str(e)
                connection_results[exchange_type] = False
        
        # Start health monitoring
        asyncio.create_task(self._health_monitor())
        
        return connection_results
    
    async def disconnect_all(self) -> None:
        """Disconnect from all exchanges"""
        for exchange_type, api in self.exchanges.items():
            try:
                await api.disconnect()
                self.exchange_status[exchange_type].is_connected = False
                logger.info(f"Disconnected from {exchange_type.value}")
            except Exception as e:
                logger.error(f"Error disconnecting from {exchange_type.value}: {str(e)}")
    
    # Market Data Methods
    async def get_market_data(self, symbol: str, timeframe: str = "1m", 
                            limit: int = 100, 
                            exchange: Optional[ExchangeType] = None) -> List[MarketData]:
        """Get market data from specified exchange or best available"""
        
        if exchange:
            # Use specific exchange
            if exchange in self.enabled_exchanges and self.exchange_status[exchange].is_connected:
                api = self.exchanges[exchange]
                return await api.get_market_data(symbol, timeframe, limit)
            else:
                raise APIError(f"Exchange {exchange.value} not available")
        
        # Try exchanges in order of preference
        asset_type = self._guess_asset_type(symbol)
        preferred_exchanges = self._get_preferred_exchanges(asset_type)
        
        for exchange_type in preferred_exchanges:
            if self.exchange_status[exchange_type].is_connected:
                try:
                    api = self.exchanges[exchange_type]
                    data = await api.get_market_data(symbol, timeframe, limit)
                    
                    # Cache the data
                    cache_key = f"{symbol}:{timeframe}:{limit}"
                    self.market_data_cache[cache_key] = data
                    
                    return data
                    
                except Exception as e:
                    logger.warning(f"Failed to get market data from {exchange_type.value}: {str(e)}")
                    await self._handle_exchange_error(exchange_type, e)
                    continue
        
        raise APIError("No exchanges available for market data")
    
    async def get_current_price(self, symbol: str, 
                              exchange: Optional[ExchangeType] = None) -> TickData:
        """Get current price from specified exchange or best available"""
        
        if exchange:
            if exchange in self.enabled_exchanges and self.exchange_status[exchange].is_connected:
                api = self.exchanges[exchange]
                return await api.get_current_price(symbol)
            else:
                raise APIError(f"Exchange {exchange.value} not available")
        
        # Get prices from all available exchanges
        prices = {}
        asset_type = self._guess_asset_type(symbol)
        preferred_exchanges = self._get_preferred_exchanges(asset_type)
        
        for exchange_type in preferred_exchanges:
            if self.exchange_status[exchange_type].is_connected:
                try:
                    api = self.exchanges[exchange_type]
                    price = await api.get_current_price(symbol)
                    prices[exchange_type] = price
                    
                    # Update price cache
                    self.price_cache[symbol][exchange_type] = price
                    
                except Exception as e:
                    logger.warning(f"Failed to get price from {exchange_type.value}: {str(e)}")
                    continue
        
        if not prices:
            raise APIError("No exchanges available for current price")
        
        # Update unified quote
        self.latest_quotes[symbol] = UnifiedQuote(symbol=symbol, exchanges=prices)
        
        # Return price from primary exchange or first available
        primary_exchange = self.primary_exchanges.get(asset_type)
        if primary_exchange and primary_exchange in prices:
            return prices[primary_exchange]
        
        return next(iter(prices.values()))
    
    async def get_unified_quote(self, symbol: str) -> UnifiedQuote:
        """Get unified quote across all exchanges"""
        await self.get_current_price(symbol)  # This updates the unified quote
        return self.latest_quotes.get(symbol)
    
    # Trading Methods
    async def place_order(self, order_request: OrderRequest, 
                         exchange: Optional[ExchangeType] = None) -> Order:
        """Place order on specified exchange or best available"""
        
        if exchange:
            if exchange in self.enabled_exchanges and self.exchange_status[exchange].is_connected:
                api = self.exchanges[exchange]
                order = await api.place_order(order_request)
                log_trade(
                    action=f"PLACE_ORDER_{order_request.side.value}",
                    symbol=order_request.symbol,
                    quantity=float(order_request.quantity),
                    price=float(order_request.price or 0),
                    exchange=exchange.value
                )
                return order
            else:
                raise TradingError(f"Exchange {exchange.value} not available for trading")
        
        # Auto-select exchange based on asset type
        asset_type = self._guess_asset_type(order_request.symbol)
        preferred_exchanges = self._get_preferred_exchanges(asset_type)
        
        for exchange_type in preferred_exchanges:
            if self.exchange_status[exchange_type].is_connected:
                try:
                    api = self.exchanges[exchange_type]
                    order = await api.place_order(order_request)
                    log_trade(
                        action=f"PLACE_ORDER_{order_request.side.value}",
                        symbol=order_request.symbol,
                        quantity=float(order_request.quantity),
                        price=float(order_request.price or 0),
                        exchange=exchange_type.value
                    )
                    return order
                    
                except Exception as e:
                    logger.warning(f"Failed to place order on {exchange_type.value}: {str(e)}")
                    await self._handle_exchange_error(exchange_type, e)
                    continue
        
        raise TradingError("No exchanges available for trading")
    
    async def cancel_order(self, order_id: str, symbol: str = None,
                          exchange: Optional[ExchangeType] = None) -> Order:
        """Cancel order on specified exchange"""
        
        if not exchange:
            # Try to find order across all exchanges
            for exchange_type in self.enabled_exchanges:
                if self.exchange_status[exchange_type].is_connected:
                    try:
                        api = self.exchanges[exchange_type]
                        return await api.cancel_order(order_id, symbol)
                    except Exception:
                        continue
            raise TradingError("Order not found on any exchange")
        
        if exchange in self.enabled_exchanges and self.exchange_status[exchange].is_connected:
            api = self.exchanges[exchange]
            return await api.cancel_order(order_id, symbol)
        else:
            raise TradingError(f"Exchange {exchange.value} not available")
    
    # Account Methods
    async def get_all_balances(self) -> Dict[ExchangeType, List[Balance]]:
        """Get balances from all connected exchanges"""
        balances = {}
        
        for exchange_type in self.enabled_exchanges:
            if self.exchange_status[exchange_type].is_connected:
                try:
                    api = self.exchanges[exchange_type]
                    balances[exchange_type] = await api.get_balance()
                except Exception as e:
                    logger.warning(f"Failed to get balance from {exchange_type.value}: {str(e)}")
                    balances[exchange_type] = []
        
        return balances
    
    async def get_all_positions(self) -> Dict[ExchangeType, List[Position]]:
        """Get positions from all connected exchanges"""
        positions = {}
        
        for exchange_type in self.enabled_exchanges:
            if self.exchange_status[exchange_type].is_connected:
                try:
                    api = self.exchanges[exchange_type]
                    positions[exchange_type] = await api.get_positions()
                except Exception as e:
                    logger.warning(f"Failed to get positions from {exchange_type.value}: {str(e)}")
                    positions[exchange_type] = []
        
        return positions
    
    async def get_unified_portfolio(self) -> Dict[str, Any]:
        """Get unified portfolio view across all exchanges"""
        all_balances = await self.get_all_balances()
        all_positions = await self.get_all_positions()
        
        # Aggregate balances by currency
        total_balances = defaultdict(Decimal)
        total_usd_value = Decimal('0')
        
        for exchange_balances in all_balances.values():
            for balance in exchange_balances:
                total_balances[balance.currency] += balance.total
                if balance.usd_value:
                    total_usd_value += balance.usd_value
        
        # Aggregate positions by symbol
        total_positions = defaultdict(lambda: {
            'quantity': Decimal('0'),
            'market_value': Decimal('0'),
            'unrealized_pnl': Decimal('0'),
            'exchanges': []
        })
        
        for exchange_type, positions in all_positions.items():
            for position in positions:
                pos_data = total_positions[position.symbol]
                pos_data['quantity'] += position.quantity
                pos_data['market_value'] += position.market_value
                pos_data['unrealized_pnl'] += position.unrealized_pnl
                pos_data['exchanges'].append(exchange_type.value)
        
        return {
            'balances': dict(total_balances),
            'total_usd_value': total_usd_value,
            'positions': dict(total_positions),
            'exchanges': list(self.enabled_exchanges),
            'timestamp': datetime.now()
        }
    
    # Health Monitoring
    async def _health_monitor(self) -> None:
        """Monitor exchange health and perform auto-recovery"""
        while True:
            try:
                await asyncio.sleep(self.health_check_interval)
                
                for exchange_type in self.enabled_exchanges:
                    try:
                        api = self.exchanges[exchange_type]
                        health = await api.health_check()
                        
                        status = self.exchange_status[exchange_type]
                        status.last_heartbeat = datetime.now()
                        
                        # Update health score based on API response
                        if health.get('api_connection') == 'healthy':
                            status.health_score = min(1.0, status.health_score + 0.1)
                            status.error_count = max(0, status.error_count - 1)
                        else:
                            status.health_score = max(0.0, status.health_score - 0.2)
                            status.error_count += 1
                        
                        # Check if reconnection is needed
                        if (not status.is_connected or 
                            status.error_count >= self.max_error_count):
                            await self._attempt_reconnection(exchange_type)
                        
                        log_system_health(
                            component=f"exchange_{exchange_type.value}",
                            status="healthy" if status.is_connected else "unhealthy",
                            health_score=status.health_score,
                            error_count=status.error_count
                        )
                        
                    except Exception as e:
                        await self._handle_exchange_error(exchange_type, e)
                
            except Exception as e:
                logger.error(f"Health monitor error: {str(e)}")
    
    async def _attempt_reconnection(self, exchange_type: ExchangeType) -> None:
        """Attempt to reconnect to exchange"""
        status = self.exchange_status[exchange_type]
        
        # Check cooldown period
        if (datetime.now() - status.last_heartbeat).seconds < self.failover_cooldown:
            return
        
        try:
            logger.info(f"Attempting to reconnect to {exchange_type.value}")
            api = self.exchanges[exchange_type]
            
            if status.is_connected:
                await api.disconnect()
            
            await api.connect()
            
            status.is_connected = True
            status.error_count = 0
            status.health_score = 1.0
            status.last_error = None
            
            logger.info(f"Successfully reconnected to {exchange_type.value}")
            
        except Exception as e:
            logger.error(f"Reconnection failed for {exchange_type.value}: {str(e)}")
            status.is_connected = False
            status.error_count += 1
            status.last_error = str(e)
    
    # Event Handlers
    async def _handle_market_data(self, market_data: MarketData) -> None:
        """Handle market data from exchanges"""
        if self.on_market_data:
            self.on_market_data(market_data)
    
    async def _handle_tick_data(self, tick_data: TickData) -> None:
        """Handle tick data from exchanges"""
        # Update price cache
        self.price_cache[tick_data.symbol][tick_data.exchange] = tick_data
        
        # Update unified quote
        if tick_data.symbol in self.latest_quotes:
            self.latest_quotes[tick_data.symbol].exchanges[tick_data.exchange] = tick_data
            self.latest_quotes[tick_data.symbol]._calculate_best_prices()
        
        if self.on_tick_data:
            self.on_tick_data(tick_data)
    
    async def _handle_order_update(self, order: Order) -> None:
        """Handle order updates from exchanges"""
        if self.on_order_update:
            self.on_order_update(order)
    
    async def _handle_trade(self, trade: Trade) -> None:
        """Handle trade executions from exchanges"""
        log_trade(
            action=f"EXECUTION_{trade.side.value}",
            symbol=trade.symbol,
            quantity=float(trade.quantity),
            price=float(trade.price),
            exchange=trade.exchange.value
        )
        
        if self.on_trade:
            self.on_trade(trade)
    
    async def _handle_exchange_error(self, exchange_type: ExchangeType, error: Exception) -> None:
        """Handle exchange errors"""
        status = self.exchange_status[exchange_type]
        status.error_count += 1
        status.last_error = str(error)
        status.health_score = max(0.0, status.health_score - 0.3)
        
        if status.error_count >= self.max_error_count:
            status.is_connected = False
            logger.error(f"Exchange {exchange_type.value} marked as disconnected due to errors")
        
        if self.on_exchange_error:
            self.on_exchange_error(exchange_type, error)
    
    # Utility Methods
    def _guess_asset_type(self, symbol: str) -> AssetType:
        """Guess asset type from symbol"""
        # Crypto patterns
        if any(pair in symbol.upper() for pair in ['BTC', 'ETH', 'USDT', 'BNB']):
            return AssetType.CRYPTO
        
        # Korean stock patterns (6 digits)
        if symbol.isdigit() and len(symbol) == 6:
            return AssetType.STOCK
        
        # US stock patterns (letters)
        if symbol.isalpha() and len(symbol) <= 5:
            return AssetType.STOCK
        
        # Default to stock
        return AssetType.STOCK
    
    def _get_preferred_exchanges(self, asset_type: AssetType) -> List[ExchangeType]:
        """Get preferred exchanges for asset type"""
        preferred = []
        
        # Add primary exchange first
        if asset_type in self.primary_exchanges:
            preferred.append(self.primary_exchanges[asset_type])
        
        # Add other available exchanges
        for exchange_type in self.enabled_exchanges:
            if exchange_type not in preferred:
                preferred.append(exchange_type)
        
        return preferred
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status"""
        connected_count = sum(1 for status in self.exchange_status.values() if status.is_connected)
        total_count = len(self.enabled_exchanges)
        
        avg_health_score = sum(status.health_score for status in self.exchange_status.values()) / total_count if total_count > 0 else 0
        
        return {
            'exchanges_connected': f"{connected_count}/{total_count}",
            'average_health_score': avg_health_score,
            'exchange_status': {
                exchange_type.value: {
                    'connected': status.is_connected,
                    'health_score': status.health_score,
                    'error_count': status.error_count,
                    'last_error': status.last_error
                }
                for exchange_type, status in self.exchange_status.items()
            },
            'cache_stats': {
                'price_cache_size': len(self.price_cache),
                'quotes_cached': len(self.latest_quotes),
                'market_data_cached': len(self.market_data_cache)
            },
            'timestamp': datetime.now()
        }
    
    def set_callbacks(self, **callbacks) -> None:
        """Set event callbacks"""
        for event, callback in callbacks.items():
            if hasattr(self, f"on_{event}"):
                setattr(self, f"on_{event}", callback)
            else:
                logger.warning(f"Unknown callback event: {event}")
    
    async def close(self) -> None:
        """Close all connections"""
        await self.disconnect_all()
        
        for api in self.exchanges.values():
            if hasattr(api, 'close'):
                await api.close()


# Global API manager instance
api_manager = APIManager()