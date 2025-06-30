"""
Market Data Collector for real-time market data collection and processing
Collects candle data, orderbook, trades, and volume profile across multiple exchanges
"""

import asyncio
import json
from datetime import datetime, timedelta
from decimal import Decimal, InvalidOperation
from typing import Dict, List, Optional, Set, Any, Tuple
import statistics
from dataclasses import dataclass
from collections import defaultdict, deque

import aiohttp
try:
    import websockets
except ImportError:
    websockets = None

from .api_manager import api_manager
from .models import (
    MarketData, TickData, ExchangeType, TimeFrame, AssetType
)
from utils.logger import get_logger
from utils.exceptions import DataCollectionError, ValidationError

logger = get_logger(__name__)


@dataclass
class VolumeProfile:
    """Volume profile data structure"""
    price_levels: Dict[Decimal, Decimal]  # price -> volume
    poc: Decimal  # Point of Control (highest volume price)
    vah: Decimal  # Value Area High
    val: Decimal  # Value Area Low
    total_volume: Decimal
    timestamp: datetime


@dataclass
class OrderbookData:
    """Orderbook snapshot data"""
    symbol: str
    exchange: ExchangeType
    timestamp: datetime
    bids: List[Tuple[Decimal, Decimal]]  # (price, size)
    asks: List[Tuple[Decimal, Decimal]]  # (price, size)
    spread: Decimal
    mid_price: Decimal


@dataclass
class TradeData:
    """Individual trade data"""
    symbol: str
    exchange: ExchangeType
    timestamp: datetime
    price: Decimal
    size: Decimal
    side: str  # 'buy' or 'sell'
    trade_id: str


class DataValidator:
    """Data validation and filtering for market data"""
    
    def __init__(self):
        self.price_thresholds = {
            ExchangeType.UPBIT: (Decimal('1'), Decimal('1000000000')),  # KRW
            ExchangeType.ALPACA: (Decimal('0.01'), Decimal('10000')),   # USD
            ExchangeType.KIS: (Decimal('1'), Decimal('10000000'))       # KRW
        }
        self.volume_thresholds = {
            ExchangeType.UPBIT: Decimal('0.00000001'),
            ExchangeType.ALPACA: Decimal('1'),
            ExchangeType.KIS: Decimal('1')
        }
    
    def validate_price(self, price: Decimal, exchange: ExchangeType) -> bool:
        """Validate price data"""
        if not isinstance(price, Decimal) or price <= 0:
            return False
        
        min_price, max_price = self.price_thresholds.get(exchange, (Decimal('0'), Decimal('999999999')))
        return min_price <= price <= max_price
    
    def validate_volume(self, volume: Decimal, exchange: ExchangeType) -> bool:
        """Validate volume data"""
        if not isinstance(volume, Decimal) or volume < 0:
            return False
        
        min_volume = self.volume_thresholds.get(exchange, Decimal('0'))
        return volume >= min_volume
    
    def validate_candle(self, candle: MarketData) -> bool:
        """Validate complete candle data"""
        try:
            # Price validation
            prices = [candle.open, candle.high, candle.low, candle.close]
            for price in prices:
                if not self.validate_price(price, candle.exchange):
                    return False
            
            # OHLC logic validation
            if not (candle.low <= candle.open <= candle.high and
                   candle.low <= candle.close <= candle.high):
                return False
            
            # Volume validation
            if not self.validate_volume(candle.volume, candle.exchange):
                return False
            
            # Timestamp validation
            if candle.timestamp > datetime.now() + timedelta(minutes=5):
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Candle validation error: {e}")
            return False
    
    def filter_outliers(self, prices: List[Decimal], threshold: float = 3.0) -> List[Decimal]:
        """Filter price outliers using standard deviation"""
        if len(prices) < 3:
            return prices
        
        try:
            mean_price = statistics.mean(float(p) for p in prices)
            std_dev = statistics.stdev(float(p) for p in prices)
            
            if std_dev == 0:
                return prices
            
            filtered = []
            for price in prices:
                z_score = abs(float(price) - mean_price) / std_dev
                if z_score <= threshold:
                    filtered.append(price)
            
            return filtered if filtered else prices
            
        except Exception as e:
            logger.error(f"Outlier filtering error: {e}")
            return prices


class VolumeProfileCalculator:
    """Calculate volume profile from trade data"""
    
    def __init__(self, num_levels: int = 100):
        self.num_levels = num_levels
    
    def calculate(self, trades: List[TradeData], price_range: Tuple[Decimal, Decimal] = None) -> VolumeProfile:
        """Calculate volume profile from trades"""
        if not trades:
            raise ValueError("No trades provided for volume profile calculation")
        
        # Determine price range
        if not price_range:
            prices = [trade.price for trade in trades]
            min_price = min(prices)
            max_price = max(prices)
        else:
            min_price, max_price = price_range
        
        # Create price levels
        price_step = (max_price - min_price) / self.num_levels
        if price_step == 0:
            price_step = Decimal('0.01')
        
        volume_by_level = defaultdict(Decimal)
        
        # Aggregate volume by price level
        for trade in trades:
            level = int((trade.price - min_price) / price_step)
            level = max(0, min(level, self.num_levels - 1))
            level_price = min_price + (level * price_step)
            volume_by_level[level_price] += trade.size
        
        if not volume_by_level:
            raise ValueError("No volume data calculated")
        
        # Find Point of Control (POC)
        poc = max(volume_by_level.items(), key=lambda x: x[1])[0]
        
        # Calculate Value Area (70% of volume)
        total_volume = sum(volume_by_level.values())
        value_area_volume = total_volume * Decimal('0.7')
        
        # Find Value Area High and Low
        sorted_levels = sorted(volume_by_level.items(), key=lambda x: x[1], reverse=True)
        cumulative_volume = Decimal('0')
        value_area_prices = []
        
        for price, volume in sorted_levels:
            cumulative_volume += volume
            value_area_prices.append(price)
            if cumulative_volume >= value_area_volume:
                break
        
        vah = max(value_area_prices) if value_area_prices else poc
        val = min(value_area_prices) if value_area_prices else poc
        
        return VolumeProfile(
            price_levels=dict(volume_by_level),
            poc=poc,
            vah=vah,
            val=val,
            total_volume=total_volume,
            timestamp=datetime.now()
        )


class MarketDataCollector:
    """Main market data collector class"""
    
    def __init__(self):
        self.validator = DataValidator()
        self.volume_calculator = VolumeProfileCalculator()
        self.is_collecting = False
        self.collected_data = {
            'candles': defaultdict(list),
            'orderbooks': defaultdict(list),
            'trades': defaultdict(list),
            'volume_profiles': defaultdict(list)
        }
        self.trade_buffers = defaultdict(lambda: deque(maxlen=1000))
        
        # Websocket connections
        self.websocket_connections = {}
        self.reconnect_attempts = defaultdict(int)
        self.max_reconnect_attempts = 5
    
    async def start_collection(self, symbols: List[str], timeframes: List[str] = None):
        """Start collecting market data for specified symbols"""
        if timeframes is None:
            timeframes = ['1m', '5m', '15m', '1h', '1d']
        
        logger.info(f"Starting market data collection for {len(symbols)} symbols")
        self.is_collecting = True
        
        # Start collection tasks
        tasks = []
        
        # Historical data collection
        for symbol in symbols:
            for timeframe in timeframes:
                task = asyncio.create_task(
                    self.collect_candle_data(symbol, timeframe)
                )
                tasks.append(task)
        
        # Real-time data collection
        for symbol in symbols:
            tasks.extend([
                asyncio.create_task(self.collect_orderbook_data(symbol)),
                asyncio.create_task(self.collect_trade_data(symbol))
            ])
        
        # Volume profile calculation
        tasks.append(asyncio.create_task(self.calculate_volume_profiles()))
        
        try:
            await asyncio.gather(*tasks, return_exceptions=True)
        except Exception as e:
            logger.error(f"Collection error: {e}")
        finally:
            self.is_collecting = False
    
    async def collect_candle_data(self, symbol: str, timeframe: str):
        """Collect historical and real-time candle data"""
        while self.is_collecting:
            try:
                # Get data from all available exchanges
                for exchange_type in [ExchangeType.UPBIT, ExchangeType.KIS, ExchangeType.ALPACA]:
                    try:
                        candles = await api_manager.get_market_data(
                            symbol=symbol,
                            timeframe=timeframe,
                            limit=100,
                            exchange=exchange_type
                        )
                        
                        # Validate and filter candles
                        valid_candles = []
                        for candle in candles:
                            if self.validator.validate_candle(candle):
                                valid_candles.append(candle)
                        
                        if valid_candles:
                            key = f"{symbol}_{timeframe}_{exchange_type.value}"
                            self.collected_data['candles'][key].extend(valid_candles)
                            
                            # Keep only recent data (sliding window)
                            max_candles = 1000
                            if len(self.collected_data['candles'][key]) > max_candles:
                                self.collected_data['candles'][key] = \
                                    self.collected_data['candles'][key][-max_candles:]
                            
                            logger.debug(f"Collected {len(valid_candles)} candles for {key}")
                    
                    except Exception as e:
                        logger.warning(f"Failed to collect candles from {exchange_type.value}: {e}")
                
                # Sleep based on timeframe
                sleep_time = self._get_sleep_time(timeframe)
                await asyncio.sleep(sleep_time)
                
            except Exception as e:
                logger.error(f"Candle collection error for {symbol}_{timeframe}: {e}")
                await asyncio.sleep(60)  # Wait before retry
    
    async def collect_orderbook_data(self, symbol: str):
        """Collect real-time orderbook data"""
        while self.is_collecting:
            try:
                for exchange_type in [ExchangeType.UPBIT, ExchangeType.KIS, ExchangeType.ALPACA]:
                    try:
                        # Get current price (includes bid/ask)
                        tick_data = await api_manager.get_current_price(symbol, exchange_type)
                        
                        if tick_data.bid and tick_data.ask:
                            orderbook = OrderbookData(
                                symbol=symbol,
                                exchange=exchange_type,
                                timestamp=tick_data.timestamp,
                                bids=[(tick_data.bid, tick_data.bid_size or Decimal('0'))],
                                asks=[(tick_data.ask, tick_data.ask_size or Decimal('0'))],
                                spread=tick_data.ask - tick_data.bid,
                                mid_price=(tick_data.bid + tick_data.ask) / 2
                            )
                            
                            key = f"{symbol}_{exchange_type.value}"
                            self.collected_data['orderbooks'][key].append(orderbook)
                            
                            # Keep only recent orderbooks
                            max_orderbooks = 100
                            if len(self.collected_data['orderbooks'][key]) > max_orderbooks:
                                self.collected_data['orderbooks'][key] = \
                                    self.collected_data['orderbooks'][key][-max_orderbooks:]
                    
                    except Exception as e:
                        logger.warning(f"Failed to collect orderbook from {exchange_type.value}: {e}")
                
                await asyncio.sleep(5)  # Collect every 5 seconds
                
            except Exception as e:
                logger.error(f"Orderbook collection error for {symbol}: {e}")
                await asyncio.sleep(30)
    
    async def collect_trade_data(self, symbol: str):
        """Collect real-time trade data"""
        while self.is_collecting:
            try:
                for exchange_type in [ExchangeType.UPBIT, ExchangeType.KIS, ExchangeType.ALPACA]:
                    try:
                        # Get recent trades (using tick data as proxy)
                        tick_data = await api_manager.get_current_price(symbol, exchange_type)
                        
                        # Create trade data from tick
                        trade = TradeData(
                            symbol=symbol,
                            exchange=exchange_type,
                            timestamp=tick_data.timestamp,
                            price=tick_data.price,
                            size=tick_data.size,
                            side='unknown',  # Would need actual trade data
                            trade_id=f"{exchange_type.value}_{tick_data.timestamp.timestamp()}"
                        )
                        
                        # Validate trade
                        if (self.validator.validate_price(trade.price, exchange_type) and
                            self.validator.validate_volume(trade.size, exchange_type)):
                            
                            key = f"{symbol}_{exchange_type.value}"
                            self.collected_data['trades'][key].append(trade)
                            self.trade_buffers[key].append(trade)
                    
                    except Exception as e:
                        logger.warning(f"Failed to collect trades from {exchange_type.value}: {e}")
                
                await asyncio.sleep(2)  # Collect every 2 seconds
                
            except Exception as e:
                logger.error(f"Trade collection error for {symbol}: {e}")
                await asyncio.sleep(30)
    
    async def calculate_volume_profiles(self):
        """Calculate volume profiles from collected trade data"""
        while self.is_collecting:
            try:
                for key, trades in self.trade_buffers.items():
                    if len(trades) >= 10:  # Minimum trades for profile
                        try:
                            profile = self.volume_calculator.calculate(list(trades))
                            self.collected_data['volume_profiles'][key].append(profile)
                            
                            # Keep only recent profiles
                            max_profiles = 50
                            if len(self.collected_data['volume_profiles'][key]) > max_profiles:
                                self.collected_data['volume_profiles'][key] = \
                                    self.collected_data['volume_profiles'][key][-max_profiles:]
                        
                        except Exception as e:
                            logger.warning(f"Volume profile calculation error for {key}: {e}")
                
                await asyncio.sleep(300)  # Calculate every 5 minutes
                
            except Exception as e:
                logger.error(f"Volume profile calculation error: {e}")
                await asyncio.sleep(60)
    
    def _get_sleep_time(self, timeframe: str) -> int:
        """Get appropriate sleep time based on timeframe"""
        timeframe_mapping = {
            '1m': 60,
            '5m': 300,
            '15m': 900,
            '30m': 1800,
            '1h': 3600,
            '4h': 14400,
            '1d': 86400
        }
        return timeframe_mapping.get(timeframe, 300)
    
    async def get_latest_data(self, symbol: str, exchange: ExchangeType, data_type: str) -> List[Any]:
        """Get latest collected data for a symbol"""
        key = f"{symbol}_{exchange.value}"
        
        if data_type in self.collected_data:
            return self.collected_data[data_type].get(key, [])
        
        return []
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics"""
        stats = {
            'is_collecting': self.is_collecting,
            'data_counts': {},
            'last_update': datetime.now()
        }
        
        for data_type, data_dict in self.collected_data.items():
            stats['data_counts'][data_type] = {
                key: len(data_list) for key, data_list in data_dict.items()
            }
        
        return stats
    
    async def stop_collection(self):
        """Stop data collection"""
        logger.info("Stopping market data collection")
        self.is_collecting = False
        
        # Close websocket connections
        for ws in self.websocket_connections.values():
            try:
                await ws.close()
            except:
                pass
        
        self.websocket_connections.clear()


# Global collector instance
market_data_collector = MarketDataCollector()


# Usage example functions
async def collect_crypto_data():
    """Example: Collect crypto data from Upbit"""
    symbols = ['KRW-BTC', 'KRW-ETH', 'KRW-XRP', 'KRW-ADA']
    timeframes = ['1m', '5m', '1h', '1d']
    
    await market_data_collector.start_collection(symbols, timeframes)


async def collect_stock_data():
    """Example: Collect stock data"""
    korean_stocks = ['005930', '000660', '035420']  # Samsung, SK Hynix, NAVER
    us_stocks = ['AAPL', 'GOOGL', 'TSLA']
    
    # Collect from multiple markets
    all_symbols = korean_stocks + us_stocks
    await market_data_collector.start_collection(all_symbols, ['1m', '5m', '1d'])


if __name__ == "__main__":
    # Example usage
    asyncio.run(collect_crypto_data())