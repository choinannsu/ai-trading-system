"""
Advanced Stop-Loss Management System
Comprehensive stop-loss strategies including trailing stops, time-based stops,
correlation-based stops, and volatility-adjusted stops
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import warnings

from utils.logger import get_logger

logger = get_logger(__name__)


class StopType(Enum):
    """Types of stop-loss orders"""
    FIXED = "fixed"
    TRAILING = "trailing"
    TIME_BASED = "time_based"
    VOLATILITY_ADJUSTED = "volatility_adjusted"
    CORRELATION_BASED = "correlation_based"
    PROFIT_TARGET = "profit_target"
    COMBINED = "combined"


class StopStatus(Enum):
    """Status of stop-loss orders"""
    ACTIVE = "active"
    TRIGGERED = "triggered"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


@dataclass
class StopLossConfig:
    """Configuration for stop-loss management"""
    # General parameters
    default_stop_type: StopType = StopType.TRAILING
    enable_multiple_stops: bool = True
    stop_update_frequency: str = "tick"  # "tick", "minute", "hour", "daily"
    
    # Fixed stop parameters
    fixed_stop_pct: float = 0.05  # 5% fixed stop
    
    # Trailing stop parameters
    trailing_stop_pct: float = 0.03  # 3% trailing stop
    trailing_activation_pct: float = 0.02  # Activate after 2% profit
    trailing_step_size: float = 0.005  # 0.5% step size for adjustments
    
    # Time-based stop parameters
    max_holding_period: int = 30  # Maximum holding days
    time_decay_factor: float = 0.1  # Daily decay in stop distance
    
    # Volatility-adjusted parameters
    volatility_window: int = 20  # Window for volatility calculation
    volatility_multiplier: float = 2.0  # ATR multiplier for stop distance
    volatility_adjustment_speed: float = 0.1  # Speed of volatility adjustments
    
    # Correlation-based parameters
    correlation_window: int = 60  # Window for correlation calculation
    correlation_threshold: float = 0.7  # High correlation threshold
    correlation_stop_multiplier: float = 1.5  # Multiplier when correlated assets decline
    
    # Profit target parameters
    profit_target_ratio: float = 2.0  # Risk-reward ratio (2:1)
    profit_target_adjustment: bool = True  # Adjust targets based on volatility
    
    # Advanced features
    weekend_gap_protection: bool = True
    news_event_protection: bool = False
    liquidity_adjustment: bool = True


@dataclass
class StopLossOrder:
    """Stop-loss order representation"""
    symbol: str
    stop_type: StopType
    stop_price: float
    entry_price: float
    entry_time: datetime
    quantity: float
    status: StopStatus = StopStatus.ACTIVE
    
    # Trailing stop specific
    highest_price: Optional[float] = None
    trailing_distance: Optional[float] = None
    
    # Time-based specific
    expiry_time: Optional[datetime] = None
    
    # Volatility-adjusted specific
    atr_multiple: Optional[float] = None
    
    # Profit target
    profit_target: Optional[float] = None
    
    # Metadata
    last_update: datetime = field(default_factory=datetime.now)
    trigger_reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class FixedStopLoss:
    """
    Fixed stop-loss implementation
    
    Features:
    - Simple percentage-based stops
    - Support for both long and short positions
    - Immediate activation upon position entry
    """
    
    def __init__(self, config: StopLossConfig):
        self.config = config
    
    def create_stop(self, symbol: str, entry_price: float, quantity: float,
                   entry_time: datetime, stop_pct: Optional[float] = None) -> StopLossOrder:
        """Create fixed stop-loss order"""
        
        stop_percentage = stop_pct or self.config.fixed_stop_pct
        
        # Calculate stop price based on position direction
        if quantity > 0:  # Long position
            stop_price = entry_price * (1 - stop_percentage)
        else:  # Short position
            stop_price = entry_price * (1 + stop_percentage)
        
        return StopLossOrder(
            symbol=symbol,
            stop_type=StopType.FIXED,
            stop_price=stop_price,
            entry_price=entry_price,
            entry_time=entry_time,
            quantity=quantity,
            metadata={'stop_percentage': stop_percentage}
        )
    
    def check_trigger(self, stop_order: StopLossOrder, current_price: float) -> bool:
        """Check if fixed stop should be triggered"""
        
        if stop_order.status != StopStatus.ACTIVE:
            return False
        
        # Long position: trigger if price falls below stop
        if stop_order.quantity > 0:
            return current_price <= stop_order.stop_price
        # Short position: trigger if price rises above stop
        else:
            return current_price >= stop_order.stop_price


class TrailingStopLoss:
    """
    Trailing stop-loss implementation
    
    Features:
    - Dynamic stop adjustment based on favorable price movement
    - Activation threshold to prevent premature triggering
    - Configurable step size for stop adjustments
    """
    
    def __init__(self, config: StopLossConfig):
        self.config = config
    
    def create_stop(self, symbol: str, entry_price: float, quantity: float,
                   entry_time: datetime, trailing_pct: Optional[float] = None) -> StopLossOrder:
        """Create trailing stop-loss order"""
        
        trailing_percentage = trailing_pct or self.config.trailing_stop_pct
        
        # Initial stop price (before activation)
        if quantity > 0:  # Long position
            initial_stop = entry_price * (1 - trailing_percentage)
            highest_price = entry_price
        else:  # Short position
            initial_stop = entry_price * (1 + trailing_percentage)
            highest_price = entry_price
        
        return StopLossOrder(
            symbol=symbol,
            stop_type=StopType.TRAILING,
            stop_price=initial_stop,
            entry_price=entry_price,
            entry_time=entry_time,
            quantity=quantity,
            highest_price=highest_price,
            trailing_distance=trailing_percentage,
            metadata={'activation_threshold': self.config.trailing_activation_pct}
        )
    
    def update_stop(self, stop_order: StopLossOrder, current_price: float) -> bool:
        """Update trailing stop based on current price"""
        
        if stop_order.status != StopStatus.ACTIVE:
            return False
        
        updated = False
        
        if stop_order.quantity > 0:  # Long position
            # Update highest price seen
            if current_price > stop_order.highest_price:
                stop_order.highest_price = current_price
                
                # Check if activation threshold is met
                profit_pct = (current_price - stop_order.entry_price) / stop_order.entry_price
                
                if profit_pct >= self.config.trailing_activation_pct:
                    # Calculate new stop price
                    new_stop = current_price * (1 - stop_order.trailing_distance)
                    
                    # Only move stop up (never down for long positions)
                    if new_stop > stop_order.stop_price:
                        stop_order.stop_price = new_stop
                        stop_order.last_update = datetime.now()
                        updated = True
        
        else:  # Short position
            # Update lowest price seen (stored as highest_price for consistency)
            if current_price < stop_order.highest_price:
                stop_order.highest_price = current_price
                
                # Check if activation threshold is met
                profit_pct = (stop_order.entry_price - current_price) / stop_order.entry_price
                
                if profit_pct >= self.config.trailing_activation_pct:
                    # Calculate new stop price
                    new_stop = current_price * (1 + stop_order.trailing_distance)
                    
                    # Only move stop down (never up for short positions)
                    if new_stop < stop_order.stop_price:
                        stop_order.stop_price = new_stop
                        stop_order.last_update = datetime.now()
                        updated = True
        
        return updated
    
    def check_trigger(self, stop_order: StopLossOrder, current_price: float) -> bool:
        """Check if trailing stop should be triggered"""
        
        if stop_order.status != StopStatus.ACTIVE:
            return False
        
        # Same logic as fixed stop for triggering
        if stop_order.quantity > 0:
            return current_price <= stop_order.stop_price
        else:
            return current_price >= stop_order.stop_price


class TimeBasedStopLoss:
    """
    Time-based stop-loss implementation
    
    Features:
    - Maximum holding period enforcement
    - Time decay of stop distance
    - Weekend gap protection
    """
    
    def __init__(self, config: StopLossConfig):
        self.config = config
    
    def create_stop(self, symbol: str, entry_price: float, quantity: float,
                   entry_time: datetime, max_holding_days: Optional[int] = None) -> StopLossOrder:
        """Create time-based stop-loss order"""
        
        holding_period = max_holding_days or self.config.max_holding_period
        expiry_time = entry_time + timedelta(days=holding_period)
        
        # Initial stop based on fixed percentage
        if quantity > 0:
            initial_stop = entry_price * (1 - self.config.fixed_stop_pct)
        else:
            initial_stop = entry_price * (1 + self.config.fixed_stop_pct)
        
        return StopLossOrder(
            symbol=symbol,
            stop_type=StopType.TIME_BASED,
            stop_price=initial_stop,
            entry_price=entry_price,
            entry_time=entry_time,
            quantity=quantity,
            expiry_time=expiry_time,
            metadata={'max_holding_days': holding_period}
        )
    
    def update_stop(self, stop_order: StopLossOrder, current_time: datetime) -> bool:
        """Update time-based stop based on time decay"""
        
        if stop_order.status != StopStatus.ACTIVE:
            return False
        
        # Check if position has expired
        if current_time >= stop_order.expiry_time:
            stop_order.status = StopStatus.EXPIRED
            stop_order.trigger_reason = "time_expiry"
            return True
        
        # Calculate time decay adjustment
        time_elapsed = (current_time - stop_order.entry_time).days
        max_days = self.config.max_holding_period
        
        if time_elapsed > 0:
            # Gradually tighten stop as time passes
            decay_factor = 1 - (time_elapsed / max_days) * self.config.time_decay_factor
            
            if stop_order.quantity > 0:  # Long position
                # Move stop closer to current entry price
                new_stop = stop_order.entry_price * (1 - self.config.fixed_stop_pct * decay_factor)
                if new_stop > stop_order.stop_price:
                    stop_order.stop_price = new_stop
                    stop_order.last_update = current_time
                    return True
            else:  # Short position
                new_stop = stop_order.entry_price * (1 + self.config.fixed_stop_pct * decay_factor)
                if new_stop < stop_order.stop_price:
                    stop_order.stop_price = new_stop
                    stop_order.last_update = current_time
                    return True
        
        return False
    
    def check_trigger(self, stop_order: StopLossOrder, current_price: float,
                     current_time: datetime) -> bool:
        """Check if time-based stop should be triggered"""
        
        if stop_order.status != StopStatus.ACTIVE:
            return False
        
        # Check time expiry
        if current_time >= stop_order.expiry_time:
            stop_order.trigger_reason = "time_expiry"
            return True
        
        # Check price trigger
        if stop_order.quantity > 0:
            triggered = current_price <= stop_order.stop_price
        else:
            triggered = current_price >= stop_order.stop_price
        
        if triggered:
            stop_order.trigger_reason = "price_trigger"
        
        return triggered


class VolatilityAdjustedStopLoss:
    """
    Volatility-adjusted stop-loss implementation
    
    Features:
    - ATR-based stop distance calculation
    - Dynamic adjustment based on changing volatility
    - Multiple volatility estimators
    """
    
    def __init__(self, config: StopLossConfig):
        self.config = config
        self.volatility_cache: Dict[str, float] = {}
    
    def create_stop(self, symbol: str, entry_price: float, quantity: float,
                   entry_time: datetime, price_data: pd.DataFrame,
                   atr_multiple: Optional[float] = None) -> StopLossOrder:
        """Create volatility-adjusted stop-loss order"""
        
        atr_mult = atr_multiple or self.config.volatility_multiplier
        
        # Calculate ATR
        atr = self._calculate_atr(price_data, self.config.volatility_window)
        self.volatility_cache[symbol] = atr
        
        # Calculate stop distance based on ATR
        stop_distance = atr * atr_mult
        
        if quantity > 0:  # Long position
            stop_price = entry_price - stop_distance
        else:  # Short position
            stop_price = entry_price + stop_distance
        
        return StopLossOrder(
            symbol=symbol,
            stop_type=StopType.VOLATILITY_ADJUSTED,
            stop_price=stop_price,
            entry_price=entry_price,
            entry_time=entry_time,
            quantity=quantity,
            atr_multiple=atr_mult,
            metadata={'atr': atr, 'stop_distance': stop_distance}
        )
    
    def update_stop(self, stop_order: StopLossOrder, price_data: pd.DataFrame,
                   current_price: float) -> bool:
        """Update volatility-adjusted stop based on current volatility"""
        
        if stop_order.status != StopStatus.ACTIVE:
            return False
        
        # Recalculate ATR
        new_atr = self._calculate_atr(price_data, self.config.volatility_window)
        old_atr = self.volatility_cache.get(stop_order.symbol, new_atr)
        
        # Smooth volatility changes
        smoothed_atr = (old_atr * (1 - self.config.volatility_adjustment_speed) +
                       new_atr * self.config.volatility_adjustment_speed)
        
        self.volatility_cache[stop_order.symbol] = smoothed_atr
        
        # Calculate new stop distance
        new_stop_distance = smoothed_atr * stop_order.atr_multiple
        
        if stop_order.quantity > 0:  # Long position
            new_stop = current_price - new_stop_distance
            # Only move stop up (tighter) if volatility decreased, or if price moved favorably
            if new_stop > stop_order.stop_price:
                stop_order.stop_price = new_stop
                stop_order.metadata['atr'] = smoothed_atr
                stop_order.metadata['stop_distance'] = new_stop_distance
                stop_order.last_update = datetime.now()
                return True
        else:  # Short position
            new_stop = current_price + new_stop_distance
            if new_stop < stop_order.stop_price:
                stop_order.stop_price = new_stop
                stop_order.metadata['atr'] = smoothed_atr
                stop_order.metadata['stop_distance'] = new_stop_distance
                stop_order.last_update = datetime.now()
                return True
        
        return False
    
    def _calculate_atr(self, price_data: pd.DataFrame, window: int) -> float:
        """Calculate Average True Range"""
        
        if len(price_data) < window:
            # Fallback to simple volatility
            if 'close' in price_data.columns:
                returns = price_data['close'].pct_change().dropna()
                return returns.std() * price_data['close'].iloc[-1] if len(returns) > 1 else 0.01
            return 0.01
        
        # Extract OHLC data
        high_col = next((col for col in price_data.columns if 'high' in col.lower()), None)
        low_col = next((col for col in price_data.columns if 'low' in col.lower()), None)
        close_col = next((col for col in price_data.columns if 'close' in col.lower()), None)
        
        if not all([high_col, low_col, close_col]):
            # Fallback to close price volatility
            if close_col:
                returns = price_data[close_col].pct_change().dropna()
                return returns.std() * price_data[close_col].iloc[-1] if len(returns) > 1 else 0.01
            return 0.01
        
        high = price_data[high_col]
        low = price_data[low_col]
        close = price_data[close_col]
        prev_close = close.shift(1)
        
        # Calculate True Range
        tr1 = high - low
        tr2 = abs(high - prev_close)
        tr3 = abs(low - prev_close)
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Calculate ATR using exponential moving average
        atr = true_range.ewm(span=window).mean().iloc[-1]
        
        return atr if not pd.isna(atr) else 0.01
    
    def check_trigger(self, stop_order: StopLossOrder, current_price: float) -> bool:
        """Check if volatility-adjusted stop should be triggered"""
        
        if stop_order.status != StopStatus.ACTIVE:
            return False
        
        if stop_order.quantity > 0:
            return current_price <= stop_order.stop_price
        else:
            return current_price >= stop_order.stop_price


class CorrelationBasedStopLoss:
    """
    Correlation-based stop-loss implementation
    
    Features:
    - Adjust stops based on correlated asset performance
    - Sector/style momentum detection
    - Contagion risk management
    """
    
    def __init__(self, config: StopLossConfig):
        self.config = config
        self.correlation_cache: Dict[str, Dict[str, float]] = {}
    
    def create_stop(self, symbol: str, entry_price: float, quantity: float,
                   entry_time: datetime, related_assets: List[str],
                   correlation_data: pd.DataFrame) -> StopLossOrder:
        """Create correlation-based stop-loss order"""
        
        # Calculate correlations with related assets
        correlations = self._calculate_correlations(symbol, related_assets, correlation_data)
        self.correlation_cache[symbol] = correlations
        
        # Initial stop based on fixed percentage
        if quantity > 0:
            initial_stop = entry_price * (1 - self.config.fixed_stop_pct)
        else:
            initial_stop = entry_price * (1 + self.config.fixed_stop_pct)
        
        return StopLossOrder(
            symbol=symbol,
            stop_type=StopType.CORRELATION_BASED,
            stop_price=initial_stop,
            entry_price=entry_price,
            entry_time=entry_time,
            quantity=quantity,
            metadata={
                'related_assets': related_assets,
                'correlations': correlations
            }
        )
    
    def update_stop(self, stop_order: StopLossOrder, current_price: float,
                   related_prices: Dict[str, float], 
                   correlation_data: pd.DataFrame) -> bool:
        """Update correlation-based stop based on related asset performance"""
        
        if stop_order.status != StopStatus.ACTIVE:
            return False
        
        related_assets = stop_order.metadata.get('related_assets', [])
        correlations = self.correlation_cache.get(stop_order.symbol, {})
        
        # Calculate correlation signal
        correlation_signal = self._calculate_correlation_signal(
            stop_order.symbol, related_assets, related_prices, correlations
        )
        
        # Adjust stop based on correlation signal
        if correlation_signal < -self.config.correlation_threshold:
            # Highly correlated assets are declining, tighten stop
            adjustment_factor = self.config.correlation_stop_multiplier
            
            if stop_order.quantity > 0:  # Long position
                new_stop = stop_order.entry_price * (1 - self.config.fixed_stop_pct * adjustment_factor)
                if new_stop > stop_order.stop_price:
                    stop_order.stop_price = new_stop
                    stop_order.last_update = datetime.now()
                    stop_order.metadata['correlation_adjustment'] = adjustment_factor
                    return True
            else:  # Short position
                new_stop = stop_order.entry_price * (1 + self.config.fixed_stop_pct * adjustment_factor)
                if new_stop < stop_order.stop_price:
                    stop_order.stop_price = new_stop
                    stop_order.last_update = datetime.now()
                    stop_order.metadata['correlation_adjustment'] = adjustment_factor
                    return True
        
        return False
    
    def _calculate_correlations(self, symbol: str, related_assets: List[str],
                              correlation_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate correlations with related assets"""
        
        correlations = {}
        
        if symbol in correlation_data.columns:
            symbol_returns = correlation_data[symbol].dropna()
            
            for related_asset in related_assets:
                if related_asset in correlation_data.columns:
                    related_returns = correlation_data[related_asset].dropna()
                    
                    # Align data
                    aligned_data = pd.concat([symbol_returns, related_returns], axis=1, join='inner')
                    
                    if len(aligned_data) >= self.config.correlation_window:
                        corr = aligned_data.corr().iloc[0, 1]
                        if not pd.isna(corr):
                            correlations[related_asset] = corr
        
        return correlations
    
    def _calculate_correlation_signal(self, symbol: str, related_assets: List[str],
                                    related_prices: Dict[str, float],
                                    correlations: Dict[str, float]) -> float:
        """Calculate correlation-based signal"""
        
        if not related_assets or not correlations:
            return 0.0
        
        weighted_signal = 0.0
        total_weight = 0.0
        
        for asset in related_assets:
            if asset in related_prices and asset in correlations:
                # Simple price change as signal (would be enhanced with returns)
                price_signal = related_prices[asset]  # Normalized price change
                correlation = correlations[asset]
                weight = abs(correlation)
                
                weighted_signal += weight * price_signal * correlation
                total_weight += weight
        
        return weighted_signal / total_weight if total_weight > 0 else 0.0
    
    def check_trigger(self, stop_order: StopLossOrder, current_price: float) -> bool:
        """Check if correlation-based stop should be triggered"""
        
        if stop_order.status != StopStatus.ACTIVE:
            return False
        
        if stop_order.quantity > 0:
            return current_price <= stop_order.stop_price
        else:
            return current_price >= stop_order.stop_price


class StopLossManager:
    """
    Main stop-loss management system
    
    Features:
    - Multiple stop-loss strategies
    - Combined stop-loss logic
    - Real-time monitoring and updates
    - Risk management integration
    """
    
    def __init__(self, config: StopLossConfig = None):
        self.config = config or StopLossConfig()
        
        # Initialize stop-loss implementations
        self.fixed_stop = FixedStopLoss(self.config)
        self.trailing_stop = TrailingStopLoss(self.config)
        self.time_stop = TimeBasedStopLoss(self.config)
        self.volatility_stop = VolatilityAdjustedStopLoss(self.config)
        self.correlation_stop = CorrelationBasedStopLoss(self.config)
        
        # Active stop orders
        self.active_stops: Dict[str, List[StopLossOrder]] = {}
        
        logger.info("Initialized stop-loss manager")
    
    def create_stop_orders(self, symbol: str, entry_price: float, quantity: float,
                          entry_time: datetime, stop_types: Optional[List[StopType]] = None,
                          **kwargs) -> List[StopLossOrder]:
        """Create stop-loss orders for a position"""
        
        if stop_types is None:
            stop_types = [self.config.default_stop_type]
        
        stop_orders = []
        
        for stop_type in stop_types:
            try:
                if stop_type == StopType.FIXED:
                    stop_order = self.fixed_stop.create_stop(
                        symbol, entry_price, quantity, entry_time,
                        kwargs.get('stop_pct')
                    )
                elif stop_type == StopType.TRAILING:
                    stop_order = self.trailing_stop.create_stop(
                        symbol, entry_price, quantity, entry_time,
                        kwargs.get('trailing_pct')
                    )
                elif stop_type == StopType.TIME_BASED:
                    stop_order = self.time_stop.create_stop(
                        symbol, entry_price, quantity, entry_time,
                        kwargs.get('max_holding_days')
                    )
                elif stop_type == StopType.VOLATILITY_ADJUSTED:
                    stop_order = self.volatility_stop.create_stop(
                        symbol, entry_price, quantity, entry_time,
                        kwargs.get('price_data'), kwargs.get('atr_multiple')
                    )
                elif stop_type == StopType.CORRELATION_BASED:
                    stop_order = self.correlation_stop.create_stop(
                        symbol, entry_price, quantity, entry_time,
                        kwargs.get('related_assets', []),
                        kwargs.get('correlation_data')
                    )
                else:
                    continue
                
                stop_orders.append(stop_order)
                
            except Exception as e:
                logger.error(f"Failed to create {stop_type} stop for {symbol}: {e}")
                continue
        
        # Store active stops
        if symbol not in self.active_stops:
            self.active_stops[symbol] = []
        self.active_stops[symbol].extend(stop_orders)
        
        return stop_orders
    
    def update_stops(self, symbol: str, current_price: float, 
                    current_time: datetime = None, **kwargs) -> List[StopLossOrder]:
        """Update all active stops for a symbol"""
        
        if current_time is None:
            current_time = datetime.now()
        
        triggered_stops = []
        
        if symbol not in self.active_stops:
            return triggered_stops
        
        for stop_order in self.active_stops[symbol]:
            if stop_order.status != StopStatus.ACTIVE:
                continue
            
            try:
                # Update stop based on type
                if stop_order.stop_type == StopType.TRAILING:
                    self.trailing_stop.update_stop(stop_order, current_price)
                elif stop_order.stop_type == StopType.TIME_BASED:
                    self.time_stop.update_stop(stop_order, current_time)
                elif stop_order.stop_type == StopType.VOLATILITY_ADJUSTED:
                    if 'price_data' in kwargs:
                        self.volatility_stop.update_stop(
                            stop_order, kwargs['price_data'], current_price
                        )
                elif stop_order.stop_type == StopType.CORRELATION_BASED:
                    if 'related_prices' in kwargs and 'correlation_data' in kwargs:
                        self.correlation_stop.update_stop(
                            stop_order, current_price, kwargs['related_prices'],
                            kwargs['correlation_data']
                        )
                
                # Check if stop is triggered
                triggered = self._check_stop_trigger(stop_order, current_price, current_time)
                
                if triggered:
                    stop_order.status = StopStatus.TRIGGERED
                    stop_order.last_update = current_time
                    triggered_stops.append(stop_order)
                    
            except Exception as e:
                logger.error(f"Error updating stop for {symbol}: {e}")
                continue
        
        return triggered_stops
    
    def _check_stop_trigger(self, stop_order: StopLossOrder, current_price: float,
                           current_time: datetime) -> bool:
        """Check if a stop order should be triggered"""
        
        if stop_order.stop_type == StopType.FIXED:
            return self.fixed_stop.check_trigger(stop_order, current_price)
        elif stop_order.stop_type == StopType.TRAILING:
            return self.trailing_stop.check_trigger(stop_order, current_price)
        elif stop_order.stop_type == StopType.TIME_BASED:
            return self.time_stop.check_trigger(stop_order, current_price, current_time)
        elif stop_order.stop_type == StopType.VOLATILITY_ADJUSTED:
            return self.volatility_stop.check_trigger(stop_order, current_price)
        elif stop_order.stop_type == StopType.CORRELATION_BASED:
            return self.correlation_stop.check_trigger(stop_order, current_price)
        
        return False
    
    def cancel_stops(self, symbol: str, stop_types: Optional[List[StopType]] = None):
        """Cancel active stops for a symbol"""
        
        if symbol not in self.active_stops:
            return
        
        for stop_order in self.active_stops[symbol]:
            if stop_order.status == StopStatus.ACTIVE:
                if stop_types is None or stop_order.stop_type in stop_types:
                    stop_order.status = StopStatus.CANCELLED
                    stop_order.last_update = datetime.now()
    
    def get_active_stops(self, symbol: str = None) -> Dict[str, List[StopLossOrder]]:
        """Get active stop orders"""
        
        if symbol:
            return {symbol: self.active_stops.get(symbol, [])}
        else:
            return self.active_stops.copy()
    
    def get_stop_summary(self, symbol: str = None) -> Dict[str, Any]:
        """Get summary of stop-loss system status"""
        
        summary = {
            'total_active_stops': 0,
            'stops_by_type': {},
            'stops_by_symbol': {},
            'recent_triggers': []
        }
        
        symbols_to_check = [symbol] if symbol else list(self.active_stops.keys())
        
        for sym in symbols_to_check:
            if sym not in self.active_stops:
                continue
            
            symbol_stops = self.active_stops[sym]
            active_count = len([s for s in symbol_stops if s.status == StopStatus.ACTIVE])
            
            summary['stops_by_symbol'][sym] = {
                'active_count': active_count,
                'total_count': len(symbol_stops)
            }
            
            summary['total_active_stops'] += active_count
            
            # Count by type
            for stop in symbol_stops:
                if stop.status == StopStatus.ACTIVE:
                    stop_type = stop.stop_type.value
                    summary['stops_by_type'][stop_type] = summary['stops_by_type'].get(stop_type, 0) + 1
                
                # Recent triggers (last 24 hours)
                if (stop.status == StopStatus.TRIGGERED and 
                    stop.last_update > datetime.now() - timedelta(days=1)):
                    summary['recent_triggers'].append({
                        'symbol': sym,
                        'stop_type': stop.stop_type.value,
                        'trigger_time': stop.last_update,
                        'trigger_reason': stop.trigger_reason
                    })
        
        return summary


# Factory function
def create_stop_loss_manager(config: Dict[str, Any] = None) -> StopLossManager:
    """Create stop-loss manager with configuration"""
    stop_config = StopLossConfig(**(config or {}))
    return StopLossManager(stop_config)


# Export classes and functions
__all__ = [
    'StopType',
    'StopStatus',
    'StopLossConfig',
    'StopLossOrder',
    'FixedStopLoss',
    'TrailingStopLoss',
    'TimeBasedStopLoss',
    'VolatilityAdjustedStopLoss',
    'CorrelationBasedStopLoss',
    'StopLossManager',
    'create_stop_loss_manager'
]