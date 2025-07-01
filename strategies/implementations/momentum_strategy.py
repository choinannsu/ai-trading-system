"""
Momentum Strategy Implementation
Trading strategy based on price momentum and trend following
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from ..base_strategy import BaseStrategy, StrategyConfig, Signal, SignalType
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class MomentumConfig(StrategyConfig):
    """Configuration for momentum strategy"""
    name: str = "MomentumStrategy"
    
    # Momentum parameters
    momentum_window: int = 20  # Period for momentum calculation
    price_window: int = 10     # Period for price change calculation
    volume_window: int = 20    # Period for volume confirmation
    
    # Signal thresholds
    momentum_threshold: float = 0.02   # 2% momentum threshold
    volume_threshold: float = 1.5      # Volume multiplier threshold
    rsi_overbought: float = 70         # RSI overbought level
    rsi_oversold: float = 30           # RSI oversold level
    
    # Trend confirmation
    use_moving_average: bool = True
    ma_short_period: int = 10
    ma_long_period: int = 50
    
    # Risk management
    momentum_decay_factor: float = 0.95  # How quickly momentum signals decay
    max_holding_period: int = 30         # Max days to hold position
    
    # Position sizing
    volatility_adjustment: bool = True
    base_position_size: float = 0.1     # Base position size (10%)
    max_volatility: float = 0.05        # Max volatility for full position


class MomentumStrategy(BaseStrategy):
    """
    Momentum trading strategy that identifies and follows price trends
    
    The strategy uses multiple momentum indicators:
    - Price momentum over different timeframes
    - Volume confirmation 
    - RSI for overbought/oversold conditions
    - Moving average trend confirmation
    """
    
    def __init__(self, config: MomentumConfig):
        super().__init__(config)
        self.config: MomentumConfig = config
        
        # Technical indicators cache
        self.indicators_cache: Dict[str, Dict[str, np.ndarray]] = {}
        self.last_update_time: Optional[datetime] = None
        
        logger.info(f"Initialized momentum strategy with {config.momentum_window}d momentum window")
    
    def generate_signals(self, data: pd.DataFrame, timestamp: datetime) -> List[Signal]:
        """Generate momentum-based trading signals"""
        signals = []
        
        if len(data) < max(self.config.momentum_window, self.config.ma_long_period):
            logger.warning("Insufficient data for momentum calculation")
            return signals
        
        # Update indicators for all symbols
        self._update_indicators(data, timestamp)
        
        for symbol in self.config.symbols:
            if symbol not in data.columns or f"{symbol}_close" not in data.columns:
                continue
            
            try:
                signal = self._generate_symbol_signal(symbol, data, timestamp)
                if signal:
                    signals.append(signal)
            except Exception as e:
                logger.error(f"Error generating signal for {symbol}: {e}")
        
        return signals
    
    def _update_indicators(self, data: pd.DataFrame, timestamp: datetime):
        """Update technical indicators for all symbols"""
        self.last_update_time = timestamp
        
        for symbol in self.config.symbols:
            if f"{symbol}_close" not in data.columns:
                continue
            
            prices = data[f"{symbol}_close"].values
            volumes = data.get(f"{symbol}_volume", pd.Series(np.ones(len(prices)))).values
            
            if symbol not in self.indicators_cache:
                self.indicators_cache[symbol] = {}
            
            indicators = self.indicators_cache[symbol]
            
            # Price momentum
            indicators['momentum'] = self._calculate_momentum(prices)
            
            # Volume indicators
            indicators['volume_ratio'] = self._calculate_volume_ratio(volumes)
            
            # RSI
            indicators['rsi'] = self._calculate_rsi(prices)
            
            # Moving averages
            if self.config.use_moving_average:
                indicators['ma_short'] = self._calculate_sma(prices, self.config.ma_short_period)
                indicators['ma_long'] = self._calculate_sma(prices, self.config.ma_long_period)
            
            # Volatility
            if self.config.volatility_adjustment:
                indicators['volatility'] = self._calculate_volatility(prices)
    
    def _generate_symbol_signal(self, symbol: str, data: pd.DataFrame, 
                               timestamp: datetime) -> Optional[Signal]:
        """Generate signal for a specific symbol"""
        if symbol not in self.indicators_cache:
            return None
        
        indicators = self.indicators_cache[symbol]
        current_price = data[f"{symbol}_close"].iloc[-1]
        
        # Get current indicator values
        momentum = indicators['momentum'][-1] if len(indicators['momentum']) > 0 else 0
        volume_ratio = indicators['volume_ratio'][-1] if len(indicators['volume_ratio']) > 0 else 1
        rsi = indicators['rsi'][-1] if len(indicators['rsi']) > 0 else 50
        
        # Moving average trend
        trend_bullish = True
        if self.config.use_moving_average and 'ma_short' in indicators and 'ma_long' in indicators:
            ma_short = indicators['ma_short'][-1] if len(indicators['ma_short']) > 0 else current_price
            ma_long = indicators['ma_long'][-1] if len(indicators['ma_long']) > 0 else current_price
            trend_bullish = ma_short > ma_long
        
        # Check if we already have a position
        existing_position = symbol in self.positions
        
        # Generate signals based on momentum conditions
        signal_type = SignalType.HOLD
        confidence = 0.0
        
        # Buy signal conditions
        if (momentum > self.config.momentum_threshold and 
            volume_ratio > self.config.volume_threshold and
            rsi < self.config.rsi_overbought and
            trend_bullish and
            not existing_position):
            
            signal_type = SignalType.BUY
            confidence = min(1.0, (momentum / self.config.momentum_threshold) * 
                           (volume_ratio / self.config.volume_threshold))
        
        # Sell signal conditions
        elif (momentum < -self.config.momentum_threshold and
              volume_ratio > self.config.volume_threshold and
              rsi > self.config.rsi_oversold and
              not trend_bullish and
              not existing_position):
            
            signal_type = SignalType.SELL
            confidence = min(1.0, abs(momentum / self.config.momentum_threshold) * 
                           (volume_ratio / self.config.volume_threshold))
        
        # Exit conditions for existing positions
        elif existing_position:
            position = self.positions[symbol]
            days_held = (timestamp - position.entry_time).days
            
            # Exit long position
            if (position.position_type.value == "long" and
                (momentum < -self.config.momentum_threshold / 2 or
                 rsi > self.config.rsi_overbought or
                 not trend_bullish or
                 days_held > self.config.max_holding_period)):
                
                signal_type = SignalType.SELL
                confidence = 0.8
            
            # Exit short position  
            elif (position.position_type.value == "short" and
                  (momentum > self.config.momentum_threshold / 2 or
                   rsi < self.config.rsi_oversold or
                   trend_bullish or
                   days_held > self.config.max_holding_period)):
                
                signal_type = SignalType.CLOSE_SHORT
                confidence = 0.8
        
        # Create signal if conditions are met
        if signal_type != SignalType.HOLD and confidence > self.config.signal_threshold:
            # Calculate position size based on volatility
            position_size = self._calculate_momentum_position_size(symbol, confidence)
            
            return Signal(
                symbol=symbol,
                signal_type=signal_type,
                timestamp=timestamp,
                price=current_price,
                confidence=confidence,
                size=position_size,
                metadata={
                    "momentum": momentum,
                    "volume_ratio": volume_ratio,
                    "rsi": rsi,
                    "trend_bullish": trend_bullish
                }
            )
        
        return None
    
    def _calculate_momentum(self, prices: np.ndarray) -> np.ndarray:
        """Calculate price momentum"""
        if len(prices) < self.config.momentum_window:
            return np.array([0.0])
        
        momentum = np.zeros(len(prices))
        
        for i in range(self.config.momentum_window, len(prices)):
            # Calculate momentum as percentage change over window
            current_price = prices[i]
            past_price = prices[i - self.config.momentum_window]
            momentum[i] = (current_price - past_price) / past_price
        
        return momentum
    
    def _calculate_volume_ratio(self, volumes: np.ndarray) -> np.ndarray:
        """Calculate volume ratio relative to average"""
        if len(volumes) < self.config.volume_window:
            return np.ones(len(volumes))
        
        volume_ratio = np.ones(len(volumes))
        
        for i in range(self.config.volume_window, len(volumes)):
            current_volume = volumes[i]
            avg_volume = np.mean(volumes[i - self.config.volume_window:i])
            volume_ratio[i] = current_volume / avg_volume if avg_volume > 0 else 1.0
        
        return volume_ratio
    
    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> np.ndarray:
        """Calculate Relative Strength Index"""
        if len(prices) < period + 1:
            return np.full(len(prices), 50.0)
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        rsi = np.full(len(prices), 50.0)
        
        # Calculate initial average gains and losses
        avg_gain = np.mean(gains[:period])
        avg_loss = np.mean(losses[:period])
        
        for i in range(period, len(prices) - 1):
            # Exponential moving average for gains and losses
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period
            
            if avg_loss == 0:
                rsi[i + 1] = 100
            else:
                rs = avg_gain / avg_loss
                rsi[i + 1] = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _calculate_sma(self, prices: np.ndarray, period: int) -> np.ndarray:
        """Calculate Simple Moving Average"""
        if len(prices) < period:
            return np.full(len(prices), np.mean(prices))
        
        sma = np.full(len(prices), np.nan)
        
        for i in range(period - 1, len(prices)):
            sma[i] = np.mean(prices[i - period + 1:i + 1])
        
        # Fill initial NaN values with current price
        sma[:period - 1] = prices[:period - 1]
        
        return sma
    
    def _calculate_volatility(self, prices: np.ndarray, period: int = 20) -> np.ndarray:
        """Calculate rolling volatility"""
        if len(prices) < 2:
            return np.zeros(len(prices))
        
        returns = np.diff(prices) / prices[:-1]
        volatility = np.zeros(len(prices))
        
        for i in range(period, len(prices)):
            volatility[i] = np.std(returns[i - period:i]) * np.sqrt(252)  # Annualized
        
        # Fill initial values
        if len(returns) > 0:
            initial_vol = np.std(returns[:period]) * np.sqrt(252) if period <= len(returns) else 0.02
            volatility[:period] = initial_vol
        
        return volatility
    
    def _calculate_momentum_position_size(self, symbol: str, confidence: float) -> float:
        """Calculate position size based on momentum and volatility"""
        base_size = self.config.base_position_size
        
        # Adjust for confidence
        size = base_size * confidence
        
        # Adjust for volatility if enabled
        if self.config.volatility_adjustment and symbol in self.indicators_cache:
            indicators = self.indicators_cache[symbol]
            if 'volatility' in indicators and len(indicators['volatility']) > 0:
                current_vol = indicators['volatility'][-1]
                vol_adjustment = min(1.0, self.config.max_volatility / (current_vol + 1e-6))
                size *= vol_adjustment
        
        return min(size, self.config.max_position_size)
    
    def calculate_position_size(self, signal: Signal, current_price: float, 
                              available_cash: float) -> float:
        """Calculate position size for momentum strategy"""
        # Get target position size from signal
        target_size = signal.size
        
        # Calculate maximum shares we can buy
        max_value = available_cash * target_size
        
        # Account for transaction costs
        transaction_cost_rate = self.config.commission_rate + self.config.slippage_rate
        effective_cash = max_value / (1 + transaction_cost_rate)
        
        # Calculate shares
        shares = effective_cash / current_price
        
        # Apply minimum trade size constraint
        min_shares = self.config.min_trade_size / current_price
        if shares < min_shares:
            return 0.0
        
        return shares
    
    def get_strategy_metrics(self) -> Dict[str, Any]:
        """Get momentum-specific strategy metrics"""
        base_metrics = self.get_performance_metrics()
        
        # Calculate momentum-specific metrics
        momentum_metrics = {}
        
        if self.indicators_cache:
            # Average momentum across symbols
            total_momentum = 0
            symbol_count = 0
            
            for symbol, indicators in self.indicators_cache.items():
                if 'momentum' in indicators and len(indicators['momentum']) > 0:
                    total_momentum += indicators['momentum'][-1]
                    symbol_count += 1
            
            if symbol_count > 0:
                momentum_metrics['avg_momentum'] = total_momentum / symbol_count
        
        # Calculate trend consistency
        if self.trades:
            momentum_trades = [t for t in self.trades if 'momentum' in t.metadata]
            if momentum_trades:
                avg_momentum_at_entry = np.mean([t.metadata['momentum'] for t in momentum_trades])
                momentum_metrics['avg_momentum_at_entry'] = avg_momentum_at_entry
        
        return {**base_metrics, **momentum_metrics}


# Export classes
__all__ = ['MomentumStrategy', 'MomentumConfig']