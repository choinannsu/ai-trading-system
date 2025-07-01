"""
Multi-Timeframe Strategy Implementation
Trading strategy that analyzes multiple timeframes for signal confirmation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

from ..base_strategy import BaseStrategy, StrategyConfig, Signal, SignalType
from utils.logger import get_logger

logger = get_logger(__name__)


class TimeframeType(Enum):
    """Different timeframe types"""
    INTRADAY = "intraday"
    DAILY = "daily" 
    WEEKLY = "weekly"
    MONTHLY = "monthly"


class TrendDirection(Enum):
    """Trend direction"""
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


@dataclass
class TimeframeConfig:
    """Configuration for a specific timeframe"""
    name: str
    period_days: int                    # Number of days for this timeframe
    weight: float                       # Weight in final decision
    trend_method: str = "sma"          # "sma", "ema", "linear"
    momentum_method: str = "roc"       # "roc", "macd", "rsi"
    volatility_method: str = "atr"     # "atr", "std", "bollinger"
    
    # Technical indicator parameters
    fast_ma_period: int = 10
    slow_ma_period: int = 20
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    atr_period: int = 14


@dataclass
class MultiTimeframeConfig(StrategyConfig):
    """Configuration for multi-timeframe strategy"""
    name: str = "MultiTimeframeStrategy"
    
    # Timeframe configurations
    timeframes: List[TimeframeConfig] = field(default_factory=lambda: [
        TimeframeConfig("short", 5, 0.2),      # 5-day short-term
        TimeframeConfig("medium", 20, 0.3),    # 20-day medium-term  
        TimeframeConfig("long", 60, 0.5)       # 60-day long-term
    ])
    
    # Signal generation
    min_timeframe_agreement: float = 0.6    # Minimum agreement between timeframes
    trend_confirmation_required: bool = True # Require trend confirmation
    momentum_confirmation_required: bool = True # Require momentum confirmation
    
    # Position management
    pyramiding: bool = True                  # Allow adding to positions
    max_pyramid_levels: int = 3             # Maximum pyramid levels
    pyramid_scale_factor: float = 0.5       # Scale factor for pyramid positions
    
    # Risk management per timeframe
    stop_loss_method: str = "atr"           # "atr", "percent", "technical"
    stop_loss_multiplier: float = 2.0      # Multiplier for ATR-based stops
    trailing_stop: bool = True              # Use trailing stops
    
    # Volatility filtering
    volatility_filter: bool = True          # Filter trades by volatility
    min_volatility: float = 0.01           # Minimum volatility for trading
    max_volatility: float = 0.1            # Maximum volatility for trading


@dataclass
class TimeframeAnalysis:
    """Analysis results for a specific timeframe"""
    timeframe: str
    trend_direction: TrendDirection
    trend_strength: float              # 0-1 scale
    momentum_direction: TrendDirection
    momentum_strength: float           # 0-1 scale
    volatility: float
    support_level: float
    resistance_level: float
    confidence: float                  # Overall confidence in analysis


class MultiTimeframeStrategy(BaseStrategy):
    """
    Multi-timeframe trading strategy that analyzes multiple time horizons
    
    Features:
    - Multiple timeframe trend analysis
    - Momentum confirmation across timeframes
    - Volatility-adjusted position sizing
    - Pyramiding capabilities
    - Dynamic stop losses
    """
    
    def __init__(self, config: MultiTimeframeConfig):
        super().__init__(config)
        self.config: MultiTimeframeConfig = config
        
        # Timeframe analysis cache
        self.timeframe_analysis: Dict[str, Dict[str, TimeframeAnalysis]] = {}
        self.price_history: Dict[str, List[float]] = {}
        self.volume_history: Dict[str, List[float]] = {}
        self.timestamps: List[datetime] = []
        
        # Position tracking per timeframe
        self.timeframe_positions: Dict[str, Dict[str, Any]] = {}
        self.pyramid_levels: Dict[str, int] = {}
        
        # Technical indicators cache
        self.indicators: Dict[str, Dict[str, Dict[str, float]]] = {}
        
        logger.info(f"Initialized multi-timeframe strategy with {len(config.timeframes)} timeframes")
    
    def generate_signals(self, data: pd.DataFrame, timestamp: datetime) -> List[Signal]:
        """Generate multi-timeframe trading signals"""
        signals = []
        
        # Update price history
        self._update_price_history(data, timestamp)
        
        # Need sufficient data for longest timeframe
        max_timeframe_days = max(tf.period_days for tf in self.config.timeframes)
        if len(self.timestamps) < max_timeframe_days:
            return signals
        
        # Analyze each timeframe for each symbol
        self._analyze_all_timeframes(timestamp)
        
        # Generate signals based on multi-timeframe analysis
        for symbol in self.config.symbols:
            if symbol not in data.columns or f"{symbol}_close" not in data.columns:
                continue
            
            try:
                symbol_signals = self._generate_symbol_signals(symbol, data, timestamp)
                signals.extend(symbol_signals)
            except Exception as e:
                logger.error(f"Error generating multi-timeframe signals for {symbol}: {e}")
        
        return signals
    
    def _update_price_history(self, data: pd.DataFrame, timestamp: datetime):
        """Update price and volume history"""
        self.timestamps.append(timestamp)
        
        for symbol in self.config.symbols:
            if f"{symbol}_close" in data.columns:
                price = data[f"{symbol}_close"].iloc[-1]
                volume = data.get(f"{symbol}_volume", pd.Series([1000000])).iloc[-1]
                
                if symbol not in self.price_history:
                    self.price_history[symbol] = []
                    self.volume_history[symbol] = []
                
                self.price_history[symbol].append(price)
                self.volume_history[symbol].append(volume)
        
        # Keep limited history
        max_history = max(tf.period_days for tf in self.config.timeframes) + 50
        if len(self.timestamps) > max_history:
            self.timestamps = self.timestamps[-max_history:]
            for symbol in self.price_history:
                self.price_history[symbol] = self.price_history[symbol][-max_history:]
                self.volume_history[symbol] = self.volume_history[symbol][-max_history:]
    
    def _analyze_all_timeframes(self, timestamp: datetime):
        """Analyze all timeframes for all symbols"""
        for symbol in self.config.symbols:
            if symbol not in self.price_history:
                continue
            
            if symbol not in self.timeframe_analysis:
                self.timeframe_analysis[symbol] = {}
                self.indicators[symbol] = {}
            
            for timeframe_config in self.config.timeframes:
                analysis = self._analyze_timeframe(symbol, timeframe_config, timestamp)
                self.timeframe_analysis[symbol][timeframe_config.name] = analysis
    
    def _analyze_timeframe(self, symbol: str, timeframe_config: TimeframeConfig, 
                          timestamp: datetime) -> TimeframeAnalysis:
        """Analyze a specific timeframe for a symbol"""
        prices = np.array(self.price_history[symbol])
        volumes = np.array(self.volume_history[symbol])
        
        # Get data for this timeframe
        period_length = min(timeframe_config.period_days, len(prices))
        timeframe_prices = prices[-period_length:]
        timeframe_volumes = volumes[-period_length:]
        
        if len(timeframe_prices) < 5:
            return TimeframeAnalysis(
                timeframe=timeframe_config.name,
                trend_direction=TrendDirection.NEUTRAL,
                trend_strength=0.0,
                momentum_direction=TrendDirection.NEUTRAL,
                momentum_strength=0.0,
                volatility=0.02,
                support_level=timeframe_prices[-1],
                resistance_level=timeframe_prices[-1],
                confidence=0.0
            )
        
        # Calculate technical indicators for this timeframe
        indicators = self._calculate_timeframe_indicators(timeframe_prices, timeframe_volumes, timeframe_config)
        
        # Store indicators
        if timeframe_config.name not in self.indicators[symbol]:
            self.indicators[symbol][timeframe_config.name] = {}
        self.indicators[symbol][timeframe_config.name] = indicators
        
        # Analyze trend
        trend_direction, trend_strength = self._analyze_trend(timeframe_prices, indicators, timeframe_config)
        
        # Analyze momentum
        momentum_direction, momentum_strength = self._analyze_momentum(timeframe_prices, indicators, timeframe_config)
        
        # Calculate volatility
        volatility = self._calculate_volatility(timeframe_prices, timeframe_config)
        
        # Identify support and resistance
        support_level, resistance_level = self._identify_support_resistance(timeframe_prices)
        
        # Calculate overall confidence
        confidence = self._calculate_confidence(trend_strength, momentum_strength, volatility, timeframe_config)
        
        return TimeframeAnalysis(
            timeframe=timeframe_config.name,
            trend_direction=trend_direction,
            trend_strength=trend_strength,
            momentum_direction=momentum_direction,
            momentum_strength=momentum_strength,
            volatility=volatility,
            support_level=support_level,
            resistance_level=resistance_level,
            confidence=confidence
        )
    
    def _calculate_timeframe_indicators(self, prices: np.ndarray, volumes: np.ndarray, 
                                      config: TimeframeConfig) -> Dict[str, float]:
        """Calculate technical indicators for a timeframe"""
        indicators = {}
        
        # Moving averages
        if len(prices) >= config.fast_ma_period:
            indicators['fast_ma'] = np.mean(prices[-config.fast_ma_period:])
        else:
            indicators['fast_ma'] = prices[-1]
        
        if len(prices) >= config.slow_ma_period:
            indicators['slow_ma'] = np.mean(prices[-config.slow_ma_period:])
        else:
            indicators['slow_ma'] = prices[-1]
        
        # RSI
        indicators['rsi'] = self._calculate_rsi(prices, config.rsi_period)
        
        # MACD
        indicators['macd'], indicators['macd_signal'] = self._calculate_macd(
            prices, config.macd_fast, config.macd_slow, config.macd_signal
        )
        indicators['macd_histogram'] = indicators['macd'] - indicators['macd_signal']
        
        # ATR
        indicators['atr'] = self._calculate_atr(prices, config.atr_period)
        
        # Rate of Change
        roc_period = min(10, len(prices) - 1)
        if roc_period > 0:
            indicators['roc'] = (prices[-1] - prices[-roc_period-1]) / prices[-roc_period-1]
        else:
            indicators['roc'] = 0.0
        
        # Bollinger Bands
        bb_period = min(20, len(prices))
        if bb_period >= 5:
            bb_sma = np.mean(prices[-bb_period:])
            bb_std = np.std(prices[-bb_period:])
            indicators['bb_upper'] = bb_sma + (2 * bb_std)
            indicators['bb_lower'] = bb_sma - (2 * bb_std)
            indicators['bb_position'] = (prices[-1] - bb_lower) / (bb_upper - bb_lower) if bb_upper != bb_lower else 0.5
        else:
            indicators['bb_upper'] = prices[-1]
            indicators['bb_lower'] = prices[-1]
            indicators['bb_position'] = 0.5
        
        # Volume indicators
        if len(volumes) >= 10:
            indicators['volume_sma'] = np.mean(volumes[-10:])
            indicators['volume_ratio'] = volumes[-1] / indicators['volume_sma']
        else:
            indicators['volume_sma'] = volumes[-1]
            indicators['volume_ratio'] = 1.0
        
        return indicators
    
    def _analyze_trend(self, prices: np.ndarray, indicators: Dict[str, float], 
                      config: TimeframeConfig) -> Tuple[TrendDirection, float]:
        """Analyze trend direction and strength"""
        trend_signals = []
        
        # Moving average trend
        if indicators['fast_ma'] > indicators['slow_ma']:
            trend_signals.append(1)  # Bullish
        elif indicators['fast_ma'] < indicators['slow_ma']:
            trend_signals.append(-1)  # Bearish
        else:
            trend_signals.append(0)  # Neutral
        
        # Price vs moving average
        current_price = prices[-1]
        if current_price > indicators['slow_ma']:
            trend_signals.append(1)
        elif current_price < indicators['slow_ma']:
            trend_signals.append(-1)
        else:
            trend_signals.append(0)
        
        # Linear regression trend (simplified)
        if len(prices) >= 10:
            x = np.arange(len(prices[-10:]))
            y = prices[-10:]
            slope = np.polyfit(x, y, 1)[0]
            
            if slope > 0:
                trend_signals.append(1)
            elif slope < 0:
                trend_signals.append(-1)
            else:
                trend_signals.append(0)
        
        # Calculate overall trend
        avg_signal = np.mean(trend_signals)
        trend_strength = abs(avg_signal)
        
        if avg_signal > 0.3:
            trend_direction = TrendDirection.BULLISH
        elif avg_signal < -0.3:
            trend_direction = TrendDirection.BEARISH
        else:
            trend_direction = TrendDirection.NEUTRAL
        
        return trend_direction, trend_strength
    
    def _analyze_momentum(self, prices: np.ndarray, indicators: Dict[str, float], 
                         config: TimeframeConfig) -> Tuple[TrendDirection, float]:
        """Analyze momentum direction and strength"""
        momentum_signals = []
        
        # RSI momentum
        rsi = indicators['rsi']
        if rsi > 60:
            momentum_signals.append(1)  # Bullish momentum
        elif rsi < 40:
            momentum_signals.append(-1)  # Bearish momentum
        else:
            momentum_signals.append(0)  # Neutral
        
        # MACD momentum
        if indicators['macd'] > indicators['macd_signal']:
            momentum_signals.append(1)
        elif indicators['macd'] < indicators['macd_signal']:
            momentum_signals.append(-1)
        else:
            momentum_signals.append(0)
        
        # Rate of Change momentum
        roc = indicators['roc']
        if roc > 0.01:  # 1% threshold
            momentum_signals.append(1)
        elif roc < -0.01:
            momentum_signals.append(-1)
        else:
            momentum_signals.append(0)
        
        # Price momentum (short-term)
        if len(prices) >= 5:
            short_momentum = (prices[-1] - prices[-5]) / prices[-5]
            if short_momentum > 0.005:  # 0.5% threshold
                momentum_signals.append(1)
            elif short_momentum < -0.005:
                momentum_signals.append(-1)
            else:
                momentum_signals.append(0)
        
        # Calculate overall momentum
        avg_signal = np.mean(momentum_signals)
        momentum_strength = abs(avg_signal)
        
        if avg_signal > 0.3:
            momentum_direction = TrendDirection.BULLISH
        elif avg_signal < -0.3:
            momentum_direction = TrendDirection.BEARISH
        else:
            momentum_direction = TrendDirection.NEUTRAL
        
        return momentum_direction, momentum_strength
    
    def _calculate_volatility(self, prices: np.ndarray, config: TimeframeConfig) -> float:
        """Calculate volatility for the timeframe"""
        if len(prices) < 2:
            return 0.02
        
        returns = np.diff(prices) / prices[:-1]
        volatility = np.std(returns) * np.sqrt(252)  # Annualized
        
        return volatility
    
    def _identify_support_resistance(self, prices: np.ndarray) -> Tuple[float, float]:
        """Identify support and resistance levels"""
        if len(prices) < 10:
            return prices[-1], prices[-1]
        
        # Simple support/resistance based on recent highs and lows
        recent_prices = prices[-20:] if len(prices) >= 20 else prices
        
        # Support: recent low
        support_level = np.min(recent_prices)
        
        # Resistance: recent high
        resistance_level = np.max(recent_prices)
        
        return support_level, resistance_level
    
    def _calculate_confidence(self, trend_strength: float, momentum_strength: float, 
                            volatility: float, config: TimeframeConfig) -> float:
        """Calculate overall confidence in the analysis"""
        # Base confidence from trend and momentum alignment
        base_confidence = (trend_strength + momentum_strength) / 2
        
        # Adjust for volatility (moderate volatility is preferred)
        if self.config.volatility_filter:
            if self.config.min_volatility <= volatility <= self.config.max_volatility:
                volatility_adjustment = 1.0
            else:
                volatility_adjustment = 0.5
        else:
            volatility_adjustment = 1.0
        
        # Weight adjustment based on timeframe weight
        weight_adjustment = config.weight
        
        confidence = base_confidence * volatility_adjustment * weight_adjustment
        
        return min(1.0, confidence)
    
    def _generate_symbol_signals(self, symbol: str, data: pd.DataFrame, 
                               timestamp: datetime) -> List[Signal]:
        """Generate signals for a symbol based on multi-timeframe analysis"""
        signals = []
        
        if symbol not in self.timeframe_analysis:
            return signals
        
        current_price = data[f"{symbol}_close"].iloc[-1]
        timeframe_analyses = self.timeframe_analysis[symbol]
        
        # Calculate weighted consensus
        bullish_weight = 0
        bearish_weight = 0
        total_weight = 0
        
        for tf_config in self.config.timeframes:
            tf_name = tf_config.name
            if tf_name not in timeframe_analyses:
                continue
            
            analysis = timeframe_analyses[tf_name]
            weight = tf_config.weight * analysis.confidence
            
            if analysis.trend_direction == TrendDirection.BULLISH:
                if not self.config.momentum_confirmation_required or analysis.momentum_direction == TrendDirection.BULLISH:
                    bullish_weight += weight
            elif analysis.trend_direction == TrendDirection.BEARISH:
                if not self.config.momentum_confirmation_required or analysis.momentum_direction == TrendDirection.BEARISH:
                    bearish_weight += weight
            
            total_weight += weight
        
        if total_weight == 0:
            return signals
        
        # Normalize weights
        bullish_consensus = bullish_weight / total_weight
        bearish_consensus = bearish_weight / total_weight
        
        # Check for position
        existing_position = symbol in self.positions
        pyramid_level = self.pyramid_levels.get(symbol, 0)
        
        # Generate entry signals
        if not existing_position:
            if bullish_consensus >= self.config.min_timeframe_agreement:
                signals.append(self._create_entry_signal(
                    symbol, SignalType.BUY, current_price, bullish_consensus, timestamp
                ))
                self.pyramid_levels[symbol] = 1
            
            elif bearish_consensus >= self.config.min_timeframe_agreement:
                signals.append(self._create_entry_signal(
                    symbol, SignalType.SELL, current_price, bearish_consensus, timestamp
                ))
                self.pyramid_levels[symbol] = 1
        
        # Generate pyramid signals
        elif (self.config.pyramiding and 
              pyramid_level < self.config.max_pyramid_levels):
            
            position = self.positions[symbol]
            
            # Add to long position
            if (position.position_type.value == "long" and 
                bullish_consensus >= self.config.min_timeframe_agreement):
                
                pyramid_signal = self._create_pyramid_signal(
                    symbol, SignalType.SCALE_IN, current_price, bullish_consensus, 
                    pyramid_level, timestamp
                )
                if pyramid_signal:
                    signals.append(pyramid_signal)
                    self.pyramid_levels[symbol] += 1
            
            # Add to short position
            elif (position.position_type.value == "short" and 
                  bearish_consensus >= self.config.min_timeframe_agreement):
                
                pyramid_signal = self._create_pyramid_signal(
                    symbol, SignalType.SCALE_IN, current_price, bearish_consensus, 
                    pyramid_level, timestamp
                )
                if pyramid_signal:
                    signals.append(pyramid_signal)
                    self.pyramid_levels[symbol] += 1
        
        # Generate exit signals
        if existing_position:
            exit_signal = self._check_exit_conditions(symbol, current_price, timestamp)
            if exit_signal:
                signals.append(exit_signal)
                self.pyramid_levels[symbol] = 0
        
        return signals
    
    def _create_entry_signal(self, symbol: str, signal_type: SignalType, 
                           price: float, confidence: float, timestamp: datetime) -> Signal:
        """Create entry signal"""
        return Signal(
            symbol=symbol,
            signal_type=signal_type,
            timestamp=timestamp,
            price=price,
            confidence=confidence,
            size=self.config.base_position_size,
            metadata={
                "entry_type": "initial",
                "timeframe_consensus": confidence,
                "pyramid_level": 1
            }
        )
    
    def _create_pyramid_signal(self, symbol: str, signal_type: SignalType, 
                             price: float, confidence: float, pyramid_level: int,
                             timestamp: datetime) -> Optional[Signal]:
        """Create pyramid signal"""
        # Scale down position size for pyramid entries
        pyramid_size = (self.config.base_position_size * 
                       (self.config.pyramid_scale_factor ** pyramid_level))
        
        if pyramid_size < self.config.min_trade_size / price:
            return None
        
        return Signal(
            symbol=symbol,
            signal_type=signal_type,
            timestamp=timestamp,
            price=price,
            confidence=confidence,
            size=pyramid_size,
            metadata={
                "entry_type": "pyramid",
                "pyramid_level": pyramid_level + 1,
                "timeframe_consensus": confidence
            }
        )
    
    def _check_exit_conditions(self, symbol: str, current_price: float, 
                             timestamp: datetime) -> Optional[Signal]:
        """Check if position should be exited"""
        if symbol not in self.positions:
            return None
        
        position = self.positions[symbol]
        timeframe_analyses = self.timeframe_analysis[symbol]
        
        # Check trend reversal across timeframes
        reversal_weight = 0
        total_weight = 0
        
        for tf_config in self.config.timeframes:
            tf_name = tf_config.name
            if tf_name not in timeframe_analyses:
                continue
            
            analysis = timeframe_analyses[tf_name]
            weight = tf_config.weight * analysis.confidence
            
            # Check for trend reversal
            if position.position_type.value == "long":
                if analysis.trend_direction == TrendDirection.BEARISH:
                    reversal_weight += weight
            else:  # short position
                if analysis.trend_direction == TrendDirection.BULLISH:
                    reversal_weight += weight
            
            total_weight += weight
        
        if total_weight > 0:
            reversal_consensus = reversal_weight / total_weight
            
            if reversal_consensus >= self.config.min_timeframe_agreement:
                signal_type = SignalType.SELL if position.position_type.value == "long" else SignalType.CLOSE_SHORT
                
                return Signal(
                    symbol=symbol,
                    signal_type=signal_type,
                    timestamp=timestamp,
                    price=current_price,
                    confidence=reversal_consensus,
                    size=1.0,  # Close entire position
                    metadata={
                        "exit_reason": "trend_reversal",
                        "reversal_consensus": reversal_consensus
                    }
                )
        
        return None
    
    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """Calculate RSI"""
        if len(prices) < period + 1:
            return 50.0
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _calculate_macd(self, prices: np.ndarray, fast: int = 12, 
                       slow: int = 26, signal: int = 9) -> Tuple[float, float]:
        """Calculate MACD"""
        if len(prices) < slow:
            return 0.0, 0.0
        
        # Simple EMA approximation
        ema_fast = np.mean(prices[-fast:])
        ema_slow = np.mean(prices[-slow:])
        
        macd = ema_fast - ema_slow
        macd_signal = macd  # Simplified - normally would be EMA of MACD
        
        return macd, macd_signal
    
    def _calculate_atr(self, prices: np.ndarray, period: int = 14) -> float:
        """Calculate Average True Range"""
        if len(prices) < period + 1:
            return 0.01
        
        price_ranges = np.abs(np.diff(prices[-period-1:]))
        atr = np.mean(price_ranges)
        
        return atr
    
    def calculate_position_size(self, signal: Signal, current_price: float, 
                              available_cash: float) -> float:
        """Calculate position size for multi-timeframe strategy"""
        # Base position size from signal
        base_size = signal.size
        
        # Adjust for volatility if enabled
        if self.config.volatility_filter and signal.symbol in self.timeframe_analysis:
            # Use medium-term timeframe for volatility adjustment
            medium_tf_name = next((tf.name for tf in self.config.timeframes 
                                 if 15 <= tf.period_days <= 30), None)
            
            if medium_tf_name and medium_tf_name in self.timeframe_analysis[signal.symbol]:
                analysis = self.timeframe_analysis[signal.symbol][medium_tf_name]
                volatility = analysis.volatility
                
                # Scale position size inversely with volatility
                vol_adjustment = min(2.0, max(0.5, 0.15 / (volatility + 1e-6)))
                base_size *= vol_adjustment
        
        # Calculate target value
        target_value = available_cash * base_size
        
        # Account for transaction costs
        transaction_cost_rate = self.config.commission_rate + self.config.slippage_rate
        effective_cash = target_value / (1 + transaction_cost_rate)
        
        shares = effective_cash / current_price
        
        # Apply minimum trade size constraint
        min_shares = self.config.min_trade_size / current_price
        if shares < min_shares:
            return 0.0
        
        return shares
    
    def get_timeframe_summary(self) -> Dict[str, Dict[str, Any]]:
        """Get summary of all timeframe analyses"""
        summary = {}
        
        for symbol, timeframe_analyses in self.timeframe_analysis.items():
            summary[symbol] = {}
            
            for tf_name, analysis in timeframe_analyses.items():
                summary[symbol][tf_name] = {
                    "trend_direction": analysis.trend_direction.value,
                    "trend_strength": analysis.trend_strength,
                    "momentum_direction": analysis.momentum_direction.value,
                    "momentum_strength": analysis.momentum_strength,
                    "volatility": analysis.volatility,
                    "confidence": analysis.confidence
                }
        
        return summary
    
    def get_strategy_metrics(self) -> Dict[str, Any]:
        """Get multi-timeframe specific metrics"""
        base_metrics = self.get_performance_metrics()
        
        mtf_metrics = {
            "active_timeframes": len(self.config.timeframes),
            "pyramid_positions": len([p for p in self.pyramid_levels.values() if p > 1]),
            "avg_pyramid_level": np.mean(list(self.pyramid_levels.values())) if self.pyramid_levels else 0
        }
        
        # Timeframe consensus metrics
        if self.timeframe_analysis:
            consensus_scores = []
            volatility_scores = []
            
            for symbol_analyses in self.timeframe_analysis.values():
                for analysis in symbol_analyses.values():
                    consensus_scores.append(analysis.confidence)
                    volatility_scores.append(analysis.volatility)
            
            if consensus_scores:
                mtf_metrics["avg_timeframe_consensus"] = np.mean(consensus_scores)
                mtf_metrics["avg_timeframe_volatility"] = np.mean(volatility_scores)
        
        return {**base_metrics, **mtf_metrics}


# Export classes
__all__ = ['MultiTimeframeStrategy', 'MultiTimeframeConfig', 'TimeframeConfig', 
          'TimeframeAnalysis', 'TimeframeType', 'TrendDirection']