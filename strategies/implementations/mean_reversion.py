"""
Mean Reversion Strategy Implementation
Trading strategy based on mean reversion patterns and statistical arbitrage
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from scipy import stats

from ..base_strategy import BaseStrategy, StrategyConfig, Signal, SignalType
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class MeanReversionConfig(StrategyConfig):
    """Configuration for mean reversion strategy"""
    name: str = "MeanReversionStrategy"
    
    # Mean reversion parameters
    lookback_window: int = 20          # Period for mean calculation
    zscore_threshold: float = 2.0      # Z-score threshold for signals
    zscore_exit_threshold: float = 0.5 # Z-score threshold for exits
    
    # Bollinger Bands parameters
    bollinger_period: int = 20
    bollinger_std_dev: float = 2.0
    
    # Relative Strength Index parameters
    rsi_period: int = 14
    rsi_overbought: float = 75
    rsi_oversold: float = 25
    
    # Volume confirmation
    volume_confirmation: bool = True
    volume_threshold: float = 0.8  # Minimum volume ratio for signals
    
    # Statistical tests
    use_normality_test: bool = True
    normality_p_value: float = 0.05
    
    # Risk management
    max_adverse_move: float = 0.05     # 5% adverse move to exit
    profit_target: float = 0.03       # 3% profit target
    holding_period_limit: int = 10     # Max days to hold position
    
    # Position sizing
    volatility_scaling: bool = True
    target_volatility: float = 0.15   # Target portfolio volatility


class MeanReversionStrategy(BaseStrategy):
    """
    Mean reversion trading strategy that identifies overbought/oversold conditions
    
    The strategy uses multiple mean reversion indicators:
    - Z-score relative to moving average
    - Bollinger Bands for volatility-adjusted levels
    - RSI for momentum divergence
    - Statistical tests for mean reversion validity
    """
    
    def __init__(self, config: MeanReversionConfig):
        super().__init__(config)
        self.config: MeanReversionConfig = config
        
        # Statistical indicators cache
        self.indicators_cache: Dict[str, Dict[str, np.ndarray]] = {}
        self.statistical_cache: Dict[str, Dict[str, float]] = {}
        
        logger.info(f"Initialized mean reversion strategy with {config.lookback_window}d lookback")
    
    def generate_signals(self, data: pd.DataFrame, timestamp: datetime) -> List[Signal]:
        """Generate mean reversion trading signals"""
        signals = []
        
        if len(data) < self.config.lookback_window * 2:
            logger.warning("Insufficient data for mean reversion calculation")
            return signals
        
        # Update indicators for all symbols
        self._update_indicators(data, timestamp)
        
        for symbol in self.config.symbols:
            if symbol not in data.columns or f"{symbol}_close" not in data.columns:
                continue
            
            try:
                signal = self._generate_mean_reversion_signal(symbol, data, timestamp)
                if signal:
                    signals.append(signal)
            except Exception as e:
                logger.error(f"Error generating mean reversion signal for {symbol}: {e}")
        
        return signals
    
    def _update_indicators(self, data: pd.DataFrame, timestamp: datetime):
        """Update mean reversion indicators for all symbols"""
        for symbol in self.config.symbols:
            if f"{symbol}_close" not in data.columns:
                continue
            
            prices = data[f"{symbol}_close"].values
            volumes = data.get(f"{symbol}_volume", pd.Series(np.ones(len(prices)))).values
            
            if symbol not in self.indicators_cache:
                self.indicators_cache[symbol] = {}
                self.statistical_cache[symbol] = {}
            
            indicators = self.indicators_cache[symbol]
            
            # Z-score calculation
            indicators['zscore'] = self._calculate_zscore(prices)
            
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(prices)
            indicators['bb_upper'] = bb_upper
            indicators['bb_middle'] = bb_middle
            indicators['bb_lower'] = bb_lower
            indicators['bb_position'] = self._calculate_bb_position(prices, bb_upper, bb_lower)
            
            # RSI
            indicators['rsi'] = self._calculate_rsi(prices)
            
            # Moving average and standard deviation
            indicators['sma'] = self._calculate_sma(prices, self.config.lookback_window)
            indicators['rolling_std'] = self._calculate_rolling_std(prices)
            
            # Volume indicators
            if self.config.volume_confirmation:
                indicators['volume_ratio'] = self._calculate_volume_ratio(volumes)
            
            # Price velocity (rate of change)
            indicators['price_velocity'] = self._calculate_price_velocity(prices)
            
            # Statistical properties
            self._update_statistical_properties(symbol, prices)
    
    def _generate_mean_reversion_signal(self, symbol: str, data: pd.DataFrame, 
                                      timestamp: datetime) -> Optional[Signal]:
        """Generate mean reversion signal for a specific symbol"""
        if symbol not in self.indicators_cache:
            return None
        
        indicators = self.indicators_cache[symbol]
        current_price = data[f"{symbol}_close"].iloc[-1]
        
        # Get current indicator values
        zscore = indicators['zscore'][-1] if len(indicators['zscore']) > 0 else 0
        bb_position = indicators['bb_position'][-1] if len(indicators['bb_position']) > 0 else 0
        rsi = indicators['rsi'][-1] if len(indicators['rsi']) > 0 else 50
        price_velocity = indicators['price_velocity'][-1] if len(indicators['price_velocity']) > 0 else 0
        
        # Volume confirmation
        volume_ok = True
        if self.config.volume_confirmation:
            volume_ratio = indicators['volume_ratio'][-1] if len(indicators['volume_ratio']) > 0 else 1
            volume_ok = volume_ratio >= self.config.volume_threshold
        
        # Check statistical validity
        stats_valid = self._check_statistical_validity(symbol)
        
        # Check existing position
        existing_position = symbol in self.positions
        
        # Generate signals
        signal_type = SignalType.HOLD
        confidence = 0.0
        
        # Oversold condition - buy signal
        if (zscore < -self.config.zscore_threshold and
            bb_position < -0.8 and  # Near lower Bollinger Band
            rsi < self.config.rsi_oversold and
            price_velocity < 0 and  # Declining momentum
            volume_ok and
            stats_valid and
            not existing_position):
            
            signal_type = SignalType.BUY
            confidence = min(1.0, abs(zscore) / self.config.zscore_threshold)
        
        # Overbought condition - sell signal
        elif (zscore > self.config.zscore_threshold and
              bb_position > 0.8 and  # Near upper Bollinger Band
              rsi > self.config.rsi_overbought and
              price_velocity > 0 and  # Rising momentum
              volume_ok and
              stats_valid and
              not existing_position):
            
            signal_type = SignalType.SELL
            confidence = min(1.0, zscore / self.config.zscore_threshold)
        
        # Exit conditions for existing positions
        elif existing_position:
            position = self.positions[symbol]
            entry_price = position.entry_price
            current_return = (current_price - entry_price) / entry_price
            days_held = (timestamp - position.entry_time).days
            
            # Exit long position
            if position.position_type.value == "long":
                # Profit target or mean reversion complete
                if (current_return >= self.config.profit_target or
                    zscore > -self.config.zscore_exit_threshold or
                    current_return <= -self.config.max_adverse_move or
                    days_held >= self.config.holding_period_limit):
                    
                    signal_type = SignalType.SELL
                    confidence = 0.8
            
            # Exit short position
            elif position.position_type.value == "short":
                # Profit target or mean reversion complete
                if (-current_return >= self.config.profit_target or
                    zscore < self.config.zscore_exit_threshold or
                    -current_return <= -self.config.max_adverse_move or
                    days_held >= self.config.holding_period_limit):
                    
                    signal_type = SignalType.CLOSE_SHORT
                    confidence = 0.8
        
        # Create signal if conditions are met
        if signal_type != SignalType.HOLD and confidence > self.config.signal_threshold:
            position_size = self._calculate_mean_reversion_position_size(symbol, confidence)
            
            return Signal(
                symbol=symbol,
                signal_type=signal_type,
                timestamp=timestamp,
                price=current_price,
                confidence=confidence,
                size=position_size,
                metadata={
                    "zscore": zscore,
                    "bb_position": bb_position,
                    "rsi": rsi,
                    "price_velocity": price_velocity,
                    "volume_ratio": indicators.get('volume_ratio', [1])[-1] if 'volume_ratio' in indicators else 1
                }
            )
        
        return None
    
    def _calculate_zscore(self, prices: np.ndarray) -> np.ndarray:
        """Calculate rolling Z-score"""
        if len(prices) < self.config.lookback_window:
            return np.zeros(len(prices))
        
        zscore = np.zeros(len(prices))
        
        for i in range(self.config.lookback_window, len(prices)):
            window_prices = prices[i - self.config.lookback_window:i]
            mean_price = np.mean(window_prices)
            std_price = np.std(window_prices)
            
            if std_price > 0:
                zscore[i] = (prices[i] - mean_price) / std_price
        
        return zscore
    
    def _calculate_bollinger_bands(self, prices: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate Bollinger Bands"""
        sma = self._calculate_sma(prices, self.config.bollinger_period)
        rolling_std = self._calculate_rolling_std(prices, self.config.bollinger_period)
        
        bb_upper = sma + (rolling_std * self.config.bollinger_std_dev)
        bb_lower = sma - (rolling_std * self.config.bollinger_std_dev)
        
        return bb_upper, sma, bb_lower
    
    def _calculate_bb_position(self, prices: np.ndarray, bb_upper: np.ndarray, 
                              bb_lower: np.ndarray) -> np.ndarray:
        """Calculate position within Bollinger Bands (-1 to 1)"""
        bb_position = np.zeros(len(prices))
        
        for i in range(len(prices)):
            if bb_upper[i] != bb_lower[i]:
                bb_position[i] = (prices[i] - bb_lower[i]) / (bb_upper[i] - bb_lower[i]) * 2 - 1
        
        return bb_position
    
    def _calculate_rsi(self, prices: np.ndarray) -> np.ndarray:
        """Calculate Relative Strength Index"""
        if len(prices) < self.config.rsi_period + 1:
            return np.full(len(prices), 50.0)
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        rsi = np.full(len(prices), 50.0)
        
        # Calculate initial average gains and losses
        avg_gain = np.mean(gains[:self.config.rsi_period])
        avg_loss = np.mean(losses[:self.config.rsi_period])
        
        for i in range(self.config.rsi_period, len(prices) - 1):
            avg_gain = (avg_gain * (self.config.rsi_period - 1) + gains[i]) / self.config.rsi_period
            avg_loss = (avg_loss * (self.config.rsi_period - 1) + losses[i]) / self.config.rsi_period
            
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
        
        # Forward fill NaN values
        sma = pd.Series(sma).fillna(method='ffill').fillna(method='bfill').values
        
        return sma
    
    def _calculate_rolling_std(self, prices: np.ndarray, period: int = None) -> np.ndarray:
        """Calculate rolling standard deviation"""
        if period is None:
            period = self.config.lookback_window
        
        if len(prices) < period:
            return np.full(len(prices), np.std(prices))
        
        rolling_std = np.full(len(prices), np.nan)
        
        for i in range(period - 1, len(prices)):
            rolling_std[i] = np.std(prices[i - period + 1:i + 1])
        
        # Forward fill NaN values
        rolling_std = pd.Series(rolling_std).fillna(method='ffill').fillna(method='bfill').values
        
        return rolling_std
    
    def _calculate_volume_ratio(self, volumes: np.ndarray) -> np.ndarray:
        """Calculate volume ratio relative to average"""
        if len(volumes) < self.config.lookback_window:
            return np.ones(len(volumes))
        
        volume_ratio = np.ones(len(volumes))
        
        for i in range(self.config.lookback_window, len(volumes)):
            current_volume = volumes[i]
            avg_volume = np.mean(volumes[i - self.config.lookback_window:i])
            volume_ratio[i] = current_volume / avg_volume if avg_volume > 0 else 1.0
        
        return volume_ratio
    
    def _calculate_price_velocity(self, prices: np.ndarray, period: int = 5) -> np.ndarray:
        """Calculate price velocity (rate of change)"""
        if len(prices) < period + 1:
            return np.zeros(len(prices))
        
        velocity = np.zeros(len(prices))
        
        for i in range(period, len(prices)):
            velocity[i] = (prices[i] - prices[i - period]) / prices[i - period]
        
        return velocity
    
    def _update_statistical_properties(self, symbol: str, prices: np.ndarray):
        """Update statistical properties for mean reversion validation"""
        if len(prices) < self.config.lookback_window * 2:
            return
        
        # Recent price returns
        returns = np.diff(prices[-self.config.lookback_window:]) / prices[-self.config.lookback_window:-1]
        
        stats_cache = self.statistical_cache[symbol]
        
        # Test for normality (if returns are normally distributed, mean reversion is more reliable)
        if self.config.use_normality_test and len(returns) > 8:
            try:
                _, p_value = stats.normaltest(returns)
                stats_cache['normality_p_value'] = p_value
            except:
                stats_cache['normality_p_value'] = 0.0
        
        # Calculate Hurst exponent (< 0.5 indicates mean reversion)
        stats_cache['hurst_exponent'] = self._calculate_hurst_exponent(prices[-self.config.lookback_window:])
        
        # Calculate half-life of mean reversion
        stats_cache['half_life'] = self._calculate_half_life(prices[-self.config.lookback_window:])
    
    def _check_statistical_validity(self, symbol: str) -> bool:
        """Check if statistical conditions support mean reversion"""
        if symbol not in self.statistical_cache:
            return True  # Default to valid if no stats available
        
        stats_cache = self.statistical_cache[symbol]
        
        # Check normality if enabled
        if self.config.use_normality_test:
            normality_p = stats_cache.get('normality_p_value', 1.0)
            if normality_p < self.config.normality_p_value:
                return False  # Reject null hypothesis of normality
        
        # Check Hurst exponent
        hurst = stats_cache.get('hurst_exponent', 0.5)
        if hurst > 0.6:  # Too trending
            return False
        
        # Check half-life
        half_life = stats_cache.get('half_life', 10)
        if half_life > self.config.holding_period_limit:
            return False  # Mean reversion too slow
        
        return True
    
    def _calculate_hurst_exponent(self, prices: np.ndarray) -> float:
        """Calculate Hurst exponent using R/S analysis"""
        if len(prices) < 10:
            return 0.5
        
        try:
            # Convert prices to log returns
            log_returns = np.diff(np.log(prices))
            
            # Calculate R/S statistic for different time lags
            lags = range(2, min(len(log_returns) // 2, 20))
            rs_values = []
            
            for lag in lags:
                # Calculate mean return over lag period
                mean_return = np.mean(log_returns[:lag])
                
                # Calculate cumulative deviations
                cumulative_deviations = np.cumsum(log_returns[:lag] - mean_return)
                
                # Calculate range and standard deviation
                R = np.max(cumulative_deviations) - np.min(cumulative_deviations)
                S = np.std(log_returns[:lag])
                
                if S > 0:
                    rs_values.append(R / S)
            
            if len(rs_values) < 3:
                return 0.5
            
            # Fit log(R/S) = H * log(lag) + constant
            log_lags = np.log(list(lags)[:len(rs_values)])
            log_rs = np.log(rs_values)
            
            # Simple linear regression
            hurst = np.polyfit(log_lags, log_rs, 1)[0]
            
            return max(0.0, min(1.0, hurst))
        
        except:
            return 0.5
    
    def _calculate_half_life(self, prices: np.ndarray) -> float:
        """Calculate half-life of mean reversion"""
        if len(prices) < 3:
            return 10.0
        
        try:
            # Calculate price deviations from mean
            mean_price = np.mean(prices)
            deviations = prices - mean_price
            
            # Fit AR(1) model: deviation(t) = a * deviation(t-1) + error
            y = deviations[1:]
            x = deviations[:-1]
            
            if len(x) > 0 and np.var(x) > 0:
                a = np.cov(x, y)[0, 1] / np.var(x)
                
                # Half-life = -log(2) / log(a)
                if 0 < a < 1:
                    half_life = -np.log(2) / np.log(a)
                    return max(1.0, min(100.0, half_life))
            
            return 10.0
        
        except:
            return 10.0
    
    def _calculate_mean_reversion_position_size(self, symbol: str, confidence: float) -> float:
        """Calculate position size based on mean reversion strength"""
        base_size = self.config.base_position_size
        
        # Adjust for confidence
        size = base_size * confidence
        
        # Adjust for volatility if enabled
        if self.config.volatility_scaling and symbol in self.indicators_cache:
            indicators = self.indicators_cache[symbol]
            if 'rolling_std' in indicators and len(indicators['rolling_std']) > 0:
                current_vol = indicators['rolling_std'][-1]
                # Scale inversely with volatility
                vol_adjustment = self.config.target_volatility / (current_vol + 1e-6)
                size *= min(2.0, max(0.5, vol_adjustment))
        
        return min(size, self.config.max_position_size)
    
    def calculate_position_size(self, signal: Signal, current_price: float, 
                              available_cash: float) -> float:
        """Calculate position size for mean reversion strategy"""
        target_size = signal.size
        max_value = available_cash * target_size
        
        # Account for transaction costs
        transaction_cost_rate = self.config.commission_rate + self.config.slippage_rate
        effective_cash = max_value / (1 + transaction_cost_rate)
        
        shares = effective_cash / current_price
        
        # Apply minimum trade size constraint
        min_shares = self.config.min_trade_size / current_price
        if shares < min_shares:
            return 0.0
        
        return shares
    
    def get_strategy_metrics(self) -> Dict[str, Any]:
        """Get mean reversion specific metrics"""
        base_metrics = self.get_performance_metrics()
        
        # Calculate mean reversion specific metrics
        mr_metrics = {}
        
        if self.indicators_cache:
            # Average Z-score across symbols
            total_zscore = 0
            symbol_count = 0
            
            for symbol, indicators in self.indicators_cache.items():
                if 'zscore' in indicators and len(indicators['zscore']) > 0:
                    total_zscore += abs(indicators['zscore'][-1])
                    symbol_count += 1
            
            if symbol_count > 0:
                mr_metrics['avg_zscore_magnitude'] = total_zscore / symbol_count
        
        # Statistical properties
        if self.statistical_cache:
            hurst_values = [stats.get('hurst_exponent', 0.5) for stats in self.statistical_cache.values()]
            if hurst_values:
                mr_metrics['avg_hurst_exponent'] = np.mean(hurst_values)
        
        return {**base_metrics, **mr_metrics}


# Export classes
__all__ = ['MeanReversionStrategy', 'MeanReversionConfig']