"""
Market Regime Detection and Classification System
Identifies market states, volatility regimes, and trend strength for adaptive trading strategies
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import warnings
from datetime import datetime, timedelta
from scipy import stats
from scipy.signal import find_peaks
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from utils.logger import get_logger

logger = get_logger(__name__)

warnings.filterwarnings('ignore', category=RuntimeWarning)


class MarketState(Enum):
    """Market trend states"""
    STRONG_UPTREND = "strong_uptrend"
    WEAK_UPTREND = "weak_uptrend"
    SIDEWAYS = "sideways"
    WEAK_DOWNTREND = "weak_downtrend"
    STRONG_DOWNTREND = "strong_downtrend"


class VolatilityRegime(Enum):
    """Volatility regimes"""
    LOW_VOLATILITY = "low_volatility"
    NORMAL_VOLATILITY = "normal_volatility"
    HIGH_VOLATILITY = "high_volatility"
    EXTREME_VOLATILITY = "extreme_volatility"


class LiquidityRegime(Enum):
    """Liquidity regimes"""
    HIGH_LIQUIDITY = "high_liquidity"
    NORMAL_LIQUIDITY = "normal_liquidity"
    LOW_LIQUIDITY = "low_liquidity"
    ILLIQUID = "illiquid"


@dataclass
class MarketRegimeState:
    """Complete market regime analysis result"""
    timestamp: datetime
    market_state: MarketState
    volatility_regime: VolatilityRegime
    liquidity_regime: LiquidityRegime
    trend_strength: float  # 0-1 scale
    trend_direction: float  # -1 to 1
    volatility_percentile: float  # 0-100
    momentum_score: float  # -1 to 1
    risk_level: str  # LOW, MEDIUM, HIGH
    confidence: float  # 0-1
    regime_persistence: float  # Expected duration in days
    key_metrics: Dict[str, float]


@dataclass
class RegimeTransition:
    """Market regime transition detection"""
    from_regime: MarketRegimeState
    to_regime: MarketRegimeState
    transition_probability: float
    transition_date: datetime
    catalyst_factors: List[str]
    expected_duration: float


class MarketRegimeDetector:
    """Advanced market regime detection and classification system"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Detection parameters
        self.lookback_period = self.config.get('lookback_period', 252)  # 1 year
        self.short_window = self.config.get('short_window', 20)
        self.medium_window = self.config.get('medium_window', 50)
        self.long_window = self.config.get('long_window', 200)
        
        # Regime thresholds
        self.trend_threshold = self.config.get('trend_threshold', 0.3)
        self.volatility_threshold = self.config.get('volatility_threshold', 0.02)
        self.momentum_threshold = self.config.get('momentum_threshold', 0.1)
        
        # Historical regime data for persistence analysis
        self.regime_history: List[MarketRegimeState] = []
        
    def detect_market_regime(self, df: pd.DataFrame, volume_data: Optional[pd.DataFrame] = None) -> MarketRegimeState:
        """
        Detect current market regime based on price and volume data
        
        Args:
            df: OHLCV DataFrame
            volume_data: Additional volume metrics if available
            
        Returns:
            Complete market regime analysis
        """
        logger.info("Detecting current market regime")
        
        try:
            if len(df) < self.lookback_period:
                logger.warning(f"Insufficient data for regime detection: {len(df)} < {self.lookback_period}")
                return self._create_default_regime()
            
            # Calculate regime components
            market_state = self._detect_market_state(df)
            volatility_regime = self._detect_volatility_regime(df)
            liquidity_regime = self._detect_liquidity_regime(df, volume_data)
            
            # Calculate strength and direction metrics
            trend_strength = self._calculate_trend_strength(df)
            trend_direction = self._calculate_trend_direction(df)
            volatility_percentile = self._calculate_volatility_percentile(df)
            momentum_score = self._calculate_momentum_score(df)
            
            # Assess risk level
            risk_level = self._assess_risk_level(volatility_regime, market_state, momentum_score)
            
            # Calculate confidence
            confidence = self._calculate_regime_confidence(df, market_state, volatility_regime)
            
            # Estimate regime persistence
            regime_persistence = self._estimate_regime_persistence(market_state, volatility_regime)
            
            # Gather key metrics
            key_metrics = self._calculate_key_metrics(df)
            
            regime_state = MarketRegimeState(
                timestamp=datetime.now(),
                market_state=market_state,
                volatility_regime=volatility_regime,
                liquidity_regime=liquidity_regime,
                trend_strength=trend_strength,
                trend_direction=trend_direction,
                volatility_percentile=volatility_percentile,
                momentum_score=momentum_score,
                risk_level=risk_level,
                confidence=confidence,
                regime_persistence=regime_persistence,
                key_metrics=key_metrics
            )
            
            # Update history
            self.regime_history.append(regime_state)
            if len(self.regime_history) > 100:  # Keep last 100 regime states
                self.regime_history = self.regime_history[-100:]
            
            logger.info(f"Market regime detected: {market_state.value}, Volatility: {volatility_regime.value}, "
                       f"Trend Strength: {trend_strength:.2f}, Confidence: {confidence:.2f}")
            
            return regime_state
            
        except Exception as e:
            logger.error(f"Error detecting market regime: {e}")
            return self._create_default_regime()
    
    def detect_regime_transitions(self, df: pd.DataFrame, lookback_days: int = 30) -> List[RegimeTransition]:
        """
        Detect potential regime transitions and predict future changes
        
        Args:
            df: OHLCV DataFrame
            lookback_days: Days to look back for transition analysis
            
        Returns:
            List of detected and predicted regime transitions
        """
        logger.info("Analyzing regime transitions")
        
        transitions = []
        
        try:
            if len(self.regime_history) < 5:  # Need some history
                return transitions
            
            # Analyze recent regime changes
            recent_regimes = self.regime_history[-lookback_days:]
            
            for i in range(1, len(recent_regimes)):
                current_regime = recent_regimes[i]
                previous_regime = recent_regimes[i-1]
                
                # Check for regime change
                if (current_regime.market_state != previous_regime.market_state or
                    current_regime.volatility_regime != previous_regime.volatility_regime):
                    
                    # Calculate transition probability
                    transition_prob = self._calculate_transition_probability(previous_regime, current_regime)
                    
                    # Identify catalyst factors
                    catalysts = self._identify_transition_catalysts(df, previous_regime, current_regime)
                    
                    # Estimate expected duration
                    expected_duration = self._estimate_transition_duration(current_regime)
                    
                    transition = RegimeTransition(
                        from_regime=previous_regime,
                        to_regime=current_regime,
                        transition_probability=transition_prob,
                        transition_date=current_regime.timestamp,
                        catalyst_factors=catalysts,
                        expected_duration=expected_duration
                    )
                    
                    transitions.append(transition)
            
            # Predict potential future transitions
            future_transitions = self._predict_future_transitions(df)
            transitions.extend(future_transitions)
            
            logger.info(f"Found {len(transitions)} regime transitions")
            
        except Exception as e:
            logger.error(f"Error detecting regime transitions: {e}")
            
        return transitions
    
    def calculate_regime_probabilities(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate probabilities of different market regimes
        
        Args:
            df: OHLCV DataFrame
            
        Returns:
            Dictionary of regime probabilities
        """
        logger.debug("Calculating regime probabilities")
        
        try:
            # Calculate features for regime classification
            features = self._extract_regime_features(df)
            
            # Use machine learning approach for probability estimation
            probabilities = self._calculate_ml_probabilities(features)
            
            return probabilities
            
        except Exception as e:
            logger.error(f"Error calculating regime probabilities: {e}")
            return {}
    
    def get_trading_implications(self, regime_state: MarketRegimeState) -> Dict[str, Any]:
        """
        Get trading strategy implications for current market regime
        
        Args:
            regime_state: Current market regime state
            
        Returns:
            Trading strategy recommendations
        """
        logger.debug("Calculating trading implications for current regime")
        
        implications = {
            'strategy_type': 'balanced',
            'position_sizing': 'normal',
            'stop_loss_adjustment': 1.0,
            'profit_target_adjustment': 1.0,
            'preferred_timeframes': ['4h', '1d'],
            'risk_management': 'standard',
            'entry_timing': 'normal',
            'instruments': ['stocks', 'etfs'],
            'leverage_recommendation': 1.0,
            'hedging_recommendation': False
        }
        
        try:
            # Strategy type based on market state
            if regime_state.market_state in [MarketState.STRONG_UPTREND, MarketState.STRONG_DOWNTREND]:
                implications['strategy_type'] = 'trend_following'
                implications['preferred_timeframes'] = ['1h', '4h', '1d']
                implications['leverage_recommendation'] = 1.5
                
            elif regime_state.market_state == MarketState.SIDEWAYS:
                implications['strategy_type'] = 'mean_reversion'
                implications['preferred_timeframes'] = ['15m', '1h']
                implications['leverage_recommendation'] = 0.8
                
            # Position sizing based on volatility
            if regime_state.volatility_regime == VolatilityRegime.LOW_VOLATILITY:
                implications['position_sizing'] = 'increased'
                implications['leverage_recommendation'] *= 1.2
                
            elif regime_state.volatility_regime in [VolatilityRegime.HIGH_VOLATILITY, VolatilityRegime.EXTREME_VOLATILITY]:
                implications['position_sizing'] = 'reduced'
                implications['stop_loss_adjustment'] = 1.5
                implications['leverage_recommendation'] *= 0.7
                implications['hedging_recommendation'] = True
                
            # Risk management adjustments
            if regime_state.risk_level == 'HIGH':
                implications['risk_management'] = 'conservative'
                implications['position_sizing'] = 'minimal'
                implications['entry_timing'] = 'selective'
                
            elif regime_state.risk_level == 'LOW':
                implications['risk_management'] = 'aggressive'
                implications['position_sizing'] = 'increased'
                
            # Liquidity considerations
            if regime_state.liquidity_regime in [LiquidityRegime.LOW_LIQUIDITY, LiquidityRegime.ILLIQUID]:
                implications['instruments'] = ['large_cap_stocks', 'major_etfs']
                implications['entry_timing'] = 'patient'
                
            # Confidence-based adjustments
            if regime_state.confidence < 0.6:
                implications['position_sizing'] = 'reduced'
                implications['leverage_recommendation'] *= 0.8
                
            logger.debug(f"Trading implications: {implications['strategy_type']}, "
                        f"Position sizing: {implications['position_sizing']}, "
                        f"Risk level: {regime_state.risk_level}")
            
        except Exception as e:
            logger.error(f"Error calculating trading implications: {e}")
            
        return implications
    
    # Market State Detection Methods
    def _detect_market_state(self, df: pd.DataFrame) -> MarketState:
        """Detect overall market trend state"""
        try:
            # Calculate multiple trend indicators
            ma_short = df['close'].rolling(self.short_window).mean()
            ma_medium = df['close'].rolling(self.medium_window).mean()
            ma_long = df['close'].rolling(self.long_window).mean()
            
            current_price = df['close'].iloc[-1]
            
            # Trend alignment
            short_above_medium = ma_short.iloc[-1] > ma_medium.iloc[-1]
            medium_above_long = ma_medium.iloc[-1] > ma_long.iloc[-1]
            price_above_short = current_price > ma_short.iloc[-1]
            
            # Price momentum
            price_change_short = (current_price - ma_short.iloc[-self.short_window]) / ma_short.iloc[-self.short_window]
            price_change_medium = (current_price - ma_medium.iloc[-self.medium_window]) / ma_medium.iloc[-self.medium_window]
            
            # ADX for trend strength
            adx = self._calculate_adx(df)
            current_adx = adx.iloc[-1] if not pd.isna(adx.iloc[-1]) else 25
            
            # Classify market state
            if (short_above_medium and medium_above_long and price_above_short and 
                price_change_short > self.trend_threshold and current_adx > 30):
                return MarketState.STRONG_UPTREND
                
            elif (short_above_medium and price_above_short and 
                  price_change_short > self.trend_threshold / 2):
                return MarketState.WEAK_UPTREND
                
            elif (not short_above_medium and not medium_above_long and not price_above_short and 
                  price_change_short < -self.trend_threshold and current_adx > 30):
                return MarketState.STRONG_DOWNTREND
                
            elif (not short_above_medium and not price_above_short and 
                  price_change_short < -self.trend_threshold / 2):
                return MarketState.WEAK_DOWNTREND
                
            else:
                return MarketState.SIDEWAYS
                
        except Exception as e:
            logger.error(f"Error detecting market state: {e}")
            return MarketState.SIDEWAYS
    
    def _detect_volatility_regime(self, df: pd.DataFrame) -> VolatilityRegime:
        """Detect volatility regime"""
        try:
            # Calculate realized volatility
            returns = df['close'].pct_change().dropna()
            
            # Multiple volatility measures
            rv_20 = returns.rolling(20).std() * np.sqrt(252)  # 20-day annualized vol
            rv_60 = returns.rolling(60).std() * np.sqrt(252)  # 60-day annualized vol
            
            # GARCH-like volatility clustering
            vol_squared = returns.rolling(20).var()
            vol_persistence = vol_squared.rolling(10).mean()
            
            # ATR-based volatility
            atr = self._calculate_atr(df, 20)
            atr_pct = (atr / df['close']).rolling(20).mean()
            
            current_rv = rv_20.iloc[-1] if not pd.isna(rv_20.iloc[-1]) else 0.2
            current_atr_pct = atr_pct.iloc[-1] if not pd.isna(atr_pct.iloc[-1]) else 0.02
            
            # Historical percentiles
            rv_percentile = self._calculate_percentile(rv_20, current_rv)
            
            # Classify volatility regime
            if rv_percentile > 90 or current_rv > 0.4:
                return VolatilityRegime.EXTREME_VOLATILITY
            elif rv_percentile > 75 or current_rv > 0.25:
                return VolatilityRegime.HIGH_VOLATILITY
            elif rv_percentile < 25 and current_rv < 0.15:
                return VolatilityRegime.LOW_VOLATILITY
            else:
                return VolatilityRegime.NORMAL_VOLATILITY
                
        except Exception as e:
            logger.error(f"Error detecting volatility regime: {e}")
            return VolatilityRegime.NORMAL_VOLATILITY
    
    def _detect_liquidity_regime(self, df: pd.DataFrame, volume_data: Optional[pd.DataFrame] = None) -> LiquidityRegime:
        """Detect liquidity regime based on volume and spread data"""
        try:
            if 'volume' not in df.columns:
                return LiquidityRegime.NORMAL_LIQUIDITY
            
            # Volume-based liquidity measures
            volume_ma = df['volume'].rolling(20).mean()
            current_volume = df['volume'].iloc[-1]
            volume_ratio = current_volume / volume_ma.iloc[-1] if volume_ma.iloc[-1] > 0 else 1
            
            # Volume trend
            volume_trend = df['volume'].rolling(10).mean() / df['volume'].rolling(30).mean()
            current_volume_trend = volume_trend.iloc[-1] if not pd.isna(volume_trend.iloc[-1]) else 1
            
            # Price-volume relationship (Amihud illiquidity measure)
            returns = df['close'].pct_change().abs()
            dollar_volume = df['volume'] * df['close']
            illiquidity = returns / (dollar_volume / 1e6)  # Amihud measure
            illiquidity_ma = illiquidity.rolling(20).mean()
            current_illiquidity = illiquidity_ma.iloc[-1] if not pd.isna(illiquidity_ma.iloc[-1]) else 0
            
            # Bid-ask spread (if available)
            if 'bid' in df.columns and 'ask' in df.columns:
                spread = (df['ask'] - df['bid']) / ((df['ask'] + df['bid']) / 2)
                avg_spread = spread.rolling(20).mean().iloc[-1]
            else:
                avg_spread = 0.001  # Assume tight spread if not available
            
            # Classify liquidity regime
            if (volume_ratio > 1.5 and current_volume_trend > 1.2 and 
                current_illiquidity < 0.5 and avg_spread < 0.002):
                return LiquidityRegime.HIGH_LIQUIDITY
                
            elif (volume_ratio < 0.5 or current_volume_trend < 0.8 or 
                  current_illiquidity > 2.0 or avg_spread > 0.01):
                return LiquidityRegime.LOW_LIQUIDITY
                
            elif current_illiquidity > 5.0 or avg_spread > 0.05:
                return LiquidityRegime.ILLIQUID
                
            else:
                return LiquidityRegime.NORMAL_LIQUIDITY
                
        except Exception as e:
            logger.error(f"Error detecting liquidity regime: {e}")
            return LiquidityRegime.NORMAL_LIQUIDITY
    
    # Calculation Methods
    def _calculate_trend_strength(self, df: pd.DataFrame) -> float:
        """Calculate overall trend strength (0-1)"""
        try:
            # Multiple trend strength measures
            adx = self._calculate_adx(df)
            current_adx = adx.iloc[-1] if not pd.isna(adx.iloc[-1]) else 25
            
            # R-squared of price vs time (linearity of trend)
            prices = df['close'].iloc[-50:].values  # Last 50 days
            time_index = np.arange(len(prices))
            if len(prices) > 10:
                slope, intercept, r_value, _, _ = stats.linregress(time_index, prices)
                r_squared = r_value ** 2
            else:
                r_squared = 0
            
            # Moving average alignment
            ma_20 = df['close'].rolling(20).mean()
            ma_50 = df['close'].rolling(50).mean()
            ma_alignment = abs(ma_20.iloc[-1] - ma_50.iloc[-1]) / ma_50.iloc[-1] if ma_50.iloc[-1] > 0 else 0
            
            # Combine measures
            adx_strength = min(current_adx / 50, 1.0)  # Normalize ADX
            linearity_strength = r_squared
            alignment_strength = min(ma_alignment * 10, 1.0)  # Scale alignment
            
            # Weighted average
            trend_strength = (0.4 * adx_strength + 0.4 * linearity_strength + 0.2 * alignment_strength)
            
            return max(0, min(1, trend_strength))
            
        except Exception as e:
            logger.error(f"Error calculating trend strength: {e}")
            return 0.5
    
    def _calculate_trend_direction(self, df: pd.DataFrame) -> float:
        """Calculate trend direction (-1 to 1)"""
        try:
            # Multiple timeframe price slopes
            short_slope = self._calculate_price_slope(df['close'], 20)
            medium_slope = self._calculate_price_slope(df['close'], 50)
            long_slope = self._calculate_price_slope(df['close'], 100)
            
            # Normalize slopes
            price_std = df['close'].rolling(100).std().iloc[-1]
            if price_std > 0:
                short_norm = np.tanh(short_slope / price_std * 100)
                medium_norm = np.tanh(medium_slope / price_std * 100)
                long_norm = np.tanh(long_slope / price_std * 100)
            else:
                short_norm = medium_norm = long_norm = 0
            
            # Weighted average (more weight to recent trends)
            trend_direction = 0.5 * short_norm + 0.3 * medium_norm + 0.2 * long_norm
            
            return max(-1, min(1, trend_direction))
            
        except Exception as e:
            logger.error(f"Error calculating trend direction: {e}")
            return 0
    
    def _calculate_volatility_percentile(self, df: pd.DataFrame) -> float:
        """Calculate current volatility percentile (0-100)"""
        try:
            returns = df['close'].pct_change()
            current_vol = returns.rolling(20).std().iloc[-1] * np.sqrt(252)
            
            # Historical volatility series
            vol_series = returns.rolling(20).std() * np.sqrt(252)
            vol_series = vol_series.dropna()
            
            if len(vol_series) > 50:
                percentile = stats.percentileofscore(vol_series, current_vol)
                return percentile
            else:
                return 50.0  # Default to median
                
        except Exception as e:
            logger.error(f"Error calculating volatility percentile: {e}")
            return 50.0
    
    def _calculate_momentum_score(self, df: pd.DataFrame) -> float:
        """Calculate momentum score (-1 to 1)"""
        try:
            # Multiple momentum measures
            
            # Price momentum
            price_mom_5 = df['close'].pct_change(5).iloc[-1]
            price_mom_20 = df['close'].pct_change(20).iloc[-1]
            price_mom_60 = df['close'].pct_change(60).iloc[-1]
            
            # RSI momentum
            rsi = self._calculate_rsi(df['close'], 14)
            rsi_momentum = (rsi.iloc[-1] - 50) / 50 if not pd.isna(rsi.iloc[-1]) else 0
            
            # MACD momentum
            macd, signal, _ = self._calculate_macd(df['close'])
            macd_momentum = np.tanh(macd.iloc[-1]) if not pd.isna(macd.iloc[-1]) else 0
            
            # Combine momentum measures
            price_momentum = np.tanh((0.5 * price_mom_5 + 0.3 * price_mom_20 + 0.2 * price_mom_60) * 100)
            
            # Weighted combination
            momentum_score = 0.4 * price_momentum + 0.3 * rsi_momentum + 0.3 * macd_momentum
            
            return max(-1, min(1, momentum_score))
            
        except Exception as e:
            logger.error(f"Error calculating momentum score: {e}")
            return 0
    
    def _assess_risk_level(self, volatility_regime: VolatilityRegime, market_state: MarketState, 
                          momentum_score: float) -> str:
        """Assess overall risk level"""
        risk_score = 0
        
        # Volatility contribution
        if volatility_regime == VolatilityRegime.EXTREME_VOLATILITY:
            risk_score += 3
        elif volatility_regime == VolatilityRegime.HIGH_VOLATILITY:
            risk_score += 2
        elif volatility_regime == VolatilityRegime.LOW_VOLATILITY:
            risk_score += 0
        else:
            risk_score += 1
        
        # Market state contribution
        if market_state in [MarketState.STRONG_DOWNTREND, MarketState.WEAK_DOWNTREND]:
            risk_score += 2
        elif market_state == MarketState.SIDEWAYS:
            risk_score += 1
        
        # Momentum contribution
        if abs(momentum_score) > 0.7:
            risk_score += 1
        
        # Classification
        if risk_score >= 5:
            return "HIGH"
        elif risk_score >= 3:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _calculate_regime_confidence(self, df: pd.DataFrame, market_state: MarketState, 
                                   volatility_regime: VolatilityRegime) -> float:
        """Calculate confidence in regime detection"""
        try:
            confidence_factors = []
            
            # Data quality factor
            data_completeness = 1 - df.isnull().sum().sum() / (len(df) * len(df.columns))
            confidence_factors.append(data_completeness)
            
            # Trend consistency factor
            ma_20 = df['close'].rolling(20).mean()
            ma_50 = df['close'].rolling(50).mean()
            trend_consistency = 1 - abs(ma_20.iloc[-1] - ma_50.iloc[-1]) / ma_50.iloc[-1] if ma_50.iloc[-1] > 0 else 0.5
            confidence_factors.append(min(trend_consistency, 1.0))
            
            # Volatility stability factor
            vol_20 = df['close'].pct_change().rolling(20).std()
            vol_stability = 1 - (vol_20.rolling(10).std().iloc[-1] / vol_20.iloc[-1]) if vol_20.iloc[-1] > 0 else 0.5
            confidence_factors.append(min(vol_stability, 1.0))
            
            # Pattern confirmation factor
            adx = self._calculate_adx(df)
            pattern_strength = min(adx.iloc[-1] / 50, 1.0) if not pd.isna(adx.iloc[-1]) else 0.5
            confidence_factors.append(pattern_strength)
            
            # Average confidence
            confidence = sum(confidence_factors) / len(confidence_factors)
            
            return max(0.1, min(1.0, confidence))
            
        except Exception as e:
            logger.error(f"Error calculating regime confidence: {e}")
            return 0.5
    
    def _estimate_regime_persistence(self, market_state: MarketState, volatility_regime: VolatilityRegime) -> float:
        """Estimate how long the current regime is likely to persist (in days)"""
        try:
            # Base persistence by regime type
            base_persistence = {
                MarketState.STRONG_UPTREND: 45,
                MarketState.WEAK_UPTREND: 25,
                MarketState.SIDEWAYS: 60,
                MarketState.WEAK_DOWNTREND: 20,
                MarketState.STRONG_DOWNTREND: 30
            }
            
            # Volatility adjustment
            vol_multiplier = {
                VolatilityRegime.LOW_VOLATILITY: 1.3,
                VolatilityRegime.NORMAL_VOLATILITY: 1.0,
                VolatilityRegime.HIGH_VOLATILITY: 0.7,
                VolatilityRegime.EXTREME_VOLATILITY: 0.4
            }
            
            persistence = base_persistence.get(market_state, 30) * vol_multiplier.get(volatility_regime, 1.0)
            
            # Historical regime analysis adjustment
            if len(self.regime_history) > 10:
                similar_regimes = [r for r in self.regime_history[-50:] 
                                 if r.market_state == market_state and r.volatility_regime == volatility_regime]
                if similar_regimes:
                    avg_historical = sum(r.regime_persistence for r in similar_regimes) / len(similar_regimes)
                    persistence = 0.7 * persistence + 0.3 * avg_historical
            
            return max(5, min(120, persistence))  # Between 5 and 120 days
            
        except Exception as e:
            logger.error(f"Error estimating regime persistence: {e}")
            return 30
    
    def _calculate_key_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate key metrics for regime analysis"""
        metrics = {}
        
        try:
            # Price metrics
            metrics['price_change_1d'] = df['close'].pct_change(1).iloc[-1]
            metrics['price_change_5d'] = df['close'].pct_change(5).iloc[-1]
            metrics['price_change_20d'] = df['close'].pct_change(20).iloc[-1]
            
            # Volatility metrics
            returns = df['close'].pct_change()
            metrics['volatility_20d'] = returns.rolling(20).std().iloc[-1] * np.sqrt(252)
            metrics['volatility_60d'] = returns.rolling(60).std().iloc[-1] * np.sqrt(252)
            
            # Volume metrics
            if 'volume' in df.columns:
                metrics['volume_ratio_20d'] = df['volume'].iloc[-1] / df['volume'].rolling(20).mean().iloc[-1]
                metrics['volume_trend'] = df['volume'].rolling(5).mean().iloc[-1] / df['volume'].rolling(20).mean().iloc[-1]
            
            # Technical indicators
            rsi = self._calculate_rsi(df['close'], 14)
            metrics['rsi_14'] = rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50
            
            adx = self._calculate_adx(df)
            metrics['adx_14'] = adx.iloc[-1] if not pd.isna(adx.iloc[-1]) else 25
            
            # Moving averages
            ma_20 = df['close'].rolling(20).mean()
            ma_50 = df['close'].rolling(50).mean()
            metrics['price_vs_ma20'] = (df['close'].iloc[-1] - ma_20.iloc[-1]) / ma_20.iloc[-1]
            metrics['price_vs_ma50'] = (df['close'].iloc[-1] - ma_50.iloc[-1]) / ma_50.iloc[-1]
            metrics['ma20_vs_ma50'] = (ma_20.iloc[-1] - ma_50.iloc[-1]) / ma_50.iloc[-1]
            
        except Exception as e:
            logger.error(f"Error calculating key metrics: {e}")
            
        return metrics
    
    # Transition Analysis Methods
    def _calculate_transition_probability(self, previous_regime: MarketRegimeState, 
                                        current_regime: MarketRegimeState) -> float:
        """Calculate probability of regime transition"""
        try:
            # Base transition probabilities (simplified)
            transition_matrix = {
                (MarketState.STRONG_UPTREND, MarketState.WEAK_UPTREND): 0.3,
                (MarketState.STRONG_UPTREND, MarketState.SIDEWAYS): 0.2,
                (MarketState.WEAK_UPTREND, MarketState.SIDEWAYS): 0.4,
                (MarketState.SIDEWAYS, MarketState.WEAK_UPTREND): 0.3,
                (MarketState.SIDEWAYS, MarketState.WEAK_DOWNTREND): 0.3,
                (MarketState.WEAK_DOWNTREND, MarketState.STRONG_DOWNTREND): 0.2,
                # Add more transitions as needed
            }
            
            state_transition = (previous_regime.market_state, current_regime.market_state)
            base_prob = transition_matrix.get(state_transition, 0.1)
            
            # Adjust based on volatility change
            vol_change = abs(previous_regime.volatility_percentile - current_regime.volatility_percentile) / 100
            vol_adjustment = 1 + vol_change
            
            # Adjust based on momentum change
            momentum_change = abs(previous_regime.momentum_score - current_regime.momentum_score)
            momentum_adjustment = 1 + momentum_change
            
            # Confidence adjustment
            confidence_adjustment = (previous_regime.confidence + current_regime.confidence) / 2
            
            probability = base_prob * vol_adjustment * momentum_adjustment * confidence_adjustment
            
            return max(0.05, min(0.95, probability))
            
        except Exception as e:
            logger.error(f"Error calculating transition probability: {e}")
            return 0.5
    
    def _identify_transition_catalysts(self, df: pd.DataFrame, previous_regime: MarketRegimeState, 
                                     current_regime: MarketRegimeState) -> List[str]:
        """Identify factors that may have caused regime transition"""
        catalysts = []
        
        try:
            # Volume spike
            if 'volume' in df.columns:
                recent_volume = df['volume'].iloc[-5:].mean()
                historical_volume = df['volume'].rolling(20).mean().iloc[-1]
                if recent_volume > historical_volume * 2:
                    catalysts.append("Volume spike")
            
            # Price gap
            recent_returns = df['close'].pct_change().iloc[-5:]
            if any(abs(ret) > 0.05 for ret in recent_returns):  # 5% single-day move
                catalysts.append("Large price movement")
            
            # Volatility shift
            vol_change = current_regime.volatility_percentile - previous_regime.volatility_percentile
            if abs(vol_change) > 30:
                catalysts.append("Volatility regime shift")
            
            # Momentum reversal
            momentum_change = current_regime.momentum_score - previous_regime.momentum_score
            if abs(momentum_change) > 0.5:
                catalysts.append("Momentum reversal")
            
            # Trend break
            if (previous_regime.market_state in [MarketState.STRONG_UPTREND, MarketState.WEAK_UPTREND] and
                current_regime.market_state in [MarketState.WEAK_DOWNTREND, MarketState.STRONG_DOWNTREND]):
                catalysts.append("Trend reversal")
            
        except Exception as e:
            logger.error(f"Error identifying transition catalysts: {e}")
            
        return catalysts if catalysts else ["Market dynamics"]
    
    def _estimate_transition_duration(self, current_regime: MarketRegimeState) -> float:
        """Estimate how long the transition will last"""
        try:
            # Base duration by regime type
            base_duration = {
                MarketState.STRONG_UPTREND: 3,
                MarketState.WEAK_UPTREND: 5,
                MarketState.SIDEWAYS: 10,
                MarketState.WEAK_DOWNTREND: 7,
                MarketState.STRONG_DOWNTREND: 5
            }
            
            duration = base_duration.get(current_regime.market_state, 7)
            
            # Adjust for volatility
            if current_regime.volatility_regime == VolatilityRegime.HIGH_VOLATILITY:
                duration *= 0.7
            elif current_regime.volatility_regime == VolatilityRegime.LOW_VOLATILITY:
                duration *= 1.3
            
            return max(1, min(20, duration))
            
        except Exception as e:
            logger.error(f"Error estimating transition duration: {e}")
            return 7
    
    def _predict_future_transitions(self, df: pd.DataFrame) -> List[RegimeTransition]:
        """Predict potential future regime transitions"""
        predictions = []
        
        try:
            if not self.regime_history:
                return predictions
            
            current_regime = self.regime_history[-1]
            
            # Simple prediction based on regime persistence
            if current_regime.regime_persistence < 10:  # Regime is likely to change soon
                # Predict most likely next regime
                next_regime = self._predict_next_regime(current_regime, df)
                
                if next_regime and next_regime.market_state != current_regime.market_state:
                    prediction = RegimeTransition(
                        from_regime=current_regime,
                        to_regime=next_regime,
                        transition_probability=0.6,
                        transition_date=datetime.now() + timedelta(days=5),
                        catalyst_factors=["Regime persistence exhaustion"],
                        expected_duration=7
                    )
                    predictions.append(prediction)
            
        except Exception as e:
            logger.error(f"Error predicting future transitions: {e}")
            
        return predictions
    
    def _predict_next_regime(self, current_regime: MarketRegimeState, df: pd.DataFrame) -> Optional[MarketRegimeState]:
        """Predict the most likely next regime state"""
        try:
            # Simplified next regime prediction
            # In practice, this would use more sophisticated ML models
            
            # Trend mean reversion tendency
            if current_regime.market_state == MarketState.STRONG_UPTREND:
                next_state = MarketState.WEAK_UPTREND
            elif current_regime.market_state == MarketState.STRONG_DOWNTREND:
                next_state = MarketState.WEAK_DOWNTREND
            elif current_regime.market_state in [MarketState.WEAK_UPTREND, MarketState.WEAK_DOWNTREND]:
                next_state = MarketState.SIDEWAYS
            else:
                # Sideways can go either direction based on momentum
                if current_regime.momentum_score > 0:
                    next_state = MarketState.WEAK_UPTREND
                else:
                    next_state = MarketState.WEAK_DOWNTREND
            
            # Volatility tends to mean revert
            if current_regime.volatility_regime == VolatilityRegime.EXTREME_VOLATILITY:
                next_vol = VolatilityRegime.HIGH_VOLATILITY
            elif current_regime.volatility_regime == VolatilityRegime.LOW_VOLATILITY:
                next_vol = VolatilityRegime.NORMAL_VOLATILITY
            else:
                next_vol = current_regime.volatility_regime
            
            # Create predicted regime
            predicted_regime = MarketRegimeState(
                timestamp=datetime.now() + timedelta(days=7),
                market_state=next_state,
                volatility_regime=next_vol,
                liquidity_regime=current_regime.liquidity_regime,
                trend_strength=current_regime.trend_strength * 0.8,  # Assuming some decay
                trend_direction=current_regime.trend_direction * 0.7,
                volatility_percentile=min(75, current_regime.volatility_percentile),
                momentum_score=current_regime.momentum_score * 0.5,
                risk_level=current_regime.risk_level,
                confidence=0.3,  # Lower confidence for predictions
                regime_persistence=20,
                key_metrics={}
            )
            
            return predicted_regime
            
        except Exception as e:
            logger.error(f"Error predicting next regime: {e}")
            return None
    
    # Feature Extraction for ML
    def _extract_regime_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extract features for ML-based regime classification"""
        features = []
        
        try:
            # Price features
            returns = df['close'].pct_change().dropna()
            features.extend([
                returns.mean(),
                returns.std(),
                returns.skew(),
                returns.kurtosis()
            ])
            
            # Trend features
            ma_20 = df['close'].rolling(20).mean()
            ma_50 = df['close'].rolling(50).mean()
            features.extend([
                (df['close'].iloc[-1] - ma_20.iloc[-1]) / ma_20.iloc[-1],
                (ma_20.iloc[-1] - ma_50.iloc[-1]) / ma_50.iloc[-1],
                self._calculate_price_slope(df['close'], 20),
                self._calculate_price_slope(df['close'], 50)
            ])
            
            # Technical indicators
            rsi = self._calculate_rsi(df['close'], 14)
            features.append(rsi.iloc[-1] / 100)  # Normalize RSI
            
            adx = self._calculate_adx(df)
            features.append(adx.iloc[-1] / 100)  # Normalize ADX
            
            # Volume features
            if 'volume' in df.columns:
                volume_ma = df['volume'].rolling(20).mean()
                features.extend([
                    df['volume'].iloc[-1] / volume_ma.iloc[-1],
                    df['volume'].rolling(5).mean().iloc[-1] / volume_ma.iloc[-1]
                ])
            else:
                features.extend([1.0, 1.0])  # Default values
            
            # Volatility features
            vol_20 = returns.rolling(20).std()
            vol_60 = returns.rolling(60).std()
            features.extend([
                vol_20.iloc[-1],
                vol_60.iloc[-1],
                vol_20.iloc[-1] / vol_60.iloc[-1] if vol_60.iloc[-1] > 0 else 1.0
            ])
            
            return np.array(features)
            
        except Exception as e:
            logger.error(f"Error extracting regime features: {e}")
            return np.zeros(15)  # Return default feature vector
    
    def _calculate_ml_probabilities(self, features: np.ndarray) -> Dict[str, float]:
        """Calculate regime probabilities using simple heuristics (could be replaced with ML model)"""
        try:
            # Simple heuristic-based probabilities
            # In practice, this would use a trained ML model
            
            probabilities = {}
            
            # Extract key features
            returns_mean = features[0] if len(features) > 0 else 0
            returns_std = features[1] if len(features) > 1 else 0.02
            price_vs_ma20 = features[4] if len(features) > 4 else 0
            rsi_norm = features[8] if len(features) > 8 else 0.5
            
            # Market state probabilities
            if returns_mean > 0.01 and price_vs_ma20 > 0.05:
                probabilities['strong_uptrend'] = 0.7
                probabilities['weak_uptrend'] = 0.2
                probabilities['sideways'] = 0.1
                probabilities['weak_downtrend'] = 0.0
                probabilities['strong_downtrend'] = 0.0
            elif returns_mean > 0.005 and price_vs_ma20 > 0:
                probabilities['strong_uptrend'] = 0.2
                probabilities['weak_uptrend'] = 0.6
                probabilities['sideways'] = 0.2
                probabilities['weak_downtrend'] = 0.0
                probabilities['strong_downtrend'] = 0.0
            elif returns_mean < -0.01 and price_vs_ma20 < -0.05:
                probabilities['strong_uptrend'] = 0.0
                probabilities['weak_uptrend'] = 0.0
                probabilities['sideways'] = 0.1
                probabilities['weak_downtrend'] = 0.2
                probabilities['strong_downtrend'] = 0.7
            elif returns_mean < -0.005 and price_vs_ma20 < 0:
                probabilities['strong_uptrend'] = 0.0
                probabilities['weak_uptrend'] = 0.0
                probabilities['sideways'] = 0.2
                probabilities['weak_downtrend'] = 0.6
                probabilities['strong_downtrend'] = 0.2
            else:
                probabilities['strong_uptrend'] = 0.1
                probabilities['weak_uptrend'] = 0.2
                probabilities['sideways'] = 0.4
                probabilities['weak_downtrend'] = 0.2
                probabilities['strong_downtrend'] = 0.1
            
            # Volatility regime probabilities
            if returns_std > 0.04:
                probabilities['high_volatility'] = 0.8
                probabilities['normal_volatility'] = 0.2
                probabilities['low_volatility'] = 0.0
            elif returns_std < 0.015:
                probabilities['high_volatility'] = 0.0
                probabilities['normal_volatility'] = 0.3
                probabilities['low_volatility'] = 0.7
            else:
                probabilities['high_volatility'] = 0.2
                probabilities['normal_volatility'] = 0.6
                probabilities['low_volatility'] = 0.2
            
            return probabilities
            
        except Exception as e:
            logger.error(f"Error calculating ML probabilities: {e}")
            return {}
    
    # Utility Methods
    def _create_default_regime(self) -> MarketRegimeState:
        """Create a default regime state when detection fails"""
        return MarketRegimeState(
            timestamp=datetime.now(),
            market_state=MarketState.SIDEWAYS,
            volatility_regime=VolatilityRegime.NORMAL_VOLATILITY,
            liquidity_regime=LiquidityRegime.NORMAL_LIQUIDITY,
            trend_strength=0.5,
            trend_direction=0.0,
            volatility_percentile=50.0,
            momentum_score=0.0,
            risk_level="MEDIUM",
            confidence=0.3,
            regime_persistence=30,
            key_metrics={}
        )
    
    def _calculate_percentile(self, series: pd.Series, value: float) -> float:
        """Calculate percentile of value in series"""
        try:
            clean_series = series.dropna()
            if len(clean_series) > 10:
                return stats.percentileofscore(clean_series, value)
            else:
                return 50.0
        except:
            return 50.0
    
    def _calculate_price_slope(self, prices: pd.Series, window: int) -> float:
        """Calculate price slope over window"""
        try:
            recent_prices = prices.iloc[-window:].values
            if len(recent_prices) < 5:
                return 0
            
            x = np.arange(len(recent_prices))
            slope, _, _, _, _ = stats.linregress(x, recent_prices)
            return slope
            
        except:
            return 0
    
    # Technical indicator calculations (reused from indicators.py)
    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_adx(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average Directional Index"""
        high = df['high']
        low = df['low']
        close = df['close']
        
        plus_dm = high.diff()
        minus_dm = low.diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm > 0] = 0
        minus_dm = abs(minus_dm)
        
        tr1 = pd.DataFrame(high - low)
        tr2 = pd.DataFrame(abs(high - close.shift(1)))
        tr3 = pd.DataFrame(abs(low - close.shift(1)))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        atr = tr.rolling(period).mean()
        plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(period).mean() / atr)
        
        dx = (abs(plus_di - minus_di) / abs(plus_di + minus_di)) * 100
        adx = dx.rolling(period).mean()
        
        return adx
    
    def _calculate_atr(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Average True Range"""
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(period).mean()
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram


# Usage example and factory function
def create_market_regime_detector(config: Dict[str, Any] = None) -> MarketRegimeDetector:
    """Create an instance of MarketRegimeDetector with configuration"""
    return MarketRegimeDetector(config)


# Export classes and functions
__all__ = [
    'MarketRegimeDetector',
    'MarketRegimeState',
    'RegimeTransition',
    'MarketState',
    'VolatilityRegime',
    'LiquidityRegime',
    'create_market_regime_detector'
]