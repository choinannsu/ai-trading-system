"""
Advanced Technical Analysis Indicators
Multi-timeframe analysis, custom indicators, dynamic support/resistance, Elliott Wave analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import warnings
from datetime import datetime, timedelta
from utils.logger import get_logger

logger = get_logger(__name__)

warnings.filterwarnings('ignore', category=RuntimeWarning)


class TimeFrame(Enum):
    """Supported timeframes for analysis"""
    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    M30 = "30m"
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"
    W1 = "1w"


@dataclass
class TechnicalIndicator:
    """Technical indicator result"""
    name: str
    value: float
    signal: str  # BUY, SELL, NEUTRAL
    confidence: float
    timeframe: TimeFrame
    timestamp: datetime


@dataclass
class SupportResistanceLevel:
    """Support/Resistance level"""
    level: float
    strength: float
    touches: int
    level_type: str  # SUPPORT, RESISTANCE
    confidence: float
    timeframe: TimeFrame


@dataclass
class ElliottWave:
    """Elliott Wave analysis result"""
    wave_count: int
    current_wave: str  # 1, 2, 3, 4, 5, A, B, C
    wave_degree: str  # PRIMARY, INTERMEDIATE, MINOR
    completion_percent: float
    next_target: float
    confidence: float


class AdvancedIndicators:
    """Advanced technical analysis indicators with multi-timeframe support"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Indicator parameters
        self.rsi_period = self.config.get('rsi_period', 14)
        self.macd_fast = self.config.get('macd_fast', 12)
        self.macd_slow = self.config.get('macd_slow', 26)
        self.macd_signal = self.config.get('macd_signal', 9)
        self.bb_period = self.config.get('bb_period', 20)
        self.bb_std = self.config.get('bb_std', 2)
        
        # Multi-timeframe settings
        self.timeframes = [TimeFrame.M5, TimeFrame.M15, TimeFrame.H1, TimeFrame.D1]
        
    def analyze_multi_timeframe(self, data: Dict[str, pd.DataFrame]) -> Dict[str, List[TechnicalIndicator]]:
        """
        Perform multi-timeframe technical analysis
        
        Args:
            data: Dictionary with timeframe as key and OHLCV data as value
            
        Returns:
            Dictionary of indicators for each timeframe
        """
        logger.info("Starting multi-timeframe technical analysis")
        
        results = {}
        
        for timeframe_str, df in data.items():
            try:
                timeframe = TimeFrame(timeframe_str)
                indicators = self._analyze_single_timeframe(df, timeframe)
                results[timeframe_str] = indicators
                
                logger.debug(f"Completed analysis for {timeframe_str}: {len(indicators)} indicators")
                
            except Exception as e:
                logger.error(f"Error analyzing timeframe {timeframe_str}: {e}")
                continue
        
        # Perform cross-timeframe analysis
        cross_tf_signals = self._cross_timeframe_analysis(results)
        if cross_tf_signals:
            results['cross_timeframe'] = cross_tf_signals
            
        logger.info(f"Multi-timeframe analysis completed for {len(results)} timeframes")
        return results
    
    def _analyze_single_timeframe(self, df: pd.DataFrame, timeframe: TimeFrame) -> List[TechnicalIndicator]:
        """Analyze single timeframe data"""
        indicators = []
        current_time = datetime.now()
        
        if len(df) < 50:  # Minimum data points required
            logger.warning(f"Insufficient data for {timeframe.value}: {len(df)} rows")
            return indicators
        
        try:
            # Standard indicators
            indicators.extend(self._calculate_momentum_indicators(df, timeframe, current_time))
            indicators.extend(self._calculate_trend_indicators(df, timeframe, current_time))
            indicators.extend(self._calculate_volatility_indicators(df, timeframe, current_time))
            indicators.extend(self._calculate_volume_indicators(df, timeframe, current_time))
            
            # Custom indicators
            indicators.extend(self._calculate_custom_indicators(df, timeframe, current_time))
            
        except Exception as e:
            logger.error(f"Error calculating indicators for {timeframe.value}: {e}")
            
        return indicators
    
    def _calculate_momentum_indicators(self, df: pd.DataFrame, timeframe: TimeFrame, 
                                     current_time: datetime) -> List[TechnicalIndicator]:
        """Calculate momentum-based indicators"""
        indicators = []
        
        try:
            # RSI
            rsi = self._calculate_rsi(df['close'], self.rsi_period)
            if not pd.isna(rsi.iloc[-1]):
                rsi_signal = "BUY" if rsi.iloc[-1] < 30 else "SELL" if rsi.iloc[-1] > 70 else "NEUTRAL"
                rsi_confidence = min(abs(rsi.iloc[-1] - 50) / 20, 1.0)
                
                indicators.append(TechnicalIndicator(
                    name="RSI",
                    value=float(rsi.iloc[-1]),
                    signal=rsi_signal,
                    confidence=rsi_confidence,
                    timeframe=timeframe,
                    timestamp=current_time
                ))
            
            # MACD
            macd_line, macd_signal, macd_histogram = self._calculate_macd(
                df['close'], self.macd_fast, self.macd_slow, self.macd_signal
            )
            
            if not pd.isna(macd_histogram.iloc[-1]):
                macd_sig = "BUY" if macd_histogram.iloc[-1] > 0 and macd_histogram.iloc[-2] <= 0 else \
                          "SELL" if macd_histogram.iloc[-1] < 0 and macd_histogram.iloc[-2] >= 0 else "NEUTRAL"
                macd_confidence = min(abs(macd_histogram.iloc[-1]) / df['close'].iloc[-1] * 1000, 1.0)
                
                indicators.append(TechnicalIndicator(
                    name="MACD",
                    value=float(macd_histogram.iloc[-1]),
                    signal=macd_sig,
                    confidence=macd_confidence,
                    timeframe=timeframe,
                    timestamp=current_time
                ))
            
            # Stochastic
            stoch_k, stoch_d = self._calculate_stochastic(df, 14, 3)
            if not pd.isna(stoch_k.iloc[-1]):
                stoch_signal = "BUY" if stoch_k.iloc[-1] < 20 and stoch_k.iloc[-1] > stoch_d.iloc[-1] else \
                              "SELL" if stoch_k.iloc[-1] > 80 and stoch_k.iloc[-1] < stoch_d.iloc[-1] else "NEUTRAL"
                stoch_confidence = min(abs(stoch_k.iloc[-1] - 50) / 30, 1.0)
                
                indicators.append(TechnicalIndicator(
                    name="Stochastic",
                    value=float(stoch_k.iloc[-1]),
                    signal=stoch_signal,
                    confidence=stoch_confidence,
                    timeframe=timeframe,
                    timestamp=current_time
                ))
                
        except Exception as e:
            logger.error(f"Error calculating momentum indicators: {e}")
            
        return indicators
    
    def _calculate_trend_indicators(self, df: pd.DataFrame, timeframe: TimeFrame,
                                  current_time: datetime) -> List[TechnicalIndicator]:
        """Calculate trend-following indicators"""
        indicators = []
        
        try:
            # Moving Averages
            ma_fast = df['close'].rolling(20).mean()
            ma_slow = df['close'].rolling(50).mean()
            
            if not pd.isna(ma_fast.iloc[-1]) and not pd.isna(ma_slow.iloc[-1]):
                ma_signal = "BUY" if ma_fast.iloc[-1] > ma_slow.iloc[-1] else "SELL"
                ma_confidence = abs(ma_fast.iloc[-1] - ma_slow.iloc[-1]) / df['close'].iloc[-1]
                
                indicators.append(TechnicalIndicator(
                    name="MA_Cross",
                    value=float(ma_fast.iloc[-1] - ma_slow.iloc[-1]),
                    signal=ma_signal,
                    confidence=min(ma_confidence * 10, 1.0),
                    timeframe=timeframe,
                    timestamp=current_time
                ))
            
            # ADX (Average Directional Index)
            adx = self._calculate_adx(df, 14)
            if not pd.isna(adx.iloc[-1]):
                adx_signal = "BUY" if adx.iloc[-1] > 25 else "NEUTRAL"
                adx_confidence = min(adx.iloc[-1] / 50, 1.0)
                
                indicators.append(TechnicalIndicator(
                    name="ADX",
                    value=float(adx.iloc[-1]),
                    signal=adx_signal,
                    confidence=adx_confidence,
                    timeframe=timeframe,
                    timestamp=current_time
                ))
            
            # Parabolic SAR
            sar = self._calculate_parabolic_sar(df)
            if not pd.isna(sar.iloc[-1]):
                sar_signal = "BUY" if df['close'].iloc[-1] > sar.iloc[-1] else "SELL"
                sar_confidence = abs(df['close'].iloc[-1] - sar.iloc[-1]) / df['close'].iloc[-1]
                
                indicators.append(TechnicalIndicator(
                    name="Parabolic_SAR",
                    value=float(sar.iloc[-1]),
                    signal=sar_signal,
                    confidence=min(sar_confidence * 20, 1.0),
                    timeframe=timeframe,
                    timestamp=current_time
                ))
                
        except Exception as e:
            logger.error(f"Error calculating trend indicators: {e}")
            
        return indicators
    
    def _calculate_volatility_indicators(self, df: pd.DataFrame, timeframe: TimeFrame,
                                       current_time: datetime) -> List[TechnicalIndicator]:
        """Calculate volatility-based indicators"""
        indicators = []
        
        try:
            # Bollinger Bands
            bb_middle = df['close'].rolling(self.bb_period).mean()
            bb_std = df['close'].rolling(self.bb_period).std()
            bb_upper = bb_middle + (bb_std * self.bb_std)
            bb_lower = bb_middle - (bb_std * self.bb_std)
            
            if not pd.isna(bb_upper.iloc[-1]):
                current_price = df['close'].iloc[-1]
                bb_position = (current_price - bb_lower.iloc[-1]) / (bb_upper.iloc[-1] - bb_lower.iloc[-1])
                
                bb_signal = "SELL" if bb_position > 0.9 else "BUY" if bb_position < 0.1 else "NEUTRAL"
                bb_confidence = abs(bb_position - 0.5) * 2
                
                indicators.append(TechnicalIndicator(
                    name="Bollinger_Bands",
                    value=float(bb_position),
                    signal=bb_signal,
                    confidence=bb_confidence,
                    timeframe=timeframe,
                    timestamp=current_time
                ))
            
            # Average True Range (ATR)
            atr = self._calculate_atr(df, 14)
            if not pd.isna(atr.iloc[-1]):
                atr_pct = atr.iloc[-1] / df['close'].iloc[-1] * 100
                volatility_signal = "HIGH" if atr_pct > 3 else "LOW" if atr_pct < 1 else "NORMAL"
                
                indicators.append(TechnicalIndicator(
                    name="ATR",
                    value=float(atr_pct),
                    signal=volatility_signal,
                    confidence=min(atr_pct / 5, 1.0),
                    timeframe=timeframe,
                    timestamp=current_time
                ))
                
        except Exception as e:
            logger.error(f"Error calculating volatility indicators: {e}")
            
        return indicators
    
    def _calculate_volume_indicators(self, df: pd.DataFrame, timeframe: TimeFrame,
                                   current_time: datetime) -> List[TechnicalIndicator]:
        """Calculate volume-based indicators"""
        indicators = []
        
        if 'volume' not in df.columns:
            return indicators
            
        try:
            # On-Balance Volume (OBV)
            obv = self._calculate_obv(df)
            if len(obv) > 20:
                obv_ma = obv.rolling(20).mean()
                if not pd.isna(obv_ma.iloc[-1]):
                    obv_signal = "BUY" if obv.iloc[-1] > obv_ma.iloc[-1] else "SELL"
                    obv_confidence = abs(obv.iloc[-1] - obv_ma.iloc[-1]) / obv_ma.iloc[-1]
                    
                    indicators.append(TechnicalIndicator(
                        name="OBV",
                        value=float(obv.iloc[-1]),
                        signal=obv_signal,
                        confidence=min(obv_confidence, 1.0),
                        timeframe=timeframe,
                        timestamp=current_time
                    ))
            
            # Volume Rate of Change
            volume_roc = df['volume'].pct_change(periods=20)
            if not pd.isna(volume_roc.iloc[-1]):
                vol_signal = "BUY" if volume_roc.iloc[-1] > 0.5 else "NEUTRAL"
                vol_confidence = min(abs(volume_roc.iloc[-1]), 1.0)
                
                indicators.append(TechnicalIndicator(
                    name="Volume_ROC",
                    value=float(volume_roc.iloc[-1]),
                    signal=vol_signal,
                    confidence=vol_confidence,
                    timeframe=timeframe,
                    timestamp=current_time
                ))
                
        except Exception as e:
            logger.error(f"Error calculating volume indicators: {e}")
            
        return indicators
    
    def _calculate_custom_indicators(self, df: pd.DataFrame, timeframe: TimeFrame,
                                   current_time: datetime) -> List[TechnicalIndicator]:
        """Calculate custom proprietary indicators"""
        indicators = []
        
        try:
            # Market Strength Index (Custom)
            msi = self._calculate_market_strength_index(df)
            if not pd.isna(msi.iloc[-1]):
                msi_signal = "BUY" if msi.iloc[-1] > 0.6 else "SELL" if msi.iloc[-1] < 0.4 else "NEUTRAL"
                msi_confidence = abs(msi.iloc[-1] - 0.5) * 2
                
                indicators.append(TechnicalIndicator(
                    name="Market_Strength_Index",
                    value=float(msi.iloc[-1]),
                    signal=msi_signal,
                    confidence=msi_confidence,
                    timeframe=timeframe,
                    timestamp=current_time
                ))
            
            # Momentum Oscillator (Custom)
            mom_osc = self._calculate_momentum_oscillator(df)
            if not pd.isna(mom_osc.iloc[-1]):
                mom_signal = "BUY" if mom_osc.iloc[-1] > 0 and mom_osc.iloc[-2] <= 0 else \
                            "SELL" if mom_osc.iloc[-1] < 0 and mom_osc.iloc[-2] >= 0 else "NEUTRAL"
                mom_confidence = min(abs(mom_osc.iloc[-1]), 1.0)
                
                indicators.append(TechnicalIndicator(
                    name="Momentum_Oscillator",
                    value=float(mom_osc.iloc[-1]),
                    signal=mom_signal,
                    confidence=mom_confidence,
                    timeframe=timeframe,
                    timestamp=current_time
                ))
            
            # Adaptive Moving Average (Custom)
            ama = self._calculate_adaptive_ma(df)
            if not pd.isna(ama.iloc[-1]):
                ama_signal = "BUY" if df['close'].iloc[-1] > ama.iloc[-1] else "SELL"
                ama_confidence = abs(df['close'].iloc[-1] - ama.iloc[-1]) / df['close'].iloc[-1]
                
                indicators.append(TechnicalIndicator(
                    name="Adaptive_MA",
                    value=float(ama.iloc[-1]),
                    signal=ama_signal,
                    confidence=min(ama_confidence * 50, 1.0),
                    timeframe=timeframe,
                    timestamp=current_time
                ))
                
        except Exception as e:
            logger.error(f"Error calculating custom indicators: {e}")
            
        return indicators
    
    def calculate_support_resistance(self, df: pd.DataFrame, timeframe: TimeFrame) -> List[SupportResistanceLevel]:
        """
        Calculate dynamic support and resistance levels
        
        Args:
            df: OHLCV data
            timeframe: Analysis timeframe
            
        Returns:
            List of support/resistance levels
        """
        logger.info(f"Calculating support/resistance levels for {timeframe.value}")
        
        levels = []
        
        try:
            # Pivot points method
            pivot_levels = self._calculate_pivot_points(df)
            levels.extend(pivot_levels)
            
            # Fractal method
            fractal_levels = self._calculate_fractal_levels(df, timeframe)
            levels.extend(fractal_levels)
            
            # Volume profile method
            volume_levels = self._calculate_volume_profile_levels(df, timeframe)
            levels.extend(volume_levels)
            
            # Fibonacci retracement levels
            fib_levels = self._calculate_fibonacci_levels(df, timeframe)
            levels.extend(fib_levels)
            
            # Filter and rank levels by strength
            levels = self._filter_and_rank_levels(levels, df['close'].iloc[-1])
            
            logger.info(f"Found {len(levels)} support/resistance levels")
            
        except Exception as e:
            logger.error(f"Error calculating support/resistance: {e}")
            
        return levels
    
    def analyze_elliott_wave(self, df: pd.DataFrame, timeframe: TimeFrame) -> Optional[ElliottWave]:
        """
        Perform Elliott Wave analysis
        
        Args:
            df: OHLCV data
            timeframe: Analysis timeframe
            
        Returns:
            Elliott Wave analysis result
        """
        logger.info(f"Performing Elliott Wave analysis for {timeframe.value}")
        
        try:
            if len(df) < 100:  # Need sufficient data for wave analysis
                logger.warning("Insufficient data for Elliott Wave analysis")
                return None
            
            # Identify swing highs and lows
            swings = self._identify_swings(df)
            
            if len(swings) < 8:  # Need at least 8 swings for 5-wave pattern
                logger.warning("Insufficient swing points for Elliott Wave analysis")
                return None
            
            # Analyze wave patterns
            wave_analysis = self._analyze_wave_patterns(swings, df)
            
            if wave_analysis:
                logger.info(f"Elliott Wave detected: {wave_analysis.current_wave} wave")
                return wave_analysis
            else:
                logger.info("No clear Elliott Wave pattern detected")
                return None
                
        except Exception as e:
            logger.error(f"Error in Elliott Wave analysis: {e}")
            return None
    
    def _cross_timeframe_analysis(self, results: Dict[str, List[TechnicalIndicator]]) -> List[TechnicalIndicator]:
        """Perform cross-timeframe analysis for stronger signals"""
        cross_signals = []
        current_time = datetime.now()
        
        try:
            # Look for confluences across timeframes
            indicator_names = set()
            for indicators in results.values():
                indicator_names.update([ind.name for ind in indicators])
            
            for indicator_name in indicator_names:
                timeframe_signals = {}
                
                for tf, indicators in results.items():
                    for ind in indicators:
                        if ind.name == indicator_name:
                            timeframe_signals[tf] = ind
                            break
                
                if len(timeframe_signals) >= 2:  # At least 2 timeframes agree
                    confluence = self._calculate_confluence(timeframe_signals)
                    if confluence['confidence'] > 0.7:
                        cross_signals.append(TechnicalIndicator(
                            name=f"CrossTF_{indicator_name}",
                            value=confluence['value'],
                            signal=confluence['signal'],
                            confidence=confluence['confidence'],
                            timeframe=TimeFrame.D1,  # Representative timeframe
                            timestamp=current_time
                        ))
                        
        except Exception as e:
            logger.error(f"Error in cross-timeframe analysis: {e}")
            
        return cross_signals
    
    # Core calculation methods
    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_macd(self, prices: pd.Series, fast: int, slow: int, signal: int) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    def _calculate_stochastic(self, df: pd.DataFrame, k_period: int, d_period: int) -> Tuple[pd.Series, pd.Series]:
        """Calculate Stochastic oscillator"""
        lowest_low = df['low'].rolling(window=k_period).min()
        highest_high = df['high'].rolling(window=k_period).max()
        
        k_percent = 100 * ((df['close'] - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_period).mean()
        
        return k_percent, d_percent
    
    def _calculate_adx(self, df: pd.DataFrame, period: int) -> pd.Series:
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
    
    def _calculate_parabolic_sar(self, df: pd.DataFrame, af_start: float = 0.02, af_increment: float = 0.02, af_max: float = 0.2) -> pd.Series:
        """Calculate Parabolic SAR"""
        high = df['high'].values
        low = df['low'].values
        
        sar = np.zeros(len(df))
        trend = np.zeros(len(df))
        af = np.zeros(len(df))
        ep = np.zeros(len(df))
        
        # Initialize
        sar[0] = low[0]
        trend[0] = 1
        af[0] = af_start
        ep[0] = high[0]
        
        for i in range(1, len(df)):
            if trend[i-1] == 1:  # Uptrend
                sar[i] = sar[i-1] + af[i-1] * (ep[i-1] - sar[i-1])
                
                if high[i] > ep[i-1]:
                    ep[i] = high[i]
                    af[i] = min(af[i-1] + af_increment, af_max)
                else:
                    ep[i] = ep[i-1]
                    af[i] = af[i-1]
                
                if low[i] <= sar[i]:
                    trend[i] = -1
                    sar[i] = ep[i-1]
                    af[i] = af_start
                    ep[i] = low[i]
                else:
                    trend[i] = 1
            else:  # Downtrend
                sar[i] = sar[i-1] + af[i-1] * (ep[i-1] - sar[i-1])
                
                if low[i] < ep[i-1]:
                    ep[i] = low[i]
                    af[i] = min(af[i-1] + af_increment, af_max)
                else:
                    ep[i] = ep[i-1]
                    af[i] = af[i-1]
                
                if high[i] >= sar[i]:
                    trend[i] = 1
                    sar[i] = ep[i-1]
                    af[i] = af_start
                    ep[i] = high[i]
                else:
                    trend[i] = -1
        
        return pd.Series(sar, index=df.index)
    
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
    
    def _calculate_obv(self, df: pd.DataFrame) -> pd.Series:
        """Calculate On-Balance Volume"""
        obv = np.zeros(len(df))
        obv[0] = df['volume'].iloc[0]
        
        for i in range(1, len(df)):
            if df['close'].iloc[i] > df['close'].iloc[i-1]:
                obv[i] = obv[i-1] + df['volume'].iloc[i]
            elif df['close'].iloc[i] < df['close'].iloc[i-1]:
                obv[i] = obv[i-1] - df['volume'].iloc[i]
            else:
                obv[i] = obv[i-1]
        
        return pd.Series(obv, index=df.index)
    
    def _calculate_market_strength_index(self, df: pd.DataFrame) -> pd.Series:
        """Custom Market Strength Index"""
        # Combine multiple factors: trend, momentum, volume
        price_change = df['close'].pct_change(20)
        volume_ratio = df['volume'] / df['volume'].rolling(20).mean()
        volatility = df['close'].rolling(20).std() / df['close'].rolling(20).mean()
        
        # Normalize and combine
        price_norm = (price_change - price_change.rolling(50).mean()) / price_change.rolling(50).std()
        volume_norm = (volume_ratio - 1) / volume_ratio.rolling(50).std()
        vol_norm = -volatility / volatility.rolling(50).mean()  # Lower volatility is better
        
        # Weighted combination
        msi = (0.5 * price_norm + 0.3 * volume_norm + 0.2 * vol_norm)
        
        # Convert to 0-1 scale
        msi_scaled = (msi - msi.rolling(100).min()) / (msi.rolling(100).max() - msi.rolling(100).min())
        
        return msi_scaled
    
    def _calculate_momentum_oscillator(self, df: pd.DataFrame) -> pd.Series:
        """Custom Momentum Oscillator"""
        # Multiple momentum timeframes
        mom_5 = df['close'].pct_change(5)
        mom_10 = df['close'].pct_change(10)
        mom_20 = df['close'].pct_change(20)
        
        # Weighted average with recent periods having more weight
        momentum = (0.5 * mom_5 + 0.3 * mom_10 + 0.2 * mom_20)
        
        # Smooth with exponential moving average
        momentum_smooth = momentum.ewm(span=5).mean()
        
        return momentum_smooth
    
    def _calculate_adaptive_ma(self, df: pd.DataFrame) -> pd.Series:
        """Adaptive Moving Average based on market volatility"""
        volatility = df['close'].rolling(20).std()
        vol_ratio = volatility / volatility.rolling(50).mean()
        
        # Adjust period based on volatility
        min_period = 5
        max_period = 50
        adaptive_period = min_period + (max_period - min_period) / (1 + vol_ratio)
        
        # Calculate adaptive MA
        ama = pd.Series(index=df.index, dtype=float)
        ama.iloc[0] = df['close'].iloc[0]
        
        for i in range(1, len(df)):
            period = int(adaptive_period.iloc[i]) if not pd.isna(adaptive_period.iloc[i]) else 20
            alpha = 2.0 / (period + 1)
            ama.iloc[i] = alpha * df['close'].iloc[i] + (1 - alpha) * ama.iloc[i-1]
        
        return ama
    
    def _calculate_pivot_points(self, df: pd.DataFrame) -> List[SupportResistanceLevel]:
        """Calculate pivot point levels"""
        levels = []
        
        try:
            # Get last complete day data
            last_high = df['high'].iloc[-1]
            last_low = df['low'].iloc[-1]
            last_close = df['close'].iloc[-1]
            
            # Classic pivot point
            pivot = (last_high + last_low + last_close) / 3
            
            # Support and resistance levels
            r1 = 2 * pivot - last_low
            r2 = pivot + (last_high - last_low)
            r3 = last_high + 2 * (pivot - last_low)
            
            s1 = 2 * pivot - last_high
            s2 = pivot - (last_high - last_low)
            s3 = last_low - 2 * (last_high - pivot)
            
            # Create level objects
            pivot_levels = [
                (pivot, "PIVOT", 0.8),
                (r1, "RESISTANCE", 0.7),
                (r2, "RESISTANCE", 0.6),
                (r3, "RESISTANCE", 0.5),
                (s1, "SUPPORT", 0.7),
                (s2, "SUPPORT", 0.6),
                (s3, "SUPPORT", 0.5)
            ]
            
            for level, level_type, strength in pivot_levels:
                levels.append(SupportResistanceLevel(
                    level=level,
                    strength=strength,
                    touches=1,
                    level_type=level_type,
                    confidence=strength,
                    timeframe=TimeFrame.D1
                ))
                
        except Exception as e:
            logger.error(f"Error calculating pivot points: {e}")
            
        return levels
    
    def _calculate_fractal_levels(self, df: pd.DataFrame, timeframe: TimeFrame) -> List[SupportResistanceLevel]:
        """Calculate fractal-based support/resistance levels"""
        levels = []
        
        try:
            # Find fractal highs and lows (5-period fractals)
            fractal_highs = []
            fractal_lows = []
            
            for i in range(2, len(df) - 2):
                # Fractal high: current high is higher than 2 highs before and after
                if (df['high'].iloc[i] > df['high'].iloc[i-2] and 
                    df['high'].iloc[i] > df['high'].iloc[i-1] and
                    df['high'].iloc[i] > df['high'].iloc[i+1] and 
                    df['high'].iloc[i] > df['high'].iloc[i+2]):
                    fractal_highs.append((i, df['high'].iloc[i]))
                
                # Fractal low: current low is lower than 2 lows before and after
                if (df['low'].iloc[i] < df['low'].iloc[i-2] and 
                    df['low'].iloc[i] < df['low'].iloc[i-1] and
                    df['low'].iloc[i] < df['low'].iloc[i+1] and 
                    df['low'].iloc[i] < df['low'].iloc[i+2]):
                    fractal_lows.append((i, df['low'].iloc[i]))
            
            # Convert to support/resistance levels
            for idx, level in fractal_highs[-10:]:  # Last 10 fractal highs
                levels.append(SupportResistanceLevel(
                    level=level,
                    strength=0.6,
                    touches=1,
                    level_type="RESISTANCE",
                    confidence=0.6,
                    timeframe=timeframe
                ))
            
            for idx, level in fractal_lows[-10:]:  # Last 10 fractal lows
                levels.append(SupportResistanceLevel(
                    level=level,
                    strength=0.6,
                    touches=1,
                    level_type="SUPPORT",
                    confidence=0.6,
                    timeframe=timeframe
                ))
                
        except Exception as e:
            logger.error(f"Error calculating fractal levels: {e}")
            
        return levels
    
    def _calculate_volume_profile_levels(self, df: pd.DataFrame, timeframe: TimeFrame) -> List[SupportResistanceLevel]:
        """Calculate volume profile-based levels"""
        levels = []
        
        if 'volume' not in df.columns:
            return levels
            
        try:
            # Create price bins
            price_min = df['low'].min()
            price_max = df['high'].max()
            bins = np.linspace(price_min, price_max, 50)
            
            # Calculate volume at each price level
            volume_profile = np.zeros(len(bins) - 1)
            
            for i in range(len(df)):
                # Distribute volume across the price range for this candle
                low_bin = np.digitize(df['low'].iloc[i], bins) - 1
                high_bin = np.digitize(df['high'].iloc[i], bins) - 1
                
                if low_bin == high_bin:
                    if 0 <= low_bin < len(volume_profile):
                        volume_profile[low_bin] += df['volume'].iloc[i]
                else:
                    # Distribute volume proportionally
                    bin_range = max(1, high_bin - low_bin)
                    for j in range(low_bin, min(high_bin + 1, len(volume_profile))):
                        if j >= 0:
                            volume_profile[j] += df['volume'].iloc[i] / bin_range
            
            # Find high volume areas (potential support/resistance)
            volume_threshold = np.percentile(volume_profile, 80)
            
            for i, vol in enumerate(volume_profile):
                if vol > volume_threshold:
                    price_level = (bins[i] + bins[i+1]) / 2
                    current_price = df['close'].iloc[-1]
                    
                    level_type = "SUPPORT" if price_level < current_price else "RESISTANCE"
                    strength = min(vol / volume_profile.max(), 1.0)
                    
                    levels.append(SupportResistanceLevel(
                        level=price_level,
                        strength=strength,
                        touches=1,
                        level_type=level_type,
                        confidence=strength,
                        timeframe=timeframe
                    ))
                    
        except Exception as e:
            logger.error(f"Error calculating volume profile levels: {e}")
            
        return levels
    
    def _calculate_fibonacci_levels(self, df: pd.DataFrame, timeframe: TimeFrame) -> List[SupportResistanceLevel]:
        """Calculate Fibonacci retracement levels"""
        levels = []
        
        try:
            # Find significant swing high and low in recent data
            lookback = min(100, len(df))
            recent_data = df.tail(lookback)
            
            swing_high = recent_data['high'].max()
            swing_low = recent_data['low'].min()
            
            # Fibonacci retracement levels
            fib_levels = [0.236, 0.382, 0.5, 0.618, 0.786]
            
            for fib_ratio in fib_levels:
                # Uptrend retracement
                fib_level = swing_high - (swing_high - swing_low) * fib_ratio
                
                current_price = df['close'].iloc[-1]
                level_type = "SUPPORT" if fib_level < current_price else "RESISTANCE"
                
                levels.append(SupportResistanceLevel(
                    level=fib_level,
                    strength=0.5 + fib_ratio * 0.3,  # Higher ratios get more strength
                    touches=1,
                    level_type=level_type,
                    confidence=0.5 + fib_ratio * 0.3,
                    timeframe=timeframe
                ))
                
        except Exception as e:
            logger.error(f"Error calculating Fibonacci levels: {e}")
            
        return levels
    
    def _filter_and_rank_levels(self, levels: List[SupportResistanceLevel], current_price: float) -> List[SupportResistanceLevel]:
        """Filter and rank support/resistance levels by relevance"""
        if not levels:
            return levels
            
        # Remove levels too far from current price (more than 10%)
        filtered_levels = []
        price_threshold = current_price * 0.1
        
        for level in levels:
            if abs(level.level - current_price) <= price_threshold:
                filtered_levels.append(level)
        
        # Merge nearby levels (within 0.5% of each other)
        merged_levels = []
        merge_threshold = current_price * 0.005
        
        sorted_levels = sorted(filtered_levels, key=lambda x: x.level)
        
        i = 0
        while i < len(sorted_levels):
            current_level = sorted_levels[i]
            merge_group = [current_level]
            
            j = i + 1
            while j < len(sorted_levels) and abs(sorted_levels[j].level - current_level.level) <= merge_threshold:
                merge_group.append(sorted_levels[j])
                j += 1
            
            if len(merge_group) > 1:
                # Merge levels
                avg_level = sum(l.level for l in merge_group) / len(merge_group)
                avg_strength = sum(l.strength for l in merge_group) / len(merge_group)
                total_touches = sum(l.touches for l in merge_group)
                
                merged_level = SupportResistanceLevel(
                    level=avg_level,
                    strength=min(avg_strength * len(merge_group), 1.0),
                    touches=total_touches,
                    level_type=current_level.level_type,
                    confidence=min(avg_strength * len(merge_group), 1.0),
                    timeframe=current_level.timeframe
                )
                merged_levels.append(merged_level)
            else:
                merged_levels.append(current_level)
            
            i = j
        
        # Sort by strength and return top levels
        merged_levels.sort(key=lambda x: x.strength, reverse=True)
        return merged_levels[:20]  # Return top 20 levels
    
    def _identify_swings(self, df: pd.DataFrame) -> List[Tuple[int, float, str]]:
        """Identify swing highs and lows for Elliott Wave analysis"""
        swings = []
        
        try:
            # Use a more sophisticated swing detection
            window = 5
            
            for i in range(window, len(df) - window):
                # Check for swing high
                if all(df['high'].iloc[i] >= df['high'].iloc[i-j] for j in range(1, window+1)) and \
                   all(df['high'].iloc[i] >= df['high'].iloc[i+j] for j in range(1, window+1)):
                    swings.append((i, df['high'].iloc[i], 'HIGH'))
                
                # Check for swing low
                elif all(df['low'].iloc[i] <= df['low'].iloc[i-j] for j in range(1, window+1)) and \
                     all(df['low'].iloc[i] <= df['low'].iloc[i+j] for j in range(1, window+1)):
                    swings.append((i, df['low'].iloc[i], 'LOW'))
            
            # Filter swings to ensure alternating pattern
            filtered_swings = []
            last_type = None
            
            for swing in swings:
                if swing[2] != last_type:
                    filtered_swings.append(swing)
                    last_type = swing[2]
                else:
                    # Keep the more extreme swing
                    if len(filtered_swings) > 0:
                        last_swing = filtered_swings[-1]
                        if (swing[2] == 'HIGH' and swing[1] > last_swing[1]) or \
                           (swing[2] == 'LOW' and swing[1] < last_swing[1]):
                            filtered_swings[-1] = swing
            
            return filtered_swings
            
        except Exception as e:
            logger.error(f"Error identifying swings: {e}")
            return []
    
    def _analyze_wave_patterns(self, swings: List[Tuple[int, float, str]], df: pd.DataFrame) -> Optional[ElliottWave]:
        """Analyze Elliott Wave patterns from swing points"""
        try:
            if len(swings) < 8:  # Need at least 8 swings for a complete 5-3 pattern
                return None
            
            # Take the most recent swings for analysis
            recent_swings = swings[-8:]
            
            # Check for 5-wave impulse pattern
            impulse_wave = self._check_impulse_pattern(recent_swings)
            if impulse_wave:
                return impulse_wave
            
            # Check for 3-wave corrective pattern
            corrective_wave = self._check_corrective_pattern(recent_swings)
            if corrective_wave:
                return corrective_wave
            
            return None
            
        except Exception as e:
            logger.error(f"Error analyzing wave patterns: {e}")
            return None
    
    def _check_impulse_pattern(self, swings: List[Tuple[int, float, str]]) -> Optional[ElliottWave]:
        """Check for 5-wave impulse pattern"""
        try:
            if len(swings) < 6:  # Need at least 6 swings for 5 waves
                return None
            
            # Assume uptrend for simplicity (can be extended for downtrend)
            if swings[0][2] != 'LOW':  # Should start with a low
                return None
            
            # Extract wave levels
            wave_0 = swings[0][1]  # Start
            wave_1 = swings[1][1]  # Wave 1 high
            wave_2 = swings[2][1]  # Wave 2 low
            wave_3 = swings[3][1]  # Wave 3 high
            wave_4 = swings[4][1]  # Wave 4 low
            
            if len(swings) > 5:
                wave_5 = swings[5][1]  # Wave 5 high
            else:
                # Wave 5 in progress
                wave_5 = None
            
            # Elliott Wave rules validation
            # Rule 1: Wave 2 never retraces more than 100% of wave 1
            if wave_2 <= wave_0:
                return None
            
            # Rule 2: Wave 3 is never the shortest of waves 1, 3, and 5
            wave_1_length = wave_1 - wave_0
            wave_3_length = wave_3 - wave_2
            
            if wave_5:
                wave_5_length = wave_5 - wave_4
                if wave_3_length < wave_1_length and wave_3_length < wave_5_length:
                    return None
            
            # Rule 3: Wave 4 never enters the price territory of wave 1
            if wave_4 <= wave_1:
                return None
            
            # Determine current wave
            current_price = swings[-1][1]
            
            if not wave_5:
                current_wave = "5"
                completion_percent = 0.8  # Estimate
                next_target = wave_3 + (wave_3 - wave_2) * 0.618  # Fibonacci extension
            else:
                current_wave = "A"  # Corrective phase
                completion_percent = 1.0
                next_target = wave_5 - (wave_5 - wave_4) * 0.618
            
            confidence = 0.7  # Base confidence for impulse pattern
            
            return ElliottWave(
                wave_count=5,
                current_wave=current_wave,
                wave_degree="INTERMEDIATE",
                completion_percent=completion_percent,
                next_target=next_target,
                confidence=confidence
            )
            
        except Exception as e:
            logger.error(f"Error checking impulse pattern: {e}")
            return None
    
    def _check_corrective_pattern(self, swings: List[Tuple[int, float, str]]) -> Optional[ElliottWave]:
        """Check for 3-wave corrective pattern (A-B-C)"""
        try:
            if len(swings) < 4:  # Need at least 4 swings for 3 waves
                return None
            
            # Take last 4 swings for A-B-C pattern
            recent_swings = swings[-4:]
            
            # Simple A-B-C pattern detection
            wave_a_start = recent_swings[0][1]
            wave_a_end = recent_swings[1][1]
            wave_b_end = recent_swings[2][1]
            wave_c_end = recent_swings[3][1]
            
            # Basic validation for corrective pattern
            # Wave B should be a partial retracement of Wave A
            wave_a_length = abs(wave_a_end - wave_a_start)
            wave_b_retracement = abs(wave_b_end - wave_a_end) / wave_a_length
            
            if 0.2 <= wave_b_retracement <= 0.8:  # Valid retracement range
                current_wave = "C"
                completion_percent = 0.7
                
                # Estimate C wave target (often equals A wave length)
                next_target = wave_b_end + (wave_a_end - wave_a_start)
                
                confidence = 0.6  # Lower confidence for corrective patterns
                
                return ElliottWave(
                    wave_count=3,
                    current_wave=current_wave,
                    wave_degree="INTERMEDIATE",
                    completion_percent=completion_percent,
                    next_target=next_target,
                    confidence=confidence
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error checking corrective pattern: {e}")
            return None
    
    def _calculate_confluence(self, timeframe_signals: Dict[str, TechnicalIndicator]) -> Dict[str, Any]:
        """Calculate confluence across multiple timeframes"""
        signals = []
        values = []
        confidences = []
        
        for indicator in timeframe_signals.values():
            signals.append(indicator.signal)
            values.append(indicator.value)
            confidences.append(indicator.confidence)
        
        # Determine overall signal
        buy_count = signals.count('BUY')
        sell_count = signals.count('SELL')
        
        if buy_count > sell_count:
            overall_signal = 'BUY'
        elif sell_count > buy_count:
            overall_signal = 'SELL'
        else:
            overall_signal = 'NEUTRAL'
        
        # Calculate confluence confidence
        total_signals = len(signals)
        dominant_count = max(buy_count, sell_count)
        agreement_ratio = dominant_count / total_signals
        
        avg_confidence = sum(confidences) / len(confidences)
        confluence_confidence = agreement_ratio * avg_confidence
        
        return {
            'signal': overall_signal,
            'value': sum(values) / len(values),
            'confidence': confluence_confidence
        }


# Usage example and factory function
def create_advanced_indicators(config: Dict[str, Any] = None) -> AdvancedIndicators:
    """Create an instance of AdvancedIndicators with configuration"""
    return AdvancedIndicators(config)


# Export classes and functions
__all__ = [
    'AdvancedIndicators',
    'TechnicalIndicator', 
    'SupportResistanceLevel',
    'ElliottWave',
    'TimeFrame',
    'create_advanced_indicators'
]