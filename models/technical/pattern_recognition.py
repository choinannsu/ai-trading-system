"""
Advanced Pattern Recognition System
Candlestick patterns, chart patterns, harmonic patterns with confidence scoring
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import warnings
from datetime import datetime
from scipy.signal import find_peaks, find_peaks_cwt
from scipy.stats import linregress
from utils.logger import get_logger

logger = get_logger(__name__)

warnings.filterwarnings('ignore', category=RuntimeWarning)


class PatternType(Enum):
    """Pattern classification types"""
    CANDLESTICK = "candlestick"
    CHART = "chart"
    HARMONIC = "harmonic"


class PatternSignal(Enum):
    """Pattern trading signals"""
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"
    REVERSAL = "reversal"
    CONTINUATION = "continuation"


@dataclass
class PatternMatch:
    """Pattern recognition result"""
    pattern_name: str
    pattern_type: PatternType
    signal: PatternSignal
    confidence: float
    start_idx: int
    end_idx: int
    key_levels: List[float]
    target_price: Optional[float]
    stop_loss: Optional[float]
    description: str
    timestamp: datetime


@dataclass
class HarmonicRatio:
    """Harmonic pattern ratio validation"""
    name: str
    required_ratios: List[Tuple[str, float, float]]  # (point_pair, min_ratio, max_ratio)
    pattern_completion: float


class PatternRecognition:
    """Advanced pattern recognition system for financial markets"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Pattern recognition parameters
        self.min_pattern_length = self.config.get('min_pattern_length', 5)
        self.max_pattern_length = self.config.get('max_pattern_length', 50)
        self.confidence_threshold = self.config.get('confidence_threshold', 0.6)
        
        # Initialize pattern definitions
        self._initialize_harmonic_patterns()
        
    def recognize_all_patterns(self, df: pd.DataFrame) -> List[PatternMatch]:
        """
        Recognize all types of patterns in the given data
        
        Args:
            df: OHLCV DataFrame
            
        Returns:
            List of recognized patterns with confidence scores
        """
        logger.info("Starting comprehensive pattern recognition")
        
        all_patterns = []
        
        try:
            # Candlestick patterns
            candlestick_patterns = self.recognize_candlestick_patterns(df)
            all_patterns.extend(candlestick_patterns)
            
            # Chart patterns
            chart_patterns = self.recognize_chart_patterns(df)
            all_patterns.extend(chart_patterns)
            
            # Harmonic patterns
            harmonic_patterns = self.recognize_harmonic_patterns(df)
            all_patterns.extend(harmonic_patterns)
            
            # Sort by confidence and remove duplicates
            all_patterns = self._filter_and_rank_patterns(all_patterns)
            
            logger.info(f"Pattern recognition completed: {len(all_patterns)} patterns found")
            
        except Exception as e:
            logger.error(f"Error in pattern recognition: {e}")
            
        return all_patterns
    
    def recognize_candlestick_patterns(self, df: pd.DataFrame) -> List[PatternMatch]:
        """
        Recognize candlestick patterns (30+ patterns)
        
        Args:
            df: OHLCV DataFrame
            
        Returns:
            List of candlestick pattern matches
        """
        logger.debug("Recognizing candlestick patterns")
        
        patterns = []
        current_time = datetime.now()
        
        # Ensure minimum data length
        if len(df) < 10:
            return patterns
        
        try:
            # Single candle patterns
            patterns.extend(self._detect_single_candle_patterns(df, current_time))
            
            # Two candle patterns
            patterns.extend(self._detect_two_candle_patterns(df, current_time))
            
            # Three candle patterns
            patterns.extend(self._detect_three_candle_patterns(df, current_time))
            
            # Five candle patterns
            patterns.extend(self._detect_five_candle_patterns(df, current_time))
            
            logger.debug(f"Found {len(patterns)} candlestick patterns")
            
        except Exception as e:
            logger.error(f"Error recognizing candlestick patterns: {e}")
            
        return patterns
    
    def recognize_chart_patterns(self, df: pd.DataFrame) -> List[PatternMatch]:
        """
        Recognize chart patterns (triangles, flags, cup and handle, etc.)
        
        Args:
            df: OHLCV DataFrame
            
        Returns:
            List of chart pattern matches
        """
        logger.debug("Recognizing chart patterns")
        
        patterns = []
        current_time = datetime.now()
        
        if len(df) < 20:  # Need more data for chart patterns
            return patterns
        
        try:
            # Triangle patterns
            patterns.extend(self._detect_triangle_patterns(df, current_time))
            
            # Flag and pennant patterns
            patterns.extend(self._detect_flag_patterns(df, current_time))
            
            # Head and shoulders patterns
            patterns.extend(self._detect_head_shoulders_patterns(df, current_time))
            
            # Cup and handle patterns
            patterns.extend(self._detect_cup_handle_patterns(df, current_time))
            
            # Double top/bottom patterns
            patterns.extend(self._detect_double_patterns(df, current_time))
            
            # Wedge patterns
            patterns.extend(self._detect_wedge_patterns(df, current_time))
            
            # Channel patterns
            patterns.extend(self._detect_channel_patterns(df, current_time))
            
            logger.debug(f"Found {len(patterns)} chart patterns")
            
        except Exception as e:
            logger.error(f"Error recognizing chart patterns: {e}")
            
        return patterns
    
    def recognize_harmonic_patterns(self, df: pd.DataFrame) -> List[PatternMatch]:
        """
        Recognize harmonic patterns (Gartley, Bat, Crab, etc.)
        
        Args:
            df: OHLCV DataFrame
            
        Returns:
            List of harmonic pattern matches
        """
        logger.debug("Recognizing harmonic patterns")
        
        patterns = []
        current_time = datetime.now()
        
        if len(df) < 30:  # Need sufficient data for harmonic patterns
            return patterns
        
        try:
            # Find swing points for harmonic analysis
            swing_points = self._find_swing_points(df)
            
            if len(swing_points) < 5:  # Need at least 5 points for XABCD pattern
                return patterns
            
            # Check each harmonic pattern type
            for pattern_name, pattern_def in self.harmonic_patterns.items():
                pattern_matches = self._detect_harmonic_pattern(
                    swing_points, pattern_def, pattern_name, df, current_time
                )
                patterns.extend(pattern_matches)
            
            logger.debug(f"Found {len(patterns)} harmonic patterns")
            
        except Exception as e:
            logger.error(f"Error recognizing harmonic patterns: {e}")
            
        return patterns
    
    # Single Candle Pattern Detection
    def _detect_single_candle_patterns(self, df: pd.DataFrame, current_time: datetime) -> List[PatternMatch]:
        """Detect single candlestick patterns"""
        patterns = []
        
        for i in range(1, len(df)):  # Start from index 1 to have previous candle
            try:
                current = df.iloc[i]
                previous = df.iloc[i-1]
                
                # Doji patterns
                doji_pattern = self._check_doji(current, previous)
                if doji_pattern:
                    patterns.append(PatternMatch(
                        pattern_name=doji_pattern['name'],
                        pattern_type=PatternType.CANDLESTICK,
                        signal=doji_pattern['signal'],
                        confidence=doji_pattern['confidence'],
                        start_idx=i,
                        end_idx=i,
                        key_levels=[current['close']],
                        target_price=None,
                        stop_loss=None,
                        description=doji_pattern['description'],
                        timestamp=current_time
                    ))
                
                # Hammer patterns
                hammer_pattern = self._check_hammer(current, previous)
                if hammer_pattern:
                    patterns.append(PatternMatch(
                        pattern_name=hammer_pattern['name'],
                        pattern_type=PatternType.CANDLESTICK,
                        signal=hammer_pattern['signal'],
                        confidence=hammer_pattern['confidence'],
                        start_idx=i,
                        end_idx=i,
                        key_levels=[current['close'], current['low']],
                        target_price=hammer_pattern.get('target'),
                        stop_loss=hammer_pattern.get('stop_loss'),
                        description=hammer_pattern['description'],
                        timestamp=current_time
                    ))
                
                # Spinning top
                spinning_top = self._check_spinning_top(current)
                if spinning_top:
                    patterns.append(PatternMatch(
                        pattern_name="Spinning Top",
                        pattern_type=PatternType.CANDLESTICK,
                        signal=PatternSignal.NEUTRAL,
                        confidence=spinning_top['confidence'],
                        start_idx=i,
                        end_idx=i,
                        key_levels=[current['close']],
                        target_price=None,
                        stop_loss=None,
                        description="Indecision pattern indicating potential reversal",
                        timestamp=current_time
                    ))
                
                # Marubozu patterns
                marubozu = self._check_marubozu(current)
                if marubozu:
                    patterns.append(PatternMatch(
                        pattern_name=marubozu['name'],
                        pattern_type=PatternType.CANDLESTICK,
                        signal=marubozu['signal'],
                        confidence=marubozu['confidence'],
                        start_idx=i,
                        end_idx=i,
                        key_levels=[current['open'], current['close']],
                        target_price=None,
                        stop_loss=None,
                        description=marubozu['description'],
                        timestamp=current_time
                    ))
                
            except Exception as e:
                logger.debug(f"Error detecting single candle pattern at index {i}: {e}")
                continue
                
        return patterns
    
    def _detect_two_candle_patterns(self, df: pd.DataFrame, current_time: datetime) -> List[PatternMatch]:
        """Detect two-candle patterns"""
        patterns = []
        
        for i in range(1, len(df)):
            try:
                current = df.iloc[i]
                previous = df.iloc[i-1]
                
                # Engulfing patterns
                engulfing = self._check_engulfing(previous, current)
                if engulfing:
                    patterns.append(PatternMatch(
                        pattern_name=engulfing['name'],
                        pattern_type=PatternType.CANDLESTICK,
                        signal=engulfing['signal'],
                        confidence=engulfing['confidence'],
                        start_idx=i-1,
                        end_idx=i,
                        key_levels=[previous['close'], current['close']],
                        target_price=engulfing.get('target'),
                        stop_loss=engulfing.get('stop_loss'),
                        description=engulfing['description'],
                        timestamp=current_time
                    ))
                
                # Harami patterns
                harami = self._check_harami(previous, current)
                if harami:
                    patterns.append(PatternMatch(
                        pattern_name=harami['name'],
                        pattern_type=PatternType.CANDLESTICK,
                        signal=harami['signal'],
                        confidence=harami['confidence'],
                        start_idx=i-1,
                        end_idx=i,
                        key_levels=[previous['close'], current['close']],
                        target_price=None,
                        stop_loss=None,
                        description=harami['description'],
                        timestamp=current_time
                    ))
                
                # Piercing line / Dark cloud cover
                piercing = self._check_piercing_patterns(previous, current)
                if piercing:
                    patterns.append(PatternMatch(
                        pattern_name=piercing['name'],
                        pattern_type=PatternType.CANDLESTICK,
                        signal=piercing['signal'],
                        confidence=piercing['confidence'],
                        start_idx=i-1,
                        end_idx=i,
                        key_levels=[previous['close'], current['close']],
                        target_price=piercing.get('target'),
                        stop_loss=piercing.get('stop_loss'),
                        description=piercing['description'],
                        timestamp=current_time
                    ))
                
            except Exception as e:
                logger.debug(f"Error detecting two candle pattern at index {i}: {e}")
                continue
                
        return patterns
    
    def _detect_three_candle_patterns(self, df: pd.DataFrame, current_time: datetime) -> List[PatternMatch]:
        """Detect three-candle patterns"""
        patterns = []
        
        for i in range(2, len(df)):
            try:
                candles = [df.iloc[i-2], df.iloc[i-1], df.iloc[i]]
                
                # Morning/Evening star
                star_pattern = self._check_star_patterns(candles)
                if star_pattern:
                    patterns.append(PatternMatch(
                        pattern_name=star_pattern['name'],
                        pattern_type=PatternType.CANDLESTICK,
                        signal=star_pattern['signal'],
                        confidence=star_pattern['confidence'],
                        start_idx=i-2,
                        end_idx=i,
                        key_levels=[candles[0]['close'], candles[1]['close'], candles[2]['close']],
                        target_price=star_pattern.get('target'),
                        stop_loss=star_pattern.get('stop_loss'),
                        description=star_pattern['description'],
                        timestamp=current_time
                    ))
                
                # Three white soldiers / Three black crows
                soldiers_crows = self._check_soldiers_crows(candles)
                if soldiers_crows:
                    patterns.append(PatternMatch(
                        pattern_name=soldiers_crows['name'],
                        pattern_type=PatternType.CANDLESTICK,
                        signal=soldiers_crows['signal'],
                        confidence=soldiers_crows['confidence'],
                        start_idx=i-2,
                        end_idx=i,
                        key_levels=[c['close'] for c in candles],
                        target_price=soldiers_crows.get('target'),
                        stop_loss=soldiers_crows.get('stop_loss'),
                        description=soldiers_crows['description'],
                        timestamp=current_time
                    ))
                
                # Inside/Outside patterns
                inside_outside = self._check_inside_outside_patterns(candles)
                if inside_outside:
                    patterns.append(PatternMatch(
                        pattern_name=inside_outside['name'],
                        pattern_type=PatternType.CANDLESTICK,
                        signal=inside_outside['signal'],
                        confidence=inside_outside['confidence'],
                        start_idx=i-2,
                        end_idx=i,
                        key_levels=[c['close'] for c in candles],
                        target_price=None,
                        stop_loss=None,
                        description=inside_outside['description'],
                        timestamp=current_time
                    ))
                
            except Exception as e:
                logger.debug(f"Error detecting three candle pattern at index {i}: {e}")
                continue
                
        return patterns
    
    def _detect_five_candle_patterns(self, df: pd.DataFrame, current_time: datetime) -> List[PatternMatch]:
        """Detect five-candle patterns"""
        patterns = []
        
        for i in range(4, len(df)):
            try:
                candles = [df.iloc[j] for j in range(i-4, i+1)]
                
                # Three inside up/down with confirmation
                three_inside = self._check_three_inside_patterns(candles)
                if three_inside:
                    patterns.append(PatternMatch(
                        pattern_name=three_inside['name'],
                        pattern_type=PatternType.CANDLESTICK,
                        signal=three_inside['signal'],
                        confidence=three_inside['confidence'],
                        start_idx=i-4,
                        end_idx=i,
                        key_levels=[c['close'] for c in candles[-3:]],
                        target_price=three_inside.get('target'),
                        stop_loss=three_inside.get('stop_loss'),
                        description=three_inside['description'],
                        timestamp=current_time
                    ))
                
            except Exception as e:
                logger.debug(f"Error detecting five candle pattern at index {i}: {e}")
                continue
                
        return patterns
    
    # Chart Pattern Detection
    def _detect_triangle_patterns(self, df: pd.DataFrame, current_time: datetime) -> List[PatternMatch]:
        """Detect triangle patterns (ascending, descending, symmetrical)"""
        patterns = []
        
        try:
            # Find swing points
            swing_highs, swing_lows = self._find_swings_for_charts(df)
            
            if len(swing_highs) < 4 or len(swing_lows) < 4:
                return patterns
            
            # Look for triangle patterns in recent data
            for i in range(len(swing_highs) - 3):
                for j in range(len(swing_lows) - 3):
                    try:
                        # Take 4 consecutive swing points
                        highs = swing_highs[i:i+4]
                        lows = swing_lows[j:j+4]
                        
                        # Check if they form a triangle
                        triangle = self._analyze_triangle(highs, lows, df)
                        if triangle:
                            patterns.append(PatternMatch(
                                pattern_name=triangle['name'],
                                pattern_type=PatternType.CHART,
                                signal=triangle['signal'],
                                confidence=triangle['confidence'],
                                start_idx=triangle['start_idx'],
                                end_idx=triangle['end_idx'],
                                key_levels=triangle['key_levels'],
                                target_price=triangle.get('target'),
                                stop_loss=triangle.get('stop_loss'),
                                description=triangle['description'],
                                timestamp=current_time
                            ))
                    except:
                        continue
                        
        except Exception as e:
            logger.error(f"Error detecting triangle patterns: {e}")
            
        return patterns
    
    def _detect_flag_patterns(self, df: pd.DataFrame, current_time: datetime) -> List[PatternMatch]:
        """Detect flag and pennant patterns"""
        patterns = []
        
        try:
            # Flag patterns require a strong move followed by consolidation
            for i in range(20, len(df) - 10):  # Need history and future data
                # Check for strong move (flagpole)
                flagpole_start = i - 15
                flagpole_end = i
                
                flagpole_move = abs(df['close'].iloc[flagpole_end] - df['close'].iloc[flagpole_start])
                avg_range = df['high'].rolling(20).max().iloc[i] - df['low'].rolling(20).min().iloc[i]
                
                if flagpole_move < avg_range * 0.5:  # Not strong enough move
                    continue
                
                # Check for consolidation (flag)
                consolidation_data = df.iloc[i:i+10]
                flag_pattern = self._analyze_flag_consolidation(consolidation_data, flagpole_move > 0)
                
                if flag_pattern:
                    patterns.append(PatternMatch(
                        pattern_name=flag_pattern['name'],
                        pattern_type=PatternType.CHART,
                        signal=flag_pattern['signal'],
                        confidence=flag_pattern['confidence'],
                        start_idx=flagpole_start,
                        end_idx=i+10,
                        key_levels=flag_pattern['key_levels'],
                        target_price=flag_pattern.get('target'),
                        stop_loss=flag_pattern.get('stop_loss'),
                        description=flag_pattern['description'],
                        timestamp=current_time
                    ))
                    
        except Exception as e:
            logger.error(f"Error detecting flag patterns: {e}")
            
        return patterns
    
    def _detect_head_shoulders_patterns(self, df: pd.DataFrame, current_time: datetime) -> List[PatternMatch]:
        """Detect head and shoulders patterns"""
        patterns = []
        
        try:
            swing_highs, swing_lows = self._find_swings_for_charts(df)
            
            # Need at least 3 swing highs for head and shoulders
            if len(swing_highs) < 3:
                return patterns
            
            for i in range(len(swing_highs) - 2):
                try:
                    left_shoulder = swing_highs[i]
                    head = swing_highs[i+1]
                    right_shoulder = swing_highs[i+2]
                    
                    # Check head and shoulders criteria
                    hs_pattern = self._analyze_head_shoulders(left_shoulder, head, right_shoulder, swing_lows, df)
                    
                    if hs_pattern:
                        patterns.append(PatternMatch(
                            pattern_name=hs_pattern['name'],
                            pattern_type=PatternType.CHART,
                            signal=hs_pattern['signal'],
                            confidence=hs_pattern['confidence'],
                            start_idx=hs_pattern['start_idx'],
                            end_idx=hs_pattern['end_idx'],
                            key_levels=hs_pattern['key_levels'],
                            target_price=hs_pattern.get('target'),
                            stop_loss=hs_pattern.get('stop_loss'),
                            description=hs_pattern['description'],
                            timestamp=current_time
                        ))
                except:
                    continue
                    
        except Exception as e:
            logger.error(f"Error detecting head and shoulders patterns: {e}")
            
        return patterns
    
    def _detect_cup_handle_patterns(self, df: pd.DataFrame, current_time: datetime) -> List[PatternMatch]:
        """Detect cup and handle patterns"""
        patterns = []
        
        try:
            # Cup and handle requires substantial data
            if len(df) < 50:
                return patterns
            
            # Look for cup formation in recent data
            for i in range(30, len(df) - 20):
                cup_data = df.iloc[i-30:i]
                handle_data = df.iloc[i:i+20]
                
                cup_handle = self._analyze_cup_handle(cup_data, handle_data)
                
                if cup_handle:
                    patterns.append(PatternMatch(
                        pattern_name="Cup and Handle",
                        pattern_type=PatternType.CHART,
                        signal=PatternSignal.BULLISH,
                        confidence=cup_handle['confidence'],
                        start_idx=i-30,
                        end_idx=i+20,
                        key_levels=cup_handle['key_levels'],
                        target_price=cup_handle.get('target'),
                        stop_loss=cup_handle.get('stop_loss'),
                        description="Bullish continuation pattern with cup and handle formation",
                        timestamp=current_time
                    ))
                    
        except Exception as e:
            logger.error(f"Error detecting cup and handle patterns: {e}")
            
        return patterns
    
    def _detect_double_patterns(self, df: pd.DataFrame, current_time: datetime) -> List[PatternMatch]:
        """Detect double top and double bottom patterns"""
        patterns = []
        
        try:
            swing_highs, swing_lows = self._find_swings_for_charts(df)
            
            # Double tops
            if len(swing_highs) >= 2:
                for i in range(len(swing_highs) - 1):
                    for j in range(i + 1, len(swing_highs)):
                        double_top = self._analyze_double_top(swing_highs[i], swing_highs[j], swing_lows, df)
                        if double_top:
                            patterns.append(PatternMatch(
                                pattern_name="Double Top",
                                pattern_type=PatternType.CHART,
                                signal=PatternSignal.BEARISH,
                                confidence=double_top['confidence'],
                                start_idx=double_top['start_idx'],
                                end_idx=double_top['end_idx'],
                                key_levels=double_top['key_levels'],
                                target_price=double_top.get('target'),
                                stop_loss=double_top.get('stop_loss'),
                                description="Bearish reversal pattern with two similar highs",
                                timestamp=current_time
                            ))
            
            # Double bottoms
            if len(swing_lows) >= 2:
                for i in range(len(swing_lows) - 1):
                    for j in range(i + 1, len(swing_lows)):
                        double_bottom = self._analyze_double_bottom(swing_lows[i], swing_lows[j], swing_highs, df)
                        if double_bottom:
                            patterns.append(PatternMatch(
                                pattern_name="Double Bottom",
                                pattern_type=PatternType.CHART,
                                signal=PatternSignal.BULLISH,
                                confidence=double_bottom['confidence'],
                                start_idx=double_bottom['start_idx'],
                                end_idx=double_bottom['end_idx'],
                                key_levels=double_bottom['key_levels'],
                                target_price=double_bottom.get('target'),
                                stop_loss=double_bottom.get('stop_loss'),
                                description="Bullish reversal pattern with two similar lows",
                                timestamp=current_time
                            ))
                            
        except Exception as e:
            logger.error(f"Error detecting double patterns: {e}")
            
        return patterns
    
    def _detect_wedge_patterns(self, df: pd.DataFrame, current_time: datetime) -> List[PatternMatch]:
        """Detect rising and falling wedge patterns"""
        patterns = []
        
        try:
            swing_highs, swing_lows = self._find_swings_for_charts(df)
            
            if len(swing_highs) < 4 or len(swing_lows) < 4:
                return patterns
            
            # Analyze recent swing points for wedge formation
            for i in range(len(swing_highs) - 3):
                for j in range(len(swing_lows) - 3):
                    highs = swing_highs[i:i+4]
                    lows = swing_lows[j:j+4]
                    
                    wedge = self._analyze_wedge(highs, lows, df)
                    if wedge:
                        patterns.append(PatternMatch(
                            pattern_name=wedge['name'],
                            pattern_type=PatternType.CHART,
                            signal=wedge['signal'],
                            confidence=wedge['confidence'],
                            start_idx=wedge['start_idx'],
                            end_idx=wedge['end_idx'],
                            key_levels=wedge['key_levels'],
                            target_price=wedge.get('target'),
                            stop_loss=wedge.get('stop_loss'),
                            description=wedge['description'],
                            timestamp=current_time
                        ))
                        
        except Exception as e:
            logger.error(f"Error detecting wedge patterns: {e}")
            
        return patterns
    
    def _detect_channel_patterns(self, df: pd.DataFrame, current_time: datetime) -> List[PatternMatch]:
        """Detect channel patterns (ascending, descending, horizontal)"""
        patterns = []
        
        try:
            # Look for channel patterns in recent data
            for window_size in [20, 30, 40]:
                if len(df) < window_size:
                    continue
                    
                for i in range(window_size, len(df)):
                    channel_data = df.iloc[i-window_size:i]
                    
                    channel = self._analyze_channel(channel_data)
                    if channel:
                        patterns.append(PatternMatch(
                            pattern_name=channel['name'],
                            pattern_type=PatternType.CHART,
                            signal=channel['signal'],
                            confidence=channel['confidence'],
                            start_idx=i-window_size,
                            end_idx=i,
                            key_levels=channel['key_levels'],
                            target_price=channel.get('target'),
                            stop_loss=channel.get('stop_loss'),
                            description=channel['description'],
                            timestamp=current_time
                        ))
                        
        except Exception as e:
            logger.error(f"Error detecting channel patterns: {e}")
            
        return patterns
    
    # Harmonic Pattern Detection
    def _initialize_harmonic_patterns(self):
        """Initialize harmonic pattern definitions"""
        self.harmonic_patterns = {
            'Gartley': HarmonicRatio(
                name='Gartley',
                required_ratios=[
                    ('XA_AB', 0.618, 0.618),  # AB = 61.8% of XA
                    ('AB_BC', 0.382, 0.886),  # BC = 38.2-88.6% of AB
                    ('XA_AD', 0.786, 0.786),  # AD = 78.6% of XA
                ],
                pattern_completion=0.786
            ),
            'Bat': HarmonicRatio(
                name='Bat',
                required_ratios=[
                    ('XA_AB', 0.382, 0.5),    # AB = 38.2-50% of XA
                    ('AB_BC', 0.382, 0.886),  # BC = 38.2-88.6% of AB
                    ('XA_AD', 0.886, 0.886),  # AD = 88.6% of XA
                ],
                pattern_completion=0.886
            ),
            'Butterfly': HarmonicRatio(
                name='Butterfly',
                required_ratios=[
                    ('XA_AB', 0.786, 0.786),  # AB = 78.6% of XA
                    ('AB_BC', 0.382, 0.886),  # BC = 38.2-88.6% of AB
                    ('XA_AD', 1.27, 1.618),   # AD = 127-161.8% of XA
                ],
                pattern_completion=1.27
            ),
            'Crab': HarmonicRatio(
                name='Crab',
                required_ratios=[
                    ('XA_AB', 0.382, 0.618),  # AB = 38.2-61.8% of XA
                    ('AB_BC', 0.382, 0.886),  # BC = 38.2-88.6% of AB
                    ('XA_AD', 1.618, 1.618),  # AD = 161.8% of XA
                ],
                pattern_completion=1.618
            ),
            'Shark': HarmonicRatio(
                name='Shark',
                required_ratios=[
                    ('XA_AB', 0.382, 0.618),  # AB = 38.2-61.8% of XA
                    ('AB_BC', 1.13, 1.618),   # BC = 113-161.8% of AB
                    ('XA_AD', 0.886, 0.886),  # AD = 88.6% of XA
                ],
                pattern_completion=0.886
            )
        }
    
    def _detect_harmonic_pattern(self, swing_points: List[Tuple[int, float, str]], 
                                pattern_def: HarmonicRatio, pattern_name: str,
                                df: pd.DataFrame, current_time: datetime) -> List[PatternMatch]:
        """Detect specific harmonic pattern"""
        patterns = []
        
        try:
            # Need at least 5 swing points for XABCD pattern
            if len(swing_points) < 5:
                return patterns
            
            # Check all possible 5-point combinations
            for i in range(len(swing_points) - 4):
                five_points = swing_points[i:i+5]
                
                # Ensure alternating high-low pattern
                if not self._validate_alternating_pattern(five_points):
                    continue
                
                # Extract XABCD points
                X = five_points[0]
                A = five_points[1]
                B = five_points[2]
                C = five_points[3]
                D = five_points[4]
                
                # Validate harmonic ratios
                ratios_valid, confidence = self._validate_harmonic_ratios(X, A, B, C, D, pattern_def)
                
                if ratios_valid and confidence > self.confidence_threshold:
                    # Determine pattern signal
                    signal = self._determine_harmonic_signal(pattern_name, five_points)
                    
                    # Calculate target and stop loss
                    target, stop_loss = self._calculate_harmonic_targets(X, A, B, C, D, signal)
                    
                    patterns.append(PatternMatch(
                        pattern_name=f"Harmonic {pattern_name}",
                        pattern_type=PatternType.HARMONIC,
                        signal=signal,
                        confidence=confidence,
                        start_idx=X[0],
                        end_idx=D[0],
                        key_levels=[point[1] for point in five_points],
                        target_price=target,
                        stop_loss=stop_loss,
                        description=f"Harmonic {pattern_name} pattern with {confidence:.1%} confidence",
                        timestamp=current_time
                    ))
                    
        except Exception as e:
            logger.debug(f"Error detecting {pattern_name} pattern: {e}")
            
        return patterns
    
    # Helper methods for pattern analysis
    def _find_swing_points(self, df: pd.DataFrame, window: int = 5) -> List[Tuple[int, float, str]]:
        """Find swing highs and lows for harmonic analysis"""
        swing_points = []
        
        try:
            # Find peaks and troughs
            high_peaks, _ = find_peaks(df['high'].values, distance=window)
            low_peaks, _ = find_peaks(-df['low'].values, distance=window)
            
            # Combine and sort by index
            all_swings = []
            
            for peak in high_peaks:
                all_swings.append((peak, df['high'].iloc[peak], 'HIGH'))
            
            for peak in low_peaks:
                all_swings.append((peak, df['low'].iloc[peak], 'LOW'))
            
            # Sort by index and ensure alternating pattern
            all_swings.sort(key=lambda x: x[0])
            
            # Filter to ensure alternating highs and lows
            if all_swings:
                swing_points = [all_swings[0]]
                
                for swing in all_swings[1:]:
                    if swing[2] != swing_points[-1][2]:  # Different type from last
                        swing_points.append(swing)
                    else:
                        # Keep the more extreme swing
                        if (swing[2] == 'HIGH' and swing[1] > swing_points[-1][1]) or \
                           (swing[2] == 'LOW' and swing[1] < swing_points[-1][1]):
                            swing_points[-1] = swing
                            
        except Exception as e:
            logger.error(f"Error finding swing points: {e}")
            
        return swing_points
    
    def _find_swings_for_charts(self, df: pd.DataFrame) -> Tuple[List[Tuple[int, float]], List[Tuple[int, float]]]:
        """Find swing highs and lows for chart pattern analysis"""
        try:
            # Use a smaller window for chart patterns
            window = 3
            
            # Find swing highs
            high_peaks, _ = find_peaks(df['high'].values, distance=window)
            swing_highs = [(peak, df['high'].iloc[peak]) for peak in high_peaks]
            
            # Find swing lows
            low_peaks, _ = find_peaks(-df['low'].values, distance=window)
            swing_lows = [(peak, df['low'].iloc[peak]) for peak in low_peaks]
            
            return swing_highs, swing_lows
            
        except Exception as e:
            logger.error(f"Error finding swings for charts: {e}")
            return [], []
    
    # Candlestick pattern detection methods
    def _check_doji(self, current: pd.Series, previous: pd.Series) -> Optional[Dict]:
        """Check for Doji patterns"""
        try:
            body_size = abs(current['close'] - current['open'])
            total_range = current['high'] - current['low']
            
            if total_range == 0:
                return None
            
            body_ratio = body_size / total_range
            
            # Doji: small body relative to total range
            if body_ratio < 0.1:
                upper_shadow = current['high'] - max(current['open'], current['close'])
                lower_shadow = min(current['open'], current['close']) - current['low']
                
                # Standard Doji
                if abs(upper_shadow - lower_shadow) / total_range < 0.2:
                    return {
                        'name': 'Doji',
                        'signal': PatternSignal.NEUTRAL,
                        'confidence': 0.7,
                        'description': 'Indecision pattern indicating potential reversal'
                    }
                
                # Dragonfly Doji
                elif lower_shadow > upper_shadow * 2:
                    return {
                        'name': 'Dragonfly Doji',
                        'signal': PatternSignal.BULLISH,
                        'confidence': 0.75,
                        'description': 'Bullish reversal pattern with long lower shadow'
                    }
                
                # Gravestone Doji
                elif upper_shadow > lower_shadow * 2:
                    return {
                        'name': 'Gravestone Doji',
                        'signal': PatternSignal.BEARISH,
                        'confidence': 0.75,
                        'description': 'Bearish reversal pattern with long upper shadow'
                    }
            
            return None
            
        except:
            return None
    
    def _check_hammer(self, current: pd.Series, previous: pd.Series) -> Optional[Dict]:
        """Check for Hammer and Hanging Man patterns"""
        try:
            body_size = abs(current['close'] - current['open'])
            total_range = current['high'] - current['low']
            
            if total_range == 0:
                return None
            
            upper_shadow = current['high'] - max(current['open'], current['close'])
            lower_shadow = min(current['open'], current['close']) - current['low']
            
            # Hammer/Hanging Man criteria
            if (lower_shadow > body_size * 2 and 
                upper_shadow < body_size * 0.5 and
                body_size / total_range > 0.1):
                
                # Determine if it's a hammer or hanging man based on trend
                prev_trend = previous['close'] > previous['open']  # Simplified trend detection
                
                if current['close'] > previous['close']:  # Bullish context
                    return {
                        'name': 'Hammer',
                        'signal': PatternSignal.BULLISH,
                        'confidence': 0.8,
                        'description': 'Bullish reversal pattern at bottom of downtrend',
                        'target': current['close'] + (current['close'] - current['low']),
                        'stop_loss': current['low']
                    }
                else:  # Bearish context
                    return {
                        'name': 'Hanging Man',
                        'signal': PatternSignal.BEARISH,
                        'confidence': 0.7,
                        'description': 'Bearish reversal pattern at top of uptrend',
                        'target': current['close'] - (current['high'] - current['close']),
                        'stop_loss': current['high']
                    }
            
            return None
            
        except:
            return None
    
    def _check_spinning_top(self, current: pd.Series) -> Optional[Dict]:
        """Check for Spinning Top pattern"""
        try:
            body_size = abs(current['close'] - current['open'])
            total_range = current['high'] - current['low']
            
            if total_range == 0:
                return None
            
            upper_shadow = current['high'] - max(current['open'], current['close'])
            lower_shadow = min(current['open'], current['close']) - current['low']
            
            # Spinning top: small body with roughly equal upper and lower shadows
            if (body_size / total_range < 0.3 and
                min(upper_shadow, lower_shadow) > body_size and
                abs(upper_shadow - lower_shadow) / total_range < 0.3):
                
                return {
                    'confidence': 0.6
                }
            
            return None
            
        except:
            return None
    
    def _check_marubozu(self, current: pd.Series) -> Optional[Dict]:
        """Check for Marubozu patterns"""
        try:
            body_size = abs(current['close'] - current['open'])
            total_range = current['high'] - current['low']
            
            if total_range == 0 or body_size == 0:
                return None
            
            # Marubozu: body takes up most of the total range
            if body_size / total_range > 0.9:
                if current['close'] > current['open']:
                    return {
                        'name': 'White Marubozu',
                        'signal': PatternSignal.BULLISH,
                        'confidence': 0.8,
                        'description': 'Strong bullish continuation pattern'
                    }
                else:
                    return {
                        'name': 'Black Marubozu',
                        'signal': PatternSignal.BEARISH,
                        'confidence': 0.8,
                        'description': 'Strong bearish continuation pattern'
                    }
            
            return None
            
        except:
            return None
    
    def _check_engulfing(self, previous: pd.Series, current: pd.Series) -> Optional[Dict]:
        """Check for Bullish/Bearish Engulfing patterns"""
        try:
            prev_bullish = previous['close'] > previous['open']
            curr_bullish = current['close'] > current['open']
            
            # Opposite colored candles
            if prev_bullish == curr_bullish:
                return None
            
            # Current candle engulfs previous candle
            if (current['open'] < min(previous['open'], previous['close']) and
                current['close'] > max(previous['open'], previous['close'])) or \
               (current['open'] > max(previous['open'], previous['close']) and
                current['close'] < min(previous['open'], previous['close'])):
                
                if curr_bullish:  # Bullish engulfing
                    return {
                        'name': 'Bullish Engulfing',
                        'signal': PatternSignal.BULLISH,
                        'confidence': 0.85,
                        'description': 'Bullish reversal pattern where current candle engulfs previous',
                        'target': current['close'] + abs(current['close'] - current['open']),
                        'stop_loss': current['low']
                    }
                else:  # Bearish engulfing
                    return {
                        'name': 'Bearish Engulfing',
                        'signal': PatternSignal.BEARISH,
                        'confidence': 0.85,
                        'description': 'Bearish reversal pattern where current candle engulfs previous',
                        'target': current['close'] - abs(current['open'] - current['close']),
                        'stop_loss': current['high']
                    }
            
            return None
            
        except:
            return None
    
    def _check_harami(self, previous: pd.Series, current: pd.Series) -> Optional[Dict]:
        """Check for Harami patterns"""
        try:
            prev_bullish = previous['close'] > previous['open']
            curr_bullish = current['close'] > current['open']
            
            # Opposite colored candles and current is inside previous
            if (prev_bullish != curr_bullish and
                current['open'] > min(previous['open'], previous['close']) and
                current['close'] < max(previous['open'], previous['close'])):
                
                if curr_bullish:  # Bullish harami
                    return {
                        'name': 'Bullish Harami',
                        'signal': PatternSignal.BULLISH,
                        'confidence': 0.7,
                        'description': 'Bullish reversal pattern with small candle inside large bearish candle'
                    }
                else:  # Bearish harami
                    return {
                        'name': 'Bearish Harami',
                        'signal': PatternSignal.BEARISH,
                        'confidence': 0.7,
                        'description': 'Bearish reversal pattern with small candle inside large bullish candle'
                    }
            
            return None
            
        except:
            return None
    
    def _check_piercing_patterns(self, previous: pd.Series, current: pd.Series) -> Optional[Dict]:
        """Check for Piercing Line and Dark Cloud Cover patterns"""
        try:
            prev_bullish = previous['close'] > previous['open']
            curr_bullish = current['close'] > current['open']
            
            # Piercing Line: bearish candle followed by bullish candle that opens below low and closes above midpoint
            if (not prev_bullish and curr_bullish and
                current['open'] < previous['low'] and
                current['close'] > (previous['open'] + previous['close']) / 2 and
                current['close'] < previous['open']):
                
                return {
                    'name': 'Piercing Line',
                    'signal': PatternSignal.BULLISH,
                    'confidence': 0.8,
                    'description': 'Bullish reversal pattern with strong upward momentum',
                    'target': current['close'] + (current['close'] - current['open']),
                    'stop_loss': current['low']
                }
            
            # Dark Cloud Cover: bullish candle followed by bearish candle that opens above high and closes below midpoint
            elif (prev_bullish and not curr_bullish and
                  current['open'] > previous['high'] and
                  current['close'] < (previous['open'] + previous['close']) / 2 and
                  current['close'] > previous['close']):
                
                return {
                    'name': 'Dark Cloud Cover',
                    'signal': PatternSignal.BEARISH,
                    'confidence': 0.8,
                    'description': 'Bearish reversal pattern with strong downward momentum',
                    'target': current['close'] - (current['open'] - current['close']),
                    'stop_loss': current['high']
                }
            
            return None
            
        except:
            return None
    
    def _check_star_patterns(self, candles: List[pd.Series]) -> Optional[Dict]:
        """Check for Morning Star and Evening Star patterns"""
        try:
            first, second, third = candles
            
            first_bullish = first['close'] > first['open']
            second_body = abs(second['close'] - second['open'])
            second_range = second['high'] - second['low']
            third_bullish = third['close'] > third['open']
            
            # Small middle candle (star)
            if second_range > 0 and second_body / second_range < 0.3:
                
                # Morning Star: bearish, small, bullish
                if (not first_bullish and third_bullish and
                    second['high'] < min(first['close'], third['open']) and
                    third['close'] > (first['open'] + first['close']) / 2):
                    
                    return {
                        'name': 'Morning Star',
                        'signal': PatternSignal.BULLISH,
                        'confidence': 0.85,
                        'description': 'Strong bullish reversal pattern with three candles',
                        'target': third['close'] + (first['open'] - first['close']),
                        'stop_loss': min(first['low'], second['low'], third['low'])
                    }
                
                # Evening Star: bullish, small, bearish
                elif (first_bullish and not third_bullish and
                      second['low'] > max(first['close'], third['open']) and
                      third['close'] < (first['open'] + first['close']) / 2):
                    
                    return {
                        'name': 'Evening Star',
                        'signal': PatternSignal.BEARISH,
                        'confidence': 0.85,
                        'description': 'Strong bearish reversal pattern with three candles',
                        'target': third['close'] - (first['close'] - first['open']),
                        'stop_loss': max(first['high'], second['high'], third['high'])
                    }
            
            return None
            
        except:
            return None
    
    def _check_soldiers_crows(self, candles: List[pd.Series]) -> Optional[Dict]:
        """Check for Three White Soldiers and Three Black Crows patterns"""
        try:
            # Check if all three candles are the same color
            bullish_candles = [c['close'] > c['open'] for c in candles]
            
            if all(bullish_candles):  # Three White Soldiers
                # Each candle should open within the body of the previous and close higher
                if (candles[1]['open'] > candles[0]['open'] and candles[1]['open'] < candles[0]['close'] and
                    candles[2]['open'] > candles[1]['open'] and candles[2]['open'] < candles[1]['close'] and
                    candles[1]['close'] > candles[0]['close'] and
                    candles[2]['close'] > candles[1]['close']):
                    
                    return {
                        'name': 'Three White Soldiers',
                        'signal': PatternSignal.BULLISH,
                        'confidence': 0.8,
                        'description': 'Strong bullish continuation pattern with three consecutive bullish candles',
                        'target': candles[2]['close'] + (candles[2]['close'] - candles[0]['open']),
                        'stop_loss': min(c['low'] for c in candles)
                    }
            
            elif not any(bullish_candles):  # Three Black Crows
                # Each candle should open within the body of the previous and close lower
                if (candles[1]['open'] < candles[0]['open'] and candles[1]['open'] > candles[0]['close'] and
                    candles[2]['open'] < candles[1]['open'] and candles[2]['open'] > candles[1]['close'] and
                    candles[1]['close'] < candles[0]['close'] and
                    candles[2]['close'] < candles[1]['close']):
                    
                    return {
                        'name': 'Three Black Crows',
                        'signal': PatternSignal.BEARISH,
                        'confidence': 0.8,
                        'description': 'Strong bearish continuation pattern with three consecutive bearish candles',
                        'target': candles[2]['close'] - (candles[0]['open'] - candles[2]['close']),
                        'stop_loss': max(c['high'] for c in candles)
                    }
            
            return None
            
        except:
            return None
    
    def _check_inside_outside_patterns(self, candles: List[pd.Series]) -> Optional[Dict]:
        """Check for Inside and Outside patterns"""
        try:
            first, second, third = candles
            
            # Inside pattern: second candle is completely within the first
            if (second['high'] <= first['high'] and second['low'] >= first['low'] and
                third['close'] != second['close']):  # Third candle shows direction
                
                if third['close'] > second['high']:
                    return {
                        'name': 'Inside Up',
                        'signal': PatternSignal.BULLISH,
                        'confidence': 0.7,
                        'description': 'Bullish breakout from inside pattern'
                    }
                elif third['close'] < second['low']:
                    return {
                        'name': 'Inside Down',
                        'signal': PatternSignal.BEARISH,
                        'confidence': 0.7,
                        'description': 'Bearish breakdown from inside pattern'
                    }
            
            # Outside pattern: second candle completely engulfs the first
            elif (second['high'] >= first['high'] and second['low'] <= first['low'] and
                  abs(second['close'] - second['open']) > abs(first['close'] - first['open'])):
                
                if third['close'] > second['close']:
                    return {
                        'name': 'Outside Up',
                        'signal': PatternSignal.BULLISH,
                        'confidence': 0.75,
                        'description': 'Bullish continuation from outside pattern'
                    }
                elif third['close'] < second['close']:
                    return {
                        'name': 'Outside Down',
                        'signal': PatternSignal.BEARISH,
                        'confidence': 0.75,
                        'description': 'Bearish continuation from outside pattern'
                    }
            
            return None
            
        except:
            return None
    
    def _check_three_inside_patterns(self, candles: List[pd.Series]) -> Optional[Dict]:
        """Check for Three Inside Up/Down patterns with confirmation"""
        try:
            # Use first three candles for the pattern, last two for confirmation
            pattern_candles = candles[:3]
            confirmation_candles = candles[3:]
            
            # First check if we have a basic inside pattern
            inside_pattern = self._check_inside_outside_patterns(pattern_candles)
            
            if inside_pattern and len(confirmation_candles) >= 2:
                # Check for confirmation in the next candles
                if inside_pattern['signal'] == PatternSignal.BULLISH:
                    # Confirm with continued upward movement
                    if all(c['close'] > pattern_candles[2]['close'] for c in confirmation_candles):
                        return {
                            'name': 'Three Inside Up Confirmed',
                            'signal': PatternSignal.BULLISH,
                            'confidence': 0.85,
                            'description': 'Confirmed bullish reversal with inside pattern and continuation',
                            'target': confirmation_candles[-1]['close'] + (confirmation_candles[-1]['close'] - pattern_candles[0]['low']),
                            'stop_loss': pattern_candles[0]['low']
                        }
                
                elif inside_pattern['signal'] == PatternSignal.BEARISH:
                    # Confirm with continued downward movement
                    if all(c['close'] < pattern_candles[2]['close'] for c in confirmation_candles):
                        return {
                            'name': 'Three Inside Down Confirmed',
                            'signal': PatternSignal.BEARISH,
                            'confidence': 0.85,
                            'description': 'Confirmed bearish reversal with inside pattern and continuation',
                            'target': confirmation_candles[-1]['close'] - (pattern_candles[0]['high'] - confirmation_candles[-1]['close']),
                            'stop_loss': pattern_candles[0]['high']
                        }
            
            return None
            
        except:
            return None
    
    # Chart pattern analysis methods
    def _analyze_triangle(self, highs: List[Tuple[int, float]], lows: List[Tuple[int, float]], df: pd.DataFrame) -> Optional[Dict]:
        """Analyze triangle pattern formation"""
        try:
            if len(highs) < 2 or len(lows) < 2:
                return None
            
            # Calculate trend lines for highs and lows
            high_slope, high_intercept, high_r, _, _ = linregress([h[0] for h in highs], [h[1] for h in highs])
            low_slope, low_intercept, low_r, _, _ = linregress([l[0] for l in lows], [l[1] for l in lows])
            
            # Check for triangle patterns based on slopes
            convergence_threshold = 0.8  # R-squared threshold for trend line validity
            
            if abs(high_r) > convergence_threshold and abs(low_r) > convergence_threshold:
                # Ascending triangle: flat resistance, rising support
                if abs(high_slope) < 0.001 and low_slope > 0:
                    return {
                        'name': 'Ascending Triangle',
                        'signal': PatternSignal.BULLISH,
                        'confidence': min(abs(high_r), abs(low_r)),
                        'start_idx': min(highs[0][0], lows[0][0]),
                        'end_idx': max(highs[-1][0], lows[-1][0]),
                        'key_levels': [highs[-1][1], lows[0][1], lows[-1][1]],
                        'target': highs[-1][1] + (highs[-1][1] - lows[0][1]),
                        'stop_loss': lows[-1][1],
                        'description': 'Bullish triangle with flat resistance and rising support'
                    }
                
                # Descending triangle: declining resistance, flat support
                elif high_slope < 0 and abs(low_slope) < 0.001:
                    return {
                        'name': 'Descending Triangle',
                        'signal': PatternSignal.BEARISH,
                        'confidence': min(abs(high_r), abs(low_r)),
                        'start_idx': min(highs[0][0], lows[0][0]),
                        'end_idx': max(highs[-1][0], lows[-1][0]),
                        'key_levels': [highs[0][1], highs[-1][1], lows[-1][1]],
                        'target': lows[-1][1] - (highs[0][1] - lows[-1][1]),
                        'stop_loss': highs[-1][1],
                        'description': 'Bearish triangle with declining resistance and flat support'
                    }
                
                # Symmetrical triangle: converging trend lines
                elif high_slope < 0 and low_slope > 0 and abs(high_slope + low_slope) < abs(high_slope) * 0.5:
                    return {
                        'name': 'Symmetrical Triangle',
                        'signal': PatternSignal.NEUTRAL,
                        'confidence': min(abs(high_r), abs(low_r)) * 0.8,  # Lower confidence for neutral
                        'start_idx': min(highs[0][0], lows[0][0]),
                        'end_idx': max(highs[-1][0], lows[-1][0]),
                        'key_levels': [highs[0][1], highs[-1][1], lows[0][1], lows[-1][1]],
                        'description': 'Neutral triangle pattern awaiting breakout direction'
                    }
            
            return None
            
        except:
            return None
    
    def _analyze_flag_consolidation(self, consolidation_data: pd.DataFrame, uptrend: bool) -> Optional[Dict]:
        """Analyze flag/pennant consolidation pattern"""
        try:
            if len(consolidation_data) < 5:
                return None
            
            # Calculate consolidation characteristics
            price_range = consolidation_data['high'].max() - consolidation_data['low'].min()
            avg_price = consolidation_data['close'].mean()
            
            # Check for tight consolidation
            if price_range / avg_price < 0.05:  # Less than 5% range
                # Check for slight counter-trend in consolidation
                first_half = consolidation_data.iloc[:len(consolidation_data)//2]
                second_half = consolidation_data.iloc[len(consolidation_data)//2:]
                
                trend_change = second_half['close'].mean() - first_half['close'].mean()
                
                if uptrend:  # Bull flag
                    if trend_change <= 0:  # Slight decline or sideways in consolidation
                        return {
                            'name': 'Bull Flag',
                            'signal': PatternSignal.BULLISH,
                            'confidence': 0.75,
                            'key_levels': [consolidation_data['high'].max(), consolidation_data['low'].min()],
                            'target': consolidation_data['close'].iloc[-1] + price_range * 2,
                            'stop_loss': consolidation_data['low'].min(),
                            'description': 'Bullish continuation pattern after strong upward move'
                        }
                else:  # Bear flag
                    if trend_change >= 0:  # Slight rise or sideways in consolidation
                        return {
                            'name': 'Bear Flag',
                            'signal': PatternSignal.BEARISH,
                            'confidence': 0.75,
                            'key_levels': [consolidation_data['high'].max(), consolidation_data['low'].min()],
                            'target': consolidation_data['close'].iloc[-1] - price_range * 2,
                            'stop_loss': consolidation_data['high'].max(),
                            'description': 'Bearish continuation pattern after strong downward move'
                        }
            
            return None
            
        except:
            return None
    
    def _analyze_head_shoulders(self, left_shoulder: Tuple[int, float], head: Tuple[int, float], 
                               right_shoulder: Tuple[int, float], swing_lows: List[Tuple[int, float]], 
                               df: pd.DataFrame) -> Optional[Dict]:
        """Analyze head and shoulders pattern"""
        try:
            # Head should be higher than both shoulders
            if not (head[1] > left_shoulder[1] and head[1] > right_shoulder[1]):
                return None
            
            # Shoulders should be roughly at similar levels (within 5%)
            shoulder_diff = abs(left_shoulder[1] - right_shoulder[1]) / max(left_shoulder[1], right_shoulder[1])
            if shoulder_diff > 0.05:
                return None
            
            # Find neckline (lows between shoulders and head)
            neckline_lows = [low for low in swing_lows 
                           if left_shoulder[0] < low[0] < right_shoulder[0]]
            
            if len(neckline_lows) < 1:
                return None
            
            # Calculate neckline level
            neckline_level = sum(low[1] for low in neckline_lows) / len(neckline_lows)
            
            # Head should be significantly above neckline
            head_height = head[1] - neckline_level
            if head_height < (head[1] * 0.02):  # At least 2% above neckline
                return None
            
            # Calculate target (head height projected below neckline)
            target_price = neckline_level - head_height
            
            return {
                'name': 'Head and Shoulders',
                'signal': PatternSignal.BEARISH,
                'confidence': 0.8,
                'start_idx': left_shoulder[0],
                'end_idx': right_shoulder[0],
                'key_levels': [left_shoulder[1], head[1], right_shoulder[1], neckline_level],
                'target': target_price,
                'stop_loss': head[1],
                'description': 'Bearish reversal pattern with head higher than two shoulders'
            }
            
        except:
            return None
    
    def _analyze_cup_handle(self, cup_data: pd.DataFrame, handle_data: pd.DataFrame) -> Optional[Dict]:
        """Analyze cup and handle pattern"""
        try:
            # Cup should show a rounded bottom
            cup_low = cup_data['low'].min()
            cup_high_start = cup_data['high'].iloc[0]
            cup_high_end = cup_data['high'].iloc[-1]
            
            # Cup should be roughly U-shaped (not V-shaped)
            cup_depth = min(cup_high_start, cup_high_end) - cup_low
            if cup_depth < (cup_high_start * 0.15):  # At least 15% depth
                return None
            
            # Handle should be a small consolidation near the cup rim
            handle_high = handle_data['high'].max()
            handle_low = handle_data['low'].min()
            handle_range = handle_high - handle_low
            
            # Handle should be smaller than cup (typically 1/3 or less)
            if handle_range > cup_depth * 0.33:
                return None
            
            # Handle should be in upper half of cup
            if handle_low < cup_low + cup_depth * 0.5:
                return None
            
            # Calculate target (cup depth added to breakout point)
            breakout_level = handle_high
            target_price = breakout_level + cup_depth
            
            return {
                'confidence': 0.75,
                'key_levels': [cup_low, cup_high_start, cup_high_end, handle_high, handle_low],
                'target': target_price,
                'stop_loss': handle_low
            }
            
        except:
            return None
    
    def _analyze_double_top(self, first_top: Tuple[int, float], second_top: Tuple[int, float], 
                           swing_lows: List[Tuple[int, float]], df: pd.DataFrame) -> Optional[Dict]:
        """Analyze double top pattern"""
        try:
            # Tops should be at similar levels (within 3%)
            top_diff = abs(first_top[1] - second_top[1]) / max(first_top[1], second_top[1])
            if top_diff > 0.03:
                return None
            
            # Find the valley between the two tops
            valley_lows = [low for low in swing_lows 
                          if first_top[0] < low[0] < second_top[0]]
            
            if not valley_lows:
                return None
            
            valley_level = min(low[1] for low in valley_lows)
            
            # Valley should be significantly below tops
            top_level = (first_top[1] + second_top[1]) / 2
            if (top_level - valley_level) < (top_level * 0.05):  # At least 5% difference
                return None
            
            # Calculate target
            pattern_height = top_level - valley_level
            target_price = valley_level - pattern_height
            
            return {
                'confidence': 0.8,
                'start_idx': first_top[0],
                'end_idx': second_top[0],
                'key_levels': [first_top[1], second_top[1], valley_level],
                'target': target_price,
                'stop_loss': top_level
            }
            
        except:
            return None
    
    def _analyze_double_bottom(self, first_bottom: Tuple[int, float], second_bottom: Tuple[int, float], 
                              swing_highs: List[Tuple[int, float]], df: pd.DataFrame) -> Optional[Dict]:
        """Analyze double bottom pattern"""
        try:
            # Bottoms should be at similar levels (within 3%)
            bottom_diff = abs(first_bottom[1] - second_bottom[1]) / max(first_bottom[1], second_bottom[1])
            if bottom_diff > 0.03:
                return None
            
            # Find the peak between the two bottoms
            peak_highs = [high for high in swing_highs 
                         if first_bottom[0] < high[0] < second_bottom[0]]
            
            if not peak_highs:
                return None
            
            peak_level = max(high[1] for high in peak_highs)
            
            # Peak should be significantly above bottoms
            bottom_level = (first_bottom[1] + second_bottom[1]) / 2
            if (peak_level - bottom_level) < (peak_level * 0.05):  # At least 5% difference
                return None
            
            # Calculate target
            pattern_height = peak_level - bottom_level
            target_price = peak_level + pattern_height
            
            return {
                'confidence': 0.8,
                'start_idx': first_bottom[0],
                'end_idx': second_bottom[0],
                'key_levels': [first_bottom[1], second_bottom[1], peak_level],
                'target': target_price,
                'stop_loss': bottom_level
            }
            
        except:
            return None
    
    def _analyze_wedge(self, highs: List[Tuple[int, float]], lows: List[Tuple[int, float]], df: pd.DataFrame) -> Optional[Dict]:
        """Analyze rising and falling wedge patterns"""
        try:
            if len(highs) < 3 or len(lows) < 3:
                return None
            
            # Calculate trend lines
            high_slope, _, high_r, _, _ = linregress([h[0] for h in highs], [h[1] for h in highs])
            low_slope, _, low_r, _, _ = linregress([l[0] for l in lows], [l[1] for l in lows])
            
            # Both trend lines should be significant
            if abs(high_r) < 0.7 or abs(low_r) < 0.7:
                return None
            
            # Rising wedge: both lines slope up, but resistance rises slower than support
            if high_slope > 0 and low_slope > 0 and low_slope > high_slope:
                return {
                    'name': 'Rising Wedge',
                    'signal': PatternSignal.BEARISH,
                    'confidence': min(abs(high_r), abs(low_r)),
                    'start_idx': min(highs[0][0], lows[0][0]),
                    'end_idx': max(highs[-1][0], lows[-1][0]),
                    'key_levels': [highs[0][1], highs[-1][1], lows[0][1], lows[-1][1]],
                    'target': lows[0][1] - (highs[-1][1] - lows[0][1]),
                    'stop_loss': highs[-1][1],
                    'description': 'Bearish pattern with converging upward trend lines'
                }
            
            # Falling wedge: both lines slope down, but support falls slower than resistance
            elif high_slope < 0 and low_slope < 0 and high_slope < low_slope:
                return {
                    'name': 'Falling Wedge',
                    'signal': PatternSignal.BULLISH,
                    'confidence': min(abs(high_r), abs(low_r)),
                    'start_idx': min(highs[0][0], lows[0][0]),
                    'end_idx': max(highs[-1][0], lows[-1][0]),
                    'key_levels': [highs[0][1], highs[-1][1], lows[0][1], lows[-1][1]],
                    'target': highs[0][1] + (highs[0][1] - lows[-1][1]),
                    'stop_loss': lows[-1][1],
                    'description': 'Bullish pattern with converging downward trend lines'
                }
            
            return None
            
        except:
            return None
    
    def _analyze_channel(self, data: pd.DataFrame) -> Optional[Dict]:
        """Analyze channel patterns"""
        try:
            if len(data) < 10:
                return None
            
            # Find highs and lows
            highs = []
            lows = []
            
            for i in range(2, len(data) - 2):
                if (data['high'].iloc[i] > data['high'].iloc[i-1] and 
                    data['high'].iloc[i] > data['high'].iloc[i+1]):
                    highs.append((i, data['high'].iloc[i]))
                
                if (data['low'].iloc[i] < data['low'].iloc[i-1] and 
                    data['low'].iloc[i] < data['low'].iloc[i+1]):
                    lows.append((i, data['low'].iloc[i]))
            
            if len(highs) < 3 or len(lows) < 3:
                return None
            
            # Calculate trend lines
            high_slope, _, high_r, _, _ = linregress([h[0] for h in highs], [h[1] for h in highs])
            low_slope, _, low_r, _, _ = linregress([l[0] for l in lows], [l[1] for l in lows])
            
            # Check for parallel lines (channel)
            if (abs(high_r) > 0.6 and abs(low_r) > 0.6 and 
                abs(high_slope - low_slope) < abs(high_slope) * 0.3):
                
                if high_slope > 0.001:  # Ascending channel
                    return {
                        'name': 'Ascending Channel',
                        'signal': PatternSignal.BULLISH,
                        'confidence': min(abs(high_r), abs(low_r)),
                        'key_levels': [highs[0][1], highs[-1][1], lows[0][1], lows[-1][1]],
                        'description': 'Bullish channel with parallel upward trend lines'
                    }
                elif high_slope < -0.001:  # Descending channel
                    return {
                        'name': 'Descending Channel',
                        'signal': PatternSignal.BEARISH,
                        'confidence': min(abs(high_r), abs(low_r)),
                        'key_levels': [highs[0][1], highs[-1][1], lows[0][1], lows[-1][1]],
                        'description': 'Bearish channel with parallel downward trend lines'
                    }
                else:  # Horizontal channel
                    return {
                        'name': 'Horizontal Channel',
                        'signal': PatternSignal.NEUTRAL,
                        'confidence': min(abs(high_r), abs(low_r)),
                        'key_levels': [highs[0][1], lows[0][1]],
                        'description': 'Sideways channel with horizontal support and resistance'
                    }
            
            return None
            
        except:
            return None
    
    # Harmonic pattern analysis methods
    def _validate_alternating_pattern(self, points: List[Tuple[int, float, str]]) -> bool:
        """Validate that swing points alternate between highs and lows"""
        for i in range(1, len(points)):
            if points[i][2] == points[i-1][2]:  # Same type as previous
                return False
        return True
    
    def _validate_harmonic_ratios(self, X: Tuple, A: Tuple, B: Tuple, C: Tuple, D: Tuple, 
                                 pattern_def: HarmonicRatio) -> Tuple[bool, float]:
        """Validate harmonic pattern ratios"""
        try:
            # Calculate actual ratios
            XA = abs(A[1] - X[1])
            AB = abs(B[1] - A[1])
            BC = abs(C[1] - B[1])
            CD = abs(D[1] - C[1])
            XD = abs(D[1] - X[1])
            
            if XA == 0 or AB == 0 or XA == 0:  # Avoid division by zero
                return False, 0.0
            
            # Calculate ratios
            ratios = {
                'XA_AB': AB / XA,
                'AB_BC': BC / AB,
                'XA_AD': XD / XA,
                'BC_CD': CD / BC if BC != 0 else 0
            }
            
            # Check each required ratio
            valid_ratios = 0
            total_ratios = len(pattern_def.required_ratios)
            
            for ratio_name, min_ratio, max_ratio in pattern_def.required_ratios:
                if ratio_name in ratios:
                    actual_ratio = ratios[ratio_name]
                    if min_ratio <= actual_ratio <= max_ratio:
                        valid_ratios += 1
                    elif abs(actual_ratio - min_ratio) / min_ratio < 0.1:  # 10% tolerance
                        valid_ratios += 0.5
                    elif abs(actual_ratio - max_ratio) / max_ratio < 0.1:  # 10% tolerance
                        valid_ratios += 0.5
            
            # Calculate confidence based on ratio accuracy
            confidence = valid_ratios / total_ratios
            
            return confidence > 0.7, confidence
            
        except:
            return False, 0.0
    
    def _determine_harmonic_signal(self, pattern_name: str, points: List[Tuple]) -> PatternSignal:
        """Determine trading signal for harmonic pattern"""
        # Check if pattern is bullish or bearish based on overall structure
        start_price = points[0][1]
        end_price = points[-1][1]
        
        # Most harmonic patterns are reversal patterns
        if end_price > start_price:
            return PatternSignal.BEARISH  # Expecting reversal down
        else:
            return PatternSignal.BULLISH  # Expecting reversal up
    
    def _calculate_harmonic_targets(self, X: Tuple, A: Tuple, B: Tuple, C: Tuple, D: Tuple, 
                                   signal: PatternSignal) -> Tuple[Optional[float], Optional[float]]:
        """Calculate target and stop loss for harmonic patterns"""
        try:
            XA = abs(A[1] - X[1])
            
            if signal == PatternSignal.BULLISH:
                # Target: 38.2% or 61.8% retracement of XA from D
                target = D[1] + (XA * 0.618)
                stop_loss = D[1] - (XA * 0.1)  # 10% of XA below D
            else:  # BEARISH
                # Target: 38.2% or 61.8% retracement of XA from D
                target = D[1] - (XA * 0.618)
                stop_loss = D[1] + (XA * 0.1)  # 10% of XA above D
            
            return target, stop_loss
            
        except:
            return None, None
    
    def _filter_and_rank_patterns(self, patterns: List[PatternMatch]) -> List[PatternMatch]:
        """Filter and rank patterns by confidence and relevance"""
        if not patterns:
            return patterns
        
        # Remove low confidence patterns
        filtered_patterns = [p for p in patterns if p.confidence >= self.confidence_threshold]
        
        # Remove overlapping patterns (keep highest confidence)
        final_patterns = []
        
        for pattern in filtered_patterns:
            # Check for overlap with existing patterns
            overlap_found = False
            
            for existing in final_patterns:
                # Check if patterns overlap significantly
                overlap_start = max(pattern.start_idx, existing.start_idx)
                overlap_end = min(pattern.end_idx, existing.end_idx)
                
                if overlap_end > overlap_start:  # There is overlap
                    overlap_length = overlap_end - overlap_start
                    pattern_length = pattern.end_idx - pattern.start_idx
                    
                    if overlap_length > pattern_length * 0.5:  # More than 50% overlap
                        if pattern.confidence > existing.confidence:
                            # Replace existing with higher confidence pattern
                            final_patterns.remove(existing)
                            final_patterns.append(pattern)
                        overlap_found = True
                        break
            
            if not overlap_found:
                final_patterns.append(pattern)
        
        # Sort by confidence (highest first)
        final_patterns.sort(key=lambda x: x.confidence, reverse=True)
        
        # Return top patterns
        return final_patterns[:20]  # Limit to top 20 patterns


# Usage example and factory function
def create_pattern_recognition(config: Dict[str, Any] = None) -> PatternRecognition:
    """Create an instance of PatternRecognition with configuration"""
    return PatternRecognition(config)


# Export classes and functions
__all__ = [
    'PatternRecognition',
    'PatternMatch',
    'PatternType',
    'PatternSignal',
    'HarmonicRatio',
    'create_pattern_recognition'
]