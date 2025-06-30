"""
Market-Specific Data Validator
Specialized validation for financial market data including trading halts, price gaps, and volume anomalies
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta, time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import pytz

from .quality_checker import QualityIssue, QualityIssueType
from ..data_collectors.models import ExchangeType
from utils.logger import get_logger

logger = get_logger(__name__)


class MarketIssueType(Enum):
    """Market-specific issue types"""
    TRADING_HALT = "trading_halt"
    PRICE_GAP = "price_gap"
    VOLUME_ANOMALY = "volume_anomaly"
    SPREAD_ANOMALY = "spread_anomaly"
    AFTER_HOURS_TRADING = "after_hours_trading"
    CIRCUIT_BREAKER = "circuit_breaker"
    LOW_LIQUIDITY = "low_liquidity"


@dataclass
class MarketValidationResult:
    """Result of market-specific validation"""
    exchange: ExchangeType
    symbol: str
    validation_time: datetime
    issues: List[QualityIssue]
    trading_status: str
    market_health_score: float
    recommendations: List[str]


class MarketDataValidator:
    """Market-specific data validation for financial instruments"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Market-specific thresholds
        self.thresholds = {
            'max_price_gap_percent': self.config.get('max_price_gap_percent', 10.0),  # 10%
            'volume_spike_multiplier': self.config.get('volume_spike_multiplier', 5.0),  # 5x normal
            'min_volume_threshold': self.config.get('min_volume_threshold', 100),
            'max_spread_percent': self.config.get('max_spread_percent', 5.0),  # 5%
            'circuit_breaker_percent': self.config.get('circuit_breaker_percent', 20.0),  # 20%
            'low_liquidity_threshold': self.config.get('low_liquidity_threshold', 1000)
        }
        
        # Exchange trading hours (UTC)
        self.trading_hours = {
            ExchangeType.UPBIT: {
                'timezone': 'Asia/Seoul',
                'open': time(0, 0),  # 24/7 for crypto
                'close': time(23, 59),
                'days': list(range(7))  # All days
            },
            ExchangeType.KIS: {
                'timezone': 'Asia/Seoul', 
                'open': time(9, 0),
                'close': time(15, 30),
                'days': list(range(5))  # Monday to Friday
            },
            ExchangeType.ALPACA: {
                'timezone': 'America/New_York',
                'open': time(9, 30),
                'close': time(16, 0),
                'days': list(range(5))  # Monday to Friday
            }
        }
    
    def validate_market_data(self, df: pd.DataFrame, symbol: str, 
                           exchange: ExchangeType) -> MarketValidationResult:
        """Perform comprehensive market data validation"""
        logger.info(f"Starting market validation for {symbol} on {exchange.value}")
        
        validation_time = datetime.now()
        issues = []
        
        # 1. Check trading halts
        halt_issues = self._check_trading_halts(df, symbol, exchange)
        issues.extend(halt_issues)
        
        # 2. Detect price gaps
        gap_issues = self._detect_price_gaps(df, symbol)
        issues.extend(gap_issues)
        
        # 3. Analyze volume anomalies
        volume_issues = self._analyze_volume_anomalies(df, symbol)
        issues.extend(volume_issues)
        
        # 4. Check spread anomalies
        spread_issues = self._check_spread_anomalies(df, symbol)
        issues.extend(spread_issues)
        
        # 5. Validate trading hours
        hours_issues = self._validate_trading_hours(df, symbol, exchange)
        issues.extend(hours_issues)
        
        # 6. Detect circuit breakers
        circuit_issues = self._detect_circuit_breakers(df, symbol)
        issues.extend(circuit_issues)
        
        # 7. Check liquidity
        liquidity_issues = self._check_liquidity(df, symbol)
        issues.extend(liquidity_issues)
        
        # Calculate market health score
        market_health_score = self._calculate_market_health_score(issues, len(df))
        
        # Determine trading status
        trading_status = self._determine_trading_status(issues, df, exchange)
        
        # Generate recommendations
        recommendations = self._generate_market_recommendations(issues, market_health_score)
        
        result = MarketValidationResult(
            exchange=exchange,
            symbol=symbol,
            validation_time=validation_time,
            issues=issues,
            trading_status=trading_status,
            market_health_score=market_health_score,
            recommendations=recommendations
        )
        
        logger.info(f"Market validation completed. Health score: {market_health_score:.3f}, Issues: {len(issues)}")
        return result
    
    def _check_trading_halts(self, df: pd.DataFrame, symbol: str, 
                           exchange: ExchangeType) -> List[QualityIssue]:
        """Detect potential trading halts based on volume and price patterns"""
        issues = []
        
        if len(df) < 10:  # Need sufficient data
            return issues
        
        # Look for periods with zero volume
        if 'volume' in df.columns:
            zero_volume_periods = self._find_zero_volume_periods(df)
            
            for start_idx, end_idx, duration in zero_volume_periods:
                if duration > timedelta(minutes=30):  # Significant halt
                    issue = QualityIssue(
                        issue_type=QualityIssueType.INCONSISTENT,
                        field="volume",
                        value=0,
                        row_index=start_idx,
                        description=f"Potential trading halt detected: {duration} of zero volume",
                        severity="high" if duration > timedelta(hours=1) else "medium",
                        timestamp=datetime.now(),
                        suggestion="Verify if trading was officially halted during this period"
                    )
                    issues.append(issue)
        
        # Look for price stagnation
        if 'close' in df.columns:
            stagnation_periods = self._find_price_stagnation(df)
            
            for start_idx, end_idx, duration in stagnation_periods:
                if duration > timedelta(hours=1):
                    issue = QualityIssue(
                        issue_type=QualityIssueType.INCONSISTENT,
                        field="close",
                        value=df.loc[start_idx, 'close'],
                        row_index=start_idx,
                        description=f"Price stagnation detected: {duration} with no price movement",
                        severity="medium",
                        timestamp=datetime.now(),
                        suggestion="Check for trading halt or low activity period"
                    )
                    issues.append(issue)
        
        return issues
    
    def _detect_price_gaps(self, df: pd.DataFrame, symbol: str) -> List[QualityIssue]:
        """Detect significant price gaps between consecutive periods"""
        issues = []
        
        if 'close' not in df.columns or len(df) < 2:
            return issues
        
        # Calculate price changes
        df_sorted = df.sort_values('timestamp') if 'timestamp' in df.columns else df
        price_changes = df_sorted['close'].pct_change()
        
        # Find significant gaps
        gap_threshold = self.thresholds['max_price_gap_percent'] / 100
        significant_gaps = price_changes[abs(price_changes) > gap_threshold]
        
        for idx, gap in significant_gaps.items():
            if pd.isna(gap):
                continue
                
            current_price = df_sorted.loc[idx, 'close']
            previous_idx = df_sorted.index[df_sorted.index.get_loc(idx) - 1]
            previous_price = df_sorted.loc[previous_idx, 'close']
            
            gap_percent = gap * 100
            severity = self._get_gap_severity(abs(gap_percent))
            
            issue = QualityIssue(
                issue_type=QualityIssueType.OUTLIER,
                field="close",
                value=current_price,
                row_index=idx,
                description=f"Significant price gap: {gap_percent:.2f}% (from {previous_price} to {current_price})",
                severity=severity,
                timestamp=datetime.now(),
                suggestion="Investigate news or events that might explain the price gap"
            )
            issues.append(issue)
        
        return issues
    
    def _analyze_volume_anomalies(self, df: pd.DataFrame, symbol: str) -> List[QualityIssue]:
        """Detect unusual volume patterns"""
        issues = []
        
        if 'volume' not in df.columns or len(df) < 10:
            return issues
        
        # Calculate volume statistics
        volume_series = df['volume'].replace(0, np.nan)  # Exclude zero volumes
        if volume_series.isna().all():
            return issues
        
        mean_volume = volume_series.mean()
        median_volume = volume_series.median()
        std_volume = volume_series.std()
        
        # Detect volume spikes
        spike_threshold = mean_volume * self.thresholds['volume_spike_multiplier']
        volume_spikes = df[df['volume'] > spike_threshold]
        
        for idx, row in volume_spikes.iterrows():
            volume = row['volume']
            spike_ratio = volume / mean_volume
            
            issue = QualityIssue(
                issue_type=QualityIssueType.OUTLIER,
                field="volume",
                value=volume,
                row_index=idx,
                description=f"Volume spike detected: {volume:,.0f} ({spike_ratio:.1f}x normal volume)",
                severity="high" if spike_ratio > 10 else "medium",
                timestamp=datetime.now(),
                suggestion="Check for news announcements or unusual market activity"
            )
            issues.append(issue)
        
        # Detect abnormally low volume
        low_volume_threshold = self.thresholds['min_volume_threshold']
        low_volume_periods = df[df['volume'] < low_volume_threshold]
        
        for idx, row in low_volume_periods.iterrows():
            volume = row['volume']
            
            issue = QualityIssue(
                issue_type=QualityIssueType.INCONSISTENT,
                field="volume",
                value=volume,
                row_index=idx,
                description=f"Abnormally low volume: {volume:,.0f} (threshold: {low_volume_threshold:,.0f})",
                severity="medium",
                timestamp=datetime.now(),
                suggestion="Verify if market was active during this period"
            )
            issues.append(issue)
        
        return issues
    
    def _check_spread_anomalies(self, df: pd.DataFrame, symbol: str) -> List[QualityIssue]:
        """Check for unusual bid-ask spreads"""
        issues = []
        
        if not all(col in df.columns for col in ['bid', 'ask']):
            return issues
        
        # Calculate spreads
        df_copy = df.copy()
        df_copy['spread'] = df_copy['ask'] - df_copy['bid']
        df_copy['spread_percent'] = (df_copy['spread'] / df_copy['bid']) * 100
        
        # Find wide spreads
        wide_spread_threshold = self.thresholds['max_spread_percent']
        wide_spreads = df_copy[df_copy['spread_percent'] > wide_spread_threshold]
        
        for idx, row in wide_spreads.iterrows():
            spread_percent = row['spread_percent']
            
            issue = QualityIssue(
                issue_type=QualityIssueType.OUTLIER,
                field="spread",
                value=row['spread'],
                row_index=idx,
                description=f"Wide bid-ask spread: {spread_percent:.2f}% (bid: {row['bid']}, ask: {row['ask']})",
                severity="high" if spread_percent > 10 else "medium",
                timestamp=datetime.now(),
                suggestion="Check market liquidity and trading conditions"
            )
            issues.append(issue)
        
        return issues
    
    def _validate_trading_hours(self, df: pd.DataFrame, symbol: str, 
                              exchange: ExchangeType) -> List[QualityIssue]:
        """Validate data against exchange trading hours"""
        issues = []
        
        if 'timestamp' not in df.columns or exchange not in self.trading_hours:
            return issues
        
        trading_config = self.trading_hours[exchange]
        timezone = pytz.timezone(trading_config['timezone'])
        
        for idx, row in df.iterrows():
            timestamp = row['timestamp']
            if pd.isna(timestamp):
                continue
            
            # Convert to exchange timezone
            if timestamp.tzinfo is None:
                timestamp = pytz.utc.localize(timestamp)
            local_time = timestamp.astimezone(timezone)
            
            # Check if within trading hours
            is_trading_day = local_time.weekday() in trading_config['days']
            is_trading_time = (trading_config['open'] <= local_time.time() <= trading_config['close'])
            
            if not (is_trading_day and is_trading_time) and exchange != ExchangeType.UPBIT:  # Crypto trades 24/7
                issue = QualityIssue(
                    issue_type=QualityIssueType.INCONSISTENT,
                    field="timestamp",
                    value=timestamp,
                    row_index=idx,
                    description=f"Trading data outside market hours: {local_time}",
                    severity="medium",
                    timestamp=datetime.now(),
                    suggestion="Verify if after-hours or pre-market trading data"
                )
                issues.append(issue)
        
        return issues
    
    def _detect_circuit_breakers(self, df: pd.DataFrame, symbol: str) -> List[QualityIssue]:
        """Detect potential circuit breaker events"""
        issues = []
        
        if 'close' not in df.columns or len(df) < 2:
            return issues
        
        # Calculate intraday price movements
        if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
            # Calculate max intraday movement
            df_copy = df.copy()
            df_copy['max_up_move'] = ((df_copy['high'] - df_copy['open']) / df_copy['open']) * 100
            df_copy['max_down_move'] = ((df_copy['low'] - df_copy['open']) / df_copy['open']) * 100
            
            circuit_threshold = self.thresholds['circuit_breaker_percent']
            
            # Check for large movements
            large_moves = df_copy[
                (abs(df_copy['max_up_move']) > circuit_threshold) | 
                (abs(df_copy['max_down_move']) > circuit_threshold)
            ]
            
            for idx, row in large_moves.iterrows():
                max_move = max(abs(row['max_up_move']), abs(row['max_down_move']))
                
                issue = QualityIssue(
                    issue_type=QualityIssueType.OUTLIER,
                    field="price_movement",
                    value=max_move,
                    row_index=idx,
                    description=f"Potential circuit breaker event: {max_move:.2f}% intraday movement",
                    severity="critical" if max_move > 30 else "high",
                    timestamp=datetime.now(),
                    suggestion="Check for circuit breaker activation or significant news events"
                )
                issues.append(issue)
        
        return issues
    
    def _check_liquidity(self, df: pd.DataFrame, symbol: str) -> List[QualityIssue]:
        """Check market liquidity indicators"""
        issues = []
        
        if 'volume' not in df.columns:
            return issues
        
        # Check overall liquidity
        low_liquidity_threshold = self.thresholds['low_liquidity_threshold']
        total_volume = df['volume'].sum()
        avg_volume = df['volume'].mean()
        
        if avg_volume < low_liquidity_threshold:
            issue = QualityIssue(
                issue_type=QualityIssueType.INCONSISTENT,
                field="volume",
                value=avg_volume,
                row_index=0,
                description=f"Low liquidity detected: average volume {avg_volume:,.0f} below threshold {low_liquidity_threshold:,.0f}",
                severity="medium",
                timestamp=datetime.now(),
                suggestion="Consider liquidity risk in trading decisions"
            )
            issues.append(issue)
        
        # Check for liquidity dry-ups
        if len(df) > 5:
            rolling_volume = df['volume'].rolling(window=5).mean()
            volume_drops = rolling_volume.pct_change()
            
            significant_drops = volume_drops[volume_drops < -0.8]  # 80% drop
            for idx, drop in significant_drops.items():
                if pd.isna(drop):
                    continue
                
                issue = QualityIssue(
                    issue_type=QualityIssueType.INCONSISTENT,
                    field="volume",
                    value=rolling_volume.loc[idx],
                    row_index=idx,
                    description=f"Liquidity dry-up detected: {drop*100:.1f}% volume drop",
                    severity="high",
                    timestamp=datetime.now(),
                    suggestion="Monitor for market stress or institutional selling"
                )
                issues.append(issue)
        
        return issues
    
    def _find_zero_volume_periods(self, df: pd.DataFrame) -> List[Tuple[int, int, timedelta]]:
        """Find periods with zero trading volume"""
        periods = []
        
        if 'volume' not in df.columns or 'timestamp' not in df.columns:
            return periods
        
        zero_volume_mask = df['volume'] == 0
        zero_indices = df[zero_volume_mask].index.tolist()
        
        if not zero_indices:
            return periods
        
        # Group consecutive zero volume periods
        current_start = zero_indices[0]
        current_end = zero_indices[0]
        
        for i in range(1, len(zero_indices)):
            if zero_indices[i] == current_end + 1:
                current_end = zero_indices[i]
            else:
                # End of consecutive period
                start_time = df.loc[current_start, 'timestamp']
                end_time = df.loc[current_end, 'timestamp']
                duration = end_time - start_time
                periods.append((current_start, current_end, duration))
                
                current_start = zero_indices[i]
                current_end = zero_indices[i]
        
        # Add final period
        start_time = df.loc[current_start, 'timestamp']
        end_time = df.loc[current_end, 'timestamp']
        duration = end_time - start_time
        periods.append((current_start, current_end, duration))
        
        return periods
    
    def _find_price_stagnation(self, df: pd.DataFrame) -> List[Tuple[int, int, timedelta]]:
        """Find periods with no price movement"""
        periods = []
        
        if 'close' not in df.columns or 'timestamp' not in df.columns:
            return periods
        
        # Find consecutive periods with same price
        price_changes = df['close'].diff()
        stagnant_mask = price_changes == 0
        stagnant_indices = df[stagnant_mask].index.tolist()
        
        if not stagnant_indices:
            return periods
        
        # Group consecutive stagnant periods (similar to zero volume logic)
        current_start = stagnant_indices[0]
        current_end = stagnant_indices[0]
        
        for i in range(1, len(stagnant_indices)):
            if stagnant_indices[i] == current_end + 1:
                current_end = stagnant_indices[i]
            else:
                start_time = df.loc[current_start, 'timestamp']
                end_time = df.loc[current_end, 'timestamp']
                duration = end_time - start_time
                periods.append((current_start, current_end, duration))
                
                current_start = stagnant_indices[i]
                current_end = stagnant_indices[i]
        
        # Add final period
        start_time = df.loc[current_start, 'timestamp']
        end_time = df.loc[current_end, 'timestamp']
        duration = end_time - start_time
        periods.append((current_start, current_end, duration))
        
        return periods
    
    def _calculate_market_health_score(self, issues: List[QualityIssue], total_records: int) -> float:
        """Calculate market health score based on detected issues"""
        if total_records == 0:
            return 0.0
        
        # Weight market issues by severity
        severity_weights = {'low': 0.1, 'medium': 0.3, 'high': 0.7, 'critical': 1.0}
        
        # Additional weights for market-specific issues
        market_issue_weights = {
            MarketIssueType.TRADING_HALT: 1.5,
            MarketIssueType.CIRCUIT_BREAKER: 2.0,
            MarketIssueType.VOLUME_ANOMALY: 1.2,
            MarketIssueType.PRICE_GAP: 1.3
        }
        
        total_penalty = 0.0
        for issue in issues:
            base_penalty = severity_weights.get(issue.severity, 0.5)
            
            # Apply market-specific multiplier if applicable
            market_multiplier = 1.0
            for market_issue_type in MarketIssueType:
                if market_issue_type.value in issue.description.lower():
                    market_multiplier = market_issue_weights.get(market_issue_type, 1.0)
                    break
            
            total_penalty += base_penalty * market_multiplier
        
        # Normalize by dataset size
        max_possible_penalty = total_records * 2.0  # Adjusted for market issues
        penalty_ratio = min(total_penalty / max_possible_penalty, 1.0) if max_possible_penalty > 0 else 0.0
        
        health_score = max(1.0 - penalty_ratio, 0.0)
        return round(health_score, 3)
    
    def _determine_trading_status(self, issues: List[QualityIssue], df: pd.DataFrame, 
                                exchange: ExchangeType) -> str:
        """Determine current trading status based on issues"""
        critical_issues = [issue for issue in issues if issue.severity == 'critical']
        high_issues = [issue for issue in issues if issue.severity == 'high']
        
        # Check for trading halts
        halt_keywords = ['halt', 'suspended', 'circuit breaker']
        has_halt_issues = any(
            any(keyword in issue.description.lower() for keyword in halt_keywords)
            for issue in critical_issues + high_issues
        )
        
        if has_halt_issues:
            return "HALTED"
        elif len(critical_issues) > 0:
            return "CRITICAL"
        elif len(high_issues) > 5:  # Many high severity issues
            return "DEGRADED"
        elif len(df) == 0:
            return "NO_DATA"
        else:
            return "NORMAL"
    
    def _generate_market_recommendations(self, issues: List[QualityIssue], 
                                       health_score: float) -> List[str]:
        """Generate market-specific recommendations"""
        recommendations = []
        
        # Analyze issue patterns
        issue_types = [issue.issue_type for issue in issues]
        severities = [issue.severity for issue in issues]
        
        # Volume-related recommendations
        volume_issues = [issue for issue in issues if 'volume' in issue.field.lower()]
        if len(volume_issues) > 0:
            recommendations.append("Monitor volume patterns closely - unusual activity detected")
        
        # Price-related recommendations
        price_issues = [issue for issue in issues if any(keyword in issue.description.lower() 
                       for keyword in ['gap', 'circuit', 'spike'])]
        if len(price_issues) > 0:
            recommendations.append("Exercise caution with price-based strategies due to volatility")
        
        # Liquidity recommendations
        liquidity_issues = [issue for issue in issues if 'liquidity' in issue.description.lower()]
        if len(liquidity_issues) > 0:
            recommendations.append("Consider market impact when placing large orders")
        
        # Overall health recommendations
        if health_score < 0.5:
            recommendations.append("Market conditions are poor - consider reducing exposure")
        elif health_score < 0.7:
            recommendations.append("Market conditions are suboptimal - increase monitoring")
        
        # Trading halt recommendations
        halt_issues = [issue for issue in issues if 'halt' in issue.description.lower()]
        if len(halt_issues) > 0:
            recommendations.append("Trading may be restricted - verify exchange announcements")
        
        return recommendations
    
    def _get_gap_severity(self, gap_percent: float) -> str:
        """Determine gap severity based on percentage"""
        if gap_percent > 50:
            return 'critical'
        elif gap_percent > 20:
            return 'high'
        elif gap_percent > 10:
            return 'medium'
        else:
            return 'low'
    
    def get_market_summary(self, df: pd.DataFrame, symbol: str, 
                          exchange: ExchangeType) -> Dict[str, Any]:
        """Get comprehensive market summary"""
        summary = {
            'symbol': symbol,
            'exchange': exchange.value,
            'period_start': df['timestamp'].min() if 'timestamp' in df.columns else None,
            'period_end': df['timestamp'].max() if 'timestamp' in df.columns else None,
            'total_records': len(df),
            'trading_days': len(df['timestamp'].dt.date.unique()) if 'timestamp' in df.columns else None
        }
        
        # Price statistics
        if 'close' in df.columns:
            summary.update({
                'price_range': {
                    'min': df['close'].min(),
                    'max': df['close'].max(),
                    'current': df['close'].iloc[-1] if len(df) > 0 else None
                },
                'volatility': df['close'].pct_change().std() * np.sqrt(252) if len(df) > 1 else None
            })
        
        # Volume statistics
        if 'volume' in df.columns:
            summary.update({
                'volume_stats': {
                    'total': df['volume'].sum(),
                    'average': df['volume'].mean(),
                    'max_daily': df['volume'].max()
                }
            })
        
        return summary