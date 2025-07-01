"""
Pairs Trading Strategy Implementation
Statistical arbitrage strategy based on mean-reverting relationships between correlated assets
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from scipy import stats
from sklearn.linear_model import LinearRegression
from itertools import combinations

from ..base_strategy import BaseStrategy, StrategyConfig, Signal, SignalType
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class PairsTradingConfig(StrategyConfig):
    """Configuration for pairs trading strategy"""
    name: str = "PairsTradingStrategy"
    
    # Pair selection parameters
    min_correlation: float = 0.7       # Minimum correlation for pair selection
    lookback_period: int = 252         # Period for correlation and cointegration tests
    formation_period: int = 126        # Period for pair formation
    trading_period: int = 63           # Period for active trading
    
    # Cointegration parameters
    cointegration_threshold: float = 0.05  # P-value threshold for cointegration test
    use_cointegration: bool = True         # Whether to use cointegration test
    
    # Signal generation parameters
    zscore_entry: float = 2.0          # Z-score threshold for entry
    zscore_exit: float = 0.5           # Z-score threshold for exit
    zscore_stop: float = 3.5           # Z-score threshold for stop loss
    
    # Position management
    max_pairs: int = 10                # Maximum number of active pairs
    pair_allocation: float = 0.1       # Allocation per pair (10%)
    hedge_ratio_method: str = "ols"    # "ols", "tls", or "kalman"
    
    # Risk management
    max_holding_days: int = 30         # Maximum days to hold pair position
    correlation_decay_threshold: float = 0.5  # Exit if correlation drops below this
    
    # Rebalancing
    rebalance_frequency: int = 21      # Rebalance pairs every 21 days
    min_trading_volume: float = 1000000 # Minimum daily volume for pair assets


@dataclass
class PairInfo:
    """Information about a trading pair"""
    asset1: str
    asset2: str
    hedge_ratio: float
    correlation: float
    cointegration_pvalue: float
    spread_mean: float
    spread_std: float
    formation_start: datetime
    formation_end: datetime
    last_rebalance: datetime
    is_active: bool = True
    
    def get_pair_name(self) -> str:
        return f"{self.asset1}_{self.asset2}"


class PairsTradingStrategy(BaseStrategy):
    """
    Pairs trading strategy that exploits mean-reverting relationships between correlated assets
    
    The strategy:
    1. Identifies pairs of correlated and cointegrated assets
    2. Monitors the spread between pairs for mean reversion opportunities
    3. Goes long the underperforming asset and short the outperforming asset
    4. Closes positions when spread reverts to mean
    """
    
    def __init__(self, config: PairsTradingConfig):
        super().__init__(config)
        self.config: PairsTradingConfig = config
        
        # Pairs management
        self.pairs: Dict[str, PairInfo] = {}
        self.pair_positions: Dict[str, Dict[str, float]] = {}  # pair_name -> {asset1: pos, asset2: pos}
        self.pair_spreads: Dict[str, List[float]] = {}
        self.pair_zscores: Dict[str, List[float]] = {}
        
        # Market data cache
        self.price_data: Dict[str, List[float]] = {}
        self.volume_data: Dict[str, List[float]] = {}
        self.timestamps_data: List[datetime] = []
        
        logger.info(f"Initialized pairs trading strategy for {len(config.symbols)} assets")
    
    def generate_signals(self, data: pd.DataFrame, timestamp: datetime) -> List[Signal]:
        """Generate pairs trading signals"""
        signals = []
        
        # Update market data cache
        self._update_market_data(data, timestamp)
        
        if len(self.timestamps_data) < self.config.formation_period:
            logger.debug("Insufficient data for pair formation")
            return signals
        
        # Update or form new pairs periodically
        if self._should_rebalance_pairs(timestamp):
            self._update_pairs(timestamp)
        
        # Generate signals for existing pairs
        for pair_name, pair_info in self.pairs.items():
            if not pair_info.is_active:
                continue
            
            try:
                pair_signals = self._generate_pair_signals(pair_info, timestamp)
                signals.extend(pair_signals)
            except Exception as e:
                logger.error(f"Error generating signals for pair {pair_name}: {e}")
        
        return signals
    
    def _update_market_data(self, data: pd.DataFrame, timestamp: datetime):
        """Update market data cache"""
        self.timestamps_data.append(timestamp)
        
        for symbol in self.config.symbols:
            if f"{symbol}_close" in data.columns:
                price = data[f"{symbol}_close"].iloc[-1]
                
                if symbol not in self.price_data:
                    self.price_data[symbol] = []
                    self.volume_data[symbol] = []
                
                self.price_data[symbol].append(price)
                
                # Volume data
                volume = data.get(f"{symbol}_volume", pd.Series([1000000])).iloc[-1]
                self.volume_data[symbol].append(volume)
        
        # Keep only recent data
        max_length = self.config.lookback_period + 50
        if len(self.timestamps_data) > max_length:
            self.timestamps_data = self.timestamps_data[-max_length:]
            for symbol in self.price_data:
                self.price_data[symbol] = self.price_data[symbol][-max_length:]
                self.volume_data[symbol] = self.volume_data[symbol][-max_length:]
    
    def _should_rebalance_pairs(self, timestamp: datetime) -> bool:
        """Check if pairs should be rebalanced"""
        if not self.pairs:
            return True
        
        # Check if enough time has passed since last rebalance
        for pair_info in self.pairs.values():
            days_since_rebalance = (timestamp - pair_info.last_rebalance).days
            if days_since_rebalance >= self.config.rebalance_frequency:
                return True
        
        return False
    
    def _update_pairs(self, timestamp: datetime):
        """Update or form new trading pairs"""
        logger.info("Updating trading pairs...")
        
        # Get recent price data for analysis
        formation_length = min(self.config.formation_period, len(self.timestamps_data))
        
        if formation_length < 30:  # Need minimum data
            return
        
        # Create price matrix
        price_matrix = {}
        valid_symbols = []
        
        for symbol in self.config.symbols:
            if (symbol in self.price_data and 
                len(self.price_data[symbol]) >= formation_length and
                np.mean(self.volume_data[symbol][-formation_length:]) >= self.config.min_trading_volume):
                
                price_matrix[symbol] = np.array(self.price_data[symbol][-formation_length:])
                valid_symbols.append(symbol)
        
        if len(valid_symbols) < 2:
            logger.warning("Insufficient valid symbols for pair formation")
            return
        
        # Find best pairs
        new_pairs = self._find_best_pairs(valid_symbols, price_matrix, timestamp)
        
        # Update pairs dictionary
        self.pairs.clear()
        for pair_info in new_pairs[:self.config.max_pairs]:
            pair_name = pair_info.get_pair_name()
            self.pairs[pair_name] = pair_info
            self.pair_spreads[pair_name] = []
            self.pair_zscores[pair_name] = []
        
        logger.info(f"Updated pairs: {list(self.pairs.keys())}")
    
    def _find_best_pairs(self, symbols: List[str], price_matrix: Dict[str, np.ndarray], 
                        timestamp: datetime) -> List[PairInfo]:
        """Find the best trading pairs based on correlation and cointegration"""
        pair_candidates = []
        
        # Test all possible pairs
        for asset1, asset2 in combinations(symbols, 2):
            prices1 = price_matrix[asset1]
            prices2 = price_matrix[asset2]
            
            # Calculate correlation
            correlation = np.corrcoef(prices1, prices2)[0, 1]
            
            if abs(correlation) < self.config.min_correlation:
                continue
            
            # Test for cointegration
            cointegration_pvalue = 1.0
            if self.config.use_cointegration:
                cointegration_pvalue = self._test_cointegration(prices1, prices2)
            
            if cointegration_pvalue > self.config.cointegration_threshold:
                continue
            
            # Calculate hedge ratio
            hedge_ratio = self._calculate_hedge_ratio(prices1, prices2)
            
            # Calculate spread statistics
            spread = prices1 - hedge_ratio * prices2
            spread_mean = np.mean(spread)
            spread_std = np.std(spread)
            
            if spread_std == 0:
                continue
            
            # Create pair info
            pair_info = PairInfo(
                asset1=asset1,
                asset2=asset2,
                hedge_ratio=hedge_ratio,
                correlation=correlation,
                cointegration_pvalue=cointegration_pvalue,
                spread_mean=spread_mean,
                spread_std=spread_std,
                formation_start=timestamp - timedelta(days=len(prices1)),
                formation_end=timestamp,
                last_rebalance=timestamp
            )
            
            # Score pairs (lower p-value and higher correlation is better)
            score = abs(correlation) / (cointegration_pvalue + 1e-6)
            pair_candidates.append((score, pair_info))
        
        # Sort by score and return best pairs
        pair_candidates.sort(key=lambda x: x[0], reverse=True)
        return [pair_info for score, pair_info in pair_candidates]
    
    def _test_cointegration(self, prices1: np.ndarray, prices2: np.ndarray) -> float:
        """Test for cointegration using Engle-Granger test"""
        try:
            from statsmodels.tsa.stattools import coint
            _, p_value, _ = coint(prices1, prices2)
            return p_value
        except ImportError:
            # Simplified cointegration test using linear regression residuals
            X = prices2.reshape(-1, 1)
            reg = LinearRegression().fit(X, prices1)
            residuals = prices1 - reg.predict(X)
            
            # ADF test approximation using variance ratio
            residuals_diff = np.diff(residuals)
            residuals_lag = residuals[:-1]
            
            if np.var(residuals_lag) > 0:
                correlation = np.corrcoef(residuals_diff, residuals_lag)[0, 1]
                # Convert correlation to approximate p-value
                p_value = max(0.001, 1 - abs(correlation))
                return p_value
            
            return 1.0
        except Exception:
            return 1.0
    
    def _calculate_hedge_ratio(self, prices1: np.ndarray, prices2: np.ndarray) -> float:
        """Calculate hedge ratio between two assets"""
        if self.config.hedge_ratio_method == "ols":
            # Ordinary Least Squares
            X = prices2.reshape(-1, 1)
            reg = LinearRegression().fit(X, prices1)
            return reg.coef_[0]
        
        elif self.config.hedge_ratio_method == "tls":
            # Total Least Squares (more robust to noise in both variables)
            try:
                from sklearn.decomposition import PCA
                data = np.column_stack([prices1, prices2])
                pca = PCA(n_components=1)
                pca.fit(data)
                # Hedge ratio from principal component
                return -pca.components_[0][0] / pca.components_[0][1]
            except:
                # Fallback to OLS
                X = prices2.reshape(-1, 1)
                reg = LinearRegression().fit(X, prices1)
                return reg.coef_[0]
        
        else:  # Default to simple ratio
            return np.mean(prices1 / prices2)
    
    def _generate_pair_signals(self, pair_info: PairInfo, timestamp: datetime) -> List[Signal]:
        """Generate trading signals for a specific pair"""
        signals = []
        
        # Get current prices
        asset1_price = self.price_data[pair_info.asset1][-1]
        asset2_price = self.price_data[pair_info.asset2][-1]
        
        # Calculate current spread and z-score
        current_spread = asset1_price - pair_info.hedge_ratio * asset2_price
        spread_zscore = (current_spread - pair_info.spread_mean) / pair_info.spread_std
        
        # Update spread history
        pair_name = pair_info.get_pair_name()
        self.pair_spreads[pair_name].append(current_spread)
        self.pair_zscores[pair_name].append(spread_zscore)
        
        # Keep limited history
        if len(self.pair_spreads[pair_name]) > 100:
            self.pair_spreads[pair_name] = self.pair_spreads[pair_name][-100:]
            self.pair_zscores[pair_name] = self.pair_zscores[pair_name][-100:]
        
        # Check if pair is currently active
        has_pair_position = pair_name in self.pair_positions
        
        # Entry signals
        if not has_pair_position:
            # Long spread (asset1 relatively cheap)
            if spread_zscore < -self.config.zscore_entry:
                # Buy asset1, sell asset2
                signals.extend([
                    Signal(
                        symbol=pair_info.asset1,
                        signal_type=SignalType.BUY,
                        timestamp=timestamp,
                        price=asset1_price,
                        confidence=min(1.0, abs(spread_zscore) / self.config.zscore_entry),
                        size=self.config.pair_allocation,
                        metadata={
                            "pair_name": pair_name,
                            "spread_zscore": spread_zscore,
                            "hedge_ratio": pair_info.hedge_ratio,
                            "pair_trade": "long_spread"
                        }
                    ),
                    Signal(
                        symbol=pair_info.asset2,
                        signal_type=SignalType.SELL,
                        timestamp=timestamp,
                        price=asset2_price,
                        confidence=min(1.0, abs(spread_zscore) / self.config.zscore_entry),
                        size=self.config.pair_allocation * pair_info.hedge_ratio,
                        metadata={
                            "pair_name": pair_name,
                            "spread_zscore": spread_zscore,
                            "hedge_ratio": pair_info.hedge_ratio,
                            "pair_trade": "short_spread"
                        }
                    )
                ])
                
                # Mark pair as having position
                self.pair_positions[pair_name] = {
                    pair_info.asset1: 1.0,  # Long
                    pair_info.asset2: -pair_info.hedge_ratio  # Short
                }
            
            # Short spread (asset1 relatively expensive)
            elif spread_zscore > self.config.zscore_entry:
                # Sell asset1, buy asset2
                signals.extend([
                    Signal(
                        symbol=pair_info.asset1,
                        signal_type=SignalType.SELL,
                        timestamp=timestamp,
                        price=asset1_price,
                        confidence=min(1.0, spread_zscore / self.config.zscore_entry),
                        size=self.config.pair_allocation,
                        metadata={
                            "pair_name": pair_name,
                            "spread_zscore": spread_zscore,
                            "hedge_ratio": pair_info.hedge_ratio,
                            "pair_trade": "short_spread"
                        }
                    ),
                    Signal(
                        symbol=pair_info.asset2,
                        signal_type=SignalType.BUY,
                        timestamp=timestamp,
                        price=asset2_price,
                        confidence=min(1.0, spread_zscore / self.config.zscore_entry),
                        size=self.config.pair_allocation * pair_info.hedge_ratio,
                        metadata={
                            "pair_name": pair_name,
                            "spread_zscore": spread_zscore,
                            "hedge_ratio": pair_info.hedge_ratio,
                            "pair_trade": "long_spread"
                        }
                    )
                ])
                
                # Mark pair as having position
                self.pair_positions[pair_name] = {
                    pair_info.asset1: -1.0,  # Short
                    pair_info.asset2: pair_info.hedge_ratio  # Long
                }
        
        # Exit signals
        else:
            pair_position = self.pair_positions[pair_name]
            asset1_position = pair_position[pair_info.asset1]
            
            # Check exit conditions
            should_exit = False
            exit_reason = ""
            
            # Mean reversion exit
            if abs(spread_zscore) < self.config.zscore_exit:
                should_exit = True
                exit_reason = "mean_reversion"
            
            # Stop loss exit
            elif abs(spread_zscore) > self.config.zscore_stop:
                should_exit = True
                exit_reason = "stop_loss"
            
            # Time-based exit
            elif len(self.pair_zscores[pair_name]) > self.config.max_holding_days:
                should_exit = True
                exit_reason = "time_limit"
            
            # Correlation decay exit
            elif self._check_correlation_decay(pair_info):
                should_exit = True
                exit_reason = "correlation_decay"
            
            if should_exit:
                # Close existing positions
                if asset1_position > 0:  # Long asset1, short asset2
                    signals.extend([
                        Signal(
                            symbol=pair_info.asset1,
                            signal_type=SignalType.SELL,
                            timestamp=timestamp,
                            price=asset1_price,
                            confidence=0.9,
                            size=self.config.pair_allocation,
                            metadata={
                                "pair_name": pair_name,
                                "exit_reason": exit_reason,
                                "spread_zscore": spread_zscore
                            }
                        ),
                        Signal(
                            symbol=pair_info.asset2,
                            signal_type=SignalType.CLOSE_SHORT,
                            timestamp=timestamp,
                            price=asset2_price,
                            confidence=0.9,
                            size=self.config.pair_allocation * pair_info.hedge_ratio,
                            metadata={
                                "pair_name": pair_name,
                                "exit_reason": exit_reason,
                                "spread_zscore": spread_zscore
                            }
                        )
                    ])
                else:  # Short asset1, long asset2
                    signals.extend([
                        Signal(
                            symbol=pair_info.asset1,
                            signal_type=SignalType.CLOSE_SHORT,
                            timestamp=timestamp,
                            price=asset1_price,
                            confidence=0.9,
                            size=self.config.pair_allocation,
                            metadata={
                                "pair_name": pair_name,
                                "exit_reason": exit_reason,
                                "spread_zscore": spread_zscore
                            }
                        ),
                        Signal(
                            symbol=pair_info.asset2,
                            signal_type=SignalType.SELL,
                            timestamp=timestamp,
                            price=asset2_price,
                            confidence=0.9,
                            size=self.config.pair_allocation * pair_info.hedge_ratio,
                            metadata={
                                "pair_name": pair_name,
                                "exit_reason": exit_reason,
                                "spread_zscore": spread_zscore
                            }
                        )
                    ])
                
                # Remove pair position
                del self.pair_positions[pair_name]
        
        return signals
    
    def _check_correlation_decay(self, pair_info: PairInfo) -> bool:
        """Check if correlation has decayed significantly"""
        if len(self.price_data[pair_info.asset1]) < 20:
            return False
        
        # Calculate recent correlation
        recent_prices1 = np.array(self.price_data[pair_info.asset1][-20:])
        recent_prices2 = np.array(self.price_data[pair_info.asset2][-20:])
        
        recent_correlation = np.corrcoef(recent_prices1, recent_prices2)[0, 1]
        
        return abs(recent_correlation) < self.config.correlation_decay_threshold
    
    def calculate_position_size(self, signal: Signal, current_price: float, 
                              available_cash: float) -> float:
        """Calculate position size for pairs trading"""
        # For pairs trading, position size is predetermined by pair allocation
        target_value = available_cash * signal.size
        
        # Account for transaction costs
        transaction_cost_rate = self.config.commission_rate + self.config.slippage_rate
        effective_cash = target_value / (1 + transaction_cost_rate)
        
        shares = effective_cash / current_price
        
        # Apply minimum trade size constraint
        min_shares = self.config.min_trade_size / current_price
        if shares < min_shares:
            return 0.0
        
        return shares
    
    def get_pairs_summary(self) -> Dict[str, Dict[str, Any]]:
        """Get summary of all trading pairs"""
        summary = {}
        
        for pair_name, pair_info in self.pairs.items():
            has_position = pair_name in self.pair_positions
            current_zscore = self.pair_zscores[pair_name][-1] if self.pair_zscores[pair_name] else 0
            
            summary[pair_name] = {
                "asset1": pair_info.asset1,
                "asset2": pair_info.asset2,
                "hedge_ratio": pair_info.hedge_ratio,
                "correlation": pair_info.correlation,
                "cointegration_pvalue": pair_info.cointegration_pvalue,
                "current_zscore": current_zscore,
                "has_position": has_position,
                "is_active": pair_info.is_active,
                "days_since_formation": (datetime.now() - pair_info.formation_end).days
            }
        
        return summary
    
    def get_strategy_metrics(self) -> Dict[str, Any]:
        """Get pairs trading specific metrics"""
        base_metrics = self.get_performance_metrics()
        
        pairs_metrics = {
            "active_pairs": len([p for p in self.pairs.values() if p.is_active]),
            "pairs_with_positions": len(self.pair_positions),
            "total_pairs_formed": len(self.pairs)
        }
        
        # Average correlation and cointegration metrics
        if self.pairs:
            correlations = [p.correlation for p in self.pairs.values()]
            p_values = [p.cointegration_pvalue for p in self.pairs.values()]
            
            pairs_metrics.update({
                "avg_correlation": np.mean([abs(c) for c in correlations]),
                "avg_cointegration_pvalue": np.mean(p_values),
                "min_correlation": min([abs(c) for c in correlations]),
                "max_cointegration_pvalue": max(p_values)
            })
        
        # Current z-score statistics
        if self.pair_zscores:
            all_zscores = [zscores[-1] for zscores in self.pair_zscores.values() if zscores]
            if all_zscores:
                pairs_metrics.update({
                    "avg_zscore_magnitude": np.mean([abs(z) for z in all_zscores]),
                    "max_zscore_magnitude": max([abs(z) for z in all_zscores])
                })
        
        return {**base_metrics, **pairs_metrics}


# Export classes
__all__ = ['PairsTradingStrategy', 'PairsTradingConfig', 'PairInfo']