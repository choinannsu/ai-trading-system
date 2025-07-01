"""
Position Sizing System
Advanced position sizing algorithms including Kelly Criterion, volatility-based sizing,
correlation-aware optimization, and dynamic leverage adjustment
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import warnings
from scipy.optimize import minimize
from scipy import stats
import cvxpy as cp

from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class PositionSizingConfig:
    """Configuration for position sizing system"""
    # Kelly Criterion parameters
    kelly_window: int = 252  # Window for Kelly calculation
    kelly_fraction: float = 0.25  # Maximum Kelly fraction to use
    kelly_lookback: int = 100  # Number of trades for Kelly calculation
    
    # Volatility-based sizing
    target_volatility: float = 0.15  # Target portfolio volatility (15%)
    volatility_window: int = 30  # Window for volatility calculation
    vol_adjustment_factor: float = 1.0  # Volatility adjustment factor
    
    # Correlation parameters
    correlation_window: int = 60  # Window for correlation calculation
    max_correlation_exposure: float = 0.5  # Max exposure to correlated assets
    correlation_threshold: float = 0.7  # Correlation threshold
    
    # Dynamic leverage
    base_leverage: float = 1.0  # Base leverage
    max_leverage: float = 3.0  # Maximum leverage
    leverage_adjustment_factor: float = 0.1  # Leverage adjustment speed
    
    # Risk limits
    max_position_size: float = 0.2  # Maximum position size (20%)
    min_position_size: float = 0.01  # Minimum position size (1%)
    max_portfolio_exposure: float = 1.0  # Maximum total exposure


class KellyCriterion:
    """
    Kelly Criterion implementation for optimal position sizing
    
    Features:
    - Classical Kelly formula
    - Fractional Kelly for risk management
    - Multi-asset Kelly optimization
    - Kelly with transaction costs
    """
    
    def __init__(self, config: PositionSizingConfig):
        self.config = config
        self.trade_history: Dict[str, List[Dict[str, Any]]] = {}
        
    def calculate_kelly_fraction(self, returns: pd.Series, 
                                win_rate: Optional[float] = None,
                                avg_win: Optional[float] = None,
                                avg_loss: Optional[float] = None) -> float:
        """Calculate Kelly fraction from returns series"""
        if len(returns) < 10:
            return 0.0
        
        # Method 1: Direct from returns
        if win_rate is None or avg_win is None or avg_loss is None:
            mean_return = returns.mean()
            return_std = returns.std()
            
            if return_std == 0:
                return 0.0
            
            # Simplified Kelly: f = μ/σ²
            kelly_fraction = mean_return / (return_std ** 2)
        else:
            # Method 2: Win/Loss ratio method
            if avg_loss == 0:
                return 0.0
            
            win_loss_ratio = abs(avg_win / avg_loss)
            kelly_fraction = win_rate - (1 - win_rate) / win_loss_ratio
        
        # Apply fractional Kelly for safety
        kelly_fraction = kelly_fraction * self.config.kelly_fraction
        
        # Bound the result
        return max(-0.25, min(0.25, kelly_fraction))
    
    def calculate_multi_asset_kelly(self, returns_matrix: pd.DataFrame,
                                   correlation_matrix: Optional[pd.DataFrame] = None) -> Dict[str, float]:
        """Calculate Kelly fractions for multiple assets considering correlations"""
        if returns_matrix.empty:
            return {}
        
        assets = returns_matrix.columns
        n_assets = len(assets)
        
        if n_assets == 1:
            asset = assets[0]
            kelly_frac = self.calculate_kelly_fraction(returns_matrix[asset])
            return {asset: kelly_frac}
        
        # Calculate mean returns and covariance matrix
        mean_returns = returns_matrix.mean().values
        cov_matrix = returns_matrix.cov().values
        
        # Regularize covariance matrix to ensure positive definiteness
        cov_matrix = self._regularize_covariance(cov_matrix)
        
        try:
            # Solve: f = Σ⁻¹μ (Kelly formula for multiple assets)
            inv_cov = np.linalg.inv(cov_matrix)
            kelly_weights = inv_cov @ mean_returns
            
            # Scale down if total allocation is too high
            total_allocation = np.sum(np.abs(kelly_weights))
            if total_allocation > self.config.kelly_fraction:
                kelly_weights = kelly_weights * (self.config.kelly_fraction / total_allocation)
            
            # Create result dictionary
            result = {}
            for i, asset in enumerate(assets):
                result[asset] = max(-0.25, min(0.25, kelly_weights[i]))
            
            return result
            
        except np.linalg.LinAlgError:
            # Fallback to individual Kelly calculations
            logger.warning("Covariance matrix inversion failed, using individual Kelly calculations")
            result = {}
            for asset in assets:
                result[asset] = self.calculate_kelly_fraction(returns_matrix[asset])
            return result
    
    def _regularize_covariance(self, cov_matrix: np.ndarray, 
                              regularization: float = 1e-6) -> np.ndarray:
        """Regularize covariance matrix to ensure numerical stability"""
        # Add small value to diagonal
        cov_regularized = cov_matrix + regularization * np.eye(cov_matrix.shape[0])
        
        # Ensure positive definiteness
        eigenvals, eigenvecs = np.linalg.eigh(cov_regularized)
        eigenvals = np.maximum(eigenvals, regularization)
        cov_regularized = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
        
        return cov_regularized
    
    def update_trade_history(self, symbol: str, entry_price: float, exit_price: float,
                           trade_return: float, trade_size: float):
        """Update trade history for Kelly calculation"""
        if symbol not in self.trade_history:
            self.trade_history[symbol] = []
        
        trade_record = {
            'entry_price': entry_price,
            'exit_price': exit_price,
            'return': trade_return,
            'size': trade_size,
            'timestamp': datetime.now()
        }
        
        self.trade_history[symbol].append(trade_record)
        
        # Keep only recent trades
        if len(self.trade_history[symbol]) > self.config.kelly_lookback:
            self.trade_history[symbol] = self.trade_history[symbol][-self.config.kelly_lookback:]
    
    def get_kelly_from_trades(self, symbol: str) -> float:
        """Calculate Kelly fraction from trade history"""
        if symbol not in self.trade_history or len(self.trade_history[symbol]) < 10:
            return 0.0
        
        trades = self.trade_history[symbol]
        returns = [trade['return'] for trade in trades]
        
        return self.calculate_kelly_fraction(pd.Series(returns))


class VolatilityBasedSizing:
    """
    Volatility-based position sizing system
    
    Features:
    - Target volatility position sizing
    - EWMA volatility estimation
    - Volatility regime detection
    - Dynamic volatility adjustment
    """
    
    def __init__(self, config: PositionSizingConfig):
        self.config = config
        self.volatility_cache: Dict[str, float] = {}
        
    def calculate_volatility_adjusted_size(self, symbol: str, price_data: pd.Series,
                                         base_size: float) -> float:
        """Calculate position size adjusted for volatility"""
        if len(price_data) < self.config.volatility_window:
            return base_size
        
        # Calculate returns
        returns = price_data.pct_change().dropna()
        
        if len(returns) < self.config.volatility_window:
            return base_size
        
        # Calculate volatility using EWMA
        volatility = self._calculate_ewma_volatility(returns)
        
        # Calculate target volatility contribution
        target_vol_contribution = self.config.target_volatility
        
        # Adjust position size
        vol_adjustment = target_vol_contribution / (volatility * np.sqrt(252))
        vol_adjustment = vol_adjustment * self.config.vol_adjustment_factor
        
        adjusted_size = base_size * vol_adjustment
        
        # Apply bounds
        adjusted_size = max(self.config.min_position_size, 
                          min(self.config.max_position_size, adjusted_size))
        
        # Cache volatility
        self.volatility_cache[symbol] = volatility
        
        return adjusted_size
    
    def _calculate_ewma_volatility(self, returns: pd.Series, alpha: float = 0.06) -> float:
        """Calculate EWMA volatility"""
        if len(returns) < 2:
            return 0.01  # Default volatility
        
        # EWMA calculation
        ewma_var = returns.iloc[0] ** 2
        
        for ret in returns.iloc[1:]:
            ewma_var = alpha * (ret ** 2) + (1 - alpha) * ewma_var
        
        return np.sqrt(ewma_var)
    
    def detect_volatility_regime(self, returns: pd.Series) -> str:
        """Detect volatility regime (low, medium, high)"""
        if len(returns) < 60:
            return "medium"
        
        # Calculate rolling volatility
        rolling_vol = returns.rolling(30).std() * np.sqrt(252)
        current_vol = rolling_vol.iloc[-1]
        
        # Define thresholds based on historical distribution
        vol_percentiles = rolling_vol.quantile([0.33, 0.67])
        
        if current_vol < vol_percentiles.iloc[0]:
            return "low"
        elif current_vol > vol_percentiles.iloc[1]:
            return "high"
        else:
            return "medium"
    
    def get_regime_adjustment(self, regime: str) -> float:
        """Get position size adjustment based on volatility regime"""
        adjustments = {
            "low": 1.2,    # Increase size in low vol
            "medium": 1.0,  # Normal size
            "high": 0.7    # Reduce size in high vol
        }
        return adjustments.get(regime, 1.0)


class CorrelationAwareOptimizer:
    """
    Correlation-aware position sizing optimization
    
    Features:
    - Dynamic correlation monitoring
    - Correlation-adjusted position limits
    - Sector/style diversification
    - Risk budgeting
    """
    
    def __init__(self, config: PositionSizingConfig):
        self.config = config
        self.correlation_matrix: Optional[pd.DataFrame] = None
        self.last_correlation_update: Optional[datetime] = None
        
    def optimize_positions_with_correlation(self, proposed_positions: Dict[str, float],
                                          returns_data: pd.DataFrame) -> Dict[str, float]:
        """Optimize positions considering correlations"""
        if not proposed_positions or returns_data.empty:
            return proposed_positions
        
        # Update correlation matrix if needed
        self._update_correlation_matrix(returns_data)
        
        if self.correlation_matrix is None:
            return proposed_positions
        
        # Get symbols that have both position and correlation data
        symbols = list(set(proposed_positions.keys()) & set(self.correlation_matrix.index))
        
        if len(symbols) < 2:
            return proposed_positions
        
        # Create position vector
        position_vector = np.array([proposed_positions[symbol] for symbol in symbols])
        
        # Get correlation submatrix
        corr_matrix = self.correlation_matrix.loc[symbols, symbols].values
        
        # Optimize positions
        optimized_positions = self._solve_correlation_optimization(position_vector, corr_matrix, symbols)
        
        # Update proposed positions
        result = proposed_positions.copy()
        for i, symbol in enumerate(symbols):
            result[symbol] = optimized_positions[i]
        
        return result
    
    def _update_correlation_matrix(self, returns_data: pd.DataFrame):
        """Update correlation matrix if needed"""
        now = datetime.now()
        
        if (self.last_correlation_update is None or 
            (now - self.last_correlation_update).days >= 1):
            
            if len(returns_data) >= self.config.correlation_window:
                recent_data = returns_data.tail(self.config.correlation_window)
                self.correlation_matrix = recent_data.corr()
                self.last_correlation_update = now
    
    def _solve_correlation_optimization(self, positions: np.ndarray, 
                                      correlation_matrix: np.ndarray,
                                      symbols: List[str]) -> np.ndarray:
        """Solve correlation-constrained optimization"""
        n = len(positions)
        
        # Decision variables
        x = cp.Variable(n)
        
        # Objective: minimize deviation from target positions
        objective = cp.Minimize(cp.sum_squares(x - positions))
        
        # Constraints
        constraints = []
        
        # Position bounds
        constraints.append(x >= -self.config.max_position_size)
        constraints.append(x <= self.config.max_position_size)
        
        # Total exposure constraint
        constraints.append(cp.sum(cp.abs(x)) <= self.config.max_portfolio_exposure)
        
        # Correlation constraints
        for i in range(n):
            for j in range(i+1, n):
                correlation = correlation_matrix[i, j]
                if abs(correlation) > self.config.correlation_threshold:
                    # Limit combined exposure of highly correlated assets
                    constraints.append(cp.abs(x[i]) + cp.abs(x[j]) <= 
                                     self.config.max_correlation_exposure)
        
        # Solve optimization
        problem = cp.Problem(objective, constraints)
        
        try:
            problem.solve()
            
            if x.value is not None:
                return x.value
            else:
                logger.warning("Correlation optimization failed, returning original positions")
                return positions
                
        except Exception as e:
            logger.error(f"Correlation optimization error: {e}")
            return positions
    
    def calculate_diversification_ratio(self, positions: Dict[str, float],
                                      returns_data: pd.DataFrame) -> float:
        """Calculate portfolio diversification ratio"""
        if not positions or returns_data.empty:
            return 1.0
        
        symbols = list(set(positions.keys()) & set(returns_data.columns))
        
        if len(symbols) < 2:
            return 1.0
        
        # Get position weights
        weights = np.array([abs(positions[symbol]) for symbol in symbols])
        total_weight = np.sum(weights)
        
        if total_weight == 0:
            return 1.0
        
        weights = weights / total_weight
        
        # Calculate volatilities
        volatilities = returns_data[symbols].std().values * np.sqrt(252)
        
        # Calculate correlation matrix
        correlation_matrix = returns_data[symbols].corr().values
        
        # Portfolio volatility
        portfolio_vol = np.sqrt(weights.T @ correlation_matrix @ weights)
        
        # Weighted average volatility
        weighted_avg_vol = np.sum(weights * volatilities)
        
        # Diversification ratio
        diversification_ratio = weighted_avg_vol / portfolio_vol
        
        return diversification_ratio


class DynamicLeverageManager:
    """
    Dynamic leverage adjustment system
    
    Features:
    - Market regime-based leverage
    - Volatility-adjusted leverage
    - Risk-adjusted leverage scaling
    - Maximum drawdown-based adjustments
    """
    
    def __init__(self, config: PositionSizingConfig):
        self.config = config
        self.current_leverage = config.base_leverage
        self.equity_curve: List[float] = []
        self.last_peak = 0.0
        
    def calculate_dynamic_leverage(self, portfolio_value: float, 
                                 market_conditions: Dict[str, Any]) -> float:
        """Calculate dynamic leverage based on market conditions"""
        # Update equity curve
        self.equity_curve.append(portfolio_value)
        if len(self.equity_curve) > 252:  # Keep 1 year of data
            self.equity_curve = self.equity_curve[-252:]
        
        # Calculate leverage adjustments
        vol_adjustment = self._get_volatility_adjustment(market_conditions)
        drawdown_adjustment = self._get_drawdown_adjustment()
        momentum_adjustment = self._get_momentum_adjustment()
        
        # Combine adjustments
        total_adjustment = vol_adjustment * drawdown_adjustment * momentum_adjustment
        
        # Update leverage
        target_leverage = self.config.base_leverage * total_adjustment
        
        # Smooth leverage changes
        leverage_change = (target_leverage - self.current_leverage) * self.config.leverage_adjustment_factor
        self.current_leverage += leverage_change
        
        # Apply bounds
        self.current_leverage = max(0.1, min(self.config.max_leverage, self.current_leverage))
        
        return self.current_leverage
    
    def _get_volatility_adjustment(self, market_conditions: Dict[str, Any]) -> float:
        """Get leverage adjustment based on volatility"""
        current_vol = market_conditions.get('portfolio_volatility', 0.15)
        target_vol = self.config.target_volatility
        
        # Inverse relationship: higher vol = lower leverage
        vol_ratio = target_vol / max(current_vol, 0.01)
        
        # Bound the adjustment
        return max(0.5, min(2.0, vol_ratio))
    
    def _get_drawdown_adjustment(self) -> float:
        """Get leverage adjustment based on current drawdown"""
        if len(self.equity_curve) < 2:
            return 1.0
        
        current_value = self.equity_curve[-1]
        peak_value = max(self.equity_curve)
        
        if peak_value <= 0:
            return 1.0
        
        drawdown = (peak_value - current_value) / peak_value
        
        # Reduce leverage based on drawdown
        if drawdown < 0.05:  # Less than 5% drawdown
            return 1.1
        elif drawdown < 0.10:  # 5-10% drawdown
            return 1.0
        elif drawdown < 0.15:  # 10-15% drawdown
            return 0.8
        else:  # More than 15% drawdown
            return 0.6
    
    def _get_momentum_adjustment(self) -> float:
        """Get leverage adjustment based on recent performance momentum"""
        if len(self.equity_curve) < 20:
            return 1.0
        
        # Calculate recent returns
        recent_values = self.equity_curve[-20:]
        returns = [(recent_values[i] - recent_values[i-1]) / recent_values[i-1] 
                  for i in range(1, len(recent_values))]
        
        if not returns:
            return 1.0
        
        # Calculate momentum
        avg_return = np.mean(returns)
        
        # Adjust leverage based on momentum
        if avg_return > 0.001:  # Positive momentum
            return 1.1
        elif avg_return < -0.001:  # Negative momentum
            return 0.9
        else:
            return 1.0


class PositionSizer:
    """
    Main position sizing system combining all strategies
    
    Features:
    - Integrated Kelly Criterion
    - Volatility-based adjustments
    - Correlation-aware optimization
    - Dynamic leverage management
    - Risk limit enforcement
    """
    
    def __init__(self, config: PositionSizingConfig = None):
        self.config = config or PositionSizingConfig()
        
        # Initialize subsystems
        self.kelly_criterion = KellyCriterion(self.config)
        self.volatility_sizer = VolatilityBasedSizing(self.config)
        self.correlation_optimizer = CorrelationAwareOptimizer(self.config)
        self.leverage_manager = DynamicLeverageManager(self.config)
        
        logger.info("Initialized position sizing system")
    
    def calculate_position_sizes(self, signals: Dict[str, float],
                               market_data: pd.DataFrame,
                               portfolio_value: float,
                               current_positions: Dict[str, float] = None) -> Dict[str, float]:
        """Calculate optimal position sizes for given signals"""
        if not signals or market_data.empty:
            return {}
        
        current_positions = current_positions or {}
        
        # Step 1: Calculate base sizes using Kelly Criterion
        base_sizes = self._calculate_kelly_sizes(signals, market_data)
        
        # Step 2: Apply volatility adjustments
        vol_adjusted_sizes = self._apply_volatility_adjustments(base_sizes, market_data)
        
        # Step 3: Optimize for correlations
        correlation_optimized = self.correlation_optimizer.optimize_positions_with_correlation(
            vol_adjusted_sizes, market_data.pct_change().dropna()
        )
        
        # Step 4: Apply dynamic leverage
        market_conditions = self._get_market_conditions(market_data)
        leverage = self.leverage_manager.calculate_dynamic_leverage(portfolio_value, market_conditions)
        
        leveraged_sizes = {symbol: size * leverage 
                          for symbol, size in correlation_optimized.items()}
        
        # Step 5: Apply final risk limits
        final_sizes = self._apply_risk_limits(leveraged_sizes, current_positions)
        
        return final_sizes
    
    def _calculate_kelly_sizes(self, signals: Dict[str, float],
                             market_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate base position sizes using Kelly Criterion"""
        base_sizes = {}
        
        # Calculate returns for each symbol
        returns_data = {}
        for symbol in signals.keys():
            symbol_cols = [col for col in market_data.columns if symbol in col and 'close' in col]
            if symbol_cols:
                prices = market_data[symbol_cols[0]].dropna()
                if len(prices) > self.config.kelly_window:
                    returns = prices.pct_change().dropna().tail(self.config.kelly_window)
                    returns_data[symbol] = returns
        
        if returns_data:
            # Convert to DataFrame for multi-asset Kelly
            returns_df = pd.DataFrame(returns_data)
            kelly_fractions = self.kelly_criterion.calculate_multi_asset_kelly(returns_df)
            
            # Apply signal direction and strength
            for symbol, signal_strength in signals.items():
                if symbol in kelly_fractions:
                    kelly_fraction = kelly_fractions[symbol]
                    # Scale by signal strength (assuming signals are between -1 and 1)
                    base_sizes[symbol] = kelly_fraction * signal_strength
                else:
                    # Fallback for symbols without enough data
                    base_sizes[symbol] = self.config.min_position_size * signal_strength
        else:
            # Fallback: equal weight based on signal strength
            for symbol, signal_strength in signals.items():
                base_sizes[symbol] = self.config.min_position_size * signal_strength
        
        return base_sizes
    
    def _apply_volatility_adjustments(self, base_sizes: Dict[str, float],
                                    market_data: pd.DataFrame) -> Dict[str, float]:
        """Apply volatility-based adjustments to position sizes"""
        adjusted_sizes = {}
        
        for symbol, base_size in base_sizes.items():
            symbol_cols = [col for col in market_data.columns if symbol in col and 'close' in col]
            if symbol_cols:
                prices = market_data[symbol_cols[0]].dropna()
                adjusted_size = self.volatility_sizer.calculate_volatility_adjusted_size(
                    symbol, prices, base_size
                )
                adjusted_sizes[symbol] = adjusted_size
            else:
                adjusted_sizes[symbol] = base_size
        
        return adjusted_sizes
    
    def _get_market_conditions(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Extract current market conditions"""
        if market_data.empty:
            return {'portfolio_volatility': 0.15}
        
        # Calculate portfolio volatility (simplified)
        close_cols = [col for col in market_data.columns if 'close' in col]
        if close_cols:
            prices = market_data[close_cols].dropna()
            returns = prices.pct_change().dropna()
            
            if not returns.empty:
                portfolio_returns = returns.mean(axis=1)  # Equal weight average
                portfolio_vol = portfolio_returns.std() * np.sqrt(252)
            else:
                portfolio_vol = 0.15
        else:
            portfolio_vol = 0.15
        
        return {
            'portfolio_volatility': portfolio_vol,
            'market_regime': 'normal'  # Could be enhanced with regime detection
        }
    
    def _apply_risk_limits(self, sizes: Dict[str, float],
                          current_positions: Dict[str, float]) -> Dict[str, float]:
        """Apply final risk limits and constraints"""
        final_sizes = {}
        
        # Calculate total exposure
        total_exposure = sum(abs(size) for size in sizes.values())
        
        # Scale down if total exposure exceeds limit
        if total_exposure > self.config.max_portfolio_exposure:
            scale_factor = self.config.max_portfolio_exposure / total_exposure
            sizes = {symbol: size * scale_factor for symbol, size in sizes.items()}
        
        # Apply individual position limits
        for symbol, size in sizes.items():
            # Bound individual positions
            bounded_size = max(-self.config.max_position_size,
                             min(self.config.max_position_size, size))
            
            # Apply minimum size threshold
            if abs(bounded_size) < self.config.min_position_size:
                bounded_size = 0.0
            
            final_sizes[symbol] = bounded_size
        
        return final_sizes
    
    def update_trade_history(self, symbol: str, entry_price: float, exit_price: float,
                           trade_return: float, trade_size: float):
        """Update trade history for Kelly calculations"""
        self.kelly_criterion.update_trade_history(symbol, entry_price, exit_price,
                                                 trade_return, trade_size)
    
    def get_sizing_metrics(self) -> Dict[str, Any]:
        """Get current sizing system metrics"""
        return {
            'current_leverage': self.leverage_manager.current_leverage,
            'volatility_cache': self.volatility_sizer.volatility_cache.copy(),
            'kelly_trade_counts': {symbol: len(trades) 
                                  for symbol, trades in self.kelly_criterion.trade_history.items()},
            'correlation_last_update': self.correlation_optimizer.last_correlation_update
        }


# Factory function
def create_position_sizer(config: Dict[str, Any] = None) -> PositionSizer:
    """Create position sizing system with configuration"""
    sizing_config = PositionSizingConfig(**(config or {}))
    return PositionSizer(sizing_config)


# Export classes and functions
__all__ = [
    'PositionSizingConfig',
    'KellyCriterion',
    'VolatilityBasedSizing',
    'CorrelationAwareOptimizer',
    'DynamicLeverageManager',
    'PositionSizer',
    'create_position_sizer'
]