"""
Advanced Reward Functions for Trading RL
Risk-adjusted returns, drawdown penalties, and portfolio optimization rewards
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
from enum import Enum
import warnings

from utils.logger import get_logger

logger = get_logger(__name__)


class RewardType(Enum):
    """Types of reward functions"""
    SIMPLE_RETURN = "simple_return"
    SHARPE_RATIO = "sharpe_ratio"
    SORTINO_RATIO = "sortino_ratio"
    CALMAR_RATIO = "calmar_ratio"
    MAX_DRAWDOWN = "max_drawdown"
    TRADING_FREQUENCY = "trading_frequency"
    PORTFOLIO_DIVERSITY = "portfolio_diversity"
    RISK_PARITY = "risk_parity"
    VALUE_AT_RISK = "value_at_risk"


@dataclass
class RewardConfig:
    """Configuration for reward functions"""
    # Base reward settings
    reward_scaling: float = 1.0
    risk_free_rate: float = 0.02  # Annual risk-free rate
    
    # Sharpe ratio settings
    sharpe_window: int = 252  # Trading days in a year
    min_sharpe_periods: int = 30
    
    # Drawdown settings
    max_drawdown_penalty: float = 10.0
    drawdown_threshold: float = 0.05  # 5%
    
    # Trading frequency settings
    optimal_trades_per_day: float = 2.0
    trading_penalty_coef: float = 0.01
    
    # Portfolio diversity settings
    diversity_target: float = 0.8  # Target diversification ratio
    concentration_penalty: float = 5.0
    min_positions: int = 3
    
    # Risk management
    volatility_penalty: float = 1.0
    var_confidence: float = 0.05  # 5% VaR
    var_window: int = 252
    
    # Transaction cost penalty
    cost_penalty_multiplier: float = 10.0


class RewardFunction(ABC):
    """Abstract base class for reward functions"""
    
    def __init__(self, config: RewardConfig = None):
        self.config = config or RewardConfig()
        self.returns_history = []
        self.portfolio_values = []
        self.trades_history = []
        self.positions_history = []
        
    @abstractmethod
    def calculate(self, portfolio_return: float, portfolio_value: float,
                 positions: Dict[str, float], trades: List[Any],
                 market_data: Dict[str, Any]) -> float:
        """Calculate reward for current step"""
        pass
    
    def reset(self):
        """Reset reward function for new episode"""
        self.returns_history = []
        self.portfolio_values = []
        self.trades_history = []
        self.positions_history = []
    
    def update_history(self, portfolio_return: float, portfolio_value: float,
                      positions: Dict[str, float], trades: List[Any]):
        """Update historical data"""
        self.returns_history.append(portfolio_return)
        self.portfolio_values.append(portfolio_value)
        self.trades_history.extend(trades)
        self.positions_history.append(positions.copy())


class SimpleReturnReward(RewardFunction):
    """Simple portfolio return reward"""
    
    def calculate(self, portfolio_return: float, portfolio_value: float,
                 positions: Dict[str, float], trades: List[Any],
                 market_data: Dict[str, Any]) -> float:
        """Calculate simple return reward"""
        self.update_history(portfolio_return, portfolio_value, positions, trades)
        return portfolio_return * self.config.reward_scaling


class SharpeReward(RewardFunction):
    """Sharpe ratio based reward function"""
    
    def calculate(self, portfolio_return: float, portfolio_value: float,
                 positions: Dict[str, float], trades: List[Any],
                 market_data: Dict[str, Any]) -> float:
        """Calculate Sharpe ratio reward"""
        self.update_history(portfolio_return, portfolio_value, positions, trades)
        
        if len(self.returns_history) < self.config.min_sharpe_periods:
            # Use simple return for early periods
            return portfolio_return * self.config.reward_scaling
        
        # Calculate rolling Sharpe ratio
        recent_returns = np.array(self.returns_history[-self.config.sharpe_window:])
        
        if len(recent_returns) == 0:
            return 0.0
        
        excess_returns = recent_returns - (self.config.risk_free_rate / 252)  # Daily risk-free rate
        
        if np.std(excess_returns) == 0:
            return 0.0
        
        sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns)
        
        # Scale Sharpe ratio to reasonable reward range
        reward = sharpe_ratio * self.config.reward_scaling * 0.1
        
        return reward


class SortinoReward(RewardFunction):
    """Sortino ratio based reward (focuses on downside risk)"""
    
    def calculate(self, portfolio_return: float, portfolio_value: float,
                 positions: Dict[str, float], trades: List[Any],
                 market_data: Dict[str, Any]) -> float:
        """Calculate Sortino ratio reward"""
        self.update_history(portfolio_return, portfolio_value, positions, trades)
        
        if len(self.returns_history) < self.config.min_sharpe_periods:
            return portfolio_return * self.config.reward_scaling
        
        recent_returns = np.array(self.returns_history[-self.config.sharpe_window:])
        excess_returns = recent_returns - (self.config.risk_free_rate / 252)
        
        # Calculate downside deviation
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0:
            downside_std = 1e-6  # Avoid division by zero
        else:
            downside_std = np.std(downside_returns)
        
        sortino_ratio = np.mean(excess_returns) / downside_std
        
        return sortino_ratio * self.config.reward_scaling * 0.1


class MaxDrawdownPenalty(RewardFunction):
    """Maximum drawdown penalty reward function"""
    
    def __init__(self, config: RewardConfig = None):
        super().__init__(config)
        self.peak_value = 0
        self.max_drawdown = 0
    
    def calculate(self, portfolio_return: float, portfolio_value: float,
                 positions: Dict[str, float], trades: List[Any],
                 market_data: Dict[str, Any]) -> float:
        """Calculate drawdown penalty reward"""
        self.update_history(portfolio_return, portfolio_value, positions, trades)
        
        # Update peak value
        self.peak_value = max(self.peak_value, portfolio_value)
        
        # Calculate current drawdown
        if self.peak_value > 0:
            current_drawdown = (self.peak_value - portfolio_value) / self.peak_value
        else:
            current_drawdown = 0
        
        # Update maximum drawdown
        self.max_drawdown = max(self.max_drawdown, current_drawdown)
        
        # Base reward from return
        base_reward = portfolio_return * self.config.reward_scaling
        
        # Drawdown penalty
        if current_drawdown > self.config.drawdown_threshold:
            drawdown_penalty = (current_drawdown - self.config.drawdown_threshold) * self.config.max_drawdown_penalty
            base_reward -= drawdown_penalty
        
        # Additional penalty for new maximum drawdown
        if current_drawdown >= self.max_drawdown * 0.95:  # Within 5% of max drawdown
            base_reward -= self.config.max_drawdown_penalty * 0.5
        
        return base_reward
    
    def reset(self):
        """Reset for new episode"""
        super().reset()
        self.peak_value = 0
        self.max_drawdown = 0


class TradingFrequencyPenalty(RewardFunction):
    """Trading frequency penalty reward function"""
    
    def calculate(self, portfolio_return: float, portfolio_value: float,
                 positions: Dict[str, float], trades: List[Any],
                 market_data: Dict[str, Any]) -> float:
        """Calculate trading frequency penalty"""
        self.update_history(portfolio_return, portfolio_value, positions, trades)
        
        # Base reward from return
        base_reward = portfolio_return * self.config.reward_scaling
        
        # Calculate trading frequency penalty
        num_trades = len(trades)
        
        if num_trades > self.config.optimal_trades_per_day:
            excess_trades = num_trades - self.config.optimal_trades_per_day
            trading_penalty = excess_trades * self.config.trading_penalty_coef
            base_reward -= trading_penalty
        
        # Calculate transaction costs penalty
        total_costs = sum(getattr(trade, 'commission', 0) + getattr(trade, 'slippage', 0) 
                         for trade in trades)
        
        if portfolio_value > 0:
            cost_ratio = total_costs / portfolio_value
            cost_penalty = cost_ratio * self.config.cost_penalty_multiplier
            base_reward -= cost_penalty
        
        return base_reward


class PortfolioDiversityReward(RewardFunction):
    """Portfolio diversification reward function"""
    
    def calculate(self, portfolio_return: float, portfolio_value: float,
                 positions: Dict[str, float], trades: List[Any],
                 market_data: Dict[str, Any]) -> float:
        """Calculate portfolio diversity reward"""
        self.update_history(portfolio_return, portfolio_value, positions, trades)
        
        # Base reward from return
        base_reward = portfolio_return * self.config.reward_scaling
        
        # Calculate portfolio concentration
        active_positions = {k: v for k, v in positions.items() 
                          if k != 'cash' and abs(v) > 0.01}  # Positions > 1%
        
        if len(active_positions) < self.config.min_positions:
            # Penalty for insufficient diversification
            concentration_penalty = (self.config.min_positions - len(active_positions)) * 0.1
            base_reward -= concentration_penalty
        
        if len(active_positions) > 0:
            # Calculate Herfindahl-Hirschman Index (HHI) for concentration
            weights = np.array(list(active_positions.values()))
            weights = np.abs(weights) / np.sum(np.abs(weights))  # Normalize to sum to 1
            hhi = np.sum(weights ** 2)
            
            # Diversification ratio (lower HHI = better diversification)
            diversification_ratio = 1 - hhi
            
            # Reward for good diversification
            if diversification_ratio > self.config.diversity_target:
                diversity_bonus = (diversification_ratio - self.config.diversity_target) * 0.5
                base_reward += diversity_bonus
            else:
                # Penalty for concentration
                concentration_penalty = (self.config.diversity_target - diversification_ratio) * self.config.concentration_penalty
                base_reward -= concentration_penalty
        
        return base_reward


class RiskParityReward(RewardFunction):
    """Risk parity reward function"""
    
    def calculate(self, portfolio_return: float, portfolio_value: float,
                 positions: Dict[str, float], trades: List[Any],
                 market_data: Dict[str, Any]) -> float:
        """Calculate risk parity reward"""
        self.update_history(portfolio_return, portfolio_value, positions, trades)
        
        base_reward = portfolio_return * self.config.reward_scaling
        
        # Get volatilities from market data
        volatilities = market_data.get('volatilities', {})
        
        if not volatilities:
            return base_reward
        
        # Calculate risk contributions
        active_positions = {k: v for k, v in positions.items() 
                          if k != 'cash' and abs(v) > 0.01}
        
        if len(active_positions) < 2:
            return base_reward
        
        risk_contributions = []
        total_weight = sum(abs(v) for v in active_positions.values())
        
        for asset, weight in active_positions.items():
            if asset in volatilities:
                normalized_weight = abs(weight) / total_weight
                vol = volatilities[asset]
                risk_contribution = normalized_weight * vol
                risk_contributions.append(risk_contribution)
        
        if len(risk_contributions) > 1:
            # Calculate risk parity score (lower variance = better)
            risk_var = np.var(risk_contributions)
            risk_penalty = risk_var * self.config.volatility_penalty
            base_reward -= risk_penalty
        
        return base_reward


class ValueAtRiskPenalty(RewardFunction):
    """Value at Risk penalty reward function"""
    
    def calculate(self, portfolio_return: float, portfolio_value: float,
                 positions: Dict[str, float], trades: List[Any],
                 market_data: Dict[str, Any]) -> float:
        """Calculate VaR penalty reward"""
        self.update_history(portfolio_return, portfolio_value, positions, trades)
        
        base_reward = portfolio_return * self.config.reward_scaling
        
        if len(self.returns_history) < 30:  # Need minimum history for VaR
            return base_reward
        
        # Calculate rolling VaR
        recent_returns = np.array(self.returns_history[-self.config.var_window:])
        
        if len(recent_returns) > 0:
            # Calculate VaR at specified confidence level
            var_cutoff = np.percentile(recent_returns, self.config.var_confidence * 100)
            
            # Penalty if current return is worse than VaR
            if portfolio_return < var_cutoff:
                var_penalty = abs(portfolio_return - var_cutoff) * 5.0
                base_reward -= var_penalty
        
        return base_reward


class CompositeReward(RewardFunction):
    """Composite reward function combining multiple reward types"""
    
    def __init__(self, config: RewardConfig = None, reward_weights: Dict[str, float] = None):
        super().__init__(config)
        
        # Default reward weights
        if reward_weights is None:
            reward_weights = {
                'sharpe': 0.4,
                'drawdown': 0.3,
                'trading_frequency': 0.1,
                'diversity': 0.2
            }
        
        self.reward_weights = reward_weights
        
        # Initialize component reward functions
        self.reward_functions = {
            'sharpe': SharpeReward(config),
            'drawdown': MaxDrawdownPenalty(config),
            'trading_frequency': TradingFrequencyPenalty(config),
            'diversity': PortfolioDiversityReward(config),
            'sortino': SortinoReward(config),
            'risk_parity': RiskParityReward(config),
            'var_penalty': ValueAtRiskPenalty(config)
        }
    
    def calculate(self, portfolio_return: float, portfolio_value: float,
                 positions: Dict[str, float], trades: List[Any],
                 market_data: Dict[str, Any]) -> float:
        """Calculate composite reward"""
        total_reward = 0.0
        reward_components = {}
        
        for reward_name, weight in self.reward_weights.items():
            if reward_name in self.reward_functions:
                component_reward = self.reward_functions[reward_name].calculate(
                    portfolio_return, portfolio_value, positions, trades, market_data
                )
                weighted_reward = component_reward * weight
                total_reward += weighted_reward
                reward_components[reward_name] = component_reward
        
        # Store component breakdown for analysis
        self.last_reward_components = reward_components
        
        return total_reward
    
    def reset(self):
        """Reset all component reward functions"""
        super().reset()
        for reward_func in self.reward_functions.values():
            reward_func.reset()
    
    def get_reward_breakdown(self) -> Dict[str, float]:
        """Get breakdown of last reward calculation"""
        return getattr(self, 'last_reward_components', {})


class AdaptiveReward(RewardFunction):
    """Adaptive reward function that adjusts weights based on performance"""
    
    def __init__(self, config: RewardConfig = None):
        super().__init__(config)
        self.base_weights = {
            'return': 0.3,
            'sharpe': 0.2,
            'drawdown': 0.2,
            'trading_frequency': 0.15,
            'diversity': 0.15
        }
        self.current_weights = self.base_weights.copy()
        self.performance_history = []
        self.adaptation_frequency = 100  # Adapt weights every 100 steps
        self.step_count = 0
    
    def calculate(self, portfolio_return: float, portfolio_value: float,
                 positions: Dict[str, float], trades: List[Any],
                 market_data: Dict[str, Any]) -> float:
        """Calculate adaptive reward"""
        self.update_history(portfolio_return, portfolio_value, positions, trades)
        self.step_count += 1
        
        # Calculate individual reward components
        components = {}
        components['return'] = portfolio_return * self.config.reward_scaling
        
        # Sharpe component
        if len(self.returns_history) >= self.config.min_sharpe_periods:
            recent_returns = np.array(self.returns_history[-30:])
            excess_returns = recent_returns - (self.config.risk_free_rate / 252)
            if np.std(excess_returns) > 0:
                components['sharpe'] = np.mean(excess_returns) / np.std(excess_returns) * 0.1
            else:
                components['sharpe'] = 0.0
        else:
            components['sharpe'] = 0.0
        
        # Drawdown component
        if len(self.portfolio_values) > 0:
            peak = max(self.portfolio_values)
            drawdown = (peak - portfolio_value) / peak if peak > 0 else 0
            components['drawdown'] = -drawdown * self.config.max_drawdown_penalty
        else:
            components['drawdown'] = 0.0
        
        # Trading frequency component
        num_trades = len(trades)
        if num_trades > self.config.optimal_trades_per_day:
            components['trading_frequency'] = -(num_trades - self.config.optimal_trades_per_day) * self.config.trading_penalty_coef
        else:
            components['trading_frequency'] = 0.0
        
        # Diversity component
        active_positions = {k: v for k, v in positions.items() if k != 'cash' and abs(v) > 0.01}
        if len(active_positions) > 0:
            weights = np.array(list(active_positions.values()))
            weights = np.abs(weights) / np.sum(np.abs(weights))
            hhi = np.sum(weights ** 2)
            diversification_ratio = 1 - hhi
            components['diversity'] = (diversification_ratio - 0.5) * 2.0  # Reward good diversification
        else:
            components['diversity'] = -1.0  # Penalty for no positions
        
        # Calculate weighted reward
        total_reward = sum(self.current_weights[comp] * value for comp, value in components.items())
        
        # Store performance for adaptation
        self.performance_history.append({
            'total_reward': total_reward,
            'components': components.copy(),
            'portfolio_value': portfolio_value
        })
        
        # Adapt weights periodically
        if self.step_count % self.adaptation_frequency == 0:
            self._adapt_weights()
        
        return total_reward
    
    def _adapt_weights(self):
        """Adapt reward weights based on recent performance"""
        if len(self.performance_history) < self.adaptation_frequency:
            return
        
        recent_performance = self.performance_history[-self.adaptation_frequency:]
        
        # Calculate correlation between component rewards and total performance
        total_values = [p['portfolio_value'] for p in recent_performance]
        if len(set(total_values)) <= 1:  # No variance in portfolio values
            return
        
        value_returns = np.diff(total_values) / np.array(total_values[:-1])
        
        # Update weights based on component effectiveness
        for component in self.current_weights:
            component_values = [p['components'][component] for p in recent_performance[:-1]]
            
            if len(set(component_values)) > 1:  # Component has variance
                correlation = np.corrcoef(component_values, value_returns)[0, 1]
                if not np.isnan(correlation):
                    # Adjust weight based on correlation with performance
                    adjustment = correlation * 0.1  # Small adjustment
                    self.current_weights[component] = max(0.05, min(0.5, 
                        self.current_weights[component] + adjustment))
        
        # Renormalize weights
        total_weight = sum(self.current_weights.values())
        self.current_weights = {k: v / total_weight for k, v in self.current_weights.items()}
        
        logger.info(f"Adapted reward weights: {self.current_weights}")
    
    def reset(self):
        """Reset adaptive reward function"""
        super().reset()
        self.current_weights = self.base_weights.copy()
        self.performance_history = []
        self.step_count = 0


# Factory function
def create_reward_function(reward_type: str = 'composite', 
                          config: Dict[str, Any] = None,
                          reward_weights: Dict[str, float] = None) -> RewardFunction:
    """Create reward function with configuration"""
    if config:
        reward_config = RewardConfig(**config)
    else:
        reward_config = RewardConfig()
    
    if reward_type == 'simple':
        return SimpleReturnReward(reward_config)
    elif reward_type == 'sharpe':
        return SharpeReward(reward_config)
    elif reward_type == 'sortino':
        return SortinoReward(reward_config)
    elif reward_type == 'drawdown':
        return MaxDrawdownPenalty(reward_config)
    elif reward_type == 'trading_frequency':
        return TradingFrequencyPenalty(reward_config)
    elif reward_type == 'diversity':
        return PortfolioDiversityReward(reward_config)
    elif reward_type == 'risk_parity':
        return RiskParityReward(reward_config)
    elif reward_type == 'var_penalty':
        return ValueAtRiskPenalty(reward_config)
    elif reward_type == 'composite':
        return CompositeReward(reward_config, reward_weights)
    elif reward_type == 'adaptive':
        return AdaptiveReward(reward_config)
    else:
        raise ValueError(f"Unknown reward type: {reward_type}")


# Export classes and functions
__all__ = [
    'RewardFunction',
    'RewardConfig',
    'RewardType',
    'SimpleReturnReward',
    'SharpeReward',
    'SortinoReward',
    'MaxDrawdownPenalty',
    'TradingFrequencyPenalty',
    'PortfolioDiversityReward',
    'RiskParityReward',
    'ValueAtRiskPenalty',
    'CompositeReward',
    'AdaptiveReward',
    'create_reward_function'
]