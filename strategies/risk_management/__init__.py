"""
Risk Management Module
Comprehensive risk management system with position sizing, risk metrics,
portfolio optimization, and stop-loss management
"""

from .position_sizing import (
    PositionSizingConfig, KellyCriterion, VolatilityBasedSizing,
    CorrelationAwareOptimizer, DynamicLeverageManager, PositionSizer,
    create_position_sizer
)

from .risk_metrics import (
    RiskMetricsConfig, VaRResult, StressTestResult, VaRCalculator,
    StressTester, RiskMetricsCalculator, create_risk_calculator
)

from .portfolio_optimizer import (
    OptimizerConfig, OptimizationResult, MeanVarianceOptimizer,
    BlackLittermanOptimizer, RiskParityOptimizer, MaxDiversificationOptimizer,
    CovarianceEstimator, PortfolioOptimizer, create_portfolio_optimizer
)

from .stop_loss_manager import (
    StopType, StopStatus, StopLossConfig, StopLossOrder,
    FixedStopLoss, TrailingStopLoss, TimeBasedStopLoss,
    VolatilityAdjustedStopLoss, CorrelationBasedStopLoss,
    StopLossManager, create_stop_loss_manager
)

__all__ = [
    # Position Sizing
    'PositionSizingConfig',
    'KellyCriterion',
    'VolatilityBasedSizing',
    'CorrelationAwareOptimizer',
    'DynamicLeverageManager',
    'PositionSizer',
    'create_position_sizer',
    
    # Risk Metrics
    'RiskMetricsConfig',
    'VaRResult',
    'StressTestResult',
    'VaRCalculator',
    'StressTester',
    'RiskMetricsCalculator',
    'create_risk_calculator',
    
    # Portfolio Optimization
    'OptimizerConfig',
    'OptimizationResult',
    'MeanVarianceOptimizer',
    'BlackLittermanOptimizer',
    'RiskParityOptimizer',
    'MaxDiversificationOptimizer',
    'CovarianceEstimator',
    'PortfolioOptimizer',
    'create_portfolio_optimizer',
    
    # Stop Loss Management
    'StopType',
    'StopStatus',
    'StopLossConfig',
    'StopLossOrder',
    'FixedStopLoss',
    'TrailingStopLoss',
    'TimeBasedStopLoss',
    'VolatilityAdjustedStopLoss',
    'CorrelationBasedStopLoss',
    'StopLossManager',
    'create_stop_loss_manager'
]