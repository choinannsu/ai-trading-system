"""
Trading Strategies Module
Comprehensive trading strategy framework with backtesting capabilities
"""

from .base_strategy import (
    BaseStrategy, StrategyConfig, Signal, SignalType, Position, 
    PositionType, Trade, TradeStatus, create_strategy
)

# Import strategy implementations
from .implementations import (
    MomentumStrategy, MomentumConfig,
    MeanReversionStrategy, MeanReversionConfig,
    PairsTradingStrategy, PairsTradingConfig, PairInfo,
    MLEnsembleStrategy, MLEnsembleConfig,
    MultiTimeframeStrategy, MultiTimeframeConfig, TimeframeConfig,
    TimeframeAnalysis, TimeframeType, TrendDirection
)

# Import backtesting components
from .backtesting import (
    BacktestEngine, BacktestConfig, BacktestResult,
    PerformanceAnalytics, PerformanceMetrics, MonteCarloAnalysis,
    AttributionAnalysis, RiskMetrics, create_backtest_engine, run_backtest,
    create_performance_analytics
)

__all__ = [
    # Base classes
    'BaseStrategy',
    'StrategyConfig', 
    'Signal',
    'SignalType',
    'Position',
    'PositionType',
    'Trade',
    'TradeStatus',
    'create_strategy',
    
    # Strategy implementations
    'MomentumStrategy',
    'MomentumConfig',
    'MeanReversionStrategy',
    'MeanReversionConfig',
    'PairsTradingStrategy',
    'PairsTradingConfig',
    'PairInfo',
    'MLEnsembleStrategy',
    'MLEnsembleConfig',
    'MultiTimeframeStrategy',
    'MultiTimeframeConfig',
    'TimeframeConfig',
    'TimeframeAnalysis',
    'TimeframeType',
    'TrendDirection',
    
    # Backtesting
    'BacktestEngine',
    'BacktestConfig',
    'BacktestResult',
    'PerformanceAnalytics',
    'PerformanceMetrics',
    'MonteCarloAnalysis',
    'AttributionAnalysis',
    'RiskMetrics',
    'create_backtest_engine',
    'run_backtest',
    'create_performance_analytics'
]