#!/usr/bin/env python3
"""
Test script for trading strategies and backtesting system
"""

import numpy as np
import pandas as pd
import sys
from datetime import datetime, timedelta
import traceback

# Add the project root to the path
sys.path.append('/Users/user/Projects/fire/ai-trading-system')

def generate_sample_market_data(start_date: str = "2020-01-01", 
                               end_date: str = "2023-12-31",
                               symbols: list = None) -> pd.DataFrame:
    """Generate sample market data for testing"""
    if symbols is None:
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    dates = dates[dates.dayofweek < 5]  # Remove weekends
    
    np.random.seed(42)
    data = {}
    
    for symbol in symbols:
        # Generate realistic price data
        n_days = len(dates)
        base_price = np.random.uniform(50, 300)
        
        # Generate returns with some trend and volatility clustering
        returns = np.random.normal(0.0005, 0.02, n_days)
        
        # Add trend
        trend = np.linspace(-0.1, 0.1, n_days) * np.random.choice([-1, 1])
        returns += trend / n_days
        
        # Volatility clustering
        for i in range(1, n_days):
            if abs(returns[i-1]) > 0.03:  # High volatility day
                returns[i] *= 1.5
        
        # Generate OHLCV data
        prices = base_price * np.exp(np.cumsum(returns))
        
        data[f'{symbol}_open'] = prices * np.random.uniform(0.995, 1.005, n_days)
        data[f'{symbol}_high'] = prices * np.random.uniform(1.0, 1.02, n_days)
        data[f'{symbol}_low'] = prices * np.random.uniform(0.98, 1.0, n_days)
        data[f'{symbol}_close'] = prices
        data[f'{symbol}_volume'] = np.random.randint(1000000, 10000000, n_days)
    
    df = pd.DataFrame(data, index=dates)
    return df

def test_base_strategy():
    """Test base strategy functionality"""
    print("Testing Base Strategy...")
    
    try:
        from strategies import BaseStrategy, StrategyConfig, Signal, SignalType
        
        # Create a simple test strategy
        class TestStrategy(BaseStrategy):
            def generate_signals(self, data, timestamp):
                # Simple momentum strategy
                if len(data) < 20:
                    return []
                
                signals = []
                for symbol in self.config.symbols:
                    if f"{symbol}_close" in data.columns:
                        prices = data[f"{symbol}_close"].values
                        if len(prices) >= 20:
                            recent_return = (prices[-1] - prices[-20]) / prices[-20]
                            if recent_return > 0.05:  # 5% momentum
                                signal = Signal(
                                    symbol=symbol,
                                    signal_type=SignalType.BUY,
                                    timestamp=timestamp,
                                    price=prices[-1],
                                    confidence=0.8,
                                    size=0.1
                                )
                                signals.append(signal)
                
                return signals
            
            def calculate_position_size(self, signal, current_price, available_cash):
                return (available_cash * signal.size) / current_price
        
        # Test configuration
        config = StrategyConfig(
            symbols=['AAPL', 'MSFT'],
            initial_capital=100000,
            max_position_size=0.2
        )
        
        strategy = TestStrategy(config)
        print(f"  ✓ Created test strategy with ${config.initial_capital:,.0f} capital")
        
        # Test signal generation
        market_data = generate_sample_market_data(symbols=['AAPL', 'MSFT'])
        recent_data = market_data.tail(50)
        
        signals = strategy.generate_signals(recent_data, datetime.now())
        print(f"  ✓ Generated {len(signals)} signals")
        
        # Test position management
        if signals:
            signal = signals[0]
            current_prices = {symbol: recent_data[f"{symbol}_close"].iloc[-1] 
                            for symbol in config.symbols}
            
            executed_trades = strategy.process_signals([signal], current_prices, datetime.now())
            print(f"  ✓ Processed signals, executed {len(executed_trades)} trades")
            
            strategy.update_positions(current_prices, datetime.now())
            portfolio_value = strategy.get_portfolio_value(current_prices)
            print(f"  ✓ Portfolio value: ${portfolio_value:,.2f}")
        
        # Test metrics
        metrics = strategy.get_performance_metrics()
        print(f"  ✓ Performance metrics: {len(metrics)} metrics calculated")
        
        print("✓ Base Strategy test passed!\n")
        return True
        
    except Exception as e:
        print(f"✗ Base Strategy test failed: {e}")
        traceback.print_exc()
        return False

def test_momentum_strategy():
    """Test momentum strategy"""
    print("Testing Momentum Strategy...")
    
    try:
        from strategies import MomentumStrategy, MomentumConfig
        
        config = MomentumConfig(
            symbols=['AAPL', 'MSFT', 'GOOGL'],
            initial_capital=100000,
            momentum_window=20,
            momentum_threshold=0.02
        )
        
        strategy = MomentumStrategy(config)
        print(f"  ✓ Created momentum strategy")
        
        # Generate test data
        market_data = generate_sample_market_data(symbols=config.symbols)
        
        # Test signal generation
        signals = strategy.generate_signals(market_data.tail(100), datetime.now())
        print(f"  ✓ Generated {len(signals)} momentum signals")
        
        # Test strategy-specific metrics
        if hasattr(strategy, 'get_strategy_metrics'):
            strategy_metrics = strategy.get_strategy_metrics()
            print(f"  ✓ Strategy-specific metrics available")
        
        print("✓ Momentum Strategy test passed!\n")
        return True
        
    except Exception as e:
        print(f"✗ Momentum Strategy test failed: {e}")
        traceback.print_exc()
        return False

def test_mean_reversion_strategy():
    """Test mean reversion strategy"""
    print("Testing Mean Reversion Strategy...")
    
    try:
        from strategies import MeanReversionStrategy, MeanReversionConfig
        
        config = MeanReversionConfig(
            symbols=['AAPL', 'MSFT'],
            initial_capital=100000,
            zscore_threshold=2.0,
            lookback_window=20
        )
        
        strategy = MeanReversionStrategy(config)
        print(f"  ✓ Created mean reversion strategy")
        
        # Generate test data
        market_data = generate_sample_market_data(symbols=config.symbols)
        
        # Test signal generation
        signals = strategy.generate_signals(market_data.tail(100), datetime.now())
        print(f"  ✓ Generated {len(signals)} mean reversion signals")
        
        print("✓ Mean Reversion Strategy test passed!\n")
        return True
        
    except Exception as e:
        print(f"✗ Mean Reversion Strategy test failed: {e}")
        traceback.print_exc()
        return False

def test_pairs_trading_strategy():
    """Test pairs trading strategy"""
    print("Testing Pairs Trading Strategy...")
    
    try:
        from strategies import PairsTradingStrategy, PairsTradingConfig
        
        config = PairsTradingConfig(
            symbols=['AAPL', 'MSFT', 'GOOGL', 'AMZN'],
            initial_capital=100000,
            min_correlation=0.6,
            zscore_entry=2.0
        )
        
        strategy = PairsTradingStrategy(config)
        print(f"  ✓ Created pairs trading strategy")
        
        # Generate test data with some correlation
        market_data = generate_sample_market_data(symbols=config.symbols)
        
        # Test signal generation
        signals = strategy.generate_signals(market_data.tail(300), datetime.now())
        print(f"  ✓ Generated {len(signals)} pairs trading signals")
        
        # Test pairs summary
        if hasattr(strategy, 'get_pairs_summary'):
            pairs_summary = strategy.get_pairs_summary()
            print(f"  ✓ Pairs summary: {len(pairs_summary)} pairs analyzed")
        
        print("✓ Pairs Trading Strategy test passed!\n")
        return True
        
    except Exception as e:
        print(f"✗ Pairs Trading Strategy test failed: {e}")
        traceback.print_exc()
        return False

def test_ml_ensemble_strategy():
    """Test ML ensemble strategy"""
    print("Testing ML Ensemble Strategy...")
    
    try:
        from strategies import MLEnsembleStrategy, MLEnsembleConfig
        
        config = MLEnsembleConfig(
            symbols=['AAPL', 'MSFT'],
            initial_capital=100000,
            models=['random_forest', 'logistic_regression'],
            min_training_samples=100
        )
        
        strategy = MLEnsembleStrategy(config)
        print(f"  ✓ Created ML ensemble strategy with {len(config.models)} models")
        
        # Generate test data
        market_data = generate_sample_market_data(symbols=config.symbols)
        
        # Test signal generation (may not generate signals initially due to training requirements)
        signals = strategy.generate_signals(market_data.tail(200), datetime.now())
        print(f"  ✓ Generated {len(signals)} ML signals")
        
        print("✓ ML Ensemble Strategy test passed!\n")
        return True
        
    except Exception as e:
        print(f"✗ ML Ensemble Strategy test failed: {e}")
        traceback.print_exc()
        return False

def test_multi_timeframe_strategy():
    """Test multi-timeframe strategy"""
    print("Testing Multi-Timeframe Strategy...")
    
    try:
        from strategies import MultiTimeframeStrategy, MultiTimeframeConfig, TimeframeConfig
        
        timeframes = [
            TimeframeConfig("short", 5, 0.3),
            TimeframeConfig("medium", 20, 0.4),
            TimeframeConfig("long", 60, 0.3)
        ]
        
        config = MultiTimeframeConfig(
            symbols=['AAPL', 'MSFT'],
            initial_capital=100000,
            timeframes=timeframes,
            min_timeframe_agreement=0.6
        )
        
        strategy = MultiTimeframeStrategy(config)
        print(f"  ✓ Created multi-timeframe strategy with {len(timeframes)} timeframes")
        
        # Generate test data
        market_data = generate_sample_market_data(symbols=config.symbols)
        
        # Test signal generation
        signals = strategy.generate_signals(market_data.tail(100), datetime.now())
        print(f"  ✓ Generated {len(signals)} multi-timeframe signals")
        
        # Test timeframe summary
        if hasattr(strategy, 'get_timeframe_summary'):
            timeframe_summary = strategy.get_timeframe_summary()
            print(f"  ✓ Timeframe analysis available for {len(timeframe_summary)} symbols")
        
        print("✓ Multi-Timeframe Strategy test passed!\n")
        return True
        
    except Exception as e:
        print(f"✗ Multi-Timeframe Strategy test failed: {e}")
        traceback.print_exc()
        return False

def test_backtest_engine():
    """Test backtesting engine"""
    print("Testing Backtest Engine...")
    
    try:
        from strategies import (
            BacktestEngine, BacktestConfig, MomentumStrategy, MomentumConfig,
            create_backtest_engine, run_backtest
        )
        
        # Create test strategy
        strategy_config = MomentumConfig(
            symbols=['AAPL', 'MSFT'],
            initial_capital=100000,
            momentum_window=20
        )
        strategy = MomentumStrategy(strategy_config)
        
        # Generate test data
        start_date = datetime(2022, 1, 1)
        end_date = datetime(2023, 6, 30)
        market_data = generate_sample_market_data(
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d"),
            symbols=['AAPL', 'MSFT']
        )
        
        print(f"  ✓ Generated market data: {len(market_data)} periods")
        
        # Test backtest engine creation
        engine = create_backtest_engine(
            start_date=start_date,
            end_date=end_date,
            initial_capital=100000,
            commission_rate=0.001
        )
        print(f"  ✓ Created backtest engine")
        
        # Load data and run backtest
        engine.load_data(market_data)
        result = engine.run_backtest(strategy)
        
        print(f"  ✓ Backtest completed:")
        print(f"    Total Return: {result.total_return:.2%}")
        print(f"    Sharpe Ratio: {result.sharpe_ratio:.2f}")
        print(f"    Max Drawdown: {result.max_drawdown:.2%}")
        print(f"    Total Trades: {result.total_trades}")
        print(f"    Win Rate: {result.win_rate:.2%}")
        
        # Test convenience function
        result2 = run_backtest(
            strategy, market_data, start_date, end_date, 
            initial_capital=100000
        )
        print(f"  ✓ Convenience function test passed")
        
        print("✓ Backtest Engine test passed!\n")
        return True
        
    except Exception as e:
        print(f"✗ Backtest Engine test failed: {e}")
        traceback.print_exc()
        return False

def test_performance_analytics():
    """Test performance analytics"""
    print("Testing Performance Analytics...")
    
    try:
        from strategies import PerformanceAnalytics, create_performance_analytics
        
        # Generate sample returns
        np.random.seed(42)
        dates = pd.date_range('2022-01-01', '2023-12-31', freq='D')
        returns = pd.Series(np.random.normal(0.0008, 0.015, len(dates)), index=dates)
        benchmark_returns = pd.Series(np.random.normal(0.0005, 0.012, len(dates)), index=dates)
        
        # Create analytics instance
        analytics = create_performance_analytics(risk_free_rate=0.02)
        print(f"  ✓ Created performance analytics")
        
        # Test performance metrics
        metrics = analytics.calculate_performance_metrics(returns, benchmark_returns)
        print(f"  ✓ Performance metrics calculated:")
        print(f"    Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
        print(f"    Max Drawdown: {metrics.max_drawdown:.2%}")
        print(f"    Alpha: {metrics.alpha:.2%}")
        print(f"    Beta: {metrics.beta:.2f}")
        
        # Test attribution analysis
        attribution = analytics.calculate_attribution_analysis(returns, benchmark_returns)
        print(f"  ✓ Attribution analysis completed")
        print(f"    Total Active Return: {attribution.total_active_return:.2%}")
        
        # Test Monte Carlo simulation
        monte_carlo = analytics.run_monte_carlo_simulation(returns, num_simulations=1000)
        print(f"  ✓ Monte Carlo simulation completed:")
        print(f"    Probability of Loss: {monte_carlo.probability_of_loss:.2%}")
        print(f"    95th Percentile Return: {monte_carlo.percentile_returns[95]:.2%}")
        
        # Test risk metrics
        risk_metrics = analytics.calculate_risk_metrics(returns)
        print(f"  ✓ Risk metrics calculated:")
        print(f"    Realized Volatility: {risk_metrics.realized_volatility:.2%}")
        print(f"    Tail Ratio: {risk_metrics.tail_ratio:.2f}")
        
        # Test comprehensive report
        report = analytics.generate_performance_report(returns, benchmark_returns)
        print(f"  ✓ Performance report generated")
        print(f"    Overall Rating: {report['summary']['overall_rating']}")
        
        print("✓ Performance Analytics test passed!\n")
        return True
        
    except Exception as e:
        print(f"✗ Performance Analytics test failed: {e}")
        traceback.print_exc()
        return False

def test_integrated_workflow():
    """Test integrated workflow combining strategies and backtesting"""
    print("Testing Integrated Workflow...")
    
    try:
        from strategies import (
            MomentumStrategy, MomentumConfig,
            BacktestEngine, BacktestConfig,
            PerformanceAnalytics
        )
        
        # Create strategy
        strategy = MomentumStrategy(MomentumConfig(
            symbols=['AAPL', 'MSFT', 'GOOGL'],
            initial_capital=100000,
            momentum_window=15,
            momentum_threshold=0.03
        ))
        
        # Generate data
        market_data = generate_sample_market_data(
            start_date="2022-01-01",
            end_date="2023-12-31",
            symbols=['AAPL', 'MSFT', 'GOOGL']
        )
        
        # Run backtest
        engine = BacktestEngine(BacktestConfig(
            start_date=datetime(2022, 1, 1),
            end_date=datetime(2023, 12, 31),
            initial_capital=100000,
            commission_rate=0.001,
            market_impact_coef=0.0001
        ))
        
        engine.load_data(market_data)
        result = engine.run_backtest(strategy)
        
        # Analyze performance
        analytics = PerformanceAnalytics()
        performance_report = analytics.generate_performance_report(result.returns)
        
        print(f"  ✓ Integrated workflow completed:")
        print(f"    Strategy: Momentum")
        print(f"    Period: 2022-2023")
        print(f"    Final Return: {result.total_return:.2%}")
        print(f"    Sharpe Ratio: {result.sharpe_ratio:.2f}")
        print(f"    Max Drawdown: {result.max_drawdown:.2%}")
        print(f"    Performance Rating: {performance_report['summary']['overall_rating']}")
        
        print("✓ Integrated Workflow test passed!\n")
        return True
        
    except Exception as e:
        print(f"✗ Integrated Workflow test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all strategy tests"""
    print("=" * 70)
    print("TRADING STRATEGIES AND BACKTESTING TEST SUITE")
    print("=" * 70)
    print(f"Test started at: {datetime.now()}")
    print()
    
    tests = [
        test_base_strategy,
        test_momentum_strategy,
        test_mean_reversion_strategy,
        test_pairs_trading_strategy,
        test_ml_ensemble_strategy,
        test_multi_timeframe_strategy,
        test_backtest_engine,
        test_performance_analytics,
        test_integrated_workflow
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} crashed: {e}")
            failed += 1
    
    print("=" * 70)
    print("TEST RESULTS")
    print("=" * 70)
    print(f"✓ Passed: {passed}")
    print(f"✗ Failed: {failed}")
    print(f"Total: {passed + failed}")
    
    if failed == 0:
        print("\n🎉 ALL TESTS PASSED! 🎉")
        print("Trading strategies and backtesting system are ready!")
    else:
        print(f"\n⚠️  {failed} tests failed. Please check the errors above.")
    
    print(f"\nTest completed at: {datetime.now()}")
    
    return failed == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)