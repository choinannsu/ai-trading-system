#!/usr/bin/env python3
"""
Test script for risk management system
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

def test_position_sizing():
    """Test position sizing system"""
    print("Testing Position Sizing System...")
    
    try:
        from strategies.risk_management import create_position_sizer, PositionSizingConfig
        
        # Create configuration
        config = PositionSizingConfig(
            kelly_window=100,
            target_volatility=0.15,
            max_position_size=0.2
        )
        
        # Create position sizer
        position_sizer = create_position_sizer(config.__dict__)
        print(f"  ✓ Created position sizer with target volatility {config.target_volatility}")
        
        # Generate test data
        market_data = generate_sample_market_data(symbols=['AAPL', 'MSFT', 'GOOGL'])
        
        # Test signals
        signals = {
            'AAPL': 0.8,   # Strong buy signal
            'MSFT': -0.3,  # Weak sell signal
            'GOOGL': 0.5   # Medium buy signal
        }
        
        # Calculate position sizes
        portfolio_value = 100000
        position_sizes = position_sizer.calculate_position_sizes(
            signals, market_data, portfolio_value
        )
        
        print(f"  ✓ Calculated position sizes for {len(position_sizes)} symbols:")
        for symbol, size in position_sizes.items():
            print(f"    {symbol}: {size:.3f}")
        
        # Test Kelly Criterion
        kelly_fraction = position_sizer.kelly_criterion.calculate_kelly_fraction(
            market_data['AAPL_close'].pct_change().dropna()
        )
        print(f"  ✓ Kelly fraction for AAPL: {kelly_fraction:.3f}")
        
        # Test sizing metrics
        metrics = position_sizer.get_sizing_metrics()
        print(f"  ✓ Retrieved sizing metrics: {len(metrics)} metrics")
        
        print("✓ Position Sizing test passed!\n")
        return True
        
    except Exception as e:
        print(f"✗ Position Sizing test failed: {e}")
        traceback.print_exc()
        return False

def test_risk_metrics():
    """Test risk metrics system"""
    print("Testing Risk Metrics System...")
    
    try:
        from strategies.risk_management import create_risk_calculator, RiskMetricsConfig
        
        # Create configuration
        config = RiskMetricsConfig(
            var_confidence_levels=[0.95, 0.99],
            var_method="historical",
            mc_simulations=1000
        )
        
        # Create risk calculator
        risk_calculator = create_risk_calculator(config.__dict__)
        print(f"  ✓ Created risk calculator with {config.var_method} VaR method")
        
        # Generate test data
        market_data = generate_sample_market_data(symbols=['AAPL', 'MSFT'])
        
        # Create portfolio returns
        portfolio_returns = (market_data['AAPL_close'].pct_change() * 0.6 + 
                           market_data['MSFT_close'].pct_change() * 0.4).dropna()
        
        # Asset returns for analysis
        asset_returns = pd.DataFrame({
            'AAPL': market_data['AAPL_close'].pct_change(),
            'MSFT': market_data['MSFT_close'].pct_change()
        }).dropna()
        
        # Portfolio weights
        portfolio_weights = {'AAPL': 0.6, 'MSFT': 0.4}
        
        # Calculate comprehensive risk metrics
        risk_metrics = risk_calculator.calculate_comprehensive_risk_metrics(
            portfolio_returns, asset_returns, portfolio_weights, 100000
        )
        
        print(f"  ✓ Calculated comprehensive risk metrics")
        
        # Test VaR metrics
        for var_key, var_result in risk_metrics['var_metrics'].items():
            print(f"    {var_key}: {var_result.var_value:.4f}")
        
        # Test stress testing
        if risk_metrics['stress_test_results']:
            print(f"  ✓ Completed {len(risk_metrics['stress_test_results'])} stress tests")
            for stress_result in risk_metrics['stress_test_results'][:2]:  # Show first 2
                print(f"    {stress_result.scenario_name}: {stress_result.portfolio_impact:.4f}")
        
        # Generate risk report
        risk_report = risk_calculator.generate_risk_report(risk_metrics)
        print(f"  ✓ Generated risk report ({len(risk_report)} characters)")
        
        print("✓ Risk Metrics test passed!\n")
        return True
        
    except Exception as e:
        print(f"✗ Risk Metrics test failed: {e}")
        traceback.print_exc()
        return False

def test_portfolio_optimizer():
    """Test portfolio optimization system"""
    print("Testing Portfolio Optimizer...")
    
    try:
        from strategies.risk_management import create_portfolio_optimizer, OptimizerConfig
        
        # Create configuration
        config = OptimizerConfig(
            optimization_method="mean_variance",
            max_weight=0.4,
            risk_aversion=5.0
        )
        
        # Create optimizer
        optimizer = create_portfolio_optimizer(config.__dict__)
        print(f"  ✓ Created portfolio optimizer with {config.optimization_method} method")
        
        # Generate test data
        market_data = generate_sample_market_data(symbols=['AAPL', 'MSFT', 'GOOGL'])
        
        # Create returns data
        returns_data = pd.DataFrame({
            'AAPL': market_data['AAPL_close'].pct_change(),
            'MSFT': market_data['MSFT_close'].pct_change(),
            'GOOGL': market_data['GOOGL_close'].pct_change()
        }).dropna()
        
        # Test mean-variance optimization
        result = optimizer.optimize_portfolio(returns_data)
        
        print(f"  ✓ Mean-variance optimization completed:")
        print(f"    Expected return: {result.expected_return:.4f}")
        print(f"    Expected volatility: {result.expected_volatility:.4f}")
        print(f"    Sharpe ratio: {result.sharpe_ratio:.4f}")
        
        # Print optimal weights
        print(f"    Optimal weights:")
        for asset, weight in result.weights.items():
            print(f"      {asset}: {weight:.3f}")
        
        # Test optimization comparison
        comparison_results = optimizer.run_optimization_comparison(returns_data)
        print(f"  ✓ Optimization comparison completed: {len(comparison_results)} methods")
        
        for method, opt_result in comparison_results.items():
            print(f"    {method}: Sharpe={opt_result.sharpe_ratio:.3f}")
        
        print("✓ Portfolio Optimizer test passed!\n")
        return True
        
    except Exception as e:
        print(f"✗ Portfolio Optimizer test failed: {e}")
        traceback.print_exc()
        return False

def test_stop_loss_manager():
    """Test stop-loss management system"""
    print("Testing Stop-Loss Manager...")
    
    try:
        from strategies.risk_management import (
            create_stop_loss_manager, StopLossConfig, StopType
        )
        
        # Create configuration
        config = StopLossConfig(
            default_stop_type=StopType.TRAILING,
            trailing_stop_pct=0.05,
            fixed_stop_pct=0.03
        )
        
        # Create stop-loss manager
        stop_manager = create_stop_loss_manager(config.__dict__)
        print(f"  ✓ Created stop-loss manager with {config.default_stop_type.value} default")
        
        # Generate test data
        market_data = generate_sample_market_data(symbols=['AAPL'])
        
        # Test position entry
        symbol = 'AAPL'
        entry_price = 150.0
        quantity = 100  # Long position
        entry_time = datetime.now()
        
        # Create stop orders
        stop_types = [StopType.FIXED, StopType.TRAILING]
        stop_orders = stop_manager.create_stop_orders(
            symbol, entry_price, quantity, entry_time, stop_types,
            price_data=market_data[['AAPL_high', 'AAPL_low', 'AAPL_close']]
        )
        
        print(f"  ✓ Created {len(stop_orders)} stop orders:")
        for stop in stop_orders:
            print(f"    {stop.stop_type.value}: stop_price={stop.stop_price:.2f}")
        
        # Test stop updates
        current_price = 155.0  # Price moved up
        triggered_stops = stop_manager.update_stops(
            symbol, current_price, datetime.now(),
            price_data=market_data[['AAPL_high', 'AAPL_low', 'AAPL_close']]
        )
        
        print(f"  ✓ Updated stops, {len(triggered_stops)} stops triggered")
        
        # Test stop summary
        summary = stop_manager.get_stop_summary()
        print(f"  ✓ Stop summary: {summary['total_active_stops']} active stops")
        
        # Test stop cancellation
        stop_manager.cancel_stops(symbol, [StopType.FIXED])
        print(f"  ✓ Cancelled fixed stops for {symbol}")
        
        print("✓ Stop-Loss Manager test passed!\n")
        return True
        
    except Exception as e:
        print(f"✗ Stop-Loss Manager test failed: {e}")
        traceback.print_exc()
        return False

def test_integrated_risk_management():
    """Test integrated risk management workflow"""
    print("Testing Integrated Risk Management...")
    
    try:
        from strategies.risk_management import (
            create_position_sizer, create_risk_calculator, 
            create_portfolio_optimizer, create_stop_loss_manager
        )
        
        # Generate market data
        market_data = generate_sample_market_data(symbols=['AAPL', 'MSFT', 'GOOGL'])
        
        # Create returns data
        returns_data = pd.DataFrame({
            'AAPL': market_data['AAPL_close'].pct_change(),
            'MSFT': market_data['MSFT_close'].pct_change(),
            'GOOGL': market_data['GOOGL_close'].pct_change()
        }).dropna()
        
        # 1. Portfolio Optimization
        optimizer = create_portfolio_optimizer()
        optimization_result = optimizer.optimize_portfolio(returns_data)
        print(f"  ✓ Portfolio optimization: Sharpe={optimization_result.sharpe_ratio:.3f}")
        
        # 2. Position Sizing
        position_sizer = create_position_sizer()
        signals = {symbol: 0.5 for symbol in optimization_result.weights.keys()}
        position_sizes = position_sizer.calculate_position_sizes(
            signals, market_data, 100000
        )
        print(f"  ✓ Position sizing completed for {len(position_sizes)} assets")
        
        # 3. Risk Metrics
        portfolio_returns = sum(
            returns_data[asset] * weight 
            for asset, weight in optimization_result.weights.items()
        )
        
        risk_calculator = create_risk_calculator()
        risk_metrics = risk_calculator.calculate_comprehensive_risk_metrics(
            portfolio_returns, returns_data, optimization_result.weights, 100000
        )
        print(f"  ✓ Risk metrics calculated: VaR_95={list(risk_metrics['var_metrics'].values())[0].var_value:.4f}")
        
        # 4. Stop-Loss Management
        stop_manager = create_stop_loss_manager()
        
        for symbol, weight in optimization_result.weights.items():
            if abs(weight) > 0.01:  # Only create stops for significant positions
                current_price = market_data[f'{symbol}_close'].iloc[-1]
                quantity = (100000 * weight) / current_price
                
                stop_orders = stop_manager.create_stop_orders(
                    symbol, current_price, quantity, datetime.now()
                )
                
        summary = stop_manager.get_stop_summary()
        print(f"  ✓ Stop-loss orders created: {summary['total_active_stops']} active stops")
        
        # Integration summary
        print(f"  ✓ Integrated risk management workflow completed:")
        print(f"    Portfolio expected return: {optimization_result.expected_return:.4f}")
        print(f"    Portfolio volatility: {optimization_result.expected_volatility:.4f}")
        print(f"    Active risk controls: {summary['total_active_stops']} stop orders")
        
        print("✓ Integrated Risk Management test passed!\n")
        return True
        
    except Exception as e:
        print(f"✗ Integrated Risk Management test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all risk management tests"""
    print("=" * 70)
    print("RISK MANAGEMENT SYSTEM TEST SUITE")
    print("=" * 70)
    print(f"Test started at: {datetime.now()}")
    print()
    
    tests = [
        test_position_sizing,
        test_risk_metrics,
        test_portfolio_optimizer,
        test_stop_loss_manager,
        test_integrated_risk_management
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
        print("Risk management system is ready!")
    else:
        print(f"\n⚠️  {failed} tests failed. Please check the errors above.")
    
    print(f"\nTest completed at: {datetime.now()}")
    
    return failed == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)