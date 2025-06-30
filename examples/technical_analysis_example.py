#!/usr/bin/env python3
"""
Advanced Technical Analysis Example
Demonstrates usage of indicators, pattern recognition, and market regime detection
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import asyncio

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.technical import (
    create_advanced_indicators,
    create_pattern_recognition, 
    create_market_regime_detector,
    TimeFrame,
    PatternType,
    MarketState
)
from utils.logger import get_logger

logger = get_logger(__name__)


def generate_sample_data(days: int = 252) -> pd.DataFrame:
    """Generate realistic sample OHLCV data"""
    
    # Generate random walk with trend and volatility clustering
    np.random.seed(42)
    returns = np.random.normal(0.0005, 0.02, days)  # Daily returns
    
    # Add trend component
    trend = np.linspace(0, 0.3, days)
    returns += trend / days
    
    # Add volatility clustering (simple GARCH effect)
    vol = np.zeros(days)
    vol[0] = 0.02
    for i in range(1, days):
        vol[i] = 0.9 * vol[i-1] + 0.1 * abs(returns[i-1])
    
    returns = returns + np.random.normal(0, vol)
    
    # Generate price series
    prices = 100 * np.exp(np.cumsum(returns))
    
    # Generate OHLC from close prices
    data = []
    for i in range(days):
        close = prices[i]
        
        # Generate intraday high/low around close
        daily_range = abs(np.random.normal(0, vol[i] * close * 2))
        high = close + daily_range * np.random.uniform(0.3, 0.7)
        low = close - daily_range * np.random.uniform(0.3, 0.7)
        
        # Generate open (close to previous close)
        if i == 0:
            open_price = close * np.random.uniform(0.99, 1.01)
        else:
            open_price = prices[i-1] * np.random.uniform(0.995, 1.005)
        
        # Ensure OHLC logic
        high = max(high, open_price, close)
        low = min(low, open_price, close)
        
        # Generate volume (correlated with volatility)
        base_volume = 1000000
        volume = base_volume * (1 + vol[i] * 10) * np.random.uniform(0.5, 2.0)
        
        data.append({
            'timestamp': datetime.now() - timedelta(days=days-i),
            'open': round(open_price, 2),
            'high': round(high, 2),
            'low': round(low, 2),
            'close': round(close, 2),
            'volume': int(volume)
        })
    
    df = pd.DataFrame(data)
    df.set_index('timestamp', inplace=True)
    
    return df


def demonstrate_indicators():
    """Demonstrate advanced technical indicators"""
    print("\n" + "="*60)
    print("🔍 ADVANCED TECHNICAL INDICATORS DEMO")
    print("="*60)
    
    # Generate sample data for multiple timeframes
    daily_data = generate_sample_data(252)  # 1 year daily
    
    # Create simulated multi-timeframe data
    timeframe_data = {
        '5m': daily_data.iloc[-100:].copy(),   # Last 100 days as 5min data
        '15m': daily_data.iloc[-200:].copy(),  # Last 200 days as 15min data
        '1h': daily_data.iloc[-252:].copy(),   # Full year as hourly data
        '1d': daily_data.copy()                # Daily data
    }
    
    # Create indicators analyzer
    indicators = create_advanced_indicators({
        'rsi_period': 14,
        'macd_fast': 12,
        'macd_slow': 26,
        'bb_period': 20
    })
    
    # Analyze multi-timeframe indicators
    print("\n📊 Multi-Timeframe Analysis:")
    results = indicators.analyze_multi_timeframe(timeframe_data)
    
    for timeframe, indicator_list in results.items():
        print(f"\n⏱️  {timeframe.upper()} Timeframe:")
        
        # Show top 5 indicators
        for indicator in indicator_list[:5]:
            confidence_emoji = "🟢" if indicator.confidence > 0.7 else "🟡" if indicator.confidence > 0.4 else "🔴"
            signal_emoji = "📈" if indicator.signal == "BUY" else "📉" if indicator.signal == "SELL" else "➡️"
            
            print(f"  {confidence_emoji} {signal_emoji} {indicator.name}: {indicator.value:.3f} "
                  f"({indicator.signal}, {indicator.confidence:.2f})")
    
    # Calculate support/resistance levels
    print("\n🎯 Support/Resistance Levels:")
    sr_levels = indicators.calculate_support_resistance(daily_data, TimeFrame.D1)
    
    current_price = daily_data['close'].iloc[-1]
    print(f"Current Price: ${current_price:.2f}")
    
    for level in sr_levels[:5]:  # Show top 5 levels
        distance = abs(level.level - current_price) / current_price * 100
        level_emoji = "🔴" if level.level_type == "RESISTANCE" else "🟢"
        
        print(f"  {level_emoji} {level.level_type}: ${level.level:.2f} "
              f"(Strength: {level.strength:.2f}, Distance: {distance:.1f}%)")
    
    # Elliott Wave analysis
    print("\n🌊 Elliott Wave Analysis:")
    elliott_wave = indicators.analyze_elliott_wave(daily_data, TimeFrame.D1)
    
    if elliott_wave:
        print(f"  📈 Current Wave: {elliott_wave.current_wave}")
        print(f"  🎯 Wave Degree: {elliott_wave.wave_degree}")
        print(f"  📊 Completion: {elliott_wave.completion_percent:.1%}")
        print(f"  🎯 Next Target: ${elliott_wave.next_target:.2f}")
        print(f"  🎲 Confidence: {elliott_wave.confidence:.2f}")
    else:
        print("  ❌ No clear Elliott Wave pattern detected")


def demonstrate_pattern_recognition():
    """Demonstrate pattern recognition capabilities"""
    print("\n" + "="*60)
    print("🔍 PATTERN RECOGNITION DEMO")
    print("="*60)
    
    # Generate data with some artificial patterns
    df = generate_sample_data(100)
    
    # Create pattern recognition analyzer
    pattern_analyzer = create_pattern_recognition({
        'min_pattern_length': 5,
        'max_pattern_length': 20,
        'confidence_threshold': 0.6
    })
    
    # Recognize all patterns
    print("\n🔍 Comprehensive Pattern Analysis:")
    patterns = pattern_analyzer.recognize_all_patterns(df)
    
    # Group patterns by type
    pattern_groups = {}
    for pattern in patterns:
        pattern_type = pattern.pattern_type.value
        if pattern_type not in pattern_groups:
            pattern_groups[pattern_type] = []
        pattern_groups[pattern_type].append(pattern)
    
    for pattern_type, pattern_list in pattern_groups.items():
        print(f"\n📊 {pattern_type.upper()} Patterns:")
        
        for pattern in pattern_list[:3]:  # Show top 3 patterns per type
            signal_emoji = "📈" if pattern.signal.value in ['bullish'] else "📉" if pattern.signal.value in ['bearish'] else "↔️"
            confidence_emoji = "🟢" if pattern.confidence > 0.8 else "🟡" if pattern.confidence > 0.6 else "🔴"
            
            print(f"  {confidence_emoji} {signal_emoji} {pattern.pattern_name}")
            print(f"    Signal: {pattern.signal.value.title()}")
            print(f"    Confidence: {pattern.confidence:.2%}")
            print(f"    Duration: {pattern.end_idx - pattern.start_idx} periods")
            
            if pattern.target_price:
                print(f"    Target: ${pattern.target_price:.2f}")
            if pattern.stop_loss:
                print(f"    Stop Loss: ${pattern.stop_loss:.2f}")
            print(f"    Description: {pattern.description}")
            print()
    
    # Specific pattern type analysis
    print("🕯️ Candlestick Patterns:")
    candlestick_patterns = pattern_analyzer.recognize_candlestick_patterns(df)
    
    for pattern in candlestick_patterns[:5]:
        signal_emoji = "📈" if "bullish" in pattern.signal.value.lower() else "📉" if "bearish" in pattern.signal.value.lower() else "❓"
        print(f"  {signal_emoji} {pattern.pattern_name} (Confidence: {pattern.confidence:.2%})")
    
    print("\n📈 Chart Patterns:")
    chart_patterns = pattern_analyzer.recognize_chart_patterns(df)
    
    for pattern in chart_patterns[:3]:
        signal_emoji = "📈" if "bullish" in pattern.signal.value.lower() else "📉" if "bearish" in pattern.signal.value.lower() else "❓"
        print(f"  {signal_emoji} {pattern.pattern_name} (Confidence: {pattern.confidence:.2%})")
    
    print("\n🔷 Harmonic Patterns:")
    harmonic_patterns = pattern_analyzer.recognize_harmonic_patterns(df)
    
    if harmonic_patterns:
        for pattern in harmonic_patterns[:2]:
            signal_emoji = "📈" if "bullish" in pattern.signal.value.lower() else "📉" if "bearish" in pattern.signal.value.lower() else "❓"
            print(f"  {signal_emoji} {pattern.pattern_name} (Confidence: {pattern.confidence:.2%})")
    else:
        print("  ❌ No harmonic patterns detected")


def demonstrate_market_regime():
    """Demonstrate market regime detection"""
    print("\n" + "="*60)
    print("🔍 MARKET REGIME DETECTION DEMO")
    print("="*60)
    
    # Generate data with different market conditions
    df = generate_sample_data(252)
    
    # Create market regime detector
    regime_detector = create_market_regime_detector({
        'lookback_period': 200,
        'short_window': 20,
        'medium_window': 50,
        'long_window': 100
    })
    
    # Detect current market regime
    print("\n🎯 Current Market Regime:")
    current_regime = regime_detector.detect_market_regime(df)
    
    # Display regime information
    print(f"📊 Market State: {current_regime.market_state.value.replace('_', ' ').title()}")
    print(f"📈 Trend Strength: {current_regime.trend_strength:.2%}")
    print(f"➡️  Trend Direction: {current_regime.trend_direction:.2f}")
    
    print(f"\n💥 Volatility Regime: {current_regime.volatility_regime.value.replace('_', ' ').title()}")
    print(f"📊 Volatility Percentile: {current_regime.volatility_percentile:.1f}%")
    
    print(f"\n💧 Liquidity Regime: {current_regime.liquidity_regime.value.replace('_', ' ').title()}")
    
    print(f"\n⚡ Momentum Score: {current_regime.momentum_score:.2f}")
    print(f"⚠️  Risk Level: {current_regime.risk_level}")
    print(f"🎲 Confidence: {current_regime.confidence:.2%}")
    print(f"⏳ Expected Persistence: {current_regime.regime_persistence:.0f} days")
    
    # Show key metrics
    print(f"\n📋 Key Metrics:")
    for metric, value in current_regime.key_metrics.items():
        if isinstance(value, float):
            print(f"  📊 {metric.replace('_', ' ').title()}: {value:.3f}")
    
    # Calculate regime probabilities
    print(f"\n🎲 Regime Probabilities:")
    probabilities = regime_detector.calculate_regime_probabilities(df)
    
    for regime, prob in probabilities.items():
        if prob > 0.1:  # Only show significant probabilities
            print(f"  📊 {regime.replace('_', ' ').title()}: {prob:.1%}")
    
    # Get trading implications
    print(f"\n💼 Trading Implications:")
    implications = regime_detector.get_trading_implications(current_regime)
    
    strategy_emoji = "📈" if "trend" in implications['strategy_type'] else "🔄" if "mean" in implications['strategy_type'] else "⚖️"
    print(f"  {strategy_emoji} Strategy Type: {implications['strategy_type'].replace('_', ' ').title()}")
    
    position_emoji = "🔺" if implications['position_sizing'] == 'increased' else "🔻" if implications['position_sizing'] == 'reduced' else "◼️"
    print(f"  {position_emoji} Position Sizing: {implications['position_sizing'].title()}")
    
    print(f"  ⚖️  Leverage Recommendation: {implications['leverage_recommendation']:.1f}x")
    print(f"  ⏰ Preferred Timeframes: {', '.join(implications['preferred_timeframes'])}")
    print(f"  🛡️  Risk Management: {implications['risk_management'].title()}")
    
    if implications['hedging_recommendation']:
        print(f"  🛡️  Hedging: Recommended")
    
    # Analyze regime transitions
    print(f"\n🔄 Regime Transition Analysis:")
    
    # Simulate some historical regimes for transition analysis
    for i in range(3):
        past_data = df.iloc[:-50*(i+1)].copy()
        if len(past_data) > 100:
            past_regime = regime_detector.detect_market_regime(past_data)
    
    transitions = regime_detector.detect_regime_transitions(df, lookback_days=10)
    
    if transitions:
        for transition in transitions[:2]:  # Show recent transitions
            print(f"  🔄 Transition detected:")
            print(f"    From: {transition.from_regime.market_state.value} → To: {transition.to_regime.market_state.value}")
            print(f"    Probability: {transition.transition_probability:.2%}")
            print(f"    Catalysts: {', '.join(transition.catalyst_factors)}")
            print(f"    Expected Duration: {transition.expected_duration:.0f} days")
            print()
    else:
        print("  ✅ No recent regime transitions detected")


async def run_comprehensive_analysis():
    """Run comprehensive technical analysis demo"""
    print("🚀 ADVANCED TECHNICAL ANALYSIS SYSTEM")
    print("="*60)
    print("Demonstrating multi-timeframe indicators, pattern recognition, and market regime detection")
    
    try:
        # Run all demonstrations
        demonstrate_indicators()
        demonstrate_pattern_recognition() 
        demonstrate_market_regime()
        
        print("\n" + "="*60)
        print("✅ ANALYSIS COMPLETE")
        print("="*60)
        print("🎯 Summary:")
        print("• Advanced technical indicators with multi-timeframe analysis")
        print("• Comprehensive pattern recognition (30+ candlestick, chart, harmonic patterns)")
        print("• Market regime detection and classification")
        print("• Support/resistance levels with dynamic calculation")
        print("• Elliott Wave analysis with automatic detection")
        print("• Trading strategy implications based on market conditions")
        print("• Regime transition analysis and prediction")
        
    except Exception as e:
        logger.error(f"Error in comprehensive analysis: {e}")
        raise


if __name__ == "__main__":
    # Run the comprehensive demo
    asyncio.run(run_comprehensive_analysis())