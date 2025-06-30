#!/usr/bin/env python3
"""
Sentiment Analysis System Example
Demonstrates news analysis, social sentiment, and event detection
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import asyncio

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.sentiment import (
    create_news_analyzer,
    create_social_analyzer,
    create_event_detector,
    FinancialNewsConfig,
    SocialConfig,
    EventConfig
)
from models.sentiment.news_analyzer import generate_sample_news
from models.sentiment.social_sentiment import generate_sample_social_posts
from models.sentiment.event_detector import generate_sample_news_events
from utils.logger import get_logger

logger = get_logger(__name__)


def demonstrate_news_analysis():
    """Demonstrate financial news sentiment analysis"""
    print("\n" + "="*60)
    print("📰 FINANCIAL NEWS SENTIMENT ANALYSIS DEMO")
    print("="*60)
    
    # Create news analyzer
    news_config = {
        'sentiment_threshold': 0.1,
        'impact_weight': 0.7,
        'time_decay_hours': 24,
        'use_korean_model': True
    }
    
    analyzer = create_news_analyzer(news_config)
    
    # Generate sample news
    sample_news = generate_sample_news(15)
    print(f"📊 Analyzing {len(sample_news)} news items...")
    
    try:
        # Analyze news batch
        analysis_results = analyzer.analyze_news_batch(sample_news)
        
        print(f"✅ Successfully analyzed {len(analysis_results)} news items")
        
        # Show individual analysis results
        print(f"\n🔍 Sample Analysis Results:")
        for i, result in enumerate(analysis_results[:5]):  # Show first 5
            sentiment_emoji = "📈" if result.sentiment.compound > 0.1 else "📉" if result.sentiment.compound < -0.1 else "➡️"
            confidence_emoji = "🟢" if result.sentiment.confidence > 0.7 else "🟡" if result.sentiment.confidence > 0.4 else "🔴"
            
            print(f"\n  {i+1}. {sentiment_emoji} {confidence_emoji} {result.news_item.title[:60]}...")
            print(f"     Source: {result.news_item.source} | Language: {result.news_item.language}")
            print(f"     Sentiment: {result.sentiment.label.title()} ({result.sentiment.compound:.3f})")
            print(f"     Confidence: {result.sentiment.confidence:.2f} | Impact: {result.impact_score:.3f}")
            
            if result.financial_keywords:
                print(f"     Keywords: {', '.join(result.financial_keywords[:5])}")
            
            if result.key_entities:
                print(f"     Entities: {', '.join(result.key_entities[:3])}")
        
        # Create sentiment time series
        print(f"\n📊 Creating Sentiment Time Series...")
        timeseries_df = analyzer.create_sentiment_timeseries(analysis_results, '1H')
        
        if not timeseries_df.empty:
            print(f"✅ Time series created with {len(timeseries_df)} data points")
            print(f"📈 Latest sentiment: {timeseries_df['weighted_sentiment'].iloc[-1]:.3f}")
            print(f"📊 Average impact score: {timeseries_df['impact_score'].mean():.3f}")
        
        # Get market sentiment summary
        print(f"\n📋 Market Sentiment Summary:")
        summary = analyzer.get_market_sentiment_summary(analysis_results)
        
        overall_emoji = "📈" if summary['overall_sentiment'] > 0.1 else "📉" if summary['overall_sentiment'] < -0.1 else "➡️"
        print(f"  {overall_emoji} Overall Sentiment: {summary['sentiment_label'].title()} ({summary['overall_sentiment']:.3f})")
        print(f"  🎯 Confidence: {summary['confidence']:.2f}")
        print(f"  📰 News Count: {summary['news_count']}")
        print(f"  📊 Average Impact: {summary['average_impact']:.3f}")
        
        print(f"\n  📊 Sentiment Distribution:")
        for sentiment, count in summary['sentiment_distribution'].items():
            percentage = count / summary['news_count'] * 100 if summary['news_count'] > 0 else 0
            print(f"    {sentiment.title()}: {count} ({percentage:.1f}%)")
        
        if summary['top_topics']:
            print(f"  🔥 Top Topics: {', '.join(summary['top_topics'][:5])}")
        
        if summary['top_entities']:
            print(f"  🏢 Top Entities: {', '.join(summary['top_entities'][:5])}")
        
    except Exception as e:
        logger.error(f"News analysis demo failed: {e}")
        print(f"❌ News analysis demo failed: {e}")


def demonstrate_social_sentiment():
    """Demonstrate social media sentiment analysis"""
    print("\n" + "="*60)
    print("🐦 SOCIAL MEDIA SENTIMENT ANALYSIS DEMO")
    print("="*60)
    
    # Create social sentiment analyzer
    social_config = {
        'platforms': ['reddit', 'twitter'],
        'min_followers': 10,
        'enable_bot_filtering': True,
        'bot_detection_threshold': 0.7,
        'community_weights': {
            'wallstreetbets': 1.5,
            'investing': 1.2,
            'stocks': 1.0,
            'default': 0.5
        }
    }
    
    analyzer = create_social_analyzer(social_config)
    
    # Generate sample social posts
    sample_posts = generate_sample_social_posts(25)
    print(f"📊 Analyzing {len(sample_posts)} social media posts...")
    
    try:
        # Analyze posts
        sentiment_index = analyzer.analyze_posts(sample_posts)
        
        print(f"✅ Successfully analyzed social sentiment")
        
        # Display sentiment index
        print(f"\n📊 Real-time Sentiment Index:")
        overall_emoji = "📈" if sentiment_index.overall_sentiment > 0.1 else "📉" if sentiment_index.overall_sentiment < -0.1 else "➡️"
        print(f"  {overall_emoji} Overall Sentiment: {sentiment_index.overall_sentiment:.3f}")
        print(f"  📊 Volume Weighted: {sentiment_index.volume_weighted_sentiment:.3f}")
        print(f"  🎯 Confidence: {sentiment_index.confidence_score:.2f}")
        print(f"  📱 Total Posts: {sentiment_index.total_posts}")
        print(f"  🤖 Bot Ratio: {sentiment_index.bot_ratio:.2%}")
        
        # Sentiment distribution
        print(f"\n  📊 Sentiment Distribution:")
        total_classified = sum(sentiment_index.sentiment_distribution.values())
        for sentiment, count in sentiment_index.sentiment_distribution.items():
            percentage = count / total_classified * 100 if total_classified > 0 else 0
            emoji = "😊" if sentiment == 'positive' else "😢" if sentiment == 'negative' else "😐"
            print(f"    {emoji} {sentiment.title()}: {count} ({percentage:.1f}%)")
        
        # Community breakdown
        if sentiment_index.community_breakdown:
            print(f"\n  🏘️  Community Sentiment:")
            for community, sentiment in sentiment_index.community_breakdown.items():
                community_emoji = "📈" if sentiment > 0.1 else "📉" if sentiment < -0.1 else "➡️"
                print(f"    {community_emoji} r/{community}: {sentiment:.3f}")
        
        # Platform breakdown
        if sentiment_index.platform_breakdown:
            print(f"\n  📱 Platform Sentiment:")
            for platform, sentiment in sentiment_index.platform_breakdown.items():
                platform_emoji = "📈" if sentiment > 0.1 else "📉" if sentiment < -0.1 else "➡️"
                print(f"    {platform_emoji} {platform.title()}: {sentiment:.3f}")
        
        # Top tickers
        if sentiment_index.top_tickers:
            print(f"\n  📈 Top Mentioned Tickers:")
            for ticker, count in sentiment_index.top_tickers[:5]:
                print(f"    💼 ${ticker}: {count} mentions")
        
        # Trending topics
        if sentiment_index.trending_topics:
            print(f"\n  🔥 Trending Topics: {', '.join(sentiment_index.trending_topics[:5])}")
        
        # Create time series (simulate multiple data points)
        print(f"\n📊 Creating Social Sentiment Time Series...")
        multiple_indices = [sentiment_index]
        
        # Generate additional synthetic data points
        for i in range(1, 6):
            synthetic_posts = generate_sample_social_posts(20)
            synthetic_index = analyzer.analyze_posts(synthetic_posts)
            synthetic_index.timestamp = datetime.now() - timedelta(hours=i)
            multiple_indices.append(synthetic_index)
        
        timeseries_df = analyzer.create_sentiment_timeseries(multiple_indices)
        
        if not timeseries_df.empty:
            print(f"✅ Social sentiment time series created with {len(timeseries_df)} data points")
            print(f"📈 Latest sentiment: {timeseries_df['overall_sentiment'].iloc[-1]:.3f}")
            print(f"📊 Sentiment volatility: {timeseries_df['overall_sentiment'].std():.3f}")
        
    except Exception as e:
        logger.error(f"Social sentiment demo failed: {e}")
        print(f"❌ Social sentiment demo failed: {e}")


def demonstrate_event_detection():
    """Demonstrate event detection and impact prediction"""
    print("\n" + "="*60)
    print("⚡ EVENT DETECTION & IMPACT PREDICTION DEMO")
    print("="*60)
    
    # Create event detector
    event_config = {
        'min_news_threshold': 2,
        'time_window_hours': 6,
        'similarity_threshold': 0.3,
        'min_confidence': 0.6,
        'max_events_per_day': 20
    }
    
    detector = create_event_detector(event_config)
    
    # Generate sample news with events
    sample_news = generate_sample_news_events(20)
    print(f"📊 Analyzing {len(sample_news)} news items for events...")
    
    try:
        # Detect and analyze events
        event_result = detector.detect_and_analyze(sample_news)
        
        print(f"✅ Event detection completed")
        print(f"⚡ Detected {len(event_result.detected_events)} events")
        print(f"📊 Market stress level: {event_result.market_stress_level:.2f}")
        
        # Show detected events
        if event_result.detected_events:
            print(f"\n🔍 Detected Events:")
            for i, event in enumerate(event_result.detected_events[:5]):  # Show first 5
                severity_emoji = "🔴" if event.confidence > 0.8 else "🟡" if event.confidence > 0.6 else "🟢"
                event_emoji = "📈" if event.sentiment_shift > 0.1 else "📉" if event.sentiment_shift < -0.1 else "⚡"
                
                print(f"\n  {i+1}. {severity_emoji} {event_emoji} {event.description}")
                print(f"     Type: {event.event_type.value.title()} | Confidence: {event.confidence:.2f}")
                print(f"     Sentiment Shift: {event.sentiment_shift:+.3f} | News Count: {event.news_count}")
                print(f"     Time: {event.timestamp.strftime('%Y-%m-%d %H:%M')}")
                
                if event.affected_entities:
                    print(f"     Affected: {', '.join(event.affected_entities[:3])}")
                
                if event.keywords:
                    print(f"     Keywords: {', '.join(event.keywords[:5])}")
        
        # Show impact predictions
        if event_result.impact_predictions:
            print(f"\n🎯 Impact Predictions:")
            for i, prediction in enumerate(event_result.impact_predictions[:3]):  # Show first 3
                risk_emoji = "🔴" if prediction.risk_level == "High" else "🟡" if prediction.risk_level == "Medium" else "🟢"
                impact_emoji = "📈" if prediction.predicted_price_impact > 0 else "📉" if prediction.predicted_price_impact < 0 else "➡️"
                
                print(f"\n  {i+1}. {risk_emoji} {impact_emoji} {prediction.event_signal.description}")
                print(f"     Risk Level: {prediction.risk_level}")
                print(f"     Price Impact: {prediction.predicted_price_impact:+.2%}")
                print(f"     Volatility: {prediction.predicted_volatility:.2%}")
                print(f"     Volume Ratio: {prediction.predicted_volume_ratio:.1f}x")
                print(f"     Confidence: {prediction.confidence:.2f}")
                print(f"     Time Horizon: {prediction.time_horizon_hours} hours")
                
                if prediction.similar_historical_events:
                    print(f"     Similar Events: {len(prediction.similar_historical_events)} found")
                
                if prediction.recommended_actions:
                    print(f"     Recommendations:")
                    for action in prediction.recommended_actions[:3]:
                        print(f"       • {action}")
        
        # Show event clusters
        if len(event_result.event_clusters) > 1:
            print(f"\n🔗 Event Clusters:")
            for cluster_name, cluster_events in event_result.event_clusters.items():
                if len(cluster_events) > 1:
                    print(f"  📊 {cluster_name}: {len(cluster_events)} related events")
                    for event in cluster_events[:2]:  # Show first 2 in cluster
                        print(f"    • {event.description}")
        
        # Show trending themes
        if event_result.trending_themes:
            print(f"\n🔥 Trending Themes:")
            theme_str = ", ".join(event_result.trending_themes[:8])
            print(f"  {theme_str}")
        
        # Market stress analysis
        stress_level = event_result.market_stress_level
        if stress_level > 0.7:
            stress_emoji = "🔴"
            stress_desc = "High Stress"
        elif stress_level > 0.4:
            stress_emoji = "🟡"
            stress_desc = "Medium Stress"
        else:
            stress_emoji = "🟢"
            stress_desc = "Low Stress"
        
        print(f"\n📊 Market Stress Analysis:")
        print(f"  {stress_emoji} Stress Level: {stress_desc} ({stress_level:.2f})")
        
        if stress_level > 0.5:
            print(f"  ⚠️  Recommendations:")
            print(f"    • Increase risk monitoring")
            print(f"    • Consider defensive positioning")
            print(f"    • Watch for volatility spikes")
        
    except Exception as e:
        logger.error(f"Event detection demo failed: {e}")
        print(f"❌ Event detection demo failed: {e}")


def demonstrate_integrated_analysis():
    """Demonstrate integrated sentiment analysis"""
    print("\n" + "="*60)
    print("🔄 INTEGRATED SENTIMENT ANALYSIS DEMO")
    print("="*60)
    
    print("📊 Running integrated analysis combining news, social, and events...")
    
    try:
        # Create all analyzers
        news_analyzer = create_news_analyzer()
        social_analyzer = create_social_analyzer()
        event_detector = create_event_detector()
        
        # Generate data
        news_data = generate_sample_news(10)
        social_posts = generate_sample_social_posts(15)
        news_events = generate_sample_news_events(8)
        
        # Analyze news sentiment
        news_results = news_analyzer.analyze_news_batch(news_data)
        news_summary = news_analyzer.get_market_sentiment_summary(news_results)
        
        # Analyze social sentiment
        social_index = social_analyzer.analyze_posts(social_posts)
        
        # Detect events
        event_result = event_detector.detect_and_analyze(news_events)
        
        # Combine insights
        print(f"\n📊 Integrated Market Sentiment Dashboard:")
        print(f"  📰 News Sentiment: {news_summary['overall_sentiment']:+.3f} ({news_summary['news_count']} articles)")
        print(f"  🐦 Social Sentiment: {social_index.overall_sentiment:+.3f} ({social_index.total_posts} posts)")
        print(f"  ⚡ Event Impact: {len(event_result.detected_events)} events detected")
        print(f"  📊 Market Stress: {event_result.market_stress_level:.2f}")
        
        # Calculate combined sentiment score
        news_weight = 0.4
        social_weight = 0.3
        event_weight = 0.3
        
        combined_sentiment = (
            news_summary['overall_sentiment'] * news_weight +
            social_index.overall_sentiment * social_weight +
            (-event_result.market_stress_level) * event_weight  # Events add stress (negative)
        )
        
        combined_emoji = "📈" if combined_sentiment > 0.1 else "📉" if combined_sentiment < -0.1 else "➡️"
        print(f"\n🎯 Combined Market Sentiment: {combined_emoji} {combined_sentiment:+.3f}")
        
        # Risk assessment
        total_risk_factors = 0
        risk_factors = []
        
        if abs(news_summary['overall_sentiment']) > 0.3:
            risk_factors.append("High news sentiment volatility")
            total_risk_factors += 1
        
        if social_index.bot_ratio > 0.3:
            risk_factors.append("High bot activity in social media")
            total_risk_factors += 1
        
        if event_result.market_stress_level > 0.5:
            risk_factors.append("Elevated market stress from events")
            total_risk_factors += 1
        
        if len([p for p in event_result.impact_predictions if p.risk_level == "High"]) > 0:
            risk_factors.append("High-risk events detected")
            total_risk_factors += 1
        
        print(f"\n⚠️  Risk Assessment:")
        if total_risk_factors == 0:
            print(f"  🟢 Low Risk: Market sentiment appears stable")
        elif total_risk_factors <= 2:
            print(f"  🟡 Medium Risk: {total_risk_factors} risk factors identified")
        else:
            print(f"  🔴 High Risk: {total_risk_factors} risk factors identified")
        
        if risk_factors:
            print(f"  Risk Factors:")
            for factor in risk_factors:
                print(f"    • {factor}")
        
        # Trading implications
        print(f"\n💼 Trading Implications:")
        if combined_sentiment > 0.2:
            print(f"  📈 Bullish bias - Consider long positions")
            print(f"  🎯 Look for breakout opportunities")
        elif combined_sentiment < -0.2:
            print(f"  📉 Bearish bias - Consider defensive measures")
            print(f"  🛡️  Hedge existing positions")
        else:
            print(f"  ➡️  Neutral sentiment - Range-bound trading")
            print(f"  ⚖️  Balanced portfolio approach")
        
        if event_result.market_stress_level > 0.6:
            print(f"  ⚠️  High stress - Reduce position sizes")
            print(f"  📊 Increase monitoring frequency")
        
    except Exception as e:
        logger.error(f"Integrated analysis demo failed: {e}")
        print(f"❌ Integrated analysis demo failed: {e}")


async def run_comprehensive_sentiment_demo():
    """Run comprehensive sentiment analysis demo"""
    print("🚀 SENTIMENT ANALYSIS SYSTEM")
    print("="*60)
    print("Demonstrating financial news, social media, and event detection")
    
    try:
        # Run all demonstrations
        demonstrate_news_analysis()
        demonstrate_social_sentiment()
        demonstrate_event_detection()
        demonstrate_integrated_analysis()
        
        print("\n" + "="*60)
        print("✅ SENTIMENT ANALYSIS DEMO COMPLETE")
        print("="*60)
        print("🎯 Summary:")
        print("• FinBERT-based financial news sentiment analysis")
        print("• KoBERT support for Korean language news")
        print("• Social media sentiment with bot detection")
        print("• Reddit/Twitter data collection and analysis")
        print("• Automatic event detection and impact prediction")
        print("• Historical event pattern matching")
        print("• Market stress level calculation")
        print("• Integrated sentiment dashboard")
        print("• Real-time sentiment indices and time series")
        
    except Exception as e:
        logger.error(f"Error in comprehensive sentiment demo: {e}")
        raise


if __name__ == "__main__":
    # Run the comprehensive demo
    asyncio.run(run_comprehensive_sentiment_demo())