#!/usr/bin/env python3
"""
System Integration Test
Tests the real-time market data collection and storage system
"""

import asyncio
import sys
from datetime import datetime

from infrastructure.data_collectors.market_data_collector import market_data_collector
from infrastructure.data_collectors.news_collector import initialize_news_collector
from infrastructure.databases.data_storage import initialize_data_storage, DatabaseConfig
from infrastructure.data_collectors.api_manager import api_manager
from utils.logger import get_logger

logger = get_logger(__name__)


async def test_api_connections():
    """Test API connections"""
    logger.info("Testing API connections...")
    
    try:
        # Connect to all exchanges
        connection_results = await api_manager.connect_all()
        
        for exchange, connected in connection_results.items():
            status = "✓ Connected" if connected else "✗ Failed"
            logger.info(f"  {exchange}: {status}")
        
        connected_count = sum(connection_results.values())
        total_count = len(connection_results)
        
        logger.info(f"API Connections: {connected_count}/{total_count} successful")
        return connected_count > 0
        
    except Exception as e:
        logger.error(f"API connection test failed: {e}")
        return False


async def test_database_connection():
    """Test database connection and schema"""
    logger.info("Testing database connection...")
    
    try:
        # Test with default config
        db_config = DatabaseConfig(
            host="localhost",
            database="trading_system",
            username="postgres",
            password="password"
        )
        
        storage = await initialize_data_storage(db_config)
        
        # Test basic query
        result = await storage.db_manager.execute_query("SELECT 1 as test")
        
        if result and result[0]['test'] == 1:
            logger.info("✓ Database connection successful")
            await storage.close()
            return True
        else:
            logger.error("✗ Database query failed")
            return False
            
    except Exception as e:
        logger.error(f"✗ Database connection failed: {e}")
        logger.info("Make sure PostgreSQL/TimescaleDB is running:")
        logger.info("  brew services start postgresql  # macOS")
        logger.info("  sudo systemctl start postgresql  # Linux")
        return False


async def test_market_data_collection():
    """Test market data collection"""
    logger.info("Testing market data collection...")
    
    try:
        # Test with a single symbol for short duration
        test_symbols = ['KRW-BTC']
        test_timeframes = ['1m']
        
        # Start collection briefly
        collection_task = asyncio.create_task(
            market_data_collector.start_collection(test_symbols, test_timeframes)
        )
        
        # Let it run for 10 seconds
        await asyncio.sleep(10)
        
        # Stop collection
        await market_data_collector.stop_collection()
        
        # Check stats
        stats = market_data_collector.get_collection_stats()
        
        if stats['data_counts']:
            logger.info("✓ Market data collection working")
            logger.info(f"  Collected data: {stats['data_counts']}")
            return True
        else:
            logger.warning("✗ No market data collected")
            return False
            
    except Exception as e:
        logger.error(f"✗ Market data collection test failed: {e}")
        return False


async def test_news_collection():
    """Test news collection"""
    logger.info("Testing news collection...")
    
    try:
        # Initialize with minimal config (no API keys needed for RSS)
        news_config = {}
        
        collector = await initialize_news_collector(news_config)
        
        # Try to collect some news (RSS feeds don't need API keys)
        articles = await collector.collect_all_news(hours_back=24)
        
        if articles:
            logger.info(f"✓ News collection working - collected {len(articles)} articles")
            
            # Show sample article
            if articles:
                sample = articles[0]
                logger.info(f"  Sample: '{sample.title[:50]}...' from {sample.source}")
            
            await collector.close()
            return True
        else:
            logger.warning("✗ No news articles collected")
            return False
            
    except Exception as e:
        logger.error(f"✗ News collection test failed: {e}")
        return False


async def test_data_storage():
    """Test data storage functionality"""
    logger.info("Testing data storage...")
    
    try:
        db_config = DatabaseConfig(
            host="localhost",
            database="trading_system",
            username="postgres",
            password="password"
        )
        
        storage = await initialize_data_storage(db_config)
        
        # Test symbol creation
        from infrastructure.data_collectors.models import ExchangeType
        symbol_id = await storage.get_or_create_symbol('TEST-SYMBOL', ExchangeType.UPBIT)
        
        if symbol_id:
            logger.info("✓ Data storage working")
            logger.info(f"  Created/retrieved symbol ID: {symbol_id}")
            
            await storage.close()
            return True
        else:
            logger.error("✗ Symbol creation failed")
            return False
            
    except Exception as e:
        logger.error(f"✗ Data storage test failed: {e}")
        return False


async def main():
    """Run system integration tests"""
    logger.info("=== AI Trading System Integration Test ===")
    logger.info(f"Test started: {datetime.now()}")
    
    tests = [
        ("API Connections", test_api_connections),
        ("Database Connection", test_database_connection),
        ("Data Storage", test_data_storage),
        ("Market Data Collection", test_market_data_collection),
        ("News Collection", test_news_collection),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n--- Testing {test_name} ---")
        try:
            results[test_name] = await test_func()
        except Exception as e:
            logger.error(f"Test {test_name} crashed: {e}")
            results[test_name] = False
    
    # Summary
    logger.info("\n=== Test Results Summary ===")
    passed = 0
    total = len(tests)
    
    for test_name, result in results.items():
        status = "PASS" if result else "FAIL"
        logger.info(f"  {test_name}: {status}")
        if result:
            passed += 1
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! System is ready for deployment.")
        return 0
    else:
        logger.warning("⚠️  Some tests failed. Check the logs above for details.")
        return 1


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Test suite crashed: {e}")
        sys.exit(1)