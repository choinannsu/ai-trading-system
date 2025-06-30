#!/usr/bin/env python3
"""
Data Collection System Runner
Starts the real-time market data collection and news gathering system
"""

import asyncio
import signal
import sys
from datetime import datetime
from typing import List

from infrastructure.data_collectors.market_data_collector import market_data_collector
from infrastructure.data_collectors.news_collector import initialize_news_collector
from infrastructure.databases.data_storage import initialize_data_storage, DatabaseConfig
from utils.config import get_config
from utils.logger import get_logger

logger = get_logger(__name__)


class DataCollectionSystem:
    """Main data collection system orchestrator"""
    
    def __init__(self):
        self.config = get_config()
        self.is_running = False
        self.tasks = []
        
        # Initialize components
        self.storage = None
        self.news_collector = None
        
    async def initialize(self):
        """Initialize all system components"""
        logger.info("Initializing data collection system...")
        
        try:
            # Initialize database
            db_config = DatabaseConfig(
                host=getattr(self.config.database, 'host', 'localhost'),
                database=getattr(self.config.database, 'name', 'trading_system'),
                username=getattr(self.config.database, 'username', 'postgres'),
                password=getattr(self.config.database, 'password', 'password')
            )
            
            self.storage = await initialize_data_storage(db_config)
            logger.info("Database initialized successfully")
            
            # Initialize news collector
            news_config = {
                'newsapi_key': getattr(self.config.news, 'newsapi_key', None),
                'reddit_client_id': getattr(self.config.news, 'reddit_client_id', None),
                'reddit_client_secret': getattr(self.config.news, 'reddit_client_secret', None),
                'reddit_user_agent': getattr(self.config.news, 'reddit_user_agent', 'TradingBot/1.0')
            }
            
            self.news_collector = await initialize_news_collector(news_config)
            logger.info("News collector initialized successfully")
            
            return True
            
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            return False
    
    async def start_market_data_collection(self):
        """Start market data collection"""
        try:
            # Define symbols to collect
            crypto_symbols = ['KRW-BTC', 'KRW-ETH', 'KRW-XRP', 'KRW-ADA', 'KRW-SOL']
            stock_symbols = ['005930', '000660', '035420', 'AAPL', 'GOOGL', 'TSLA']
            
            all_symbols = crypto_symbols + stock_symbols
            timeframes = ['1m', '5m', '15m', '1h', '1d']
            
            logger.info(f"Starting market data collection for {len(all_symbols)} symbols")
            
            # Start collection
            task = asyncio.create_task(
                market_data_collector.start_collection(all_symbols, timeframes)
            )
            self.tasks.append(task)
            
            return task
            
        except Exception as e:
            logger.error(f"Failed to start market data collection: {e}")
            return None
    
    async def start_news_collection(self):
        """Start news collection with periodic updates"""
        async def news_collection_loop():
            while self.is_running:
                try:
                    logger.info("Starting news collection cycle...")
                    
                    # Collect news
                    articles = await self.news_collector.collect_all_news(hours_back=6)
                    
                    if articles and self.storage:
                        # Store in database
                        await self.storage.store_news_articles(articles)
                        logger.info(f"Stored {len(articles)} news articles")
                    
                    # Wait 30 minutes before next collection
                    await asyncio.sleep(1800)
                    
                except Exception as e:
                    logger.error(f"News collection error: {e}")
                    await asyncio.sleep(300)  # Wait 5 minutes on error
        
        task = asyncio.create_task(news_collection_loop())
        self.tasks.append(task)
        return task
    
    async def start_data_storage_monitoring(self):
        """Monitor data storage and system health"""
        async def storage_monitoring_loop():
            while self.is_running:
                try:
                    # Get collection stats
                    stats = market_data_collector.get_collection_stats()
                    logger.info(f"Collection stats: {stats['data_counts']}")
                    
                    # Check database connection
                    if self.storage and self.storage.db_manager.is_connected:
                        logger.debug("Database connection healthy")
                    else:
                        logger.warning("Database connection issues detected")
                    
                    # Wait 5 minutes between checks
                    await asyncio.sleep(300)
                    
                except Exception as e:
                    logger.error(f"Storage monitoring error: {e}")
                    await asyncio.sleep(60)
        
        task = asyncio.create_task(storage_monitoring_loop())
        self.tasks.append(task)
        return task
    
    async def start(self):
        """Start the complete data collection system"""
        if not await self.initialize():
            logger.error("System initialization failed")
            return False
        
        self.is_running = True
        logger.info("Starting data collection system...")
        
        try:
            # Start all collection tasks
            await self.start_market_data_collection()
            await self.start_news_collection()
            await self.start_data_storage_monitoring()
            
            logger.info("All collection tasks started successfully")
            
            # Wait for all tasks
            await asyncio.gather(*self.tasks, return_exceptions=True)
            
        except Exception as e:
            logger.error(f"System error: {e}")
            return False
        
        return True
    
    async def stop(self):
        """Stop the data collection system"""
        logger.info("Stopping data collection system...")
        self.is_running = False
        
        # Stop market data collector
        await market_data_collector.stop_collection()
        
        # Cancel all tasks
        for task in self.tasks:
            if not task.done():
                task.cancel()
        
        # Close connections
        if self.news_collector:
            await self.news_collector.close()
        
        if self.storage:
            await self.storage.close()
        
        logger.info("Data collection system stopped")


async def main():
    """Main entry point"""
    system = DataCollectionSystem()
    
    # Setup signal handlers for graceful shutdown
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, initiating shutdown...")
        asyncio.create_task(system.stop())
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        logger.info("=== AI Trading System Data Collection Started ===")
        logger.info(f"Start time: {datetime.now()}")
        
        await system.start()
        
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    except Exception as e:
        logger.error(f"System error: {e}")
    finally:
        await system.stop()
        logger.info("=== Data Collection System Shutdown Complete ===")


if __name__ == "__main__":
    asyncio.run(main())