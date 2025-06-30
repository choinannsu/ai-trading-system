"""
Celery tasks for automated data collection and processing
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any
import logging

from celery import Task
from .celery_app import app
from ..data_collectors.market_data_collector import market_data_collector
from ..data_collectors.news_collector import NewsCollector, initialize_news_collector
from ..databases.data_storage import initialize_data_storage, DatabaseConfig
from ..data_collectors.api_manager import api_manager
from utils.logger import get_logger
from utils.config import get_config

logger = get_logger(__name__)


class AsyncTask(Task):
    """Base class for async tasks"""
    
    def run_async(self, coro):
        """Run async coroutine in task"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()


# Market Data Collection Tasks
@app.task(base=AsyncTask, bind=True, max_retries=3)
def collect_crypto_data(self, symbols: List[str] = None, timeframes: List[str] = None):
    """Collect cryptocurrency market data"""
    if symbols is None:
        symbols = ['KRW-BTC', 'KRW-ETH', 'KRW-XRP', 'KRW-ADA', 'KRW-SOL', 'KRW-DOGE']
    
    if timeframes is None:
        timeframes = ['1m', '5m', '15m', '1h', '1d']
    
    async def _collect():
        try:
            logger.info(f"Starting crypto data collection for {len(symbols)} symbols")
            
            # Initialize components
            config = get_config()
            db_config = DatabaseConfig(
                host=config.database.host,
                database=config.database.name,
                username=config.database.username,
                password=config.database.password
            )
            
            storage = await initialize_data_storage(db_config)
            
            # Collect data for each symbol and timeframe
            for symbol in symbols:
                for timeframe in timeframes:
                    try:
                        # Get market data from Upbit
                        market_data = await api_manager.get_market_data(
                            symbol=symbol,
                            timeframe=timeframe,
                            limit=100
                        )
                        
                        if market_data:
                            await storage.store_market_data(market_data)
                            logger.info(f"Stored {len(market_data)} records for {symbol} {timeframe}")
                        
                        # Small delay to avoid rate limiting
                        await asyncio.sleep(0.5)
                        
                    except Exception as e:
                        logger.error(f"Failed to collect {symbol} {timeframe}: {e}")
                        continue
            
            await storage.close()
            return {"status": "success", "symbols": len(symbols), "timeframes": len(timeframes)}
            
        except Exception as e:
            logger.error(f"Crypto data collection failed: {e}")
            raise self.retry(countdown=60)
    
    return self.run_async(_collect())


@app.task(base=AsyncTask, bind=True, max_retries=3)
def collect_korean_stocks(self, symbols: List[str] = None):
    """Collect Korean stock market data"""
    if symbols is None:
        # Top Korean stocks
        symbols = ['005930', '000660', '035420', '051910', '006400', '035720']
    
    async def _collect():
        try:
            logger.info(f"Starting Korean stock data collection for {len(symbols)} symbols")
            
            config = get_config()
            db_config = DatabaseConfig(
                host=config.database.host,
                database=config.database.name,
                username=config.database.username,
                password=config.database.password
            )
            
            storage = await initialize_data_storage(db_config)
            
            # Check if market is open (9:00-15:30 KST, Mon-Fri)
            now = datetime.now()
            is_trading_hours = (
                now.weekday() < 5 and  # Monday to Friday
                9 <= now.hour < 15 or (now.hour == 15 and now.minute <= 30)
            )
            
            timeframes = ['5m', '15m', '1h'] if is_trading_hours else ['1d']
            
            for symbol in symbols:
                for timeframe in timeframes:
                    try:
                        market_data = await api_manager.get_market_data(
                            symbol=symbol,
                            timeframe=timeframe,
                            limit=50
                        )
                        
                        if market_data:
                            await storage.store_market_data(market_data)
                            logger.info(f"Stored Korean stock data for {symbol} {timeframe}")
                        
                        await asyncio.sleep(1)  # Respectful rate limiting
                        
                    except Exception as e:
                        logger.error(f"Failed to collect Korean stock {symbol}: {e}")
                        continue
            
            await storage.close()
            return {"status": "success", "symbols": len(symbols), "trading_hours": is_trading_hours}
            
        except Exception as e:
            logger.error(f"Korean stock collection failed: {e}")
            raise self.retry(countdown=120)
    
    return self.run_async(_collect())


@app.task(base=AsyncTask, bind=True, max_retries=3)
def collect_us_stocks(self, symbols: List[str] = None):
    """Collect US stock market data"""
    if symbols is None:
        symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN', 'NVDA']
    
    async def _collect():
        try:
            logger.info(f"Starting US stock data collection for {len(symbols)} symbols")
            
            config = get_config()
            db_config = DatabaseConfig(
                host=config.database.host,
                database=config.database.name,
                username=config.database.username,
                password=config.database.password
            )
            
            storage = await initialize_data_storage(db_config)
            
            for symbol in symbols:
                try:
                    market_data = await api_manager.get_market_data(
                        symbol=symbol,
                        timeframe='1h',
                        limit=24  # Last 24 hours
                    )
                    
                    if market_data:
                        await storage.store_market_data(market_data)
                        logger.info(f"Stored US stock data for {symbol}")
                    
                    await asyncio.sleep(1)
                    
                except Exception as e:
                    logger.error(f"Failed to collect US stock {symbol}: {e}")
                    continue
            
            await storage.close()
            return {"status": "success", "symbols": len(symbols)}
            
        except Exception as e:
            logger.error(f"US stock collection failed: {e}")
            raise self.retry(countdown=120)
    
    return self.run_async(_collect())


# News Collection Tasks
@app.task(base=AsyncTask, bind=True, max_retries=3)
def collect_news_data(self):
    """Collect financial news from multiple sources"""
    
    async def _collect():
        try:
            logger.info("Starting news data collection")
            
            config = get_config()
            
            # Initialize news collector
            news_config = {
                'newsapi_key': getattr(config.news, 'newsapi_key', None),
                'reddit_client_id': getattr(config.news, 'reddit_client_id', None),
                'reddit_client_secret': getattr(config.news, 'reddit_client_secret', None),
                'reddit_user_agent': getattr(config.news, 'reddit_user_agent', 'TradingBot/1.0')
            }
            
            collector = await initialize_news_collector(news_config)
            
            # Collect news
            articles = await collector.collect_all_news(hours_back=6)
            
            if articles:
                # Initialize storage
                db_config = DatabaseConfig(
                    host=config.database.host,
                    database=config.database.name,
                    username=config.database.username,
                    password=config.database.password
                )
                
                storage = await initialize_data_storage(db_config)
                await storage.store_news_articles(articles)
                await storage.close()
                
                logger.info(f"Stored {len(articles)} news articles")
            
            await collector.close()
            return {"status": "success", "articles_collected": len(articles)}
            
        except Exception as e:
            logger.error(f"News collection failed: {e}")
            raise self.retry(countdown=300)  # 5 minute delay
    
    return self.run_async(_collect())


@app.task(base=AsyncTask, bind=True, max_retries=3)
def collect_korean_news(self):
    """Collect Korean financial news"""
    
    async def _collect():
        try:
            logger.info("Starting Korean news collection")
            
            config = get_config()
            collector = await initialize_news_collector({})
            
            # Get Korean news only
            if hasattr(collector, 'collectors') and 'korean' in collector.collectors:
                articles = await collector.collectors['korean'].collect_all_categories()
                
                if articles:
                    db_config = DatabaseConfig(
                        host=config.database.host,
                        database=config.database.name,
                        username=config.database.username,
                        password=config.database.password
                    )
                    
                    storage = await initialize_data_storage(db_config)
                    await storage.store_news_articles(articles)
                    await storage.close()
                    
                    logger.info(f"Stored {len(articles)} Korean news articles")
                
                return {"status": "success", "articles_collected": len(articles)}
            
            return {"status": "skipped", "reason": "Korean collector not available"}
            
        except Exception as e:
            logger.error(f"Korean news collection failed: {e}")
            raise self.retry(countdown=600)  # 10 minute delay
    
    return self.run_async(_collect())


# Analytics Tasks
@app.task(base=AsyncTask, bind=True)
def calculate_technical_indicators(self):
    """Calculate technical indicators for all symbols"""
    
    async def _calculate():
        try:
            logger.info("Starting technical indicators calculation")
            
            # This would integrate with technical analysis models
            # For now, just log the task execution
            
            return {"status": "success", "indicators_calculated": 0}
            
        except Exception as e:
            logger.error(f"Technical indicators calculation failed: {e}")
            return {"status": "error", "error": str(e)}
    
    return self.run_async(_calculate())


@app.task(base=AsyncTask, bind=True)
def update_volume_profiles(self):
    """Update volume profiles for major symbols"""
    
    async def _update():
        try:
            logger.info("Starting volume profile updates")
            
            # Get collected trade data and calculate volume profiles
            symbols = ['KRW-BTC', 'KRW-ETH', '005930', 'AAPL']
            
            for symbol in symbols:
                try:
                    # This would calculate volume profiles from stored trade data
                    logger.debug(f"Updating volume profile for {symbol}")
                    
                except Exception as e:
                    logger.error(f"Volume profile update failed for {symbol}: {e}")
                    continue
            
            return {"status": "success", "symbols_updated": len(symbols)}
            
        except Exception as e:
            logger.error(f"Volume profile update failed: {e}")
            return {"status": "error", "error": str(e)}
    
    return self.run_async(_update())


@app.task(base=AsyncTask, bind=True)
def analyze_news_sentiment(self):
    """Analyze sentiment of collected news articles"""
    
    async def _analyze():
        try:
            logger.info("Starting news sentiment analysis")
            
            # This would integrate with sentiment analysis models
            # Process unprocessed articles and update sentiment scores
            
            return {"status": "success", "articles_analyzed": 0}
            
        except Exception as e:
            logger.error(f"News sentiment analysis failed: {e}")
            return {"status": "error", "error": str(e)}
    
    return self.run_async(_analyze())


# Daily Tasks
@app.task(base=AsyncTask, bind=True)
def create_portfolio_snapshot(self):
    """Create daily portfolio snapshot"""
    
    async def _snapshot():
        try:
            logger.info("Creating portfolio snapshot")
            
            # Get portfolio data from all exchanges
            portfolio_data = await api_manager.get_unified_portfolio()
            
            # Store snapshot in database
            config = get_config()
            db_config = DatabaseConfig(
                host=config.database.host,
                database=config.database.name,
                username=config.database.username,
                password=config.database.password
            )
            
            storage = await initialize_data_storage(db_config)
            
            # This would store the portfolio snapshot
            # await storage.store_portfolio_snapshot(portfolio_data)
            
            await storage.close()
            
            return {"status": "success", "snapshot_time": datetime.now().isoformat()}
            
        except Exception as e:
            logger.error(f"Portfolio snapshot failed: {e}")
            return {"status": "error", "error": str(e)}
    
    return self.run_async(_snapshot())


@app.task(base=AsyncTask, bind=True)
def generate_performance_report(self):
    """Generate daily performance report"""
    
    async def _report():
        try:
            logger.info("Generating performance report")
            
            # Calculate daily performance metrics
            # This would analyze portfolio performance, P&L, etc.
            
            return {"status": "success", "report_generated": datetime.now().isoformat()}
            
        except Exception as e:
            logger.error(f"Performance report generation failed: {e}")
            return {"status": "error", "error": str(e)}
    
    return self.run_async(_report())


@app.task(bind=True)
def cleanup_old_data(self):
    """Clean up old data based on retention policies"""
    try:
        logger.info("Starting data cleanup")
        
        # This would clean up old tick data, logs, etc.
        # TimescaleDB retention policies handle most of this automatically
        
        return {"status": "success", "cleanup_time": datetime.now().isoformat()}
        
    except Exception as e:
        logger.error(f"Data cleanup failed: {e}")
        return {"status": "error", "error": str(e)}


# Monitoring Tasks
@app.task(base=AsyncTask, bind=True)
def system_health_check(self):
    """Check overall system health"""
    
    async def _check():
        try:
            logger.info("Performing system health check")
            
            health_status = {
                "timestamp": datetime.now().isoformat(),
                "database": "unknown",
                "apis": {},
                "overall": "unknown"
            }
            
            # Check database connection
            try:
                config = get_config()
                db_config = DatabaseConfig(
                    host=config.database.host,
                    database=config.database.name,
                    username=config.database.username,
                    password=config.database.password
                )
                
                storage = await initialize_data_storage(db_config)
                await storage.close()
                health_status["database"] = "healthy"
                
            except Exception as e:
                health_status["database"] = f"unhealthy: {str(e)}"
            
            # Check API connections
            try:
                system_status = api_manager.get_system_status()
                health_status["apis"] = system_status.get("exchange_status", {})
                
            except Exception as e:
                health_status["apis"] = {"error": str(e)}
            
            # Determine overall health
            db_healthy = health_status["database"] == "healthy"
            apis_healthy = any(
                status.get("connected", False) 
                for status in health_status["apis"].values()
            )
            
            if db_healthy and apis_healthy:
                health_status["overall"] = "healthy"
            elif db_healthy or apis_healthy:
                health_status["overall"] = "degraded"
            else:
                health_status["overall"] = "unhealthy"
            
            logger.info(f"System health: {health_status['overall']}")
            return health_status
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {"status": "error", "error": str(e)}
    
    return self.run_async(_check())


@app.task(base=AsyncTask, bind=True)
def check_api_connections(self):
    """Check API connection status"""
    
    async def _check():
        try:
            logger.info("Checking API connections")
            
            # Test connections to all exchanges
            connection_results = await api_manager.connect_all()
            
            healthy_connections = sum(1 for status in connection_results.values() if status)
            total_connections = len(connection_results)
            
            return {
                "status": "success",
                "connections": connection_results,
                "healthy": healthy_connections,
                "total": total_connections,
                "health_percentage": (healthy_connections / total_connections * 100) if total_connections > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"API connection check failed: {e}")
            return {"status": "error", "error": str(e)}
    
    return self.run_async(_check())


# Weekly Tasks
@app.task(base=AsyncTask, bind=True)
def retrain_ml_models(self):
    """Retrain machine learning models with new data"""
    
    async def _retrain():
        try:
            logger.info("Starting ML model retraining")
            
            # This would retrain technical analysis and prediction models
            # with the latest data
            
            return {"status": "success", "retrain_time": datetime.now().isoformat()}
            
        except Exception as e:
            logger.error(f"ML model retraining failed: {e}")
            return {"status": "error", "error": str(e)}
    
    return self.run_async(_retrain())


@app.task(bind=True)
def backup_database(self):
    """Create database backup"""
    try:
        logger.info("Starting database backup")
        
        # This would create a backup of the TimescaleDB database
        # Could use pg_dump or TimescaleDB-specific backup tools
        
        return {"status": "success", "backup_time": datetime.now().isoformat()}
        
    except Exception as e:
        logger.error(f"Database backup failed: {e}")
        return {"status": "error", "error": str(e)}