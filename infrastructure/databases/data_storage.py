"""
TimescaleDB Data Storage System for Financial Time Series Data
Handles market data, news, and analytics with optimized schema design
"""

import asyncio
import json
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
import logging

try:
    import asyncpg
except ImportError:
    asyncpg = None

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
except ImportError:
    psycopg2 = None

from ..data_collectors.market_data_collector import VolumeProfile, OrderbookData, TradeData
from ..data_collectors.news_collector import NewsArticle
from ..data_collectors.models import MarketData, TickData, ExchangeType
from utils.logger import get_logger
from utils.exceptions import DatabaseError

logger = get_logger(__name__)


@dataclass
class DatabaseConfig:
    """Database configuration"""
    host: str = "localhost"
    port: int = 5432
    database: str = "trading_system"
    username: str = "postgres"
    password: str = "password"
    pool_size: int = 10
    max_connections: int = 20
    command_timeout: int = 60


class TimescaleDBManager:
    """TimescaleDB connection and schema management"""
    
    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.pool = None
        self.is_connected = False
    
    async def connect(self) -> bool:
        """Establish database connection pool"""
        if not asyncpg:
            logger.error("asyncpg not installed. Install with: pip install asyncpg")
            return False
        
        try:
            dsn = f"postgresql://{self.config.username}:{self.config.password}@{self.config.host}:{self.config.port}/{self.config.database}"
            
            self.pool = await asyncpg.create_pool(
                dsn,
                min_size=1,
                max_size=self.config.max_connections,
                command_timeout=self.config.command_timeout
            )
            
            # Test connection
            async with self.pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
            
            self.is_connected = True
            logger.info("Connected to TimescaleDB successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to TimescaleDB: {e}")
            return False
    
    async def disconnect(self):
        """Close database connection pool"""
        if self.pool:
            await self.pool.close()
            self.is_connected = False
            logger.info("Disconnected from TimescaleDB")
    
    async def execute_query(self, query: str, *args) -> Any:
        """Execute a query and return results"""
        if not self.is_connected:
            raise DatabaseError("Not connected to database")
        
        try:
            async with self.pool.acquire() as conn:
                return await conn.fetch(query, *args)
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            raise DatabaseError(f"Query failed: {e}")
    
    async def execute_command(self, command: str, *args) -> str:
        """Execute a command and return status"""
        if not self.is_connected:
            raise DatabaseError("Not connected to database")
        
        try:
            async with self.pool.acquire() as conn:
                return await conn.execute(command, *args)
        except Exception as e:
            logger.error(f"Command execution failed: {e}")
            raise DatabaseError(f"Command failed: {e}")


class SchemaManager:
    """Database schema creation and management"""
    
    def __init__(self, db_manager: TimescaleDBManager):
        self.db = db_manager
    
    async def create_all_schemas(self):
        """Create all database schemas and tables"""
        await self.create_extensions()
        await self.create_market_data_schema()
        await self.create_news_schema()
        await self.create_analytics_schema()
        await self.create_hypertables()
        await self.create_indexes()
        await self.create_compression_policies()
        await self.create_retention_policies()
        logger.info("All database schemas created successfully")
    
    async def create_extensions(self):
        """Create required PostgreSQL extensions"""
        extensions = [
            "CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;",
            "CREATE EXTENSION IF NOT EXISTS pg_stat_statements;",
            "CREATE EXTENSION IF NOT EXISTS btree_gin;",
            "CREATE EXTENSION IF NOT EXISTS btree_gist;"
        ]
        
        for ext in extensions:
            try:
                await self.db.execute_command(ext)
            except Exception as e:
                logger.warning(f"Extension creation warning: {e}")
    
    async def create_market_data_schema(self):
        """Create market data tables"""
        
        # Exchanges table
        exchanges_table = """
        CREATE TABLE IF NOT EXISTS exchanges (
            id SERIAL PRIMARY KEY,
            name VARCHAR(50) UNIQUE NOT NULL,
            display_name VARCHAR(100),
            is_active BOOLEAN DEFAULT true,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
        """
        
        # Symbols table
        symbols_table = """
        CREATE TABLE IF NOT EXISTS symbols (
            id SERIAL PRIMARY KEY,
            symbol VARCHAR(20) NOT NULL,
            exchange_id INTEGER REFERENCES exchanges(id),
            asset_type VARCHAR(20) NOT NULL,
            base_asset VARCHAR(10),
            quote_asset VARCHAR(10),
            min_quantity DECIMAL(20,8),
            max_quantity DECIMAL(20,8),
            quantity_increment DECIMAL(20,8),
            min_price DECIMAL(20,8),
            max_price DECIMAL(20,8),
            price_increment DECIMAL(20,8),
            is_tradable BOOLEAN DEFAULT true,
            market_open TIME,
            market_close TIME,
            timezone VARCHAR(50),
            created_at TIMESTAMPTZ DEFAULT NOW(),
            UNIQUE(symbol, exchange_id)
        );
        """
        
        # Market data (OHLCV) table
        market_data_table = """
        CREATE TABLE IF NOT EXISTS market_data (
            timestamp TIMESTAMPTZ NOT NULL,
            symbol_id INTEGER NOT NULL REFERENCES symbols(id),
            exchange_id INTEGER NOT NULL REFERENCES exchanges(id),
            timeframe VARCHAR(10) NOT NULL,
            open_price DECIMAL(20,8) NOT NULL,
            high_price DECIMAL(20,8) NOT NULL,
            low_price DECIMAL(20,8) NOT NULL,
            close_price DECIMAL(20,8) NOT NULL,
            volume DECIMAL(20,8) NOT NULL,
            vwap DECIMAL(20,8),
            trade_count INTEGER,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
        """
        
        # Tick data table (real-time prices)
        tick_data_table = """
        CREATE TABLE IF NOT EXISTS tick_data (
            timestamp TIMESTAMPTZ NOT NULL,
            symbol_id INTEGER NOT NULL REFERENCES symbols(id),
            exchange_id INTEGER NOT NULL REFERENCES exchanges(id),
            price DECIMAL(20,8) NOT NULL,
            size DECIMAL(20,8) NOT NULL,
            bid DECIMAL(20,8),
            ask DECIMAL(20,8),
            bid_size DECIMAL(20,8),
            ask_size DECIMAL(20,8),
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
        """
        
        # Orderbook snapshots table
        orderbook_table = """
        CREATE TABLE IF NOT EXISTS orderbook_snapshots (
            timestamp TIMESTAMPTZ NOT NULL,
            symbol_id INTEGER NOT NULL REFERENCES symbols(id),
            exchange_id INTEGER NOT NULL REFERENCES exchanges(id),
            bids JSONB NOT NULL,
            asks JSONB NOT NULL,
            spread DECIMAL(20,8),
            mid_price DECIMAL(20,8),
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
        """
        
        # Trade data table
        trades_table = """
        CREATE TABLE IF NOT EXISTS trades (
            timestamp TIMESTAMPTZ NOT NULL,
            symbol_id INTEGER NOT NULL REFERENCES symbols(id),
            exchange_id INTEGER NOT NULL REFERENCES exchanges(id),
            trade_id VARCHAR(100),
            price DECIMAL(20,8) NOT NULL,
            size DECIMAL(20,8) NOT NULL,
            side VARCHAR(10),
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
        """
        
        # Volume profile table
        volume_profile_table = """
        CREATE TABLE IF NOT EXISTS volume_profiles (
            timestamp TIMESTAMPTZ NOT NULL,
            symbol_id INTEGER NOT NULL REFERENCES symbols(id),
            exchange_id INTEGER NOT NULL REFERENCES exchanges(id),
            price_levels JSONB NOT NULL,
            poc DECIMAL(20,8) NOT NULL,
            vah DECIMAL(20,8) NOT NULL,
            val DECIMAL(20,8) NOT NULL,
            total_volume DECIMAL(20,8) NOT NULL,
            period_minutes INTEGER DEFAULT 60,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
        """
        
        tables = [
            exchanges_table,
            symbols_table,
            market_data_table,
            tick_data_table,
            orderbook_table,
            trades_table,
            volume_profile_table
        ]
        
        for table in tables:
            await self.db.execute_command(table)
        
        # Insert default exchanges
        await self.insert_default_exchanges()
        
        logger.info("Market data schema created")
    
    async def create_news_schema(self):
        """Create news and sentiment data tables"""
        
        # News sources table
        news_sources_table = """
        CREATE TABLE IF NOT EXISTS news_sources (
            id SERIAL PRIMARY KEY,
            name VARCHAR(100) UNIQUE NOT NULL,
            source_type VARCHAR(50) NOT NULL,
            base_url VARCHAR(200),
            is_active BOOLEAN DEFAULT true,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
        """
        
        # News articles table
        news_articles_table = """
        CREATE TABLE IF NOT EXISTS news_articles (
            id SERIAL PRIMARY KEY,
            article_id VARCHAR(64) UNIQUE NOT NULL,
            title TEXT NOT NULL,
            content TEXT,
            url TEXT,
            source_id INTEGER REFERENCES news_sources(id),
            published_at TIMESTAMPTZ NOT NULL,
            collected_at TIMESTAMPTZ DEFAULT NOW(),
            author VARCHAR(200),
            category VARCHAR(50),
            language VARCHAR(10) DEFAULT 'en',
            sentiment_score DECIMAL(5,4),
            relevance_score DECIMAL(5,4),
            symbols TEXT[], -- Array of related symbols
            processed BOOLEAN DEFAULT false
        );
        """
        
        # News sentiment analysis table
        news_sentiment_table = """
        CREATE TABLE IF NOT EXISTS news_sentiment (
            timestamp TIMESTAMPTZ NOT NULL,
            symbol_id INTEGER REFERENCES symbols(id),
            article_id INTEGER REFERENCES news_articles(id),
            sentiment_score DECIMAL(5,4) NOT NULL,
            confidence DECIMAL(5,4),
            sentiment_label VARCHAR(20),
            keywords TEXT[],
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
        """
        
        tables = [
            news_sources_table,
            news_articles_table,
            news_sentiment_table
        ]
        
        for table in tables:
            await self.db.execute_command(table)
        
        logger.info("News schema created")
    
    async def create_analytics_schema(self):
        """Create analytics and derived data tables"""
        
        # Technical indicators table
        technical_indicators_table = """
        CREATE TABLE IF NOT EXISTS technical_indicators (
            timestamp TIMESTAMPTZ NOT NULL,
            symbol_id INTEGER NOT NULL REFERENCES symbols(id),
            exchange_id INTEGER NOT NULL REFERENCES exchanges(id),
            timeframe VARCHAR(10) NOT NULL,
            indicator_name VARCHAR(50) NOT NULL,
            indicator_value DECIMAL(20,8),
            parameters JSONB,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
        """
        
        # Trading signals table
        trading_signals_table = """
        CREATE TABLE IF NOT EXISTS trading_signals (
            timestamp TIMESTAMPTZ NOT NULL,
            symbol_id INTEGER NOT NULL REFERENCES symbols(id),
            exchange_id INTEGER NOT NULL REFERENCES exchanges(id),
            signal_type VARCHAR(20) NOT NULL, -- 'BUY', 'SELL', 'HOLD'
            signal_strength DECIMAL(5,4) NOT NULL, -- 0.0 to 1.0
            model_name VARCHAR(100) NOT NULL,
            model_version VARCHAR(20),
            confidence DECIMAL(5,4),
            features JSONB,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
        """
        
        # Portfolio snapshots table
        portfolio_snapshots_table = """
        CREATE TABLE IF NOT EXISTS portfolio_snapshots (
            timestamp TIMESTAMPTZ NOT NULL,
            exchange_id INTEGER NOT NULL REFERENCES exchanges(id),
            total_value DECIMAL(20,8) NOT NULL,
            available_balance DECIMAL(20,8) NOT NULL,
            locked_balance DECIMAL(20,8) NOT NULL,
            positions JSONB NOT NULL,
            pnl_unrealized DECIMAL(20,8),
            pnl_realized DECIMAL(20,8),
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
        """
        
        # Performance metrics table
        performance_metrics_table = """
        CREATE TABLE IF NOT EXISTS performance_metrics (
            timestamp TIMESTAMPTZ NOT NULL,
            metric_name VARCHAR(100) NOT NULL,
            metric_value DECIMAL(20,8) NOT NULL,
            period VARCHAR(20) NOT NULL, -- 'daily', 'weekly', 'monthly'
            additional_data JSONB,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
        """
        
        tables = [
            technical_indicators_table,
            trading_signals_table,
            portfolio_snapshots_table,
            performance_metrics_table
        ]
        
        for table in tables:
            await self.db.execute_command(table)
        
        logger.info("Analytics schema created")
    
    async def create_hypertables(self):
        """Convert tables to TimescaleDB hypertables"""
        hypertables = [
            ("market_data", "timestamp"),
            ("tick_data", "timestamp"),
            ("orderbook_snapshots", "timestamp"),
            ("trades", "timestamp"),
            ("volume_profiles", "timestamp"),
            ("news_articles", "published_at"),
            ("news_sentiment", "timestamp"),
            ("technical_indicators", "timestamp"),
            ("trading_signals", "timestamp"),
            ("portfolio_snapshots", "timestamp"),
            ("performance_metrics", "timestamp")
        ]
        
        for table_name, time_column in hypertables:
            try:
                query = f"SELECT create_hypertable('{table_name}', '{time_column}', if_not_exists => TRUE);"
                await self.db.execute_command(query)
                logger.info(f"Created hypertable: {table_name}")
            except Exception as e:
                logger.warning(f"Hypertable creation warning for {table_name}: {e}")
    
    async def create_indexes(self):
        """Create optimized indexes"""
        indexes = [
            # Market data indexes
            "CREATE INDEX IF NOT EXISTS idx_market_data_symbol_time ON market_data (symbol_id, timestamp DESC);",
            "CREATE INDEX IF NOT EXISTS idx_market_data_exchange_time ON market_data (exchange_id, timestamp DESC);",
            "CREATE INDEX IF NOT EXISTS idx_market_data_timeframe ON market_data (timeframe, timestamp DESC);",
            
            # Tick data indexes
            "CREATE INDEX IF NOT EXISTS idx_tick_data_symbol_time ON tick_data (symbol_id, timestamp DESC);",
            "CREATE INDEX IF NOT EXISTS idx_tick_data_exchange_time ON tick_data (exchange_id, timestamp DESC);",
            
            # Trades indexes
            "CREATE INDEX IF NOT EXISTS idx_trades_symbol_time ON trades (symbol_id, timestamp DESC);",
            "CREATE INDEX IF NOT EXISTS idx_trades_trade_id ON trades (trade_id);",
            
            # News indexes
            "CREATE INDEX IF NOT EXISTS idx_news_published_at ON news_articles (published_at DESC);",
            "CREATE INDEX IF NOT EXISTS idx_news_symbols ON news_articles USING GIN (symbols);",
            "CREATE INDEX IF NOT EXISTS idx_news_source ON news_articles (source_id, published_at DESC);",
            
            # Analytics indexes
            "CREATE INDEX IF NOT EXISTS idx_technical_indicators_symbol ON technical_indicators (symbol_id, indicator_name, timestamp DESC);",
            "CREATE INDEX IF NOT EXISTS idx_trading_signals_symbol ON trading_signals (symbol_id, timestamp DESC);",
            "CREATE INDEX IF NOT EXISTS idx_trading_signals_model ON trading_signals (model_name, timestamp DESC);",
        ]
        
        for index in indexes:
            try:
                await self.db.execute_command(index)
            except Exception as e:
                logger.warning(f"Index creation warning: {e}")
        
        logger.info("Database indexes created")
    
    async def create_compression_policies(self):
        """Create data compression policies"""
        compression_policies = [
            ("market_data", "1 day"),
            ("tick_data", "2 hours"),
            ("orderbook_snapshots", "1 hour"),
            ("trades", "2 hours"),
            ("news_articles", "7 days"),
            ("technical_indicators", "1 day"),
        ]
        
        for table_name, compress_after in compression_policies:
            try:
                query = f"""
                SELECT add_compression_policy('{table_name}', INTERVAL '{compress_after}');
                """
                await self.db.execute_command(query)
                logger.info(f"Added compression policy for {table_name}")
            except Exception as e:
                logger.warning(f"Compression policy warning for {table_name}: {e}")
    
    async def create_retention_policies(self):
        """Create data retention policies"""
        retention_policies = [
            ("tick_data", "30 days"),
            ("orderbook_snapshots", "7 days"),
            ("trades", "90 days"),
            ("market_data", "5 years"),  # Keep OHLCV data longer
            ("news_articles", "1 year"),
            ("technical_indicators", "1 year"),
            ("trading_signals", "6 months"),
            ("portfolio_snapshots", "2 years"),
        ]
        
        for table_name, drop_after in retention_policies:
            try:
                query = f"""
                SELECT add_retention_policy('{table_name}', INTERVAL '{drop_after}');
                """
                await self.db.execute_command(query)
                logger.info(f"Added retention policy for {table_name}")
            except Exception as e:
                logger.warning(f"Retention policy warning for {table_name}: {e}")
    
    async def insert_default_exchanges(self):
        """Insert default exchange data"""
        exchanges = [
            ("upbit", "Upbit"),
            ("kis", "Korea Investment Securities"),
            ("alpaca", "Alpaca Markets"),
            ("binance", "Binance"),
        ]
        
        for name, display_name in exchanges:
            try:
                query = """
                INSERT INTO exchanges (name, display_name)
                VALUES ($1, $2)
                ON CONFLICT (name) DO UPDATE SET
                display_name = EXCLUDED.display_name;
                """
                await self.db.execute_command(query, name, display_name)
            except Exception as e:
                logger.warning(f"Exchange insertion warning: {e}")


class DataStorage:
    """Main data storage interface"""
    
    def __init__(self, config: DatabaseConfig):
        self.db_manager = TimescaleDBManager(config)
        self.schema_manager = SchemaManager(self.db_manager)
        self._symbol_cache = {}
        self._exchange_cache = {}
    
    async def initialize(self, create_schema: bool = True):
        """Initialize database connection and schema"""
        success = await self.db_manager.connect()
        if not success:
            raise DatabaseError("Failed to connect to database")
        
        if create_schema:
            await self.schema_manager.create_all_schemas()
        
        await self._load_caches()
        logger.info("Data storage initialized successfully")
    
    async def _load_caches(self):
        """Load symbol and exchange caches"""
        # Load exchanges
        exchanges = await self.db_manager.execute_query("SELECT id, name FROM exchanges WHERE is_active = true")
        self._exchange_cache = {row['name']: row['id'] for row in exchanges}
        
        # Load symbols
        symbols = await self.db_manager.execute_query("""
            SELECT s.id, s.symbol, e.name as exchange_name
            FROM symbols s
            JOIN exchanges e ON s.exchange_id = e.id
            WHERE s.is_tradable = true
        """)
        
        for row in symbols:
            key = f"{row['symbol']}_{row['exchange_name']}"
            self._symbol_cache[key] = row['id']
    
    async def get_or_create_symbol(self, symbol: str, exchange: ExchangeType, **kwargs) -> int:
        """Get or create symbol and return ID"""
        cache_key = f"{symbol}_{exchange.value}"
        
        if cache_key in self._symbol_cache:
            return self._symbol_cache[cache_key]
        
        exchange_id = self._exchange_cache.get(exchange.value)
        if not exchange_id:
            raise DatabaseError(f"Exchange {exchange.value} not found")
        
        # Try to get existing symbol
        result = await self.db_manager.execute_query("""
            SELECT id FROM symbols WHERE symbol = $1 AND exchange_id = $2
        """, symbol, exchange_id)
        
        if result:
            symbol_id = result[0]['id']
        else:
            # Create new symbol
            query = """
            INSERT INTO symbols (symbol, exchange_id, asset_type, min_quantity, 
                               quantity_increment, price_increment, is_tradable)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            RETURNING id
            """
            
            asset_type = kwargs.get('asset_type', 'unknown')
            min_quantity = kwargs.get('min_quantity', 1)
            quantity_increment = kwargs.get('quantity_increment', 1)
            price_increment = kwargs.get('price_increment', 0.01)
            is_tradable = kwargs.get('is_tradable', True)
            
            result = await self.db_manager.execute_query(
                query, symbol, exchange_id, asset_type, min_quantity,
                quantity_increment, price_increment, is_tradable
            )
            symbol_id = result[0]['id']
        
        self._symbol_cache[cache_key] = symbol_id
        return symbol_id
    
    # Market Data Storage Methods
    async def store_market_data(self, market_data: List[MarketData]):
        """Store OHLCV market data"""
        if not market_data:
            return
        
        query = """
        INSERT INTO market_data (timestamp, symbol_id, exchange_id, timeframe,
                               open_price, high_price, low_price, close_price,
                               volume, vwap, trade_count)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
        ON CONFLICT DO NOTHING
        """
        
        data_rows = []
        for data in market_data:
            symbol_id = await self.get_or_create_symbol(data.symbol, data.exchange)
            exchange_id = self._exchange_cache[data.exchange.value]
            
            row = (
                data.timestamp, symbol_id, exchange_id, data.timeframe.value,
                float(data.open), float(data.high), float(data.low), float(data.close),
                float(data.volume), float(data.vwap) if data.vwap else None,
                data.trade_count
            )
            data_rows.append(row)
        
        try:
            async with self.db_manager.pool.acquire() as conn:
                await conn.executemany(query, data_rows)
            logger.info(f"Stored {len(data_rows)} market data records")
        except Exception as e:
            logger.error(f"Failed to store market data: {e}")
            raise DatabaseError(f"Market data storage failed: {e}")
    
    async def store_tick_data(self, tick_data: List[TickData]):
        """Store real-time tick data"""
        if not tick_data:
            return
        
        query = """
        INSERT INTO tick_data (timestamp, symbol_id, exchange_id, price, size,
                             bid, ask, bid_size, ask_size)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
        """
        
        data_rows = []
        for tick in tick_data:
            symbol_id = await self.get_or_create_symbol(tick.symbol, tick.exchange)
            exchange_id = self._exchange_cache[tick.exchange.value]
            
            row = (
                tick.timestamp, symbol_id, exchange_id, float(tick.price), float(tick.size),
                float(tick.bid) if tick.bid else None,
                float(tick.ask) if tick.ask else None,
                float(tick.bid_size) if tick.bid_size else None,
                float(tick.ask_size) if tick.ask_size else None
            )
            data_rows.append(row)
        
        try:
            async with self.db_manager.pool.acquire() as conn:
                await conn.executemany(query, data_rows)
            logger.debug(f"Stored {len(data_rows)} tick data records")
        except Exception as e:
            logger.error(f"Failed to store tick data: {e}")
    
    async def store_news_articles(self, articles: List[NewsArticle]):
        """Store news articles"""
        if not articles:
            return
        
        # First ensure news sources exist
        await self._ensure_news_sources(articles)
        
        query = """
        INSERT INTO news_articles (article_id, title, content, url, source_id,
                                 published_at, author, category, language,
                                 sentiment_score, relevance_score, symbols)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
        ON CONFLICT (article_id) DO UPDATE SET
        sentiment_score = EXCLUDED.sentiment_score,
        relevance_score = EXCLUDED.relevance_score,
        symbols = EXCLUDED.symbols
        """
        
        source_cache = await self._get_news_source_cache()
        
        data_rows = []
        for article in articles:
            source_id = source_cache.get(article.source)
            if not source_id:
                continue
            
            row = (
                article.article_id, article.title, article.content, article.url,
                source_id, article.published_at, article.author, article.category,
                article.language, 
                float(article.sentiment_score) if article.sentiment_score else None,
                float(article.relevance_score) if article.relevance_score else None,
                article.symbols
            )
            data_rows.append(row)
        
        try:
            async with self.db_manager.pool.acquire() as conn:
                await conn.executemany(query, data_rows)
            logger.info(f"Stored {len(data_rows)} news articles")
        except Exception as e:
            logger.error(f"Failed to store news articles: {e}")
    
    async def _ensure_news_sources(self, articles: List[NewsArticle]):
        """Ensure news sources exist in database"""
        sources = set(article.source for article in articles)
        
        for source in sources:
            query = """
            INSERT INTO news_sources (name, source_type)
            VALUES ($1, $2)
            ON CONFLICT (name) DO NOTHING
            """
            
            try:
                await self.db_manager.execute_command(query, source, 'news')
            except Exception as e:
                logger.warning(f"Failed to insert news source {source}: {e}")
    
    async def _get_news_source_cache(self) -> Dict[str, int]:
        """Get news source ID cache"""
        sources = await self.db_manager.execute_query("SELECT id, name FROM news_sources")
        return {row['name']: row['id'] for row in sources}
    
    # Query Methods
    async def get_latest_market_data(self, symbol: str, exchange: ExchangeType, 
                                   timeframe: str, limit: int = 100) -> List[Dict]:
        """Get latest market data"""
        try:
            symbol_id = await self.get_or_create_symbol(symbol, exchange)
            
            query = """
            SELECT timestamp, open_price, high_price, low_price, close_price,
                   volume, vwap, trade_count
            FROM market_data
            WHERE symbol_id = $1 AND timeframe = $2
            ORDER BY timestamp DESC
            LIMIT $3
            """
            
            result = await self.db_manager.execute_query(query, symbol_id, timeframe, limit)
            return [dict(row) for row in result]
            
        except Exception as e:
            logger.error(f"Failed to get market data: {e}")
            return []
    
    async def get_symbol_news(self, symbol: str, hours_back: int = 24) -> List[Dict]:
        """Get news articles for a symbol"""
        try:
            query = """
            SELECT na.title, na.content, na.url, ns.name as source,
                   na.published_at, na.sentiment_score, na.relevance_score
            FROM news_articles na
            JOIN news_sources ns ON na.source_id = ns.id
            WHERE $1 = ANY(na.symbols)
            AND na.published_at >= NOW() - INTERVAL '%s hours'
            ORDER BY na.published_at DESC
            """ % hours_back
            
            result = await self.db_manager.execute_query(query, symbol.upper())
            return [dict(row) for row in result]
            
        except Exception as e:
            logger.error(f"Failed to get symbol news: {e}")
            return []
    
    async def close(self):
        """Close database connections"""
        await self.db_manager.disconnect()


# Global storage instance
data_storage = None


async def initialize_data_storage(config: DatabaseConfig) -> DataStorage:
    """Initialize global data storage"""
    global data_storage
    data_storage = DataStorage(config)
    await data_storage.initialize()
    return data_storage


# Usage example
async def example_usage():
    """Example usage of data storage"""
    config = DatabaseConfig(
        host="localhost",
        database="trading_system",
        username="postgres",
        password="password"
    )
    
    storage = await initialize_data_storage(config)
    
    # Query latest market data
    btc_data = await storage.get_latest_market_data('KRW-BTC', ExchangeType.UPBIT, '1h', 24)
    print(f"Retrieved {len(btc_data)} BTC hourly candles")
    
    # Query news
    btc_news = await storage.get_symbol_news('BTC', 48)
    print(f"Found {len(btc_news)} BTC news articles")
    
    await storage.close()


if __name__ == "__main__":
    asyncio.run(example_usage())