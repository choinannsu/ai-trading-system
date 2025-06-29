"""
Configuration management for AI Trading System
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml
from pydantic import BaseSettings, Field, validator
from pydantic_settings import SettingsConfigDict


class DatabaseConfig(BaseSettings):
    """Database configuration"""
    
    url: str = Field(..., description="Database connection URL")
    pool_size: int = Field(default=20, description="Connection pool size")
    max_overflow: int = Field(default=30, description="Maximum pool overflow")
    echo: bool = Field(default=False, description="Echo SQL queries")


class RedisConfig(BaseSettings):
    """Redis configuration"""
    
    url: str = Field(..., description="Redis connection URL")
    max_connections: int = Field(default=50, description="Maximum connections")
    socket_timeout: int = Field(default=30, description="Socket timeout in seconds")


class TradingConfig(BaseSettings):
    """Trading configuration"""
    
    initial_capital: float = Field(..., description="Initial trading capital")
    monthly_investment: float = Field(..., description="Monthly additional investment")
    max_risk_per_trade: float = Field(default=0.02, description="Maximum risk per trade")
    max_daily_loss: float = Field(default=0.05, description="Maximum daily loss")
    max_drawdown: float = Field(default=0.15, description="Maximum drawdown")
    max_positions: int = Field(default=10, description="Maximum concurrent positions")
    trading_mode: str = Field(default="paper", description="Trading mode: paper or live")
    
    @validator('max_risk_per_trade', 'max_daily_loss', 'max_drawdown')
    def validate_risk_percentages(cls, v):
        if not 0 < v <= 1:
            raise ValueError('Risk percentages must be between 0 and 1')
        return v


class APIConfig(BaseSettings):
    """API configuration"""
    
    # Alpaca
    alpaca_api_key: Optional[str] = Field(default=None, env="ALPACA_API_KEY")
    alpaca_secret_key: Optional[str] = Field(default=None, env="ALPACA_SECRET_KEY")
    alpaca_base_url: str = Field(default="https://paper-api.alpaca.markets", env="ALPACA_BASE_URL")
    
    # Binance
    binance_api_key: Optional[str] = Field(default=None, env="BINANCE_API_KEY")
    binance_secret_key: Optional[str] = Field(default=None, env="BINANCE_SECRET_KEY")
    
    # Upbit
    upbit_access_key: Optional[str] = Field(default=None, env="UPBIT_ACCESS_KEY")
    upbit_secret_key: Optional[str] = Field(default=None, env="UPBIT_SECRET_KEY")
    
    # News API
    news_api_key: Optional[str] = Field(default=None, env="NEWS_API_KEY")
    alpha_vantage_api_key: Optional[str] = Field(default=None, env="ALPHA_VANTAGE_API_KEY")
    
    # Social Media
    twitter_bearer_token: Optional[str] = Field(default=None, env="TWITTER_BEARER_TOKEN")


class MLConfig(BaseSettings):
    """Machine Learning configuration"""
    
    model_retrain_frequency: str = Field(default="weekly", description="Model retrain frequency")
    prediction_horizon: int = Field(default=5, description="Prediction horizon in days")
    feature_window: int = Field(default=252, description="Feature engineering window")
    train_test_split: float = Field(default=0.8, description="Train/test split ratio")
    
    @validator('train_test_split')
    def validate_split_ratio(cls, v):
        if not 0 < v < 1:
            raise ValueError('Train/test split must be between 0 and 1')
        return v


class MonitoringConfig(BaseSettings):
    """Monitoring and alerting configuration"""
    
    # Telegram
    telegram_bot_token: Optional[str] = Field(default=None, env="TELEGRAM_BOT_TOKEN")
    telegram_chat_id: Optional[str] = Field(default=None, env="TELEGRAM_CHAT_ID")
    
    # Discord
    discord_webhook_url: Optional[str] = Field(default=None, env="DISCORD_WEBHOOK_URL")
    
    # Sentry
    sentry_dsn: Optional[str] = Field(default=None, env="SENTRY_DSN")
    
    # Alert thresholds
    daily_loss_threshold: float = Field(default=0.05, description="Daily loss alert threshold")
    drawdown_threshold: float = Field(default=0.10, description="Drawdown alert threshold")


class SystemConfig(BaseSettings):
    """Main system configuration"""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False
    )
    
    # Environment
    environment: str = Field(default="development", env="ENVIRONMENT")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    secret_key: str = Field(..., env="SECRET_KEY")
    
    # Database
    database_url: str = Field(..., env="DATABASE_URL")
    timescale_url: str = Field(..., env="TIMESCALE_URL")
    redis_url: str = Field(..., env="REDIS_URL")
    
    # Celery
    celery_broker_url: str = Field(..., env="CELERY_BROKER_URL")
    celery_result_backend: str = Field(..., env="CELERY_RESULT_BACKEND")
    
    # Sub-configurations
    trading: TradingConfig = Field(default_factory=TradingConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    ml: MLConfig = Field(default_factory=MLConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)


class ConfigManager:
    """Configuration manager class"""
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        self.config_path = Path(config_path) if config_path else Path("configs/config.yaml")
        self._config: Optional[Dict[str, Any]] = None
        self._system_config: Optional[SystemConfig] = None
        
    def load_yaml_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        if self._config is None:
            if self.config_path.exists():
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    self._config = yaml.safe_load(f)
            else:
                self._config = {}
        return self._config
    
    def get_system_config(self) -> SystemConfig:
        """Get system configuration"""
        if self._system_config is None:
            # Load YAML config first
            yaml_config = self.load_yaml_config()
            
            # Create system config from environment and YAML
            self._system_config = SystemConfig()
            
            # Override with YAML values if present
            if 'trading' in yaml_config:
                for key, value in yaml_config['trading'].items():
                    if hasattr(self._system_config.trading, key):
                        setattr(self._system_config.trading, key, value)
            
            if 'models' in yaml_config:
                for key, value in yaml_config['models'].items():
                    if hasattr(self._system_config.ml, key):
                        setattr(self._system_config.ml, key, value)
        
        return self._system_config
    
    def get_database_config(self) -> DatabaseConfig:
        """Get database configuration"""
        config = self.get_system_config()
        yaml_config = self.load_yaml_config()
        
        db_config = {
            'url': config.database_url,
            'pool_size': yaml_config.get('database', {}).get('main', {}).get('pool_size', 20),
            'max_overflow': yaml_config.get('database', {}).get('main', {}).get('max_overflow', 30),
            'echo': config.environment == 'development'
        }
        
        return DatabaseConfig(**db_config)
    
    def get_redis_config(self) -> RedisConfig:
        """Get Redis configuration"""
        config = self.get_system_config()
        return RedisConfig(url=config.redis_url)
    
    def get_config_value(self, key_path: str, default: Any = None) -> Any:
        """Get configuration value by dot-separated key path"""
        yaml_config = self.load_yaml_config()
        keys = key_path.split('.')
        
        value = yaml_config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def get_market_config(self, market_name: str) -> Optional[Dict[str, Any]]:
        """Get configuration for specific market"""
        markets = self.get_config_value('trading.markets', [])
        for market in markets:
            if market.get('name') == market_name:
                return market
        return None
    
    def is_market_enabled(self, market_name: str) -> bool:
        """Check if market is enabled"""
        market_config = self.get_market_config(market_name)
        return market_config.get('enabled', False) if market_config else False
    
    def get_alert_thresholds(self) -> Dict[str, float]:
        """Get alert thresholds"""
        config = self.get_system_config()
        return {
            'daily_loss': config.monitoring.daily_loss_threshold,
            'drawdown': config.monitoring.drawdown_threshold,
            'max_risk_per_trade': config.trading.max_risk_per_trade,
            'max_daily_loss': config.trading.max_daily_loss
        }


# Global configuration instance
config_manager = ConfigManager()


def get_config() -> SystemConfig:
    """Get system configuration"""
    return config_manager.get_system_config()


def get_database_config() -> DatabaseConfig:
    """Get database configuration"""
    return config_manager.get_database_config()


def get_redis_config() -> RedisConfig:
    """Get Redis configuration"""
    return config_manager.get_redis_config()


def is_production() -> bool:
    """Check if running in production environment"""
    return get_config().environment == "production"


def is_development() -> bool:
    """Check if running in development environment"""
    return get_config().environment == "development"