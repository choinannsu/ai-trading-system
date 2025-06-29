"""
Logging configuration for AI Trading System
"""

import sys
from pathlib import Path
from typing import Optional

from loguru import logger


class LoggerConfig:
    """Logger configuration class"""
    
    def __init__(self, log_level: str = "INFO", log_dir: str = "logs"):
        self.log_level = log_level
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
    def setup_logger(self, module_name: Optional[str] = None) -> None:
        """Setup logger with file and console handlers"""
        
        # Remove default handler
        logger.remove()
        
        # Console handler with colors
        logger.add(
            sys.stdout,
            level=self.log_level,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                   "<level>{level: <8}</level> | "
                   "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
                   "<level>{message}</level>",
            colorize=True
        )
        
        # File handler for all logs
        logger.add(
            self.log_dir / "trading_system.log",
            level="DEBUG",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
            rotation="10 MB",
            retention="30 days",
            compression="zip"
        )
        
        # Separate file handler for errors
        logger.add(
            self.log_dir / "errors.log",
            level="ERROR",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
            rotation="5 MB",
            retention="60 days",
            compression="zip"
        )
        
        # Trading specific logs
        logger.add(
            self.log_dir / "trading.log",
            level="INFO",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
            filter=lambda record: "TRADE" in record["extra"],
            rotation="daily",
            retention="1 year",
            compression="zip"
        )
        
        # Performance logs
        logger.add(
            self.log_dir / "performance.log",
            level="INFO",
            format="{time:YYYY-MM-DD HH:mm:ss} | {message}",
            filter=lambda record: "PERFORMANCE" in record["extra"],
            rotation="daily",
            retention="1 year",
            compression="zip"
        )
        
        if module_name:
            logger.bind(module=module_name)


def get_logger(module_name: str, log_level: str = "INFO") -> logger:
    """Get configured logger for a module"""
    config = LoggerConfig(log_level=log_level)
    config.setup_logger(module_name)
    return logger.bind(module=module_name)


def log_trade(action: str, symbol: str, quantity: float, price: float, **kwargs) -> None:
    """Log trading activity"""
    logger.bind(TRADE=True).info(
        f"TRADE | {action} | {symbol} | Qty: {quantity} | Price: {price} | {kwargs}"
    )


def log_performance(metric: str, value: float, **kwargs) -> None:
    """Log performance metrics"""
    logger.bind(PERFORMANCE=True).info(
        f"PERFORMANCE | {metric}: {value} | {kwargs}"
    )


def log_error_with_context(error: Exception, context: dict) -> None:
    """Log error with additional context"""
    logger.error(f"Error: {str(error)} | Context: {context}", exc_info=True)


def log_api_call(api_name: str, endpoint: str, status_code: int, response_time: float) -> None:
    """Log API call details"""
    logger.info(
        f"API_CALL | {api_name} | {endpoint} | Status: {status_code} | Time: {response_time:.2f}s"
    )


def log_model_performance(model_name: str, accuracy: float, loss: float, **metrics) -> None:
    """Log ML model performance"""
    logger.bind(PERFORMANCE=True).info(
        f"MODEL | {model_name} | Accuracy: {accuracy:.4f} | Loss: {loss:.4f} | {metrics}"
    )


def log_data_collection(source: str, records_count: int, time_taken: float) -> None:
    """Log data collection activities"""
    logger.info(
        f"DATA_COLLECTION | {source} | Records: {records_count} | Time: {time_taken:.2f}s"
    )


def log_system_health(component: str, status: str, **metrics) -> None:
    """Log system health metrics"""
    logger.info(f"HEALTH | {component} | Status: {status} | {metrics}")