"""
Custom exceptions for AI Trading System
"""

from typing import Any, Dict, Optional


class TradingSystemException(Exception):
    """Base exception for trading system"""
    
    def __init__(self, message: str, error_code: str = None, context: Dict[str, Any] = None):
        self.message = message
        self.error_code = error_code
        self.context = context or {}
        super().__init__(self.message)
    
    def __str__(self) -> str:
        if self.error_code:
            return f"[{self.error_code}] {self.message}"
        return self.message


class ConfigurationError(TradingSystemException):
    """Configuration related errors"""
    pass


class DataCollectionError(TradingSystemException):
    """Data collection and processing errors"""
    pass


class APIError(TradingSystemException):
    """API related errors"""
    
    def __init__(self, message: str, api_name: str = None, status_code: int = None, 
                 response: str = None, **kwargs):
        self.api_name = api_name
        self.status_code = status_code
        self.response = response
        
        context = kwargs.get('context', {})
        context.update({
            'api_name': api_name,
            'status_code': status_code,
            'response': response
        })
        
        super().__init__(message, kwargs.get('error_code'), context)


class DatabaseError(TradingSystemException):
    """Database related errors"""
    pass


class ModelError(TradingSystemException):
    """Machine learning model errors"""
    
    def __init__(self, message: str, model_name: str = None, model_type: str = None, **kwargs):
        self.model_name = model_name
        self.model_type = model_type
        
        context = kwargs.get('context', {})
        context.update({
            'model_name': model_name,
            'model_type': model_type
        })
        
        super().__init__(message, kwargs.get('error_code'), context)


class TradingError(TradingSystemException):
    """Trading execution errors"""
    
    def __init__(self, message: str, symbol: str = None, order_type: str = None, 
                 quantity: float = None, price: float = None, **kwargs):
        self.symbol = symbol
        self.order_type = order_type
        self.quantity = quantity
        self.price = price
        
        context = kwargs.get('context', {})
        context.update({
            'symbol': symbol,
            'order_type': order_type,
            'quantity': quantity,
            'price': price
        })
        
        super().__init__(message, kwargs.get('error_code'), context)


class OrderExecutionError(TradingError):
    """Order execution specific errors"""
    pass


class InsufficientFundsError(TradingError):
    """Insufficient funds for trading"""
    pass


class RiskManagementError(TradingError):
    """Risk management violations"""
    
    def __init__(self, message: str, risk_type: str = None, current_value: float = None, 
                 limit_value: float = None, **kwargs):
        self.risk_type = risk_type
        self.current_value = current_value
        self.limit_value = limit_value
        
        context = kwargs.get('context', {})
        context.update({
            'risk_type': risk_type,
            'current_value': current_value,
            'limit_value': limit_value
        })
        
        super().__init__(message, kwargs.get('error_code'), context)


class PortfolioError(TradingSystemException):
    """Portfolio management errors"""
    
    def __init__(self, message: str, portfolio_id: str = None, **kwargs):
        self.portfolio_id = portfolio_id
        
        context = kwargs.get('context', {})
        context.update({'portfolio_id': portfolio_id})
        
        super().__init__(message, kwargs.get('error_code'), context)


class BacktestError(TradingSystemException):
    """Backtesting related errors"""
    
    def __init__(self, message: str, strategy_name: str = None, start_date: str = None, 
                 end_date: str = None, **kwargs):
        self.strategy_name = strategy_name
        self.start_date = start_date
        self.end_date = end_date
        
        context = kwargs.get('context', {})
        context.update({
            'strategy_name': strategy_name,
            'start_date': start_date,
            'end_date': end_date
        })
        
        super().__init__(message, kwargs.get('error_code'), context)


class ValidationError(TradingSystemException):
    """Data validation errors"""
    
    def __init__(self, message: str, field_name: str = None, field_value: Any = None, **kwargs):
        self.field_name = field_name
        self.field_value = field_value
        
        context = kwargs.get('context', {})
        context.update({
            'field_name': field_name,
            'field_value': field_value
        })
        
        super().__init__(message, kwargs.get('error_code'), context)


class AuthenticationError(TradingSystemException):
    """Authentication related errors"""
    pass


class AuthorizationError(TradingSystemException):
    """Authorization related errors"""
    pass


class RateLimitError(APIError):
    """Rate limit exceeded errors"""
    
    def __init__(self, message: str, api_name: str = None, reset_time: int = None, **kwargs):
        self.reset_time = reset_time
        
        context = kwargs.get('context', {})
        context.update({'reset_time': reset_time})
        
        super().__init__(message, api_name=api_name, error_code='RATE_LIMIT_EXCEEDED', 
                        context=context)


class MarketClosedError(TradingError):
    """Market closed errors"""
    
    def __init__(self, message: str, market_name: str = None, **kwargs):
        self.market_name = market_name
        
        context = kwargs.get('context', {})
        context.update({'market_name': market_name})
        
        super().__init__(message, error_code='MARKET_CLOSED', context=context)


class DataQualityError(DataCollectionError):
    """Data quality issues"""
    
    def __init__(self, message: str, data_source: str = None, data_type: str = None, 
                 quality_issue: str = None, **kwargs):
        self.data_source = data_source
        self.data_type = data_type
        self.quality_issue = quality_issue
        
        context = kwargs.get('context', {})
        context.update({
            'data_source': data_source,
            'data_type': data_type,
            'quality_issue': quality_issue
        })
        
        super().__init__(message, error_code='DATA_QUALITY_ISSUE', context=context)


class ModelNotFoundError(ModelError):
    """Model not found errors"""
    
    def __init__(self, message: str, model_name: str = None, **kwargs):
        super().__init__(message, model_name=model_name, error_code='MODEL_NOT_FOUND', **kwargs)


class ModelTrainingError(ModelError):
    """Model training errors"""
    
    def __init__(self, message: str, model_name: str = None, epoch: int = None, **kwargs):
        self.epoch = epoch
        
        context = kwargs.get('context', {})
        context.update({'epoch': epoch})
        
        super().__init__(message, model_name=model_name, error_code='MODEL_TRAINING_FAILED', 
                        context=context, **kwargs)


class PredictionError(ModelError):
    """Model prediction errors"""
    
    def __init__(self, message: str, model_name: str = None, input_shape: tuple = None, **kwargs):
        self.input_shape = input_shape
        
        context = kwargs.get('context', {})
        context.update({'input_shape': input_shape})
        
        super().__init__(message, model_name=model_name, error_code='PREDICTION_FAILED', 
                        context=context, **kwargs)


class SystemHealthError(TradingSystemException):
    """System health check errors"""
    
    def __init__(self, message: str, component: str = None, health_status: str = None, **kwargs):
        self.component = component
        self.health_status = health_status
        
        context = kwargs.get('context', {})
        context.update({
            'component': component,
            'health_status': health_status
        })
        
        super().__init__(message, error_code='SYSTEM_HEALTH_CHECK_FAILED', context=context)


# Exception mapping for error codes
ERROR_CODE_MAPPING = {
    'CONFIGURATION_ERROR': ConfigurationError,
    'DATA_COLLECTION_ERROR': DataCollectionError,
    'API_ERROR': APIError,
    'DATABASE_ERROR': DatabaseError,
    'MODEL_ERROR': ModelError,
    'TRADING_ERROR': TradingError,
    'ORDER_EXECUTION_ERROR': OrderExecutionError,
    'INSUFFICIENT_FUNDS': InsufficientFundsError,
    'RISK_MANAGEMENT_ERROR': RiskManagementError,
    'PORTFOLIO_ERROR': PortfolioError,
    'BACKTEST_ERROR': BacktestError,
    'VALIDATION_ERROR': ValidationError,
    'AUTHENTICATION_ERROR': AuthenticationError,
    'AUTHORIZATION_ERROR': AuthorizationError,
    'RATE_LIMIT_EXCEEDED': RateLimitError,
    'MARKET_CLOSED': MarketClosedError,
    'DATA_QUALITY_ISSUE': DataQualityError,
    'MODEL_NOT_FOUND': ModelNotFoundError,
    'MODEL_TRAINING_FAILED': ModelTrainingError,
    'PREDICTION_FAILED': PredictionError,
    'SYSTEM_HEALTH_CHECK_FAILED': SystemHealthError,
}


def create_exception_from_code(error_code: str, message: str, **kwargs) -> TradingSystemException:
    """Create exception from error code"""
    exception_class = ERROR_CODE_MAPPING.get(error_code, TradingSystemException)
    return exception_class(message, error_code=error_code, **kwargs)


def handle_exception(func):
    """Decorator to handle and log exceptions"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except TradingSystemException:
            # Re-raise trading system exceptions
            raise
        except Exception as e:
            # Convert other exceptions to TradingSystemException
            raise TradingSystemException(
                message=f"Unexpected error in {func.__name__}: {str(e)}",
                error_code='UNEXPECTED_ERROR',
                context={'function': func.__name__, 'args': str(args), 'kwargs': str(kwargs)}
            ) from e
    
    return wrapper