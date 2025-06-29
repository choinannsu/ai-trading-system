"""
Rate limiter for API calls
"""

import asyncio
import time
from collections import defaultdict, deque
from typing import Dict, Optional

from utils.logger import get_logger
from utils.exceptions import RateLimitError

logger = get_logger(__name__)


class RateLimiter:
    """Rate limiter with multiple window support"""
    
    def __init__(self, calls_per_second: int = 10, calls_per_minute: int = 100, 
                 calls_per_hour: int = 1000):
        self.calls_per_second = calls_per_second
        self.calls_per_minute = calls_per_minute
        self.calls_per_hour = calls_per_hour
        
        # Track calls for different time windows
        self.calls_second = deque()
        self.calls_minute = deque()  
        self.calls_hour = deque()
        
        self._lock = asyncio.Lock()
    
    async def acquire(self, endpoint: str = "default") -> None:
        """Acquire rate limit permission"""
        async with self._lock:
            now = time.time()
            
            # Clean old calls
            self._clean_old_calls(now)
            
            # Check limits
            if (len(self.calls_second) >= self.calls_per_second or
                len(self.calls_minute) >= self.calls_per_minute or
                len(self.calls_hour) >= self.calls_per_hour):
                
                wait_time = self._calculate_wait_time(now)
                logger.warning(f"Rate limit reached for {endpoint}, waiting {wait_time:.2f}s")
                
                if wait_time > 60:  # If wait time is too long, raise error
                    raise RateLimitError(
                        f"Rate limit exceeded for {endpoint}",
                        api_name=endpoint,
                        reset_time=int(now + wait_time)
                    )
                
                await asyncio.sleep(wait_time)
                now = time.time()
                self._clean_old_calls(now)
            
            # Record the call
            self.calls_second.append(now)
            self.calls_minute.append(now)
            self.calls_hour.append(now)
    
    def _clean_old_calls(self, now: float) -> None:
        """Remove old calls outside time windows"""
        # Clean calls older than 1 second
        while self.calls_second and now - self.calls_second[0] > 1:
            self.calls_second.popleft()
        
        # Clean calls older than 1 minute
        while self.calls_minute and now - self.calls_minute[0] > 60:
            self.calls_minute.popleft()
        
        # Clean calls older than 1 hour
        while self.calls_hour and now - self.calls_hour[0] > 3600:
            self.calls_hour.popleft()
    
    def _calculate_wait_time(self, now: float) -> float:
        """Calculate minimum wait time"""
        wait_times = []
        
        # Check second limit
        if len(self.calls_second) >= self.calls_per_second and self.calls_second:
            wait_times.append(1 - (now - self.calls_second[0]))
        
        # Check minute limit
        if len(self.calls_minute) >= self.calls_per_minute and self.calls_minute:
            wait_times.append(60 - (now - self.calls_minute[0]))
        
        # Check hour limit
        if len(self.calls_hour) >= self.calls_per_hour and self.calls_hour:
            wait_times.append(3600 - (now - self.calls_hour[0]))
        
        return max(wait_times) if wait_times else 0
    
    def get_remaining_calls(self) -> Dict[str, int]:
        """Get remaining calls for each time window"""
        now = time.time()
        self._clean_old_calls(now)
        
        return {
            'per_second': self.calls_per_second - len(self.calls_second),
            'per_minute': self.calls_per_minute - len(self.calls_minute),
            'per_hour': self.calls_per_hour - len(self.calls_hour)
        }


class ExchangeRateLimiter:
    """Exchange-specific rate limiter manager"""
    
    def __init__(self):
        self.limiters: Dict[str, RateLimiter] = {}
        self.exchange_configs = {
            'alpaca': {
                'calls_per_second': 200,
                'calls_per_minute': 200,
                'calls_per_hour': 10000
            },
            'binance': {
                'calls_per_second': 10,
                'calls_per_minute': 1200,
                'calls_per_hour': 100000
            },
            'kiwoom': {
                'calls_per_second': 4,
                'calls_per_minute': 200,
                'calls_per_hour': 1000
            },
            'upbit': {
                'calls_per_second': 8,
                'calls_per_minute': 600,
                'calls_per_hour': 10000
            }
        }
    
    def get_limiter(self, exchange: str) -> RateLimiter:
        """Get rate limiter for specific exchange"""
        if exchange not in self.limiters:
            config = self.exchange_configs.get(exchange, {
                'calls_per_second': 5,
                'calls_per_minute': 100,
                'calls_per_hour': 1000
            })
            self.limiters[exchange] = RateLimiter(**config)
        
        return self.limiters[exchange]
    
    async def acquire(self, exchange: str, endpoint: str = "default") -> None:
        """Acquire rate limit for specific exchange and endpoint"""
        limiter = self.get_limiter(exchange)
        await limiter.acquire(f"{exchange}:{endpoint}")
    
    def get_status(self, exchange: str) -> Dict[str, int]:
        """Get rate limit status for exchange"""
        if exchange not in self.limiters:
            return self.exchange_configs.get(exchange, {})
        
        limiter = self.limiters[exchange]
        return limiter.get_remaining_calls()


# Global rate limiter instance
rate_limiter = ExchangeRateLimiter()


def rate_limit(exchange: str, endpoint: str = "default"):
    """Decorator for rate limiting API calls"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            await rate_limiter.acquire(exchange, endpoint)
            return await func(*args, **kwargs)
        return wrapper
    return decorator