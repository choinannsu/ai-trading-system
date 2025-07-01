"""
Base Strategy Abstract Class
Foundation for all trading strategy implementations
"""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import warnings

from utils.logger import get_logger

logger = get_logger(__name__)


class SignalType(Enum):
    """Trading signal types"""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    CLOSE_LONG = "close_long"
    CLOSE_SHORT = "close_short"
    SCALE_IN = "scale_in"
    SCALE_OUT = "scale_out"


class PositionType(Enum):
    """Position types"""
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


class TradeStatus(Enum):
    """Trade execution status"""
    PENDING = "pending"
    FILLED = "filled"
    PARTIAL_FILL = "partial_fill"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class Signal:
    """Trading signal with metadata"""
    symbol: str
    signal_type: SignalType
    timestamp: datetime
    price: float
    confidence: float = 1.0
    size: float = 1.0  # Relative position size (0-1)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        # Validate confidence and size
        self.confidence = max(0.0, min(1.0, self.confidence))
        self.size = max(0.0, min(1.0, self.size))


@dataclass
class Position:
    """Position tracking"""
    symbol: str
    position_type: PositionType
    quantity: float
    entry_price: float
    entry_time: datetime
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    commission_paid: float = 0.0
    
    def update_price(self, new_price: float):
        """Update current price and unrealized P&L"""
        self.current_price = new_price
        
        if self.position_type == PositionType.LONG:
            self.unrealized_pnl = (new_price - self.entry_price) * self.quantity
        elif self.position_type == PositionType.SHORT:
            self.unrealized_pnl = (self.entry_price - new_price) * self.quantity
        else:
            self.unrealized_pnl = 0.0
    
    def get_total_pnl(self) -> float:
        """Get total P&L including realized and unrealized"""
        return self.realized_pnl + self.unrealized_pnl - self.commission_paid


@dataclass
class Trade:
    """Individual trade record"""
    symbol: str
    signal_type: SignalType
    timestamp: datetime
    price: float
    quantity: float
    commission: float = 0.0
    slippage: float = 0.0
    status: TradeStatus = TradeStatus.PENDING
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def total_cost(self) -> float:
        """Total transaction cost"""
        return self.commission + abs(self.slippage * self.quantity)
    
    @property
    def notional_value(self) -> float:
        """Notional value of the trade"""
        return abs(self.price * self.quantity)


@dataclass 
class StrategyConfig:
    """Base configuration for all strategies"""
    # Basic settings
    name: str = "BaseStrategy"
    symbols: List[str] = field(default_factory=lambda: ["AAPL"])
    initial_capital: float = 100000.0
    
    # Position management
    max_position_size: float = 0.1  # Max 10% per position
    max_portfolio_leverage: float = 1.0  # No leverage by default
    position_timeout: int = 0  # Days to hold position (0 = no timeout)
    
    # Risk management
    stop_loss_pct: float = 0.0  # Stop loss percentage (0 = disabled)
    take_profit_pct: float = 0.0  # Take profit percentage (0 = disabled)
    max_drawdown_pct: float = 0.2  # Max 20% drawdown
    
    # Transaction costs
    commission_rate: float = 0.001  # 0.1% commission
    slippage_rate: float = 0.0005  # 0.05% slippage
    
    # Rebalancing
    rebalance_frequency: str = "daily"  # daily, weekly, monthly
    min_trade_size: float = 100.0  # Minimum trade size in dollars
    
    # Strategy specific
    lookback_period: int = 20
    signal_threshold: float = 0.5
    
    def validate(self):
        """Validate configuration parameters"""
        assert 0 < self.max_position_size <= 1.0, "max_position_size must be between 0 and 1"
        assert self.max_portfolio_leverage >= 1.0, "max_portfolio_leverage must be >= 1"
        assert self.initial_capital > 0, "initial_capital must be positive"
        assert 0 <= self.stop_loss_pct <= 1.0, "stop_loss_pct must be between 0 and 1"
        assert 0 <= self.take_profit_pct <= 1.0, "take_profit_pct must be between 0 and 1"


class BaseStrategy(ABC):
    """Abstract base class for all trading strategies"""
    
    def __init__(self, config: StrategyConfig):
        self.config = config
        self.config.validate()
        
        # Portfolio state
        self.cash = config.initial_capital
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.equity_curve: List[float] = [config.initial_capital]
        self.timestamps: List[datetime] = []
        
        # Performance tracking
        self.total_returns = 0.0
        self.total_trades = 0
        self.winning_trades = 0
        self.max_drawdown = 0.0
        self.peak_equity = config.initial_capital
        
        # Strategy state
        self.is_active = True
        self.last_rebalance = None
        self.strategy_data: Dict[str, Any] = {}
        
        logger.info(f"Initialized {config.name} strategy with ${config.initial_capital:,.2f}")
    
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame, timestamp: datetime) -> List[Signal]:
        """
        Generate trading signals based on market data
        
        Args:
            data: Market data DataFrame with OHLCV columns
            timestamp: Current timestamp
            
        Returns:
            List of trading signals
        """
        pass
    
    @abstractmethod
    def calculate_position_size(self, signal: Signal, current_price: float, 
                              available_cash: float) -> float:
        """
        Calculate position size for a given signal
        
        Args:
            signal: Trading signal
            current_price: Current market price
            available_cash: Available cash for trading
            
        Returns:
            Position size (number of shares/contracts)
        """
        pass
    
    def process_signals(self, signals: List[Signal], market_data: Dict[str, float], 
                       timestamp: datetime) -> List[Trade]:
        """
        Process signals and generate trades
        
        Args:
            signals: List of trading signals
            market_data: Current market prices
            timestamp: Current timestamp
            
        Returns:
            List of executed trades
        """
        executed_trades = []
        
        for signal in signals:
            if signal.symbol not in market_data:
                logger.warning(f"No market data for {signal.symbol}")
                continue
            
            current_price = market_data[signal.symbol]
            
            # Apply risk management checks
            if not self._risk_check(signal, current_price):
                continue
            
            # Calculate position size
            position_size = self.calculate_position_size(
                signal, current_price, self.cash
            )
            
            if abs(position_size) < self.config.min_trade_size / current_price:
                continue  # Skip small trades
            
            # Execute trade
            trade = self._execute_trade(signal, current_price, position_size, timestamp)
            if trade:
                executed_trades.append(trade)
                self.trades.append(trade)
        
        return executed_trades
    
    def _execute_trade(self, signal: Signal, price: float, quantity: float, 
                      timestamp: datetime) -> Optional[Trade]:
        """Execute a trade with transaction costs"""
        if quantity == 0:
            return None
        
        # Calculate transaction costs
        notional_value = abs(price * quantity)
        commission = notional_value * self.config.commission_rate
        slippage = price * self.config.slippage_rate * np.sign(quantity)
        
        # Adjust execution price for slippage
        execution_price = price + slippage
        
        # Check if we have enough cash for buying
        if quantity > 0:  # Buying
            total_cost = notional_value + commission
            if total_cost > self.cash:
                # Adjust quantity based on available cash
                quantity = (self.cash - commission) / execution_price
                if quantity <= 0:
                    return None
        
        # Create trade
        trade = Trade(
            symbol=signal.symbol,
            signal_type=signal.signal_type,
            timestamp=timestamp,
            price=execution_price,
            quantity=quantity,
            commission=commission,
            slippage=slippage,
            status=TradeStatus.FILLED
        )
        
        # Update positions and cash
        self._update_position(trade)
        
        return trade
    
    def _update_position(self, trade: Trade):
        """Update position and cash based on executed trade"""
        symbol = trade.symbol
        
        # Update cash
        cash_flow = -(trade.price * trade.quantity + trade.commission)
        self.cash += cash_flow
        
        # Update position
        if symbol in self.positions:
            position = self.positions[symbol]
            
            if ((position.position_type == PositionType.LONG and trade.quantity > 0) or
                (position.position_type == PositionType.SHORT and trade.quantity < 0)):
                # Adding to existing position
                old_value = position.quantity * position.entry_price
                new_value = trade.quantity * trade.price
                total_quantity = position.quantity + trade.quantity
                
                if total_quantity != 0:
                    position.entry_price = (old_value + new_value) / total_quantity
                    position.quantity = total_quantity
                else:
                    # Position closed
                    del self.positions[symbol]
            else:
                # Reducing or reversing position
                if abs(trade.quantity) >= abs(position.quantity):
                    # Closing or reversing position
                    realized_pnl = self._calculate_realized_pnl(position, trade)
                    position.realized_pnl += realized_pnl
                    
                    remaining_quantity = trade.quantity + position.quantity
                    if abs(remaining_quantity) > 1e-6:
                        # Reversing position
                        position.quantity = remaining_quantity
                        position.position_type = PositionType.LONG if remaining_quantity > 0 else PositionType.SHORT
                        position.entry_price = trade.price
                        position.entry_time = trade.timestamp
                    else:
                        # Position completely closed
                        del self.positions[symbol]
                else:
                    # Partial close
                    realized_pnl = self._calculate_realized_pnl(position, trade)
                    position.realized_pnl += realized_pnl
                    position.quantity += trade.quantity
        else:
            # New position
            if abs(trade.quantity) > 1e-6:
                self.positions[symbol] = Position(
                    symbol=symbol,
                    position_type=PositionType.LONG if trade.quantity > 0 else PositionType.SHORT,
                    quantity=abs(trade.quantity),
                    entry_price=trade.price,
                    entry_time=trade.timestamp,
                    current_price=trade.price
                )
    
    def _calculate_realized_pnl(self, position: Position, trade: Trade) -> float:
        """Calculate realized P&L for a trade"""
        if position.position_type == PositionType.LONG:
            return trade.quantity * (trade.price - position.entry_price)
        else:
            return -trade.quantity * (trade.price - position.entry_price)
    
    def _risk_check(self, signal: Signal, current_price: float) -> bool:
        """Perform risk management checks"""
        # Check maximum position size
        portfolio_value = self.get_portfolio_value({signal.symbol: current_price})
        max_position_value = portfolio_value * self.config.max_position_size
        
        if signal.signal_type in [SignalType.BUY, SignalType.SCALE_IN]:
            current_position_value = 0
            if signal.symbol in self.positions:
                pos = self.positions[signal.symbol]
                if pos.position_type == PositionType.LONG:
                    current_position_value = pos.quantity * current_price
            
            # Check if adding to position would exceed limit
            proposed_value = current_position_value + (signal.size * max_position_value)
            if proposed_value > max_position_value:
                logger.warning(f"Position size limit exceeded for {signal.symbol}")
                return False
        
        # Check maximum drawdown
        current_equity = self.get_portfolio_value({signal.symbol: current_price})
        drawdown = (self.peak_equity - current_equity) / self.peak_equity
        
        if drawdown > self.config.max_drawdown_pct:
            logger.warning(f"Maximum drawdown exceeded: {drawdown:.2%}")
            self.is_active = False
            return False
        
        return True
    
    def update_positions(self, market_data: Dict[str, float], timestamp: datetime):
        """Update all positions with current market prices"""
        for symbol, position in self.positions.items():
            if symbol in market_data:
                position.update_price(market_data[symbol])
                
                # Check stop loss and take profit
                self._check_exit_conditions(position, timestamp)
        
        # Update equity curve
        portfolio_value = self.get_portfolio_value(market_data)
        self.equity_curve.append(portfolio_value)
        self.timestamps.append(timestamp)
        
        # Update peak equity and max drawdown
        if portfolio_value > self.peak_equity:
            self.peak_equity = portfolio_value
        
        current_drawdown = (self.peak_equity - portfolio_value) / self.peak_equity
        self.max_drawdown = max(self.max_drawdown, current_drawdown)
    
    def _check_exit_conditions(self, position: Position, timestamp: datetime):
        """Check stop loss and take profit conditions"""
        if self.config.stop_loss_pct > 0:
            stop_loss_price = position.entry_price * (1 - self.config.stop_loss_pct)
            if position.position_type == PositionType.LONG and position.current_price <= stop_loss_price:
                self._create_exit_signal(position, SignalType.SELL, "Stop Loss", timestamp)
            elif position.position_type == PositionType.SHORT and position.current_price >= stop_loss_price:
                self._create_exit_signal(position, SignalType.CLOSE_SHORT, "Stop Loss", timestamp)
        
        if self.config.take_profit_pct > 0:
            take_profit_price = position.entry_price * (1 + self.config.take_profit_pct)
            if position.position_type == PositionType.LONG and position.current_price >= take_profit_price:
                self._create_exit_signal(position, SignalType.SELL, "Take Profit", timestamp)
            elif position.position_type == PositionType.SHORT and position.current_price <= take_profit_price:
                self._create_exit_signal(position, SignalType.CLOSE_SHORT, "Take Profit", timestamp)
        
        # Check position timeout
        if self.config.position_timeout > 0:
            days_held = (timestamp - position.entry_time).days
            if days_held >= self.config.position_timeout:
                exit_signal_type = SignalType.SELL if position.position_type == PositionType.LONG else SignalType.CLOSE_SHORT
                self._create_exit_signal(position, exit_signal_type, "Timeout", timestamp)
    
    def _create_exit_signal(self, position: Position, signal_type: SignalType, 
                           reason: str, timestamp: datetime):
        """Create exit signal for position"""
        signal = Signal(
            symbol=position.symbol,
            signal_type=signal_type,
            timestamp=timestamp,
            price=position.current_price,
            confidence=1.0,
            size=1.0,  # Close entire position
            metadata={"reason": reason}
        )
        
        # Process the exit signal immediately
        self.process_signals([signal], {position.symbol: position.current_price}, timestamp)
    
    def get_portfolio_value(self, market_data: Dict[str, float]) -> float:
        """Calculate total portfolio value"""
        total_value = self.cash
        
        for symbol, position in self.positions.items():
            if symbol in market_data:
                position.update_price(market_data[symbol])
                total_value += position.quantity * position.current_price
        
        return total_value
    
    def get_positions_summary(self) -> Dict[str, Dict[str, Any]]:
        """Get summary of all positions"""
        summary = {}
        
        for symbol, position in self.positions.items():
            summary[symbol] = {
                "position_type": position.position_type.value,
                "quantity": position.quantity,
                "entry_price": position.entry_price,
                "current_price": position.current_price,
                "unrealized_pnl": position.unrealized_pnl,
                "realized_pnl": position.realized_pnl,
                "total_pnl": position.get_total_pnl(),
                "days_held": (datetime.now() - position.entry_time).days
            }
        
        return summary
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Calculate basic performance metrics"""
        if len(self.equity_curve) < 2:
            return {}
        
        returns = np.diff(self.equity_curve) / np.array(self.equity_curve[:-1])
        
        total_return = (self.equity_curve[-1] - self.config.initial_capital) / self.config.initial_capital
        annualized_return = (1 + total_return) ** (252 / len(returns)) - 1 if len(returns) > 0 else 0
        
        volatility = np.std(returns) * np.sqrt(252) if len(returns) > 1 else 0
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        
        # Calculate win rate
        winning_trades = sum(1 for trade in self.trades if trade.quantity * (trade.price - trade.price) > 0)
        win_rate = winning_trades / len(self.trades) if self.trades else 0
        
        return {
            "total_return": total_return,
            "annualized_return": annualized_return,
            "volatility": volatility,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": self.max_drawdown,
            "total_trades": len(self.trades),
            "win_rate": win_rate,
            "current_cash": self.cash,
            "portfolio_value": self.equity_curve[-1] if self.equity_curve else self.config.initial_capital
        }
    
    def reset(self):
        """Reset strategy state for new backtest"""
        self.cash = self.config.initial_capital
        self.positions.clear()
        self.trades.clear()
        self.equity_curve = [self.config.initial_capital]
        self.timestamps.clear()
        
        self.total_returns = 0.0
        self.total_trades = 0
        self.winning_trades = 0
        self.max_drawdown = 0.0
        self.peak_equity = self.config.initial_capital
        
        self.is_active = True
        self.last_rebalance = None
        self.strategy_data.clear()
        
        logger.info(f"Reset {self.config.name} strategy")


# Factory function
def create_strategy(strategy_type: str, config: StrategyConfig) -> BaseStrategy:
    """
    Factory function to create strategy instances
    
    Args:
        strategy_type: Type of strategy to create
        config: Strategy configuration
        
    Returns:
        Strategy instance
    """
    strategy_map = {
        "momentum": "MomentumStrategy",
        "mean_reversion": "MeanReversionStrategy", 
        "pairs_trading": "PairsTradingStrategy",
        "ml_ensemble": "MLEnsembleStrategy",
        "multi_timeframe": "MultiTimeframeStrategy"
    }
    
    if strategy_type not in strategy_map:
        raise ValueError(f"Unknown strategy type: {strategy_type}")
    
    # This would be implemented when we have the specific strategy classes
    raise NotImplementedError(f"Strategy {strategy_type} not yet implemented")


# Export classes and functions
__all__ = [
    'BaseStrategy',
    'StrategyConfig',
    'Signal', 
    'SignalType',
    'Position',
    'PositionType',
    'Trade',
    'TradeStatus',
    'create_strategy'
]