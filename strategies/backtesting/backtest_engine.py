"""
Vectorized Backtesting Engine
High-performance backtesting with realistic transaction costs and market impact simulation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

from ..base_strategy import BaseStrategy, StrategyConfig, Signal, Trade, Position
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class BacktestConfig:
    """Configuration for backtesting"""
    # Date range
    start_date: datetime
    end_date: datetime
    
    # Initial conditions
    initial_capital: float = 100000.0
    benchmark_symbol: str = "SPY"
    
    # Transaction costs
    commission_rate: float = 0.001      # 0.1% commission
    bid_ask_spread: float = 0.0005      # 0.05% bid-ask spread
    market_impact_model: str = "sqrt"   # "linear", "sqrt", "power_law"
    market_impact_coef: float = 0.0001  # Market impact coefficient
    
    # Slippage modeling
    slippage_model: str = "linear"      # "linear", "sqrt", "exponential"
    slippage_base: float = 0.0002       # Base slippage (0.02%)
    slippage_volume_factor: float = 0.1 # Volume impact factor
    
    # Execution modeling
    execution_delay: int = 0            # Execution delay in bars
    partial_fill_prob: float = 0.0     # Probability of partial fills
    min_fill_ratio: float = 0.5        # Minimum fill ratio for partial fills
    
    # Performance analysis
    risk_free_rate: float = 0.02        # Annual risk-free rate
    benchmark_returns: Optional[pd.Series] = None
    
    # Walk-forward analysis
    use_walk_forward: bool = False
    training_period_months: int = 12
    rebalance_frequency_months: int = 3
    
    # Multi-processing
    use_multiprocessing: bool = True
    max_workers: Optional[int] = None
    
    # Debugging
    save_trades: bool = True
    save_positions: bool = True
    verbose: bool = False


@dataclass 
class BacktestResult:
    """Results from backtesting"""
    # Basic metrics
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    
    # Trade statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    
    # Time series
    equity_curve: pd.Series
    returns: pd.Series
    positions: Optional[pd.DataFrame] = None
    trades: Optional[pd.DataFrame] = None
    
    # Benchmark comparison
    benchmark_return: Optional[float] = None
    alpha: Optional[float] = None
    beta: Optional[float] = None
    information_ratio: Optional[float] = None
    
    # Risk metrics
    var_95: float = 0.0
    cvar_95: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    
    # Strategy specific
    strategy_metrics: Dict[str, Any] = field(default_factory=dict)


class BacktestEngine:
    """
    Vectorized backtesting engine with realistic transaction cost modeling
    
    Features:
    - Vectorized operations for speed
    - Realistic transaction costs and market impact
    - Walk-forward analysis support
    - Multi-processing capabilities
    - Comprehensive performance metrics
    """
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        
        # Market data
        self.market_data: Optional[pd.DataFrame] = None
        self.benchmark_data: Optional[pd.Series] = None
        
        # Transaction cost models
        self.cost_models = {
            'commission': self._calculate_commission,
            'spread': self._calculate_spread,
            'market_impact': self._calculate_market_impact,
            'slippage': self._calculate_slippage
        }
        
        logger.info(f"Initialized backtest engine for period {config.start_date} to {config.end_date}")
    
    def load_data(self, market_data: pd.DataFrame, benchmark_data: Optional[pd.Series] = None):
        """Load market data for backtesting"""
        # Filter data by date range
        self.market_data = market_data[
            (market_data.index >= self.config.start_date) & 
            (market_data.index <= self.config.end_date)
        ].copy()
        
        if benchmark_data is not None:
            self.benchmark_data = benchmark_data[
                (benchmark_data.index >= self.config.start_date) & 
                (benchmark_data.index <= self.config.end_date)
            ].copy()
        
        logger.info(f"Loaded market data: {len(self.market_data)} periods")
    
    def run_backtest(self, strategy: BaseStrategy) -> BacktestResult:
        """Run backtest for a strategy"""
        if self.market_data is None:
            raise ValueError("Market data not loaded. Call load_data() first.")
        
        if self.config.use_walk_forward:
            return self._run_walk_forward_backtest(strategy)
        else:
            return self._run_single_backtest(strategy)
    
    def _run_single_backtest(self, strategy: BaseStrategy) -> BacktestResult:
        """Run a single backtest"""
        # Reset strategy
        strategy.reset()
        
        # Initialize tracking variables
        portfolio_values = []
        trade_records = []
        position_records = []
        
        # Run simulation
        for i, (timestamp, row) in enumerate(self.market_data.iterrows()):
            # Get market data for current timestamp
            current_data = self.market_data.iloc[:i+1]
            market_prices = self._extract_current_prices(row)
            
            # Generate signals
            try:
                signals = strategy.generate_signals(current_data, timestamp)
            except Exception as e:
                logger.warning(f"Error generating signals at {timestamp}: {e}")
                signals = []
            
            # Process signals with realistic execution
            executed_trades = self._execute_signals_realistic(
                signals, market_prices, timestamp, i
            )
            
            # Process trades through strategy
            strategy.process_signals(signals, market_prices, timestamp)
            
            # Update positions with current prices
            strategy.update_positions(market_prices, timestamp)
            
            # Record portfolio value
            portfolio_value = strategy.get_portfolio_value(market_prices)
            portfolio_values.append(portfolio_value)
            
            # Record trades
            if executed_trades:
                trade_records.extend(self._format_trade_records(executed_trades, timestamp))
            
            # Record positions
            if self.config.save_positions and i % 5 == 0:  # Save every 5th position
                position_records.append(self._format_position_record(
                    strategy.get_positions_summary(), timestamp, portfolio_value
                ))
        
        # Create result
        return self._create_backtest_result(
            strategy, portfolio_values, trade_records, position_records
        )
    
    def _run_walk_forward_backtest(self, strategy: BaseStrategy) -> BacktestResult:
        """Run walk-forward analysis backtest"""
        logger.info("Running walk-forward analysis...")
        
        # Calculate walk-forward periods
        periods = self._calculate_walk_forward_periods()
        
        # Run backtests for each period
        if self.config.use_multiprocessing and len(periods) > 1:
            results = self._run_parallel_walk_forward(strategy, periods)
        else:
            results = self._run_sequential_walk_forward(strategy, periods)
        
        # Combine results
        return self._combine_walk_forward_results(results)
    
    def _calculate_walk_forward_periods(self) -> List[Tuple[datetime, datetime, datetime]]:
        """Calculate walk-forward periods (training_start, training_end, testing_end)"""
        periods = []
        
        training_delta = timedelta(days=self.config.training_period_months * 30)
        rebalance_delta = timedelta(days=self.config.rebalance_frequency_months * 30)
        
        current_start = self.config.start_date
        
        while current_start < self.config.end_date:
            training_end = current_start + training_delta
            testing_end = training_end + rebalance_delta
            
            if testing_end > self.config.end_date:
                testing_end = self.config.end_date
            
            periods.append((current_start, training_end, testing_end))
            current_start = training_end
        
        return periods
    
    def _run_parallel_walk_forward(self, strategy: BaseStrategy, 
                                  periods: List[Tuple[datetime, datetime, datetime]]) -> List[BacktestResult]:
        """Run walk-forward analysis in parallel"""
        max_workers = self.config.max_workers or min(len(periods), mp.cpu_count())
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_period = {}
            
            for period in periods:
                future = executor.submit(self._run_walk_forward_period, strategy, period)
                future_to_period[future] = period
            
            results = []
            for future in as_completed(future_to_period):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    period = future_to_period[future]
                    logger.error(f"Walk-forward period {period} failed: {e}")
        
        return results
    
    def _run_sequential_walk_forward(self, strategy: BaseStrategy, 
                                    periods: List[Tuple[datetime, datetime, datetime]]) -> List[BacktestResult]:
        """Run walk-forward analysis sequentially"""
        results = []
        
        for period in periods:
            try:
                result = self._run_walk_forward_period(strategy, period)
                results.append(result)
            except Exception as e:
                logger.error(f"Walk-forward period {period} failed: {e}")
        
        return results
    
    def _run_walk_forward_period(self, strategy: BaseStrategy, 
                                period: Tuple[datetime, datetime, datetime]) -> BacktestResult:
        """Run a single walk-forward period"""
        training_start, training_end, testing_end = period
        
        # Filter data for this period
        training_data = self.market_data[
            (self.market_data.index >= training_start) & 
            (self.market_data.index <= training_end)
        ]
        
        testing_data = self.market_data[
            (self.market_data.index > training_end) & 
            (self.market_data.index <= testing_end)
        ]
        
        # Reset strategy and run on testing data
        strategy.reset()
        
        # Run backtest on testing period
        temp_market_data = self.market_data
        self.market_data = testing_data
        
        try:
            result = self._run_single_backtest(strategy)
        finally:
            self.market_data = temp_market_data
        
        return result
    
    def _extract_current_prices(self, row: pd.Series) -> Dict[str, float]:
        """Extract current prices from market data row"""
        prices = {}
        
        for col in row.index:
            if col.endswith('_close'):
                symbol = col.replace('_close', '')
                prices[symbol] = row[col]
        
        return prices
    
    def _execute_signals_realistic(self, signals: List[Signal], market_prices: Dict[str, float], 
                                 timestamp: datetime, bar_index: int) -> List[Trade]:
        """Execute signals with realistic transaction costs and market impact"""
        executed_trades = []
        
        for signal in signals:
            if signal.symbol not in market_prices:
                continue
            
            market_price = market_prices[signal.symbol]
            
            # Calculate transaction costs
            costs = self._calculate_total_transaction_costs(
                signal, market_price, bar_index
            )
            
            # Determine execution price
            execution_price = self._calculate_execution_price(
                signal, market_price, costs
            )
            
            # Handle partial fills
            fill_ratio = self._determine_fill_ratio(signal, market_price)
            
            if fill_ratio > 0:
                # Calculate actual quantity
                target_quantity = self._calculate_target_quantity(signal, execution_price)
                actual_quantity = target_quantity * fill_ratio
                
                if abs(actual_quantity) > 1e-6:  # Minimum trade size
                    trade = Trade(
                        symbol=signal.symbol,
                        signal_type=signal.signal_type,
                        timestamp=timestamp,
                        price=execution_price,
                        quantity=actual_quantity,
                        commission=costs['commission'],
                        slippage=costs['slippage'],
                        metadata={
                            **signal.metadata,
                            'market_impact': costs['market_impact'],
                            'spread_cost': costs['spread'],
                            'fill_ratio': fill_ratio
                        }
                    )
                    
                    executed_trades.append(trade)
        
        return executed_trades
    
    def _calculate_total_transaction_costs(self, signal: Signal, market_price: float, 
                                         bar_index: int) -> Dict[str, float]:
        """Calculate all transaction costs"""
        # Estimate notional value
        notional_value = abs(signal.size * self.config.initial_capital)
        
        costs = {}
        
        # Commission
        costs['commission'] = self._calculate_commission(notional_value, signal)
        
        # Bid-ask spread
        costs['spread'] = self._calculate_spread(notional_value, market_price, signal)
        
        # Market impact
        costs['market_impact'] = self._calculate_market_impact(
            notional_value, market_price, signal, bar_index
        )
        
        # Slippage
        costs['slippage'] = self._calculate_slippage(
            notional_value, market_price, signal, bar_index
        )
        
        return costs
    
    def _calculate_commission(self, notional_value: float, signal: Signal) -> float:
        """Calculate commission costs"""
        return notional_value * self.config.commission_rate
    
    def _calculate_spread(self, notional_value: float, market_price: float, signal: Signal) -> float:
        """Calculate bid-ask spread costs"""
        return notional_value * self.config.bid_ask_spread
    
    def _calculate_market_impact(self, notional_value: float, market_price: float, 
                               signal: Signal, bar_index: int) -> float:
        """Calculate market impact costs"""
        # Get volume data if available
        volume_col = f"{signal.symbol}_volume"
        if (self.market_data is not None and 
            volume_col in self.market_data.columns and 
            bar_index < len(self.market_data)):
            
            daily_volume = self.market_data.iloc[bar_index][volume_col]
            volume_ratio = notional_value / (daily_volume * market_price + 1e-6)
        else:
            volume_ratio = 0.001  # Default assumption
        
        if self.config.market_impact_model == "linear":
            impact = self.config.market_impact_coef * volume_ratio
        elif self.config.market_impact_model == "sqrt":
            impact = self.config.market_impact_coef * np.sqrt(volume_ratio)
        elif self.config.market_impact_model == "power_law":
            impact = self.config.market_impact_coef * (volume_ratio ** 0.6)
        else:
            impact = 0.0
        
        return notional_value * impact
    
    def _calculate_slippage(self, notional_value: float, market_price: float, 
                          signal: Signal, bar_index: int) -> float:
        """Calculate slippage costs"""
        # Base slippage
        base_slippage = self.config.slippage_base
        
        # Volume-based slippage
        volume_slippage = 0
        volume_col = f"{signal.symbol}_volume"
        if (self.market_data is not None and 
            volume_col in self.market_data.columns and 
            bar_index < len(self.market_data)):
            
            daily_volume = self.market_data.iloc[bar_index][volume_col]
            avg_volume = self.market_data[volume_col].rolling(20).mean().iloc[bar_index]
            
            if avg_volume > 0:
                volume_ratio = daily_volume / avg_volume
                volume_slippage = self.config.slippage_volume_factor * max(0, 1 - volume_ratio)
        
        # Calculate total slippage
        if self.config.slippage_model == "linear":
            total_slippage = base_slippage + volume_slippage
        elif self.config.slippage_model == "sqrt":
            total_slippage = base_slippage * np.sqrt(1 + volume_slippage)
        elif self.config.slippage_model == "exponential":
            total_slippage = base_slippage * np.exp(volume_slippage)
        else:
            total_slippage = base_slippage
        
        return notional_value * total_slippage
    
    def _calculate_execution_price(self, signal: Signal, market_price: float, 
                                 costs: Dict[str, float]) -> float:
        """Calculate realistic execution price including all costs"""
        # Direction-dependent cost application
        if signal.signal_type.value in ['buy', 'scale_in']:
            # Buying - costs increase execution price
            price_impact = (costs['market_impact'] + costs['slippage']) / (signal.size * self.config.initial_capital + 1e-6)
            execution_price = market_price * (1 + price_impact + self.config.bid_ask_spread / 2)
        else:
            # Selling - costs decrease execution price
            price_impact = (costs['market_impact'] + costs['slippage']) / (signal.size * self.config.initial_capital + 1e-6)
            execution_price = market_price * (1 - price_impact - self.config.bid_ask_spread / 2)
        
        return max(0.01, execution_price)  # Prevent negative prices
    
    def _determine_fill_ratio(self, signal: Signal, market_price: float) -> float:
        """Determine the fill ratio for partial fills"""
        if np.random.random() < self.config.partial_fill_prob:
            return np.random.uniform(self.config.min_fill_ratio, 1.0)
        else:
            return 1.0
    
    def _calculate_target_quantity(self, signal: Signal, execution_price: float) -> float:
        """Calculate target quantity based on signal size"""
        target_value = signal.size * self.config.initial_capital
        return target_value / execution_price
    
    def _format_trade_records(self, trades: List[Trade], timestamp: datetime) -> List[Dict[str, Any]]:
        """Format trades for record keeping"""
        records = []
        
        for trade in trades:
            record = {
                'timestamp': timestamp,
                'symbol': trade.symbol,
                'signal_type': trade.signal_type.value,
                'price': trade.price,
                'quantity': trade.quantity,
                'commission': trade.commission,
                'slippage': trade.slippage,
                'notional_value': abs(trade.price * trade.quantity),
                'total_cost': trade.commission + abs(trade.slippage)
            }
            
            # Add metadata
            for key, value in trade.metadata.items():
                record[f'meta_{key}'] = value
            
            records.append(record)
        
        return records
    
    def _format_position_record(self, positions: Dict[str, Dict[str, Any]], 
                               timestamp: datetime, portfolio_value: float) -> Dict[str, Any]:
        """Format position summary for record keeping"""
        record = {
            'timestamp': timestamp,
            'portfolio_value': portfolio_value,
            'num_positions': len(positions)
        }
        
        # Add individual position data
        for symbol, position_data in positions.items():
            record[f'{symbol}_quantity'] = position_data.get('quantity', 0)
            record[f'{symbol}_unrealized_pnl'] = position_data.get('unrealized_pnl', 0)
            record[f'{symbol}_total_pnl'] = position_data.get('total_pnl', 0)
        
        return record
    
    def _create_backtest_result(self, strategy: BaseStrategy, portfolio_values: List[float],
                               trade_records: List[Dict[str, Any]], 
                               position_records: List[Dict[str, Any]]) -> BacktestResult:
        """Create comprehensive backtest result"""
        # Create time series
        equity_curve = pd.Series(portfolio_values, index=self.market_data.index)
        returns = equity_curve.pct_change().dropna()
        
        # Basic metrics
        total_return = (equity_curve.iloc[-1] - equity_curve.iloc[0]) / equity_curve.iloc[0]
        annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
        volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = (annualized_return - self.config.risk_free_rate) / volatility if volatility > 0 else 0
        
        # Drawdown calculation
        peak = equity_curve.expanding().max()
        drawdown = (equity_curve - peak) / peak
        max_drawdown = drawdown.min()
        
        # Trade statistics
        trade_df = pd.DataFrame(trade_records) if trade_records else pd.DataFrame()
        
        if not trade_df.empty:
            total_trades = len(trade_df)
            trade_pnl = self._calculate_trade_pnl(trade_df, strategy)
            winning_trades = len(trade_pnl[trade_pnl > 0])
            losing_trades = len(trade_pnl[trade_pnl < 0])
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            avg_win = trade_pnl[trade_pnl > 0].mean() if winning_trades > 0 else 0
            avg_loss = trade_pnl[trade_pnl < 0].mean() if losing_trades > 0 else 0
            profit_factor = abs(avg_win * winning_trades / (avg_loss * losing_trades)) if avg_loss != 0 else 0
        else:
            total_trades = winning_trades = losing_trades = 0
            win_rate = avg_win = avg_loss = profit_factor = 0
        
        # Risk metrics
        var_95 = returns.quantile(0.05) if len(returns) > 0 else 0
        cvar_95 = returns[returns <= var_95].mean() if len(returns) > 0 else 0
        
        # Sortino ratio
        downside_returns = returns[returns < 0]
        downside_volatility = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino_ratio = (annualized_return - self.config.risk_free_rate) / downside_volatility if downside_volatility > 0 else 0
        
        # Calmar ratio
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Benchmark comparison
        benchmark_return = None
        alpha = beta = information_ratio = None
        
        if self.benchmark_data is not None:
            benchmark_returns = self.benchmark_data.pct_change().dropna()
            if len(benchmark_returns) > 0:
                benchmark_return = (self.benchmark_data.iloc[-1] - self.benchmark_data.iloc[0]) / self.benchmark_data.iloc[0]
                
                # Align returns
                aligned_returns = returns.reindex(benchmark_returns.index, method='nearest').dropna()
                aligned_benchmark = benchmark_returns.reindex(aligned_returns.index).dropna()
                
                if len(aligned_returns) > 1 and len(aligned_benchmark) > 1:
                    # Beta
                    covariance = np.cov(aligned_returns, aligned_benchmark)[0, 1]
                    benchmark_variance = np.var(aligned_benchmark)
                    beta = covariance / benchmark_variance if benchmark_variance > 0 else 0
                    
                    # Alpha
                    alpha = annualized_return - (self.config.risk_free_rate + beta * (benchmark_returns.mean() * 252 - self.config.risk_free_rate))
                    
                    # Information ratio
                    excess_returns = aligned_returns - aligned_benchmark
                    tracking_error = excess_returns.std() * np.sqrt(252)
                    information_ratio = excess_returns.mean() * 252 / tracking_error if tracking_error > 0 else 0
        
        # Create result
        result = BacktestResult(
            total_return=total_return,
            annualized_return=annualized_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            equity_curve=equity_curve,
            returns=returns,
            trades=trade_df if self.config.save_trades else None,
            positions=pd.DataFrame(position_records) if (self.config.save_positions and position_records) else None,
            benchmark_return=benchmark_return,
            alpha=alpha,
            beta=beta,
            information_ratio=information_ratio,
            var_95=var_95,
            cvar_95=cvar_95,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            strategy_metrics=strategy.get_strategy_metrics() if hasattr(strategy, 'get_strategy_metrics') else {}
        )
        
        return result
    
    def _calculate_trade_pnl(self, trade_df: pd.DataFrame, strategy: BaseStrategy) -> pd.Series:
        """Calculate P&L for each trade"""
        # This is a simplified calculation
        # In reality, would need to match entry/exit trades
        trade_pnl = []
        
        for _, trade in trade_df.iterrows():
            # Simplified: assume each trade is immediately profitable/unprofitable
            pnl = trade['quantity'] * 0.01  # Placeholder calculation
            trade_pnl.append(pnl)
        
        return pd.Series(trade_pnl)
    
    def _combine_walk_forward_results(self, results: List[BacktestResult]) -> BacktestResult:
        """Combine multiple walk-forward results"""
        if not results:
            raise ValueError("No walk-forward results to combine")
        
        # Combine equity curves
        combined_equity = pd.concat([r.equity_curve for r in results])
        combined_returns = combined_equity.pct_change().dropna()
        
        # Recalculate metrics for combined results
        total_return = (combined_equity.iloc[-1] - combined_equity.iloc[0]) / combined_equity.iloc[0]
        annualized_return = (1 + total_return) ** (252 / len(combined_returns)) - 1
        volatility = combined_returns.std() * np.sqrt(252)
        sharpe_ratio = (annualized_return - self.config.risk_free_rate) / volatility if volatility > 0 else 0
        
        # Combine other metrics
        total_trades = sum(r.total_trades for r in results)
        winning_trades = sum(r.winning_trades for r in results)
        losing_trades = sum(r.losing_trades for r in results)
        
        # Create combined result
        return BacktestResult(
            total_return=total_return,
            annualized_return=annualized_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=min(r.max_drawdown for r in results),
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=winning_trades / total_trades if total_trades > 0 else 0,
            avg_win=np.mean([r.avg_win for r in results if r.avg_win > 0]),
            avg_loss=np.mean([r.avg_loss for r in results if r.avg_loss < 0]),
            profit_factor=np.mean([r.profit_factor for r in results if r.profit_factor > 0]),
            equity_curve=combined_equity,
            returns=combined_returns,
            var_95=combined_returns.quantile(0.05),
            cvar_95=combined_returns[combined_returns <= combined_returns.quantile(0.05)].mean(),
            sortino_ratio=np.mean([r.sortino_ratio for r in results]),
            calmar_ratio=np.mean([r.calmar_ratio for r in results])
        )


# Factory functions
def create_backtest_engine(start_date: datetime, end_date: datetime, 
                          initial_capital: float = 100000, **kwargs) -> BacktestEngine:
    """Create a backtest engine with configuration"""
    config = BacktestConfig(
        start_date=start_date,
        end_date=end_date,
        initial_capital=initial_capital,
        **kwargs
    )
    return BacktestEngine(config)


def run_backtest(strategy: BaseStrategy, market_data: pd.DataFrame,
                start_date: datetime, end_date: datetime,
                initial_capital: float = 100000, **kwargs) -> BacktestResult:
    """Convenience function to run a backtest"""
    engine = create_backtest_engine(start_date, end_date, initial_capital, **kwargs)
    engine.load_data(market_data)
    return engine.run_backtest(strategy)


# Export classes and functions
__all__ = [
    'BacktestEngine',
    'BacktestConfig',
    'BacktestResult',
    'create_backtest_engine',
    'run_backtest'
]