"""
OpenAI Gym Compatible Trading Environment
Realistic trading simulation with transaction costs, slippage, and multi-asset support
"""

import numpy as np
import pandas as pd
try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    import gym
    from gym import spaces
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import warnings
from datetime import datetime, timedelta
from collections import deque
import random

from utils.logger import get_logger

logger = get_logger(__name__)
warnings.filterwarnings('ignore', category=FutureWarning)


class ActionType(Enum):
    """Trading action types"""
    HOLD = 0
    BUY = 1
    SELL = 2
    CLOSE_LONG = 3
    CLOSE_SHORT = 4


class OrderType(Enum):
    """Order types"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


@dataclass
class TradingEnvConfig:
    """Trading environment configuration"""
    # Market data
    initial_balance: float = 100000.0
    max_position_size: float = 0.3  # Max 30% of portfolio per asset
    leverage: float = 1.0
    
    # Transaction costs
    commission_rate: float = 0.001  # 0.1% commission
    spread_rate: float = 0.0005  # 0.05% bid-ask spread
    slippage_rate: float = 0.0002  # 0.02% slippage
    
    # Market impact
    market_impact_coef: float = 0.0001
    liquidity_threshold: float = 0.01
    
    # Environment settings
    lookback_window: int = 50
    max_steps: int = 1000
    observation_features: List[str] = None
    
    # Risk management
    max_drawdown: float = 0.2  # 20% max drawdown
    margin_call_threshold: float = 0.1
    
    # Rewards
    reward_scaling: float = 1.0
    risk_free_rate: float = 0.02  # 2% annual risk-free rate
    
    # Multi-asset
    assets: List[str] = None
    correlation_penalty: float = 0.1
    
    def __post_init__(self):
        if self.observation_features is None:
            self.observation_features = [
                'open', 'high', 'low', 'close', 'volume',
                'returns', 'volatility', 'rsi', 'macd', 'bb_position'
            ]
        
        if self.assets is None:
            self.assets = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']


@dataclass
class MarketState:
    """Current market state"""
    timestamp: datetime
    prices: Dict[str, float]
    volumes: Dict[str, float]
    features: Dict[str, np.ndarray]
    market_open: bool = True
    volatility: Dict[str, float] = None
    
    def __post_init__(self):
        if self.volatility is None:
            self.volatility = {asset: 0.02 for asset in self.prices.keys()}


@dataclass
class Position:
    """Trading position"""
    asset: str
    quantity: float
    entry_price: float
    timestamp: datetime
    position_type: str = 'long'  # 'long' or 'short'
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0


@dataclass
class Trade:
    """Executed trade"""
    asset: str
    action: ActionType
    quantity: float
    price: float
    commission: float
    slippage: float
    timestamp: datetime
    order_type: OrderType = OrderType.MARKET


class ActionSpace:
    """Custom action space for multi-asset trading"""
    
    def __init__(self, assets: List[str], discrete: bool = True):
        self.assets = assets
        self.n_assets = len(assets)
        self.discrete = discrete
        
        if discrete:
            # Discrete actions: [HOLD, BUY_WEAK, BUY_STRONG, SELL_WEAK, SELL_STRONG] for each asset
            self.n_actions_per_asset = 5
            self.n = self.n_assets * self.n_actions_per_asset
            self.space = spaces.Discrete(self.n)
        else:
            # Continuous actions: target allocation for each asset [-1, 1]
            self.space = spaces.Box(
                low=-1.0, high=1.0, 
                shape=(self.n_assets,), 
                dtype=np.float32
            )
    
    def sample(self):
        """Sample random action"""
        return self.space.sample()
    
    def decode_action(self, action):
        """Decode action to asset-specific actions"""
        if self.discrete:
            actions = {}
            for i, asset in enumerate(self.assets):
                asset_action = (action // (self.n_actions_per_asset ** i)) % self.n_actions_per_asset
                actions[asset] = asset_action
            return actions
        else:
            return {asset: action[i] for i, asset in enumerate(self.assets)}


class TransactionCostModel:
    """Realistic transaction cost modeling"""
    
    def __init__(self, config: TradingEnvConfig):
        self.config = config
        
    def calculate_costs(self, asset: str, quantity: float, price: float, 
                       market_state: MarketState, portfolio_value: float) -> Dict[str, float]:
        """Calculate transaction costs"""
        notional_value = abs(quantity * price)
        
        # Base commission
        commission = notional_value * self.config.commission_rate
        
        # Bid-ask spread
        spread_cost = notional_value * self.config.spread_rate
        
        # Market impact (depends on order size relative to typical volume)
        volume = market_state.volumes.get(asset, 1000000)
        volume_ratio = notional_value / (volume * price)
        market_impact = notional_value * self.config.market_impact_coef * np.sqrt(volume_ratio)
        
        # Slippage (random component)
        base_slippage = notional_value * self.config.slippage_rate
        volatility = market_state.volatility.get(asset, 0.02)
        random_slippage = base_slippage * (1 + volatility * np.random.normal(0, 1))
        slippage = max(0, random_slippage)
        
        return {
            'commission': commission,
            'spread': spread_cost,
            'market_impact': market_impact,
            'slippage': slippage,
            'total': commission + spread_cost + market_impact + slippage
        }
    
    def get_execution_price(self, asset: str, quantity: float, market_price: float,
                          market_state: MarketState) -> float:
        """Get realistic execution price including slippage"""
        volatility = market_state.volatility.get(asset, 0.02)
        
        # Slippage direction based on trade direction
        slippage_direction = 1 if quantity > 0 else -1
        
        # Slippage magnitude
        base_slippage = self.config.slippage_rate
        volume_impact = min(0.01, abs(quantity) * market_price / market_state.volumes.get(asset, 1000000))
        total_slippage = (base_slippage + volume_impact) * slippage_direction
        
        # Add random component
        random_component = volatility * np.random.normal(0, 0.1)
        
        execution_price = market_price * (1 + total_slippage + random_component)
        return max(0.01, execution_price)  # Prevent negative prices


class PortfolioManager:
    """Manage portfolio positions and calculations"""
    
    def __init__(self, config: TradingEnvConfig):
        self.config = config
        self.positions: Dict[str, Position] = {}
        self.cash = config.initial_balance
        self.trades: List[Trade] = []
        self.transaction_costs = 0.0
        
    def get_portfolio_value(self, market_state: MarketState) -> float:
        """Calculate total portfolio value"""
        total_value = self.cash
        
        for asset, position in self.positions.items():
            current_price = market_state.prices.get(asset, position.entry_price)
            position_value = position.quantity * current_price
            total_value += position_value
            
        return total_value
    
    def get_asset_allocation(self, market_state: MarketState) -> Dict[str, float]:
        """Get current asset allocation percentages"""
        portfolio_value = self.get_portfolio_value(market_state)
        allocations = {}
        
        for asset in self.config.assets:
            if asset in self.positions:
                position = self.positions[asset]
                current_price = market_state.prices.get(asset, position.entry_price)
                asset_value = position.quantity * current_price
                allocations[asset] = asset_value / portfolio_value if portfolio_value > 0 else 0.0
            else:
                allocations[asset] = 0.0
                
        allocations['cash'] = self.cash / portfolio_value if portfolio_value > 0 else 1.0
        return allocations
    
    def execute_trade(self, asset: str, action: ActionType, quantity: float,
                     market_state: MarketState, cost_model: TransactionCostModel) -> bool:
        """Execute a trade with realistic costs"""
        if quantity == 0:
            return True
            
        market_price = market_state.prices[asset]
        execution_price = cost_model.get_execution_price(asset, quantity, market_price, market_state)
        
        # Calculate costs
        portfolio_value = self.get_portfolio_value(market_state)
        costs = cost_model.calculate_costs(asset, quantity, execution_price, market_state, portfolio_value)
        
        notional_value = abs(quantity * execution_price)
        total_cost = costs['total']
        
        # Check if we have enough cash for the trade
        if quantity > 0:  # Buying
            required_cash = notional_value + total_cost
            if required_cash > self.cash:
                # Partial fill based on available cash
                max_quantity = (self.cash - total_cost) / execution_price
                if max_quantity <= 0:
                    return False
                quantity = max(0, max_quantity)
                notional_value = quantity * execution_price
                costs = cost_model.calculate_costs(asset, quantity, execution_price, market_state, portfolio_value)
                total_cost = costs['total']
        
        # Execute the trade
        if asset in self.positions:
            position = self.positions[asset]
            
            if (position.quantity > 0 and quantity > 0) or (position.quantity < 0 and quantity < 0):
                # Adding to existing position
                total_quantity = position.quantity + quantity
                weighted_price = ((position.quantity * position.entry_price) + 
                                (quantity * execution_price)) / total_quantity
                position.quantity = total_quantity
                position.entry_price = weighted_price
            else:
                # Reducing or closing position
                if abs(quantity) >= abs(position.quantity):
                    # Closing position completely
                    realized_pnl = position.quantity * (execution_price - position.entry_price)
                    position.realized_pnl += realized_pnl
                    
                    remaining_quantity = quantity + position.quantity
                    if abs(remaining_quantity) > 1e-6:
                        # Opening new position in opposite direction
                        position.quantity = remaining_quantity
                        position.entry_price = execution_price
                        position.position_type = 'long' if remaining_quantity > 0 else 'short'
                    else:
                        # Remove position
                        del self.positions[asset]
                else:
                    # Partial close
                    realized_pnl = -quantity * (execution_price - position.entry_price)
                    position.realized_pnl += realized_pnl
                    position.quantity += quantity
        else:
            # New position
            self.positions[asset] = Position(
                asset=asset,
                quantity=quantity,
                entry_price=execution_price,
                timestamp=market_state.timestamp,
                position_type='long' if quantity > 0 else 'short'
            )
        
        # Update cash
        self.cash -= quantity * execution_price + total_cost
        self.transaction_costs += total_cost
        
        # Record trade
        trade = Trade(
            asset=asset,
            action=action,
            quantity=quantity,
            price=execution_price,
            commission=costs['commission'],
            slippage=costs['slippage'],
            timestamp=market_state.timestamp
        )
        self.trades.append(trade)
        
        return True
    
    def get_unrealized_pnl(self, market_state: MarketState) -> Dict[str, float]:
        """Calculate unrealized P&L for all positions"""
        unrealized_pnl = {}
        
        for asset, position in self.positions.items():
            current_price = market_state.prices.get(asset, position.entry_price)
            pnl = position.quantity * (current_price - position.entry_price)
            unrealized_pnl[asset] = pnl
            position.unrealized_pnl = pnl
            
        return unrealized_pnl
    
    def get_total_pnl(self, market_state: MarketState) -> float:
        """Get total P&L (realized + unrealized)"""
        total_realized = sum(pos.realized_pnl for pos in self.positions.values())
        total_unrealized = sum(self.get_unrealized_pnl(market_state).values())
        return total_realized + total_unrealized


class TradingEnvironment(gym.Env):
    """OpenAI Gym compatible trading environment"""
    
    def __init__(self, config: TradingEnvConfig = None, data: pd.DataFrame = None):
        super().__init__()
        
        self.config = config or TradingEnvConfig()
        self.data = data
        self.current_step = 0
        self.max_steps = self.config.max_steps
        
        # Initialize components
        self.portfolio_manager = PortfolioManager(self.config)
        self.cost_model = TransactionCostModel(self.config)
        
        # Action and observation spaces
        self.action_space_handler = ActionSpace(self.config.assets, discrete=True)
        self.action_space = self.action_space_handler.space
        
        # Observation space
        n_features = len(self.config.observation_features)
        n_assets = len(self.config.assets)
        n_portfolio_features = n_assets + 3  # allocations + cash + total_value + drawdown
        
        obs_dim = (n_features * n_assets * self.config.lookback_window + 
                  n_portfolio_features)
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(obs_dim,), dtype=np.float32
        )
        
        # State tracking
        self.initial_portfolio_value = self.config.initial_balance
        self.peak_portfolio_value = self.config.initial_balance
        self.episode_trades = []
        self.episode_returns = []
        
        # Initialize data
        if self.data is not None:
            self._prepare_data()
    
    def _prepare_data(self):
        """Prepare market data for training"""
        # Ensure we have all required features
        for asset in self.config.assets:
            if asset not in self.data.columns:
                logger.warning(f"Asset {asset} not found in data")
                continue
                
        # Calculate technical features if not present
        for feature in self.config.observation_features:
            if feature not in self.data.columns:
                self._calculate_feature(feature)
        
        # Normalize features
        self.feature_means = {}
        self.feature_stds = {}
        
        for feature in self.config.observation_features:
            if feature in self.data.columns:
                self.feature_means[feature] = self.data[feature].mean()
                self.feature_stds[feature] = self.data[feature].std()
    
    def _calculate_feature(self, feature: str):
        """Calculate missing technical features"""
        if feature == 'returns':
            for asset in self.config.assets:
                if f'{asset}_close' in self.data.columns:
                    self.data[f'{asset}_{feature}'] = self.data[f'{asset}_close'].pct_change()
        
        elif feature == 'volatility':
            for asset in self.config.assets:
                if f'{asset}_returns' in self.data.columns:
                    self.data[f'{asset}_{feature}'] = self.data[f'{asset}_returns'].rolling(20).std()
        
        elif feature == 'rsi':
            for asset in self.config.assets:
                if f'{asset}_close' in self.data.columns:
                    self.data[f'{asset}_{feature}'] = self._calculate_rsi(self.data[f'{asset}_close'])
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def reset(self, seed=None, options=None):
        """Reset environment for new episode"""
        super().reset(seed=seed)
        
        # Reset portfolio
        self.portfolio_manager = PortfolioManager(self.config)
        self.current_step = self.config.lookback_window
        self.initial_portfolio_value = self.config.initial_balance
        self.peak_portfolio_value = self.config.initial_balance
        self.episode_trades = []
        self.episode_returns = []
        
        # Random start position if data is available
        if self.data is not None and len(self.data) > self.max_steps + self.config.lookback_window:
            max_start = len(self.data) - self.max_steps - self.config.lookback_window
            self.current_step = np.random.randint(self.config.lookback_window, max_start)
        
        return self._get_observation(), {}
    
    def step(self, action):
        """Execute one step in the environment"""
        # Decode action
        asset_actions = self.action_space_handler.decode_action(action)
        
        # Get current market state
        market_state = self._get_market_state()
        
        # Execute trades based on actions
        executed_trades = self._execute_actions(asset_actions, market_state)
        
        # Calculate reward
        reward = self._calculate_reward(market_state, executed_trades)
        
        # Update state
        self.current_step += 1
        
        # Check termination conditions
        done = self._is_done(market_state)
        
        # Get new observation
        obs = self._get_observation()
        
        # Info dictionary
        info = {
            'portfolio_value': self.portfolio_manager.get_portfolio_value(market_state),
            'cash': self.portfolio_manager.cash,
            'positions': len(self.portfolio_manager.positions),
            'total_trades': len(self.portfolio_manager.trades),
            'transaction_costs': self.portfolio_manager.transaction_costs,
            'executed_trades': executed_trades
        }
        
        return obs, reward, done, False, info
    
    def _get_market_state(self) -> MarketState:
        """Get current market state"""
        if self.data is not None:
            current_row = self.data.iloc[self.current_step]
            
            prices = {}
            volumes = {}
            features = {feature: [] for feature in self.config.observation_features}
            
            for asset in self.config.assets:
                if f'{asset}_close' in current_row:
                    prices[asset] = current_row[f'{asset}_close']
                else:
                    # Generate synthetic price if not available
                    prices[asset] = 100.0 * (1 + np.random.normal(0, 0.02))
                
                if f'{asset}_volume' in current_row:
                    volumes[asset] = current_row[f'{asset}_volume']
                else:
                    volumes[asset] = 1000000  # Default volume
                
                for feature in self.config.observation_features:
                    feature_col = f'{asset}_{feature}'
                    if feature_col in current_row:
                        features[feature].append(current_row[feature_col])
                    else:
                        features[feature].append(0.0)
            
            return MarketState(
                timestamp=datetime.now(),
                prices=prices,
                volumes=volumes,
                features=features
            )
        else:
            # Generate synthetic market state
            return self._generate_synthetic_market_state()
    
    def _generate_synthetic_market_state(self) -> MarketState:
        """Generate synthetic market data for testing"""
        prices = {}
        volumes = {}
        features = {feature: [] for feature in self.config.observation_features}
        
        for asset in self.config.assets:
            # Generate random price
            base_price = 100.0
            price_change = np.random.normal(0, 0.02)
            prices[asset] = base_price * (1 + price_change)
            volumes[asset] = np.random.randint(500000, 2000000)
            
            # Generate random features
            for feature in self.config.observation_features:
                if feature == 'returns':
                    features[feature].append(price_change)
                elif feature == 'rsi':
                    features[feature].append(np.random.uniform(20, 80))
                elif feature == 'volatility':
                    features[feature].append(np.random.uniform(0.01, 0.05))
                else:
                    features[feature].append(np.random.normal(0, 1))
        
        return MarketState(
            timestamp=datetime.now(),
            prices=prices,
            volumes=volumes,
            features=features
        )
    
    def _execute_actions(self, asset_actions: Dict[str, int], 
                        market_state: MarketState) -> List[Trade]:
        """Execute trading actions"""
        executed_trades = []
        portfolio_value = self.portfolio_manager.get_portfolio_value(market_state)
        
        for asset, action in asset_actions.items():
            if asset not in market_state.prices:
                continue
                
            current_price = market_state.prices[asset]
            
            # Convert discrete action to trading quantity
            if action == 0:  # HOLD
                continue
            elif action == 1:  # BUY_WEAK (5% of portfolio)
                target_value = portfolio_value * 0.05
                quantity = target_value / current_price
            elif action == 2:  # BUY_STRONG (15% of portfolio)
                target_value = portfolio_value * 0.15
                quantity = target_value / current_price
            elif action == 3:  # SELL_WEAK (sell 25% of position)
                if asset in self.portfolio_manager.positions:
                    current_quantity = self.portfolio_manager.positions[asset].quantity
                    quantity = -current_quantity * 0.25
                else:
                    continue
            elif action == 4:  # SELL_STRONG (sell 75% of position)
                if asset in self.portfolio_manager.positions:
                    current_quantity = self.portfolio_manager.positions[asset].quantity
                    quantity = -current_quantity * 0.75
                else:
                    continue
            else:
                continue
            
            # Execute trade
            if abs(quantity) > 1e-6:  # Minimum trade size
                success = self.portfolio_manager.execute_trade(
                    asset, ActionType.BUY if quantity > 0 else ActionType.SELL,
                    quantity, market_state, self.cost_model
                )
                
                if success and self.portfolio_manager.trades:
                    executed_trades.append(self.portfolio_manager.trades[-1])
        
        return executed_trades
    
    def _calculate_reward(self, market_state: MarketState, executed_trades: List[Trade]) -> float:
        """Calculate reward for current step"""
        # Portfolio return
        current_value = self.portfolio_manager.get_portfolio_value(market_state)
        portfolio_return = (current_value - self.initial_portfolio_value) / self.initial_portfolio_value
        
        # Update peak value for drawdown calculation
        self.peak_portfolio_value = max(self.peak_portfolio_value, current_value)
        drawdown = (self.peak_portfolio_value - current_value) / self.peak_portfolio_value
        
        # Base reward: portfolio return
        reward = portfolio_return * self.config.reward_scaling
        
        # Penalty for drawdown
        reward -= drawdown * 2.0
        
        # Penalty for excessive trading
        trade_penalty = len(executed_trades) * 0.01
        reward -= trade_penalty
        
        # Penalty for transaction costs
        cost_penalty = sum(trade.commission + trade.slippage for trade in executed_trades)
        cost_penalty /= current_value  # Normalize by portfolio value
        reward -= cost_penalty * 10.0
        
        return reward
    
    def _is_done(self, market_state: MarketState) -> bool:
        """Check if episode should terminate"""
        # Max steps reached
        if self.current_step >= self.max_steps:
            return True
        
        # Maximum drawdown exceeded
        current_value = self.portfolio_manager.get_portfolio_value(market_state)
        drawdown = (self.peak_portfolio_value - current_value) / self.peak_portfolio_value
        
        if drawdown > self.config.max_drawdown:
            return True
        
        # Portfolio value too low
        if current_value < self.initial_portfolio_value * 0.1:
            return True
        
        return False
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation"""
        obs_components = []
        
        # Market features for lookback window
        if self.data is not None:
            start_idx = max(0, self.current_step - self.config.lookback_window)
            end_idx = self.current_step
            
            for feature in self.config.observation_features:
                for asset in self.config.assets:
                    feature_col = f'{asset}_{feature}'
                    if feature_col in self.data.columns:
                        feature_values = self.data[feature_col].iloc[start_idx:end_idx].values
                        
                        # Pad if necessary
                        if len(feature_values) < self.config.lookback_window:
                            padding = np.zeros(self.config.lookback_window - len(feature_values))
                            feature_values = np.concatenate([padding, feature_values])
                        
                        # Normalize
                        if feature in self.feature_means:
                            feature_values = (feature_values - self.feature_means[feature]) / (self.feature_stds[feature] + 1e-8)
                        
                        obs_components.extend(feature_values)
                    else:
                        # Fill with zeros if feature not available
                        obs_components.extend(np.zeros(self.config.lookback_window))
        else:
            # Synthetic observations
            total_features = len(self.config.observation_features) * len(self.config.assets) * self.config.lookback_window
            obs_components.extend(np.random.normal(0, 1, total_features))
        
        # Portfolio state
        market_state = self._get_market_state()
        allocations = self.portfolio_manager.get_asset_allocation(market_state)
        
        # Asset allocations
        for asset in self.config.assets:
            obs_components.append(allocations.get(asset, 0.0))
        
        # Cash allocation
        obs_components.append(allocations.get('cash', 0.0))
        
        # Portfolio value (normalized)
        current_value = self.portfolio_manager.get_portfolio_value(market_state)
        normalized_value = current_value / self.initial_portfolio_value
        obs_components.append(normalized_value)
        
        # Current drawdown
        drawdown = (self.peak_portfolio_value - current_value) / self.peak_portfolio_value
        obs_components.append(drawdown)
        
        return np.array(obs_components, dtype=np.float32)
    
    def render(self, mode='human'):
        """Render environment state"""
        market_state = self._get_market_state()
        portfolio_value = self.portfolio_manager.get_portfolio_value(market_state)
        allocations = self.portfolio_manager.get_asset_allocation(market_state)
        
        print(f"\n=== Trading Environment State ===")
        print(f"Step: {self.current_step}")
        print(f"Portfolio Value: ${portfolio_value:,.2f}")
        print(f"Cash: ${self.portfolio_manager.cash:,.2f}")
        print(f"Total Trades: {len(self.portfolio_manager.trades)}")
        print(f"Transaction Costs: ${self.portfolio_manager.transaction_costs:,.2f}")
        
        print(f"\nAsset Allocations:")
        for asset, allocation in allocations.items():
            print(f"  {asset}: {allocation:.2%}")
        
        if self.portfolio_manager.positions:
            print(f"\nActive Positions:")
            for asset, position in self.portfolio_manager.positions.items():
                current_price = market_state.prices.get(asset, position.entry_price)
                unrealized_pnl = position.quantity * (current_price - position.entry_price)
                print(f"  {asset}: {position.quantity:.2f} @ ${position.entry_price:.2f} "
                      f"(Current: ${current_price:.2f}, PnL: ${unrealized_pnl:,.2f})")


# Factory function
def create_trading_env(config: Dict[str, Any] = None, data: pd.DataFrame = None) -> TradingEnvironment:
    """Create trading environment with configuration"""
    if config:
        env_config = TradingEnvConfig(**config)
    else:
        env_config = TradingEnvConfig()
    
    return TradingEnvironment(env_config, data)


# Generate sample market data for testing
def generate_sample_market_data(n_days: int = 1000, assets: List[str] = None) -> pd.DataFrame:
    """Generate sample market data for testing"""
    if assets is None:
        assets = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', periods=n_days, freq='D')
    
    data = {'date': dates}
    
    for asset in assets:
        # Generate price series with realistic patterns
        base_price = np.random.uniform(50, 200)
        returns = np.random.normal(0.0005, 0.02, n_days)
        
        # Add some trend
        trend = np.linspace(0, 0.5, n_days) * np.random.choice([-1, 1])
        returns += trend / n_days
        
        # Generate OHLCV data
        prices = base_price * np.exp(np.cumsum(returns))
        
        data[f'{asset}_open'] = prices * np.random.uniform(0.995, 1.005, n_days)
        data[f'{asset}_high'] = prices * np.random.uniform(1.0, 1.02, n_days)
        data[f'{asset}_low'] = prices * np.random.uniform(0.98, 1.0, n_days)
        data[f'{asset}_close'] = prices
        data[f'{asset}_volume'] = np.random.randint(1000000, 10000000, n_days)
        
        # Technical indicators
        data[f'{asset}_returns'] = np.concatenate([[0], np.diff(prices) / prices[:-1]])
        data[f'{asset}_volatility'] = pd.Series(data[f'{asset}_returns']).rolling(20).std().fillna(0.02)
        
        # RSI
        close_series = pd.Series(data[f'{asset}_close'])
        delta = close_series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data[f'{asset}_rsi'] = (100 - (100 / (1 + rs))).fillna(50)
        
        # MACD (simplified)
        ema_12 = close_series.ewm(span=12).mean()
        ema_26 = close_series.ewm(span=26).mean()
        data[f'{asset}_macd'] = ema_12 - ema_26
        
        # Bollinger Bands position
        sma_20 = close_series.rolling(20).mean()
        std_20 = close_series.rolling(20).std()
        data[f'{asset}_bb_position'] = ((close_series - sma_20) / (2 * std_20)).fillna(0)
    
    return pd.DataFrame(data)


# Export classes and functions
__all__ = [
    'TradingEnvironment',
    'TradingEnvConfig',
    'MarketState',
    'Position',
    'Trade',
    'ActionSpace',
    'ActionType',
    'OrderType',
    'TransactionCostModel',
    'PortfolioManager',
    'create_trading_env',
    'generate_sample_market_data'
]