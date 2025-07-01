"""
Reinforcement Learning Module for Trading
Advanced RL algorithms and environments for automated trading
"""

from .trading_env import (
    TradingEnvironment, TradingEnvConfig, MarketState, Position, Trade,
    ActionSpace, ActionType, OrderType, TransactionCostModel, PortfolioManager,
    create_trading_env, generate_sample_market_data
)

from .reward_functions import (
    RewardFunction, RewardConfig, RewardType, SimpleReturnReward, SharpeReward,
    SortinoReward, MaxDrawdownPenalty, TradingFrequencyPenalty, PortfolioDiversityReward,
    RiskParityReward, ValueAtRiskPenalty, CompositeReward, AdaptiveReward,
    create_reward_function
)

# Import all agent modules
from .agents import (
    PPOAgent, PPOConfig, ActorCriticNetwork, PPOBuffer, create_ppo_agent,
    SACAgent, SACConfig, PolicyNetwork, QNetwork, ReplayBuffer, create_sac_agent,
    RainbowDQNAgent, RainbowConfig, DuelingDistributionalNetwork, NoisyLinear,
    PrioritizedReplayBuffer, create_rainbow_dqn_agent,
    MultiAgentTradingSystem, MultiAgentConfig, SpecializedAgent, CommunicationNetwork,
    CentralCritic, AgentType, CommunicationMode, create_multi_agent_system
)

# Import distributed training components
from .distributed_training import (
    DistributedConfig, TrainingMetrics, DistributedWorker, ParameterServer,
    DistributedTrainingManager, create_distributed_trainer, run_distributed_training
)

# Import hyperparameter optimization
from .hyperparameter_optimization import (
    HyperparameterSpace, OptimizationConfig, OptimizationResult,
    HyperparameterOptimizer, create_hyperparameter_optimizer, optimize_agent_hyperparameters
)

# Import monitoring system
from .monitoring import (
    MonitoringConfig, MetricSnapshot, RealTimeMonitor, 
    create_monitor, start_monitoring_dashboard
)


# Factory functions for easy agent creation
def create_agent(agent_type: str, obs_dim: int, action_dim: int, config: dict = None):
    """
    Factory function to create RL agents
    
    Args:
        agent_type: 'ppo', 'sac', 'rainbow_dqn', or 'multi_agent'
        obs_dim: Observation space dimension
        action_dim: Action space dimension  
        config: Agent-specific configuration dictionary
    
    Returns:
        Configured RL agent
    """
    if agent_type.lower() == 'ppo':
        return create_ppo_agent(obs_dim, action_dim, config, discrete=False)
    elif agent_type.lower() == 'sac':
        return create_sac_agent(obs_dim, action_dim, config)
    elif agent_type.lower() == 'rainbow_dqn':
        return create_rainbow_dqn_agent(obs_dim, action_dim, config)
    elif agent_type.lower() == 'multi_agent':
        return create_multi_agent_system(obs_dim, action_dim, config)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")


def create_trading_system(env_config: dict = None, agent_config: dict = None, 
                         reward_config: dict = None, agent_type: str = 'ppo'):
    """
    Factory function to create complete trading system
    
    Args:
        env_config: Environment configuration
        agent_config: Agent configuration  
        reward_config: Reward function configuration
        agent_type: Type of RL agent to use
    
    Returns:
        Tuple of (environment, agent, reward_function)
    """
    # Create environment
    env = create_trading_env(env_config)
    
    # Create reward function
    reward_function = create_reward_function('composite', reward_config)
    
    # Create agent
    obs_dim = env.observation_space.shape[0]
    if agent_type == 'rainbow_dqn':
        action_dim = env.action_space.n if hasattr(env.action_space, 'n') else env.action_space.shape[0]
    else:
        action_dim = env.action_space.shape[0] if hasattr(env.action_space, 'shape') else env.action_space.n
    
    agent = create_agent(agent_type, obs_dim, action_dim, agent_config)
    
    return env, agent, reward_function


# Export all classes and functions
__all__ = [
    # Environment components
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
    'generate_sample_market_data',
    
    # Reward functions
    'RewardFunction',
    'RewardConfig',
    'RewardType',
    'SimpleReturnReward',
    'SharpeReward',
    'SortinoReward', 
    'MaxDrawdownPenalty',
    'TradingFrequencyPenalty',
    'PortfolioDiversityReward',
    'RiskParityReward',
    'ValueAtRiskPenalty',
    'CompositeReward',
    'AdaptiveReward',
    'create_reward_function',
    
    # PPO Agent
    'PPOAgent',
    'PPOConfig',
    'ActorCriticNetwork',
    'PPOBuffer',
    'create_ppo_agent',
    
    # SAC Agent  
    'SACAgent',
    'SACConfig',
    'PolicyNetwork',
    'QNetwork',
    'ReplayBuffer',
    'create_sac_agent',
    
    # Rainbow DQN Agent
    'RainbowDQNAgent',
    'RainbowConfig',
    'DuelingDistributionalNetwork',
    'NoisyLinear',
    'PrioritizedReplayBuffer', 
    'create_rainbow_dqn_agent',
    
    # Multi-Agent System
    'MultiAgentTradingSystem',
    'MultiAgentConfig',
    'SpecializedAgent',
    'CommunicationNetwork',
    'CentralCritic',
    'AgentType',
    'CommunicationMode',
    'create_multi_agent_system',
    
    # Distributed Training
    'DistributedConfig',
    'TrainingMetrics',
    'DistributedWorker',
    'ParameterServer',
    'DistributedTrainingManager',
    'create_distributed_trainer',
    'run_distributed_training',
    
    # Hyperparameter Optimization
    'HyperparameterSpace',
    'OptimizationConfig',
    'OptimizationResult',
    'HyperparameterOptimizer',
    'create_hyperparameter_optimizer',
    'optimize_agent_hyperparameters',
    
    # Monitoring System
    'MonitoringConfig',
    'MetricSnapshot',
    'RealTimeMonitor',
    'create_monitor',
    'start_monitoring_dashboard',
    
    # Factory functions
    'create_agent',
    'create_trading_system'
]