"""
Multi-Agent Trading System
Collaborative and competitive RL agents for portfolio management
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import warnings
from collections import deque
import random
from enum import Enum

from .ppo_agent import PPOAgent, PPOConfig
from .sac_agent import SACAgent, SACConfig  
from .rainbow_dqn import RainbowDQNAgent, RainbowConfig
from utils.logger import get_logger

logger = get_logger(__name__)


class AgentType(Enum):
    """Types of agents in the multi-agent system"""
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    TREND_FOLLOWING = "trend_following"
    RISK_PARITY = "risk_parity"
    MARKET_MAKING = "market_making"
    ARBITRAGE = "arbitrage"


class CommunicationMode(Enum):
    """Communication modes between agents"""
    NONE = "none"
    BROADCAST = "broadcast"
    SELECTIVE = "selective"
    HIERARCHICAL = "hierarchical"


@dataclass
class MultiAgentConfig:
    """Multi-agent system configuration"""
    # Agent composition
    agent_types: List[AgentType] = None
    num_agents: int = 4
    
    # Communication
    communication_mode: CommunicationMode = CommunicationMode.SELECTIVE
    communication_frequency: int = 10  # Steps between communications
    message_dim: int = 32
    
    # Coordination
    use_central_critic: bool = True
    use_parameter_sharing: bool = False
    use_attention_mechanism: bool = True
    
    # Portfolio allocation
    dynamic_allocation: bool = True
    min_agent_allocation: float = 0.05  # Minimum 5% allocation per agent
    max_agent_allocation: float = 0.4   # Maximum 40% allocation per agent
    
    # Training
    coordination_reward_weight: float = 0.2
    diversity_reward_weight: float = 0.1
    competition_weight: float = 0.3
    
    # Environment
    lookback_window: int = 50
    action_dim: int = 5  # Per agent action dimension
    
    def __post_init__(self):
        if self.agent_types is None:
            self.agent_types = [
                AgentType.MOMENTUM,
                AgentType.MEAN_REVERSION, 
                AgentType.TREND_FOLLOWING,
                AgentType.RISK_PARITY
            ]


class CommunicationNetwork(nn.Module):
    """Neural network for agent communication"""
    
    def __init__(self, agent_dim: int, message_dim: int, num_agents: int):
        super().__init__()
        
        self.agent_dim = agent_dim
        self.message_dim = message_dim
        self.num_agents = num_agents
        
        # Message encoding
        self.message_encoder = nn.Sequential(
            nn.Linear(agent_dim, message_dim),
            nn.ReLU(),
            nn.Linear(message_dim, message_dim)
        )
        
        # Attention mechanism for selective communication
        self.attention = nn.MultiheadAttention(
            embed_dim=message_dim,
            num_heads=4,
            batch_first=True
        )
        
        # Message integration
        self.message_integrator = nn.Sequential(
            nn.Linear(message_dim + agent_dim, agent_dim),
            nn.ReLU(),
            nn.Linear(agent_dim, agent_dim)
        )
    
    def forward(self, agent_states: torch.Tensor, 
                communication_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for communication
        
        Args:
            agent_states: (batch_size, num_agents, agent_dim)
            communication_mask: (batch_size, num_agents, num_agents)
        
        Returns:
            Updated agent states: (batch_size, num_agents, agent_dim)
        """
        batch_size, num_agents, agent_dim = agent_states.shape
        
        # Encode messages
        messages = self.message_encoder(agent_states)  # (batch_size, num_agents, message_dim)
        
        # Apply attention for selective communication
        attended_messages, attention_weights = self.attention(
            messages, messages, messages,
            attn_mask=communication_mask
        )
        
        # Integrate messages with agent states
        combined = torch.cat([agent_states, attended_messages], dim=-1)
        updated_states = self.message_integrator(combined)
        
        return updated_states


class CentralCritic(nn.Module):
    """Centralized critic for multi-agent training"""
    
    def __init__(self, total_obs_dim: int, total_action_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        self.critic = nn.Sequential(
            nn.Linear(total_obs_dim + total_action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, joint_obs: torch.Tensor, joint_actions: torch.Tensor) -> torch.Tensor:
        """Forward pass for centralized value function"""
        combined = torch.cat([joint_obs, joint_actions], dim=-1)
        return self.critic(combined)


class SpecializedAgent:
    """Wrapper for specialized trading agents"""
    
    def __init__(self, agent_type: AgentType, base_agent: Union[PPOAgent, SACAgent, RainbowDQNAgent],
                 obs_dim: int, action_dim: int):
        self.agent_type = agent_type
        self.base_agent = base_agent
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        # Specialization parameters
        self.specialization_params = self._get_specialization_params()
        
        # Performance tracking
        self.performance_history = deque(maxlen=1000)
        self.allocation_history = deque(maxlen=1000)
        
    def _get_specialization_params(self) -> Dict[str, Any]:
        """Get specialization parameters based on agent type"""
        params = {}
        
        if self.agent_type == AgentType.MOMENTUM:
            params = {
                'lookback_preference': 20,
                'trend_threshold': 0.02,
                'momentum_decay': 0.95
            }
        elif self.agent_type == AgentType.MEAN_REVERSION:
            params = {
                'lookback_preference': 50,
                'reversion_threshold': 2.0,  # Standard deviations
                'mean_window': 20
            }
        elif self.agent_type == AgentType.TREND_FOLLOWING:
            params = {
                'lookback_preference': 100,
                'trend_confirmation': 5,  # Days
                'stop_loss_threshold': 0.05
            }
        elif self.agent_type == AgentType.RISK_PARITY:
            params = {
                'risk_target': 0.1,  # Target volatility
                'rebalance_frequency': 20,
                'correlation_threshold': 0.7
            }
        elif self.agent_type == AgentType.MARKET_MAKING:
            params = {
                'spread_target': 0.001,
                'inventory_limit': 0.1,
                'order_size': 0.01
            }
        elif self.agent_type == AgentType.ARBITRAGE:
            params = {
                'price_threshold': 0.0005,
                'execution_speed': 1,  # Priority level
                'risk_limit': 0.05
            }
        
        return params
    
    def preprocess_observation(self, obs: np.ndarray, market_data: Dict[str, Any]) -> np.ndarray:
        """Preprocess observations based on agent specialization"""
        processed_obs = obs.copy()
        
        if self.agent_type == AgentType.MOMENTUM:
            # Add momentum indicators
            returns = market_data.get('returns', np.zeros(len(obs)))
            momentum = np.convolve(returns, np.ones(self.specialization_params['lookback_preference']), mode='valid')
            if len(momentum) > 0:
                processed_obs = np.concatenate([processed_obs, [momentum[-1]]])
        
        elif self.agent_type == AgentType.MEAN_REVERSION:
            # Add mean reversion indicators
            prices = market_data.get('prices', np.zeros(len(obs)))
            if len(prices) >= self.specialization_params['mean_window']:
                mean_price = np.mean(prices[-self.specialization_params['mean_window']:])
                std_price = np.std(prices[-self.specialization_params['mean_window']:])
                z_score = (prices[-1] - mean_price) / (std_price + 1e-8)
                processed_obs = np.concatenate([processed_obs, [z_score]])
        
        elif self.agent_type == AgentType.RISK_PARITY:
            # Add risk metrics
            volatilities = market_data.get('volatilities', {})
            correlations = market_data.get('correlations', {})
            if volatilities:
                avg_vol = np.mean(list(volatilities.values()))
                processed_obs = np.concatenate([processed_obs, [avg_vol]])
        
        return processed_obs
    
    def get_action(self, obs: np.ndarray, market_data: Dict[str, Any] = None, 
                   deterministic: bool = False) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Get action from specialized agent"""
        if market_data is None:
            market_data = {}
            
        # Preprocess observation
        processed_obs = self.preprocess_observation(obs, market_data)
        
        # Get action from base agent
        action, info = self.base_agent.get_action(processed_obs, deterministic)
        
        # Apply specialization modifications
        action = self._apply_specialization(action, market_data)
        
        # Update info
        info['agent_type'] = self.agent_type.value
        info['specialization_params'] = self.specialization_params
        
        return action, info
    
    def _apply_specialization(self, action: np.ndarray, market_data: Dict[str, Any]) -> np.ndarray:
        """Apply agent-specific modifications to actions"""
        modified_action = action.copy()
        
        if self.agent_type == AgentType.MOMENTUM:
            # Amplify actions in trending markets
            momentum = market_data.get('momentum', 0)
            if abs(momentum) > self.specialization_params['trend_threshold']:
                modified_action *= (1 + abs(momentum))
        
        elif self.agent_type == AgentType.MEAN_REVERSION:
            # Reverse actions when prices deviate from mean
            z_score = market_data.get('z_score', 0)
            if abs(z_score) > self.specialization_params['reversion_threshold']:
                modified_action *= -np.sign(z_score)
        
        elif self.agent_type == AgentType.RISK_PARITY:
            # Scale actions based on volatility
            volatility = market_data.get('volatility', 0.02)
            target_vol = self.specialization_params['risk_target']
            vol_adjustment = target_vol / (volatility + 1e-8)
            modified_action *= vol_adjustment
        
        return modified_action
    
    def update_performance(self, reward: float, allocation: float):
        """Update performance tracking"""
        self.performance_history.append(reward)
        self.allocation_history.append(allocation)
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get performance metrics for allocation decisions"""
        if len(self.performance_history) == 0:
            return {'avg_reward': 0.0, 'volatility': 0.0, 'sharpe': 0.0}
        
        rewards = np.array(self.performance_history)
        avg_reward = np.mean(rewards)
        volatility = np.std(rewards)
        sharpe = avg_reward / (volatility + 1e-8)
        
        return {
            'avg_reward': avg_reward,
            'volatility': volatility,
            'sharpe': sharpe,
            'recent_performance': np.mean(rewards[-20:]) if len(rewards) >= 20 else avg_reward
        }


class MultiAgentTradingSystem:
    """Multi-agent reinforcement learning trading system"""
    
    def __init__(self, obs_dim: int, action_dim: int, config: MultiAgentConfig = None):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.config = config or MultiAgentConfig()
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize agents
        self.agents = self._create_specialized_agents()
        
        # Communication network
        if self.config.communication_mode != CommunicationMode.NONE:
            self.communication_network = CommunicationNetwork(
                obs_dim, self.config.message_dim, self.config.num_agents
            ).to(self.device)
            self.comm_optimizer = optim.Adam(self.communication_network.parameters(), lr=1e-4)
        
        # Central critic
        if self.config.use_central_critic:
            total_obs_dim = obs_dim * self.config.num_agents
            total_action_dim = action_dim * self.config.num_agents
            self.central_critic = CentralCritic(total_obs_dim, total_action_dim).to(self.device)
            self.critic_optimizer = optim.Adam(self.central_critic.parameters(), lr=1e-3)
        
        # Portfolio allocation manager
        self.allocation_manager = self._create_allocation_manager()
        
        # Training state
        self.total_steps = 0
        self.episodes = 0
        self.communication_step = 0
        
        # Metrics
        self.training_metrics = {
            'total_reward': [],
            'agent_rewards': {i: [] for i in range(self.config.num_agents)},
            'allocations': {i: [] for i in range(self.config.num_agents)},
            'communication_frequency': [],
            'coordination_reward': []
        }
    
    def _create_specialized_agents(self) -> List[SpecializedAgent]:
        """Create specialized trading agents"""
        agents = []
        
        for i, agent_type in enumerate(self.config.agent_types[:self.config.num_agents]):
            # Create base agent (alternate between different RL algorithms)
            if i % 3 == 0:
                base_agent = PPOAgent(self.obs_dim, self.action_dim, PPOConfig(), discrete=False)
            elif i % 3 == 1:
                base_agent = SACAgent(self.obs_dim, self.action_dim, SACConfig())
            else:
                base_agent = RainbowDQNAgent(self.obs_dim, self.action_dim, RainbowConfig())
            
            # Wrap in specialized agent
            specialized_agent = SpecializedAgent(agent_type, base_agent, self.obs_dim, self.action_dim)
            agents.append(specialized_agent)
        
        return agents
    
    def _create_allocation_manager(self):
        """Create portfolio allocation manager"""
        class AllocationManager:
            def __init__(self, config: MultiAgentConfig):
                self.config = config
                self.allocations = np.ones(config.num_agents) / config.num_agents
                self.performance_weights = np.ones(config.num_agents)
            
            def update_allocations(self, agent_performances: List[Dict[str, float]]) -> np.ndarray:
                """Update portfolio allocations based on agent performance"""
                if not self.config.dynamic_allocation:
                    return self.allocations
                
                # Calculate performance-based weights
                sharpe_ratios = np.array([perf.get('sharpe', 0) for perf in agent_performances])
                recent_performances = np.array([perf.get('recent_performance', 0) for perf in agent_performances])
                
                # Combine Sharpe ratio and recent performance
                combined_scores = 0.7 * sharpe_ratios + 0.3 * recent_performances
                
                # Apply softmax for allocation
                exp_scores = np.exp(combined_scores - np.max(combined_scores))
                new_allocations = exp_scores / np.sum(exp_scores)
                
                # Apply constraints
                new_allocations = np.clip(
                    new_allocations, 
                    self.config.min_agent_allocation,
                    self.config.max_agent_allocation
                )
                
                # Renormalize
                new_allocations = new_allocations / np.sum(new_allocations)
                
                # Smooth transition
                self.allocations = 0.8 * self.allocations + 0.2 * new_allocations
                
                return self.allocations
        
        return AllocationManager(self.config)
    
    def get_action(self, obs: np.ndarray, market_data: Dict[str, Any] = None, 
                   deterministic: bool = False) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Get actions from all agents"""
        if market_data is None:
            market_data = {}
        
        agent_actions = []
        agent_infos = []
        
        # Get individual agent actions
        for agent in self.agents:
            action, info = agent.get_action(obs, market_data, deterministic)
            agent_actions.append(action)
            agent_infos.append(info)
        
        # Communication step
        if (self.config.communication_mode != CommunicationMode.NONE and 
            self.total_steps % self.config.communication_frequency == 0):
            agent_actions = self._communicate_and_coordinate(agent_actions, obs)
        
        # Aggregate actions using current allocations
        allocations = self.allocation_manager.allocations
        aggregated_action = np.zeros_like(agent_actions[0])
        
        for i, (action, allocation) in enumerate(zip(agent_actions, allocations)):
            aggregated_action += allocation * action
        
        # Combined info
        combined_info = {
            'agent_actions': agent_actions,
            'agent_infos': agent_infos,
            'allocations': allocations.copy(),
            'communication_step': self.communication_step
        }
        
        return aggregated_action, combined_info
    
    def _communicate_and_coordinate(self, agent_actions: List[np.ndarray], 
                                   obs: np.ndarray) -> List[np.ndarray]:
        """Perform communication and coordination between agents"""
        if self.config.communication_mode == CommunicationMode.NONE:
            return agent_actions
        
        # Convert to tensors
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).unsqueeze(0).repeat(1, len(self.agents), 1).to(self.device)
        
        # Apply communication network
        if hasattr(self, 'communication_network'):
            with torch.no_grad():
                updated_states = self.communication_network(obs_tensor)
            
            # Update agent actions based on communication
            coordinated_actions = []
            for i, (agent, action) in enumerate(zip(self.agents, agent_actions)):
                # Simple coordination: blend individual action with communication signal
                comm_signal = updated_states[0, i].cpu().numpy()[:len(action)]
                coordinated_action = 0.8 * action + 0.2 * comm_signal
                coordinated_actions.append(coordinated_action)
            
            self.communication_step += 1
            return coordinated_actions
        
        return agent_actions
    
    def store_transition(self, obs: np.ndarray, action: np.ndarray, reward: float,
                        next_obs: np.ndarray, done: bool, info: Dict[str, Any]):
        """Store transitions for all agents"""
        agent_actions = info.get('agent_actions', [action] * len(self.agents))
        allocations = info.get('allocations', self.allocation_manager.allocations)
        
        # Calculate individual agent rewards
        agent_rewards = self._calculate_agent_rewards(reward, agent_actions, allocations)
        
        # Store transitions for each agent
        for i, (agent, agent_action, agent_reward) in enumerate(zip(self.agents, agent_actions, agent_rewards)):
            agent.base_agent.store_transition(obs, agent_action, agent_reward, next_obs, done, info)
            agent.update_performance(agent_reward, allocations[i])
        
        self.total_steps += 1
    
    def _calculate_agent_rewards(self, total_reward: float, agent_actions: List[np.ndarray],
                                allocations: np.ndarray) -> List[float]:
        """Calculate individual rewards for each agent"""
        base_rewards = [total_reward * allocation for allocation in allocations]
        
        # Add coordination reward
        if self.config.coordination_reward_weight > 0:
            coordination_reward = self._calculate_coordination_reward(agent_actions)
            for i in range(len(base_rewards)):
                base_rewards[i] += self.config.coordination_reward_weight * coordination_reward
        
        # Add diversity reward
        if self.config.diversity_reward_weight > 0:
            diversity_reward = self._calculate_diversity_reward(agent_actions)
            for i in range(len(base_rewards)):
                base_rewards[i] += self.config.diversity_reward_weight * diversity_reward
        
        # Add competition component
        if self.config.competition_weight > 0:
            performance_metrics = [agent.get_performance_metrics() for agent in self.agents]
            relative_performance = self._calculate_relative_performance(performance_metrics)
            
            for i in range(len(base_rewards)):
                competition_bonus = self.config.competition_weight * relative_performance[i] * total_reward
                base_rewards[i] += competition_bonus
        
        return base_rewards
    
    def _calculate_coordination_reward(self, agent_actions: List[np.ndarray]) -> float:
        """Calculate reward for agent coordination"""
        if len(agent_actions) < 2:
            return 0.0
        
        # Measure action alignment
        action_correlations = []
        for i in range(len(agent_actions)):
            for j in range(i + 1, len(agent_actions)):
                correlation = np.corrcoef(agent_actions[i].flatten(), agent_actions[j].flatten())[0, 1]
                if not np.isnan(correlation):
                    action_correlations.append(abs(correlation))
        
        return np.mean(action_correlations) if action_correlations else 0.0
    
    def _calculate_diversity_reward(self, agent_actions: List[np.ndarray]) -> float:
        """Calculate reward for maintaining action diversity"""
        if len(agent_actions) < 2:
            return 0.0
        
        # Measure action diversity (negative correlation is good for diversity)
        action_distances = []
        for i in range(len(agent_actions)):
            for j in range(i + 1, len(agent_actions)):
                distance = np.linalg.norm(agent_actions[i] - agent_actions[j])
                action_distances.append(distance)
        
        return np.mean(action_distances) if action_distances else 0.0
    
    def _calculate_relative_performance(self, performance_metrics: List[Dict[str, float]]) -> List[float]:
        """Calculate relative performance scores for competition"""
        sharpe_ratios = [metrics.get('sharpe', 0) for metrics in performance_metrics]
        
        if len(set(sharpe_ratios)) <= 1:  # All same performance
            return [0.0] * len(sharpe_ratios)
        
        # Rank-based relative performance
        ranked_indices = np.argsort(sharpe_ratios)
        relative_scores = np.zeros(len(sharpe_ratios))
        
        for rank, idx in enumerate(ranked_indices):
            relative_scores[idx] = (rank - len(sharpe_ratios) / 2) / len(sharpe_ratios)
        
        return relative_scores.tolist()
    
    def update(self) -> Dict[str, float]:
        """Update all agents and coordination mechanisms"""
        metrics = {}
        
        # Update individual agents
        agent_metrics = []
        for i, agent in enumerate(self.agents):
            agent_metric = agent.base_agent.update()
            agent_metrics.append(agent_metric)
            
            # Store agent-specific metrics
            if agent_metric:
                for key, value in agent_metric.items():
                    metrics[f'agent_{i}_{key}'] = value
        
        # Update allocations based on performance
        performance_metrics = [agent.get_performance_metrics() for agent in self.agents]
        new_allocations = self.allocation_manager.update_allocations(performance_metrics)
        
        # Update communication network
        if (hasattr(self, 'communication_network') and 
            self.config.communication_mode != CommunicationMode.NONE):
            comm_loss = self._update_communication_network()
            metrics['communication_loss'] = comm_loss
        
        # Update central critic
        if hasattr(self, 'central_critic'):
            critic_loss = self._update_central_critic()
            metrics['critic_loss'] = critic_loss
        
        # Store system-level metrics
        total_reward = sum(sum(agent.performance_history) for agent in self.agents if agent.performance_history)
        metrics['total_system_reward'] = total_reward
        metrics['allocation_entropy'] = -np.sum(new_allocations * np.log(new_allocations + 1e-8))
        
        # Update training metrics
        self.training_metrics['total_reward'].append(total_reward)
        for i, allocation in enumerate(new_allocations):
            self.training_metrics['allocations'][i].append(allocation)
        
        return metrics
    
    def _update_communication_network(self) -> float:
        """Update communication network (placeholder)"""
        # This would involve training the communication network
        # based on coordination rewards and system performance
        return 0.0
    
    def _update_central_critic(self) -> float:
        """Update central critic (placeholder)"""
        # This would involve training the centralized value function
        # for better coordination between agents
        return 0.0
    
    def save(self, filepath: str):
        """Save multi-agent system state"""
        checkpoint = {
            'config': self.config,
            'agents': [],
            'allocation_manager_state': {
                'allocations': self.allocation_manager.allocations,
                'performance_weights': self.allocation_manager.performance_weights
            },
            'total_steps': self.total_steps,
            'episodes': self.episodes,
            'training_metrics': self.training_metrics
        }
        
        # Save individual agents
        for i, agent in enumerate(self.agents):
            agent_filepath = filepath.replace('.pt', f'_agent_{i}.pt')
            agent.base_agent.save(agent_filepath)
            checkpoint['agents'].append({
                'agent_type': agent.agent_type.value,
                'filepath': agent_filepath,
                'specialization_params': agent.specialization_params
            })
        
        # Save communication network
        if hasattr(self, 'communication_network'):
            checkpoint['communication_network_state_dict'] = self.communication_network.state_dict()
        
        # Save central critic
        if hasattr(self, 'central_critic'):
            checkpoint['central_critic_state_dict'] = self.central_critic.state_dict()
        
        torch.save(checkpoint, filepath)
        logger.info(f"Multi-agent system saved to {filepath}")
    
    def load(self, filepath: str):
        """Load multi-agent system state"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # Load agents
        for i, (agent, agent_info) in enumerate(zip(self.agents, checkpoint['agents'])):
            agent_filepath = agent_info['filepath']
            agent.base_agent.load(agent_filepath)
            agent.specialization_params = agent_info['specialization_params']
        
        # Load allocation manager
        allocation_state = checkpoint['allocation_manager_state']
        self.allocation_manager.allocations = allocation_state['allocations']
        self.allocation_manager.performance_weights = allocation_state['performance_weights']
        
        # Load communication network
        if hasattr(self, 'communication_network') and 'communication_network_state_dict' in checkpoint:
            self.communication_network.load_state_dict(checkpoint['communication_network_state_dict'])
        
        # Load central critic
        if hasattr(self, 'central_critic') and 'central_critic_state_dict' in checkpoint:
            self.central_critic.load_state_dict(checkpoint['central_critic_state_dict'])
        
        self.total_steps = checkpoint.get('total_steps', 0)
        self.episodes = checkpoint.get('episodes', 0)
        self.training_metrics = checkpoint.get('training_metrics', self.training_metrics)
        
        logger.info(f"Multi-agent system loaded from {filepath}")
    
    def get_training_metrics(self) -> Dict[str, Any]:
        """Get training metrics"""
        return self.training_metrics.copy()
    
    def reset_episode(self):
        """Reset for new episode"""
        self.episodes += 1
        for agent in self.agents:
            agent.base_agent.reset_episode()


# Factory function
def create_multi_agent_system(obs_dim: int, action_dim: int, 
                             config: Dict[str, Any] = None) -> MultiAgentTradingSystem:
    """Create multi-agent trading system with configuration"""
    if config:
        multi_agent_config = MultiAgentConfig(**config)
    else:
        multi_agent_config = MultiAgentConfig()
    
    return MultiAgentTradingSystem(obs_dim, action_dim, multi_agent_config)


# Export classes and functions
__all__ = [
    'MultiAgentTradingSystem',
    'MultiAgentConfig',
    'SpecializedAgent',
    'CommunicationNetwork',
    'CentralCritic',
    'AgentType',
    'CommunicationMode',
    'create_multi_agent_system'
]