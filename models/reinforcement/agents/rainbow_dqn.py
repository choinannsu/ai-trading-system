"""
Rainbow DQN Agent for Trading
Advanced DQN with distributional RL, prioritized replay, dueling networks, and multi-step learning
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import warnings
from collections import deque, namedtuple
import random
import math

from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class RainbowConfig:
    """Rainbow DQN agent configuration"""
    # Network architecture
    hidden_sizes: List[int] = None
    activation: str = 'relu'
    
    # Rainbow components
    use_dueling: bool = True
    use_noisy_networks: bool = True
    use_distributional: bool = True
    use_multi_step: bool = True
    use_prioritized_replay: bool = True
    use_double_dqn: bool = True
    
    # DQN hyperparameters
    learning_rate: float = 6.25e-5
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: int = 1000000
    
    # Multi-step learning
    n_step: int = 3
    
    # Distributional RL
    v_min: float = -10.0
    v_max: float = 10.0
    n_atoms: int = 51
    
    # Prioritized Experience Replay
    alpha: float = 0.5  # PER alpha parameter
    beta_start: float = 0.4  # PER beta parameter
    beta_frames: int = 100000
    
    # Training settings
    batch_size: int = 32
    buffer_size: int = 1000000
    min_buffer_size: int = 80000
    target_update_frequency: int = 8000
    train_frequency: int = 4
    
    # Noisy networks
    noisy_std: float = 0.1
    
    # Financial specific
    action_scaling: float = 1.0
    risk_penalty_coef: float = 0.1
    
    def __post_init__(self):
        if self.hidden_sizes is None:
            self.hidden_sizes = [512, 512]


class NoisyLinear(nn.Module):
    """Noisy linear layer for exploration"""
    
    def __init__(self, in_features: int, out_features: int, std_init: float = 0.1):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        
        # Learnable parameters
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        
        # Register noise buffers
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))
        
        self.reset_parameters()
        self.reset_noise()
    
    def reset_parameters(self):
        """Initialize parameters"""
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))
    
    def _scale_noise(self, size: int) -> torch.Tensor:
        """Generate scaled noise"""
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())
    
    def reset_noise(self):
        """Reset noise"""
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass with noise"""
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        
        return F.linear(input, weight, bias)


class DuelingDistributionalNetwork(nn.Module):
    """Dueling distributional Q-network for Rainbow DQN"""
    
    def __init__(self, obs_dim: int, action_dim: int, config: RainbowConfig):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.config = config
        self.n_atoms = config.n_atoms
        
        # Value range
        self.v_min = config.v_min
        self.v_max = config.v_max
        self.delta_z = (config.v_max - config.v_min) / (config.n_atoms - 1)
        self.register_buffer('support', torch.linspace(config.v_min, config.v_max, config.n_atoms))
        
        # Shared feature layers
        self.shared_layers = self._build_shared_network()
        
        # Dueling architecture
        if config.use_dueling:
            # Value stream
            if config.use_noisy_networks:
                self.value_hidden = NoisyLinear(config.hidden_sizes[-1], config.hidden_sizes[-1], config.noisy_std)
                self.value_out = NoisyLinear(config.hidden_sizes[-1], config.n_atoms, config.noisy_std)
            else:
                self.value_hidden = nn.Linear(config.hidden_sizes[-1], config.hidden_sizes[-1])
                self.value_out = nn.Linear(config.hidden_sizes[-1], config.n_atoms)
            
            # Advantage stream
            if config.use_noisy_networks:
                self.advantage_hidden = NoisyLinear(config.hidden_sizes[-1], config.hidden_sizes[-1], config.noisy_std)
                self.advantage_out = NoisyLinear(config.hidden_sizes[-1], action_dim * config.n_atoms, config.noisy_std)
            else:
                self.advantage_hidden = nn.Linear(config.hidden_sizes[-1], config.hidden_sizes[-1])
                self.advantage_out = nn.Linear(config.hidden_sizes[-1], action_dim * config.n_atoms)
        else:
            # Standard DQN head
            if config.use_noisy_networks:
                self.q_out = NoisyLinear(config.hidden_sizes[-1], action_dim * config.n_atoms, config.noisy_std)
            else:
                self.q_out = nn.Linear(config.hidden_sizes[-1], action_dim * config.n_atoms)
        
        # Initialize weights
        self._init_weights()
    
    def _build_shared_network(self) -> nn.Module:
        """Build shared feature extraction network"""
        layers = []
        input_size = self.obs_dim
        
        for hidden_size in self.config.hidden_sizes:
            if self.config.use_noisy_networks:
                layers.append(NoisyLinear(input_size, hidden_size, self.config.noisy_std))
            else:
                layers.append(nn.Linear(input_size, hidden_size))
            
            if self.config.activation == 'relu':
                layers.append(nn.ReLU())
            elif self.config.activation == 'tanh':
                layers.append(nn.Tanh())
            elif self.config.activation == 'elu':
                layers.append(nn.ELU())
            
            input_size = hidden_size
        
        return nn.Sequential(*layers)
    
    def _init_weights(self):
        """Initialize network weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        features = self.shared_layers(obs)
        
        if self.config.use_dueling:
            # Value stream
            value_hidden = F.relu(self.value_hidden(features))
            value = self.value_out(value_hidden)  # (batch_size, n_atoms)
            
            # Advantage stream
            advantage_hidden = F.relu(self.advantage_hidden(features))
            advantage = self.advantage_out(advantage_hidden)  # (batch_size, action_dim * n_atoms)
            advantage = advantage.view(-1, self.action_dim, self.n_atoms)
            
            # Combine value and advantage
            value = value.unsqueeze(1)  # (batch_size, 1, n_atoms)
            q_atoms = value + advantage - advantage.mean(dim=1, keepdim=True)
        else:
            # Standard DQN
            q_atoms = self.q_out(features)
            q_atoms = q_atoms.view(-1, self.action_dim, self.n_atoms)
        
        # Apply softmax to get probability distributions
        q_dist = F.softmax(q_atoms, dim=-1)
        q_dist = q_dist.clamp(min=1e-3)  # For numerical stability
        
        return q_dist
    
    def reset_noise(self):
        """Reset noise in noisy layers"""
        if self.config.use_noisy_networks:
            for module in self.modules():
                if isinstance(module, NoisyLinear):
                    module.reset_noise()
    
    def get_q_values(self, obs: torch.Tensor) -> torch.Tensor:
        """Get Q-values from probability distributions"""
        q_dist = self.forward(obs)
        q_values = torch.sum(q_dist * self.support, dim=-1)
        return q_values


class PrioritizedReplayBuffer:
    """Prioritized Experience Replay buffer"""
    
    def __init__(self, obs_dim: int, buffer_size: int, alpha: float = 0.6):
        self.obs_dim = obs_dim
        self.buffer_size = buffer_size
        self.alpha = alpha
        
        # Buffers
        self.observations = np.zeros((buffer_size, obs_dim), dtype=np.float32)
        self.actions = np.zeros(buffer_size, dtype=np.int64)
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.next_observations = np.zeros((buffer_size, obs_dim), dtype=np.float32)
        self.dones = np.zeros(buffer_size, dtype=np.bool_)
        
        # Priority tree
        self.priorities = np.zeros(buffer_size, dtype=np.float32)
        self.ptr = 0
        self.size = 0
        
        # For multi-step learning
        self.n_step_buffer = deque(maxlen=10)
    
    def store(self, obs: np.ndarray, action: int, reward: float, 
             next_obs: np.ndarray, done: bool, priority: float = None):
        """Store experience with priority"""
        idx = self.ptr % self.buffer_size
        
        self.observations[idx] = obs
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.next_observations[idx] = next_obs
        self.dones[idx] = done
        
        # Set priority
        if priority is None:
            max_priority = self.priorities[:self.size].max() if self.size > 0 else 1.0
            self.priorities[idx] = max_priority
        else:
            self.priorities[idx] = priority
        
        self.ptr += 1
        self.size = min(self.size + 1, self.buffer_size)
    
    def sample(self, batch_size: int, beta: float = 0.4) -> Dict[str, torch.Tensor]:
        """Sample batch with importance sampling"""
        if self.size < batch_size:
            indices = np.arange(self.size)
        else:
            # Sample based on priorities
            priorities = self.priorities[:self.size] + 1e-6
            probs = priorities ** self.alpha
            probs /= probs.sum()
            
            indices = np.random.choice(self.size, batch_size, p=probs, replace=False)
        
        # Importance sampling weights
        weights = (self.size * probs[indices]) ** (-beta) if self.size >= batch_size else np.ones(len(indices))
        weights /= weights.max()
        
        return {
            'observations': torch.FloatTensor(self.observations[indices]),
            'actions': torch.LongTensor(self.actions[indices]),
            'rewards': torch.FloatTensor(self.rewards[indices]),
            'next_observations': torch.FloatTensor(self.next_observations[indices]),
            'dones': torch.FloatTensor(self.dones[indices]),
            'weights': torch.FloatTensor(weights),
            'indices': indices
        }
    
    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """Update priorities for sampled experiences"""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
    
    def store_n_step(self, obs: np.ndarray, action: int, reward: float, 
                     next_obs: np.ndarray, done: bool, n_step: int, gamma: float):
        """Store experience with n-step returns"""
        self.n_step_buffer.append((obs, action, reward, next_obs, done))
        
        if len(self.n_step_buffer) >= n_step:
            # Calculate n-step return
            n_step_reward = 0
            for i, (_, _, r, _, d) in enumerate(self.n_step_buffer):
                n_step_reward += (gamma ** i) * r
                if d:
                    break
            
            # Get initial state and final next state
            initial_obs, initial_action = self.n_step_buffer[0][:2]
            final_next_obs, final_done = self.n_step_buffer[-1][3:]
            
            # Store n-step experience
            self.store(initial_obs, initial_action, n_step_reward, final_next_obs, final_done)


class RainbowDQNAgent:
    """Rainbow DQN agent for trading"""
    
    def __init__(self, obs_dim: int, action_dim: int, config: RainbowConfig = None):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.config = config or RainbowConfig()
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Networks
        self.q_network = DuelingDistributionalNetwork(obs_dim, action_dim, self.config).to(self.device)
        self.target_network = DuelingDistributionalNetwork(obs_dim, action_dim, self.config).to(self.device)
        
        # Copy parameters to target network
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.config.learning_rate)
        
        # Experience replay
        if self.config.use_prioritized_replay:
            self.replay_buffer = PrioritizedReplayBuffer(obs_dim, self.config.buffer_size, self.config.alpha)
        else:
            self.replay_buffer = self._create_simple_buffer()
        
        # Training state
        self.total_steps = 0
        self.episodes = 0
        self.updates = 0
        
        # Exploration
        self.epsilon = self.config.epsilon_start
        
        # Metrics
        self.training_metrics = {
            'loss': [],
            'q_values': [],
            'epsilon': [],
            'priority_weight': []
        }
    
    def _create_simple_buffer(self):
        """Create simple replay buffer if prioritized replay is disabled"""
        Experience = namedtuple('Experience', ['obs', 'action', 'reward', 'next_obs', 'done'])
        
        class SimpleBuffer:
            def __init__(self, buffer_size):
                self.buffer = deque(maxlen=buffer_size)
            
            def store(self, obs, action, reward, next_obs, done, **kwargs):
                self.buffer.append(Experience(obs, action, reward, next_obs, done))
            
            def sample(self, batch_size, **kwargs):
                batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
                
                return {
                    'observations': torch.FloatTensor([e.obs for e in batch]),
                    'actions': torch.LongTensor([e.action for e in batch]),
                    'rewards': torch.FloatTensor([e.reward for e in batch]),
                    'next_observations': torch.FloatTensor([e.next_obs for e in batch]),
                    'dones': torch.FloatTensor([e.done for e in batch]),
                    'weights': torch.ones(len(batch)),
                    'indices': np.arange(len(batch))
                }
            
            def update_priorities(self, indices, priorities):
                pass
            
            @property
            def size(self):
                return len(self.buffer)
        
        return SimpleBuffer(self.config.buffer_size)
    
    def get_action(self, obs: np.ndarray, deterministic: bool = False) -> Tuple[int, Dict[str, Any]]:
        """Get action from agent"""
        if not deterministic and np.random.random() < self.epsilon:
            action = np.random.randint(self.action_dim)
            info = {'epsilon': self.epsilon, 'random_action': True}
        else:
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                q_values = self.q_network.get_q_values(obs_tensor)
                action = q_values.argmax(dim=1).item()
            
            info = {
                'epsilon': self.epsilon, 
                'random_action': False,
                'q_values': q_values.cpu().numpy().flatten() if 'q_values' in locals() else None
            }
        
        return action, info
    
    def store_transition(self, obs: np.ndarray, action: int, reward: float,
                        next_obs: np.ndarray, done: bool, info: Dict[str, Any]):
        """Store transition in replay buffer"""
        if self.config.use_multi_step:
            self.replay_buffer.store_n_step(
                obs, action, reward, next_obs, done, 
                self.config.n_step, self.config.gamma
            )
        else:
            self.replay_buffer.store(obs, action, reward, next_obs, done)
        
        self.total_steps += 1
        
        # Decay epsilon
        self.epsilon = max(
            self.config.epsilon_end,
            self.config.epsilon_start - (self.total_steps / self.config.epsilon_decay) * 
            (self.config.epsilon_start - self.config.epsilon_end)
        )
    
    def update(self) -> Dict[str, float]:
        """Update agent using Rainbow DQN algorithm"""
        if self.replay_buffer.size < self.config.min_buffer_size:
            return {}
        
        metrics = {}
        
        # Update frequency
        if self.total_steps % self.config.train_frequency == 0:
            # Sample batch
            if self.config.use_prioritized_replay:
                beta = min(1.0, self.config.beta_start + 
                          (1.0 - self.config.beta_start) * self.total_steps / self.config.beta_frames)
                batch = self.replay_buffer.sample(self.config.batch_size, beta)
            else:
                batch = self.replay_buffer.sample(self.config.batch_size)
            
            for key in batch:
                if key != 'indices':
                    batch[key] = batch[key].to(self.device)
            
            # Compute loss
            loss, td_errors = self._compute_loss(batch)
            
            # Update network
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 10.0)
            self.optimizer.step()
            
            # Reset noise
            if self.config.use_noisy_networks:
                self.q_network.reset_noise()
                self.target_network.reset_noise()
            
            # Update priorities
            if self.config.use_prioritized_replay:
                priorities = td_errors.detach().cpu().numpy() + 1e-6
                self.replay_buffer.update_priorities(batch['indices'], priorities)
            
            # Update target network
            if self.total_steps % self.config.target_update_frequency == 0:
                self.target_network.load_state_dict(self.q_network.state_dict())
            
            self.updates += 1
            
            # Metrics
            metrics = {
                'loss': loss.item(),
                'epsilon': self.epsilon,
                'q_values': batch['observations'].shape[0],  # Batch size as proxy
                'priority_weight': batch['weights'].mean().item() if self.config.use_prioritized_replay else 1.0
            }
            
            # Store metrics
            for key, value in metrics.items():
                if key in self.training_metrics:
                    self.training_metrics[key].append(value)
        
        return metrics
    
    def _compute_loss(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute distributional DQN loss"""
        observations = batch['observations']
        actions = batch['actions']
        rewards = batch['rewards']
        next_observations = batch['next_observations']
        dones = batch['dones']
        weights = batch['weights']
        
        # Current Q distribution
        current_q_dist = self.q_network(observations)
        current_q_dist = current_q_dist[range(len(actions)), actions]
        
        with torch.no_grad():
            # Next Q distributions
            if self.config.use_double_dqn:
                # Double DQN: use main network for action selection
                next_q_values = self.q_network.get_q_values(next_observations)
                next_actions = next_q_values.argmax(dim=1)
                next_q_dist = self.target_network(next_observations)
                next_q_dist = next_q_dist[range(len(next_actions)), next_actions]
            else:
                # Standard DQN
                next_q_dist = self.target_network(next_observations)
                next_q_values = torch.sum(next_q_dist * self.q_network.support, dim=-1)
                next_actions = next_q_values.argmax(dim=1)
                next_q_dist = next_q_dist[range(len(next_actions)), next_actions]
            
            # Compute target distribution
            target_support = rewards.unsqueeze(1) + self.config.gamma * self.q_network.support.unsqueeze(0) * (1 - dones.unsqueeze(1))
            target_support = target_support.clamp(self.config.v_min, self.config.v_max)
            
            # Project target distribution
            b = (target_support - self.config.v_min) / self.q_network.delta_z
            l = b.floor().long()
            u = b.ceil().long()
            
            # Handle edge cases
            l[(u > 0) * (l == u)] -= 1
            u[(l < (self.config.n_atoms - 1)) * (l == u)] += 1
            
            # Distribute probability
            target_q_dist = torch.zeros_like(next_q_dist)
            offset = torch.linspace(0, (len(next_q_dist) - 1) * self.config.n_atoms, len(next_q_dist)).long().unsqueeze(1).to(self.device)
            
            target_q_dist.view(-1).index_add_(0, (l + offset).view(-1), (next_q_dist * (u.float() - b)).view(-1))
            target_q_dist.view(-1).index_add_(0, (u + offset).view(-1), (next_q_dist * (b - l.float())).view(-1))
        
        # KL divergence loss
        loss = -(target_q_dist * torch.log(current_q_dist + 1e-8)).sum(dim=1)
        
        # Apply importance sampling weights
        if self.config.use_prioritized_replay:
            loss = loss * weights
        
        td_errors = loss.detach()
        loss = loss.mean()
        
        return loss, td_errors
    
    def save(self, filepath: str):
        """Save agent state"""
        checkpoint = {
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'total_steps': self.total_steps,
            'episodes': self.episodes,
            'updates': self.updates,
            'epsilon': self.epsilon,
            'training_metrics': self.training_metrics
        }
        
        torch.save(checkpoint, filepath)
        logger.info(f"Rainbow DQN agent saved to {filepath}")
    
    def load(self, filepath: str):
        """Load agent state"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        self.total_steps = checkpoint.get('total_steps', 0)
        self.episodes = checkpoint.get('episodes', 0)
        self.updates = checkpoint.get('updates', 0)
        self.epsilon = checkpoint.get('epsilon', self.config.epsilon_start)
        self.training_metrics = checkpoint.get('training_metrics', self.training_metrics)
        
        logger.info(f"Rainbow DQN agent loaded from {filepath}")
    
    def get_training_metrics(self) -> Dict[str, List[float]]:
        """Get training metrics"""
        return self.training_metrics.copy()
    
    def reset_episode(self):
        """Reset for new episode"""
        self.episodes += 1


# Factory function
def create_rainbow_dqn_agent(obs_dim: int, action_dim: int, config: Dict[str, Any] = None) -> RainbowDQNAgent:
    """Create Rainbow DQN agent with configuration"""
    if config:
        rainbow_config = RainbowConfig(**config)
    else:
        rainbow_config = RainbowConfig()
    
    return RainbowDQNAgent(obs_dim, action_dim, rainbow_config)


# Export classes and functions
__all__ = [
    'RainbowDQNAgent',
    'RainbowConfig',
    'DuelingDistributionalNetwork',
    'NoisyLinear',
    'PrioritizedReplayBuffer',
    'create_rainbow_dqn_agent'
]