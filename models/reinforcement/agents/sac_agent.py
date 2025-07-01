"""
Soft Actor-Critic (SAC) Agent for Trading
Advanced SAC implementation optimized for continuous action spaces
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import warnings
from collections import deque
import random

from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class SACConfig:
    """SAC agent configuration"""
    # Network architecture
    hidden_sizes: List[int] = None
    activation: str = 'relu'
    
    # SAC hyperparameters
    learning_rate: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005  # Soft update coefficient
    alpha: float = 0.2  # Entropy regularization coefficient
    automatic_entropy_tuning: bool = True
    target_entropy: float = None  # Will be set automatically if None
    
    # Training settings
    batch_size: int = 256
    buffer_size: int = 1000000
    min_buffer_size: int = 10000
    update_frequency: int = 1
    target_update_frequency: int = 1
    
    # Network specific
    use_layer_norm: bool = False
    use_spectral_norm: bool = False
    
    # Financial specific
    action_noise: float = 0.1
    action_bounds: Tuple[float, float] = (-1.0, 1.0)
    risk_penalty_coef: float = 0.1
    
    def __post_init__(self):
        if self.hidden_sizes is None:
            self.hidden_sizes = [256, 256]
        
        if self.target_entropy is None:
            # Heuristic for target entropy
            self.target_entropy = -np.prod((2,)).item()  # Assuming 2D action space


class QNetwork(nn.Module):
    """Q-Network for SAC critic"""
    
    def __init__(self, obs_dim: int, action_dim: int, config: SACConfig):
        super().__init__()
        
        self.config = config
        
        # Build Q-network
        layers = []
        input_size = obs_dim + action_dim
        
        for hidden_size in config.hidden_sizes:
            layer = nn.Linear(input_size, hidden_size)
            
            if config.use_spectral_norm:
                layer = nn.utils.spectral_norm(layer)
            
            layers.append(layer)
            
            if config.use_layer_norm:
                layers.append(nn.LayerNorm(hidden_size))
            
            if config.activation == 'relu':
                layers.append(nn.ReLU())
            elif config.activation == 'tanh':
                layers.append(nn.Tanh())
            elif config.activation == 'elu':
                layers.append(nn.ELU())
            
            input_size = hidden_size
        
        layers.append(nn.Linear(input_size, 1))
        
        self.q_network = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        x = torch.cat([obs, action], dim=-1)
        return self.q_network(x)


class PolicyNetwork(nn.Module):
    """Policy network for SAC actor"""
    
    def __init__(self, obs_dim: int, action_dim: int, config: SACConfig):
        super().__init__()
        
        self.config = config
        self.action_dim = action_dim
        self.action_scale = (config.action_bounds[1] - config.action_bounds[0]) / 2
        self.action_bias = (config.action_bounds[1] + config.action_bounds[0]) / 2
        
        # Build shared layers
        layers = []
        input_size = obs_dim
        
        for hidden_size in config.hidden_sizes:
            layer = nn.Linear(input_size, hidden_size)
            
            if config.use_spectral_norm:
                layer = nn.utils.spectral_norm(layer)
            
            layers.append(layer)
            
            if config.use_layer_norm:
                layers.append(nn.LayerNorm(hidden_size))
            
            if config.activation == 'relu':
                layers.append(nn.ReLU())
            elif config.activation == 'tanh':
                layers.append(nn.Tanh())
            elif config.activation == 'elu':
                layers.append(nn.ELU())
            
            input_size = hidden_size
        
        self.shared_layers = nn.Sequential(*layers)
        
        # Mean and log std heads
        self.mean_head = nn.Linear(input_size, action_dim)
        self.log_std_head = nn.Linear(input_size, action_dim)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0.0)
        
        # Special initialization for policy output
        nn.init.uniform_(self.mean_head.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.mean_head.bias, -3e-3, 3e-3)
        nn.init.uniform_(self.log_std_head.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.log_std_head.bias, -3e-3, 3e-3)
    
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass"""
        features = self.shared_layers(obs)
        
        mean = self.mean_head(features)
        log_std = self.log_std_head(features)
        
        # Clamp log_std to prevent numerical instability
        log_std = torch.clamp(log_std, -20, 2)
        
        return mean, log_std
    
    def sample(self, obs: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample action from policy"""
        mean, log_std = self.forward(obs)
        
        if deterministic:
            action = mean
            log_prob = None
        else:
            std = log_std.exp()
            normal = Normal(mean, std)
            
            # Reparameterization trick
            x_t = normal.rsample()
            action = torch.tanh(x_t)
            
            # Calculate log probability
            log_prob = normal.log_prob(x_t)
            # Enforcing Action Bound
            log_prob -= torch.log(1 - action.pow(2) + 1e-6)
            log_prob = log_prob.sum(dim=-1, keepdim=True)
        
        # Scale action to action bounds
        action = action * self.action_scale + self.action_bias
        
        return action, log_prob


class ReplayBuffer:
    """Experience replay buffer for SAC"""
    
    def __init__(self, obs_dim: int, action_dim: int, buffer_size: int):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.buffer_size = buffer_size
        
        # Buffers
        self.observations = np.zeros((buffer_size, obs_dim), dtype=np.float32)
        self.actions = np.zeros((buffer_size, action_dim), dtype=np.float32)
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.next_observations = np.zeros((buffer_size, obs_dim), dtype=np.float32)
        self.dones = np.zeros(buffer_size, dtype=np.bool_)
        
        self.ptr = 0
        self.size = 0
    
    def store(self, obs: np.ndarray, action: np.ndarray, reward: float, 
             next_obs: np.ndarray, done: bool):
        """Store experience"""
        idx = self.ptr % self.buffer_size
        
        self.observations[idx] = obs
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.next_observations[idx] = next_obs
        self.dones[idx] = done
        
        self.ptr += 1
        self.size = min(self.size + 1, self.buffer_size)
    
    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample batch for training"""
        indices = np.random.choice(self.size, batch_size, replace=False)
        
        return {
            'observations': torch.FloatTensor(self.observations[indices]),
            'actions': torch.FloatTensor(self.actions[indices]),
            'rewards': torch.FloatTensor(self.rewards[indices]).unsqueeze(-1),
            'next_observations': torch.FloatTensor(self.next_observations[indices]),
            'dones': torch.FloatTensor(self.dones[indices]).unsqueeze(-1)
        }


class SACAgent:
    """Soft Actor-Critic agent for trading"""
    
    def __init__(self, obs_dim: int, action_dim: int, config: SACConfig = None):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.config = config or SACConfig()
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Networks
        self.policy = PolicyNetwork(obs_dim, action_dim, self.config).to(self.device)
        self.q1 = QNetwork(obs_dim, action_dim, self.config).to(self.device)
        self.q2 = QNetwork(obs_dim, action_dim, self.config).to(self.device)
        
        # Target networks
        self.q1_target = QNetwork(obs_dim, action_dim, self.config).to(self.device)
        self.q2_target = QNetwork(obs_dim, action_dim, self.config).to(self.device)
        
        # Copy parameters to target networks
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        
        # Optimizers
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=self.config.learning_rate)
        self.q1_optimizer = optim.Adam(self.q1.parameters(), lr=self.config.learning_rate)
        self.q2_optimizer = optim.Adam(self.q2.parameters(), lr=self.config.learning_rate)
        
        # Automatic entropy tuning
        if self.config.automatic_entropy_tuning:
            self.target_entropy = self.config.target_entropy
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.config.learning_rate)
        else:
            self.alpha = self.config.alpha
        
        # Experience replay
        self.replay_buffer = ReplayBuffer(obs_dim, action_dim, self.config.buffer_size)
        
        # Training state
        self.total_steps = 0
        self.episodes = 0
        self.updates = 0
        
        # Metrics
        self.training_metrics = {
            'policy_loss': [],
            'q1_loss': [],
            'q2_loss': [],
            'alpha_loss': [],
            'alpha_value': [],
            'q1_value': [],
            'q2_value': []
        }
    
    @property
    def alpha(self):
        """Get current alpha value"""
        if self.config.automatic_entropy_tuning:
            return self.log_alpha.exp()
        else:
            return self.config.alpha
    
    def get_action(self, obs: np.ndarray, deterministic: bool = False) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Get action from agent"""
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action, log_prob = self.policy.sample(obs_tensor, deterministic)
        
        action = action.cpu().numpy().flatten()
        
        info = {
            'log_prob': log_prob.cpu().numpy().item() if log_prob is not None else 0.0
        }
        
        return action, info
    
    def store_transition(self, obs: np.ndarray, action: np.ndarray, reward: float,
                        next_obs: np.ndarray, done: bool, info: Dict[str, Any]):
        """Store transition in replay buffer"""
        self.replay_buffer.store(obs, action, reward, next_obs, done)
        self.total_steps += 1
    
    def update(self) -> Dict[str, float]:
        """Update agent using SAC algorithm"""
        if self.replay_buffer.size < self.config.min_buffer_size:
            return {}
        
        metrics = {}
        
        # Update frequency
        if self.total_steps % self.config.update_frequency == 0:
            # Sample batch
            batch = self.replay_buffer.sample(self.config.batch_size)
            
            for key in batch:
                batch[key] = batch[key].to(self.device)
            
            # Update Q-networks
            q_metrics = self._update_q_networks(batch)
            metrics.update(q_metrics)
            
            # Update policy
            policy_metrics = self._update_policy(batch)
            metrics.update(policy_metrics)
            
            # Update target networks
            if self.updates % self.config.target_update_frequency == 0:
                self._soft_update_targets()
            
            # Update alpha (if automatic tuning)
            if self.config.automatic_entropy_tuning:
                alpha_metrics = self._update_alpha(batch)
                metrics.update(alpha_metrics)
            
            self.updates += 1
            
            # Store metrics
            for key, value in metrics.items():
                if key in self.training_metrics:
                    self.training_metrics[key].append(value)
        
        return metrics
    
    def _update_q_networks(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Update Q-networks"""
        with torch.no_grad():
            # Sample actions for next states
            next_actions, next_log_probs = self.policy.sample(batch['next_observations'])
            
            # Compute target Q-values
            q1_next = self.q1_target(batch['next_observations'], next_actions)
            q2_next = self.q2_target(batch['next_observations'], next_actions)
            min_q_next = torch.min(q1_next, q2_next)
            
            # Add entropy term
            next_q_values = min_q_next - self.alpha * next_log_probs
            
            # Compute target
            target_q = batch['rewards'] + self.config.gamma * (1 - batch['dones']) * next_q_values
        
        # Current Q-values
        q1_current = self.q1(batch['observations'], batch['actions'])
        q2_current = self.q2(batch['observations'], batch['actions'])
        
        # Q-losses
        q1_loss = F.mse_loss(q1_current, target_q)
        q2_loss = F.mse_loss(q2_current, target_q)
        
        # Update Q1
        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        self.q1_optimizer.step()
        
        # Update Q2
        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        self.q2_optimizer.step()
        
        return {
            'q1_loss': q1_loss.item(),
            'q2_loss': q2_loss.item(),
            'q1_value': q1_current.mean().item(),
            'q2_value': q2_current.mean().item()
        }
    
    def _update_policy(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Update policy network"""
        # Sample actions from current policy
        actions, log_probs = self.policy.sample(batch['observations'])
        
        # Compute Q-values for sampled actions
        q1_values = self.q1(batch['observations'], actions)
        q2_values = self.q2(batch['observations'], actions)
        min_q_values = torch.min(q1_values, q2_values)
        
        # Policy loss
        policy_loss = (self.alpha * log_probs - min_q_values).mean()
        
        # Update policy
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
        return {
            'policy_loss': policy_loss.item()
        }
    
    def _update_alpha(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Update entropy coefficient alpha"""
        with torch.no_grad():
            _, log_probs = self.policy.sample(batch['observations'])
        
        # Alpha loss
        alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy)).mean()
        
        # Update alpha
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        
        return {
            'alpha_loss': alpha_loss.item(),
            'alpha_value': self.alpha.item()
        }
    
    def _soft_update_targets(self):
        """Soft update target networks"""
        for target_param, param in zip(self.q1_target.parameters(), self.q1.parameters()):
            target_param.data.copy_(self.config.tau * param.data + (1 - self.config.tau) * target_param.data)
        
        for target_param, param in zip(self.q2_target.parameters(), self.q2.parameters()):
            target_param.data.copy_(self.config.tau * param.data + (1 - self.config.tau) * target_param.data)
    
    def save(self, filepath: str):
        """Save agent state"""
        checkpoint = {
            'policy_state_dict': self.policy.state_dict(),
            'q1_state_dict': self.q1.state_dict(),
            'q2_state_dict': self.q2.state_dict(),
            'q1_target_state_dict': self.q1_target.state_dict(),
            'q2_target_state_dict': self.q2_target.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'q1_optimizer_state_dict': self.q1_optimizer.state_dict(),
            'q2_optimizer_state_dict': self.q2_optimizer.state_dict(),
            'config': self.config,
            'total_steps': self.total_steps,
            'episodes': self.episodes,
            'updates': self.updates,
            'training_metrics': self.training_metrics
        }
        
        if self.config.automatic_entropy_tuning:
            checkpoint['log_alpha'] = self.log_alpha
            checkpoint['alpha_optimizer_state_dict'] = self.alpha_optimizer.state_dict()
        
        torch.save(checkpoint, filepath)
        logger.info(f"SAC agent saved to {filepath}")
    
    def load(self, filepath: str):
        """Load agent state"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.q1.load_state_dict(checkpoint['q1_state_dict'])
        self.q2.load_state_dict(checkpoint['q2_state_dict'])
        self.q1_target.load_state_dict(checkpoint['q1_target_state_dict'])
        self.q2_target.load_state_dict(checkpoint['q2_target_state_dict'])
        
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
        self.q1_optimizer.load_state_dict(checkpoint['q1_optimizer_state_dict'])
        self.q2_optimizer.load_state_dict(checkpoint['q2_optimizer_state_dict'])
        
        if self.config.automatic_entropy_tuning and 'log_alpha' in checkpoint:
            self.log_alpha = checkpoint['log_alpha']
            self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer_state_dict'])
        
        self.total_steps = checkpoint.get('total_steps', 0)
        self.episodes = checkpoint.get('episodes', 0)
        self.updates = checkpoint.get('updates', 0)
        self.training_metrics = checkpoint.get('training_metrics', self.training_metrics)
        
        logger.info(f"SAC agent loaded from {filepath}")
    
    def get_training_metrics(self) -> Dict[str, List[float]]:
        """Get training metrics"""
        return self.training_metrics.copy()
    
    def reset_episode(self):
        """Reset for new episode"""
        self.episodes += 1


# Factory function
def create_sac_agent(obs_dim: int, action_dim: int, config: Dict[str, Any] = None) -> SACAgent:
    """Create SAC agent with configuration"""
    if config:
        sac_config = SACConfig(**config)
    else:
        sac_config = SACConfig()
    
    return SACAgent(obs_dim, action_dim, sac_config)


# Export classes and functions
__all__ = [
    'SACAgent',
    'SACConfig',
    'PolicyNetwork',
    'QNetwork',
    'ReplayBuffer',
    'create_sac_agent'
]