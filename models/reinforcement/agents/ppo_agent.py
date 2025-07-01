"""
Proximal Policy Optimization (PPO) Agent for Trading
Advanced PPO implementation with custom features for financial markets
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical, Normal
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import warnings
from collections import deque
import random

from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class PPOConfig:
    """PPO agent configuration"""
    # Network architecture
    hidden_sizes: List[int] = None
    activation: str = 'relu'
    use_lstm: bool = False
    lstm_hidden_size: int = 256
    
    # PPO hyperparameters
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    
    # Training settings
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    target_kl: float = 0.01
    
    # Experience replay
    buffer_size: int = 100000
    use_prioritized_replay: bool = False
    
    # Exploration
    exploration_noise: float = 0.1
    noise_decay: float = 0.995
    min_noise: float = 0.01
    
    # Financial specific
    use_market_features: bool = True
    risk_penalty_coef: float = 0.1
    transaction_cost_penalty: float = 0.01
    
    def __post_init__(self):
        if self.hidden_sizes is None:
            self.hidden_sizes = [256, 256, 128]


class ActorCriticNetwork(nn.Module):
    """Actor-Critic network for PPO"""
    
    def __init__(self, obs_dim: int, action_dim: int, config: PPOConfig, discrete: bool = True):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.config = config
        self.discrete = discrete
        
        # Shared layers
        self.shared_layers = self._build_shared_network()
        
        # LSTM layer (optional)
        if config.use_lstm:
            self.lstm = nn.LSTM(config.hidden_sizes[-1], config.lstm_hidden_size, batch_first=True)
            lstm_output_size = config.lstm_hidden_size
        else:
            self.lstm = None
            lstm_output_size = config.hidden_sizes[-1]
        
        # Actor head (policy)
        if discrete:
            self.actor = nn.Linear(lstm_output_size, action_dim)
        else:
            # Continuous actions: output mean and log_std
            self.actor_mean = nn.Linear(lstm_output_size, action_dim)
            self.actor_log_std = nn.Parameter(torch.zeros(action_dim))
        
        # Critic head (value function)
        self.critic = nn.Linear(lstm_output_size, 1)
        
        # Market-specific layers
        if config.use_market_features:
            self.market_feature_extractor = nn.Sequential(
                nn.Linear(obs_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU()
            )
        
        # Initialize weights
        self._init_weights()
    
    def _build_shared_network(self) -> nn.Module:
        """Build shared feature extraction network"""
        layers = []
        input_size = self.obs_dim
        
        for hidden_size in self.config.hidden_sizes:
            layers.append(nn.Linear(input_size, hidden_size))
            
            if self.config.activation == 'relu':
                layers.append(nn.ReLU())
            elif self.config.activation == 'tanh':
                layers.append(nn.Tanh())
            elif self.config.activation == 'elu':
                layers.append(nn.ELU())
            
            layers.append(nn.Dropout(0.1))
            input_size = hidden_size
        
        return nn.Sequential(*layers)
    
    def _init_weights(self):
        """Initialize network weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, np.sqrt(2))
                nn.init.constant_(module.bias, 0.0)
        
        # Special initialization for actor output
        if self.discrete:
            nn.init.orthogonal_(self.actor.weight, 0.01)
        else:
            nn.init.orthogonal_(self.actor_mean.weight, 0.01)
    
    def forward(self, obs: torch.Tensor, hidden_state: Optional[Tuple] = None):
        """Forward pass through network"""
        # Shared feature extraction
        features = self.shared_layers(obs)
        
        # LSTM processing (if enabled)
        if self.lstm is not None:
            if len(features.shape) == 2:
                features = features.unsqueeze(1)  # Add sequence dimension
            
            lstm_out, hidden_state = self.lstm(features, hidden_state)
            features = lstm_out.squeeze(1)  # Remove sequence dimension
        
        # Actor output
        if self.discrete:
            action_logits = self.actor(features)
            action_dist = Categorical(logits=action_logits)
        else:
            action_mean = self.actor_mean(features)
            action_log_std = self.actor_log_std.expand_as(action_mean)
            action_dist = Normal(action_mean, action_log_std.exp())
        
        # Critic output
        value = self.critic(features)
        
        return action_dist, value, hidden_state
    
    def get_action(self, obs: torch.Tensor, hidden_state: Optional[Tuple] = None, deterministic: bool = False):
        """Get action from policy"""
        action_dist, value, hidden_state = self.forward(obs, hidden_state)
        
        if deterministic:
            if self.discrete:
                action = action_dist.logits.argmax(dim=-1)
            else:
                action = action_dist.mean
        else:
            action = action_dist.sample()
        
        action_log_prob = action_dist.log_prob(action)
        
        return action, action_log_prob, value, hidden_state
    
    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor, hidden_state: Optional[Tuple] = None):
        """Evaluate actions for training"""
        action_dist, value, _ = self.forward(obs, hidden_state)
        
        action_log_probs = action_dist.log_prob(actions)
        entropy = action_dist.entropy()
        
        return action_log_probs, value, entropy


class PPOBuffer:
    """Experience buffer for PPO"""
    
    def __init__(self, obs_dim: int, action_dim: int, buffer_size: int, gamma: float, gae_lambda: float):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.buffer_size = buffer_size
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        
        # Buffers
        self.observations = np.zeros((buffer_size, obs_dim), dtype=np.float32)
        self.actions = np.zeros((buffer_size, action_dim), dtype=np.float32)
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.values = np.zeros(buffer_size, dtype=np.float32)
        self.log_probs = np.zeros(buffer_size, dtype=np.float32)
        self.dones = np.zeros(buffer_size, dtype=np.bool_)
        
        # Computed advantages and returns
        self.advantages = np.zeros(buffer_size, dtype=np.float32)
        self.returns = np.zeros(buffer_size, dtype=np.float32)
        
        self.ptr = 0
        self.size = 0
    
    def store(self, obs: np.ndarray, action: np.ndarray, reward: float, 
             value: float, log_prob: float, done: bool):
        """Store experience"""
        idx = self.ptr % self.buffer_size
        
        self.observations[idx] = obs
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.values[idx] = value
        self.log_probs[idx] = log_prob
        self.dones[idx] = done
        
        self.ptr += 1
        self.size = min(self.size + 1, self.buffer_size)
    
    def compute_gae(self, last_value: float = 0.0):
        """Compute Generalized Advantage Estimation"""
        advantages = np.zeros_like(self.rewards[:self.size])
        last_gae_lam = 0
        
        for step in reversed(range(self.size)):
            if step == self.size - 1:
                next_non_terminal = 1.0 - self.dones[step]
                next_value = last_value
            else:
                next_non_terminal = 1.0 - self.dones[step]
                next_value = self.values[step + 1]
            
            delta = self.rewards[step] + self.gamma * next_value * next_non_terminal - self.values[step]
            advantages[step] = last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
        
        self.advantages[:self.size] = advantages
        self.returns[:self.size] = advantages + self.values[:self.size]
        
        # Normalize advantages
        self.advantages[:self.size] = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    def get_batch(self, batch_size: int):
        """Get random batch for training"""
        indices = np.random.choice(self.size, batch_size, replace=False)
        
        return {
            'observations': torch.FloatTensor(self.observations[indices]),
            'actions': torch.FloatTensor(self.actions[indices]),
            'old_log_probs': torch.FloatTensor(self.log_probs[indices]),
            'returns': torch.FloatTensor(self.returns[indices]),
            'advantages': torch.FloatTensor(self.advantages[indices])
        }
    
    def clear(self):
        """Clear buffer"""
        self.ptr = 0
        self.size = 0


class PPOAgent:
    """Proximal Policy Optimization agent for trading"""
    
    def __init__(self, obs_dim: int, action_dim: int, config: PPOConfig = None, discrete: bool = True):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.config = config or PPOConfig()
        self.discrete = discrete
        
        # Networks
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.network = ActorCriticNetwork(obs_dim, action_dim, self.config, discrete).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.config.learning_rate)
        
        # Experience buffer
        self.buffer = PPOBuffer(
            obs_dim, action_dim if not discrete else 1, 
            self.config.buffer_size, self.config.gamma, self.config.gae_lambda
        )
        
        # Training state
        self.total_steps = 0
        self.episodes = 0
        self.current_noise = self.config.exploration_noise
        
        # Metrics
        self.training_metrics = {
            'policy_loss': [],
            'value_loss': [],
            'entropy_loss': [],
            'kl_divergence': [],
            'explained_variance': []
        }
    
    def get_action(self, obs: np.ndarray, deterministic: bool = False) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Get action from agent"""
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action, log_prob, value, _ = self.network.get_action(obs_tensor, deterministic=deterministic)
        
        action = action.cpu().numpy().flatten()
        log_prob = log_prob.cpu().numpy().item()
        value = value.cpu().numpy().item()
        
        # Add exploration noise for continuous actions
        if not self.discrete and not deterministic:
            action += np.random.normal(0, self.current_noise, action.shape)
            action = np.clip(action, -1, 1)
        
        return action, {
            'log_prob': log_prob,
            'value': value,
            'exploration_noise': self.current_noise
        }
    
    def store_transition(self, obs: np.ndarray, action: np.ndarray, reward: float, 
                        done: bool, info: Dict[str, Any]):
        """Store transition in buffer"""
        log_prob = info.get('log_prob', 0.0)
        value = info.get('value', 0.0)
        
        self.buffer.store(obs, action, reward, value, log_prob, done)
        self.total_steps += 1
    
    def update(self) -> Dict[str, float]:
        """Update agent using PPO algorithm"""
        if self.buffer.size < self.config.batch_size:
            return {}
        
        # Compute advantages
        self.buffer.compute_gae()
        
        # Training metrics
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy_loss = 0
        total_kl_div = 0
        n_updates = 0
        
        # Multiple epochs of training
        for epoch in range(self.config.n_epochs):
            # Sample batches
            n_batches = max(1, self.buffer.size // self.config.batch_size)
            
            for _ in range(n_batches):
                batch = self.buffer.get_batch(self.config.batch_size)
                
                # Evaluate current policy
                action_log_probs, values, entropy = self.network.evaluate_actions(
                    batch['observations'], 
                    batch['actions'] if not self.discrete else batch['actions'].long()
                )
                
                # Calculate policy loss
                ratio = torch.exp(action_log_probs - batch['old_log_probs'])
                surr1 = ratio * batch['advantages']
                surr2 = torch.clamp(ratio, 1 - self.config.clip_ratio, 1 + self.config.clip_ratio) * batch['advantages']
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Calculate value loss
                value_loss = F.mse_loss(values.squeeze(), batch['returns'])
                
                # Calculate entropy loss
                entropy_loss = -entropy.mean()
                
                # Total loss
                loss = (policy_loss + 
                       self.config.value_coef * value_loss + 
                       self.config.entropy_coef * entropy_loss)
                
                # Update network
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.config.max_grad_norm)
                self.optimizer.step()
                
                # Track metrics
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy_loss += entropy_loss.item()
                
                # Calculate KL divergence
                with torch.no_grad():
                    kl_div = (batch['old_log_probs'] - action_log_probs).mean().item()
                    total_kl_div += kl_div
                
                n_updates += 1
                
                # Early stopping if KL divergence too large
                if kl_div > self.config.target_kl:
                    logger.info(f"Early stopping at epoch {epoch} due to large KL divergence")
                    break
            
            if kl_div > self.config.target_kl:
                break
        
        # Update exploration noise
        self.current_noise = max(self.config.min_noise, 
                               self.current_noise * self.config.noise_decay)
        
        # Clear buffer
        self.buffer.clear()
        
        # Return metrics
        if n_updates > 0:
            metrics = {
                'policy_loss': total_policy_loss / n_updates,
                'value_loss': total_value_loss / n_updates,
                'entropy_loss': total_entropy_loss / n_updates,
                'kl_divergence': total_kl_div / n_updates,
                'exploration_noise': self.current_noise,
                'buffer_size': self.buffer.size
            }
            
            # Store metrics
            for key, value in metrics.items():
                if key in self.training_metrics:
                    self.training_metrics[key].append(value)
            
            return metrics
        
        return {}
    
    def save(self, filepath: str):
        """Save agent state"""
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'total_steps': self.total_steps,
            'episodes': self.episodes,
            'current_noise': self.current_noise,
            'training_metrics': self.training_metrics
        }, filepath)
        
        logger.info(f"PPO agent saved to {filepath}")
    
    def load(self, filepath: str):
        """Load agent state"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.total_steps = checkpoint.get('total_steps', 0)
        self.episodes = checkpoint.get('episodes', 0)
        self.current_noise = checkpoint.get('current_noise', self.config.exploration_noise)
        self.training_metrics = checkpoint.get('training_metrics', self.training_metrics)
        
        logger.info(f"PPO agent loaded from {filepath}")
    
    def get_training_metrics(self) -> Dict[str, List[float]]:
        """Get training metrics"""
        return self.training_metrics.copy()
    
    def reset_episode(self):
        """Reset for new episode"""
        self.episodes += 1
        
        # Clear any episode-specific state if needed
        if hasattr(self.network, 'lstm') and self.network.lstm is not None:
            # Reset LSTM hidden state
            pass


# Factory function
def create_ppo_agent(obs_dim: int, action_dim: int, config: Dict[str, Any] = None, 
                    discrete: bool = True) -> PPOAgent:
    """Create PPO agent with configuration"""
    if config:
        ppo_config = PPOConfig(**config)
    else:
        ppo_config = PPOConfig()
    
    return PPOAgent(obs_dim, action_dim, ppo_config, discrete)


# Export classes and functions
__all__ = [
    'PPOAgent',
    'PPOConfig',
    'ActorCriticNetwork',
    'PPOBuffer',
    'create_ppo_agent'
]