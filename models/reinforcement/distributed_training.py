"""
Distributed Reinforcement Learning Training System
Ray-based distributed training infrastructure for trading agents
"""

import ray
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import warnings
import asyncio
import threading
import queue
import time
import pickle
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .trading_env import TradingEnvironment, TradingEnvConfig
from .agents import PPOAgent, SACAgent, RainbowDQNAgent, MultiAgentTradingSystem
from .reward_functions import RewardFunction
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class DistributedConfig:
    """Configuration for distributed training"""
    # Ray cluster configuration
    ray_init_config: Dict[str, Any] = field(default_factory=dict)
    num_workers: int = 4
    num_cpus_per_worker: float = 1.0
    num_gpus_per_worker: float = 0.0
    
    # Training configuration
    episodes_per_iteration: int = 100
    max_iterations: int = 1000
    evaluation_interval: int = 10
    save_interval: int = 50
    
    # Experience collection
    experience_buffer_size: int = 100000
    batch_size: int = 256
    replay_buffer_alpha: float = 0.6
    replay_buffer_beta: float = 0.4
    
    # Hyperparameter optimization
    use_population_based_training: bool = False
    population_size: int = 8
    perturbation_interval: int = 5
    hyperparam_mutations: Dict[str, Any] = field(default_factory=dict)
    
    # Model serving
    model_serving_enabled: bool = True
    model_checkpoint_dir: str = "./checkpoints"
    tensorboard_log_dir: str = "./logs"
    
    # Monitoring
    enable_monitoring: bool = True
    metrics_update_interval: float = 10.0
    performance_threshold: float = 0.05  # Early stopping threshold


@dataclass
class TrainingMetrics:
    """Training metrics tracking"""
    episode_rewards: List[float] = field(default_factory=list)
    episode_lengths: List[int] = field(default_factory=list)
    actor_losses: List[float] = field(default_factory=list)
    critic_losses: List[float] = field(default_factory=list)
    training_times: List[float] = field(default_factory=list)
    evaluation_scores: List[float] = field(default_factory=list)
    
    # Performance metrics
    sharpe_ratios: List[float] = field(default_factory=list)
    max_drawdowns: List[float] = field(default_factory=list)
    total_returns: List[float] = field(default_factory=list)
    
    # Resource usage
    cpu_usage: List[float] = field(default_factory=list)
    memory_usage: List[float] = field(default_factory=list)
    gpu_usage: List[float] = field(default_factory=list)


@ray.remote
class DistributedWorker:
    """Ray actor for distributed training worker"""
    
    def __init__(self, worker_id: int, env_config: Dict[str, Any], 
                 agent_config: Dict[str, Any], reward_config: Dict[str, Any],
                 agent_type: str = 'ppo'):
        self.worker_id = worker_id
        self.agent_type = agent_type
        
        # Initialize environment
        self.env = TradingEnvironment(TradingEnvConfig(**env_config))
        
        # Initialize reward function
        from .reward_functions import create_reward_function
        self.reward_function = create_reward_function('composite', reward_config)
        
        # Initialize agent
        obs_dim = self.env.observation_space.shape[0]
        if agent_type == 'rainbow_dqn':
            action_dim = self.env.action_space.n if hasattr(self.env.action_space, 'n') else self.env.action_space.shape[0]
        else:
            action_dim = self.env.action_space.shape[0] if hasattr(self.env.action_space, 'shape') else self.env.action_space.n
        
        self.agent = self._create_agent(agent_type, obs_dim, action_dim, agent_config)
        
        # Experience buffer for this worker
        self.experience_buffer = []
        self.episode_rewards = []
        
        logger.info(f"Initialized distributed worker {worker_id}")
    
    def _create_agent(self, agent_type: str, obs_dim: int, action_dim: int, config: Dict[str, Any]):
        """Create agent based on type"""
        if agent_type == 'ppo':
            from .agents import create_ppo_agent
            return create_ppo_agent(obs_dim, action_dim, config, discrete=False)
        elif agent_type == 'sac':
            from .agents import create_sac_agent
            return create_sac_agent(obs_dim, action_dim, config)
        elif agent_type == 'rainbow_dqn':
            from .agents import create_rainbow_dqn_agent
            return create_rainbow_dqn_agent(obs_dim, action_dim, config)
        elif agent_type == 'multi_agent':
            from .agents import create_multi_agent_system
            return create_multi_agent_system(obs_dim, action_dim, config)
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")
    
    def collect_experience(self, num_episodes: int) -> Tuple[List[Dict[str, Any]], List[float]]:
        """Collect experience from environment interactions"""
        experiences = []
        episode_rewards = []
        
        for episode in range(num_episodes):
            obs = self.env.reset()
            episode_reward = 0
            episode_experiences = []
            done = False
            
            while not done:
                # Get action from agent
                if self.agent_type in ['ppo', 'sac']:
                    action = self.agent.act(obs)
                elif self.agent_type == 'rainbow_dqn':
                    action = self.agent.select_action(obs)
                else:
                    action = self.agent.act(obs)
                
                # Take step in environment
                next_obs, reward, done, info = self.env.step(action)
                
                # Calculate custom reward
                custom_reward = self.reward_function.calculate_reward(
                    obs, action, next_obs, reward, done, info
                )
                
                # Store experience
                experience = {
                    'obs': obs.copy(),
                    'action': action,
                    'reward': custom_reward,
                    'next_obs': next_obs.copy(),
                    'done': done,
                    'info': info
                }
                episode_experiences.append(experience)
                
                obs = next_obs
                episode_reward += custom_reward
            
            experiences.extend(episode_experiences)
            episode_rewards.append(episode_reward)
        
        return experiences, episode_rewards
    
    def update_agent(self, shared_parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Update agent with shared parameters"""
        try:
            # Update agent parameters
            if hasattr(self.agent, 'load_state_dict'):
                self.agent.load_state_dict(shared_parameters)
            elif hasattr(self.agent, 'set_parameters'):
                self.agent.set_parameters(shared_parameters)
            
            return {'success': True, 'worker_id': self.worker_id}
        except Exception as e:
            logger.error(f"Worker {self.worker_id} failed to update parameters: {e}")
            return {'success': False, 'error': str(e), 'worker_id': self.worker_id}
    
    def evaluate_agent(self, num_episodes: int = 10) -> Dict[str, float]:
        """Evaluate agent performance"""
        total_rewards = []
        episode_lengths = []
        
        for episode in range(num_episodes):
            obs = self.env.reset()
            episode_reward = 0
            episode_length = 0
            done = False
            
            while not done:
                # Get action (without exploration)
                if hasattr(self.agent, 'act_deterministic'):
                    action = self.agent.act_deterministic(obs)
                elif hasattr(self.agent, 'act'):
                    action = self.agent.act(obs, deterministic=True)
                else:
                    action = self.agent.select_action(obs, deterministic=True)
                
                next_obs, reward, done, info = self.env.step(action)
                
                # Calculate custom reward
                custom_reward = self.reward_function.calculate_reward(
                    obs, action, next_obs, reward, done, info
                )
                
                obs = next_obs
                episode_reward += custom_reward
                episode_length += 1
            
            total_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
        
        # Calculate performance metrics
        portfolio_values = self.env.portfolio_manager.portfolio_history
        if len(portfolio_values) > 1:
            returns = np.diff(portfolio_values) / portfolio_values[:-1]
            sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)
            
            # Calculate max drawdown
            cumulative = np.cumprod(1 + returns)
            peak = np.maximum.accumulate(cumulative)
            drawdown = (cumulative - peak) / peak
            max_drawdown = np.min(drawdown)
            
            total_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]
        else:
            sharpe_ratio = 0.0
            max_drawdown = 0.0
            total_return = 0.0
        
        return {
            'mean_reward': np.mean(total_rewards),
            'std_reward': np.std(total_rewards),
            'mean_episode_length': np.mean(episode_lengths),
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'total_return': total_return,
            'worker_id': self.worker_id
        }
    
    def get_agent_parameters(self) -> Dict[str, Any]:
        """Get current agent parameters"""
        if hasattr(self.agent, 'state_dict'):
            return self.agent.state_dict()
        elif hasattr(self.agent, 'get_parameters'):
            return self.agent.get_parameters()
        else:
            return {}


@ray.remote
class ParameterServer:
    """Ray actor for parameter server"""
    
    def __init__(self, agent_config: Dict[str, Any], agent_type: str = 'ppo'):
        self.agent_type = agent_type
        self.parameters = {}
        self.iteration = 0
        self.worker_updates = {}
        
        # Initialize with random parameters
        self._initialize_parameters(agent_config)
        
        logger.info("Initialized parameter server")
    
    def _initialize_parameters(self, agent_config: Dict[str, Any]):
        """Initialize parameters"""
        # This would typically initialize the neural network parameters
        # For simplicity, we'll use a placeholder
        self.parameters = {
            'iteration': 0,
            'config': agent_config,
            'weights': {}  # This would contain actual neural network weights
        }
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get current parameters"""
        return self.parameters.copy()
    
    def update_parameters(self, worker_id: int, gradients: Dict[str, Any], 
                         performance_metrics: Dict[str, float]):
        """Update parameters with worker gradients"""
        self.worker_updates[worker_id] = {
            'gradients': gradients,
            'metrics': performance_metrics,
            'timestamp': time.time()
        }
        
        # Simple averaging for now (could implement more sophisticated aggregation)
        if len(self.worker_updates) >= 2:  # Minimum workers for update
            self._aggregate_updates()
            self.iteration += 1
            self.worker_updates.clear()
    
    def _aggregate_updates(self):
        """Aggregate worker updates"""
        # Simple parameter averaging
        all_metrics = [update['metrics'] for update in self.worker_updates.values()]
        avg_performance = np.mean([m.get('mean_reward', 0) for m in all_metrics])
        
        self.parameters['iteration'] = self.iteration
        self.parameters['avg_performance'] = avg_performance
        
        logger.info(f"Parameter server iteration {self.iteration}, avg performance: {avg_performance:.4f}")


class DistributedTrainingManager:
    """
    Distributed training manager for RL trading agents
    
    Features:
    - Ray-based distributed training
    - Parameter server architecture
    - Real-time monitoring and evaluation
    - Hyperparameter optimization
    - Model checkpointing and serving
    """
    
    def __init__(self, config: DistributedConfig):
        self.config = config
        self.metrics = TrainingMetrics()
        
        # Initialize Ray cluster
        if not ray.is_initialized():
            ray.init(**config.ray_init_config)
        
        # Training state
        self.current_iteration = 0
        self.best_performance = float('-inf')
        self.training_start_time = None
        
        # Monitoring
        self.monitoring_active = False
        self.monitoring_thread = None
        
        logger.info("Initialized distributed training manager")
    
    def start_training(self, env_config: Dict[str, Any], agent_config: Dict[str, Any],
                      reward_config: Dict[str, Any], agent_type: str = 'ppo'):
        """Start distributed training"""
        self.training_start_time = time.time()
        
        try:
            # Initialize parameter server
            parameter_server = ParameterServer.remote(agent_config, agent_type)
            
            # Initialize workers
            workers = []
            for i in range(self.config.num_workers):
                worker = DistributedWorker.remote(
                    worker_id=i,
                    env_config=env_config,
                    agent_config=agent_config,
                    reward_config=reward_config,
                    agent_type=agent_type
                )
                workers.append(worker)
            
            # Start monitoring
            if self.config.enable_monitoring:
                self._start_monitoring(workers, parameter_server)
            
            # Main training loop
            for iteration in range(self.config.max_iterations):
                self.current_iteration = iteration
                
                # Get current parameters
                current_params = ray.get(parameter_server.get_parameters.remote())
                
                # Update all workers with current parameters
                update_futures = [
                    worker.update_agent.remote(current_params) 
                    for worker in workers
                ]
                ray.get(update_futures)
                
                # Collect experience from workers
                experience_futures = [
                    worker.collect_experience.remote(self.config.episodes_per_iteration // self.config.num_workers)
                    for worker in workers
                ]
                
                experiences_and_rewards = ray.get(experience_futures)
                
                # Aggregate experiences and rewards
                all_experiences = []
                all_episode_rewards = []
                for experiences, episode_rewards in experiences_and_rewards:
                    all_experiences.extend(experiences)
                    all_episode_rewards.extend(episode_rewards)
                
                # Update metrics
                self.metrics.episode_rewards.extend(all_episode_rewards)
                
                # Send aggregated data to parameter server
                # (In a real implementation, would calculate gradients and send those)
                avg_reward = np.mean(all_episode_rewards) if all_episode_rewards else 0.0
                parameter_server.update_parameters.remote(
                    worker_id=-1,  # Aggregate update
                    gradients={},  # Placeholder
                    performance_metrics={'mean_reward': avg_reward}
                )
                
                # Evaluation
                if iteration % self.config.evaluation_interval == 0:
                    eval_results = self._evaluate_workers(workers)
                    self._log_evaluation_results(iteration, eval_results)
                
                # Checkpointing
                if iteration % self.config.save_interval == 0:
                    self._save_checkpoint(parameter_server, iteration)
                
                # Early stopping check
                if self._should_stop_early():
                    logger.info(f"Early stopping at iteration {iteration}")
                    break
            
            logger.info("Training completed successfully")
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
        finally:
            self._cleanup()
    
    def _evaluate_workers(self, workers: List) -> List[Dict[str, float]]:
        """Evaluate all workers"""
        eval_futures = [worker.evaluate_agent.remote(10) for worker in workers]
        return ray.get(eval_futures)
    
    def _log_evaluation_results(self, iteration: int, eval_results: List[Dict[str, float]]):
        """Log evaluation results"""
        avg_metrics = {}
        for key in eval_results[0].keys():
            if key != 'worker_id':
                avg_metrics[key] = np.mean([result[key] for result in eval_results])
        
        logger.info(f"Iteration {iteration} Evaluation:")
        for key, value in avg_metrics.items():
            logger.info(f"  {key}: {value:.4f}")
        
        # Update tracking metrics
        self.metrics.evaluation_scores.append(avg_metrics['mean_reward'])
        self.metrics.sharpe_ratios.append(avg_metrics['sharpe_ratio'])
        self.metrics.max_drawdowns.append(avg_metrics['max_drawdown'])
        self.metrics.total_returns.append(avg_metrics['total_return'])
    
    def _save_checkpoint(self, parameter_server, iteration: int):
        """Save model checkpoint"""
        checkpoint_dir = Path(self.config.model_checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        params = ray.get(parameter_server.get_parameters.remote())
        
        checkpoint_path = checkpoint_dir / f"checkpoint_iteration_{iteration}.pkl"
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(params, f)
        
        # Save metrics
        metrics_path = checkpoint_dir / f"metrics_iteration_{iteration}.json"
        metrics_dict = {
            'episode_rewards': self.metrics.episode_rewards[-1000:],  # Last 1000
            'evaluation_scores': self.metrics.evaluation_scores,
            'sharpe_ratios': self.metrics.sharpe_ratios,
            'max_drawdowns': self.metrics.max_drawdowns,
            'total_returns': self.metrics.total_returns
        }
        
        with open(metrics_path, 'w') as f:
            json.dump(metrics_dict, f, indent=2)
        
        logger.info(f"Checkpoint saved at iteration {iteration}")
    
    def _should_stop_early(self) -> bool:
        """Check if training should stop early"""
        if len(self.metrics.evaluation_scores) < 5:
            return False
        
        # Check if performance has plateaued
        recent_scores = self.metrics.evaluation_scores[-5:]
        improvement = max(recent_scores) - min(recent_scores)
        
        return improvement < self.config.performance_threshold
    
    def _start_monitoring(self, workers: List, parameter_server):
        """Start monitoring thread"""
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(workers, parameter_server),
            daemon=True
        )
        self.monitoring_thread.start()
        logger.info("Started monitoring thread")
    
    def _monitoring_loop(self, workers: List, parameter_server):
        """Monitoring loop"""
        while self.monitoring_active:
            try:
                # Collect resource usage metrics
                self._collect_resource_metrics()
                
                # Log current status
                if len(self.metrics.episode_rewards) > 0:
                    recent_rewards = self.metrics.episode_rewards[-100:]  # Last 100 episodes
                    avg_reward = np.mean(recent_rewards)
                    
                    elapsed_time = time.time() - self.training_start_time
                    logger.info(f"Training status - Iteration: {self.current_iteration}, "
                              f"Avg Reward (last 100): {avg_reward:.4f}, "
                              f"Elapsed: {elapsed_time:.1f}s")
                
                time.sleep(self.config.metrics_update_interval)
                
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                time.sleep(5)
    
    def _collect_resource_metrics(self):
        """Collect resource usage metrics"""
        try:
            import psutil
            
            # CPU and memory usage
            cpu_percent = psutil.cpu_percent()
            memory_percent = psutil.virtual_memory().percent
            
            self.metrics.cpu_usage.append(cpu_percent)
            self.metrics.memory_usage.append(memory_percent)
            
            # GPU usage (if available)
            try:
                import GPUtil
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu_usage = np.mean([gpu.load * 100 for gpu in gpus])
                    self.metrics.gpu_usage.append(gpu_usage)
            except ImportError:
                pass
                
        except Exception as e:
            logger.debug(f"Failed to collect resource metrics: {e}")
    
    def _cleanup(self):
        """Cleanup resources"""
        self.monitoring_active = False
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5)
        
        logger.info("Cleanup completed")
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get training summary"""
        if not self.metrics.episode_rewards:
            return {"status": "No training data available"}
        
        total_episodes = len(self.metrics.episode_rewards)
        avg_reward = np.mean(self.metrics.episode_rewards)
        best_reward = max(self.metrics.episode_rewards)
        
        summary = {
            "training_status": "completed" if self.current_iteration >= self.config.max_iterations else "in_progress",
            "total_episodes": total_episodes,
            "iterations_completed": self.current_iteration,
            "average_reward": avg_reward,
            "best_reward": best_reward,
            "total_training_time": time.time() - self.training_start_time if self.training_start_time else 0
        }
        
        if self.metrics.evaluation_scores:
            summary.update({
                "final_evaluation_score": self.metrics.evaluation_scores[-1],
                "best_evaluation_score": max(self.metrics.evaluation_scores),
                "final_sharpe_ratio": self.metrics.sharpe_ratios[-1] if self.metrics.sharpe_ratios else 0,
                "final_max_drawdown": self.metrics.max_drawdowns[-1] if self.metrics.max_drawdowns else 0,
                "final_total_return": self.metrics.total_returns[-1] if self.metrics.total_returns else 0
            })
        
        return summary


# Factory functions
def create_distributed_trainer(config: Dict[str, Any] = None) -> DistributedTrainingManager:
    """Create distributed training manager"""
    distributed_config = DistributedConfig(**(config or {}))
    return DistributedTrainingManager(distributed_config)


def run_distributed_training(env_config: Dict[str, Any], agent_config: Dict[str, Any],
                            reward_config: Dict[str, Any], training_config: Dict[str, Any] = None,
                            agent_type: str = 'ppo') -> Dict[str, Any]:
    """Convenience function to run distributed training"""
    trainer = create_distributed_trainer(training_config)
    trainer.start_training(env_config, agent_config, reward_config, agent_type)
    return trainer.get_training_summary()


# Export classes and functions
__all__ = [
    'DistributedConfig',
    'TrainingMetrics',
    'DistributedWorker',
    'ParameterServer',
    'DistributedTrainingManager',
    'create_distributed_trainer',
    'run_distributed_training'
]