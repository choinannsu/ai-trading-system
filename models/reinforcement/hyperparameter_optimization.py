"""
Hyperparameter Optimization for RL Trading Agents
Advanced hyperparameter tuning using Optuna, Ray Tune, and Population-Based Training
"""

import optuna
import ray
from ray import tune
from ray.tune.schedulers import PopulationBasedTraining, ASHAScheduler
from ray.tune.suggest.optuna import OptunaSearch
import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime
import json
import pickle
from pathlib import Path
import warnings

from .trading_env import TradingEnvironment, TradingEnvConfig
from .distributed_training import DistributedTrainingManager, DistributedConfig
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class HyperparameterSpace:
    """Definition of hyperparameter search space"""
    # Agent hyperparameters
    learning_rate: Tuple[float, float] = (1e-5, 1e-2)  # (min, max)
    batch_size: List[int] = field(default_factory=lambda: [32, 64, 128, 256, 512])
    hidden_dims: List[List[int]] = field(default_factory=lambda: [[64, 64], [128, 128], [256, 256], [128, 64, 32]])
    
    # PPO specific
    ppo_eps_clip: Tuple[float, float] = (0.1, 0.3)
    ppo_gae_lambda: Tuple[float, float] = (0.9, 1.0)
    ppo_entropy_coef: Tuple[float, float] = (0.0, 0.1)
    
    # SAC specific
    sac_alpha: Tuple[float, float] = (0.1, 0.5)
    sac_tau: Tuple[float, float] = (0.001, 0.01)
    sac_gamma: Tuple[float, float] = (0.95, 0.999)
    
    # Rainbow DQN specific
    dqn_epsilon_start: Tuple[float, float] = (0.9, 1.0)
    dqn_epsilon_end: Tuple[float, float] = (0.01, 0.1)
    dqn_epsilon_decay: Tuple[float, float] = (0.995, 0.9995)
    
    # Environment hyperparameters
    lookback_window: List[int] = field(default_factory=lambda: [20, 40, 60, 100])
    initial_capital: List[float] = field(default_factory=lambda: [100000, 250000, 500000])
    transaction_cost: Tuple[float, float] = (0.0001, 0.01)
    
    # Reward function hyperparameters
    sharpe_weight: Tuple[float, float] = (0.0, 1.0)
    drawdown_penalty: Tuple[float, float] = (0.0, 2.0)
    frequency_penalty: Tuple[float, float] = (0.0, 0.5)


@dataclass
class OptimizationConfig:
    """Configuration for hyperparameter optimization"""
    # Optimization method
    method: str = "optuna"  # "optuna", "ray_tune", "pbt"
    
    # Search configuration
    n_trials: int = 100
    study_direction: str = "maximize"  # "maximize" or "minimize"
    
    # Optuna specific
    optuna_sampler: str = "TPE"  # "TPE", "Random", "CmaEs"
    optuna_pruner: str = "Median"  # "Median", "Successive", "Hyperband"
    
    # Ray Tune specific
    tune_scheduler: str = "ASHA"  # "ASHA", "PBT", "Hyperband"
    tune_search_alg: str = "Optuna"  # "Optuna", "Random", "Bayesian"
    max_concurrent_trials: int = 4
    
    # Population-Based Training
    pbt_population_size: int = 8
    pbt_perturbation_interval: int = 10
    pbt_quantile_fraction: float = 0.25
    
    # Training configuration
    episodes_per_trial: int = 1000
    evaluation_episodes: int = 100
    max_training_time: float = 3600  # seconds
    
    # Early stopping
    min_improvement_threshold: float = 0.01
    patience: int = 10
    
    # Checkpointing
    save_checkpoints: bool = True
    checkpoint_dir: str = "./hyperopt_checkpoints"


@dataclass
class OptimizationResult:
    """Results from hyperparameter optimization"""
    best_params: Dict[str, Any]
    best_score: float
    optimization_history: List[Dict[str, Any]]
    study_summary: Dict[str, Any]
    total_trials: int
    total_time: float
    
    # Performance metrics for best configuration
    best_sharpe_ratio: float = 0.0
    best_max_drawdown: float = 0.0
    best_total_return: float = 0.0
    best_win_rate: float = 0.0


class OptunaTuner:
    """Optuna-based hyperparameter tuning"""
    
    def __init__(self, config: OptimizationConfig, search_space: HyperparameterSpace):
        self.config = config
        self.search_space = search_space
        
        # Create study
        self.study = optuna.create_study(
            direction=config.study_direction,
            sampler=self._create_sampler(),
            pruner=self._create_pruner()
        )
        
        logger.info("Initialized Optuna tuner")
    
    def _create_sampler(self):
        """Create Optuna sampler"""
        if self.config.optuna_sampler == "TPE":
            return optuna.samplers.TPESampler()
        elif self.config.optuna_sampler == "Random":
            return optuna.samplers.RandomSampler()
        elif self.config.optuna_sampler == "CmaEs":
            return optuna.samplers.CmaEsSampler()
        else:
            return optuna.samplers.TPESampler()
    
    def _create_pruner(self):
        """Create Optuna pruner"""
        if self.config.optuna_pruner == "Median":
            return optuna.pruners.MedianPruner()
        elif self.config.optuna_pruner == "Successive":
            return optuna.pruners.SuccessiveHalvingPruner()
        elif self.config.optuna_pruner == "Hyperband":
            return optuna.pruners.HyperbandPruner()
        else:
            return optuna.pruners.MedianPruner()
    
    def optimize(self, env_config: Dict[str, Any], agent_type: str = 'ppo') -> OptimizationResult:
        """Run hyperparameter optimization"""
        start_time = datetime.now()
        
        def objective(trial):
            return self._objective_function(trial, env_config, agent_type)
        
        # Run optimization
        self.study.optimize(objective, n_trials=self.config.n_trials)
        
        # Create result
        end_time = datetime.now()
        total_time = (end_time - start_time).total_seconds()
        
        return OptimizationResult(
            best_params=self.study.best_params,
            best_score=self.study.best_value,
            optimization_history=self._get_optimization_history(),
            study_summary=self._get_study_summary(),
            total_trials=len(self.study.trials),
            total_time=total_time
        )
    
    def _objective_function(self, trial, env_config: Dict[str, Any], agent_type: str) -> float:
        """Objective function for optimization"""
        # Sample hyperparameters
        params = self._sample_hyperparameters(trial, agent_type)
        
        # Create configurations
        agent_config = self._create_agent_config(params, agent_type)
        reward_config = self._create_reward_config(params)
        training_config = self._create_training_config(params)
        
        # Update environment config
        updated_env_config = {
            **env_config,
            'lookback_window': params['lookback_window'],
            'initial_capital': params['initial_capital'],
            'transaction_cost': params['transaction_cost']
        }
        
        try:
            # Run training
            trainer = DistributedTrainingManager(DistributedConfig(**training_config))
            trainer.start_training(updated_env_config, agent_config, reward_config, agent_type)
            
            # Get results
            summary = trainer.get_training_summary()
            
            # Return optimization metric (e.g., final evaluation score)
            score = summary.get('final_evaluation_score', summary.get('average_reward', 0))
            
            # Report intermediate values for pruning
            if hasattr(trainer.metrics, 'evaluation_scores'):
                for i, eval_score in enumerate(trainer.metrics.evaluation_scores):
                    trial.report(eval_score, i)
                    
                    # Check if trial should be pruned
                    if trial.should_prune():
                        raise optuna.TrialPruned()
            
            return score
            
        except Exception as e:
            logger.error(f"Trial failed: {e}")
            return float('-inf')
    
    def _sample_hyperparameters(self, trial, agent_type: str) -> Dict[str, Any]:
        """Sample hyperparameters from search space"""
        params = {}
        
        # Common hyperparameters
        params['learning_rate'] = trial.suggest_float(
            'learning_rate', *self.search_space.learning_rate, log=True
        )
        params['batch_size'] = trial.suggest_categorical(
            'batch_size', self.search_space.batch_size
        )
        params['hidden_dims'] = trial.suggest_categorical(
            'hidden_dims', self.search_space.hidden_dims
        )
        
        # Agent-specific hyperparameters
        if agent_type == 'ppo':
            params['eps_clip'] = trial.suggest_float(
                'eps_clip', *self.search_space.ppo_eps_clip
            )
            params['gae_lambda'] = trial.suggest_float(
                'gae_lambda', *self.search_space.ppo_gae_lambda
            )
            params['entropy_coef'] = trial.suggest_float(
                'entropy_coef', *self.search_space.ppo_entropy_coef
            )
        
        elif agent_type == 'sac':
            params['alpha'] = trial.suggest_float(
                'alpha', *self.search_space.sac_alpha
            )
            params['tau'] = trial.suggest_float(
                'tau', *self.search_space.sac_tau
            )
            params['gamma'] = trial.suggest_float(
                'gamma', *self.search_space.sac_gamma
            )
        
        elif agent_type == 'rainbow_dqn':
            params['epsilon_start'] = trial.suggest_float(
                'epsilon_start', *self.search_space.dqn_epsilon_start
            )
            params['epsilon_end'] = trial.suggest_float(
                'epsilon_end', *self.search_space.dqn_epsilon_end
            )
            params['epsilon_decay'] = trial.suggest_float(
                'epsilon_decay', *self.search_space.dqn_epsilon_decay
            )
        
        # Environment hyperparameters
        params['lookback_window'] = trial.suggest_categorical(
            'lookback_window', self.search_space.lookback_window
        )
        params['initial_capital'] = trial.suggest_categorical(
            'initial_capital', self.search_space.initial_capital
        )
        params['transaction_cost'] = trial.suggest_float(
            'transaction_cost', *self.search_space.transaction_cost, log=True
        )
        
        # Reward function hyperparameters
        params['sharpe_weight'] = trial.suggest_float(
            'sharpe_weight', *self.search_space.sharpe_weight
        )
        params['drawdown_penalty'] = trial.suggest_float(
            'drawdown_penalty', *self.search_space.drawdown_penalty
        )
        params['frequency_penalty'] = trial.suggest_float(
            'frequency_penalty', *self.search_space.frequency_penalty
        )
        
        return params
    
    def _create_agent_config(self, params: Dict[str, Any], agent_type: str) -> Dict[str, Any]:
        """Create agent configuration from parameters"""
        config = {
            'learning_rate': params['learning_rate'],
            'batch_size': params['batch_size'],
            'hidden_dims': params['hidden_dims']
        }
        
        if agent_type == 'ppo':
            config.update({
                'eps_clip': params['eps_clip'],
                'gae_lambda': params['gae_lambda'],
                'entropy_coef': params['entropy_coef']
            })
        elif agent_type == 'sac':
            config.update({
                'alpha': params['alpha'],
                'tau': params['tau'],
                'gamma': params['gamma']
            })
        elif agent_type == 'rainbow_dqn':
            config.update({
                'epsilon_start': params['epsilon_start'],
                'epsilon_end': params['epsilon_end'],
                'epsilon_decay': params['epsilon_decay']
            })
        
        return config
    
    def _create_reward_config(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Create reward configuration from parameters"""
        return {
            'rewards': [
                {'type': 'sharpe', 'weight': params['sharpe_weight']},
                {'type': 'max_drawdown_penalty', 'weight': params['drawdown_penalty']},
                {'type': 'trading_frequency_penalty', 'weight': params['frequency_penalty']}
            ]
        }
    
    def _create_training_config(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Create training configuration from parameters"""
        return {
            'episodes_per_iteration': self.config.episodes_per_trial,
            'max_iterations': 10,  # Short training for hyperopt
            'evaluation_interval': 2,
            'num_workers': 2  # Reduced for hyperopt
        }
    
    def _get_optimization_history(self) -> List[Dict[str, Any]]:
        """Get optimization history"""
        history = []
        for trial in self.study.trials:
            history.append({
                'trial_id': trial.number,
                'value': trial.value,
                'params': trial.params,
                'state': trial.state.name
            })
        return history
    
    def _get_study_summary(self) -> Dict[str, Any]:
        """Get study summary"""
        completed_trials = [t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        
        return {
            'n_trials': len(self.study.trials),
            'n_completed_trials': len(completed_trials),
            'best_value': self.study.best_value,
            'best_params': self.study.best_params
        }


class RayTuneTuner:
    """Ray Tune-based hyperparameter tuning"""
    
    def __init__(self, config: OptimizationConfig, search_space: HyperparameterSpace):
        self.config = config
        self.search_space = search_space
        
        # Initialize Ray
        if not ray.is_initialized():
            ray.init()
        
        logger.info("Initialized Ray Tune tuner")
    
    def optimize(self, env_config: Dict[str, Any], agent_type: str = 'ppo') -> OptimizationResult:
        """Run hyperparameter optimization with Ray Tune"""
        # Create search space
        tune_config = self._create_tune_config(agent_type)
        
        # Create scheduler
        scheduler = self._create_scheduler()
        
        # Create search algorithm
        search_alg = self._create_search_algorithm()
        
        # Run tuning
        analysis = tune.run(
            tune.with_parameters(self._trainable, env_config=env_config, agent_type=agent_type),
            config=tune_config,
            scheduler=scheduler,
            search_alg=search_alg,
            num_samples=self.config.n_trials,
            max_concurrent_trials=self.config.max_concurrent_trials,
            stop={"training_iteration": 20},
            checkpoint_at_end=True,
            local_dir=self.config.checkpoint_dir
        )
        
        # Get best result
        best_trial = analysis.get_best_trial("mean_reward", "max", "last")
        
        return OptimizationResult(
            best_params=best_trial.config,
            best_score=best_trial.last_result["mean_reward"],
            optimization_history=self._get_ray_tune_history(analysis),
            study_summary={"best_trial_id": best_trial.trial_id},
            total_trials=len(analysis.trials),
            total_time=0  # Would need to calculate from logs
        )
    
    def _create_tune_config(self, agent_type: str) -> Dict[str, Any]:
        """Create Ray Tune configuration"""
        config = {
            'learning_rate': tune.loguniform(*self.search_space.learning_rate),
            'batch_size': tune.choice(self.search_space.batch_size),
            'hidden_dims': tune.choice(self.search_space.hidden_dims)
        }
        
        if agent_type == 'ppo':
            config.update({
                'eps_clip': tune.uniform(*self.search_space.ppo_eps_clip),
                'gae_lambda': tune.uniform(*self.search_space.ppo_gae_lambda),
                'entropy_coef': tune.uniform(*self.search_space.ppo_entropy_coef)
            })
        
        return config
    
    def _create_scheduler(self):
        """Create Ray Tune scheduler"""
        if self.config.tune_scheduler == "ASHA":
            return ASHAScheduler(
                metric="mean_reward",
                mode="max",
                max_t=20,
                grace_period=5
            )
        elif self.config.tune_scheduler == "PBT":
            return PopulationBasedTraining(
                time_attr="training_iteration",
                perturbation_interval=self.config.pbt_perturbation_interval,
                quantile_fraction=self.config.pbt_quantile_fraction
            )
        else:
            return ASHAScheduler()
    
    def _create_search_algorithm(self):
        """Create search algorithm"""
        if self.config.tune_search_alg == "Optuna":
            return OptunaSearch()
        else:
            return None  # Use default random search
    
    def _trainable(self, config, env_config, agent_type):
        """Trainable function for Ray Tune"""
        # Create configurations
        agent_config = config
        reward_config = self._create_reward_config(config)
        training_config = {'episodes_per_iteration': 100, 'max_iterations': 20}
        
        # Run training
        trainer = DistributedTrainingManager(DistributedConfig(**training_config))
        trainer.start_training(env_config, agent_config, reward_config, agent_type)
        
        # Report metrics
        summary = trainer.get_training_summary()
        tune.report(mean_reward=summary.get('average_reward', 0))
    
    def _create_reward_config(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Create reward configuration from parameters"""
        return {'rewards': [{'type': 'sharpe', 'weight': 1.0}]}
    
    def _get_ray_tune_history(self, analysis) -> List[Dict[str, Any]]:
        """Get Ray Tune optimization history"""
        history = []
        for trial in analysis.trials:
            history.append({
                'trial_id': trial.trial_id,
                'config': trial.config,
                'status': trial.status
            })
        return history


class HyperparameterOptimizer:
    """
    Main hyperparameter optimization interface
    
    Features:
    - Multiple optimization backends (Optuna, Ray Tune)
    - Population-based training support
    - Comprehensive search space definition
    - Early stopping and pruning
    - Result analysis and visualization
    """
    
    def __init__(self, config: OptimizationConfig = None, 
                 search_space: HyperparameterSpace = None):
        self.config = config or OptimizationConfig()
        self.search_space = search_space or HyperparameterSpace()
        
        # Initialize appropriate tuner
        if self.config.method == "optuna":
            self.tuner = OptunaTuner(self.config, self.search_space)
        elif self.config.method == "ray_tune":
            self.tuner = RayTuneTuner(self.config, self.search_space)
        else:
            raise ValueError(f"Unknown optimization method: {self.config.method}")
        
        logger.info(f"Initialized hyperparameter optimizer with {self.config.method}")
    
    def optimize(self, env_config: Dict[str, Any], agent_type: str = 'ppo') -> OptimizationResult:
        """Run hyperparameter optimization"""
        logger.info(f"Starting hyperparameter optimization for {agent_type} agent")
        logger.info(f"Method: {self.config.method}, Trials: {self.config.n_trials}")
        
        result = self.tuner.optimize(env_config, agent_type)
        
        logger.info(f"Optimization completed. Best score: {result.best_score:.4f}")
        logger.info(f"Best parameters: {result.best_params}")
        
        return result
    
    def save_results(self, result: OptimizationResult, filepath: str):
        """Save optimization results"""
        with open(filepath, 'wb') as f:
            pickle.dump(result, f)
        logger.info(f"Results saved to {filepath}")
    
    def load_results(self, filepath: str) -> OptimizationResult:
        """Load optimization results"""
        with open(filepath, 'rb') as f:
            result = pickle.load(f)
        return result
    
    def analyze_results(self, result: OptimizationResult) -> Dict[str, Any]:
        """Analyze optimization results"""
        analysis = {
            'best_score': result.best_score,
            'best_params': result.best_params,
            'total_trials': result.total_trials,
            'optimization_time': result.total_time
        }
        
        if result.optimization_history:
            scores = [trial['value'] for trial in result.optimization_history 
                     if trial.get('value') is not None]
            
            if scores:
                analysis.update({
                    'score_statistics': {
                        'mean': np.mean(scores),
                        'std': np.std(scores),
                        'min': np.min(scores),
                        'max': np.max(scores),
                        'median': np.median(scores)
                    },
                    'improvement_over_random': result.best_score - np.mean(scores),
                    'convergence_analysis': self._analyze_convergence(scores)
                })
        
        return analysis
    
    def _analyze_convergence(self, scores: List[float]) -> Dict[str, Any]:
        """Analyze convergence properties"""
        if len(scores) < 10:
            return {'insufficient_data': True}
        
        # Calculate running maximum
        running_max = np.maximum.accumulate(scores)
        
        # Find when 90% of final performance was reached
        final_score = running_max[-1]
        target_score = 0.9 * final_score
        
        convergence_trial = None
        for i, score in enumerate(running_max):
            if score >= target_score:
                convergence_trial = i
                break
        
        return {
            'convergence_trial': convergence_trial,
            'convergence_efficiency': convergence_trial / len(scores) if convergence_trial else 1.0,
            'final_improvement_rate': (running_max[-1] - running_max[-10]) / 10 if len(running_max) >= 10 else 0
        }


# Factory functions
def create_hyperparameter_optimizer(method: str = "optuna", **kwargs) -> HyperparameterOptimizer:
    """Create hyperparameter optimizer"""
    config = OptimizationConfig(method=method, **kwargs)
    return HyperparameterOptimizer(config)


def optimize_agent_hyperparameters(env_config: Dict[str, Any], agent_type: str = 'ppo',
                                  method: str = "optuna", n_trials: int = 50,
                                  **kwargs) -> OptimizationResult:
    """Convenience function for hyperparameter optimization"""
    optimizer = create_hyperparameter_optimizer(method=method, n_trials=n_trials, **kwargs)
    return optimizer.optimize(env_config, agent_type)


# Export classes and functions
__all__ = [
    'HyperparameterSpace',
    'OptimizationConfig',
    'OptimizationResult',
    'OptunaTuner',
    'RayTuneTuner',
    'HyperparameterOptimizer',
    'create_hyperparameter_optimizer',
    'optimize_agent_hyperparameters'
]