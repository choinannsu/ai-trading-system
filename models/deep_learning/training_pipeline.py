"""
Automated Training Pipeline with Hyperparameter Tuning
Optuna-based hyperparameter optimization, cross-validation, and model management
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import optuna
import joblib
import os
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass, asdict
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
from pathlib import Path
import logging

from .lstm_predictor import LSTMPredictor, LSTMConfig
from .transformer_model import TimeSeriesTransformer, TransformerConfig
from .cnn_lstm_hybrid import CNNLSTMHybrid, HybridConfig
from utils.logger import get_logger

logger = get_logger(__name__)
warnings.filterwarnings('ignore', category=UserWarning)

# Set TensorFlow logging level
tf.get_logger().setLevel('ERROR')


@dataclass
class TrainingConfig:
    """Training pipeline configuration"""
    model_type: str = 'lstm'  # 'lstm', 'transformer', 'hybrid'
    n_trials: int = 100
    n_splits: int = 5
    test_size: float = 0.2
    validation_size: float = 0.2
    random_state: int = 42
    
    # Early stopping
    patience: int = 15
    min_delta: float = 0.001
    
    # Model selection
    metric: str = 'mse'  # 'mse', 'mae', 'r2'
    direction: str = 'minimize'  # 'minimize', 'maximize'
    
    # Resource management
    max_epochs: int = 100
    max_trials_per_worker: int = 10
    timeout_seconds: int = 3600
    
    # Checkpointing
    save_checkpoints: bool = True
    checkpoint_dir: str = 'checkpoints'
    
    # Experiment tracking
    experiment_name: str = None
    track_experiments: bool = True


@dataclass
class ExperimentResult:
    """Experiment result with comprehensive metrics"""
    trial_number: int
    best_params: Dict[str, Any]
    best_score: float
    cv_scores: List[float]
    cv_mean: float
    cv_std: float
    training_time: float
    model_path: str
    experiment_id: str
    timestamp: datetime
    

@dataclass
class ModelVersion:
    """Model version metadata"""
    version_id: str
    model_type: str
    config: Dict[str, Any]
    performance_metrics: Dict[str, float]
    training_data_hash: str
    created_at: datetime
    model_path: str
    tags: List[str] = None


class HyperparameterTuner:
    """Optuna-based hyperparameter tuner for deep learning models"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.study = None
        self.best_model = None
        self.feature_names = None
        
    def _suggest_lstm_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Suggest LSTM hyperparameters"""
        return {
            'sequence_length': trial.suggest_int('sequence_length', 30, 120, step=10),
            'lstm_units': [
                trial.suggest_int('lstm_units_1', 32, 256, step=32),
                trial.suggest_int('lstm_units_2', 16, 128, step=16),
                trial.suggest_int('lstm_units_3', 8, 64, step=8)
            ],
            'attention_units': trial.suggest_int('attention_units', 64, 256, step=32),
            'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.5),
            'recurrent_dropout': trial.suggest_float('recurrent_dropout', 0.1, 0.4),
            'l2_reg': trial.suggest_float('l2_reg', 1e-5, 1e-2, log=True),
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
            'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64]),
            'use_bidirectional': trial.suggest_categorical('use_bidirectional', [True, False]),
            'dense_layers': [
                trial.suggest_int('dense_1', 16, 128, step=16),
                trial.suggest_int('dense_2', 8, 64, step=8)
            ]
        }
    
    def _suggest_transformer_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Suggest Transformer hyperparameters"""
        return {
            'sequence_length': trial.suggest_int('sequence_length', 30, 120, step=10),
            'd_model': trial.suggest_categorical('d_model', [64, 128, 256, 512]),
            'n_heads': trial.suggest_categorical('n_heads', [4, 8, 12, 16]),
            'n_transformer_blocks': trial.suggest_int('n_transformer_blocks', 2, 8),
            'ff_dim': trial.suggest_categorical('ff_dim', [128, 256, 512, 1024]),
            'dropout_rate': trial.suggest_float('dropout_rate', 0.05, 0.3),
            'attention_dropout': trial.suggest_float('attention_dropout', 0.05, 0.2),
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
            'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64]),
            'warmup_steps': trial.suggest_int('warmup_steps', 1000, 8000, step=1000),
            'positional_encoding_type': trial.suggest_categorical('positional_encoding_type', ['sinusoidal', 'learnable'])
        }
    
    def _suggest_hybrid_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Suggest CNN-LSTM hybrid hyperparameters"""
        return {
            'sequence_length': trial.suggest_int('sequence_length', 30, 120, step=10),
            'cnn_filters': [
                trial.suggest_int('cnn_filters_1', 32, 128, step=16),
                trial.suggest_int('cnn_filters_2', 64, 256, step=32),
                trial.suggest_int('cnn_filters_3', 128, 512, step=64),
                trial.suggest_int('cnn_filters_4', 64, 256, step=32)
            ],
            'cnn_kernel_sizes': [3, 3, 3, 3],
            'lstm_units': [
                trial.suggest_int('lstm_units_1', 64, 256, step=32),
                trial.suggest_int('lstm_units_2', 32, 128, step=16)
            ],
            'cnn_dropout': trial.suggest_float('cnn_dropout', 0.1, 0.4),
            'lstm_dropout': trial.suggest_float('lstm_dropout', 0.1, 0.4),
            'dense_dropout': trial.suggest_float('dense_dropout', 0.2, 0.5),
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
            'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64]),
            'use_residual': trial.suggest_categorical('use_residual', [True, False]),
            'use_skip_connections': trial.suggest_categorical('use_skip_connections', [True, False]),
            'l2_reg': trial.suggest_float('l2_reg', 1e-5, 1e-2, log=True)
        }
    
    def _create_model(self, model_type: str, params: Dict[str, Any]):
        """Create model instance with given parameters"""
        if model_type == 'lstm':
            config = LSTMConfig(**params)
            return LSTMPredictor(config)
        elif model_type == 'transformer':
            config = TransformerConfig(**params)
            return TimeSeriesTransformer(config)
        elif model_type == 'hybrid':
            config = HybridConfig(**params)
            return CNNLSTMHybrid(config)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def _objective(self, trial: optuna.Trial, X: np.ndarray, y: np.ndarray, 
                   target_column: str) -> float:
        """Objective function for Optuna optimization"""
        try:
            # Suggest hyperparameters based on model type
            if self.config.model_type == 'lstm':
                params = self._suggest_lstm_params(trial)
            elif self.config.model_type == 'transformer':
                params = self._suggest_transformer_params(trial)
            elif self.config.model_type == 'hybrid':
                params = self._suggest_hybrid_params(trial)
            else:
                raise ValueError(f"Unknown model type: {self.config.model_type}")
            
            # Time series cross-validation
            tscv = TimeSeriesSplit(n_splits=self.config.n_splits)
            scores = []
            
            for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
                try:
                    # Create fold data
                    X_train_fold = X[train_idx]
                    y_train_fold = y[train_idx]
                    X_val_fold = X[val_idx]
                    y_val_fold = y[val_idx]
                    
                    # Create DataFrame for model
                    train_df = pd.DataFrame(X_train_fold, columns=self.feature_names)
                    train_df[target_column] = y_train_fold
                    
                    val_df = pd.DataFrame(X_val_fold, columns=self.feature_names)
                    val_df[target_column] = y_val_fold
                    
                    # Create and train model
                    model = self._create_model(self.config.model_type, params)
                    
                    # Set epochs for quick training during optimization
                    if hasattr(model.config, 'epochs'):
                        model.config.epochs = min(50, self.config.max_epochs)
                        model.config.patience = min(10, self.config.patience)
                    
                    # Train model
                    model.fit(train_df, target_column, self.feature_names)
                    
                    # Make predictions
                    predictions = model.predict(val_df)
                    y_pred = predictions.predictions if hasattr(predictions, 'predictions') else predictions
                    
                    # Calculate score
                    if self.config.metric == 'mse':
                        score = mean_squared_error(y_val_fold[-len(y_pred):], y_pred)
                    elif self.config.metric == 'mae':
                        score = mean_absolute_error(y_val_fold[-len(y_pred):], y_pred)
                    elif self.config.metric == 'r2':
                        score = r2_score(y_val_fold[-len(y_pred):], y_pred)
                    
                    scores.append(score)
                    logger.debug(f"Trial {trial.number}, Fold {fold}: {self.config.metric} = {score:.6f}")
                    
                except Exception as e:
                    logger.warning(f"Fold {fold} failed: {e}")
                    continue
            
            if not scores:
                raise optuna.TrialPruned("All folds failed")
            
            # Return mean score
            mean_score = np.mean(scores)
            logger.info(f"Trial {trial.number}: Mean {self.config.metric} = {mean_score:.6f}")
            
            return mean_score
            
        except Exception as e:
            logger.warning(f"Trial {trial.number} failed: {e}")
            raise optuna.TrialPruned()
    
    def optimize(self, data: pd.DataFrame, target_column: str,
                 feature_columns: List[str] = None) -> ExperimentResult:
        """Run hyperparameter optimization"""
        logger.info(f"Starting hyperparameter optimization for {self.config.model_type}")
        
        if feature_columns is None:
            feature_columns = [col for col in data.columns if col != target_column]
        
        self.feature_names = feature_columns
        
        # Prepare data
        X = data[feature_columns].values
        y = data[target_column].values
        
        # Handle missing values
        X = pd.DataFrame(X).ffill().bfill().values
        y = pd.Series(y).ffill().bfill().values
        
        # Create study
        direction = self.config.direction if self.config.metric != 'r2' else 'maximize'
        
        study_name = f"{self.config.model_type}_{self.config.experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.study = optuna.create_study(
            direction=direction,
            study_name=study_name,
            sampler=optuna.samplers.TPESampler(seed=self.config.random_state)
        )
        
        # Optimize
        start_time = datetime.now()
        
        self.study.optimize(
            lambda trial: self._objective(trial, X, y, target_column),
            n_trials=self.config.n_trials,
            timeout=self.config.timeout_seconds
        )
        
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Get best parameters and train final model
        best_params = self.study.best_params
        logger.info(f"Best parameters: {best_params}")
        
        # Train final model with best parameters
        self.best_model = self._create_model(self.config.model_type, best_params)
        
        # Set full epochs for final training
        if hasattr(self.best_model.config, 'epochs'):
            self.best_model.config.epochs = self.config.max_epochs
            self.best_model.config.patience = self.config.patience
        
        result = self.best_model.fit(data, target_column, feature_columns)
        
        # Calculate cross-validation scores
        cv_scores = []
        tscv = TimeSeriesSplit(n_splits=self.config.n_splits)
        
        for train_idx, val_idx in tscv.split(X):
            X_train_fold = X[train_idx]
            y_train_fold = y[train_idx]
            X_val_fold = X[val_idx]
            y_val_fold = y[val_idx]
            
            train_df = pd.DataFrame(X_train_fold, columns=feature_columns)
            train_df[target_column] = y_train_fold
            
            val_df = pd.DataFrame(X_val_fold, columns=feature_columns)
            
            temp_model = self._create_model(self.config.model_type, best_params)
            temp_model.fit(train_df, target_column, feature_columns)
            
            predictions = temp_model.predict(val_df)
            y_pred = predictions.predictions if hasattr(predictions, 'predictions') else predictions
            
            if self.config.metric == 'mse':
                score = mean_squared_error(y_val_fold[-len(y_pred):], y_pred)
            elif self.config.metric == 'mae':
                score = mean_absolute_error(y_val_fold[-len(y_pred):], y_pred)
            elif self.config.metric == 'r2':
                score = r2_score(y_val_fold[-len(y_pred):], y_pred)
            
            cv_scores.append(score)
        
        # Create experiment result
        experiment_result = ExperimentResult(
            trial_number=self.study.best_trial.number,
            best_params=best_params,
            best_score=self.study.best_value,
            cv_scores=cv_scores,
            cv_mean=np.mean(cv_scores),
            cv_std=np.std(cv_scores),
            training_time=training_time,
            model_path="",  # Will be set by ModelManager
            experiment_id=study_name,
            timestamp=datetime.now()
        )
        
        logger.info(f"Optimization completed. Best {self.config.metric}: {self.study.best_value:.6f}")
        return experiment_result


class ModelManager:
    """Model version management and experiment tracking"""
    
    def __init__(self, base_path: str = "models"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(exist_ok=True)
        
        self.models_dir = self.base_path / "trained_models"
        self.experiments_dir = self.base_path / "experiments"
        self.versions_dir = self.base_path / "versions"
        
        for dir_path in [self.models_dir, self.experiments_dir, self.versions_dir]:
            dir_path.mkdir(exist_ok=True)
    
    def save_model(self, model, experiment_result: ExperimentResult, 
                   tags: List[str] = None) -> ModelVersion:
        """Save model and create version metadata"""
        version_id = f"{experiment_result.experiment_id}_{experiment_result.trial_number}"
        model_path = self.models_dir / f"{version_id}"
        
        # Save model
        if hasattr(model, 'save_model'):
            model.save_model(str(model_path))
        else:
            joblib.dump(model, f"{model_path}_model.pkl")
        
        # Create version metadata
        model_version = ModelVersion(
            version_id=version_id,
            model_type=experiment_result.experiment_id.split('_')[0],
            config=experiment_result.best_params,
            performance_metrics={
                'best_score': experiment_result.best_score,
                'cv_mean': experiment_result.cv_mean,
                'cv_std': experiment_result.cv_std
            },
            training_data_hash=self._hash_data(),
            created_at=experiment_result.timestamp,
            model_path=str(model_path),
            tags=tags or []
        )
        
        # Save version metadata
        version_file = self.versions_dir / f"{version_id}.json"
        with open(version_file, 'w') as f:
            json.dump(asdict(model_version), f, default=str, indent=2)
        
        # Update experiment result with model path
        experiment_result.model_path = str(model_path)
        
        # Save experiment result
        experiment_file = self.experiments_dir / f"{experiment_result.experiment_id}.json"
        with open(experiment_file, 'w') as f:
            json.dump(asdict(experiment_result), f, default=str, indent=2)
        
        logger.info(f"Model saved as version {version_id}")
        return model_version
    
    def load_model(self, version_id: str):
        """Load model by version ID"""
        version_file = self.versions_dir / f"{version_id}.json"
        
        if not version_file.exists():
            raise FileNotFoundError(f"Version {version_id} not found")
        
        with open(version_file, 'r') as f:
            version_data = json.load(f)
        
        model_path = version_data['model_path']
        model_type = version_data['model_type']
        
        # Load model based on type
        if model_type == 'lstm':
            model = LSTMPredictor()
        elif model_type == 'transformer':
            model = TimeSeriesTransformer()
        elif model_type == 'hybrid':
            model = CNNLSTMHybrid()
        else:
            # Try to load as pickle
            return joblib.load(f"{model_path}_model.pkl")
        
        model.load_model(model_path)
        return model
    
    def list_versions(self, model_type: str = None, tags: List[str] = None) -> List[ModelVersion]:
        """List available model versions"""
        versions = []
        
        for version_file in self.versions_dir.glob("*.json"):
            with open(version_file, 'r') as f:
                version_data = json.load(f)
            
            # Filter by model type
            if model_type and version_data.get('model_type') != model_type:
                continue
            
            # Filter by tags
            if tags:
                version_tags = version_data.get('tags', [])
                if not any(tag in version_tags for tag in tags):
                    continue
            
            version = ModelVersion(**version_data)
            versions.append(version)
        
        # Sort by creation date (newest first)
        versions.sort(key=lambda x: x.created_at, reverse=True)
        return versions
    
    def get_best_model(self, model_type: str = None, metric: str = 'cv_mean') -> ModelVersion:
        """Get best performing model"""
        versions = self.list_versions(model_type)
        
        if not versions:
            raise ValueError("No models found")
        
        # Sort by metric (assuming lower is better for most metrics)
        best_version = min(versions, key=lambda x: x.performance_metrics.get(metric, float('inf')))
        return best_version
    
    def _hash_data(self) -> str:
        """Generate hash for training data (simplified)"""
        return f"data_hash_{datetime.now().strftime('%Y%m%d')}"


class TrainingPipeline:
    """Complete automated training pipeline"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.tuner = HyperparameterTuner(config)
        self.model_manager = ModelManager()
        
    def train_and_optimize(self, data: pd.DataFrame, target_column: str,
                          feature_columns: List[str] = None,
                          tags: List[str] = None) -> Tuple[Any, ModelVersion]:
        """Complete training pipeline with optimization"""
        logger.info("Starting automated training pipeline")
        
        try:
            # Run hyperparameter optimization
            experiment_result = self.tuner.optimize(data, target_column, feature_columns)
            
            # Save best model
            model_version = self.model_manager.save_model(
                self.tuner.best_model, 
                experiment_result, 
                tags
            )
            
            logger.info(f"Training pipeline completed. Model version: {model_version.version_id}")
            
            return self.tuner.best_model, model_version
            
        except Exception as e:
            logger.error(f"Training pipeline failed: {e}")
            raise
    
    def compare_models(self, data: pd.DataFrame, target_column: str,
                      model_types: List[str] = None,
                      feature_columns: List[str] = None) -> Dict[str, ModelVersion]:
        """Train and compare multiple model types"""
        if model_types is None:
            model_types = ['lstm', 'transformer', 'hybrid']
        
        results = {}
        original_model_type = self.config.model_type
        
        for model_type in model_types:
            logger.info(f"Training {model_type} model...")
            
            # Update config for current model type
            self.config.model_type = model_type
            self.tuner = HyperparameterTuner(self.config)
            
            try:
                model, version = self.train_and_optimize(
                    data, target_column, feature_columns, 
                    tags=[f"comparison_{datetime.now().strftime('%Y%m%d')}"]
                )
                results[model_type] = version
                
            except Exception as e:
                logger.error(f"Failed to train {model_type}: {e}")
                continue
        
        # Restore original config
        self.config.model_type = original_model_type
        
        # Find best model
        if results:
            best_model_type = min(results.keys(), 
                                key=lambda x: results[x].performance_metrics['cv_mean'])
            logger.info(f"Best model type: {best_model_type}")
        
        return results


# Factory function
def create_training_pipeline(config: Dict[str, Any] = None) -> TrainingPipeline:
    """Create training pipeline with configuration"""
    if config:
        training_config = TrainingConfig(**config)
    else:
        training_config = TrainingConfig()
    
    return TrainingPipeline(training_config)


# Export classes and functions
__all__ = [
    'TrainingPipeline',
    'HyperparameterTuner', 
    'ModelManager',
    'TrainingConfig',
    'ExperimentResult',
    'ModelVersion',
    'create_training_pipeline'
]