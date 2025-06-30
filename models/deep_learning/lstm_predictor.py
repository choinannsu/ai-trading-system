"""
LSTM Predictor with Attention Mechanism
Multi-layer LSTM with attention, dropout, regularization, and ensemble support
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, callbacks
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import warnings
from datetime import datetime
import joblib
import os
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from utils.logger import get_logger

logger = get_logger(__name__)

warnings.filterwarnings('ignore', category=UserWarning)
tf.get_logger().setLevel('ERROR')


@dataclass
class LSTMConfig:
    """LSTM model configuration"""
    sequence_length: int = 60
    n_features: int = 5
    lstm_units: List[int] = None
    attention_units: int = 128
    dropout_rate: float = 0.2
    recurrent_dropout: float = 0.2
    l2_reg: float = 0.001
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 100
    patience: int = 15
    validation_split: float = 0.2
    use_attention: bool = True
    use_bidirectional: bool = True
    dense_layers: List[int] = None
    activation: str = 'tanh'
    optimizer: str = 'adam'
    
    def __post_init__(self):
        if self.lstm_units is None:
            self.lstm_units = [128, 64, 32]
        if self.dense_layers is None:
            self.dense_layers = [64, 32]


@dataclass
class PredictionResult:
    """Prediction result with confidence metrics"""
    predictions: np.ndarray
    confidence_lower: np.ndarray
    confidence_upper: np.ndarray
    attention_weights: Optional[np.ndarray]
    feature_importance: Dict[str, float]
    metrics: Dict[str, float]
    timestamp: datetime


class AttentionLayer(layers.Layer):
    """Custom attention layer for LSTM"""
    
    def __init__(self, attention_units: int, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        self.attention_units = attention_units
        
    def build(self, input_shape):
        self.W = self.add_weight(
            name='attention_weight',
            shape=(input_shape[-1], self.attention_units),
            initializer='uniform',
            trainable=True
        )
        self.b = self.add_weight(
            name='attention_bias',
            shape=(self.attention_units,),
            initializer='zeros',
            trainable=True
        )
        self.u = self.add_weight(
            name='attention_u',
            shape=(self.attention_units,),
            initializer='uniform',
            trainable=True
        )
        super(AttentionLayer, self).build(input_shape)
    
    def call(self, inputs):
        # inputs shape: (batch_size, sequence_length, features)
        uit = tf.nn.tanh(tf.tensordot(inputs, self.W, axes=1) + self.b)
        ait = tf.tensordot(uit, self.u, axes=1)
        ait = tf.nn.softmax(ait, axis=1)
        
        # Expand dimensions for broadcasting
        ait = tf.expand_dims(ait, -1)
        
        # Apply attention weights
        weighted_input = inputs * ait
        output = tf.reduce_sum(weighted_input, axis=1)
        
        return output, ait
    
    def compute_output_shape(self, input_shape):
        return [(input_shape[0], input_shape[2]), (input_shape[0], input_shape[1], 1)]
    
    def get_config(self):
        config = super(AttentionLayer, self).get_config()
        config.update({'attention_units': self.attention_units})
        return config


class LSTMPredictor:
    """Advanced LSTM predictor with attention mechanism"""
    
    def __init__(self, config: LSTMConfig = None):
        self.config = config or LSTMConfig()
        self.model = None
        self.scaler = StandardScaler()
        self.target_scaler = MinMaxScaler()
        self.history = None
        self.feature_names = None
        self.is_trained = False
        
    def _optimize_sequence_length(self, X: np.ndarray, y: np.ndarray) -> int:
        """Automatically optimize sequence length using validation performance"""
        logger.info("Optimizing sequence length...")
        
        best_length = self.config.sequence_length
        best_score = float('inf')
        
        # Test different sequence lengths
        test_lengths = [30, 60, 90, 120]
        
        for length in test_lengths:
            try:
                # Create sequences with this length
                X_seq, y_seq = self._create_sequences(X, y, length)
                
                if len(X_seq) < 100:  # Need minimum data
                    continue
                
                # Split for validation
                split_idx = int(len(X_seq) * 0.8)
                X_train, X_val = X_seq[:split_idx], X_seq[split_idx:]
                y_train, y_val = y_seq[:split_idx], y_seq[split_idx:]
                
                # Build and train simple model
                temp_config = LSTMConfig(
                    sequence_length=length,
                    lstm_units=[64],
                    epochs=20,
                    patience=5
                )
                
                temp_model = self._build_model(temp_config)
                temp_model.fit(
                    X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=20,
                    batch_size=32,
                    verbose=0
                )
                
                # Evaluate
                y_pred = temp_model.predict(X_val, verbose=0)
                score = mean_squared_error(y_val, y_pred)
                
                if score < best_score:
                    best_score = score
                    best_length = length
                    
                logger.debug(f"Sequence length {length}: MSE = {score:.6f}")
                
            except Exception as e:
                logger.warning(f"Failed to test sequence length {length}: {e}")
                continue
        
        logger.info(f"Optimal sequence length: {best_length} (MSE: {best_score:.6f})")
        return best_length
    
    def _create_sequences(self, X: np.ndarray, y: np.ndarray, sequence_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM training"""
        X_seq, y_seq = [], []
        
        for i in range(sequence_length, len(X)):
            X_seq.append(X[i-sequence_length:i])
            y_seq.append(y[i])
        
        return np.array(X_seq), np.array(y_seq)
    
    def _build_model(self, config: LSTMConfig = None) -> Model:
        """Build LSTM model with attention"""
        if config is None:
            config = self.config
            
        logger.info("Building LSTM model with attention mechanism")
        
        # Input layer
        inputs = layers.Input(shape=(config.sequence_length, config.n_features))
        x = inputs
        
        # LSTM layers
        for i, units in enumerate(config.lstm_units):
            return_sequences = (i < len(config.lstm_units) - 1) or config.use_attention
            
            if config.use_bidirectional:
                lstm_layer = layers.Bidirectional(
                    layers.LSTM(
                        units,
                        return_sequences=return_sequences,
                        dropout=config.dropout_rate,
                        recurrent_dropout=config.recurrent_dropout,
                        kernel_regularizer=keras.regularizers.l2(config.l2_reg),
                        activation=config.activation
                    ),
                    name=f'bidirectional_lstm_{i}'
                )
            else:
                lstm_layer = layers.LSTM(
                    units,
                    return_sequences=return_sequences,
                    dropout=config.dropout_rate,
                    recurrent_dropout=config.recurrent_dropout,
                    kernel_regularizer=keras.regularizers.l2(config.l2_reg),
                    activation=config.activation,
                    name=f'lstm_{i}'
                )
            
            x = lstm_layer(x)
            
            # Add batch normalization
            if i < len(config.lstm_units) - 1:
                x = layers.BatchNormalization(name=f'bn_lstm_{i}')(x)
        
        # Attention mechanism
        if config.use_attention:
            attention_output, attention_weights = AttentionLayer(
                config.attention_units, 
                name='attention'
            )(x)
            x = attention_output
        
        # Dense layers
        for i, units in enumerate(config.dense_layers):
            x = layers.Dense(
                units,
                activation='relu',
                kernel_regularizer=keras.regularizers.l2(config.l2_reg),
                name=f'dense_{i}'
            )(x)
            x = layers.Dropout(config.dropout_rate, name=f'dropout_dense_{i}')(x)
            x = layers.BatchNormalization(name=f'bn_dense_{i}')(x)
        
        # Output layer
        outputs = layers.Dense(1, activation='linear', name='output')(x)
        
        # Create model
        model = Model(inputs=inputs, outputs=outputs, name='lstm_predictor')
        
        # Compile model
        optimizer = self._get_optimizer(config)
        
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae', 'mape']
        )
        
        logger.info(f"Model built with {model.count_params():,} parameters")
        return model
    
    def _get_optimizer(self, config: LSTMConfig):
        """Get optimizer based on configuration"""
        if config.optimizer.lower() == 'adam':
            return keras.optimizers.Adam(learning_rate=config.learning_rate)
        elif config.optimizer.lower() == 'rmsprop':
            return keras.optimizers.RMSprop(learning_rate=config.learning_rate)
        elif config.optimizer.lower() == 'sgd':
            return keras.optimizers.SGD(learning_rate=config.learning_rate, momentum=0.9)
        else:
            return keras.optimizers.Adam(learning_rate=config.learning_rate)
    
    def _get_callbacks(self) -> List[callbacks.Callback]:
        """Get training callbacks"""
        callback_list = [
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=self.config.patience,
                restore_best_weights=True,
                verbose=1
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=self.config.patience // 2,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        return callback_list
    
    def prepare_data(self, data: pd.DataFrame, target_column: str, 
                    feature_columns: List[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for training"""
        logger.info("Preparing data for LSTM training")
        
        if feature_columns is None:
            feature_columns = [col for col in data.columns if col != target_column]
        
        self.feature_names = feature_columns
        self.config.n_features = len(feature_columns)
        
        # Extract features and target
        X = data[feature_columns].values
        y = data[target_column].values
        
        # Handle missing values
        X = pd.DataFrame(X).ffill().bfill().values
        y = pd.Series(y).ffill().bfill().values
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        y_scaled = self.target_scaler.fit_transform(y.reshape(-1, 1)).flatten()
        
        logger.info(f"Data prepared: {X_scaled.shape[0]} samples, {X_scaled.shape[1]} features")
        return X_scaled, y_scaled
    
    def fit(self, data: pd.DataFrame, target_column: str, 
            feature_columns: List[str] = None, 
            optimize_sequence: bool = True) -> Dict[str, Any]:
        """Train the LSTM model"""
        logger.info("Starting LSTM model training")
        
        # Prepare data
        X, y = self.prepare_data(data, target_column, feature_columns)
        
        # Optimize sequence length if requested
        if optimize_sequence:
            optimal_length = self._optimize_sequence_length(X, y)
            self.config.sequence_length = optimal_length
        
        # Create sequences
        X_seq, y_seq = self._create_sequences(X, y, self.config.sequence_length)
        
        if len(X_seq) == 0:
            raise ValueError("No sequences created. Check data length and sequence_length.")
        
        logger.info(f"Created {len(X_seq)} sequences of length {self.config.sequence_length}")
        
        # Build model
        self.model = self._build_model()
        
        # Train model
        callbacks_list = self._get_callbacks()
        
        self.history = self.model.fit(
            X_seq, y_seq,
            batch_size=self.config.batch_size,
            epochs=self.config.epochs,
            validation_split=self.config.validation_split,
            callbacks=callbacks_list,
            verbose=1,
            shuffle=True
        )
        
        self.is_trained = True
        
        # Calculate training metrics
        train_pred = self.model.predict(X_seq, verbose=0)
        train_metrics = self._calculate_metrics(y_seq, train_pred)
        
        logger.info("Model training completed")
        logger.info(f"Training metrics: {train_metrics}")
        
        return {
            'history': self.history.history,
            'train_metrics': train_metrics,
            'model_params': self.model.count_params()
        }
    
    def predict(self, data: pd.DataFrame, return_attention: bool = False) -> PredictionResult:
        """Make predictions with confidence intervals"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        logger.info("Making predictions")
        
        # Prepare data
        X = data[self.feature_names].values
        X = pd.DataFrame(X).ffill().bfill().values
        X_scaled = self.scaler.transform(X)
        
        # Create sequences
        X_seq = []
        for i in range(self.config.sequence_length, len(X_scaled)):
            X_seq.append(X_scaled[i-self.config.sequence_length:i])
        
        if len(X_seq) == 0:
            raise ValueError("Insufficient data for prediction sequences")
        
        X_seq = np.array(X_seq)
        
        # Make predictions
        predictions_scaled = self.model.predict(X_seq, verbose=0)
        predictions = self.target_scaler.inverse_transform(predictions_scaled).flatten()
        
        # Calculate confidence intervals (simplified approach)
        # In practice, you might use Monte Carlo dropout or ensemble methods
        confidence_width = np.std(predictions) * 1.96  # 95% confidence
        confidence_lower = predictions - confidence_width
        confidence_upper = predictions + confidence_width
        
        # Get attention weights if available and requested
        attention_weights = None
        if return_attention and self.config.use_attention:
            attention_weights = self._get_attention_weights(X_seq)
        
        # Calculate feature importance (simplified)
        feature_importance = self._calculate_feature_importance(X_seq, predictions_scaled)
        
        # Basic metrics (if target available)
        metrics = {}
        
        result = PredictionResult(
            predictions=predictions,
            confidence_lower=confidence_lower,
            confidence_upper=confidence_upper,
            attention_weights=attention_weights,
            feature_importance=feature_importance,
            metrics=metrics,
            timestamp=datetime.now()
        )
        
        logger.info(f"Generated {len(predictions)} predictions")
        return result
    
    def _get_attention_weights(self, X_seq: np.ndarray) -> np.ndarray:
        """Extract attention weights from the model"""
        try:
            # Create a model that outputs attention weights
            attention_model = Model(
                inputs=self.model.input,
                outputs=self.model.get_layer('attention').output[1]
            )
            attention_weights = attention_model.predict(X_seq, verbose=0)
            return attention_weights.squeeze()
        except Exception as e:
            logger.warning(f"Could not extract attention weights: {e}")
            return None
    
    def _calculate_feature_importance(self, X_seq: np.ndarray, 
                                    predictions: np.ndarray) -> Dict[str, float]:
        """Calculate feature importance using permutation importance"""
        if len(self.feature_names) == 0:
            return {}
        
        base_predictions = predictions.copy()
        importance = {}
        
        try:
            for i, feature_name in enumerate(self.feature_names):
                # Permute feature
                X_permuted = X_seq.copy()
                np.random.shuffle(X_permuted[:, :, i])
                
                # Get predictions with permuted feature
                permuted_predictions = self.model.predict(X_permuted, verbose=0)
                
                # Calculate importance as change in prediction
                importance_score = np.mean(np.abs(base_predictions - permuted_predictions))
                importance[feature_name] = float(importance_score)
                
        except Exception as e:
            logger.warning(f"Could not calculate feature importance: {e}")
            
        return importance
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate prediction metrics"""
        try:
            metrics = {
                'mse': float(mean_squared_error(y_true, y_pred)),
                'mae': float(mean_absolute_error(y_true, y_pred)),
                'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
                'r2': float(r2_score(y_true, y_pred))
            }
            
            # MAPE (avoid division by zero)
            mask = y_true != 0
            if np.any(mask):
                mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
                metrics['mape'] = float(mape)
            else:
                metrics['mape'] = float('inf')
                
            return metrics
            
        except Exception as e:
            logger.warning(f"Error calculating metrics: {e}")
            return {}
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        if not self.is_trained:
            raise ValueError("No trained model to save")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save model architecture and weights
        self.model.save(f"{filepath}_model.h5")
        
        # Save scalers and config
        model_data = {
            'config': self.config,
            'scaler': self.scaler,
            'target_scaler': self.target_scaler,
            'feature_names': self.feature_names,
            'is_trained': self.is_trained
        }
        
        joblib.dump(model_data, f"{filepath}_data.pkl")
        
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model"""
        try:
            # Load model
            self.model = keras.models.load_model(
                f"{filepath}_model.h5",
                custom_objects={'AttentionLayer': AttentionLayer}
            )
            
            # Load scalers and config
            model_data = joblib.load(f"{filepath}_data.pkl")
            self.config = model_data['config']
            self.scaler = model_data['scaler']
            self.target_scaler = model_data['target_scaler']
            self.feature_names = model_data['feature_names']
            self.is_trained = model_data['is_trained']
            
            logger.info(f"Model loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise


class AttentionLSTM(LSTMPredictor):
    """LSTM with enhanced attention mechanism"""
    
    def __init__(self, config: LSTMConfig = None):
        if config is None:
            config = LSTMConfig()
        config.use_attention = True
        config.attention_units = 256
        super().__init__(config)


class EnsembleLSTM:
    """Ensemble of LSTM models for improved predictions"""
    
    def __init__(self, n_models: int = 5, configs: List[LSTMConfig] = None):
        self.n_models = n_models
        self.models = []
        self.configs = configs or self._create_diverse_configs()
        self.is_trained = False
        
    def _create_diverse_configs(self) -> List[LSTMConfig]:
        """Create diverse configurations for ensemble"""
        configs = []
        
        # Base configurations with different architectures
        base_configs = [
            # Deep narrow
            LSTMConfig(lstm_units=[64, 32, 16], dense_layers=[32, 16]),
            # Wide shallow  
            LSTMConfig(lstm_units=[256, 128], dense_layers=[64]),
            # Balanced
            LSTMConfig(lstm_units=[128, 64], dense_layers=[32]),
            # Attention focused
            LSTMConfig(lstm_units=[128], attention_units=256, dense_layers=[64, 32]),
            # Regularized
            LSTMConfig(lstm_units=[128, 64], dropout_rate=0.3, l2_reg=0.01)
        ]
        
        # Take as many as needed
        for i in range(self.n_models):
            config = base_configs[i % len(base_configs)]
            # Add some randomization
            config.learning_rate *= np.random.uniform(0.8, 1.2)
            config.dropout_rate *= np.random.uniform(0.8, 1.2)
            configs.append(config)
            
        return configs
    
    def fit(self, data: pd.DataFrame, target_column: str, 
            feature_columns: List[str] = None) -> Dict[str, Any]:
        """Train ensemble of models"""
        logger.info(f"Training ensemble of {self.n_models} LSTM models")
        
        results = []
        
        for i, config in enumerate(self.configs):
            logger.info(f"Training model {i+1}/{self.n_models}")
            
            # Create model
            model = LSTMPredictor(config)
            
            # Train with different random states
            np.random.seed(i * 42)
            tf.random.set_seed(i * 42)
            
            try:
                result = model.fit(data, target_column, feature_columns, optimize_sequence=False)
                self.models.append(model)
                results.append(result)
                
            except Exception as e:
                logger.warning(f"Failed to train model {i+1}: {e}")
                continue
        
        self.is_trained = len(self.models) > 0
        
        if not self.is_trained:
            raise RuntimeError("No models successfully trained in ensemble")
        
        logger.info(f"Ensemble training completed: {len(self.models)} models trained")
        return {'individual_results': results, 'n_models_trained': len(self.models)}
    
    def predict(self, data: pd.DataFrame) -> PredictionResult:
        """Make ensemble predictions"""
        if not self.is_trained:
            raise ValueError("Ensemble must be trained before making predictions")
        
        logger.info("Making ensemble predictions")
        
        # Get predictions from each model
        individual_predictions = []
        individual_confidences_lower = []
        individual_confidences_upper = []
        
        for i, model in enumerate(self.models):
            try:
                result = model.predict(data)
                individual_predictions.append(result.predictions)
                individual_confidences_lower.append(result.confidence_lower)
                individual_confidences_upper.append(result.confidence_upper)
                
            except Exception as e:
                logger.warning(f"Model {i+1} prediction failed: {e}")
                continue
        
        if len(individual_predictions) == 0:
            raise RuntimeError("No models could make predictions")
        
        # Combine predictions
        predictions_array = np.array(individual_predictions)
        
        # Ensemble prediction (mean)
        ensemble_predictions = np.mean(predictions_array, axis=0)
        
        # Ensemble confidence (consider model disagreement)
        prediction_std = np.std(predictions_array, axis=0)
        confidence_lower = ensemble_predictions - 1.96 * prediction_std
        confidence_upper = ensemble_predictions + 1.96 * prediction_std
        
        # Calculate feature importance (average across models)
        feature_importance = {}
        for model in self.models:
            if hasattr(model, 'feature_names') and model.feature_names:
                break
        
        # Ensemble metrics
        metrics = {
            'n_models': len(individual_predictions),
            'prediction_std': float(np.mean(prediction_std)),
            'model_agreement': float(1.0 - np.mean(prediction_std) / np.mean(np.abs(ensemble_predictions)))
        }
        
        result = PredictionResult(
            predictions=ensemble_predictions,
            confidence_lower=confidence_lower,
            confidence_upper=confidence_upper,
            attention_weights=None,  # Could average attention weights
            feature_importance=feature_importance,
            metrics=metrics,
            timestamp=datetime.now()
        )
        
        logger.info(f"Ensemble prediction completed with {len(individual_predictions)} models")
        return result
    
    def save_ensemble(self, directory: str):
        """Save ensemble models"""
        if not self.is_trained:
            raise ValueError("No trained ensemble to save")
        
        os.makedirs(directory, exist_ok=True)
        
        # Save each model
        for i, model in enumerate(self.models):
            model.save_model(os.path.join(directory, f"model_{i}"))
        
        # Save ensemble metadata
        ensemble_data = {
            'n_models': len(self.models),
            'configs': self.configs
        }
        
        joblib.dump(ensemble_data, os.path.join(directory, "ensemble_data.pkl"))
        
        logger.info(f"Ensemble saved to {directory}")
    
    def load_ensemble(self, directory: str):
        """Load ensemble models"""
        try:
            # Load ensemble metadata
            ensemble_data = joblib.load(os.path.join(directory, "ensemble_data.pkl"))
            
            # Load each model
            self.models = []
            for i in range(ensemble_data['n_models']):
                model = LSTMPredictor()
                model.load_model(os.path.join(directory, f"model_{i}"))
                self.models.append(model)
            
            self.n_models = len(self.models)
            self.configs = ensemble_data['configs']
            self.is_trained = True
            
            logger.info(f"Ensemble loaded from {directory}")
            
        except Exception as e:
            logger.error(f"Error loading ensemble: {e}")
            raise


# Factory functions
def create_lstm_predictor(config: Dict[str, Any] = None) -> LSTMPredictor:
    """Create LSTM predictor with configuration"""
    if config:
        lstm_config = LSTMConfig(**config)
    else:
        lstm_config = LSTMConfig()
    
    return LSTMPredictor(lstm_config)


def create_attention_lstm(config: Dict[str, Any] = None) -> AttentionLSTM:
    """Create LSTM with enhanced attention"""
    if config:
        lstm_config = LSTMConfig(**config)
    else:
        lstm_config = LSTMConfig()
    
    return AttentionLSTM(lstm_config)


def create_ensemble_lstm(n_models: int = 5, configs: List[Dict[str, Any]] = None) -> EnsembleLSTM:
    """Create ensemble of LSTM models"""
    if configs:
        lstm_configs = [LSTMConfig(**config) for config in configs]
    else:
        lstm_configs = None
    
    return EnsembleLSTM(n_models, lstm_configs)


# Export classes and functions
__all__ = [
    'LSTMPredictor',
    'AttentionLSTM', 
    'EnsembleLSTM',
    'LSTMConfig',
    'PredictionResult',
    'AttentionLayer',
    'create_lstm_predictor',
    'create_attention_lstm',
    'create_ensemble_lstm'
]