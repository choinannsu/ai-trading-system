"""
CNN-LSTM Hybrid Model
Combines CNN for local pattern extraction with LSTM for temporal dependencies
Includes skip connections and residual blocks for improved learning
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
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from utils.logger import get_logger

logger = get_logger(__name__)

warnings.filterwarnings('ignore', category=UserWarning)
tf.get_logger().setLevel('ERROR')


@dataclass
class HybridConfig:
    """CNN-LSTM hybrid model configuration"""
    sequence_length: int = 60
    n_features: int = 5
    
    # CNN configuration
    cnn_filters: List[int] = None
    cnn_kernel_sizes: List[int] = None
    cnn_strides: List[int] = None
    cnn_activation: str = 'relu'
    cnn_dropout: float = 0.2
    use_batch_norm: bool = True
    use_residual: bool = True
    
    # LSTM configuration
    lstm_units: List[int] = None
    lstm_dropout: float = 0.2
    lstm_recurrent_dropout: float = 0.2
    use_bidirectional: bool = True
    lstm_activation: str = 'tanh'
    
    # Skip connections
    use_skip_connections: bool = True
    skip_connection_type: str = 'add'  # 'add', 'concat'
    
    # Dense layers
    dense_layers: List[int] = None
    dense_dropout: float = 0.3
    
    # Training configuration
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 100
    patience: int = 15
    validation_split: float = 0.2
    l2_reg: float = 0.001
    optimizer: str = 'adam'
    
    def __post_init__(self):
        if self.cnn_filters is None:
            self.cnn_filters = [64, 128, 256, 128]
        if self.cnn_kernel_sizes is None:
            self.cnn_kernel_sizes = [3, 3, 3, 3]
        if self.cnn_strides is None:
            self.cnn_strides = [1, 1, 2, 1]
        if self.lstm_units is None:
            self.lstm_units = [128, 64]
        if self.dense_layers is None:
            self.dense_layers = [64, 32]


@dataclass
class HybridPredictionResult:
    """Hybrid model prediction result with feature analysis"""
    predictions: np.ndarray
    confidence_lower: np.ndarray
    confidence_upper: np.ndarray
    cnn_features: np.ndarray  # Extracted CNN features
    lstm_features: np.ndarray  # LSTM hidden states
    feature_importance: Dict[str, float]
    temporal_patterns: Dict[str, Any]  # Detected temporal patterns
    local_patterns: Dict[str, Any]  # Detected local patterns
    metrics: Dict[str, float]
    timestamp: datetime


class ResidualBlock(layers.Layer):
    """Residual block for CNN"""
    
    def __init__(self, filters: int, kernel_size: int, stride: int = 1, 
                 dropout_rate: float = 0.2, **kwargs):
        super(ResidualBlock, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.dropout_rate = dropout_rate
        
    def build(self, input_shape):
        # Main path
        self.conv1 = layers.Conv1D(
            self.filters, self.kernel_size, strides=self.stride, 
            padding='same', activation='relu', name='conv1'
        )
        self.bn1 = layers.BatchNormalization(name='bn1')
        self.dropout1 = layers.Dropout(self.dropout_rate, name='dropout1')
        
        self.conv2 = layers.Conv1D(
            self.filters, self.kernel_size, strides=1, 
            padding='same', activation=None, name='conv2'
        )
        self.bn2 = layers.BatchNormalization(name='bn2')
        
        # Skip connection
        if self.stride != 1 or input_shape[-1] != self.filters:
            self.skip_conv = layers.Conv1D(
                self.filters, 1, strides=self.stride, 
                padding='same', activation=None, name='skip_conv'
            )
            self.skip_bn = layers.BatchNormalization(name='skip_bn')
        else:
            self.skip_conv = None
            self.skip_bn = None
        
        self.activation = layers.Activation('relu', name='final_activation')
        self.dropout2 = layers.Dropout(self.dropout_rate, name='dropout2')
        
        super(ResidualBlock, self).build(input_shape)
    
    def call(self, inputs, training=None):
        # Main path
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.dropout1(x, training=training)
        
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        
        # Skip connection
        if self.skip_conv is not None:
            skip = self.skip_conv(inputs)
            skip = self.skip_bn(skip, training=training)
        else:
            skip = inputs
        
        # Add skip connection
        x = x + skip
        x = self.activation(x)
        x = self.dropout2(x, training=training)
        
        return x
    
    def get_config(self):
        config = super(ResidualBlock, self).get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'stride': self.stride,
            'dropout_rate': self.dropout_rate
        })
        return config


class AttentionPooling(layers.Layer):
    """Attention-based pooling layer"""
    
    def __init__(self, units: int = 128, **kwargs):
        super(AttentionPooling, self).__init__(**kwargs)
        self.units = units
        
    def build(self, input_shape):
        self.W = self.add_weight(
            name='attention_weight',
            shape=(input_shape[-1], self.units),
            initializer='glorot_uniform',
            trainable=True
        )
        self.b = self.add_weight(
            name='attention_bias',
            shape=(self.units,),
            initializer='zeros',
            trainable=True
        )
        self.u = self.add_weight(
            name='attention_context',
            shape=(self.units,),
            initializer='glorot_uniform',
            trainable=True
        )
        super(AttentionPooling, self).build(input_shape)
    
    def call(self, inputs):
        # Calculate attention scores
        uit = tf.nn.tanh(tf.tensordot(inputs, self.W, axes=1) + self.b)
        ait = tf.tensordot(uit, self.u, axes=1)
        ait = tf.nn.softmax(ait, axis=1)
        
        # Apply attention weights
        ait = tf.expand_dims(ait, -1)
        weighted_input = inputs * ait
        output = tf.reduce_sum(weighted_input, axis=1)
        
        return output, ait
    
    def compute_output_shape(self, input_shape):
        return [(input_shape[0], input_shape[2]), (input_shape[0], input_shape[1], 1)]
    
    def get_config(self):
        config = super(AttentionPooling, self).get_config()
        config.update({'units': self.units})
        return config


class CNNLSTMHybrid:
    """CNN-LSTM hybrid model for time series prediction"""
    
    def __init__(self, config: HybridConfig = None):
        self.config = config or HybridConfig()
        self.model = None
        self.feature_extractor = None  # For extracting intermediate features
        self.scaler = StandardScaler()
        self.target_scaler = MinMaxScaler()
        self.history = None
        self.feature_names = None
        self.is_trained = False
        
    def _build_cnn_branch(self, inputs):
        """Build CNN branch for local pattern extraction"""
        x = inputs
        skip_connections = []
        
        logger.debug("Building CNN branch for local pattern extraction")
        
        for i, (filters, kernel_size, stride) in enumerate(
            zip(self.config.cnn_filters, self.config.cnn_kernel_sizes, self.config.cnn_strides)
        ):
            
            if self.config.use_residual and i > 0:
                # Use residual blocks after first layer
                x = ResidualBlock(
                    filters, kernel_size, stride, 
                    self.config.cnn_dropout,
                    name=f'residual_block_{i}'
                )(x)
            else:
                # Standard conv layer
                x = layers.Conv1D(
                    filters, kernel_size, strides=stride,
                    padding='same', activation=self.config.cnn_activation,
                    kernel_regularizer=keras.regularizers.l2(self.config.l2_reg),
                    name=f'conv1d_{i}'
                )(x)
                
                if self.config.use_batch_norm:
                    x = layers.BatchNormalization(name=f'bn_conv_{i}')(x)
                
                x = layers.Dropout(self.config.cnn_dropout, name=f'dropout_conv_{i}')(x)
            
            # Store for skip connections
            if self.config.use_skip_connections and i < len(self.config.cnn_filters) - 1:
                skip_connections.append(x)
        
        # Global feature extraction with attention pooling
        cnn_features, attention_weights = AttentionPooling(
            units=128, name='cnn_attention_pooling'
        )(x)
        
        return cnn_features, x, skip_connections, attention_weights
    
    def _build_lstm_branch(self, inputs):
        """Build LSTM branch for temporal dependency learning"""
        x = inputs
        lstm_outputs = []
        
        logger.debug("Building LSTM branch for temporal dependencies")
        
        for i, units in enumerate(self.config.lstm_units):
            return_sequences = i < len(self.config.lstm_units) - 1
            
            if self.config.use_bidirectional:
                lstm_layer = layers.Bidirectional(
                    layers.LSTM(
                        units,
                        return_sequences=return_sequences,
                        dropout=self.config.lstm_dropout,
                        recurrent_dropout=self.config.lstm_recurrent_dropout,
                        kernel_regularizer=keras.regularizers.l2(self.config.l2_reg),
                        activation=self.config.lstm_activation,
                        return_state=return_sequences  # Return states for intermediate outputs
                    ),
                    name=f'bidirectional_lstm_{i}'
                )
            else:
                lstm_layer = layers.LSTM(
                    units,
                    return_sequences=return_sequences,
                    dropout=self.config.lstm_dropout,
                    recurrent_dropout=self.config.lstm_recurrent_dropout,
                    kernel_regularizer=keras.regularizers.l2(self.config.l2_reg),
                    activation=self.config.lstm_activation,
                    return_state=return_sequences,
                    name=f'lstm_{i}'
                )
            
            if return_sequences and self.config.use_bidirectional:
                x, forward_h, forward_c, backward_h, backward_c = lstm_layer(x)
                # Store hidden states for analysis
                lstm_outputs.append(tf.concat([forward_h, backward_h], axis=-1))
            elif return_sequences:
                x, h, c = lstm_layer(x)
                lstm_outputs.append(h)
            else:
                x = lstm_layer(x)
                lstm_outputs.append(x)
        
        return x, lstm_outputs
    
    def _apply_skip_connections(self, cnn_features, lstm_features, skip_connections):
        """Apply skip connections between CNN and LSTM branches"""
        if not self.config.use_skip_connections or not skip_connections:
            return tf.concat([cnn_features, lstm_features], axis=-1)
        
        # Adaptive pooling for skip connections to match dimensions
        pooled_skips = []
        for skip in skip_connections:
            # Global average pooling to reduce to feature vector
            pooled = layers.GlobalAveragePooling1D()(skip)
            pooled_skips.append(pooled)
        
        # Combine all features
        if self.config.skip_connection_type == 'concat':
            combined = tf.concat([cnn_features, lstm_features] + pooled_skips, axis=-1)
        else:  # 'add'
            # Project to same dimension for addition
            target_dim = min(cnn_features.shape[-1], lstm_features.shape[-1])
            
            cnn_proj = layers.Dense(target_dim, name='cnn_projection')(cnn_features)
            lstm_proj = layers.Dense(target_dim, name='lstm_projection')(lstm_features)
            
            combined = cnn_proj + lstm_proj
            
            # Add skip connections
            for i, skip in enumerate(pooled_skips):
                skip_proj = layers.Dense(target_dim, name=f'skip_projection_{i}')(skip)
                combined = combined + skip_proj
        
        return combined
    
    def _build_model(self) -> Model:
        """Build the complete CNN-LSTM hybrid model"""
        logger.info("Building CNN-LSTM hybrid model")
        
        # Input layer
        inputs = layers.Input(
            shape=(self.config.sequence_length, self.config.n_features),
            name='input_layer'
        )
        
        # CNN branch for local pattern extraction
        cnn_features, cnn_output, skip_connections, cnn_attention = self._build_cnn_branch(inputs)
        
        # LSTM branch for temporal dependencies
        lstm_features, lstm_outputs = self._build_lstm_branch(inputs)
        
        # Combine features with skip connections
        combined_features = self._apply_skip_connections(
            cnn_features, lstm_features, skip_connections
        )
        
        # Dense layers for final processing
        x = combined_features
        for i, units in enumerate(self.config.dense_layers):
            x = layers.Dense(
                units,
                activation='relu',
                kernel_regularizer=keras.regularizers.l2(self.config.l2_reg),
                name=f'dense_{i}'
            )(x)
            x = layers.Dropout(self.config.dense_dropout, name=f'dropout_dense_{i}')(x)
            
            if self.config.use_batch_norm:
                x = layers.BatchNormalization(name=f'bn_dense_{i}')(x)
        
        # Output layer
        outputs = layers.Dense(1, activation='linear', name='output')(x)
        
        # Create main model
        model = Model(inputs=inputs, outputs=outputs, name='cnn_lstm_hybrid')
        
        # Create feature extractor model for analysis
        self.feature_extractor = Model(
            inputs=inputs,
            outputs={
                'cnn_features': cnn_features,
                'lstm_features': lstm_features,
                'combined_features': combined_features,
                'cnn_attention': cnn_attention,
                'lstm_outputs': lstm_outputs
            },
            name='feature_extractor'
        )
        
        # Compile model
        optimizer = self._get_optimizer()
        
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae', 'mape']
        )
        
        logger.info(f"Hybrid model built with {model.count_params():,} parameters")
        return model
    
    def _get_optimizer(self):
        """Get optimizer based on configuration"""
        if self.config.optimizer.lower() == 'adam':
            return keras.optimizers.Adam(learning_rate=self.config.learning_rate)
        elif self.config.optimizer.lower() == 'rmsprop':
            return keras.optimizers.RMSprop(learning_rate=self.config.learning_rate)
        elif self.config.optimizer.lower() == 'sgd':
            return keras.optimizers.SGD(
                learning_rate=self.config.learning_rate, 
                momentum=0.9
            )
        else:
            return keras.optimizers.Adam(learning_rate=self.config.learning_rate)
    
    def _create_sequences(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for hybrid model training"""
        X_seq, y_seq = [], []
        
        for i in range(self.config.sequence_length, len(X)):
            X_seq.append(X[i-self.config.sequence_length:i])
            y_seq.append(y[i])
        
        return np.array(X_seq), np.array(y_seq)
    
    def prepare_data(self, data: pd.DataFrame, target_column: str,
                    feature_columns: List[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for hybrid model training"""
        logger.info("Preparing data for CNN-LSTM hybrid training")
        
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
        
        # Scale data
        X_scaled = self.scaler.fit_transform(X)
        y_scaled = self.target_scaler.fit_transform(y.reshape(-1, 1)).flatten()
        
        logger.info(f"Data prepared: {X_scaled.shape[0]} samples, {X_scaled.shape[1]} features")
        return X_scaled, y_scaled
    
    def fit(self, data: pd.DataFrame, target_column: str,
            feature_columns: List[str] = None) -> Dict[str, Any]:
        """Train the hybrid model"""
        logger.info("Starting CNN-LSTM hybrid model training")
        
        # Prepare data
        X, y = self.prepare_data(data, target_column, feature_columns)
        
        # Create sequences
        X_seq, y_seq = self._create_sequences(X, y)
        
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
        
        logger.info("Hybrid model training completed")
        logger.info(f"Training metrics: {train_metrics}")
        
        return {
            'history': self.history.history,
            'train_metrics': train_metrics,
            'model_params': self.model.count_params()
        }
    
    def predict(self, data: pd.DataFrame, extract_features: bool = True) -> HybridPredictionResult:
        """Make predictions with feature analysis"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        logger.info("Making hybrid model predictions")
        
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
        
        # Calculate confidence intervals
        confidence_width = np.std(predictions) * 1.96
        confidence_lower = predictions - confidence_width
        confidence_upper = predictions + confidence_width
        
        # Extract features if requested
        cnn_features = None
        lstm_features = None
        if extract_features and self.feature_extractor is not None:
            features = self.feature_extractor.predict(X_seq, verbose=0)
            cnn_features = features['cnn_features']
            lstm_features = features['lstm_features']
        
        # Calculate feature importance
        feature_importance = self._calculate_feature_importance(X_seq, predictions_scaled)
        
        # Analyze patterns
        temporal_patterns = self._analyze_temporal_patterns(X_seq, predictions)
        local_patterns = self._analyze_local_patterns(cnn_features if cnn_features is not None else X_seq)
        
        result = HybridPredictionResult(
            predictions=predictions,
            confidence_lower=confidence_lower,
            confidence_upper=confidence_upper,
            cnn_features=cnn_features,
            lstm_features=lstm_features,
            feature_importance=feature_importance,
            temporal_patterns=temporal_patterns,
            local_patterns=local_patterns,
            metrics={},
            timestamp=datetime.now()
        )
        
        logger.info(f"Generated {len(predictions)} hybrid predictions")
        return result
    
    def _calculate_feature_importance(self, X_seq: np.ndarray, 
                                    predictions: np.ndarray) -> Dict[str, float]:
        """Calculate feature importance using permutation importance"""
        if len(self.feature_names) == 0:
            return {}
        
        importance = {}
        base_predictions = predictions.copy()
        
        try:
            for i, feature_name in enumerate(self.feature_names):
                # Permute feature across all time steps
                X_permuted = X_seq.copy()
                for t in range(X_seq.shape[1]):
                    np.random.shuffle(X_permuted[:, t, i])
                
                # Get predictions with permuted feature
                permuted_predictions = self.model.predict(X_permuted, verbose=0)
                
                # Calculate importance
                importance_score = np.mean(np.abs(base_predictions - permuted_predictions.flatten()))
                importance[feature_name] = float(importance_score)
                
        except Exception as e:
            logger.warning(f"Could not calculate feature importance: {e}")
            
        return importance
    
    def _analyze_temporal_patterns(self, X_seq: np.ndarray, predictions: np.ndarray) -> Dict[str, Any]:
        """Analyze temporal patterns in the data"""
        patterns = {}
        
        try:
            # Trend analysis
            if len(predictions) > 1:
                trends = np.diff(predictions)
                patterns['trend_volatility'] = float(np.std(trends))
                patterns['avg_trend'] = float(np.mean(trends))
                patterns['trend_changes'] = int(np.sum(np.diff(np.sign(trends)) != 0))
            
            # Periodicity detection (simplified)
            if len(predictions) >= 12:
                # Autocorrelation for different lags
                autocorrs = []
                for lag in [1, 7, 30]:  # Daily, weekly, monthly
                    if lag < len(predictions):
                        autocorr = np.corrcoef(
                            predictions[:-lag], predictions[lag:]
                        )[0, 1]
                        if not np.isnan(autocorr):
                            autocorrs.append((lag, autocorr))
                
                patterns['autocorrelations'] = autocorrs
            
            # Volatility clustering
            if len(predictions) > 20:
                returns = np.diff(predictions) / predictions[:-1]
                volatility = np.abs(returns)
                vol_autocorr = np.corrcoef(volatility[:-1], volatility[1:])[0, 1]
                if not np.isnan(vol_autocorr):
                    patterns['volatility_clustering'] = float(vol_autocorr)
            
        except Exception as e:
            logger.warning(f"Error analyzing temporal patterns: {e}")
            
        return patterns
    
    def _analyze_local_patterns(self, features: np.ndarray) -> Dict[str, Any]:
        """Analyze local patterns extracted by CNN"""
        patterns = {}
        
        if features is None:
            return patterns
        
        try:
            # Feature activation statistics
            patterns['mean_activation'] = float(np.mean(features))
            patterns['activation_std'] = float(np.std(features))
            patterns['activation_skewness'] = float(np.mean((features - np.mean(features))**3) / np.std(features)**3)
            
            # Sparsity analysis
            activation_threshold = 0.1 * np.max(features)
            sparsity = np.mean(features < activation_threshold)
            patterns['sparsity'] = float(sparsity)
            
            # Pattern diversity (number of distinct activation patterns)
            if len(features.shape) > 1:
                # Cluster features to find distinct patterns
                unique_patterns = len(np.unique(features.round(decimals=2), axis=0))
                patterns['pattern_diversity'] = unique_patterns / len(features)
            
        except Exception as e:
            logger.warning(f"Error analyzing local patterns: {e}")
            
        return patterns
    
    def visualize_features(self, result: HybridPredictionResult, save_path: str = None) -> plt.Figure:
        """Visualize extracted features and patterns"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('CNN-LSTM Hybrid Feature Analysis', fontsize=16)
        
        # CNN features visualization
        if result.cnn_features is not None:
            cnn_sample = result.cnn_features[:min(100, len(result.cnn_features))]
            axes[0, 0].hist(cnn_sample.flatten(), bins=50, alpha=0.7, color='blue')
            axes[0, 0].set_title('CNN Feature Distribution')
            axes[0, 0].set_xlabel('Feature Value')
            axes[0, 0].set_ylabel('Frequency')
        
        # LSTM features visualization
        if result.lstm_features is not None:
            lstm_sample = result.lstm_features[:min(100, len(result.lstm_features))]
            axes[0, 1].hist(lstm_sample.flatten(), bins=50, alpha=0.7, color='green')
            axes[0, 1].set_title('LSTM Feature Distribution')
            axes[0, 1].set_xlabel('Feature Value')
            axes[0, 1].set_ylabel('Frequency')
        
        # Feature importance
        if result.feature_importance:
            features = list(result.feature_importance.keys())
            importance = list(result.feature_importance.values())
            
            axes[1, 0].barh(features, importance)
            axes[1, 0].set_title('Feature Importance')
            axes[1, 0].set_xlabel('Importance Score')
        
        # Temporal patterns
        if result.temporal_patterns:
            pattern_names = list(result.temporal_patterns.keys())
            pattern_values = [v for v in result.temporal_patterns.values() 
                            if isinstance(v, (int, float))]
            pattern_names = pattern_names[:len(pattern_values)]
            
            if pattern_values:
                axes[1, 1].bar(pattern_names, pattern_values)
                axes[1, 1].set_title('Temporal Patterns')
                axes[1, 1].set_ylabel('Pattern Strength')
                axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Feature visualization saved to {save_path}")
        
        return fig
    
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
                factor=0.7,
                patience=self.config.patience // 2,
                min_lr=1e-7,
                verbose=1
            ),
            callbacks.ModelCheckpoint(
                filepath='best_hybrid_model.h5',
                monitor='val_loss',
                save_best_only=True,
                verbose=0
            )
        ]
        
        return callback_list
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate prediction metrics"""
        try:
            metrics = {
                'mse': float(mean_squared_error(y_true, y_pred)),
                'mae': float(mean_absolute_error(y_true, y_pred)),
                'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
                'r2': float(r2_score(y_true, y_pred))
            }
            
            # MAPE
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
        
        # Save main model
        self.model.save(f"{filepath}_model.h5")
        
        # Save feature extractor if available
        if self.feature_extractor is not None:
            self.feature_extractor.save(f"{filepath}_features.h5")
        
        # Save scalers and config
        model_data = {
            'config': self.config,
            'scaler': self.scaler,
            'target_scaler': self.target_scaler,
            'feature_names': self.feature_names,
            'is_trained': self.is_trained
        }
        
        joblib.dump(model_data, f"{filepath}_data.pkl")
        
        logger.info(f"Hybrid model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model"""
        try:
            # Custom objects for loading
            custom_objects = {
                'ResidualBlock': ResidualBlock,
                'AttentionPooling': AttentionPooling
            }
            
            # Load main model
            self.model = keras.models.load_model(
                f"{filepath}_model.h5",
                custom_objects=custom_objects
            )
            
            # Load feature extractor if available
            try:
                self.feature_extractor = keras.models.load_model(
                    f"{filepath}_features.h5",
                    custom_objects=custom_objects
                )
            except:
                logger.warning("Could not load feature extractor model")
                self.feature_extractor = None
            
            # Load scalers and config
            model_data = joblib.load(f"{filepath}_data.pkl")
            self.config = model_data['config']
            self.scaler = model_data['scaler']
            self.target_scaler = model_data['target_scaler']
            self.feature_names = model_data['feature_names']
            self.is_trained = model_data['is_trained']
            
            logger.info(f"Hybrid model loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise


# Factory function
def create_hybrid_model(config: Dict[str, Any] = None) -> CNNLSTMHybrid:
    """Create CNN-LSTM hybrid model with configuration"""
    if config:
        hybrid_config = HybridConfig(**config)
    else:
        hybrid_config = HybridConfig()
    
    return CNNLSTMHybrid(hybrid_config)


# Export classes and functions
__all__ = [
    'CNNLSTMHybrid',
    'HybridConfig',
    'HybridPredictionResult',
    'ResidualBlock',
    'AttentionPooling',
    'create_hybrid_model'
]