"""
Transformer Model for Time Series Prediction
Custom transformer architecture with positional encoding, attention visualization, and pattern interpretation
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
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from utils.logger import get_logger

logger = get_logger(__name__)

warnings.filterwarnings('ignore', category=UserWarning)
tf.get_logger().setLevel('ERROR')


@dataclass
class TransformerConfig:
    """Transformer model configuration"""
    sequence_length: int = 60
    n_features: int = 5
    d_model: int = 128
    n_heads: int = 8
    n_transformer_blocks: int = 4
    ff_dim: int = 512
    dropout_rate: float = 0.1
    attention_dropout: float = 0.1
    mlp_dropout: float = 0.1
    layer_norm_eps: float = 1e-6
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 100
    patience: int = 15
    validation_split: float = 0.2
    use_positional_encoding: bool = True
    positional_encoding_type: str = 'sinusoidal'  # 'sinusoidal', 'learnable'
    mlp_units: List[int] = None
    output_activation: str = 'linear'
    optimizer: str = 'adam'
    warmup_steps: int = 4000
    
    def __post_init__(self):
        if self.mlp_units is None:
            self.mlp_units = [64, 32]


@dataclass
class AttentionVisualization:
    """Attention visualization data"""
    attention_weights: np.ndarray  # Shape: (batch, heads, seq_len, seq_len)
    feature_names: List[str]
    timestamps: List[str]
    layer_weights: List[np.ndarray]  # Attention weights per transformer layer
    head_importance: Dict[int, float]  # Importance score per attention head
    patterns: Dict[str, Any]  # Detected attention patterns


@dataclass
class TransformerPredictionResult:
    """Transformer prediction result with attention analysis"""
    predictions: np.ndarray
    confidence_lower: np.ndarray
    confidence_upper: np.ndarray
    attention_viz: AttentionVisualization
    feature_importance: Dict[str, float]
    temporal_importance: np.ndarray  # Importance per time step
    pattern_analysis: Dict[str, Any]
    metrics: Dict[str, float]
    timestamp: datetime


class SinusoidalPositionalEncoding(layers.Layer):
    """Sinusoidal positional encoding for transformers"""
    
    def __init__(self, sequence_length: int, d_model: int, **kwargs):
        super(SinusoidalPositionalEncoding, self).__init__(**kwargs)
        self.sequence_length = sequence_length
        self.d_model = d_model
        
    def build(self, input_shape):
        # Create positional encoding matrix
        pe = np.zeros((self.sequence_length, self.d_model))
        position = np.arange(0, self.sequence_length)[:, np.newaxis]
        
        div_term = np.exp(np.arange(0, self.d_model, 2) * -(np.log(10000.0) / self.d_model))
        
        pe[:, 0::2] = np.sin(position * div_term)
        if self.d_model % 2 == 1:
            pe[:, 1::2] = np.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = np.cos(position * div_term)
        
        self.pe = self.add_weight(
            name='positional_encoding',
            shape=(1, self.sequence_length, self.d_model),
            initializer='zeros',
            trainable=False
        )
        
        self.pe.assign(pe[np.newaxis, :, :])
        super(SinusoidalPositionalEncoding, self).build(input_shape)
    
    def call(self, inputs):
        return inputs + self.pe[:, :tf.shape(inputs)[1], :]
    
    def get_config(self):
        config = super(SinusoidalPositionalEncoding, self).get_config()
        config.update({
            'sequence_length': self.sequence_length,
            'd_model': self.d_model
        })
        return config


class LearnablePositionalEncoding(layers.Layer):
    """Learnable positional encoding for transformers"""
    
    def __init__(self, sequence_length: int, d_model: int, **kwargs):
        super(LearnablePositionalEncoding, self).__init__(**kwargs)
        self.sequence_length = sequence_length
        self.d_model = d_model
        
    def build(self, input_shape):
        self.pe = self.add_weight(
            name='positional_encoding',
            shape=(1, self.sequence_length, self.d_model),
            initializer='uniform',
            trainable=True
        )
        super(LearnablePositionalEncoding, self).build(input_shape)
    
    def call(self, inputs):
        return inputs + self.pe[:, :tf.shape(inputs)[1], :]
    
    def get_config(self):
        config = super(LearnablePositionalEncoding, self).get_config()
        config.update({
            'sequence_length': self.sequence_length,
            'd_model': self.d_model
        })
        return config


class MultiHeadSelfAttention(layers.Layer):
    """Multi-head self-attention with visualization support"""
    
    def __init__(self, d_model: int, n_heads: int, dropout_rate: float = 0.1, **kwargs):
        super(MultiHeadSelfAttention, self).__init__(**kwargs)
        self.d_model = d_model
        self.n_heads = n_heads
        self.dropout_rate = dropout_rate
        
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")
        
        self.depth = d_model // n_heads
        
    def build(self, input_shape):
        self.wq = self.add_weight(
            name='query_weight',
            shape=(self.d_model, self.d_model),
            initializer='glorot_uniform',
            trainable=True
        )
        self.wk = self.add_weight(
            name='key_weight', 
            shape=(self.d_model, self.d_model),
            initializer='glorot_uniform',
            trainable=True
        )
        self.wv = self.add_weight(
            name='value_weight',
            shape=(self.d_model, self.d_model),
            initializer='glorot_uniform',
            trainable=True
        )
        self.wo = self.add_weight(
            name='output_weight',
            shape=(self.d_model, self.d_model),
            initializer='glorot_uniform',
            trainable=True
        )
        
        self.dropout = layers.Dropout(self.dropout_rate)
        super(MultiHeadSelfAttention, self).build(input_shape)
    
    def split_heads(self, x, batch_size):
        """Split the last dimension into (n_heads, depth)"""
        x = tf.reshape(x, (batch_size, -1, self.n_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def call(self, inputs, training=None, return_attention_scores=False):
        batch_size = tf.shape(inputs)[0]
        
        # Linear transformations
        q = tf.matmul(inputs, self.wq)
        k = tf.matmul(inputs, self.wk)
        v = tf.matmul(inputs, self.wv)
        
        # Split heads
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)
        
        # Scaled dot-product attention
        matmul_qk = tf.matmul(q, k, transpose_b=True)
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
        
        # Apply softmax
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        attention_weights = self.dropout(attention_weights, training=training)
        
        # Apply attention to values
        attention_output = tf.matmul(attention_weights, v)
        
        # Concatenate heads
        attention_output = tf.transpose(attention_output, perm=[0, 2, 1, 3])
        attention_output = tf.reshape(attention_output, (batch_size, -1, self.d_model))
        
        # Final linear layer
        output = tf.matmul(attention_output, self.wo)
        
        if return_attention_scores:
            return output, attention_weights
        
        return output
    
    def get_config(self):
        config = super(MultiHeadSelfAttention, self).get_config()
        config.update({
            'd_model': self.d_model,
            'n_heads': self.n_heads,
            'dropout_rate': self.dropout_rate
        })
        return config


class TransformerBlock(layers.Layer):
    """Transformer encoder block"""
    
    def __init__(self, d_model: int, n_heads: int, ff_dim: int, 
                 dropout_rate: float = 0.1, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.d_model = d_model
        self.n_heads = n_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate
        
    def build(self, input_shape):
        self.attention = MultiHeadSelfAttention(
            self.d_model, self.n_heads, self.dropout_rate, name='multi_head_attention'
        )
        self.ffn = keras.Sequential([
            layers.Dense(self.ff_dim, activation='relu', name='ffn_dense_1'),
            layers.Dropout(self.dropout_rate, name='ffn_dropout'),
            layers.Dense(self.d_model, name='ffn_dense_2')
        ], name='feed_forward')
        
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6, name='layernorm_1')
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6, name='layernorm_2')
        self.dropout1 = layers.Dropout(self.dropout_rate, name='dropout_1')
        self.dropout2 = layers.Dropout(self.dropout_rate, name='dropout_2')
        
        super(TransformerBlock, self).build(input_shape)
    
    def call(self, inputs, training=None, return_attention_scores=False):
        # Self-attention
        if return_attention_scores:
            attn_output, attention_weights = self.attention(
                inputs, training=training, return_attention_scores=True
            )
        else:
            attn_output = self.attention(inputs, training=training)
            attention_weights = None
        
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        
        # Feed forward
        ffn_output = self.ffn(out1, training=training)
        ffn_output = self.dropout2(ffn_output, training=training)
        output = self.layernorm2(out1 + ffn_output)
        
        if return_attention_scores:
            return output, attention_weights
        
        return output
    
    def get_config(self):
        config = super(TransformerBlock, self).get_config()
        config.update({
            'd_model': self.d_model,
            'n_heads': self.n_heads,
            'ff_dim': self.ff_dim,
            'dropout_rate': self.dropout_rate
        })
        return config


class TimeSeriesTransformer:
    """Transformer model for time series prediction"""
    
    def __init__(self, config: TransformerConfig = None):
        self.config = config or TransformerConfig()
        self.model = None
        self.attention_model = None  # For extracting attention weights
        self.scaler = StandardScaler()
        self.target_scaler = MinMaxScaler()
        self.history = None
        self.feature_names = None
        self.is_trained = False
        
    def _build_model(self) -> Model:
        """Build transformer model"""
        logger.info("Building Transformer model for time series")
        
        # Input layer
        inputs = layers.Input(shape=(self.config.sequence_length, self.config.n_features))
        
        # Feature projection to d_model dimensions
        x = layers.Dense(self.config.d_model, name='feature_projection')(inputs)
        
        # Positional encoding
        if self.config.use_positional_encoding:
            if self.config.positional_encoding_type == 'sinusoidal':
                pos_encoding = SinusoidalPositionalEncoding(
                    self.config.sequence_length, 
                    self.config.d_model,
                    name='positional_encoding'
                )
            else:  # learnable
                pos_encoding = LearnablePositionalEncoding(
                    self.config.sequence_length,
                    self.config.d_model, 
                    name='positional_encoding'
                )
            
            x = pos_encoding(x)
        
        # Transformer blocks
        attention_outputs = []
        for i in range(self.config.n_transformer_blocks):
            transformer_block = TransformerBlock(
                self.config.d_model,
                self.config.n_heads,
                self.config.ff_dim,
                self.config.dropout_rate,
                name=f'transformer_block_{i}'
            )
            
            if i == self.config.n_transformer_blocks - 1:
                # Return attention weights from the last layer
                x, attention_weights = transformer_block(
                    x, return_attention_scores=True
                )
                attention_outputs.append(attention_weights)
            else:
                x = transformer_block(x)
        
        # Global average pooling
        x = layers.GlobalAveragePooling1D(name='global_avg_pool')(x)
        
        # MLP head
        for i, units in enumerate(self.config.mlp_units):
            x = layers.Dense(
                units, 
                activation='relu',
                name=f'mlp_dense_{i}'
            )(x)
            x = layers.Dropout(
                self.config.mlp_dropout, 
                name=f'mlp_dropout_{i}'
            )(x)
        
        # Output layer
        outputs = layers.Dense(
            1, 
            activation=self.config.output_activation,
            name='output'
        )(x)
        
        # Create model
        model = Model(inputs=inputs, outputs=outputs, name='time_series_transformer')
        
        # Create attention extraction model
        self.attention_model = Model(
            inputs=inputs,
            outputs=attention_outputs[0] if attention_outputs else None,
            name='attention_extractor'
        )
        
        # Compile model
        optimizer = self._get_optimizer()
        
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae', 'mape']
        )
        
        logger.info(f"Transformer model built with {model.count_params():,} parameters")
        return model
    
    def _get_optimizer(self):
        """Get optimizer with learning rate scheduling"""
        if self.config.optimizer.lower() == 'adam':
            # Custom learning rate schedule for transformers
            lr_schedule = self._get_lr_schedule()
            return keras.optimizers.Adam(learning_rate=lr_schedule, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
        elif self.config.optimizer.lower() == 'rmsprop':
            return keras.optimizers.RMSprop(learning_rate=self.config.learning_rate)
        else:
            return keras.optimizers.Adam(learning_rate=self.config.learning_rate)
    
    def _get_lr_schedule(self):
        """Transformer learning rate schedule with warmup"""
        def lr_schedule(step):
            step = tf.cast(step, tf.float32)
            warmup_steps = tf.cast(self.config.warmup_steps, tf.float32)
            d_model = tf.cast(self.config.d_model, tf.float32)
            
            arg1 = tf.math.rsqrt(step)
            arg2 = step * (warmup_steps ** -1.5)
            
            return tf.math.rsqrt(d_model) * tf.math.minimum(arg1, arg2)
        
        return lr_schedule
    
    def _create_sequences(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for transformer training"""
        X_seq, y_seq = [], []
        
        for i in range(self.config.sequence_length, len(X)):
            X_seq.append(X[i-self.config.sequence_length:i])
            y_seq.append(y[i])
        
        return np.array(X_seq), np.array(y_seq)
    
    def prepare_data(self, data: pd.DataFrame, target_column: str,
                    feature_columns: List[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for transformer training"""
        logger.info("Preparing data for Transformer training")
        
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
        """Train the transformer model"""
        logger.info("Starting Transformer model training")
        
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
        
        logger.info("Transformer training completed")
        logger.info(f"Training metrics: {train_metrics}")
        
        return {
            'history': self.history.history,
            'train_metrics': train_metrics,
            'model_params': self.model.count_params()
        }
    
    def predict(self, data: pd.DataFrame, return_attention: bool = True) -> TransformerPredictionResult:
        """Make predictions with attention analysis"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        logger.info("Making transformer predictions")
        
        # Prepare data
        X = data[self.feature_names].values
        X = pd.DataFrame(X).ffill().bfill().values
        X_scaled = self.scaler.transform(X)
        
        # Create sequences
        X_seq = []
        timestamps = []
        for i in range(self.config.sequence_length, len(X_scaled)):
            X_seq.append(X_scaled[i-self.config.sequence_length:i])
            if hasattr(data, 'index'):
                timestamps.append(str(data.index[i]))
            else:
                timestamps.append(str(i))
        
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
        
        # Extract attention weights and analyze patterns
        attention_viz = None
        if return_attention and self.attention_model is not None:
            attention_viz = self._analyze_attention(X_seq, timestamps)
        
        # Calculate feature importance
        feature_importance = self._calculate_feature_importance(X_seq, predictions_scaled)
        
        # Calculate temporal importance
        temporal_importance = self._calculate_temporal_importance(X_seq, predictions_scaled)
        
        # Analyze patterns
        pattern_analysis = self._analyze_patterns(X_seq, predictions, attention_viz)
        
        result = TransformerPredictionResult(
            predictions=predictions,
            confidence_lower=confidence_lower,
            confidence_upper=confidence_upper,
            attention_viz=attention_viz,
            feature_importance=feature_importance,
            temporal_importance=temporal_importance,
            pattern_analysis=pattern_analysis,
            metrics={},
            timestamp=datetime.now()
        )
        
        logger.info(f"Generated {len(predictions)} transformer predictions")
        return result
    
    def _analyze_attention(self, X_seq: np.ndarray, timestamps: List[str]) -> AttentionVisualization:
        """Analyze attention patterns"""
        try:
            # Extract attention weights
            attention_weights = self.attention_model.predict(X_seq, verbose=0)
            
            if attention_weights is None:
                return None
            
            # Analyze attention patterns
            patterns = self._detect_attention_patterns(attention_weights)
            
            # Calculate head importance
            head_importance = self._calculate_head_importance(attention_weights)
            
            return AttentionVisualization(
                attention_weights=attention_weights,
                feature_names=self.feature_names,
                timestamps=timestamps,
                layer_weights=[attention_weights],  # Single layer for now
                head_importance=head_importance,
                patterns=patterns
            )
            
        except Exception as e:
            logger.warning(f"Could not analyze attention: {e}")
            return None
    
    def _detect_attention_patterns(self, attention_weights: np.ndarray) -> Dict[str, Any]:
        """Detect common attention patterns"""
        patterns = {}
        
        try:
            # Average across batch and heads
            avg_attention = np.mean(attention_weights, axis=(0, 1))
            
            # Diagonal attention (local focus)
            diagonal_strength = np.mean(np.diag(avg_attention))
            patterns['diagonal_strength'] = float(diagonal_strength)
            
            # Recency bias (attention to recent timesteps)
            recent_weights = np.mean(avg_attention[:, -10:])  # Last 10 timesteps
            patterns['recency_bias'] = float(recent_weights)
            
            # Long-range dependencies
            long_range = np.mean(avg_attention[:-20, :20])  # Early attending to late
            patterns['long_range_dependencies'] = float(long_range)
            
            # Attention entropy (focus vs. distributed)
            entropy = -np.sum(avg_attention * np.log(avg_attention + 1e-9), axis=-1)
            patterns['attention_entropy'] = float(np.mean(entropy))
            
        except Exception as e:
            logger.warning(f"Error detecting attention patterns: {e}")
            
        return patterns
    
    def _calculate_head_importance(self, attention_weights: np.ndarray) -> Dict[int, float]:
        """Calculate importance score for each attention head"""
        head_importance = {}
        
        try:
            n_heads = attention_weights.shape[1]
            
            for head in range(n_heads):
                head_weights = attention_weights[:, head, :, :]
                
                # Measure head importance by attention variance
                variance = np.var(head_weights)
                head_importance[head] = float(variance)
                
        except Exception as e:
            logger.warning(f"Error calculating head importance: {e}")
            
        return head_importance
    
    def _calculate_feature_importance(self, X_seq: np.ndarray, 
                                    predictions: np.ndarray) -> Dict[str, float]:
        """Calculate feature importance using integrated gradients"""
        if len(self.feature_names) == 0:
            return {}
        
        importance = {}
        
        try:
            # Simple permutation importance
            base_predictions = predictions.copy()
            
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
    
    def _calculate_temporal_importance(self, X_seq: np.ndarray, 
                                     predictions: np.ndarray) -> np.ndarray:
        """Calculate importance of each time step"""
        try:
            temporal_importance = np.zeros(self.config.sequence_length)
            base_predictions = predictions.copy()
            
            for t in range(self.config.sequence_length):
                # Mask time step t
                X_masked = X_seq.copy()
                X_masked[:, t, :] = 0  # Zero out time step
                
                # Get predictions with masked time step
                masked_predictions = self.model.predict(X_masked, verbose=0)
                
                # Calculate importance
                importance = np.mean(np.abs(base_predictions - masked_predictions.flatten()))
                temporal_importance[t] = importance
            
            return temporal_importance
            
        except Exception as e:
            logger.warning(f"Could not calculate temporal importance: {e}")
            return np.zeros(self.config.sequence_length)
    
    def _analyze_patterns(self, X_seq: np.ndarray, predictions: np.ndarray,
                         attention_viz: AttentionVisualization) -> Dict[str, Any]:
        """Analyze learned patterns in the model"""
        patterns = {}
        
        try:
            # Trend analysis
            if len(predictions) > 1:
                trend_changes = np.diff(predictions)
                patterns['trend_volatility'] = float(np.std(trend_changes))
                patterns['avg_trend'] = float(np.mean(trend_changes))
            
            # Seasonality detection (simplified)
            if len(predictions) >= 12:
                # Look for monthly patterns
                monthly_avg = []
                for i in range(12):
                    month_indices = np.arange(i, len(predictions), 12)
                    if len(month_indices) > 0:
                        monthly_avg.append(np.mean(predictions[month_indices]))
                
                if len(monthly_avg) == 12:
                    patterns['seasonal_strength'] = float(np.std(monthly_avg) / np.mean(monthly_avg))
            
            # Attention-based patterns
            if attention_viz and attention_viz.patterns:
                patterns.update(attention_viz.patterns)
            
        except Exception as e:
            logger.warning(f"Error analyzing patterns: {e}")
            
        return patterns
    
    def visualize_attention(self, attention_viz: AttentionVisualization, 
                           save_path: str = None) -> plt.Figure:
        """Visualize attention patterns"""
        if attention_viz is None:
            raise ValueError("No attention visualization data available")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Transformer Attention Analysis', fontsize=16)
        
        # Average attention heatmap
        avg_attention = np.mean(attention_viz.attention_weights, axis=(0, 1))
        sns.heatmap(avg_attention, ax=axes[0, 0], cmap='Blues', 
                   xticklabels=False, yticklabels=False)
        axes[0, 0].set_title('Average Attention Weights')
        axes[0, 0].set_xlabel('Key Position')
        axes[0, 0].set_ylabel('Query Position')
        
        # Head importance
        if attention_viz.head_importance:
            heads = list(attention_viz.head_importance.keys())
            importance = list(attention_viz.head_importance.values())
            
            axes[0, 1].bar(heads, importance)
            axes[0, 1].set_title('Attention Head Importance')
            axes[0, 1].set_xlabel('Head Index')
            axes[0, 1].set_ylabel('Importance Score')
        
        # Attention patterns over time
        if len(attention_viz.attention_weights) > 1:
            # Show how attention changes over samples
            sample_attention = attention_viz.attention_weights[0, 0, :, :]  # First sample, first head
            sns.heatmap(sample_attention, ax=axes[1, 0], cmap='Reds',
                       xticklabels=False, yticklabels=False)
            axes[1, 0].set_title('Sample Attention Pattern (Head 0)')
            axes[1, 0].set_xlabel('Key Position')
            axes[1, 0].set_ylabel('Query Position')
        
        # Pattern analysis
        if attention_viz.patterns:
            pattern_names = list(attention_viz.patterns.keys())
            pattern_values = list(attention_viz.patterns.values())
            
            axes[1, 1].barh(pattern_names, pattern_values)
            axes[1, 1].set_title('Detected Attention Patterns')
            axes[1, 1].set_xlabel('Pattern Strength')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Attention visualization saved to {save_path}")
        
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
                factor=0.8,
                patience=self.config.patience // 2,
                min_lr=1e-7,
                verbose=1
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
        
        # Save model
        self.model.save(f"{filepath}_model.h5")
        
        # Save attention model if available
        if self.attention_model is not None:
            self.attention_model.save(f"{filepath}_attention.h5")
        
        # Save scalers and config
        model_data = {
            'config': self.config,
            'scaler': self.scaler,
            'target_scaler': self.target_scaler,
            'feature_names': self.feature_names,
            'is_trained': self.is_trained
        }
        
        joblib.dump(model_data, f"{filepath}_data.pkl")
        
        logger.info(f"Transformer model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model"""
        try:
            # Custom objects for loading
            custom_objects = {
                'SinusoidalPositionalEncoding': SinusoidalPositionalEncoding,
                'LearnablePositionalEncoding': LearnablePositionalEncoding,
                'MultiHeadSelfAttention': MultiHeadSelfAttention,
                'TransformerBlock': TransformerBlock
            }
            
            # Load main model
            self.model = keras.models.load_model(
                f"{filepath}_model.h5",
                custom_objects=custom_objects
            )
            
            # Load attention model if available
            try:
                self.attention_model = keras.models.load_model(
                    f"{filepath}_attention.h5",
                    custom_objects=custom_objects
                )
            except:
                logger.warning("Could not load attention model")
                self.attention_model = None
            
            # Load scalers and config
            model_data = joblib.load(f"{filepath}_data.pkl")
            self.config = model_data['config']
            self.scaler = model_data['scaler']
            self.target_scaler = model_data['target_scaler']
            self.feature_names = model_data['feature_names']
            self.is_trained = model_data['is_trained']
            
            logger.info(f"Transformer model loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise


# Factory function
def create_transformer_model(config: Dict[str, Any] = None) -> TimeSeriesTransformer:
    """Create transformer model with configuration"""
    if config:
        transformer_config = TransformerConfig(**config)
    else:
        transformer_config = TransformerConfig()
    
    return TimeSeriesTransformer(transformer_config)


# Export classes and functions
__all__ = [
    'TimeSeriesTransformer',
    'TransformerConfig',
    'TransformerPredictionResult',
    'AttentionVisualization',
    'SinusoidalPositionalEncoding',
    'LearnablePositionalEncoding',
    'MultiHeadSelfAttention',
    'TransformerBlock',
    'create_transformer_model'
]