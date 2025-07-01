"""
ML Ensemble Strategy Implementation
Machine learning ensemble strategy combining multiple predictive models
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import warnings
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score
import joblib

from ..base_strategy import BaseStrategy, StrategyConfig, Signal, SignalType
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class MLEnsembleConfig(StrategyConfig):
    """Configuration for ML ensemble strategy"""
    name: str = "MLEnsembleStrategy"
    
    # Model parameters
    models: List[str] = field(default_factory=lambda: ["random_forest", "gradient_boosting", "logistic_regression"])
    ensemble_method: str = "voting"  # "voting", "stacking", "weighted"
    
    # Feature engineering
    feature_window: int = 20           # Window for technical indicators
    price_features: List[str] = field(default_factory=lambda: ["returns", "volatility", "rsi", "macd", "bb_position"])
    volume_features: List[str] = field(default_factory=lambda: ["volume_ratio", "volume_sma"])
    macro_features: List[str] = field(default_factory=lambda: ["vix", "market_regime"])
    
    # Target definition
    prediction_horizon: int = 5        # Days ahead to predict
    return_threshold: float = 0.02     # Return threshold for classification (2%)
    
    # Training parameters
    min_training_samples: int = 500    # Minimum samples for training
    retrain_frequency: int = 21        # Retrain models every 21 days
    validation_split: float = 0.2      # Validation split ratio
    
    # Model ensemble
    model_weights: Dict[str, float] = field(default_factory=dict)
    confidence_threshold: float = 0.6  # Minimum ensemble confidence
    
    # Risk management
    max_models_disagreement: float = 0.3  # Max disagreement between models
    performance_decay_factor: float = 0.95  # Weight decay for model performance
    
    # Feature selection
    feature_selection: bool = True
    max_features: int = 50
    feature_importance_threshold: float = 0.01


class MLEnsembleStrategy(BaseStrategy):
    """
    ML ensemble trading strategy that combines multiple machine learning models
    
    Features:
    - Multiple model types (RF, GBM, LogReg, SVM)
    - Comprehensive feature engineering
    - Dynamic model retraining
    - Performance-based model weighting
    - Feature importance analysis
    """
    
    def __init__(self, config: MLEnsembleConfig):
        super().__init__(config)
        self.config: MLEnsembleConfig = config
        
        # Model management
        self.models: Dict[str, Any] = {}
        self.model_performance: Dict[str, Dict[str, float]] = {}
        self.feature_importance: Dict[str, Dict[str, float]] = {}
        self.scalers: Dict[str, Any] = {}
        
        # Training data
        self.feature_data: List[Dict[str, float]] = []
        self.target_data: List[int] = []
        self.training_timestamps: List[datetime] = []
        
        # Feature engineering
        self.feature_names: List[str] = []
        self.selected_features: List[str] = []
        
        # Performance tracking
        self.last_retrain_date: Optional[datetime] = None
        self.prediction_accuracy: Dict[str, List[float]] = {}
        
        self._initialize_models()
        logger.info(f"Initialized ML ensemble strategy with {len(self.config.models)} models")
    
    def _initialize_models(self):
        """Initialize machine learning models"""
        for model_name in self.config.models:
            if model_name == "random_forest":
                self.models[model_name] = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    min_samples_split=20,
                    min_samples_leaf=10,
                    random_state=42
                )
            elif model_name == "gradient_boosting":
                self.models[model_name] = GradientBoostingClassifier(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=6,
                    random_state=42
                )
            elif model_name == "logistic_regression":
                self.models[model_name] = LogisticRegression(
                    C=1.0,
                    max_iter=1000,
                    random_state=42
                )
            elif model_name == "svm":
                self.models[model_name] = SVC(
                    C=1.0,
                    kernel='rbf',
                    probability=True,
                    random_state=42
                )
            
            # Initialize performance tracking
            self.model_performance[model_name] = {
                "accuracy": 0.5,
                "precision": 0.5,
                "recall": 0.5,
                "weight": 1.0 / len(self.config.models)
            }
            
            self.scalers[model_name] = StandardScaler()
            self.prediction_accuracy[model_name] = []
    
    def generate_signals(self, data: pd.DataFrame, timestamp: datetime) -> List[Signal]:
        """Generate ML-based trading signals"""
        signals = []
        
        if len(data) < self.config.feature_window + self.config.prediction_horizon:
            return signals
        
        # Extract features
        features = self._extract_features(data, timestamp)
        if not features:
            return signals
        
        # Store training data
        self._store_training_data(features, data, timestamp)
        
        # Check if models need retraining
        if self._should_retrain(timestamp):
            self._retrain_models(timestamp)
        
        # Generate predictions if models are trained
        if self.models and any(hasattr(model, 'predict') for model in self.models.values()):
            for symbol in self.config.symbols:
                try:
                    signal = self._generate_ml_signal(symbol, features, data, timestamp)
                    if signal:
                        signals.append(signal)
                except Exception as e:
                    logger.error(f"Error generating ML signal for {symbol}: {e}")
        
        return signals
    
    def _extract_features(self, data: pd.DataFrame, timestamp: datetime) -> Dict[str, Dict[str, float]]:
        """Extract features for all symbols"""
        features = {}
        
        for symbol in self.config.symbols:
            if f"{symbol}_close" not in data.columns:
                continue
            
            symbol_features = {}
            
            # Get price and volume data
            prices = data[f"{symbol}_close"].values
            volumes = data.get(f"{symbol}_volume", pd.Series(np.ones(len(prices)))).values
            
            if len(prices) < self.config.feature_window:
                continue
            
            # Price-based features
            symbol_features.update(self._calculate_price_features(prices))
            
            # Volume-based features
            symbol_features.update(self._calculate_volume_features(volumes))
            
            # Technical indicators
            symbol_features.update(self._calculate_technical_features(prices, volumes))
            
            # Cross-asset features
            symbol_features.update(self._calculate_cross_asset_features(data, symbol))
            
            # Macro features
            symbol_features.update(self._calculate_macro_features(data, symbol))
            
            features[symbol] = symbol_features
        
        return features
    
    def _calculate_price_features(self, prices: np.ndarray) -> Dict[str, float]:
        """Calculate price-based features"""
        features = {}
        
        if len(prices) < self.config.feature_window:
            return features
        
        # Returns over different horizons
        for horizon in [1, 3, 5, 10, 20]:
            if len(prices) > horizon:
                returns = (prices[-1] - prices[-horizon-1]) / prices[-horizon-1]
                features[f"return_{horizon}d"] = returns
        
        # Volatility
        returns = np.diff(prices) / prices[:-1]
        features["volatility_20d"] = np.std(returns[-20:]) if len(returns) >= 20 else 0
        features["volatility_5d"] = np.std(returns[-5:]) if len(returns) >= 5 else 0
        
        # Price momentum
        features["momentum_10d"] = (prices[-1] - prices[-11]) / prices[-11] if len(prices) > 10 else 0
        features["momentum_20d"] = (prices[-1] - prices[-21]) / prices[-21] if len(prices) > 20 else 0
        
        # Moving averages
        for period in [5, 10, 20, 50]:
            if len(prices) >= period:
                ma = np.mean(prices[-period:])
                features[f"ma_ratio_{period}d"] = prices[-1] / ma - 1
        
        # Price position (percentile in recent range)
        if len(prices) >= 20:
            recent_prices = prices[-20:]
            features["price_percentile_20d"] = (np.sum(recent_prices <= prices[-1]) / len(recent_prices))
        
        return features
    
    def _calculate_volume_features(self, volumes: np.ndarray) -> Dict[str, float]:
        """Calculate volume-based features"""
        features = {}
        
        if len(volumes) < 5:
            return features
        
        # Volume ratios
        for period in [5, 10, 20]:
            if len(volumes) >= period:
                avg_volume = np.mean(volumes[-period-1:-1])
                features[f"volume_ratio_{period}d"] = volumes[-1] / avg_volume if avg_volume > 0 else 1
        
        # Volume momentum
        if len(volumes) >= 5:
            features["volume_momentum"] = (volumes[-1] - np.mean(volumes[-5:-1])) / np.mean(volumes[-5:-1])
        
        return features
    
    def _calculate_technical_features(self, prices: np.ndarray, volumes: np.ndarray) -> Dict[str, float]:
        """Calculate technical indicator features"""
        features = {}
        
        if len(prices) < 20:
            return features
        
        # RSI
        features["rsi"] = self._calculate_rsi(prices)
        
        # MACD
        macd, macd_signal = self._calculate_macd(prices)
        features["macd"] = macd
        features["macd_signal"] = macd_signal
        features["macd_histogram"] = macd - macd_signal
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(prices)
        features["bb_position"] = (prices[-1] - bb_lower) / (bb_upper - bb_lower) if bb_upper != bb_lower else 0.5
        features["bb_width"] = (bb_upper - bb_lower) / bb_middle if bb_middle > 0 else 0
        
        # Stochastic Oscillator
        features["stoch_k"], features["stoch_d"] = self._calculate_stochastic(prices)
        
        # Average True Range (volatility)
        features["atr"] = self._calculate_atr(prices)
        
        return features
    
    def _calculate_cross_asset_features(self, data: pd.DataFrame, symbol: str) -> Dict[str, float]:
        """Calculate cross-asset features"""
        features = {}
        
        # Relative performance vs other assets
        if len(self.config.symbols) > 1:
            symbol_return = 0
            other_returns = []
            
            if f"{symbol}_close" in data.columns and len(data) >= 5:
                symbol_prices = data[f"{symbol}_close"].values
                if len(symbol_prices) >= 5:
                    symbol_return = (symbol_prices[-1] - symbol_prices[-6]) / symbol_prices[-6]
            
            for other_symbol in self.config.symbols:
                if other_symbol != symbol and f"{other_symbol}_close" in data.columns:
                    other_prices = data[f"{other_symbol}_close"].values
                    if len(other_prices) >= 5:
                        other_return = (other_prices[-1] - other_prices[-6]) / other_prices[-6]
                        other_returns.append(other_return)
            
            if other_returns:
                market_return = np.mean(other_returns)
                features["relative_performance"] = symbol_return - market_return
                features["beta"] = self._calculate_beta(data, symbol)
        
        return features
    
    def _calculate_macro_features(self, data: pd.DataFrame, symbol: str) -> Dict[str, float]:
        """Calculate macro-economic features"""
        features = {}
        
        # Market regime indicators
        if len(self.config.symbols) > 1:
            # Calculate market volatility
            market_returns = []
            for s in self.config.symbols:
                if f"{s}_close" in data.columns and len(data) >= 2:
                    prices = data[f"{s}_close"].values
                    if len(prices) >= 2:
                        returns = (prices[-1] - prices[-2]) / prices[-2]
                        market_returns.append(returns)
            
            if market_returns:
                features["market_volatility"] = np.std(market_returns)
                features["market_skewness"] = self._calculate_skewness(market_returns)
        
        # Time-based features
        features["day_of_week"] = data.index[-1].dayofweek if hasattr(data.index[-1], 'dayofweek') else 0
        features["month"] = data.index[-1].month if hasattr(data.index[-1], 'month') else 1
        
        return features
    
    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """Calculate RSI"""
        if len(prices) < period + 1:
            return 50.0
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _calculate_macd(self, prices: np.ndarray) -> Tuple[float, float]:
        """Calculate MACD"""
        if len(prices) < 26:
            return 0.0, 0.0
        
        # Exponential moving averages
        ema_12 = self._calculate_ema(prices, 12)
        ema_26 = self._calculate_ema(prices, 26)
        
        macd = ema_12 - ema_26
        macd_signal = self._calculate_ema(np.array([macd]), 9)
        
        return macd, macd_signal
    
    def _calculate_ema(self, prices: np.ndarray, period: int) -> float:
        """Calculate Exponential Moving Average"""
        if len(prices) < period:
            return np.mean(prices)
        
        multiplier = 2 / (period + 1)
        ema = prices[0]
        
        for price in prices[1:]:
            ema = (price * multiplier) + (ema * (1 - multiplier))
        
        return ema
    
    def _calculate_bollinger_bands(self, prices: np.ndarray, period: int = 20) -> Tuple[float, float, float]:
        """Calculate Bollinger Bands"""
        if len(prices) < period:
            return prices[-1], prices[-1], prices[-1]
        
        sma = np.mean(prices[-period:])
        std = np.std(prices[-period:])
        
        upper = sma + (2 * std)
        lower = sma - (2 * std)
        
        return upper, sma, lower
    
    def _calculate_stochastic(self, prices: np.ndarray, period: int = 14) -> Tuple[float, float]:
        """Calculate Stochastic Oscillator"""
        if len(prices) < period:
            return 50.0, 50.0
        
        recent_prices = prices[-period:]
        high = np.max(recent_prices)
        low = np.min(recent_prices)
        close = prices[-1]
        
        if high == low:
            k_percent = 50.0
        else:
            k_percent = ((close - low) / (high - low)) * 100
        
        # Simple approximation for D%
        d_percent = k_percent  # Normally this would be a moving average of K%
        
        return k_percent, d_percent
    
    def _calculate_atr(self, prices: np.ndarray, period: int = 14) -> float:
        """Calculate Average True Range"""
        if len(prices) < period + 1:
            return 0.0
        
        # Simplified ATR using price differences
        price_ranges = []
        for i in range(len(prices) - period, len(prices)):
            if i > 0:
                true_range = abs(prices[i] - prices[i-1])
                price_ranges.append(true_range)
        
        return np.mean(price_ranges) if price_ranges else 0.0
    
    def _calculate_beta(self, data: pd.DataFrame, symbol: str) -> float:
        """Calculate beta relative to market"""
        if len(self.config.symbols) < 2 or len(data) < 20:
            return 1.0
        
        # Get symbol returns
        symbol_prices = data[f"{symbol}_close"].values
        if len(symbol_prices) < 20:
            return 1.0
        
        symbol_returns = np.diff(symbol_prices) / symbol_prices[:-1]
        
        # Calculate market returns
        market_returns = []
        for other_symbol in self.config.symbols:
            if other_symbol != symbol and f"{other_symbol}_close" in data.columns:
                other_prices = data[f"{other_symbol}_close"].values
                if len(other_prices) >= 20:
                    other_returns = np.diff(other_prices) / other_prices[:-1]
                    market_returns.append(other_returns[-len(symbol_returns):])
        
        if not market_returns:
            return 1.0
        
        market_avg_returns = np.mean(market_returns, axis=0)
        
        if len(market_avg_returns) == len(symbol_returns) and np.var(market_avg_returns) > 0:
            beta = np.cov(symbol_returns, market_avg_returns)[0, 1] / np.var(market_avg_returns)
            return beta
        
        return 1.0
    
    def _calculate_skewness(self, data: List[float]) -> float:
        """Calculate skewness"""
        if len(data) < 3:
            return 0.0
        
        data_array = np.array(data)
        mean = np.mean(data_array)
        std = np.std(data_array)
        
        if std == 0:
            return 0.0
        
        skewness = np.mean(((data_array - mean) / std) ** 3)
        return skewness
    
    def _store_training_data(self, features: Dict[str, Dict[str, float]], 
                           data: pd.DataFrame, timestamp: datetime):
        """Store features and targets for model training"""
        # Store features for each symbol
        for symbol, symbol_features in features.items():
            if f"{symbol}_close" in data.columns:
                # Calculate target (future return)
                prices = data[f"{symbol}_close"].values
                if len(prices) >= self.config.prediction_horizon + 1:
                    current_price = prices[-1]
                    future_idx = min(len(prices) - 1, len(prices) - self.config.prediction_horizon)
                    future_price = prices[future_idx] if future_idx >= 0 else current_price
                    
                    future_return = (future_price - current_price) / current_price
                    
                    # Create target classes: 0 (down), 1 (neutral), 2 (up)
                    if future_return > self.config.return_threshold:
                        target = 2  # Up
                    elif future_return < -self.config.return_threshold:
                        target = 0  # Down
                    else:
                        target = 1  # Neutral
                    
                    # Store data
                    self.feature_data.append(symbol_features)
                    self.target_data.append(target)
                    self.training_timestamps.append(timestamp)
        
        # Keep only recent training data
        max_samples = self.config.min_training_samples * 3
        if len(self.feature_data) > max_samples:
            self.feature_data = self.feature_data[-max_samples:]
            self.target_data = self.target_data[-max_samples:]
            self.training_timestamps = self.training_timestamps[-max_samples:]
    
    def _should_retrain(self, timestamp: datetime) -> bool:
        """Check if models should be retrained"""
        if self.last_retrain_date is None:
            return len(self.feature_data) >= self.config.min_training_samples
        
        days_since_retrain = (timestamp - self.last_retrain_date).days
        return (days_since_retrain >= self.config.retrain_frequency and 
                len(self.feature_data) >= self.config.min_training_samples)
    
    def _retrain_models(self, timestamp: datetime):
        """Retrain all ML models"""
        logger.info("Retraining ML models...")
        
        if len(self.feature_data) < self.config.min_training_samples:
            return
        
        # Prepare training data
        X, y = self._prepare_training_data()
        
        if X is None or len(X) < self.config.min_training_samples:
            return
        
        # Split data for validation
        split_idx = int(len(X) * (1 - self.config.validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Train each model
        for model_name, model in self.models.items():
            try:
                # Scale features
                X_train_scaled = self.scalers[model_name].fit_transform(X_train)
                X_val_scaled = self.scalers[model_name].transform(X_val)
                
                # Train model
                model.fit(X_train_scaled, y_train)
                
                # Validate model
                y_pred = model.predict(X_val_scaled)
                
                # Update performance metrics
                accuracy = accuracy_score(y_val, y_pred)
                precision = precision_score(y_val, y_pred, average='weighted', zero_division=0)
                recall = recall_score(y_val, y_pred, average='weighted', zero_division=0)
                
                self.model_performance[model_name].update({
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall
                })
                
                # Calculate feature importance
                if hasattr(model, 'feature_importances_'):
                    importance_dict = dict(zip(self.feature_names, model.feature_importances_))
                    self.feature_importance[model_name] = importance_dict
                
                logger.info(f"Retrained {model_name}: accuracy={accuracy:.3f}")
                
            except Exception as e:
                logger.error(f"Error retraining {model_name}: {e}")
        
        # Update model weights based on performance
        self._update_model_weights()
        
        # Feature selection
        if self.config.feature_selection:
            self._select_features()
        
        self.last_retrain_date = timestamp
    
    def _prepare_training_data(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Prepare training data for ML models"""
        if not self.feature_data:
            return None, None
        
        # Get all feature names
        all_features = set()
        for features in self.feature_data:
            all_features.update(features.keys())
        
        self.feature_names = sorted(list(all_features))
        
        # Create feature matrix
        X = []
        for features in self.feature_data:
            row = [features.get(feature, 0.0) for feature in self.feature_names]
            X.append(row)
        
        X = np.array(X)
        y = np.array(self.target_data)
        
        # Remove features with no variance
        feature_variance = np.var(X, axis=0)
        valid_features = feature_variance > 1e-8
        
        if not np.any(valid_features):
            return None, None
        
        X = X[:, valid_features]
        self.feature_names = [self.feature_names[i] for i, valid in enumerate(valid_features) if valid]
        
        return X, y
    
    def _update_model_weights(self):
        """Update model weights based on performance"""
        total_performance = 0
        
        for model_name in self.models:
            performance = self.model_performance[model_name]
            # Combined performance score
            score = (performance["accuracy"] + performance["precision"] + performance["recall"]) / 3
            total_performance += score
        
        # Update weights
        for model_name in self.models:
            performance = self.model_performance[model_name]
            score = (performance["accuracy"] + performance["precision"] + performance["recall"]) / 3
            weight = score / total_performance if total_performance > 0 else 1.0 / len(self.models)
            self.model_performance[model_name]["weight"] = weight
    
    def _select_features(self):
        """Select most important features"""
        if not self.feature_importance:
            return
        
        # Aggregate feature importance across models
        feature_scores = {}
        for model_name, importance in self.feature_importance.items():
            weight = self.model_performance[model_name]["weight"]
            for feature, score in importance.items():
                if feature not in feature_scores:
                    feature_scores[feature] = 0
                feature_scores[feature] += score * weight
        
        # Select top features
        sorted_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)
        self.selected_features = [feature for feature, score in sorted_features[:self.config.max_features]
                                 if score >= self.config.feature_importance_threshold]
        
        logger.info(f"Selected {len(self.selected_features)} features for prediction")
    
    def _generate_ml_signal(self, symbol: str, features: Dict[str, Dict[str, float]], 
                           data: pd.DataFrame, timestamp: datetime) -> Optional[Signal]:
        """Generate ML-based signal for a symbol"""
        if symbol not in features:
            return None
        
        symbol_features = features[symbol]
        current_price = data[f"{symbol}_close"].iloc[-1]
        
        # Prepare feature vector
        if self.selected_features:
            feature_vector = [symbol_features.get(feature, 0.0) for feature in self.selected_features]
        else:
            feature_vector = [symbol_features.get(feature, 0.0) for feature in self.feature_names]
        
        if not feature_vector:
            return None
        
        feature_vector = np.array(feature_vector).reshape(1, -1)
        
        # Get predictions from all models
        predictions = {}
        confidences = {}
        
        for model_name, model in self.models.items():
            if not hasattr(model, 'predict'):
                continue
            
            try:
                # Scale features
                feature_vector_scaled = self.scalers[model_name].transform(feature_vector)
                
                # Get prediction
                prediction = model.predict(feature_vector_scaled)[0]
                predictions[model_name] = prediction
                
                # Get confidence if available
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(feature_vector_scaled)[0]
                    confidences[model_name] = np.max(proba)
                else:
                    confidences[model_name] = 0.6  # Default confidence
                
            except Exception as e:
                logger.warning(f"Error getting prediction from {model_name}: {e}")
        
        if not predictions:
            return None
        
        # Ensemble prediction
        ensemble_prediction, ensemble_confidence = self._ensemble_predict(predictions, confidences)
        
        # Generate signal based on prediction
        signal_type = SignalType.HOLD
        
        if ensemble_prediction == 2 and ensemble_confidence >= self.config.confidence_threshold:
            signal_type = SignalType.BUY
        elif ensemble_prediction == 0 and ensemble_confidence >= self.config.confidence_threshold:
            signal_type = SignalType.SELL
        
        # Check for existing position
        if symbol in self.positions and signal_type != SignalType.HOLD:
            position = self.positions[symbol]
            # Exit opposite position
            if ((position.position_type.value == "long" and signal_type == SignalType.SELL) or
                (position.position_type.value == "short" and signal_type == SignalType.BUY)):
                signal_type = SignalType.SELL if position.position_type.value == "long" else SignalType.CLOSE_SHORT
        
        if signal_type != SignalType.HOLD:
            return Signal(
                symbol=symbol,
                signal_type=signal_type,
                timestamp=timestamp,
                price=current_price,
                confidence=ensemble_confidence,
                size=self.config.base_position_size * ensemble_confidence,
                metadata={
                    "ensemble_prediction": ensemble_prediction,
                    "model_predictions": predictions,
                    "model_confidences": confidences,
                    "feature_count": len(feature_vector[0])
                }
            )
        
        return None
    
    def _ensemble_predict(self, predictions: Dict[str, int], 
                         confidences: Dict[str, float]) -> Tuple[int, float]:
        """Combine predictions from multiple models"""
        if not predictions:
            return 1, 0.0  # Neutral prediction with low confidence
        
        if self.config.ensemble_method == "voting":
            # Weighted voting
            vote_scores = {0: 0, 1: 0, 2: 0}
            total_weight = 0
            
            for model_name, prediction in predictions.items():
                weight = self.model_performance[model_name]["weight"]
                confidence = confidences.get(model_name, 0.5)
                
                vote_scores[prediction] += weight * confidence
                total_weight += weight
            
            # Normalize scores
            if total_weight > 0:
                for class_id in vote_scores:
                    vote_scores[class_id] /= total_weight
            
            # Get best prediction
            best_prediction = max(vote_scores, key=vote_scores.get)
            best_confidence = vote_scores[best_prediction]
            
            return best_prediction, best_confidence
        
        else:
            # Simple majority voting
            vote_counts = {0: 0, 1: 0, 2: 0}
            for prediction in predictions.values():
                vote_counts[prediction] += 1
            
            best_prediction = max(vote_counts, key=vote_counts.get)
            confidence = vote_counts[best_prediction] / len(predictions)
            
            return best_prediction, confidence
    
    def calculate_position_size(self, signal: Signal, current_price: float, 
                              available_cash: float) -> float:
        """Calculate position size based on ML signal confidence"""
        base_size = signal.size
        target_value = available_cash * base_size
        
        # Account for transaction costs
        transaction_cost_rate = self.config.commission_rate + self.config.slippage_rate
        effective_cash = target_value / (1 + transaction_cost_rate)
        
        shares = effective_cash / current_price
        
        # Apply minimum trade size constraint
        min_shares = self.config.min_trade_size / current_price
        if shares < min_shares:
            return 0.0
        
        return shares
    
    def get_strategy_metrics(self) -> Dict[str, Any]:
        """Get ML ensemble specific metrics"""
        base_metrics = self.get_performance_metrics()
        
        ml_metrics = {
            "models_trained": len([m for m in self.models.values() if hasattr(m, 'predict')]),
            "training_samples": len(self.feature_data),
            "selected_features": len(self.selected_features) if self.selected_features else len(self.feature_names),
            "last_retrain": self.last_retrain_date.isoformat() if self.last_retrain_date else None
        }
        
        # Model performance
        for model_name, performance in self.model_performance.items():
            ml_metrics[f"{model_name}_accuracy"] = performance["accuracy"]
            ml_metrics[f"{model_name}_weight"] = performance["weight"]
        
        # Feature importance
        if self.feature_importance:
            avg_importance = {}
            for model_importance in self.feature_importance.values():
                for feature, importance in model_importance.items():
                    if feature not in avg_importance:
                        avg_importance[feature] = []
                    avg_importance[feature].append(importance)
            
            top_features = {}
            for feature, importances in avg_importance.items():
                top_features[feature] = np.mean(importances)
            
            # Get top 5 features
            sorted_features = sorted(top_features.items(), key=lambda x: x[1], reverse=True)[:5]
            ml_metrics["top_features"] = dict(sorted_features)
        
        return {**base_metrics, **ml_metrics}


# Export classes
__all__ = ['MLEnsembleStrategy', 'MLEnsembleConfig']