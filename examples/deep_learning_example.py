#!/usr/bin/env python3
"""
Deep Learning Models Example
Demonstrates LSTM, Transformer, CNN-LSTM Hybrid, and automated training pipeline
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import asyncio

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.deep_learning import (
    create_lstm_predictor,
    create_transformer_model,
    create_hybrid_model,
    create_training_pipeline,
    TrainingConfig
)
from utils.logger import get_logger

logger = get_logger(__name__)


def generate_financial_data(days: int = 1000) -> pd.DataFrame:
    """Generate realistic financial time series data"""
    
    np.random.seed(42)
    dates = pd.date_range(start=datetime.now() - timedelta(days=days), 
                         periods=days, freq='D')
    
    # Generate price series with realistic patterns
    returns = np.random.normal(0.0005, 0.02, days)
    
    # Add trend
    trend = np.sin(np.linspace(0, 4*np.pi, days)) * 0.001
    returns += trend
    
    # Add volatility clustering
    vol = np.zeros(days)
    vol[0] = 0.02
    for i in range(1, days):
        vol[i] = 0.9 * vol[i-1] + 0.1 * abs(returns[i-1])
    
    returns = returns + np.random.normal(0, vol)
    prices = 100 * np.exp(np.cumsum(returns))
    
    # Generate features
    data = {
        'timestamp': dates,
        'close': prices,
        'volume': np.random.lognormal(10, 0.5, days),
        'rsi': 50 + 30 * np.sin(np.linspace(0, 8*np.pi, days)) + np.random.normal(0, 5, days),
        'macd': np.random.normal(0, 0.5, days),
        'bb_position': np.random.uniform(0, 1, days)
    }
    
    # Add moving averages
    df = pd.DataFrame(data)
    df['ma_5'] = df['close'].rolling(5).mean()
    df['ma_20'] = df['close'].rolling(20).mean()
    df['volatility'] = df['close'].rolling(20).std()
    
    # Add target (next day return)
    df['target'] = df['close'].pct_change().shift(-1)
    
    # Drop NaN values
    df = df.dropna()
    
    return df


def demonstrate_lstm():
    """Demonstrate LSTM predictor"""
    print("\n" + "="*60)
    print("🧠 LSTM PREDICTOR DEMO")
    print("="*60)
    
    # Generate data
    data = generate_financial_data(500)
    feature_columns = ['close', 'volume', 'rsi', 'macd', 'bb_position', 'ma_5', 'ma_20', 'volatility']
    target_column = 'target'
    
    print(f"📊 Data shape: {data.shape}")
    print(f"📈 Features: {len(feature_columns)}")
    print(f"🎯 Target: {target_column}")
    
    # Create LSTM model
    lstm_config = {
        'sequence_length': 60,
        'lstm_units': [128, 64],
        'attention_units': 128,
        'dropout_rate': 0.2,
        'epochs': 50,
        'batch_size': 32,
        'use_attention': True
    }
    
    model = create_lstm_predictor(lstm_config)
    
    # Split data
    train_size = int(len(data) * 0.8)
    train_data = data[:train_size]
    test_data = data[train_size:]
    
    print(f"\n🔧 Training LSTM model...")
    print(f"📊 Train data: {len(train_data)} samples")
    print(f"🧪 Test data: {len(test_data)} samples")
    
    # Train model
    try:
        result = model.fit(train_data, target_column, feature_columns)
        
        print(f"✅ Training completed!")
        print(f"📊 Model parameters: {result['model_params']:,}")
        print(f"📈 Training metrics: {result['train_metrics']}")
        
        # Make predictions
        print(f"\n🔮 Making predictions...")
        predictions = model.predict(test_data, return_attention=True)
        
        print(f"📊 Predictions shape: {predictions.predictions.shape}")
        print(f"📈 Mean prediction: {np.mean(predictions.predictions):.6f}")
        print(f"📉 Prediction std: {np.std(predictions.predictions):.6f}")
        
        if predictions.attention_weights is not None:
            print(f"👁️  Attention weights shape: {predictions.attention_weights.shape}")
        
        if predictions.feature_importance:
            print(f"\n🔍 Top 3 Important Features:")
            sorted_features = sorted(predictions.feature_importance.items(), 
                                   key=lambda x: x[1], reverse=True)
            for feature, importance in sorted_features[:3]:
                print(f"  📊 {feature}: {importance:.6f}")
        
    except Exception as e:
        logger.error(f"LSTM demo failed: {e}")
        print(f"❌ LSTM demo failed: {e}")


def demonstrate_transformer():
    """Demonstrate Transformer model"""
    print("\n" + "="*60)
    print("🤖 TRANSFORMER MODEL DEMO")
    print("="*60)
    
    # Generate data
    data = generate_financial_data(400)
    feature_columns = ['close', 'volume', 'rsi', 'macd', 'bb_position', 'ma_5', 'ma_20', 'volatility']
    target_column = 'target'
    
    print(f"📊 Data shape: {data.shape}")
    
    # Create Transformer model
    transformer_config = {
        'sequence_length': 50,
        'd_model': 128,
        'n_heads': 8,
        'n_transformer_blocks': 4,
        'ff_dim': 256,
        'dropout_rate': 0.1,
        'epochs': 30,
        'batch_size': 32
    }
    
    model = create_transformer_model(transformer_config)
    
    # Split data
    train_size = int(len(data) * 0.8)
    train_data = data[:train_size]
    test_data = data[train_size:]
    
    print(f"\n🔧 Training Transformer model...")
    
    try:
        result = model.fit(train_data, target_column, feature_columns)
        
        print(f"✅ Training completed!")
        print(f"📊 Model parameters: {result['model_params']:,}")
        print(f"📈 Training metrics: {result['train_metrics']}")
        
        # Make predictions with attention analysis
        print(f"\n🔮 Making predictions with attention analysis...")
        predictions = model.predict(test_data, return_attention=True)
        
        print(f"📊 Predictions shape: {predictions.predictions.shape}")
        print(f"📈 Mean prediction: {np.mean(predictions.predictions):.6f}")
        
        if predictions.attention_viz:
            print(f"👁️  Attention analysis available")
            if predictions.attention_viz.patterns:
                print(f"🎯 Attention patterns detected: {len(predictions.attention_viz.patterns)}")
                for pattern, value in list(predictions.attention_viz.patterns.items())[:3]:
                    print(f"  📊 {pattern}: {value:.4f}")
        
        if predictions.temporal_importance is not None:
            print(f"⏰ Temporal importance shape: {predictions.temporal_importance.shape}")
            max_importance_idx = np.argmax(predictions.temporal_importance)
            print(f"🎯 Most important time step: {max_importance_idx} (importance: {predictions.temporal_importance[max_importance_idx]:.6f})")
        
    except Exception as e:
        logger.error(f"Transformer demo failed: {e}")
        print(f"❌ Transformer demo failed: {e}")


def demonstrate_hybrid():
    """Demonstrate CNN-LSTM Hybrid model"""
    print("\n" + "="*60)
    print("🔬 CNN-LSTM HYBRID DEMO")
    print("="*60)
    
    # Generate data
    data = generate_financial_data(600)
    feature_columns = ['close', 'volume', 'rsi', 'macd', 'bb_position', 'ma_5', 'ma_20', 'volatility']
    target_column = 'target'
    
    print(f"📊 Data shape: {data.shape}")
    
    # Create Hybrid model
    hybrid_config = {
        'sequence_length': 60,
        'cnn_filters': [64, 128, 64],
        'lstm_units': [128, 64],
        'cnn_dropout': 0.2,
        'lstm_dropout': 0.2,
        'epochs': 40,
        'batch_size': 32,
        'use_residual': True,
        'use_skip_connections': True
    }
    
    model = create_hybrid_model(hybrid_config)
    
    # Split data
    train_size = int(len(data) * 0.8)
    train_data = data[:train_size]
    test_data = data[train_size:]
    
    print(f"\n🔧 Training CNN-LSTM Hybrid model...")
    
    try:
        result = model.fit(train_data, target_column, feature_columns)
        
        print(f"✅ Training completed!")
        print(f"📊 Model parameters: {result['model_params']:,}")
        print(f"📈 Training metrics: {result['train_metrics']}")
        
        # Make predictions with feature extraction
        print(f"\n🔮 Making predictions with feature analysis...")
        predictions = model.predict(test_data, extract_features=True)
        
        print(f"📊 Predictions shape: {predictions.predictions.shape}")
        print(f"📈 Mean prediction: {np.mean(predictions.predictions):.6f}")
        
        if predictions.cnn_features is not None:
            print(f"🖼️  CNN features shape: {predictions.cnn_features.shape}")
        
        if predictions.lstm_features is not None:
            print(f"🧠 LSTM features shape: {predictions.lstm_features.shape}")
        
        if predictions.temporal_patterns:
            print(f"⏰ Temporal patterns detected: {len(predictions.temporal_patterns)}")
            for pattern, value in list(predictions.temporal_patterns.items())[:3]:
                if isinstance(value, (int, float)):
                    print(f"  📊 {pattern}: {value:.6f}")
        
        if predictions.local_patterns:
            print(f"🏞️  Local patterns detected: {len(predictions.local_patterns)}")
            for pattern, value in list(predictions.local_patterns.items())[:3]:
                if isinstance(value, (int, float)):
                    print(f"  📊 {pattern}: {value:.6f}")
        
    except Exception as e:
        logger.error(f"Hybrid demo failed: {e}")
        print(f"❌ Hybrid demo failed: {e}")


def demonstrate_training_pipeline():
    """Demonstrate automated training pipeline"""
    print("\n" + "="*60)
    print("⚙️ AUTOMATED TRAINING PIPELINE DEMO")
    print("="*60)
    
    # Generate data
    data = generate_financial_data(800)
    feature_columns = ['close', 'volume', 'rsi', 'macd', 'bb_position', 'ma_5', 'ma_20', 'volatility']
    target_column = 'target'
    
    print(f"📊 Data shape: {data.shape}")
    
    # Create training pipeline
    training_config = {
        'model_type': 'lstm',
        'n_trials': 20,  # Reduced for demo
        'n_splits': 3,
        'max_epochs': 30,
        'timeout_seconds': 1800,  # 30 minutes
        'metric': 'mse',
        'experiment_name': 'demo_experiment'
    }
    
    pipeline = create_training_pipeline(training_config)
    
    print(f"\n🔧 Running automated hyperparameter optimization...")
    print(f"🎯 Model type: {training_config['model_type']}")
    print(f"🔍 Trials: {training_config['n_trials']}")
    print(f"📊 CV folds: {training_config['n_splits']}")
    
    try:
        # Run training pipeline
        best_model, model_version = pipeline.train_and_optimize(
            data, target_column, feature_columns,
            tags=['demo', 'automated']
        )
        
        print(f"✅ Optimization completed!")
        print(f"📊 Model version: {model_version.version_id}")
        print(f"🎯 Best score: {model_version.performance_metrics['best_score']:.6f}")
        print(f"📈 CV mean: {model_version.performance_metrics['cv_mean']:.6f}")
        print(f"📉 CV std: {model_version.performance_metrics['cv_std']:.6f}")
        
        # Test the best model
        test_data = data[-100:]  # Last 100 samples for testing
        predictions = best_model.predict(test_data)
        
        print(f"\n🧪 Testing best model:")
        print(f"📊 Test predictions: {len(predictions.predictions) if hasattr(predictions, 'predictions') else len(predictions)}")
        
        # List available models
        print(f"\n📚 Available model versions:")
        versions = pipeline.model_manager.list_versions()
        for i, version in enumerate(versions[:3]):  # Show top 3
            print(f"  {i+1}. {version.version_id}")
            print(f"     Performance: {version.performance_metrics['cv_mean']:.6f}")
            print(f"     Created: {version.created_at}")
        
    except Exception as e:
        logger.error(f"Training pipeline demo failed: {e}")
        print(f"❌ Training pipeline demo failed: {e}")


def demonstrate_model_comparison():
    """Demonstrate model comparison"""
    print("\n" + "="*60)
    print("🏆 MODEL COMPARISON DEMO")
    print("="*60)
    
    # Generate data
    data = generate_financial_data(400)
    feature_columns = ['close', 'volume', 'rsi', 'macd', 'bb_position', 'ma_5', 'ma_20']
    target_column = 'target'
    
    print(f"📊 Data shape: {data.shape}")
    
    # Create training pipeline
    training_config = {
        'n_trials': 10,  # Small number for demo
        'n_splits': 3,
        'max_epochs': 20,
        'timeout_seconds': 900,  # 15 minutes
        'experiment_name': 'model_comparison'
    }
    
    pipeline = create_training_pipeline(training_config)
    
    print(f"\n🔧 Comparing LSTM, Transformer, and Hybrid models...")
    
    try:
        # Compare models
        results = pipeline.compare_models(
            data, target_column, 
            model_types=['lstm', 'transformer', 'hybrid'],
            feature_columns=feature_columns
        )
        
        print(f"✅ Model comparison completed!")
        print(f"\n🏆 Results:")
        
        # Sort by performance
        sorted_results = sorted(results.items(), 
                              key=lambda x: x[1].performance_metrics['cv_mean'])
        
        for rank, (model_type, version) in enumerate(sorted_results, 1):
            emoji = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉"
            print(f"{emoji} {rank}. {model_type.upper()}")
            print(f"    Score: {version.performance_metrics['cv_mean']:.6f} ± {version.performance_metrics['cv_std']:.6f}")
            print(f"    Version: {version.version_id}")
        
        # Get best model
        best_model_type = sorted_results[0][0]
        print(f"\n🎯 Best model: {best_model_type.upper()}")
        
    except Exception as e:
        logger.error(f"Model comparison demo failed: {e}")
        print(f"❌ Model comparison demo failed: {e}")


async def run_comprehensive_demo():
    """Run comprehensive deep learning demo"""
    print("🚀 DEEP LEARNING MODELS SYSTEM")
    print("="*60)
    print("Demonstrating LSTM, Transformer, CNN-LSTM Hybrid, and automated training")
    
    try:
        # Run all demonstrations
        demonstrate_lstm()
        demonstrate_transformer()
        demonstrate_hybrid()
        demonstrate_training_pipeline()
        demonstrate_model_comparison()
        
        print("\n" + "="*60)
        print("✅ DEEP LEARNING DEMO COMPLETE")
        print("="*60)
        print("🎯 Summary:")
        print("• LSTM with attention mechanism and ensemble support")
        print("• Transformer with custom positional encoding and attention visualization")
        print("• CNN-LSTM hybrid with skip connections and residual blocks")
        print("• Automated hyperparameter tuning with Optuna")
        print("• Cross-validation and model management")
        print("• Model comparison and versioning")
        print("• Comprehensive feature analysis and pattern detection")
        
    except Exception as e:
        logger.error(f"Error in comprehensive demo: {e}")
        raise


if __name__ == "__main__":
    # Run the comprehensive demo
    asyncio.run(run_comprehensive_demo())