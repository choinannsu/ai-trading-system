#!/usr/bin/env python3
"""
Test script for the reinforcement learning trading system
"""

import numpy as np
import pandas as pd
import torch
import sys
import traceback
from datetime import datetime

# Add the project root to the path
sys.path.append('/Users/user/Projects/fire/ai-trading-system')

def test_trading_environment():
    """Test the trading environment"""
    print("Testing Trading Environment...")
    
    try:
        from models.reinforcement import create_trading_env, generate_sample_market_data
        
        # Generate sample data
        data = generate_sample_market_data(n_days=500)
        print(f"✓ Generated sample data: {data.shape}")
        
        # Create environment
        env_config = {
            'initial_balance': 100000,
            'max_steps': 100,
            'lookback_window': 20
        }
        env = create_trading_env(env_config, data)
        print(f"✓ Created environment with obs space: {env.observation_space.shape}")
        
        # Test reset and step
        obs, info = env.reset()
        print(f"✓ Environment reset successful, obs shape: {obs.shape}")
        
        # Test a few steps
        for i in range(5):
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            print(f"  Step {i+1}: reward={reward:.4f}, portfolio_value=${info['portfolio_value']:,.2f}")
            
            if done:
                break
        
        print("✓ Trading Environment test passed!\n")
        return True
        
    except Exception as e:
        print(f"✗ Trading Environment test failed: {e}")
        traceback.print_exc()
        return False


def test_reward_functions():
    """Test reward functions"""
    print("Testing Reward Functions...")
    
    try:
        from models.reinforcement import create_reward_function, RewardConfig
        
        # Test different reward types
        reward_types = ['simple', 'sharpe', 'composite', 'adaptive']
        
        for reward_type in reward_types:
            reward_func = create_reward_function(reward_type)
            
            # Test reward calculation
            portfolio_return = 0.01
            portfolio_value = 105000
            positions = {'AAPL': 0.3, 'MSFT': 0.2, 'cash': 0.5}
            trades = []
            market_data = {'volatilities': {'AAPL': 0.02, 'MSFT': 0.015}}
            
            reward = reward_func.calculate(
                portfolio_return, portfolio_value, positions, trades, market_data
            )
            
            print(f"  ✓ {reward_type} reward: {reward:.4f}")
        
        print("✓ Reward Functions test passed!\n")
        return True
        
    except Exception as e:
        print(f"✗ Reward Functions test failed: {e}")
        traceback.print_exc()
        return False


def test_ppo_agent():
    """Test PPO agent"""
    print("Testing PPO Agent...")
    
    try:
        from models.reinforcement import create_ppo_agent
        
        obs_dim = 100
        action_dim = 5
        
        agent = create_ppo_agent(obs_dim, action_dim, discrete=False)
        print(f"✓ Created PPO agent")
        
        # Test action generation
        obs = np.random.randn(obs_dim)
        action, info = agent.get_action(obs)
        print(f"  ✓ Generated action: {action.shape}, info keys: {list(info.keys())}")
        
        # Test storing transition
        next_obs = np.random.randn(obs_dim)
        agent.store_transition(obs, action, 0.1, False, info)
        print(f"  ✓ Stored transition")
        
        print("✓ PPO Agent test passed!\n")
        return True
        
    except Exception as e:
        print(f"✗ PPO Agent test failed: {e}")
        traceback.print_exc()
        return False


def test_sac_agent():
    """Test SAC agent"""
    print("Testing SAC Agent...")
    
    try:
        from models.reinforcement import create_sac_agent
        
        obs_dim = 100
        action_dim = 5
        
        agent = create_sac_agent(obs_dim, action_dim)
        print(f"✓ Created SAC agent")
        
        # Test action generation
        obs = np.random.randn(obs_dim)
        action, info = agent.get_action(obs)
        print(f"  ✓ Generated action: {action.shape}, info keys: {list(info.keys())}")
        
        # Test storing transition
        next_obs = np.random.randn(obs_dim)
        agent.store_transition(obs, action, 0.1, next_obs, False, info)
        print(f"  ✓ Stored transition")
        
        print("✓ SAC Agent test passed!\n")
        return True
        
    except Exception as e:
        print(f"✗ SAC Agent test failed: {e}")
        traceback.print_exc()
        return False


def test_rainbow_dqn_agent():
    """Test Rainbow DQN agent"""
    print("Testing Rainbow DQN Agent...")
    
    try:
        from models.reinforcement import create_rainbow_dqn_agent
        
        obs_dim = 100
        action_dim = 10  # Discrete actions
        
        agent = create_rainbow_dqn_agent(obs_dim, action_dim)
        print(f"✓ Created Rainbow DQN agent")
        
        # Test action generation
        obs = np.random.randn(obs_dim)
        action, info = agent.get_action(obs)
        print(f"  ✓ Generated action: {action}, info keys: {list(info.keys())}")
        
        # Test storing transition
        next_obs = np.random.randn(obs_dim)
        agent.store_transition(obs, action, 0.1, next_obs, False, info)
        print(f"  ✓ Stored transition")
        
        print("✓ Rainbow DQN Agent test passed!\n")
        return True
        
    except Exception as e:
        print(f"✗ Rainbow DQN Agent test failed: {e}")
        traceback.print_exc()
        return False


def test_multi_agent_system():
    """Test multi-agent trading system"""
    print("Testing Multi-Agent System...")
    
    try:
        from models.reinforcement import create_multi_agent_system
        
        obs_dim = 100
        action_dim = 5
        
        config = {
            'num_agents': 3,
            'communication_mode': 'selective',
            'dynamic_allocation': True
        }
        
        system = create_multi_agent_system(obs_dim, action_dim, config)
        print(f"✓ Created multi-agent system with {len(system.agents)} agents")
        
        # Test action generation
        obs = np.random.randn(obs_dim)
        market_data = {'momentum': 0.01, 'volatility': 0.02}
        action, info = system.get_action(obs, market_data)
        print(f"  ✓ Generated system action: {action.shape}")
        print(f"  ✓ Agent allocations: {info['allocations']}")
        
        # Test storing transition
        next_obs = np.random.randn(obs_dim)
        system.store_transition(obs, action, 0.1, next_obs, False, info)
        print(f"  ✓ Stored transition for all agents")
        
        print("✓ Multi-Agent System test passed!\n")
        return True
        
    except Exception as e:
        print(f"✗ Multi-Agent System test failed: {e}")
        traceback.print_exc()
        return False


def test_complete_trading_system():
    """Test the complete trading system integration"""
    print("Testing Complete Trading System Integration...")
    
    try:
        from models.reinforcement import create_trading_system
        
        # Create complete system
        env_config = {
            'initial_balance': 100000,
            'max_steps': 50,
            'lookback_window': 10
        }
        
        agent_config = {
            'learning_rate': 0.001,
            'batch_size': 32
        }
        
        reward_config = {
            'reward_scaling': 1.0,
            'risk_free_rate': 0.02
        }
        
        env, agent, reward_function = create_trading_system(
            env_config, agent_config, reward_config, agent_type='ppo'
        )
        
        print(f"✓ Created complete trading system")
        print(f"  Environment: {type(env).__name__}")
        print(f"  Agent: {type(agent).__name__}")
        print(f"  Reward Function: {type(reward_function).__name__}")
        
        # Test a short episode
        obs, info = env.reset()
        total_reward = 0
        
        for step in range(10):
            action, agent_info = agent.get_action(obs)
            obs, env_reward, done, truncated, env_info = env.step(action)
            
            # Calculate custom reward
            custom_reward = reward_function.calculate(
                env_reward, env_info['portfolio_value'], 
                {}, env_info.get('executed_trades', []), {}
            )
            
            agent.store_transition(obs, action, custom_reward, done, agent_info)
            total_reward += custom_reward
            
            if done:
                break
        
        print(f"  ✓ Completed {step+1} steps, total reward: {total_reward:.4f}")
        print("✓ Complete Trading System test passed!\n")
        return True
        
    except Exception as e:
        print(f"✗ Complete Trading System test failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("=" * 60)
    print("REINFORCEMENT LEARNING TRADING SYSTEM TEST SUITE")
    print("=" * 60)
    print(f"Test started at: {datetime.now()}")
    print()
    
    tests = [
        test_trading_environment,
        test_reward_functions,
        test_ppo_agent,
        test_sac_agent,
        test_rainbow_dqn_agent,
        test_multi_agent_system,
        test_complete_trading_system
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} crashed: {e}")
            failed += 1
    
    print("=" * 60)
    print("TEST RESULTS")
    print("=" * 60)
    print(f"✓ Passed: {passed}")
    print(f"✗ Failed: {failed}")
    print(f"Total: {passed + failed}")
    
    if failed == 0:
        print("\n🎉 ALL TESTS PASSED! 🎉")
        print("Reinforcement Learning Trading System is ready!")
    else:
        print(f"\n⚠️  {failed} tests failed. Please check the errors above.")
    
    print(f"\nTest completed at: {datetime.now()}")
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)