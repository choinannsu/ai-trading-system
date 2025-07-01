"""
Portfolio Optimization System
Advanced portfolio optimization including Mean-Variance, Black-Litterman, 
Risk Parity, and Maximum Diversification approaches
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import warnings
from scipy.optimize import minimize
from scipy import stats
from sklearn.covariance import LedoitWolf, OAS
import cvxpy as cp

from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class OptimizerConfig:
    """Configuration for portfolio optimization"""
    # General optimization parameters
    optimization_method: str = "mean_variance"  # "mean_variance", "black_litterman", "risk_parity", "max_diversification"
    rebalancing_frequency: str = "monthly"  # "daily", "weekly", "monthly", "quarterly"
    lookback_window: int = 252  # Days for historical data
    
    # Risk parameters
    risk_aversion: float = 5.0  # Risk aversion parameter for mean-variance
    target_volatility: Optional[float] = None  # Target portfolio volatility
    max_volatility: float = 0.25  # Maximum allowed volatility
    
    # Constraints
    max_weight: float = 0.3  # Maximum weight per asset
    min_weight: float = 0.0  # Minimum weight per asset
    max_turnover: float = 0.5  # Maximum portfolio turnover
    
    # Black-Litterman parameters
    bl_tau: float = 0.025  # Uncertainty scaling factor
    bl_confidence: float = 0.5  # Confidence in views (0-1)
    
    # Risk parity parameters
    rp_risk_budget: Optional[Dict[str, float]] = None  # Custom risk budgets
    rp_volatility_target: float = 0.12  # Target volatility for risk parity
    
    # Covariance estimation
    cov_estimator: str = "sample"  # "sample", "ledoit_wolf", "oas", "shrunk"
    min_periods: int = 60  # Minimum periods for covariance estimation
    
    # Transaction costs
    transaction_cost: float = 0.001  # Transaction cost (0.1%)
    include_transaction_costs: bool = True


@dataclass
class OptimizationResult:
    """Portfolio optimization result"""
    weights: Dict[str, float]
    expected_return: float
    expected_volatility: float
    sharpe_ratio: float
    method: str
    timestamp: datetime
    
    # Additional metrics
    max_weight: float = 0.0
    turnover: float = 0.0
    effective_assets: float = 0.0
    concentration: float = 0.0
    
    # Method-specific results
    risk_contributions: Optional[Dict[str, float]] = None
    diversification_ratio: Optional[float] = None
    tracking_error: Optional[float] = None


class MeanVarianceOptimizer:
    """
    Mean-Variance optimization (Markowitz)
    
    Features:
    - Classic mean-variance optimization
    - Risk-return efficient frontier
    - Maximum Sharpe ratio portfolio
    - Minimum variance portfolio
    """
    
    def __init__(self, config: OptimizerConfig):
        self.config = config
    
    def optimize(self, expected_returns: pd.Series, 
                covariance_matrix: pd.DataFrame,
                current_weights: Optional[Dict[str, float]] = None) -> OptimizationResult:
        """Optimize portfolio using mean-variance approach"""
        
        assets = expected_returns.index.tolist()
        n_assets = len(assets)
        
        if n_assets == 0:
            raise ValueError("No assets provided for optimization")
        
        # Convert to numpy arrays
        mu = expected_returns.values
        Sigma = covariance_matrix.values
        
        # Decision variables
        w = cp.Variable(n_assets)
        
        # Portfolio return and risk
        portfolio_return = mu.T @ w
        portfolio_risk = cp.quad_form(w, Sigma)
        
        # Objective: maximize utility (return - risk penalty)
        utility = portfolio_return - 0.5 * self.config.risk_aversion * portfolio_risk
        objective = cp.Maximize(utility)
        
        # Constraints
        constraints = self._get_base_constraints(w, n_assets)
        
        # Add turnover constraint if current weights provided
        if current_weights is not None and self.config.include_transaction_costs:
            current_w = np.array([current_weights.get(asset, 0.0) for asset in assets])
            turnover_constraint = self._add_turnover_constraint(w, current_w)
            constraints.extend(turnover_constraint)
        
        # Solve optimization
        problem = cp.Problem(objective, constraints)
        
        try:
            problem.solve()
            
            if w.value is None:
                raise ValueError("Optimization failed to converge")
            
            # Create results
            optimal_weights = {asset: float(w.value[i]) for i, asset in enumerate(assets)}
            
            # Calculate metrics
            expected_return = float(mu.T @ w.value)
            expected_volatility = float(np.sqrt(w.value.T @ Sigma @ w.value))
            sharpe_ratio = expected_return / expected_volatility if expected_volatility > 0 else 0.0
            
            # Calculate additional metrics
            max_weight = float(np.max(np.abs(w.value)))
            turnover = float(np.sum(np.abs(w.value - (current_w if current_weights else 0))))
            concentration = float(np.sum(w.value ** 2))  # Herfindahl index
            effective_assets = 1 / concentration if concentration > 0 else 0.0
            
            return OptimizationResult(
                weights=optimal_weights,
                expected_return=expected_return,
                expected_volatility=expected_volatility,
                sharpe_ratio=sharpe_ratio,
                method="mean_variance",
                timestamp=datetime.now(),
                max_weight=max_weight,
                turnover=turnover,
                effective_assets=effective_assets,
                concentration=concentration
            )
            
        except Exception as e:
            logger.error(f"Mean-variance optimization failed: {e}")
            # Return equal weight portfolio as fallback
            return self._equal_weight_fallback(assets)
    
    def _get_base_constraints(self, w, n_assets):
        """Get base constraints for optimization"""
        constraints = []
        
        # Budget constraint
        constraints.append(cp.sum(w) == 1)
        
        # Weight constraints
        constraints.append(w >= self.config.min_weight)
        constraints.append(w <= self.config.max_weight)
        
        # Volatility constraint
        if self.config.max_volatility is not None:
            constraints.append(cp.norm(w, 2) <= self.config.max_volatility)
        
        return constraints
    
    def _add_turnover_constraint(self, w, current_w):
        """Add turnover constraint"""
        turnover = cp.norm(w - current_w, 1)
        return [turnover <= self.config.max_turnover]
    
    def _equal_weight_fallback(self, assets):
        """Return equal weight portfolio as fallback"""
        n_assets = len(assets)
        equal_weight = 1.0 / n_assets
        
        weights = {asset: equal_weight for asset in assets}
        
        return OptimizationResult(
            weights=weights,
            expected_return=0.0,
            expected_volatility=0.0,
            sharpe_ratio=0.0,
            method="equal_weight_fallback",
            timestamp=datetime.now()
        )


class BlackLittermanOptimizer:
    """
    Black-Litterman optimization
    
    Features:
    - Bayesian approach to portfolio optimization
    - Incorporation of investor views
    - Market equilibrium assumptions
    - Uncertainty quantification
    """
    
    def __init__(self, config: OptimizerConfig):
        self.config = config
    
    def optimize(self, historical_returns: pd.DataFrame,
                market_caps: Optional[Dict[str, float]] = None,
                views: Optional[Dict[str, Tuple[float, float]]] = None,
                current_weights: Optional[Dict[str, float]] = None) -> OptimizationResult:
        """Optimize portfolio using Black-Litterman approach"""
        
        assets = historical_returns.columns.tolist()
        n_assets = len(assets)
        
        if n_assets == 0:
            raise ValueError("No assets provided for optimization")
        
        # Step 1: Calculate historical covariance matrix
        Sigma = historical_returns.cov().values * 252  # Annualized
        
        # Step 2: Estimate market equilibrium returns
        w_market = self._get_market_weights(assets, market_caps)
        Pi = self._calculate_equilibrium_returns(w_market, Sigma)
        
        # Step 3: Incorporate investor views
        if views:
            mu_bl, Sigma_bl = self._apply_black_litterman(Pi, Sigma, views, assets)
        else:
            mu_bl = Pi
            Sigma_bl = Sigma
        
        # Step 4: Optimize portfolio with Black-Litterman inputs
        expected_returns = pd.Series(mu_bl, index=assets)
        covariance_matrix = pd.DataFrame(Sigma_bl, index=assets, columns=assets)
        
        # Use mean-variance optimizer with BL inputs
        mv_optimizer = MeanVarianceOptimizer(self.config)
        result = mv_optimizer.optimize(expected_returns, covariance_matrix, current_weights)
        
        # Update method
        result.method = "black_litterman"
        
        return result
    
    def _get_market_weights(self, assets: List[str], 
                           market_caps: Optional[Dict[str, float]]) -> np.ndarray:
        """Get market capitalization weights"""
        if market_caps:
            caps = np.array([market_caps.get(asset, 1.0) for asset in assets])
            weights = caps / np.sum(caps)
        else:
            # Equal weights if no market cap data
            weights = np.ones(len(assets)) / len(assets)
        
        return weights
    
    def _calculate_equilibrium_returns(self, w_market: np.ndarray, 
                                     Sigma: np.ndarray) -> np.ndarray:
        """Calculate implied equilibrium returns"""
        # Pi = lambda * Sigma * w_market
        # where lambda is the risk aversion parameter
        return self.config.risk_aversion * Sigma @ w_market
    
    def _apply_black_litterman(self, Pi: np.ndarray, Sigma: np.ndarray,
                              views: Dict[str, Tuple[float, float]], 
                              assets: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Apply Black-Litterman model with investor views"""
        
        n_assets = len(assets)
        n_views = len(views)
        
        if n_views == 0:
            return Pi, Sigma
        
        # Create view matrix P and view vector Q
        P = np.zeros((n_views, n_assets))
        Q = np.zeros(n_views)
        Omega = np.zeros((n_views, n_views))
        
        for i, (asset, (view_return, confidence)) in enumerate(views.items()):
            if asset in assets:
                asset_idx = assets.index(asset)
                P[i, asset_idx] = 1.0
                Q[i] = view_return
                # View uncertainty (lower confidence = higher uncertainty)
                view_variance = self.config.bl_tau * Sigma[asset_idx, asset_idx] / confidence
                Omega[i, i] = view_variance
        
        # Black-Litterman formula
        tau = self.config.bl_tau
        
        # New expected returns
        M1 = np.linalg.inv(tau * Sigma)
        M2 = P.T @ np.linalg.inv(Omega) @ P
        M3 = np.linalg.inv(M1 + M2)
        
        mu_bl = M3 @ (M1 @ Pi + P.T @ np.linalg.inv(Omega) @ Q)
        
        # New covariance matrix
        Sigma_bl = M3
        
        return mu_bl, Sigma_bl


class RiskParityOptimizer:
    """
    Risk Parity optimization
    
    Features:
    - Equal risk contribution portfolio
    - Custom risk budgeting
    - Volatility targeting
    - Hierarchical risk parity
    """
    
    def __init__(self, config: OptimizerConfig):
        self.config = config
    
    def optimize(self, covariance_matrix: pd.DataFrame,
                risk_budgets: Optional[Dict[str, float]] = None,
                current_weights: Optional[Dict[str, float]] = None) -> OptimizationResult:
        """Optimize portfolio using risk parity approach"""
        
        assets = covariance_matrix.index.tolist()
        n_assets = len(assets)
        
        if n_assets == 0:
            raise ValueError("No assets provided for optimization")
        
        # Use custom risk budgets if provided, otherwise equal risk budgets
        if risk_budgets is None:
            risk_budgets = {asset: 1.0 / n_assets for asset in assets}
        
        # Normalize risk budgets
        total_budget = sum(risk_budgets.values())
        risk_budgets = {asset: budget / total_budget for asset, budget in risk_budgets.items()}
        
        # Convert to numpy arrays
        Sigma = covariance_matrix.values
        target_risk_contributions = np.array([risk_budgets.get(asset, 0.0) for asset in assets])
        
        # Optimize using risk parity objective
        optimal_weights = self._solve_risk_parity(Sigma, target_risk_contributions)
        
        # Scale to target volatility
        if self.config.rp_volatility_target:
            portfolio_vol = np.sqrt(optimal_weights.T @ Sigma @ optimal_weights)
            scaling_factor = self.config.rp_volatility_target / portfolio_vol
            optimal_weights = optimal_weights * scaling_factor
        
        # Create results dictionary
        weights = {asset: float(optimal_weights[i]) for i, asset in enumerate(assets)}
        
        # Calculate metrics
        expected_volatility = float(np.sqrt(optimal_weights.T @ Sigma @ optimal_weights))
        
        # Calculate risk contributions
        risk_contributions = self._calculate_risk_contributions(optimal_weights, Sigma, assets)
        
        # Calculate additional metrics
        max_weight = float(np.max(np.abs(optimal_weights)))
        concentration = float(np.sum(optimal_weights ** 2))
        effective_assets = 1 / concentration if concentration > 0 else 0.0
        
        turnover = 0.0
        if current_weights:
            current_w = np.array([current_weights.get(asset, 0.0) for asset in assets])
            turnover = float(np.sum(np.abs(optimal_weights - current_w)))
        
        return OptimizationResult(
            weights=weights,
            expected_return=0.0,  # Risk parity doesn't optimize for return
            expected_volatility=expected_volatility,
            sharpe_ratio=0.0,
            method="risk_parity",
            timestamp=datetime.now(),
            max_weight=max_weight,
            turnover=turnover,
            effective_assets=effective_assets,
            concentration=concentration,
            risk_contributions=risk_contributions
        )
    
    def _solve_risk_parity(self, Sigma: np.ndarray, 
                          target_risk_contributions: np.ndarray) -> np.ndarray:
        """Solve risk parity optimization problem"""
        n_assets = Sigma.shape[0]
        
        def risk_parity_objective(weights):
            """Objective function for risk parity optimization"""
            # Calculate risk contributions
            portfolio_var = weights.T @ Sigma @ weights
            marginal_contrib = Sigma @ weights
            risk_contrib = weights * marginal_contrib / portfolio_var
            
            # Sum of squared deviations from target risk contributions
            return np.sum((risk_contrib - target_risk_contributions) ** 2)
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},  # Budget constraint
        ]
        
        # Bounds
        bounds = [(self.config.min_weight, self.config.max_weight) for _ in range(n_assets)]
        
        # Initial guess (equal weights)
        x0 = np.ones(n_assets) / n_assets
        
        # Optimize
        try:
            result = minimize(
                risk_parity_objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000}
            )
            
            if result.success:
                return result.x
            else:
                logger.warning("Risk parity optimization failed, using equal weights")
                return x0
                
        except Exception as e:
            logger.error(f"Risk parity optimization error: {e}")
            return x0
    
    def _calculate_risk_contributions(self, weights: np.ndarray, 
                                    Sigma: np.ndarray, 
                                    assets: List[str]) -> Dict[str, float]:
        """Calculate risk contributions for each asset"""
        portfolio_var = weights.T @ Sigma @ weights
        marginal_contrib = Sigma @ weights
        risk_contrib = weights * marginal_contrib / portfolio_var
        
        return {asset: float(risk_contrib[i]) for i, asset in enumerate(assets)}


class MaxDiversificationOptimizer:
    """
    Maximum Diversification optimization
    
    Features:
    - Maximize diversification ratio
    - Focus on correlation structure
    - Low correlation emphasis
    - Alternative to traditional mean-variance
    """
    
    def __init__(self, config: OptimizerConfig):
        self.config = config
    
    def optimize(self, covariance_matrix: pd.DataFrame,
                current_weights: Optional[Dict[str, float]] = None) -> OptimizationResult:
        """Optimize portfolio for maximum diversification"""
        
        assets = covariance_matrix.index.tolist()
        n_assets = len(assets)
        
        if n_assets == 0:
            raise ValueError("No assets provided for optimization")
        
        # Convert to numpy arrays
        Sigma = covariance_matrix.values
        volatilities = np.sqrt(np.diag(Sigma))
        
        # Decision variables
        w = cp.Variable(n_assets)
        
        # Diversification ratio = weighted average volatility / portfolio volatility
        # Maximize by minimizing portfolio volatility relative to weighted volatility
        portfolio_vol = cp.quad_form(w, Sigma)
        weighted_vol = volatilities.T @ w
        
        # Objective: minimize portfolio vol / weighted vol (equivalent to maximizing div ratio)
        objective = cp.Minimize(portfolio_vol)
        
        # Constraints
        constraints = [
            cp.sum(w) == 1,  # Budget constraint
            volatilities.T @ w == 1,  # Normalize weighted volatility
            w >= self.config.min_weight,  # Min weight
            w <= self.config.max_weight,  # Max weight
        ]
        
        # Add turnover constraint if current weights provided
        if current_weights is not None:
            current_w = np.array([current_weights.get(asset, 0.0) for asset in assets])
            turnover = cp.norm(w - current_w, 1)
            constraints.append(turnover <= self.config.max_turnover)
        
        # Solve optimization
        problem = cp.Problem(objective, constraints)
        
        try:
            problem.solve()
            
            if w.value is None:
                raise ValueError("Max diversification optimization failed")
            
            # Rescale weights to sum to 1
            optimal_weights = w.value / np.sum(w.value)
            
            # Create results dictionary
            weights = {asset: float(optimal_weights[i]) for i, asset in enumerate(assets)}
            
            # Calculate metrics
            portfolio_vol = float(np.sqrt(optimal_weights.T @ Sigma @ optimal_weights))
            weighted_avg_vol = float(volatilities.T @ optimal_weights)
            diversification_ratio = weighted_avg_vol / portfolio_vol if portfolio_vol > 0 else 0.0
            
            # Calculate additional metrics
            max_weight = float(np.max(np.abs(optimal_weights)))
            concentration = float(np.sum(optimal_weights ** 2))
            effective_assets = 1 / concentration if concentration > 0 else 0.0
            
            turnover = 0.0
            if current_weights:
                current_w = np.array([current_weights.get(asset, 0.0) for asset in assets])
                turnover = float(np.sum(np.abs(optimal_weights - current_w)))
            
            return OptimizationResult(
                weights=weights,
                expected_return=0.0,  # Max div doesn't optimize for return
                expected_volatility=portfolio_vol,
                sharpe_ratio=0.0,
                method="max_diversification",
                timestamp=datetime.now(),
                max_weight=max_weight,
                turnover=turnover,
                effective_assets=effective_assets,
                concentration=concentration,
                diversification_ratio=diversification_ratio
            )
            
        except Exception as e:
            logger.error(f"Max diversification optimization failed: {e}")
            # Return equal weight portfolio as fallback
            return self._equal_weight_fallback(assets)
    
    def _equal_weight_fallback(self, assets):
        """Return equal weight portfolio as fallback"""
        n_assets = len(assets)
        equal_weight = 1.0 / n_assets
        
        weights = {asset: equal_weight for asset in assets}
        
        return OptimizationResult(
            weights=weights,
            expected_return=0.0,
            expected_volatility=0.0,
            sharpe_ratio=0.0,
            method="equal_weight_fallback",
            timestamp=datetime.now()
        )


class CovarianceEstimator:
    """
    Advanced covariance matrix estimation
    
    Methods:
    - Sample covariance
    - Ledoit-Wolf shrinkage
    - Oracle Approximating Shrinkage (OAS)
    - Custom shrinkage estimators
    """
    
    def __init__(self, config: OptimizerConfig):
        self.config = config
    
    def estimate_covariance(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Estimate covariance matrix using configured method"""
        
        if len(returns) < self.config.min_periods:
            raise ValueError(f"Insufficient data: {len(returns)} < {self.config.min_periods}")
        
        if self.config.cov_estimator == "sample":
            cov_matrix = returns.cov()
        elif self.config.cov_estimator == "ledoit_wolf":
            lw = LedoitWolf()
            cov_array, _ = lw.fit(returns.values).covariance_, lw.shrinkage_
            cov_matrix = pd.DataFrame(cov_array, index=returns.columns, columns=returns.columns)
        elif self.config.cov_estimator == "oas":
            oas = OAS()
            cov_array, _ = oas.fit(returns.values).covariance_, oas.shrinkage_
            cov_matrix = pd.DataFrame(cov_array, index=returns.columns, columns=returns.columns)
        elif self.config.cov_estimator == "shrunk":
            cov_matrix = self._shrunk_covariance(returns)
        else:
            raise ValueError(f"Unknown covariance estimator: {self.config.cov_estimator}")
        
        # Ensure positive semi-definite
        cov_matrix = self._ensure_positive_definite(cov_matrix)
        
        return cov_matrix
    
    def _shrunk_covariance(self, returns: pd.DataFrame, 
                          shrinkage_factor: float = 0.1) -> pd.DataFrame:
        """Custom shrinkage estimator"""
        sample_cov = returns.cov()
        n_assets = len(sample_cov)
        
        # Shrinkage target (identity matrix scaled by average variance)
        avg_variance = np.trace(sample_cov) / n_assets
        target = np.eye(n_assets) * avg_variance
        target = pd.DataFrame(target, index=sample_cov.index, columns=sample_cov.columns)
        
        # Shrunk covariance
        shrunk_cov = (1 - shrinkage_factor) * sample_cov + shrinkage_factor * target
        
        return shrunk_cov
    
    def _ensure_positive_definite(self, cov_matrix: pd.DataFrame, 
                                 regularization: float = 1e-6) -> pd.DataFrame:
        """Ensure covariance matrix is positive definite"""
        # Convert to numpy for eigenvalue operations
        cov_array = cov_matrix.values
        
        # Eigenvalue decomposition
        eigenvals, eigenvecs = np.linalg.eigh(cov_array)
        
        # Ensure all eigenvalues are positive
        eigenvals = np.maximum(eigenvals, regularization)
        
        # Reconstruct matrix
        cov_regularized = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
        
        return pd.DataFrame(cov_regularized, index=cov_matrix.index, columns=cov_matrix.columns)


class PortfolioOptimizer:
    """
    Main portfolio optimization system
    
    Features:
    - Multiple optimization methods
    - Advanced covariance estimation
    - Dynamic rebalancing
    - Risk management integration
    """
    
    def __init__(self, config: OptimizerConfig = None):
        self.config = config or OptimizerConfig()
        
        # Initialize optimizers
        self.mv_optimizer = MeanVarianceOptimizer(self.config)
        self.bl_optimizer = BlackLittermanOptimizer(self.config)
        self.rp_optimizer = RiskParityOptimizer(self.config)
        self.md_optimizer = MaxDiversificationOptimizer(self.config)
        
        # Initialize covariance estimator
        self.cov_estimator = CovarianceEstimator(self.config)
        
        logger.info(f"Initialized portfolio optimizer with method: {self.config.optimization_method}")
    
    def optimize_portfolio(self, returns_data: pd.DataFrame,
                          current_weights: Optional[Dict[str, float]] = None,
                          expected_returns: Optional[pd.Series] = None,
                          market_caps: Optional[Dict[str, float]] = None,
                          views: Optional[Dict[str, Tuple[float, float]]] = None,
                          risk_budgets: Optional[Dict[str, float]] = None) -> OptimizationResult:
        """Optimize portfolio using configured method"""
        
        if returns_data.empty:
            raise ValueError("No returns data provided")
        
        # Estimate covariance matrix
        covariance_matrix = self.cov_estimator.estimate_covariance(returns_data)
        
        # Calculate expected returns if not provided
        if expected_returns is None:
            expected_returns = self._estimate_expected_returns(returns_data)
        
        # Run optimization based on method
        if self.config.optimization_method == "mean_variance":
            result = self.mv_optimizer.optimize(expected_returns, covariance_matrix, current_weights)
        elif self.config.optimization_method == "black_litterman":
            result = self.bl_optimizer.optimize(returns_data, market_caps, views, current_weights)
        elif self.config.optimization_method == "risk_parity":
            result = self.rp_optimizer.optimize(covariance_matrix, risk_budgets, current_weights)
        elif self.config.optimization_method == "max_diversification":
            result = self.md_optimizer.optimize(covariance_matrix, current_weights)
        else:
            raise ValueError(f"Unknown optimization method: {self.config.optimization_method}")
        
        return result
    
    def _estimate_expected_returns(self, returns_data: pd.DataFrame) -> pd.Series:
        """Estimate expected returns from historical data"""
        # Use historical mean returns (annualized)
        return returns_data.mean() * 252
    
    def run_optimization_comparison(self, returns_data: pd.DataFrame,
                                  current_weights: Optional[Dict[str, float]] = None,
                                  **kwargs) -> Dict[str, OptimizationResult]:
        """Run optimization using all methods for comparison"""
        
        methods = ["mean_variance", "black_litterman", "risk_parity", "max_diversification"]
        results = {}
        
        for method in methods:
            try:
                # Temporarily change method
                original_method = self.config.optimization_method
                self.config.optimization_method = method
                
                # Run optimization
                result = self.optimize_portfolio(returns_data, current_weights, **kwargs)
                results[method] = result
                
                # Restore original method
                self.config.optimization_method = original_method
                
            except Exception as e:
                logger.error(f"Optimization method {method} failed: {e}")
                continue
        
        return results
    
    def generate_efficient_frontier(self, returns_data: pd.DataFrame,
                                  n_portfolios: int = 50) -> Dict[str, List[float]]:
        """Generate efficient frontier for mean-variance optimization"""
        
        expected_returns = self._estimate_expected_returns(returns_data)
        covariance_matrix = self.cov_estimator.estimate_covariance(returns_data)
        
        # Risk range
        min_var_result = self._minimize_variance(expected_returns, covariance_matrix)
        max_return = expected_returns.max()
        min_return = min_var_result.expected_return
        
        # Generate portfolios along frontier
        target_returns = np.linspace(min_return, max_return, n_portfolios)
        
        frontier_returns = []
        frontier_volatilities = []
        frontier_sharpe_ratios = []
        
        for target_return in target_returns:
            try:
                result = self._target_return_optimization(expected_returns, covariance_matrix, target_return)
                frontier_returns.append(result.expected_return)
                frontier_volatilities.append(result.expected_volatility)
                frontier_sharpe_ratios.append(result.sharpe_ratio)
            except:
                continue
        
        return {
            'returns': frontier_returns,
            'volatilities': frontier_volatilities,
            'sharpe_ratios': frontier_sharpe_ratios
        }
    
    def _minimize_variance(self, expected_returns: pd.Series, 
                         covariance_matrix: pd.DataFrame) -> OptimizationResult:
        """Find minimum variance portfolio"""
        assets = expected_returns.index.tolist()
        n_assets = len(assets)
        
        # Decision variables
        w = cp.Variable(n_assets)
        
        # Objective: minimize variance
        objective = cp.Minimize(cp.quad_form(w, covariance_matrix.values))
        
        # Constraints
        constraints = [
            cp.sum(w) == 1,
            w >= self.config.min_weight,
            w <= self.config.max_weight
        ]
        
        # Solve
        problem = cp.Problem(objective, constraints)
        problem.solve()
        
        weights = {asset: float(w.value[i]) for i, asset in enumerate(assets)}
        expected_return = float(expected_returns.values.T @ w.value)
        expected_volatility = float(np.sqrt(w.value.T @ covariance_matrix.values @ w.value))
        
        return OptimizationResult(
            weights=weights,
            expected_return=expected_return,
            expected_volatility=expected_volatility,
            sharpe_ratio=expected_return / expected_volatility if expected_volatility > 0 else 0,
            method="min_variance",
            timestamp=datetime.now()
        )
    
    def _target_return_optimization(self, expected_returns: pd.Series,
                                  covariance_matrix: pd.DataFrame,
                                  target_return: float) -> OptimizationResult:
        """Optimize for target return"""
        assets = expected_returns.index.tolist()
        n_assets = len(assets)
        
        # Decision variables
        w = cp.Variable(n_assets)
        
        # Objective: minimize variance
        objective = cp.Minimize(cp.quad_form(w, covariance_matrix.values))
        
        # Constraints
        constraints = [
            cp.sum(w) == 1,
            expected_returns.values.T @ w == target_return,
            w >= self.config.min_weight,
            w <= self.config.max_weight
        ]
        
        # Solve
        problem = cp.Problem(objective, constraints)
        problem.solve()
        
        weights = {asset: float(w.value[i]) for i, asset in enumerate(assets)}
        expected_return = float(expected_returns.values.T @ w.value)
        expected_volatility = float(np.sqrt(w.value.T @ covariance_matrix.values @ w.value))
        
        return OptimizationResult(
            weights=weights,
            expected_return=expected_return,
            expected_volatility=expected_volatility,
            sharpe_ratio=expected_return / expected_volatility if expected_volatility > 0 else 0,
            method="target_return",
            timestamp=datetime.now()
        )


# Factory function
def create_portfolio_optimizer(config: Dict[str, Any] = None) -> PortfolioOptimizer:
    """Create portfolio optimizer with configuration"""
    optimizer_config = OptimizerConfig(**(config or {}))
    return PortfolioOptimizer(optimizer_config)


# Export classes and functions
__all__ = [
    'OptimizerConfig',
    'OptimizationResult',
    'MeanVarianceOptimizer',
    'BlackLittermanOptimizer',
    'RiskParityOptimizer',
    'MaxDiversificationOptimizer',
    'CovarianceEstimator',
    'PortfolioOptimizer',
    'create_portfolio_optimizer'
]