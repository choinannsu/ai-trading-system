"""
Risk Metrics System
Real-time VaR calculation, conditional VaR, stress testing, and scenario analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import warnings
from scipy import stats
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf
import matplotlib.pyplot as plt

from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class RiskMetricsConfig:
    """Configuration for risk metrics system"""
    # VaR parameters
    var_confidence_levels: List[float] = field(default_factory=lambda: [0.95, 0.99])
    var_window: int = 252  # Window for VaR calculation
    var_method: str = "historical"  # "historical", "parametric", "monte_carlo"
    
    # CVaR parameters
    cvar_confidence_levels: List[float] = field(default_factory=lambda: [0.95, 0.99])
    
    # Monte Carlo parameters
    mc_simulations: int = 10000
    mc_time_horizon: int = 22  # Trading days (1 month)
    
    # Stress testing
    stress_scenarios: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        'market_crash': {'equity': -0.20, 'bond': -0.05, 'commodity': -0.15},
        'interest_rate_shock': {'equity': -0.10, 'bond': -0.15, 'commodity': 0.05},
        'inflation_spike': {'equity': -0.05, 'bond': -0.20, 'commodity': 0.20},
        'liquidity_crisis': {'equity': -0.25, 'bond': 0.05, 'commodity': -0.10}
    })
    
    # Rolling window parameters
    rolling_window: int = 60  # Days for rolling metrics
    update_frequency: str = "daily"  # "daily", "intraday", "weekly"
    
    # Extreme value parameters
    extreme_threshold: float = 0.95  # Threshold for extreme value analysis
    
    # Correlation parameters
    correlation_window: int = 60
    correlation_threshold: float = 0.8  # High correlation threshold


@dataclass
class VaRResult:
    """Value at Risk calculation result"""
    confidence_level: float
    var_value: float
    method: str
    time_horizon: int
    timestamp: datetime
    
    # Additional statistics
    expected_shortfall: Optional[float] = None
    volatility: Optional[float] = None
    skewness: Optional[float] = None
    kurtosis: Optional[float] = None


@dataclass
class StressTestResult:
    """Stress test result"""
    scenario_name: str
    portfolio_impact: float
    asset_impacts: Dict[str, float]
    probability: Optional[float] = None
    recovery_time: Optional[int] = None


class VaRCalculator:
    """
    Value at Risk calculator with multiple methodologies
    
    Methods:
    - Historical simulation
    - Parametric (normal and t-distribution)
    - Monte Carlo simulation
    - Extreme value theory
    """
    
    def __init__(self, config: RiskMetricsConfig):
        self.config = config
        
    def calculate_var(self, returns: pd.Series, 
                     confidence_level: float = 0.95,
                     method: str = "historical",
                     time_horizon: int = 1) -> VaRResult:
        """Calculate Value at Risk using specified method"""
        if len(returns) < 30:
            logger.warning("Insufficient data for VaR calculation")
            return VaRResult(
                confidence_level=confidence_level,
                var_value=0.0,
                method=method,
                time_horizon=time_horizon,
                timestamp=datetime.now()
            )
        
        # Clean returns data
        returns_clean = returns.dropna()
        
        if method == "historical":
            var_value = self._historical_var(returns_clean, confidence_level)
        elif method == "parametric":
            var_value = self._parametric_var(returns_clean, confidence_level)
        elif method == "monte_carlo":
            var_value = self._monte_carlo_var(returns_clean, confidence_level, time_horizon)
        elif method == "extreme_value":
            var_value = self._extreme_value_var(returns_clean, confidence_level)
        else:
            raise ValueError(f"Unknown VaR method: {method}")
        
        # Scale for time horizon
        if time_horizon > 1:
            var_value = var_value * np.sqrt(time_horizon)
        
        # Calculate additional statistics
        expected_shortfall = self._calculate_expected_shortfall(returns_clean, confidence_level)
        volatility = returns_clean.std() * np.sqrt(252)
        skewness = returns_clean.skew()
        kurtosis = returns_clean.kurtosis()
        
        return VaRResult(
            confidence_level=confidence_level,
            var_value=var_value,
            method=method,
            time_horizon=time_horizon,
            timestamp=datetime.now(),
            expected_shortfall=expected_shortfall,
            volatility=volatility,
            skewness=skewness,
            kurtosis=kurtosis
        )
    
    def _historical_var(self, returns: pd.Series, confidence_level: float) -> float:
        """Calculate VaR using historical simulation"""
        return -returns.quantile(1 - confidence_level)
    
    def _parametric_var(self, returns: pd.Series, confidence_level: float) -> float:
        """Calculate VaR using parametric approach"""
        mean = returns.mean()
        std = returns.std()
        
        # Test for normality
        _, p_value = stats.jarque_bera(returns)
        
        if p_value > 0.05:  # Normal distribution
            z_score = stats.norm.ppf(1 - confidence_level)
        else:  # Use t-distribution for fat tails
            df = len(returns) - 1
            z_score = stats.t.ppf(1 - confidence_level, df)
        
        var_value = -(mean + z_score * std)
        return var_value
    
    def _monte_carlo_var(self, returns: pd.Series, confidence_level: float, 
                        time_horizon: int) -> float:
        """Calculate VaR using Monte Carlo simulation"""
        mean = returns.mean()
        std = returns.std()
        
        # Generate random scenarios
        random_returns = np.random.normal(mean, std, 
                                        (self.config.mc_simulations, time_horizon))
        
        # Calculate cumulative returns for each scenario
        cumulative_returns = np.prod(1 + random_returns, axis=1) - 1
        
        # Calculate VaR
        var_value = -np.percentile(cumulative_returns, (1 - confidence_level) * 100)
        
        return var_value / np.sqrt(time_horizon)  # Normalize for single period
    
    def _extreme_value_var(self, returns: pd.Series, confidence_level: float) -> float:
        """Calculate VaR using Extreme Value Theory"""
        # Use Peak Over Threshold (POT) method
        threshold = returns.quantile(self.config.extreme_threshold)
        excesses = returns[returns > threshold] - threshold
        
        if len(excesses) < 10:
            # Fallback to historical VaR
            return self._historical_var(returns, confidence_level)
        
        # Fit Generalized Pareto Distribution
        try:
            shape, loc, scale = stats.genpareto.fit(excesses)
            
            # Calculate VaR using GPD
            n = len(returns)
            nu = len(excesses)
            
            var_quantile = 1 - (1 - confidence_level) * n / nu
            
            if var_quantile <= 1:
                var_value = threshold + (scale / shape) * \
                           ((1 / (1 - var_quantile)) ** shape - 1)
                return -var_value
            else:
                return self._historical_var(returns, confidence_level)
                
        except Exception:
            return self._historical_var(returns, confidence_level)
    
    def _calculate_expected_shortfall(self, returns: pd.Series, 
                                    confidence_level: float) -> float:
        """Calculate Expected Shortfall (Conditional VaR)"""
        var_threshold = returns.quantile(1 - confidence_level)
        tail_returns = returns[returns <= var_threshold]
        
        if len(tail_returns) > 0:
            return -tail_returns.mean()
        else:
            return 0.0
    
    def calculate_portfolio_var(self, portfolio_returns: pd.Series,
                              asset_weights: Dict[str, float],
                              asset_returns: pd.DataFrame,
                              confidence_level: float = 0.95) -> Dict[str, Any]:
        """Calculate portfolio VaR with component contributions"""
        portfolio_var = self.calculate_var(portfolio_returns, confidence_level)
        
        # Calculate component VaR
        component_vars = {}
        
        if not asset_returns.empty and asset_weights:
            # Covariance matrix
            cov_matrix = asset_returns.cov() * 252  # Annualized
            
            # Portfolio volatility
            weights = np.array([asset_weights.get(asset, 0) for asset in asset_returns.columns])
            portfolio_vol = np.sqrt(weights.T @ cov_matrix.values @ weights)
            
            # Marginal VaR (derivative of portfolio VaR w.r.t. weights)
            marginal_vars = (cov_matrix.values @ weights) / portfolio_vol
            
            # Component VaR
            for i, asset in enumerate(asset_returns.columns):
                if asset in asset_weights:
                    component_var = asset_weights[asset] * marginal_vars[i]
                    component_vars[asset] = component_var * portfolio_var.var_value / portfolio_vol
        
        return {
            'portfolio_var': portfolio_var,
            'component_vars': component_vars,
            'total_component_var': sum(component_vars.values())
        }


class StressTester:
    """
    Stress testing system for portfolio risk assessment
    
    Features:
    - Predefined stress scenarios
    - Custom scenario testing
    - Historical scenario replay
    - Monte Carlo stress testing
    """
    
    def __init__(self, config: RiskMetricsConfig):
        self.config = config
        
    def run_stress_test(self, portfolio_weights: Dict[str, float],
                       asset_returns: pd.DataFrame,
                       scenario: Optional[Dict[str, float]] = None,
                       scenario_name: str = "custom") -> StressTestResult:
        """Run stress test on portfolio"""
        
        if scenario is None:
            scenario = self.config.stress_scenarios.get(scenario_name, {})
        
        if not scenario:
            raise ValueError(f"No scenario data for: {scenario_name}")
        
        # Calculate asset impacts
        asset_impacts = {}
        portfolio_impact = 0.0
        
        for asset, weight in portfolio_weights.items():
            if asset in scenario:
                asset_impact = scenario[asset] * weight
                asset_impacts[asset] = asset_impact
                portfolio_impact += asset_impact
            else:
                # Use average market impact for assets not in scenario
                avg_impact = np.mean(list(scenario.values()))
                asset_impact = avg_impact * weight * 0.5  # Reduced correlation
                asset_impacts[asset] = asset_impact
                portfolio_impact += asset_impact
        
        # Estimate probability based on historical data
        probability = self._estimate_scenario_probability(asset_returns, scenario)
        
        # Estimate recovery time
        recovery_time = self._estimate_recovery_time(asset_returns, scenario)
        
        return StressTestResult(
            scenario_name=scenario_name,
            portfolio_impact=portfolio_impact,
            asset_impacts=asset_impacts,
            probability=probability,
            recovery_time=recovery_time
        )
    
    def run_all_scenarios(self, portfolio_weights: Dict[str, float],
                         asset_returns: pd.DataFrame) -> List[StressTestResult]:
        """Run all predefined stress scenarios"""
        results = []
        
        for scenario_name in self.config.stress_scenarios.keys():
            try:
                result = self.run_stress_test(portfolio_weights, asset_returns, 
                                            scenario_name=scenario_name)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to run scenario {scenario_name}: {e}")
        
        return results
    
    def historical_scenario_replay(self, portfolio_weights: Dict[str, float],
                                 asset_returns: pd.DataFrame,
                                 start_date: datetime,
                                 end_date: datetime) -> StressTestResult:
        """Replay historical period as stress scenario"""
        
        # Filter returns for specified period
        historical_period = asset_returns[
            (asset_returns.index >= start_date) & 
            (asset_returns.index <= end_date)
        ]
        
        if historical_period.empty:
            raise ValueError("No data available for specified historical period")
        
        # Calculate cumulative impact
        cumulative_returns = (1 + historical_period).prod() - 1
        
        # Calculate portfolio impact
        portfolio_impact = 0.0
        asset_impacts = {}
        
        for asset, weight in portfolio_weights.items():
            if asset in cumulative_returns.index:
                asset_impact = cumulative_returns[asset] * weight
                asset_impacts[asset] = asset_impact
                portfolio_impact += asset_impact
        
        scenario_name = f"Historical_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"
        
        return StressTestResult(
            scenario_name=scenario_name,
            portfolio_impact=portfolio_impact,
            asset_impacts=asset_impacts,
            probability=None,  # Historical event
            recovery_time=len(historical_period)
        )
    
    def monte_carlo_stress_test(self, portfolio_weights: Dict[str, float],
                              asset_returns: pd.DataFrame,
                              num_simulations: int = 1000) -> Dict[str, Any]:
        """Run Monte Carlo stress testing"""
        
        # Calculate covariance matrix
        cov_matrix = asset_returns.cov()
        mean_returns = asset_returns.mean()
        
        # Generate correlated random scenarios
        portfolio_impacts = []
        worst_scenarios = []
        
        for _ in range(num_simulations):
            # Generate correlated random shocks
            random_shocks = np.random.multivariate_normal(
                mean_returns.values, cov_matrix.values
            )
            
            # Calculate portfolio impact
            portfolio_impact = 0.0
            scenario = {}
            
            for i, asset in enumerate(asset_returns.columns):
                if asset in portfolio_weights:
                    asset_impact = random_shocks[i] * portfolio_weights[asset]
                    portfolio_impact += asset_impact
                    scenario[asset] = random_shocks[i]
            
            portfolio_impacts.append(portfolio_impact)
            
            # Store worst scenarios
            if len(worst_scenarios) < 10 or portfolio_impact < min(worst_scenarios, key=lambda x: x['impact'])['impact']:
                if len(worst_scenarios) >= 10:
                    worst_scenarios.remove(min(worst_scenarios, key=lambda x: x['impact']))
                worst_scenarios.append({'impact': portfolio_impact, 'scenario': scenario})
        
        portfolio_impacts = np.array(portfolio_impacts)
        
        return {
            'portfolio_impacts': portfolio_impacts,
            'worst_case': np.min(portfolio_impacts),
            'percentile_1': np.percentile(portfolio_impacts, 1),
            'percentile_5': np.percentile(portfolio_impacts, 5),
            'mean_impact': np.mean(portfolio_impacts),
            'volatility': np.std(portfolio_impacts),
            'worst_scenarios': sorted(worst_scenarios, key=lambda x: x['impact'])
        }
    
    def _estimate_scenario_probability(self, asset_returns: pd.DataFrame,
                                     scenario: Dict[str, float]) -> float:
        """Estimate probability of scenario based on historical data"""
        if asset_returns.empty:
            return 0.0
        
        # Count how often similar conditions occurred
        similar_events = 0
        total_periods = len(asset_returns)
        
        tolerance = 0.05  # 5% tolerance
        
        for _, row in asset_returns.iterrows():
            is_similar = True
            for asset, shock in scenario.items():
                if asset in row.index:
                    if abs(row[asset] - shock) > tolerance:
                        is_similar = False
                        break
            
            if is_similar:
                similar_events += 1
        
        return similar_events / total_periods if total_periods > 0 else 0.0
    
    def _estimate_recovery_time(self, asset_returns: pd.DataFrame,
                              scenario: Dict[str, float]) -> Optional[int]:
        """Estimate recovery time from scenario impact"""
        if asset_returns.empty:
            return None
        
        # Find similar historical events and measure recovery time
        recovery_times = []
        
        for i in range(len(asset_returns) - 20):  # Need at least 20 days for recovery
            period_returns = asset_returns.iloc[i:i+5]  # 5-day window
            
            # Check if this period is similar to scenario
            is_similar = True
            for asset, shock in scenario.items():
                if asset in period_returns.columns:
                    period_impact = period_returns[asset].sum()
                    if abs(period_impact - shock) > 0.05:
                        is_similar = False
                        break
            
            if is_similar:
                # Measure recovery time
                recovery_start = i + 5
                cumulative_return = 0.0
                recovery_days = 0
                
                for j in range(recovery_start, min(recovery_start + 60, len(asset_returns))):
                    day_return = 0.0
                    for asset in scenario.keys():
                        if asset in asset_returns.columns:
                            day_return += asset_returns.iloc[j][asset] / len(scenario)
                    
                    cumulative_return += day_return
                    recovery_days += 1
                    
                    # Check if recovered (positive cumulative return)
                    if cumulative_return >= 0:
                        recovery_times.append(recovery_days)
                        break
        
        return int(np.mean(recovery_times)) if recovery_times else None


class RiskMetricsCalculator:
    """
    Comprehensive risk metrics calculation system
    
    Features:
    - Real-time VaR and CVaR
    - Multiple VaR methodologies
    - Stress testing and scenario analysis
    - Risk decomposition and attribution
    - Dynamic risk monitoring
    """
    
    def __init__(self, config: RiskMetricsConfig = None):
        self.config = config or RiskMetricsConfig()
        
        # Initialize calculators
        self.var_calculator = VaRCalculator(self.config)
        self.stress_tester = StressTester(self.config)
        
        # Cache for rolling calculations
        self.rolling_metrics: Dict[str, pd.DataFrame] = {}
        
        logger.info("Initialized risk metrics calculator")
    
    def calculate_comprehensive_risk_metrics(self, 
                                           portfolio_returns: pd.Series,
                                           asset_returns: pd.DataFrame,
                                           portfolio_weights: Dict[str, float],
                                           current_value: float) -> Dict[str, Any]:
        """Calculate comprehensive risk metrics for portfolio"""
        
        metrics = {
            'timestamp': datetime.now(),
            'portfolio_value': current_value,
            'var_metrics': {},
            'stress_test_results': [],
            'risk_decomposition': {},
            'rolling_metrics': {}
        }
        
        # Calculate VaR for all confidence levels
        for confidence_level in self.config.var_confidence_levels:
            var_result = self.var_calculator.calculate_var(
                portfolio_returns, confidence_level, self.config.var_method
            )
            metrics['var_metrics'][f'var_{int(confidence_level*100)}'] = var_result
        
        # Calculate portfolio VaR with component analysis
        if not asset_returns.empty and portfolio_weights:
            portfolio_var_analysis = self.var_calculator.calculate_portfolio_var(
                portfolio_returns, portfolio_weights, asset_returns
            )
            metrics['risk_decomposition'] = portfolio_var_analysis
        
        # Run stress tests
        if portfolio_weights:
            stress_results = self.stress_tester.run_all_scenarios(
                portfolio_weights, asset_returns
            )
            metrics['stress_test_results'] = stress_results
            
            # Monte Carlo stress test
            mc_stress = self.stress_tester.monte_carlo_stress_test(
                portfolio_weights, asset_returns, self.config.mc_simulations
            )
            metrics['monte_carlo_stress'] = mc_stress
        
        # Calculate rolling metrics
        rolling_metrics = self._calculate_rolling_metrics(portfolio_returns)
        metrics['rolling_metrics'] = rolling_metrics
        
        # Additional risk metrics
        additional_metrics = self._calculate_additional_metrics(
            portfolio_returns, asset_returns, portfolio_weights
        )
        metrics.update(additional_metrics)
        
        return metrics
    
    def _calculate_rolling_metrics(self, returns: pd.Series) -> Dict[str, Any]:
        """Calculate rolling risk metrics"""
        if len(returns) < self.config.rolling_window:
            return {}
        
        rolling_window = self.config.rolling_window
        
        rolling_metrics = {
            'rolling_volatility': returns.rolling(rolling_window).std() * np.sqrt(252),
            'rolling_var_95': returns.rolling(rolling_window).quantile(0.05) * -1,
            'rolling_sharpe': (returns.rolling(rolling_window).mean() * 252) / 
                             (returns.rolling(rolling_window).std() * np.sqrt(252)),
            'rolling_max_drawdown': self._calculate_rolling_max_drawdown(returns, rolling_window)
        }
        
        # Get latest values
        latest_metrics = {}
        for metric_name, series in rolling_metrics.items():
            if not series.empty:
                latest_metrics[metric_name] = series.iloc[-1]
        
        return latest_metrics
    
    def _calculate_rolling_max_drawdown(self, returns: pd.Series, window: int) -> pd.Series:
        """Calculate rolling maximum drawdown"""
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.rolling(window).max()
        drawdown = (cumulative - rolling_max) / rolling_max
        return drawdown.rolling(window).min()
    
    def _calculate_additional_metrics(self, portfolio_returns: pd.Series,
                                    asset_returns: pd.DataFrame,
                                    portfolio_weights: Dict[str, float]) -> Dict[str, Any]:
        """Calculate additional risk metrics"""
        additional = {}
        
        # Tail risk metrics
        if len(portfolio_returns) >= 30:
            additional['tail_ratio'] = self._calculate_tail_ratio(portfolio_returns)
            additional['gain_pain_ratio'] = self._calculate_gain_pain_ratio(portfolio_returns)
        
        # Concentration risk
        if portfolio_weights:
            additional['concentration_risk'] = self._calculate_concentration_risk(portfolio_weights)
            additional['effective_assets'] = self._calculate_effective_number_assets(portfolio_weights)
        
        # Correlation risk
        if not asset_returns.empty and len(asset_returns.columns) > 1:
            correlation_metrics = self._calculate_correlation_risk(asset_returns, portfolio_weights)
            additional['correlation_risk'] = correlation_metrics
        
        # Liquidity risk (simplified)
        additional['liquidity_risk'] = self._estimate_liquidity_risk(asset_returns)
        
        return additional
    
    def _calculate_tail_ratio(self, returns: pd.Series) -> float:
        """Calculate tail ratio (95th percentile / 5th percentile)"""
        p95 = returns.quantile(0.95)
        p5 = returns.quantile(0.05)
        
        return p95 / abs(p5) if p5 != 0 else 0.0
    
    def _calculate_gain_pain_ratio(self, returns: pd.Series) -> float:
        """Calculate gain-to-pain ratio"""
        positive_returns = returns[returns > 0]
        negative_returns = returns[returns < 0]
        
        if len(positive_returns) == 0 or len(negative_returns) == 0:
            return 0.0
        
        avg_gain = positive_returns.mean()
        avg_pain = abs(negative_returns.mean())
        
        return avg_gain / avg_pain if avg_pain > 0 else 0.0
    
    def _calculate_concentration_risk(self, weights: Dict[str, float]) -> float:
        """Calculate concentration risk using Herfindahl-Hirschman Index"""
        weight_values = np.array(list(weights.values()))
        total_weight = np.sum(np.abs(weight_values))
        
        if total_weight == 0:
            return 0.0
        
        normalized_weights = np.abs(weight_values) / total_weight
        hhi = np.sum(normalized_weights ** 2)
        
        return hhi
    
    def _calculate_effective_number_assets(self, weights: Dict[str, float]) -> float:
        """Calculate effective number of assets"""
        concentration = self._calculate_concentration_risk(weights)
        return 1 / concentration if concentration > 0 else 0.0
    
    def _calculate_correlation_risk(self, asset_returns: pd.DataFrame,
                                  weights: Dict[str, float]) -> Dict[str, Any]:
        """Calculate correlation-based risk metrics"""
        if asset_returns.empty or len(asset_returns.columns) < 2:
            return {}
        
        # Calculate correlation matrix
        corr_matrix = asset_returns.corr()
        
        # Average correlation
        n = len(corr_matrix)
        avg_correlation = (corr_matrix.sum().sum() - n) / (n * (n - 1))
        
        # Weighted average correlation
        weight_values = np.array([weights.get(asset, 0) for asset in corr_matrix.index])
        total_weight = np.sum(np.abs(weight_values))
        
        if total_weight > 0:
            norm_weights = np.abs(weight_values) / total_weight
            weighted_corr = 0.0
            
            for i in range(n):
                for j in range(i+1, n):
                    weighted_corr += norm_weights[i] * norm_weights[j] * corr_matrix.iloc[i, j]
            
            weighted_corr = weighted_corr * 2  # Account for symmetry
        else:
            weighted_corr = 0.0
        
        # Maximum correlation
        max_correlation = corr_matrix.max().max()
        
        # Count of high correlations
        high_corr_count = ((corr_matrix > self.config.correlation_threshold).sum().sum() - n) / 2
        
        return {
            'average_correlation': avg_correlation,
            'weighted_correlation': weighted_corr,
            'maximum_correlation': max_correlation,
            'high_correlation_pairs': int(high_corr_count)
        }
    
    def _estimate_liquidity_risk(self, asset_returns: pd.DataFrame) -> float:
        """Estimate liquidity risk based on return patterns"""
        if asset_returns.empty:
            return 0.0
        
        # Use return autocorrelation as proxy for liquidity
        autocorrelations = []
        
        for column in asset_returns.columns:
            returns = asset_returns[column].dropna()
            if len(returns) > 30:
                autocorr = returns.autocorr(lag=1)
                if not np.isnan(autocorr):
                    autocorrelations.append(abs(autocorr))
        
        return np.mean(autocorrelations) if autocorrelations else 0.0
    
    def generate_risk_report(self, metrics: Dict[str, Any]) -> str:
        """Generate formatted risk report"""
        report = []
        report.append("=" * 60)
        report.append("PORTFOLIO RISK REPORT")
        report.append("=" * 60)
        report.append(f"Generated: {metrics['timestamp']}")
        report.append(f"Portfolio Value: ${metrics['portfolio_value']:,.2f}")
        report.append("")
        
        # VaR metrics
        report.append("VALUE AT RISK")
        report.append("-" * 30)
        for var_key, var_result in metrics['var_metrics'].items():
            if isinstance(var_result, VaRResult):
                confidence = int(var_result.confidence_level * 100)
                report.append(f"VaR {confidence}%: ${var_result.var_value * metrics['portfolio_value']:,.2f}")
                if var_result.expected_shortfall:
                    report.append(f"ES {confidence}%: ${var_result.expected_shortfall * metrics['portfolio_value']:,.2f}")
        report.append("")
        
        # Stress test results
        if metrics['stress_test_results']:
            report.append("STRESS TEST RESULTS")
            report.append("-" * 30)
            for stress_result in metrics['stress_test_results']:
                impact_value = stress_result.portfolio_impact * metrics['portfolio_value']
                report.append(f"{stress_result.scenario_name}: ${impact_value:,.2f}")
                if stress_result.probability:
                    report.append(f"  Probability: {stress_result.probability:.2%}")
        report.append("")
        
        # Additional metrics
        if 'tail_ratio' in metrics:
            report.append("ADDITIONAL RISK METRICS")
            report.append("-" * 30)
            report.append(f"Tail Ratio: {metrics['tail_ratio']:.2f}")
            if 'gain_pain_ratio' in metrics:
                report.append(f"Gain/Pain Ratio: {metrics['gain_pain_ratio']:.2f}")
            if 'concentration_risk' in metrics:
                report.append(f"Concentration Risk: {metrics['concentration_risk']:.3f}")
        
        return "\n".join(report)


# Factory function
def create_risk_calculator(config: Dict[str, Any] = None) -> RiskMetricsCalculator:
    """Create risk metrics calculator with configuration"""
    risk_config = RiskMetricsConfig(**(config or {}))
    return RiskMetricsCalculator(risk_config)


# Export classes and functions
__all__ = [
    'RiskMetricsConfig',
    'VaRResult',
    'StressTestResult',
    'VaRCalculator',
    'StressTester',
    'RiskMetricsCalculator',
    'create_risk_calculator'
]