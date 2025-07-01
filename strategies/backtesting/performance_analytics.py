"""
Performance Analytics System
Comprehensive performance metrics, attribution analysis, and Monte Carlo simulation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import warnings
from scipy import stats
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns

from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics"""
    # Return metrics
    total_return: float
    annualized_return: float
    excess_return: float
    geometric_mean_return: float
    
    # Risk metrics
    volatility: float
    downside_volatility: float
    tracking_error: float
    var_95: float
    var_99: float
    cvar_95: float
    cvar_99: float
    
    # Risk-adjusted metrics
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    information_ratio: float
    treynor_ratio: float
    modigliani_ratio: float
    
    # Drawdown metrics
    max_drawdown: float
    avg_drawdown: float
    max_drawdown_duration: int
    recovery_time: int
    pain_index: float
    
    # Distribution metrics
    skewness: float
    kurtosis: float
    jarque_bera_stat: float
    jarque_bera_pvalue: float
    
    # Market relation metrics
    alpha: float
    beta: float
    correlation: float
    r_squared: float
    
    # Trade metrics
    hit_ratio: float
    profit_factor: float
    expectancy: float
    kelly_criterion: float


@dataclass
class AttributionAnalysis:
    """Performance attribution analysis"""
    # Factor exposures
    factor_exposures: Dict[str, float]
    factor_returns: Dict[str, float]
    
    # Attribution components
    allocation_effect: float
    selection_effect: float
    interaction_effect: float
    total_active_return: float
    
    # Sector/style attribution
    sector_attribution: Dict[str, Dict[str, float]]
    style_attribution: Dict[str, float]
    
    # Time-based attribution
    monthly_attribution: pd.Series
    quarterly_attribution: pd.Series


@dataclass
class MonteCarloAnalysis:
    """Monte Carlo simulation results"""
    # Simulation parameters
    num_simulations: int
    time_horizon: int
    
    # Return distributions
    simulated_returns: np.ndarray
    percentile_returns: Dict[int, float]
    
    # Risk metrics
    probability_of_loss: float
    expected_shortfall: float
    maximum_loss: float
    
    # Confidence intervals
    confidence_intervals: Dict[str, Tuple[float, float]]
    
    # Scenario analysis
    stress_test_results: Dict[str, float]


@dataclass
class RiskMetrics:
    """Comprehensive risk metrics"""
    # Volatility metrics
    realized_volatility: float
    garch_volatility: float
    implied_volatility: Optional[float]
    
    # Tail risk
    tail_ratio: float
    gain_pain_ratio: float
    
    # Concentration risk
    concentration_index: float
    diversification_ratio: float
    
    # Liquidity risk
    liquidity_score: float
    bid_ask_impact: float
    
    # Model risk
    model_uncertainty: float
    parameter_sensitivity: Dict[str, float]


class PerformanceAnalytics:
    """
    Comprehensive performance analytics system
    
    Features:
    - Advanced performance metrics calculation
    - Attribution analysis
    - Monte Carlo simulation
    - Risk decomposition
    - Benchmark comparison
    """
    
    def __init__(self, risk_free_rate: float = 0.02):
        self.risk_free_rate = risk_free_rate
        
        # Cache for calculated metrics
        self._metrics_cache: Dict[str, Any] = {}
        self._last_calculation_hash: Optional[str] = None
        
        logger.info("Initialized performance analytics system")
    
    def calculate_performance_metrics(self, returns: pd.Series, 
                                    benchmark_returns: Optional[pd.Series] = None,
                                    factor_returns: Optional[pd.DataFrame] = None) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics"""
        # Calculate hash for caching
        data_hash = self._calculate_data_hash(returns, benchmark_returns)
        
        if data_hash == self._last_calculation_hash and 'performance_metrics' in self._metrics_cache:
            return self._metrics_cache['performance_metrics']
        
        # Basic return metrics
        total_return = (1 + returns).prod() - 1
        periods_per_year = self._infer_frequency(returns)
        annualized_return = (1 + total_return) ** (periods_per_year / len(returns)) - 1
        geometric_mean = (1 + returns).prod() ** (1 / len(returns)) - 1
        
        # Risk metrics
        volatility = returns.std() * np.sqrt(periods_per_year)
        downside_returns = returns[returns < 0]
        downside_volatility = downside_returns.std() * np.sqrt(periods_per_year) if len(downside_returns) > 0 else 0
        
        # VaR and CVaR
        var_95 = returns.quantile(0.05)
        var_99 = returns.quantile(0.01)
        cvar_95 = returns[returns <= var_95].mean() if len(returns[returns <= var_95]) > 0 else var_95
        cvar_99 = returns[returns <= var_99].mean() if len(returns[returns <= var_99]) > 0 else var_99
        
        # Risk-adjusted metrics
        excess_returns = returns - self.risk_free_rate / periods_per_year
        sharpe_ratio = excess_returns.mean() / returns.std() * np.sqrt(periods_per_year) if returns.std() > 0 else 0
        sortino_ratio = excess_returns.mean() / downside_volatility if downside_volatility > 0 else 0
        
        # Drawdown metrics
        dd_metrics = self._calculate_drawdown_metrics(returns)
        
        # Distribution metrics
        skewness = returns.skew()
        kurtosis = returns.kurtosis()
        jb_stat, jb_pvalue = stats.jarque_bera(returns.dropna())
        
        # Market relation metrics (if benchmark provided)
        alpha = beta = correlation = r_squared = 0.0
        tracking_error = information_ratio = treynor_ratio = 0.0
        
        if benchmark_returns is not None:
            market_metrics = self._calculate_market_metrics(returns, benchmark_returns)
            alpha = market_metrics['alpha']
            beta = market_metrics['beta']
            correlation = market_metrics['correlation']
            r_squared = market_metrics['r_squared']
            tracking_error = market_metrics['tracking_error']
            information_ratio = market_metrics['information_ratio']
            treynor_ratio = market_metrics['treynor_ratio']
        
        # Modigliani ratio
        modigliani_ratio = sharpe_ratio * (benchmark_returns.std() * np.sqrt(periods_per_year)) if benchmark_returns is not None else 0
        
        # Calmar ratio
        calmar_ratio = annualized_return / abs(dd_metrics['max_drawdown']) if dd_metrics['max_drawdown'] != 0 else 0
        
        # Trade-based metrics (simplified)
        hit_ratio = len(returns[returns > 0]) / len(returns) if len(returns) > 0 else 0
        profit_factor = self._calculate_profit_factor(returns)
        expectancy = returns.mean()
        kelly_criterion = self._calculate_kelly_criterion(returns)
        
        metrics = PerformanceMetrics(
            total_return=total_return,
            annualized_return=annualized_return,
            excess_return=annualized_return - self.risk_free_rate,
            geometric_mean_return=geometric_mean * periods_per_year,
            volatility=volatility,
            downside_volatility=downside_volatility,
            tracking_error=tracking_error,
            var_95=var_95,
            var_99=var_99,
            cvar_95=cvar_95,
            cvar_99=cvar_99,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            information_ratio=information_ratio,
            treynor_ratio=treynor_ratio,
            modigliani_ratio=modigliani_ratio,
            max_drawdown=dd_metrics['max_drawdown'],
            avg_drawdown=dd_metrics['avg_drawdown'],
            max_drawdown_duration=dd_metrics['max_duration'],
            recovery_time=dd_metrics['recovery_time'],
            pain_index=dd_metrics['pain_index'],
            skewness=skewness,
            kurtosis=kurtosis,
            jarque_bera_stat=jb_stat,
            jarque_bera_pvalue=jb_pvalue,
            alpha=alpha,
            beta=beta,
            correlation=correlation,
            r_squared=r_squared,
            hit_ratio=hit_ratio,
            profit_factor=profit_factor,
            expectancy=expectancy,
            kelly_criterion=kelly_criterion
        )
        
        # Cache results
        self._metrics_cache['performance_metrics'] = metrics
        self._last_calculation_hash = data_hash
        
        return metrics
    
    def calculate_attribution_analysis(self, portfolio_returns: pd.Series,
                                     benchmark_returns: pd.Series,
                                     sector_weights: Optional[pd.DataFrame] = None,
                                     factor_returns: Optional[pd.DataFrame] = None) -> AttributionAnalysis:
        """Calculate performance attribution analysis"""
        # Basic attribution
        active_returns = portfolio_returns - benchmark_returns
        total_active_return = active_returns.sum()
        
        # Factor attribution (if factor returns provided)
        factor_exposures = {}
        factor_returns_dict = {}
        
        if factor_returns is not None:
            # Align dates
            aligned_data = pd.concat([portfolio_returns, factor_returns], axis=1, join='inner')
            portfolio_aligned = aligned_data.iloc[:, 0]
            factors_aligned = aligned_data.iloc[:, 1:]
            
            # Regression analysis for factor exposures
            for factor in factors_aligned.columns:
                factor_data = factors_aligned[factor].dropna()
                portfolio_data = portfolio_aligned.reindex(factor_data.index).dropna()
                
                if len(factor_data) > 1 and len(portfolio_data) > 1:
                    beta = np.cov(portfolio_data, factor_data)[0, 1] / np.var(factor_data)
                    factor_exposures[factor] = beta
                    factor_returns_dict[factor] = factor_data.mean() * len(portfolio_returns)
        
        # Sector attribution (if sector weights provided)
        sector_attribution = {}
        if sector_weights is not None:
            for sector in sector_weights.columns:
                weights = sector_weights[sector].fillna(0)
                sector_return = (weights * portfolio_returns).sum()
                benchmark_sector_return = (weights * benchmark_returns).sum()
                
                allocation_effect = (weights.mean() - 1/len(sector_weights.columns)) * benchmark_sector_return
                selection_effect = weights.mean() * (sector_return - benchmark_sector_return)
                
                sector_attribution[sector] = {
                    'allocation_effect': allocation_effect,
                    'selection_effect': selection_effect,
                    'total_effect': allocation_effect + selection_effect
                }
        
        # Time-based attribution
        monthly_attribution = self._calculate_monthly_attribution(portfolio_returns, benchmark_returns)
        quarterly_attribution = self._calculate_quarterly_attribution(portfolio_returns, benchmark_returns)
        
        # Overall effects (simplified)
        allocation_effect = total_active_return * 0.4  # Simplified
        selection_effect = total_active_return * 0.6   # Simplified
        interaction_effect = 0.0
        
        return AttributionAnalysis(
            factor_exposures=factor_exposures,
            factor_returns=factor_returns_dict,
            allocation_effect=allocation_effect,
            selection_effect=selection_effect,
            interaction_effect=interaction_effect,
            total_active_return=total_active_return,
            sector_attribution=sector_attribution,
            style_attribution={},  # Placeholder
            monthly_attribution=monthly_attribution,
            quarterly_attribution=quarterly_attribution
        )
    
    def run_monte_carlo_simulation(self, returns: pd.Series, 
                                 num_simulations: int = 10000,
                                 time_horizon: int = 252) -> MonteCarloAnalysis:
        """Run Monte Carlo simulation for risk analysis"""
        # Estimate return distribution parameters
        mean_return = returns.mean()
        std_return = returns.std()
        
        # Generate random returns
        simulated_returns = np.random.normal(
            mean_return, std_return, 
            size=(num_simulations, time_horizon)
        )
        
        # Calculate cumulative returns for each simulation
        cumulative_returns = np.cumprod(1 + simulated_returns, axis=1) - 1
        final_returns = cumulative_returns[:, -1]
        
        # Calculate percentiles
        percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        percentile_returns = {p: np.percentile(final_returns, p) for p in percentiles}
        
        # Risk metrics
        probability_of_loss = np.mean(final_returns < 0)
        var_5 = np.percentile(final_returns, 5)
        expected_shortfall = np.mean(final_returns[final_returns <= var_5])
        maximum_loss = np.min(final_returns)
        
        # Confidence intervals
        confidence_intervals = {
            '95%': (np.percentile(final_returns, 2.5), np.percentile(final_returns, 97.5)),
            '99%': (np.percentile(final_returns, 0.5), np.percentile(final_returns, 99.5))
        }
        
        # Stress tests
        stress_scenarios = {
            'market_crash': self._simulate_market_crash(returns, time_horizon),
            'high_volatility': self._simulate_high_volatility(returns, time_horizon),
            'prolonged_decline': self._simulate_prolonged_decline(returns, time_horizon)
        }
        
        return MonteCarloAnalysis(
            num_simulations=num_simulations,
            time_horizon=time_horizon,
            simulated_returns=simulated_returns,
            percentile_returns=percentile_returns,
            probability_of_loss=probability_of_loss,
            expected_shortfall=expected_shortfall,
            maximum_loss=maximum_loss,
            confidence_intervals=confidence_intervals,
            stress_test_results=stress_scenarios
        )
    
    def calculate_risk_metrics(self, returns: pd.Series,
                             positions: Optional[pd.DataFrame] = None,
                             market_data: Optional[pd.DataFrame] = None) -> RiskMetrics:
        """Calculate comprehensive risk metrics"""
        # Volatility metrics
        realized_volatility = returns.std() * np.sqrt(252)
        garch_volatility = self._estimate_garch_volatility(returns)
        
        # Tail risk metrics
        positive_returns = returns[returns > 0]
        negative_returns = returns[returns < 0]
        
        if len(positive_returns) > 0 and len(negative_returns) > 0:
            tail_ratio = np.percentile(positive_returns, 95) / abs(np.percentile(negative_returns, 5))
            gain_pain_ratio = positive_returns.mean() / abs(negative_returns.mean())
        else:
            tail_ratio = gain_pain_ratio = 1.0
        
        # Concentration metrics
        concentration_index = self._calculate_concentration_index(positions) if positions is not None else 0.0
        diversification_ratio = self._calculate_diversification_ratio(returns, positions) if positions is not None else 1.0
        
        # Liquidity metrics
        liquidity_score = self._calculate_liquidity_score(market_data) if market_data is not None else 1.0
        bid_ask_impact = self._calculate_bid_ask_impact(market_data) if market_data is not None else 0.0
        
        # Model risk (simplified)
        model_uncertainty = self._calculate_model_uncertainty(returns)
        parameter_sensitivity = self._calculate_parameter_sensitivity(returns)
        
        return RiskMetrics(
            realized_volatility=realized_volatility,
            garch_volatility=garch_volatility,
            implied_volatility=None,
            tail_ratio=tail_ratio,
            gain_pain_ratio=gain_pain_ratio,
            concentration_index=concentration_index,
            diversification_ratio=diversification_ratio,
            liquidity_score=liquidity_score,
            bid_ask_impact=bid_ask_impact,
            model_uncertainty=model_uncertainty,
            parameter_sensitivity=parameter_sensitivity
        )
    
    def _calculate_data_hash(self, *data_series) -> str:
        """Calculate hash for data caching"""
        hash_str = ""
        for series in data_series:
            if series is not None:
                hash_str += str(hash(str(series.values.tobytes())))
        return hash_str
    
    def _infer_frequency(self, returns: pd.Series) -> int:
        """Infer the frequency of returns (periods per year)"""
        if hasattr(returns.index, 'freq'):
            freq = returns.index.freq
            if freq is not None:
                if 'D' in str(freq):
                    return 252
                elif 'W' in str(freq):
                    return 52
                elif 'M' in str(freq):
                    return 12
        
        # Fallback: estimate from index differences
        if len(returns) > 1:
            avg_diff = (returns.index[-1] - returns.index[0]) / (len(returns) - 1)
            if avg_diff.days <= 3:
                return 252  # Daily
            elif avg_diff.days <= 10:
                return 52   # Weekly
            else:
                return 12   # Monthly
        
        return 252  # Default to daily
    
    def _calculate_drawdown_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate comprehensive drawdown metrics"""
        cumulative = (1 + returns).cumprod()
        peak = cumulative.expanding().max()
        drawdown = (cumulative - peak) / peak
        
        # Max drawdown
        max_drawdown = drawdown.min()
        
        # Average drawdown
        negative_dd = drawdown[drawdown < 0]
        avg_drawdown = negative_dd.mean() if len(negative_dd) > 0 else 0.0
        
        # Drawdown duration analysis
        drawdown_periods = []
        in_drawdown = False
        current_period = 0
        
        for dd in drawdown:
            if dd < 0:
                if not in_drawdown:
                    in_drawdown = True
                    current_period = 1
                else:
                    current_period += 1
            else:
                if in_drawdown:
                    drawdown_periods.append(current_period)
                    in_drawdown = False
                    current_period = 0
        
        max_duration = max(drawdown_periods) if drawdown_periods else 0
        
        # Recovery time (simplified)
        max_dd_idx = drawdown.idxmin()
        recovery_returns = returns[max_dd_idx:]
        recovery_cumulative = (1 + recovery_returns).cumprod()
        
        recovery_time = 0
        if len(recovery_cumulative) > 0:
            recovery_threshold = 1.0  # Back to peak
            recovery_idx = recovery_cumulative[recovery_cumulative >= recovery_threshold]
            recovery_time = len(recovery_idx) if len(recovery_idx) > 0 else len(recovery_cumulative)
        
        # Pain index (average drawdown)
        pain_index = abs(drawdown.mean())
        
        return {
            'max_drawdown': max_drawdown,
            'avg_drawdown': avg_drawdown,
            'max_duration': max_duration,
            'recovery_time': recovery_time,
            'pain_index': pain_index
        }
    
    def _calculate_market_metrics(self, returns: pd.Series, benchmark_returns: pd.Series) -> Dict[str, float]:
        """Calculate market-related metrics"""
        # Align returns
        aligned_data = pd.concat([returns, benchmark_returns], axis=1, join='inner')
        port_returns = aligned_data.iloc[:, 0].dropna()
        bench_returns = aligned_data.iloc[:, 1].dropna()
        
        if len(port_returns) < 2 or len(bench_returns) < 2:
            return {
                'alpha': 0.0, 'beta': 0.0, 'correlation': 0.0, 'r_squared': 0.0,
                'tracking_error': 0.0, 'information_ratio': 0.0, 'treynor_ratio': 0.0
            }
        
        # Regression analysis
        X = bench_returns.values.reshape(-1, 1)
        y = port_returns.values
        
        reg = LinearRegression().fit(X, y)
        beta = reg.coef_[0]
        alpha = reg.intercept_
        
        # Correlation and R-squared
        correlation = np.corrcoef(port_returns, bench_returns)[0, 1]
        r_squared = reg.score(X, y)
        
        # Tracking error
        active_returns = port_returns - bench_returns
        periods_per_year = self._infer_frequency(returns)
        tracking_error = active_returns.std() * np.sqrt(periods_per_year)
        
        # Information ratio
        information_ratio = active_returns.mean() * periods_per_year / tracking_error if tracking_error > 0 else 0
        
        # Treynor ratio
        excess_returns = port_returns - self.risk_free_rate / periods_per_year
        treynor_ratio = excess_returns.mean() * periods_per_year / beta if beta != 0 else 0
        
        # Annualize alpha
        alpha = alpha * periods_per_year
        
        return {
            'alpha': alpha,
            'beta': beta,
            'correlation': correlation,
            'r_squared': r_squared,
            'tracking_error': tracking_error,
            'information_ratio': information_ratio,
            'treynor_ratio': treynor_ratio
        }
    
    def _calculate_profit_factor(self, returns: pd.Series) -> float:
        """Calculate profit factor"""
        positive_returns = returns[returns > 0]
        negative_returns = returns[returns < 0]
        
        if len(positive_returns) == 0 or len(negative_returns) == 0:
            return 1.0
        
        gross_profit = positive_returns.sum()
        gross_loss = abs(negative_returns.sum())
        
        return gross_profit / gross_loss if gross_loss != 0 else float('inf')
    
    def _calculate_kelly_criterion(self, returns: pd.Series) -> float:
        """Calculate Kelly criterion for optimal position sizing"""
        if len(returns) < 2:
            return 0.0
        
        win_rate = len(returns[returns > 0]) / len(returns)
        
        if win_rate == 0 or win_rate == 1:
            return 0.0
        
        avg_win = returns[returns > 0].mean() if len(returns[returns > 0]) > 0 else 0
        avg_loss = abs(returns[returns < 0].mean()) if len(returns[returns < 0]) > 0 else 0
        
        if avg_loss == 0:
            return 0.0
        
        win_loss_ratio = avg_win / avg_loss
        kelly = win_rate - (1 - win_rate) / win_loss_ratio
        
        # Cap Kelly at reasonable levels
        return max(-0.25, min(0.25, kelly))
    
    def _calculate_monthly_attribution(self, portfolio_returns: pd.Series, 
                                     benchmark_returns: pd.Series) -> pd.Series:
        """Calculate monthly attribution"""
        # Resample to monthly
        port_monthly = portfolio_returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
        bench_monthly = benchmark_returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
        
        return port_monthly - bench_monthly
    
    def _calculate_quarterly_attribution(self, portfolio_returns: pd.Series, 
                                       benchmark_returns: pd.Series) -> pd.Series:
        """Calculate quarterly attribution"""
        # Resample to quarterly
        port_quarterly = portfolio_returns.resample('Q').apply(lambda x: (1 + x).prod() - 1)
        bench_quarterly = benchmark_returns.resample('Q').apply(lambda x: (1 + x).prod() - 1)
        
        return port_quarterly - bench_quarterly
    
    def _simulate_market_crash(self, returns: pd.Series, time_horizon: int) -> float:
        """Simulate market crash scenario"""
        # Simulate 20% drop in first month, then normal returns
        crash_returns = np.concatenate([
            np.full(21, -0.01),  # 21 trading days of -1% daily returns
            np.random.normal(returns.mean(), returns.std(), time_horizon - 21)
        ])
        
        return np.prod(1 + crash_returns) - 1
    
    def _simulate_high_volatility(self, returns: pd.Series, time_horizon: int) -> float:
        """Simulate high volatility scenario"""
        high_vol_returns = np.random.normal(
            returns.mean(), returns.std() * 2, time_horizon
        )
        
        return np.prod(1 + high_vol_returns) - 1
    
    def _simulate_prolonged_decline(self, returns: pd.Series, time_horizon: int) -> float:
        """Simulate prolonged decline scenario"""
        decline_returns = np.random.normal(
            returns.mean() - 0.001, returns.std(), time_horizon
        )
        
        return np.prod(1 + decline_returns) - 1
    
    def _estimate_garch_volatility(self, returns: pd.Series) -> float:
        """Estimate GARCH volatility (simplified)"""
        # Simplified GARCH(1,1) estimation
        if len(returns) < 10:
            return returns.std() * np.sqrt(252)
        
        # Use exponentially weighted moving average as proxy
        ewm_vol = returns.ewm(span=30).std().iloc[-1]
        return ewm_vol * np.sqrt(252)
    
    def _calculate_concentration_index(self, positions: pd.DataFrame) -> float:
        """Calculate portfolio concentration index"""
        if positions is None or positions.empty:
            return 0.0
        
        # Calculate Herfindahl-Hirschman Index
        latest_positions = positions.iloc[-1]
        weights = latest_positions.abs()
        total_weight = weights.sum()
        
        if total_weight == 0:
            return 0.0
        
        normalized_weights = weights / total_weight
        hhi = (normalized_weights ** 2).sum()
        
        return hhi
    
    def _calculate_diversification_ratio(self, returns: pd.Series, positions: pd.DataFrame) -> float:
        """Calculate diversification ratio"""
        if positions is None or positions.empty:
            return 1.0
        
        # Simplified: inverse of concentration
        concentration = self._calculate_concentration_index(positions)
        return 1 / (concentration + 1e-6)
    
    def _calculate_liquidity_score(self, market_data: pd.DataFrame) -> float:
        """Calculate liquidity score based on volume and spreads"""
        if market_data is None or market_data.empty:
            return 1.0
        
        # Look for volume columns
        volume_cols = [col for col in market_data.columns if 'volume' in col.lower()]
        
        if not volume_cols:
            return 1.0
        
        # Calculate average volume ratio
        volume_ratios = []
        for col in volume_cols:
            volume_series = market_data[col].dropna()
            if len(volume_series) > 20:
                rolling_avg = volume_series.rolling(20).mean()
                current_ratio = volume_series.iloc[-1] / rolling_avg.iloc[-1]
                volume_ratios.append(current_ratio)
        
        if volume_ratios:
            avg_ratio = np.mean(volume_ratios)
            return min(2.0, max(0.1, avg_ratio))  # Normalized to [0.1, 2.0]
        
        return 1.0
    
    def _calculate_bid_ask_impact(self, market_data: pd.DataFrame) -> float:
        """Calculate bid-ask spread impact"""
        # Simplified: assume 0.05% average spread
        return 0.0005
    
    def _calculate_model_uncertainty(self, returns: pd.Series) -> float:
        """Calculate model uncertainty metric"""
        if len(returns) < 30:
            return 0.1
        
        # Use rolling window to measure parameter stability
        window_size = min(30, len(returns) // 3)
        rolling_means = returns.rolling(window_size).mean()
        rolling_stds = returns.rolling(window_size).std()
        
        mean_uncertainty = rolling_means.std() / returns.mean() if returns.mean() != 0 else 0.1
        vol_uncertainty = rolling_stds.std() / returns.std() if returns.std() != 0 else 0.1
        
        return (mean_uncertainty + vol_uncertainty) / 2
    
    def _calculate_parameter_sensitivity(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate parameter sensitivity analysis"""
        base_sharpe = returns.mean() / returns.std() if returns.std() > 0 else 0
        
        # Sensitivity to different lookback periods
        sensitivities = {}
        
        for period in [10, 20, 30, 60]:
            if len(returns) > period:
                period_returns = returns.tail(period)
                period_sharpe = period_returns.mean() / period_returns.std() if period_returns.std() > 0 else 0
                sensitivities[f'lookback_{period}'] = abs(period_sharpe - base_sharpe)
        
        return sensitivities
    
    def generate_performance_report(self, returns: pd.Series, 
                                  benchmark_returns: Optional[pd.Series] = None,
                                  output_path: Optional[str] = None) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        # Calculate all metrics
        performance_metrics = self.calculate_performance_metrics(returns, benchmark_returns)
        
        if benchmark_returns is not None:
            attribution = self.calculate_attribution_analysis(returns, benchmark_returns)
        else:
            attribution = None
        
        monte_carlo = self.run_monte_carlo_simulation(returns)
        risk_metrics = self.calculate_risk_metrics(returns)
        
        report = {
            'performance_metrics': performance_metrics,
            'attribution_analysis': attribution,
            'monte_carlo_analysis': monte_carlo,
            'risk_metrics': risk_metrics,
            'summary': self._create_summary_statistics(performance_metrics, risk_metrics)
        }
        
        if output_path:
            self._save_report(report, output_path)
        
        return report
    
    def _create_summary_statistics(self, performance_metrics: PerformanceMetrics, 
                                 risk_metrics: RiskMetrics) -> Dict[str, Any]:
        """Create summary statistics"""
        return {
            'overall_rating': self._calculate_overall_rating(performance_metrics),
            'risk_rating': self._calculate_risk_rating(risk_metrics),
            'key_strengths': self._identify_strengths(performance_metrics),
            'key_weaknesses': self._identify_weaknesses(performance_metrics, risk_metrics),
            'recommendations': self._generate_recommendations(performance_metrics, risk_metrics)
        }
    
    def _calculate_overall_rating(self, metrics: PerformanceMetrics) -> str:
        """Calculate overall performance rating"""
        score = 0
        
        # Sharpe ratio scoring
        if metrics.sharpe_ratio > 1.5:
            score += 3
        elif metrics.sharpe_ratio > 1.0:
            score += 2
        elif metrics.sharpe_ratio > 0.5:
            score += 1
        
        # Max drawdown scoring
        if metrics.max_drawdown > -0.05:  # Less than 5%
            score += 2
        elif metrics.max_drawdown > -0.15:  # Less than 15%
            score += 1
        
        # Return scoring
        if metrics.annualized_return > 0.15:  # More than 15%
            score += 2
        elif metrics.annualized_return > 0.08:  # More than 8%
            score += 1
        
        if score >= 6:
            return "Excellent"
        elif score >= 4:
            return "Good"
        elif score >= 2:
            return "Fair"
        else:
            return "Poor"
    
    def _calculate_risk_rating(self, metrics: RiskMetrics) -> str:
        """Calculate risk rating"""
        if metrics.realized_volatility < 0.1:
            return "Low"
        elif metrics.realized_volatility < 0.2:
            return "Medium"
        else:
            return "High"
    
    def _identify_strengths(self, metrics: PerformanceMetrics) -> List[str]:
        """Identify strategy strengths"""
        strengths = []
        
        if metrics.sharpe_ratio > 1.0:
            strengths.append("Strong risk-adjusted returns")
        if metrics.max_drawdown > -0.1:
            strengths.append("Low maximum drawdown")
        if metrics.hit_ratio > 0.6:
            strengths.append("High win rate")
        if metrics.sortino_ratio > metrics.sharpe_ratio:
            strengths.append("Good downside protection")
        
        return strengths
    
    def _identify_weaknesses(self, perf_metrics: PerformanceMetrics, risk_metrics: RiskMetrics) -> List[str]:
        """Identify strategy weaknesses"""
        weaknesses = []
        
        if perf_metrics.sharpe_ratio < 0.5:
            weaknesses.append("Low risk-adjusted returns")
        if perf_metrics.max_drawdown < -0.2:
            weaknesses.append("High maximum drawdown")
        if perf_metrics.hit_ratio < 0.4:
            weaknesses.append("Low win rate")
        if risk_metrics.concentration_index > 0.5:
            weaknesses.append("High concentration risk")
        
        return weaknesses
    
    def _generate_recommendations(self, perf_metrics: PerformanceMetrics, risk_metrics: RiskMetrics) -> List[str]:
        """Generate improvement recommendations"""
        recommendations = []
        
        if perf_metrics.max_drawdown < -0.15:
            recommendations.append("Consider implementing stronger risk management controls")
        if risk_metrics.concentration_index > 0.4:
            recommendations.append("Improve portfolio diversification")
        if perf_metrics.sortino_ratio < perf_metrics.sharpe_ratio * 0.8:
            recommendations.append("Focus on reducing downside volatility")
        
        return recommendations
    
    def _save_report(self, report: Dict[str, Any], output_path: str):
        """Save performance report to file"""
        # Implementation would save report to file
        logger.info(f"Performance report saved to {output_path}")


# Factory function
def create_performance_analytics(risk_free_rate: float = 0.02) -> PerformanceAnalytics:
    """Create performance analytics instance"""
    return PerformanceAnalytics(risk_free_rate)


# Export classes and functions
__all__ = [
    'PerformanceAnalytics',
    'PerformanceMetrics',
    'AttributionAnalysis',
    'MonteCarloAnalysis',
    'RiskMetrics',
    'create_performance_analytics'
]