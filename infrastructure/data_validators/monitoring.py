"""
Data Quality Monitoring and Dashboard System
Real-time monitoring, alerting, and visualization of data quality metrics
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum
import pandas as pd

from .quality_checker import DataQualityChecker, QualityReport
from .market_validator import MarketDataValidator, MarketValidationResult
from ..data_collectors.models import ExchangeType
from utils.logger import get_logger

logger = get_logger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class QualityMetrics:
    """Data quality metrics for monitoring"""
    timestamp: datetime
    dataset_name: str
    total_records: int
    valid_records: int
    quality_score: float
    missing_value_rate: float
    outlier_rate: float
    duplicate_rate: float
    consistency_rate: float
    issues_count: int
    critical_issues_count: int


@dataclass
class AlertRule:
    """Alert rule configuration"""
    name: str
    condition: str  # Python expression
    severity: AlertSeverity
    message_template: str
    cooldown_minutes: int = 60
    enabled: bool = True


@dataclass
class Alert:
    """Data quality alert"""
    id: str
    rule_name: str
    severity: AlertSeverity
    message: str
    timestamp: datetime
    dataset_name: str
    metrics: Dict[str, Any]
    acknowledged: bool = False


class DataQualityMonitor:
    """Real-time data quality monitoring system"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.quality_checker = DataQualityChecker()
        self.market_validator = MarketDataValidator()
        
        # Monitoring configuration
        self.monitoring_interval = self.config.get('monitoring_interval', 300)  # 5 minutes
        self.retention_days = self.config.get('retention_days', 30)
        self.alert_channels = self.config.get('alert_channels', ['console'])
        
        # Storage for metrics and alerts
        self.metrics_history: List[QualityMetrics] = []
        self.alerts_history: List[Alert] = []
        self.last_alert_times: Dict[str, datetime] = {}
        
        # Default alert rules
        self.alert_rules = self._create_default_alert_rules()
        
        # Monitoring state
        self.is_monitoring = False
        self.monitoring_tasks = []
    
    def _create_default_alert_rules(self) -> List[AlertRule]:
        """Create default alert rules for data quality monitoring"""
        return [
            AlertRule(
                name="quality_score_low",
                condition="quality_score < 0.7",
                severity=AlertSeverity.WARNING,
                message_template="Data quality score is low: {quality_score:.3f} for dataset {dataset_name}",
                cooldown_minutes=30
            ),
            AlertRule(
                name="quality_score_critical",
                condition="quality_score < 0.5",
                severity=AlertSeverity.CRITICAL,
                message_template="Data quality score is critically low: {quality_score:.3f} for dataset {dataset_name}",
                cooldown_minutes=15
            ),
            AlertRule(
                name="high_missing_rate",
                condition="missing_value_rate > 0.2",
                severity=AlertSeverity.WARNING,
                message_template="High missing value rate: {missing_value_rate:.1%} in dataset {dataset_name}",
                cooldown_minutes=60
            ),
            AlertRule(
                name="critical_missing_rate",
                condition="missing_value_rate > 0.5",
                severity=AlertSeverity.CRITICAL,
                message_template="Critical missing value rate: {missing_value_rate:.1%} in dataset {dataset_name}",
                cooldown_minutes=30
            ),
            AlertRule(
                name="high_outlier_rate",
                condition="outlier_rate > 0.1",
                severity=AlertSeverity.WARNING,
                message_template="High outlier rate: {outlier_rate:.1%} in dataset {dataset_name}",
                cooldown_minutes=45
            ),
            AlertRule(
                name="excessive_duplicates",
                condition="duplicate_rate > 0.15",
                severity=AlertSeverity.ERROR,
                message_template="Excessive duplicate rate: {duplicate_rate:.1%} in dataset {dataset_name}",
                cooldown_minutes=60
            ),
            AlertRule(
                name="critical_issues_detected",
                condition="critical_issues_count > 0",
                severity=AlertSeverity.CRITICAL,
                message_template="Critical data issues detected: {critical_issues_count} issues in dataset {dataset_name}",
                cooldown_minutes=10
            ),
            AlertRule(
                name="no_data_received",
                condition="total_records == 0",
                severity=AlertSeverity.ERROR,
                message_template="No data received for dataset {dataset_name}",
                cooldown_minutes=30
            )
        ]
    
    async def start_monitoring(self):
        """Start continuous data quality monitoring"""
        if self.is_monitoring:
            logger.warning("Monitoring is already running")
            return
        
        self.is_monitoring = True
        logger.info("Starting data quality monitoring")
        
        # Start monitoring tasks
        self.monitoring_tasks = [
            asyncio.create_task(self._metrics_collection_loop()),
            asyncio.create_task(self._alert_processing_loop()),
            asyncio.create_task(self._cleanup_loop())
        ]
        
        try:
            await asyncio.gather(*self.monitoring_tasks, return_exceptions=True)
        except Exception as e:
            logger.error(f"Monitoring error: {e}")
        finally:
            self.is_monitoring = False
    
    async def stop_monitoring(self):
        """Stop data quality monitoring"""
        logger.info("Stopping data quality monitoring")
        self.is_monitoring = False
        
        for task in self.monitoring_tasks:
            if not task.done():
                task.cancel()
        
        await asyncio.gather(*self.monitoring_tasks, return_exceptions=True)
        self.monitoring_tasks.clear()
    
    async def _metrics_collection_loop(self):
        """Continuous metrics collection loop"""
        while self.is_monitoring:
            try:
                await self._collect_quality_metrics()
                await asyncio.sleep(self.monitoring_interval)
            except Exception as e:
                logger.error(f"Metrics collection error: {e}")
                await asyncio.sleep(60)  # Wait before retry
    
    async def _alert_processing_loop(self):
        """Continuous alert processing loop"""
        while self.is_monitoring:
            try:
                await self._process_alerts()
                await asyncio.sleep(30)  # Check alerts every 30 seconds
            except Exception as e:
                logger.error(f"Alert processing error: {e}")
                await asyncio.sleep(60)
    
    async def _cleanup_loop(self):
        """Cleanup old metrics and alerts"""
        while self.is_monitoring:
            try:
                await self._cleanup_old_data()
                await asyncio.sleep(3600)  # Cleanup every hour
            except Exception as e:
                logger.error(f"Cleanup error: {e}")
                await asyncio.sleep(3600)
    
    async def _collect_quality_metrics(self):
        """Collect quality metrics from various data sources"""
        # This would typically integrate with your data collection system
        # For now, we'll simulate with example logic
        logger.debug("Collecting quality metrics...")
        
        # Example: Monitor recent data files or database tables
        # You would implement actual data source monitoring here
        
    def validate_dataset(self, df: pd.DataFrame, dataset_name: str, 
                        exchange: Optional[ExchangeType] = None) -> QualityMetrics:
        """Validate a dataset and return quality metrics"""
        logger.info(f"Validating dataset: {dataset_name}")
        
        # Perform quality validation
        quality_report = self.quality_checker.validate_dataframe(df, dataset_name)
        
        # Calculate metrics
        metrics = self._extract_metrics_from_report(quality_report, dataset_name)
        
        # Store metrics
        self.metrics_history.append(metrics)
        
        # Check for alerts
        self._check_alert_rules(metrics)
        
        logger.info(f"Dataset validation completed. Quality score: {metrics.quality_score:.3f}")
        return metrics
    
    def validate_market_data(self, df: pd.DataFrame, symbol: str, 
                           exchange: ExchangeType) -> MarketValidationResult:
        """Validate market data and return validation result"""
        logger.info(f"Validating market data: {symbol} on {exchange.value}")
        
        # Perform market validation
        validation_result = self.market_validator.validate_market_data(df, symbol, exchange)
        
        # Convert to quality metrics for monitoring
        dataset_name = f"{symbol}_{exchange.value}"
        metrics = QualityMetrics(
            timestamp=datetime.now(),
            dataset_name=dataset_name,
            total_records=len(df),
            valid_records=len(df) - len([i for i in validation_result.issues if i.severity in ['high', 'critical']]),
            quality_score=validation_result.market_health_score,
            missing_value_rate=0.0,  # Market validator doesn't track this directly
            outlier_rate=len([i for i in validation_result.issues if 'outlier' in i.description.lower()]) / len(df) if len(df) > 0 else 0.0,
            duplicate_rate=0.0,
            consistency_rate=1.0 - (len([i for i in validation_result.issues if 'inconsistent' in i.description.lower()]) / len(df)) if len(df) > 0 else 1.0,
            issues_count=len(validation_result.issues),
            critical_issues_count=len([i for i in validation_result.issues if i.severity == 'critical'])
        )
        
        # Store metrics and check alerts
        self.metrics_history.append(metrics)
        self._check_alert_rules(metrics)
        
        return validation_result
    
    def _extract_metrics_from_report(self, report: QualityReport, dataset_name: str) -> QualityMetrics:
        """Extract monitoring metrics from quality report"""
        total_records = report.total_records
        
        # Calculate rates
        missing_rate = report.issues_by_type.get('missing_value', 0) / total_records if total_records > 0 else 0.0
        outlier_rate = report.issues_by_type.get('outlier', 0) / total_records if total_records > 0 else 0.0
        duplicate_rate = report.issues_by_type.get('duplicate', 0) / total_records if total_records > 0 else 0.0
        
        consistency_issues = report.issues_by_type.get('inconsistent', 0)
        consistency_rate = 1.0 - (consistency_issues / total_records) if total_records > 0 else 1.0
        
        return QualityMetrics(
            timestamp=report.validation_time,
            dataset_name=dataset_name,
            total_records=total_records,
            valid_records=report.valid_records,
            quality_score=report.quality_score,
            missing_value_rate=missing_rate,
            outlier_rate=outlier_rate,
            duplicate_rate=duplicate_rate,
            consistency_rate=consistency_rate,
            issues_count=sum(report.issues_by_type.values()),
            critical_issues_count=report.issues_by_severity.get('critical', 0)
        )
    
    def _check_alert_rules(self, metrics: QualityMetrics):
        """Check metrics against alert rules"""
        metrics_dict = asdict(metrics)
        
        for rule in self.alert_rules:
            if not rule.enabled:
                continue
            
            try:
                # Check if rule condition is met
                if eval(rule.condition, {"__builtins__": {}}, metrics_dict):
                    # Check cooldown period
                    last_alert_time = self.last_alert_times.get(rule.name)
                    if (last_alert_time and 
                        datetime.now() - last_alert_time < timedelta(minutes=rule.cooldown_minutes)):
                        continue
                    
                    # Create alert
                    alert = Alert(
                        id=f"{rule.name}_{datetime.now().timestamp()}",
                        rule_name=rule.name,
                        severity=rule.severity,
                        message=rule.message_template.format(**metrics_dict),
                        timestamp=datetime.now(),
                        dataset_name=metrics.dataset_name,
                        metrics=metrics_dict
                    )
                    
                    # Store alert and update last alert time
                    self.alerts_history.append(alert)
                    self.last_alert_times[rule.name] = datetime.now()
                    
                    # Send alert
                    self._send_alert(alert)
                    
            except Exception as e:
                logger.error(f"Error evaluating alert rule {rule.name}: {e}")
    
    def _send_alert(self, alert: Alert):
        """Send alert through configured channels"""
        for channel in self.alert_channels:
            try:
                if channel == 'console':
                    self._send_console_alert(alert)
                elif channel == 'email':
                    self._send_email_alert(alert)
                elif channel == 'slack':
                    self._send_slack_alert(alert)
                # Add more channels as needed
            except Exception as e:
                logger.error(f"Failed to send alert via {channel}: {e}")
    
    def _send_console_alert(self, alert: Alert):
        """Send alert to console/logs"""
        severity_colors = {
            AlertSeverity.INFO: '\033[94m',      # Blue
            AlertSeverity.WARNING: '\033[93m',   # Yellow
            AlertSeverity.ERROR: '\033[91m',     # Red
            AlertSeverity.CRITICAL: '\033[95m'   # Magenta
        }
        reset_color = '\033[0m'
        
        color = severity_colors.get(alert.severity, '')
        print(f"{color}[{alert.severity.value.upper()}] {alert.timestamp}: {alert.message}{reset_color}")
        logger.warning(f"ALERT: {alert.message}")
    
    def _send_email_alert(self, alert: Alert):
        """Send alert via email (placeholder)"""
        # Implement email sending logic
        logger.info(f"Email alert would be sent: {alert.message}")
    
    def _send_slack_alert(self, alert: Alert):
        """Send alert via Slack (placeholder)"""
        # Implement Slack webhook logic
        logger.info(f"Slack alert would be sent: {alert.message}")
    
    async def _process_alerts(self):
        """Process pending alerts"""
        # Get recent unacknowledged alerts
        recent_alerts = [
            alert for alert in self.alerts_history
            if not alert.acknowledged and 
            datetime.now() - alert.timestamp < timedelta(hours=24)
        ]
        
        # Group by severity for summary
        if recent_alerts:
            critical_count = len([a for a in recent_alerts if a.severity == AlertSeverity.CRITICAL])
            error_count = len([a for a in recent_alerts if a.severity == AlertSeverity.ERROR])
            warning_count = len([a for a in recent_alerts if a.severity == AlertSeverity.WARNING])
            
            if critical_count > 0:
                logger.error(f"Data quality monitoring: {critical_count} critical alerts active")
    
    async def _cleanup_old_data(self):
        """Clean up old metrics and alerts"""
        cutoff_time = datetime.now() - timedelta(days=self.retention_days)
        
        # Clean up old metrics
        original_metrics_count = len(self.metrics_history)
        self.metrics_history = [
            m for m in self.metrics_history if m.timestamp > cutoff_time
        ]
        
        # Clean up old alerts
        original_alerts_count = len(self.alerts_history)
        self.alerts_history = [
            a for a in self.alerts_history if a.timestamp > cutoff_time
        ]
        
        cleaned_metrics = original_metrics_count - len(self.metrics_history)
        cleaned_alerts = original_alerts_count - len(self.alerts_history)
        
        if cleaned_metrics > 0 or cleaned_alerts > 0:
            logger.info(f"Cleaned up {cleaned_metrics} old metrics and {cleaned_alerts} old alerts")
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get data for monitoring dashboard"""
        # Recent metrics (last 24 hours)
        recent_cutoff = datetime.now() - timedelta(hours=24)
        recent_metrics = [m for m in self.metrics_history if m.timestamp > recent_cutoff]
        
        # Recent alerts
        recent_alerts = [a for a in self.alerts_history if a.timestamp > recent_cutoff]
        
        # Calculate summary statistics
        if recent_metrics:
            avg_quality_score = sum(m.quality_score for m in recent_metrics) / len(recent_metrics)
            datasets_monitored = len(set(m.dataset_name for m in recent_metrics))
            total_records_processed = sum(m.total_records for m in recent_metrics)
        else:
            avg_quality_score = 0.0
            datasets_monitored = 0
            total_records_processed = 0
        
        # Alert statistics
        alert_counts = {
            'critical': len([a for a in recent_alerts if a.severity == AlertSeverity.CRITICAL]),
            'error': len([a for a in recent_alerts if a.severity == AlertSeverity.ERROR]),
            'warning': len([a for a in recent_alerts if a.severity == AlertSeverity.WARNING]),
            'info': len([a for a in recent_alerts if a.severity == AlertSeverity.INFO])
        }
        
        # Quality trends (last 7 days)
        week_cutoff = datetime.now() - timedelta(days=7)
        week_metrics = [m for m in self.metrics_history if m.timestamp > week_cutoff]
        
        quality_trend = []
        if week_metrics:
            # Group by day
            daily_metrics = {}
            for metric in week_metrics:
                day = metric.timestamp.date()
                if day not in daily_metrics:
                    daily_metrics[day] = []
                daily_metrics[day].append(metric)
            
            for day, day_metrics in sorted(daily_metrics.items()):
                avg_score = sum(m.quality_score for m in day_metrics) / len(day_metrics)
                quality_trend.append({
                    'date': day.isoformat(),
                    'quality_score': avg_score,
                    'datasets': len(set(m.dataset_name for m in day_metrics))
                })
        
        return {
            'summary': {
                'monitoring_status': 'active' if self.is_monitoring else 'inactive',
                'avg_quality_score': round(avg_quality_score, 3),
                'datasets_monitored': datasets_monitored,
                'total_records_processed': total_records_processed,
                'last_update': datetime.now().isoformat()
            },
            'alerts': {
                'counts': alert_counts,
                'recent_alerts': [
                    {
                        'id': a.id,
                        'severity': a.severity.value,
                        'message': a.message,
                        'timestamp': a.timestamp.isoformat(),
                        'dataset': a.dataset_name,
                        'acknowledged': a.acknowledged
                    }
                    for a in sorted(recent_alerts, key=lambda x: x.timestamp, reverse=True)[:10]
                ]
            },
            'quality_trends': quality_trend,
            'dataset_metrics': [
                {
                    'dataset_name': m.dataset_name,
                    'timestamp': m.timestamp.isoformat(),
                    'quality_score': m.quality_score,
                    'total_records': m.total_records,
                    'issues_count': m.issues_count,
                    'critical_issues': m.critical_issues_count
                }
                for m in sorted(recent_metrics, key=lambda x: x.timestamp, reverse=True)[:20]
            ]
        }
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert"""
        for alert in self.alerts_history:
            if alert.id == alert_id:
                alert.acknowledged = True
                logger.info(f"Alert {alert_id} acknowledged")
                return True
        return False
    
    def add_alert_rule(self, rule: AlertRule):
        """Add a custom alert rule"""
        self.alert_rules.append(rule)
        logger.info(f"Added alert rule: {rule.name}")
    
    def disable_alert_rule(self, rule_name: str) -> bool:
        """Disable an alert rule"""
        for rule in self.alert_rules:
            if rule.name == rule_name:
                rule.enabled = False
                logger.info(f"Disabled alert rule: {rule_name}")
                return True
        return False
    
    def get_metrics_history(self, dataset_name: Optional[str] = None, 
                          hours_back: int = 24) -> List[QualityMetrics]:
        """Get metrics history for analysis"""
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        
        filtered_metrics = [
            m for m in self.metrics_history 
            if m.timestamp > cutoff_time and (dataset_name is None or m.dataset_name == dataset_name)
        ]
        
        return sorted(filtered_metrics, key=lambda x: x.timestamp, reverse=True)