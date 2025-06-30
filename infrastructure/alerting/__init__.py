"""
Alerting System
Email, Slack, and other notification channels for data quality alerts
"""

from .alert_manager import AlertManager

__all__ = ['AlertManager']