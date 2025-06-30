"""
Alert Manager
Central management of all alerting channels and configurations
"""

import asyncio
import smtplib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import aiohttp
import json

from ..data_validators.monitoring import Alert, AlertSeverity
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class EmailConfig:
    """Email configuration"""
    smtp_server: str
    smtp_port: int
    username: str
    password: str
    from_email: str
    to_emails: List[str]
    use_tls: bool = True


@dataclass
class SlackConfig:
    """Slack configuration"""
    webhook_url: str
    channel: str
    username: str = "Data Quality Bot"
    icon_emoji: str = ":warning:"


@dataclass
class TelegramConfig:
    """Telegram configuration"""
    bot_token: str
    chat_id: str


class AlertManager:
    """Central alert management system"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Initialize channel configurations
        self.email_config = self._parse_email_config()
        self.slack_config = self._parse_slack_config()
        self.telegram_config = self._parse_telegram_config()
        
        # Alert routing rules
        self.routing_rules = {
            AlertSeverity.CRITICAL: ['email', 'slack', 'telegram'],
            AlertSeverity.ERROR: ['email', 'slack'],
            AlertSeverity.WARNING: ['slack'],
            AlertSeverity.INFO: ['console']
        }
        
        # Rate limiting
        self.rate_limits = {
            'email': timedelta(minutes=15),  # Max 1 email per 15 minutes per rule
            'slack': timedelta(minutes=5),   # Max 1 slack per 5 minutes per rule
            'telegram': timedelta(minutes=5),
            'console': timedelta(seconds=0)  # No rate limit for console
        }
        
        self.last_sent = {}  # Track last sent times for rate limiting
    
    def _parse_email_config(self) -> Optional[EmailConfig]:
        """Parse email configuration from config"""
        email_cfg = self.config.get('email', {})
        if not email_cfg.get('smtp_server'):
            return None
        
        return EmailConfig(
            smtp_server=email_cfg['smtp_server'],
            smtp_port=email_cfg.get('smtp_port', 587),
            username=email_cfg['username'],
            password=email_cfg['password'],
            from_email=email_cfg['from_email'],
            to_emails=email_cfg['to_emails'],
            use_tls=email_cfg.get('use_tls', True)
        )
    
    def _parse_slack_config(self) -> Optional[SlackConfig]:
        """Parse Slack configuration from config"""
        slack_cfg = self.config.get('slack', {})
        if not slack_cfg.get('webhook_url'):
            return None
        
        return SlackConfig(
            webhook_url=slack_cfg['webhook_url'],
            channel=slack_cfg.get('channel', '#data-quality'),
            username=slack_cfg.get('username', 'Data Quality Bot'),
            icon_emoji=slack_cfg.get('icon_emoji', ':warning:')
        )
    
    def _parse_telegram_config(self) -> Optional[TelegramConfig]:
        """Parse Telegram configuration from config"""
        telegram_cfg = self.config.get('telegram', {})
        if not telegram_cfg.get('bot_token'):
            return None
        
        return TelegramConfig(
            bot_token=telegram_cfg['bot_token'],
            chat_id=telegram_cfg['chat_id']
        )
    
    async def send_alert(self, alert: Alert):
        """Send alert through appropriate channels based on severity"""
        channels = self.routing_rules.get(alert.severity, ['console'])
        
        for channel in channels:
            if self._should_send_alert(alert, channel):
                try:
                    await self._send_to_channel(alert, channel)
                    self._update_rate_limit(alert, channel)
                except Exception as e:
                    logger.error(f"Failed to send alert via {channel}: {e}")
    
    def _should_send_alert(self, alert: Alert, channel: str) -> bool:
        """Check if alert should be sent based on rate limits"""
        rate_limit = self.rate_limits.get(channel, timedelta(0))
        if rate_limit.total_seconds() == 0:
            return True
        
        key = f"{alert.rule_name}_{channel}"
        last_sent_time = self.last_sent.get(key)
        
        if last_sent_time is None:
            return True
        
        return datetime.now() - last_sent_time > rate_limit
    
    def _update_rate_limit(self, alert: Alert, channel: str):
        """Update last sent time for rate limiting"""
        key = f"{alert.rule_name}_{channel}"
        self.last_sent[key] = datetime.now()
    
    async def _send_to_channel(self, alert: Alert, channel: str):
        """Send alert to specific channel"""
        if channel == 'email' and self.email_config:
            await self._send_email_alert(alert)
        elif channel == 'slack' and self.slack_config:
            await self._send_slack_alert(alert)
        elif channel == 'telegram' and self.telegram_config:
            await self._send_telegram_alert(alert)
        elif channel == 'console':
            self._send_console_alert(alert)
        else:
            logger.warning(f"Channel {channel} not configured or not available")
    
    async def _send_email_alert(self, alert: Alert):
        """Send alert via email"""
        if not self.email_config:
            logger.warning("Email not configured")
            return
        
        try:
            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.email_config.from_email
            msg['To'] = ', '.join(self.email_config.to_emails)
            msg['Subject'] = f"[{alert.severity.value.upper()}] Data Quality Alert - {alert.dataset_name}"
            
            # Create email body
            body = self._create_email_body(alert)
            msg.attach(MIMEText(body, 'html'))
            
            # Send email
            with smtplib.SMTP(self.email_config.smtp_server, self.email_config.smtp_port) as server:
                if self.email_config.use_tls:
                    server.starttls()
                server.login(self.email_config.username, self.email_config.password)
                server.send_message(msg)
            
            logger.info(f"Email alert sent for {alert.rule_name}")
            
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
    
    async def _send_slack_alert(self, alert: Alert):
        """Send alert via Slack"""
        if not self.slack_config:
            logger.warning("Slack not configured")
            return
        
        try:
            # Create Slack message
            color = self._get_slack_color(alert.severity)
            message = {
                "channel": self.slack_config.channel,
                "username": self.slack_config.username,
                "icon_emoji": self.slack_config.icon_emoji,
                "attachments": [
                    {
                        "color": color,
                        "title": f"{alert.severity.value.upper()} - Data Quality Alert",
                        "fields": [
                            {
                                "title": "Dataset",
                                "value": alert.dataset_name,
                                "short": True
                            },
                            {
                                "title": "Rule",
                                "value": alert.rule_name,
                                "short": True
                            },
                            {
                                "title": "Message",
                                "value": alert.message,
                                "short": False
                            },
                            {
                                "title": "Time",
                                "value": alert.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                                "short": True
                            }
                        ],
                        "footer": "Data Quality Monitor",
                        "ts": int(alert.timestamp.timestamp())
                    }
                ]
            }
            
            # Send to Slack
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.slack_config.webhook_url,
                    json=message,
                    headers={'Content-Type': 'application/json'}
                ) as response:
                    if response.status == 200:
                        logger.info(f"Slack alert sent for {alert.rule_name}")
                    else:
                        logger.error(f"Slack webhook returned status {response.status}")
                        
        except Exception as e:
            logger.error(f"Failed to send Slack alert: {e}")
    
    async def _send_telegram_alert(self, alert: Alert):
        """Send alert via Telegram"""
        if not self.telegram_config:
            logger.warning("Telegram not configured")
            return
        
        try:
            # Create Telegram message
            emoji = self._get_telegram_emoji(alert.severity)
            message = f"""
{emoji} *DATA QUALITY ALERT*

*Severity:* {alert.severity.value.upper()}
*Dataset:* {alert.dataset_name}
*Rule:* {alert.rule_name}

*Message:* {alert.message}

*Time:* {alert.timestamp.strftime("%Y-%m-%d %H:%M:%S")}
            """.strip()
            
            # Send to Telegram
            url = f"https://api.telegram.org/bot{self.telegram_config.bot_token}/sendMessage"
            data = {
                'chat_id': self.telegram_config.chat_id,
                'text': message,
                'parse_mode': 'Markdown'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=data) as response:
                    if response.status == 200:
                        logger.info(f"Telegram alert sent for {alert.rule_name}")
                    else:
                        logger.error(f"Telegram API returned status {response.status}")
                        
        except Exception as e:
            logger.error(f"Failed to send Telegram alert: {e}")
    
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
        timestamp = alert.timestamp.strftime("%Y-%m-%d %H:%M:%S")
        
        print(f"{color}[{alert.severity.value.upper()}] {timestamp}: {alert.message}{reset_color}")
        logger.warning(f"ALERT [{alert.severity.value.upper()}]: {alert.message}")
    
    def _create_email_body(self, alert: Alert) -> str:
        """Create HTML email body"""
        severity_colors = {
            AlertSeverity.CRITICAL: '#dc3545',
            AlertSeverity.ERROR: '#fd7e14', 
            AlertSeverity.WARNING: '#ffc107',
            AlertSeverity.INFO: '#17a2b8'
        }
        
        color = severity_colors.get(alert.severity, '#6c757d')
        
        return f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; }}
                .header {{ background-color: {color}; color: white; padding: 20px; }}
                .content {{ padding: 20px; }}
                .metrics {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; }}
                .footer {{ color: #6c757d; font-size: 12px; margin-top: 20px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h2>Data Quality Alert - {alert.severity.value.upper()}</h2>
            </div>
            <div class="content">
                <h3>Alert Details</h3>
                <p><strong>Dataset:</strong> {alert.dataset_name}</p>
                <p><strong>Rule:</strong> {alert.rule_name}</p>
                <p><strong>Message:</strong> {alert.message}</p>
                <p><strong>Time:</strong> {alert.timestamp.strftime("%Y-%m-%d %H:%M:%S")}</p>
                
                <h3>Metrics</h3>
                <div class="metrics">
                    {self._format_metrics_for_email(alert.metrics)}
                </div>
                
                <div class="footer">
                    <p>This alert was generated by the Data Quality Monitoring System.</p>
                    <p>Alert ID: {alert.id}</p>
                </div>
            </div>
        </body>
        </html>
        """
    
    def _format_metrics_for_email(self, metrics: Dict[str, Any]) -> str:
        """Format metrics for email display"""
        if not metrics:
            return "<p>No metrics available</p>"
        
        formatted = "<ul>"
        for key, value in metrics.items():
            if key in ['quality_score', 'missing_value_rate', 'outlier_rate', 'duplicate_rate']:
                if isinstance(value, float):
                    formatted += f"<li><strong>{key.replace('_', ' ').title()}:</strong> {value:.3f}</li>"
                else:
                    formatted += f"<li><strong>{key.replace('_', ' ').title()}:</strong> {value}</li>"
        formatted += "</ul>"
        return formatted
    
    def _get_slack_color(self, severity: AlertSeverity) -> str:
        """Get Slack attachment color for severity"""
        colors = {
            AlertSeverity.CRITICAL: "#dc3545",
            AlertSeverity.ERROR: "#fd7e14",
            AlertSeverity.WARNING: "#ffc107",
            AlertSeverity.INFO: "#17a2b8"
        }
        return colors.get(severity, "#6c757d")
    
    def _get_telegram_emoji(self, severity: AlertSeverity) -> str:
        """Get Telegram emoji for severity"""
        emojis = {
            AlertSeverity.CRITICAL: "🚨",
            AlertSeverity.ERROR: "❌",
            AlertSeverity.WARNING: "⚠️",
            AlertSeverity.INFO: "ℹ️"
        }
        return emojis.get(severity, "📢")
    
    async def send_test_alert(self, channel: str = 'all'):
        """Send a test alert to verify configuration"""
        test_alert = Alert(
            id="test_alert",
            rule_name="test_rule",
            severity=AlertSeverity.INFO,
            message="This is a test alert to verify the alerting system configuration.",
            timestamp=datetime.now(),
            dataset_name="test_dataset",
            metrics={
                'quality_score': 0.95,
                'total_records': 1000,
                'issues_count': 5
            }
        )
        
        if channel == 'all':
            await self.send_alert(test_alert)
        else:
            await self._send_to_channel(test_alert, channel)
        
        logger.info(f"Test alert sent via {channel}")
    
    def get_configuration_status(self) -> Dict[str, bool]:
        """Get status of all configured channels"""
        return {
            'email': self.email_config is not None,
            'slack': self.slack_config is not None,
            'telegram': self.telegram_config is not None,
            'console': True  # Always available
        }
    
    def update_routing_rules(self, new_rules: Dict[AlertSeverity, List[str]]):
        """Update alert routing rules"""
        self.routing_rules.update(new_rules)
        logger.info("Alert routing rules updated")
    
    def get_routing_rules(self) -> Dict[AlertSeverity, List[str]]:
        """Get current routing rules"""
        return self.routing_rules.copy()


# Example usage and configuration
def create_alert_manager_from_config(config_dict: Dict[str, Any]) -> AlertManager:
    """Create AlertManager from configuration dictionary"""
    
    # Example configuration structure:
    example_config = {
        'email': {
            'smtp_server': 'smtp.gmail.com',
            'smtp_port': 587,
            'username': 'your_email@gmail.com',
            'password': 'your_app_password',
            'from_email': 'alerts@yourcompany.com',
            'to_emails': ['admin@yourcompany.com', 'team@yourcompany.com'],
            'use_tls': True
        },
        'slack': {
            'webhook_url': 'https://hooks.slack.com/services/YOUR/WEBHOOK/URL',
            'channel': '#data-quality',
            'username': 'Data Quality Bot',
            'icon_emoji': ':warning:'
        },
        'telegram': {
            'bot_token': 'your_bot_token',
            'chat_id': 'your_chat_id'
        }
    }
    
    return AlertManager(config_dict)