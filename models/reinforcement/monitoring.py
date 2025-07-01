"""
Real-time Monitoring and Visualization System for RL Trading Agents
Comprehensive monitoring dashboard with live metrics and performance tracking
"""

import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import threading
import queue
import time
import json
import sqlite3
from pathlib import Path
import websocket
import asyncio
from flask import Flask
import logging

from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class MonitoringConfig:
    """Configuration for monitoring system"""
    # Dashboard configuration
    dashboard_host: str = "localhost"
    dashboard_port: int = 8050
    update_interval: float = 1.0  # seconds
    
    # Data retention
    max_data_points: int = 10000
    data_retention_days: int = 30
    
    # Database configuration
    database_path: str = "./monitoring.db"
    enable_database: bool = True
    
    # Alert configuration
    enable_alerts: bool = True
    alert_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'max_drawdown': -0.2,
        'sharpe_ratio': 0.5,
        'loss_streak': 10
    })
    
    # Visualization configuration
    theme: str = "plotly_dark"
    chart_height: int = 400
    chart_colors: Dict[str, str] = field(default_factory=lambda: {
        'profit': '#00CC96',
        'loss': '#FF6692',
        'neutral': '#636EFA'
    })


@dataclass
class MetricSnapshot:
    """Single point-in-time metric snapshot"""
    timestamp: datetime
    agent_id: str
    episode: int
    
    # Performance metrics
    episode_reward: float
    cumulative_reward: float
    portfolio_value: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    
    # Training metrics
    actor_loss: Optional[float] = None
    critic_loss: Optional[float] = None
    learning_rate: Optional[float] = None
    exploration_rate: Optional[float] = None
    
    # Environment metrics
    num_trades: int = 0
    cash_balance: float = 0.0
    total_positions: int = 0
    
    # System metrics
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    gpu_usage: float = 0.0


class MetricsDatabase:
    """Database for storing monitoring metrics"""
    
    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._init_database()
        
        logger.info(f"Initialized metrics database at {db_path}")
    
    def _init_database(self):
        """Initialize database schema"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME,
                    agent_id TEXT,
                    episode INTEGER,
                    episode_reward REAL,
                    cumulative_reward REAL,
                    portfolio_value REAL,
                    sharpe_ratio REAL,
                    max_drawdown REAL,
                    win_rate REAL,
                    actor_loss REAL,
                    critic_loss REAL,
                    learning_rate REAL,
                    exploration_rate REAL,
                    num_trades INTEGER,
                    cash_balance REAL,
                    total_positions INTEGER,
                    cpu_usage REAL,
                    memory_usage REAL,
                    gpu_usage REAL
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp_agent 
                ON metrics(timestamp, agent_id)
            """)
    
    def insert_metric(self, metric: MetricSnapshot):
        """Insert metric snapshot into database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO metrics (
                    timestamp, agent_id, episode, episode_reward, cumulative_reward,
                    portfolio_value, sharpe_ratio, max_drawdown, win_rate,
                    actor_loss, critic_loss, learning_rate, exploration_rate,
                    num_trades, cash_balance, total_positions,
                    cpu_usage, memory_usage, gpu_usage
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                metric.timestamp, metric.agent_id, metric.episode,
                metric.episode_reward, metric.cumulative_reward, metric.portfolio_value,
                metric.sharpe_ratio, metric.max_drawdown, metric.win_rate,
                metric.actor_loss, metric.critic_loss, metric.learning_rate,
                metric.exploration_rate, metric.num_trades, metric.cash_balance,
                metric.total_positions, metric.cpu_usage, metric.memory_usage,
                metric.gpu_usage
            ))
    
    def get_recent_metrics(self, agent_id: str = None, hours: int = 24) -> pd.DataFrame:
        """Get recent metrics from database"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        query = "SELECT * FROM metrics WHERE timestamp > ?"
        params = [cutoff_time]
        
        if agent_id:
            query += " AND agent_id = ?"
            params.append(agent_id)
        
        query += " ORDER BY timestamp DESC"
        
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query(query, conn, params=params)
    
    def cleanup_old_data(self, retention_days: int):
        """Remove old data beyond retention period"""
        cutoff_time = datetime.now() - timedelta(days=retention_days)
        
        with sqlite3.connect(self.db_path) as conn:
            result = conn.execute(
                "DELETE FROM metrics WHERE timestamp < ?", 
                (cutoff_time,)
            )
            logger.info(f"Cleaned up {result.rowcount} old metric records")


class MetricsCollector:
    """Collects metrics from various sources"""
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.metrics_queue = queue.Queue()
        self.is_collecting = False
        
        # Database
        if config.enable_database:
            self.database = MetricsDatabase(config.database_path)
        else:
            self.database = None
        
        # In-memory storage for dashboard
        self.recent_metrics: Dict[str, List[MetricSnapshot]] = {}
        
        logger.info("Initialized metrics collector")
    
    def start_collection(self):
        """Start metrics collection"""
        self.is_collecting = True
        
        # Start collection thread
        self.collection_thread = threading.Thread(
            target=self._collection_loop,
            daemon=True
        )
        self.collection_thread.start()
        
        # Start cleanup thread
        if self.database:
            self.cleanup_thread = threading.Thread(
                target=self._cleanup_loop,
                daemon=True
            )
            self.cleanup_thread.start()
        
        logger.info("Started metrics collection")
    
    def stop_collection(self):
        """Stop metrics collection"""
        self.is_collecting = False
        logger.info("Stopped metrics collection")
    
    def add_metric(self, metric: MetricSnapshot):
        """Add metric to collection queue"""
        self.metrics_queue.put(metric)
    
    def _collection_loop(self):
        """Main collection loop"""
        while self.is_collecting:
            try:
                # Process queued metrics
                while not self.metrics_queue.empty():
                    metric = self.metrics_queue.get_nowait()
                    self._process_metric(metric)
                
                time.sleep(0.1)  # Short sleep to prevent busy waiting
                
            except Exception as e:
                logger.error(f"Collection loop error: {e}")
                time.sleep(1)
    
    def _process_metric(self, metric: MetricSnapshot):
        """Process a single metric"""
        # Store in database
        if self.database:
            self.database.insert_metric(metric)
        
        # Store in memory for dashboard
        if metric.agent_id not in self.recent_metrics:
            self.recent_metrics[metric.agent_id] = []
        
        self.recent_metrics[metric.agent_id].append(metric)
        
        # Limit in-memory storage
        if len(self.recent_metrics[metric.agent_id]) > self.config.max_data_points:
            self.recent_metrics[metric.agent_id] = \
                self.recent_metrics[metric.agent_id][-self.config.max_data_points:]
        
        # Check alerts
        if self.config.enable_alerts:
            self._check_alerts(metric)
    
    def _check_alerts(self, metric: MetricSnapshot):
        """Check if metric triggers any alerts"""
        alerts = []
        
        # Max drawdown alert
        if metric.max_drawdown < self.config.alert_thresholds['max_drawdown']:
            alerts.append(f"High drawdown alert: {metric.max_drawdown:.2%}")
        
        # Sharpe ratio alert
        if metric.sharpe_ratio < self.config.alert_thresholds['sharpe_ratio']:
            alerts.append(f"Low Sharpe ratio alert: {metric.sharpe_ratio:.2f}")
        
        # Log alerts
        for alert in alerts:
            logger.warning(f"ALERT - Agent {metric.agent_id}: {alert}")
    
    def _cleanup_loop(self):
        """Periodic cleanup of old data"""
        while self.is_collecting:
            try:
                if self.database:
                    self.database.cleanup_old_data(self.config.data_retention_days)
                
                # Sleep for 1 hour between cleanups
                time.sleep(3600)
                
            except Exception as e:
                logger.error(f"Cleanup error: {e}")
                time.sleep(300)  # 5 minutes on error
    
    def get_metrics_dataframe(self, agent_id: str = None) -> pd.DataFrame:
        """Get recent metrics as DataFrame"""
        if self.database:
            return self.database.get_recent_metrics(agent_id)
        else:
            # Convert in-memory data to DataFrame
            all_metrics = []
            
            if agent_id and agent_id in self.recent_metrics:
                metrics_list = self.recent_metrics[agent_id]
            else:
                metrics_list = []
                for agent_metrics in self.recent_metrics.values():
                    metrics_list.extend(agent_metrics)
            
            for metric in metrics_list:
                all_metrics.append({
                    'timestamp': metric.timestamp,
                    'agent_id': metric.agent_id,
                    'episode': metric.episode,
                    'episode_reward': metric.episode_reward,
                    'cumulative_reward': metric.cumulative_reward,
                    'portfolio_value': metric.portfolio_value,
                    'sharpe_ratio': metric.sharpe_ratio,
                    'max_drawdown': metric.max_drawdown,
                    'win_rate': metric.win_rate,
                    'actor_loss': metric.actor_loss,
                    'critic_loss': metric.critic_loss,
                    'num_trades': metric.num_trades
                })
            
            return pd.DataFrame(all_metrics)


class MonitoringDashboard:
    """Real-time monitoring dashboard using Dash"""
    
    def __init__(self, config: MonitoringConfig, metrics_collector: MetricsCollector):
        self.config = config
        self.metrics_collector = metrics_collector
        
        # Initialize Dash app
        self.app = dash.Dash(__name__)
        self.app.title = "RL Trading Agent Monitor"
        
        # Setup layout
        self._setup_layout()
        
        # Setup callbacks
        self._setup_callbacks()
        
        logger.info("Initialized monitoring dashboard")
    
    def _setup_layout(self):
        """Setup dashboard layout"""
        self.app.layout = html.Div([
            html.H1("RL Trading Agent Monitor", 
                   style={'textAlign': 'center', 'marginBottom': 30}),
            
            # Controls
            html.Div([
                html.Div([
                    html.Label("Agent ID:"),
                    dcc.Dropdown(
                        id='agent-dropdown',
                        options=[],
                        value=None,
                        placeholder="Select agent..."
                    )
                ], style={'width': '30%', 'display': 'inline-block'}),
                
                html.Div([
                    html.Label("Time Range:"),
                    dcc.Dropdown(
                        id='time-range-dropdown',
                        options=[
                            {'label': 'Last Hour', 'value': 1},
                            {'label': 'Last 6 Hours', 'value': 6},
                            {'label': 'Last 24 Hours', 'value': 24},
                            {'label': 'Last Week', 'value': 168}
                        ],
                        value=6
                    )
                ], style={'width': '30%', 'display': 'inline-block', 'marginLeft': '20px'}),
                
                html.Div([
                    html.Button('Refresh', id='refresh-button', n_clicks=0,
                               style={'marginTop': '25px'})
                ], style={'width': '20%', 'display': 'inline-block', 'marginLeft': '20px'})
            ], style={'marginBottom': 30}),
            
            # Performance metrics row
            html.Div([
                html.Div([
                    dcc.Graph(id='portfolio-value-chart')
                ], style={'width': '50%', 'display': 'inline-block'}),
                
                html.Div([
                    dcc.Graph(id='episode-rewards-chart')
                ], style={'width': '50%', 'display': 'inline-block'})
            ]),
            
            # Risk metrics row
            html.Div([
                html.Div([
                    dcc.Graph(id='sharpe-ratio-chart')
                ], style={'width': '50%', 'display': 'inline-block'}),
                
                html.Div([
                    dcc.Graph(id='drawdown-chart')
                ], style={'width': '50%', 'display': 'inline-block'})
            ]),
            
            # Training metrics row
            html.Div([
                html.Div([
                    dcc.Graph(id='loss-chart')
                ], style={'width': '50%', 'display': 'inline-block'}),
                
                html.Div([
                    dcc.Graph(id='learning-rate-chart')
                ], style={'width': '50%', 'display': 'inline-block'})
            ]),
            
            # Auto-refresh interval
            dcc.Interval(
                id='interval-component',
                interval=self.config.update_interval * 1000,  # in milliseconds
                n_intervals=0
            )
        ])
    
    def _setup_callbacks(self):
        """Setup dashboard callbacks"""
        
        @self.app.callback(
            [Output('agent-dropdown', 'options'),
             Output('portfolio-value-chart', 'figure'),
             Output('episode-rewards-chart', 'figure'),
             Output('sharpe-ratio-chart', 'figure'),
             Output('drawdown-chart', 'figure'),
             Output('loss-chart', 'figure'),
             Output('learning-rate-chart', 'figure')],
            [Input('interval-component', 'n_intervals'),
             Input('refresh-button', 'n_clicks')],
            [State('agent-dropdown', 'value'),
             State('time-range-dropdown', 'value')]
        )
        def update_dashboard(n_intervals, n_clicks, selected_agent, time_range):
            # Get data
            df = self.metrics_collector.get_metrics_dataframe(selected_agent)
            
            if df.empty:
                empty_fig = go.Figure()
                empty_fig.update_layout(title="No data available")
                return [], empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, empty_fig
            
            # Filter by time range
            if time_range:
                cutoff_time = datetime.now() - timedelta(hours=time_range)
                df = df[pd.to_datetime(df['timestamp']) > cutoff_time]
            
            # Update agent options
            agent_options = [
                {'label': agent_id, 'value': agent_id}
                for agent_id in df['agent_id'].unique()
            ]
            
            # Create charts
            portfolio_fig = self._create_portfolio_chart(df)
            rewards_fig = self._create_rewards_chart(df)
            sharpe_fig = self._create_sharpe_chart(df)
            drawdown_fig = self._create_drawdown_chart(df)
            loss_fig = self._create_loss_chart(df)
            lr_fig = self._create_learning_rate_chart(df)
            
            return (agent_options, portfolio_fig, rewards_fig, 
                   sharpe_fig, drawdown_fig, loss_fig, lr_fig)
    
    def _create_portfolio_chart(self, df: pd.DataFrame) -> go.Figure:
        """Create portfolio value chart"""
        fig = go.Figure()
        
        for agent_id in df['agent_id'].unique():
            agent_df = df[df['agent_id'] == agent_id]
            
            fig.add_trace(go.Scatter(
                x=pd.to_datetime(agent_df['timestamp']),
                y=agent_df['portfolio_value'],
                mode='lines',
                name=f'Agent {agent_id}',
                line=dict(width=2)
            ))
        
        fig.update_layout(
            title='Portfolio Value Over Time',
            xaxis_title='Time',
            yaxis_title='Portfolio Value ($)',
            template=self.config.theme,
            height=self.config.chart_height
        )
        
        return fig
    
    def _create_rewards_chart(self, df: pd.DataFrame) -> go.Figure:
        """Create episode rewards chart"""
        fig = go.Figure()
        
        for agent_id in df['agent_id'].unique():
            agent_df = df[df['agent_id'] == agent_id]
            
            # Episode rewards
            fig.add_trace(go.Scatter(
                x=agent_df['episode'],
                y=agent_df['episode_reward'],
                mode='markers',
                name=f'Episode Rewards - Agent {agent_id}',
                opacity=0.6
            ))
            
            # Moving average
            if len(agent_df) > 10:
                rolling_mean = agent_df['episode_reward'].rolling(window=10).mean()
                fig.add_trace(go.Scatter(
                    x=agent_df['episode'],
                    y=rolling_mean,
                    mode='lines',
                    name=f'Moving Avg - Agent {agent_id}',
                    line=dict(width=3)
                ))
        
        fig.update_layout(
            title='Episode Rewards',
            xaxis_title='Episode',
            yaxis_title='Reward',
            template=self.config.theme,
            height=self.config.chart_height
        )
        
        return fig
    
    def _create_sharpe_chart(self, df: pd.DataFrame) -> go.Figure:
        """Create Sharpe ratio chart"""
        fig = go.Figure()
        
        for agent_id in df['agent_id'].unique():
            agent_df = df[df['agent_id'] == agent_id]
            
            fig.add_trace(go.Scatter(
                x=pd.to_datetime(agent_df['timestamp']),
                y=agent_df['sharpe_ratio'],
                mode='lines',
                name=f'Agent {agent_id}',
                line=dict(width=2)
            ))
        
        # Add reference line at Sharpe = 1
        fig.add_hline(y=1.0, line_dash="dash", line_color="gray", 
                     annotation_text="Sharpe = 1.0")
        
        fig.update_layout(
            title='Sharpe Ratio Over Time',
            xaxis_title='Time',
            yaxis_title='Sharpe Ratio',
            template=self.config.theme,
            height=self.config.chart_height
        )
        
        return fig
    
    def _create_drawdown_chart(self, df: pd.DataFrame) -> go.Figure:
        """Create drawdown chart"""
        fig = go.Figure()
        
        for agent_id in df['agent_id'].unique():
            agent_df = df[df['agent_id'] == agent_id]
            
            fig.add_trace(go.Scatter(
                x=pd.to_datetime(agent_df['timestamp']),
                y=agent_df['max_drawdown'] * 100,  # Convert to percentage
                mode='lines',
                name=f'Agent {agent_id}',
                line=dict(width=2),
                fill='tonexty' if agent_id != df['agent_id'].iloc[0] else 'tozeroy'
            ))
        
        fig.update_layout(
            title='Maximum Drawdown Over Time',
            xaxis_title='Time',
            yaxis_title='Max Drawdown (%)',
            template=self.config.theme,
            height=self.config.chart_height
        )
        
        return fig
    
    def _create_loss_chart(self, df: pd.DataFrame) -> go.Figure:
        """Create training loss chart"""
        fig = go.Figure()
        
        for agent_id in df['agent_id'].unique():
            agent_df = df[df['agent_id'] == agent_id]
            
            # Actor loss
            if 'actor_loss' in agent_df.columns and agent_df['actor_loss'].notna().any():
                fig.add_trace(go.Scatter(
                    x=agent_df['episode'],
                    y=agent_df['actor_loss'],
                    mode='lines',
                    name=f'Actor Loss - Agent {agent_id}',
                    line=dict(width=2)
                ))
            
            # Critic loss
            if 'critic_loss' in agent_df.columns and agent_df['critic_loss'].notna().any():
                fig.add_trace(go.Scatter(
                    x=agent_df['episode'],
                    y=agent_df['critic_loss'],
                    mode='lines',
                    name=f'Critic Loss - Agent {agent_id}',
                    line=dict(width=2, dash='dash')
                ))
        
        fig.update_layout(
            title='Training Losses',
            xaxis_title='Episode',
            yaxis_title='Loss',
            template=self.config.theme,
            height=self.config.chart_height
        )
        
        return fig
    
    def _create_learning_rate_chart(self, df: pd.DataFrame) -> go.Figure:
        """Create learning rate chart"""
        fig = go.Figure()
        
        for agent_id in df['agent_id'].unique():
            agent_df = df[df['agent_id'] == agent_id]
            
            if 'learning_rate' in agent_df.columns and agent_df['learning_rate'].notna().any():
                fig.add_trace(go.Scatter(
                    x=agent_df['episode'],
                    y=agent_df['learning_rate'],
                    mode='lines',
                    name=f'Agent {agent_id}',
                    line=dict(width=2)
                ))
        
        fig.update_layout(
            title='Learning Rate Schedule',
            xaxis_title='Episode',
            yaxis_title='Learning Rate',
            template=self.config.theme,
            height=self.config.chart_height
        )
        
        return fig
    
    def run(self, debug: bool = False):
        """Run the dashboard"""
        self.app.run_server(
            host=self.config.dashboard_host,
            port=self.config.dashboard_port,
            debug=debug
        )


class RealTimeMonitor:
    """
    Main real-time monitoring system
    
    Features:
    - Metrics collection from multiple agents
    - Real-time dashboard with live updates
    - Database storage and historical analysis
    - Alert system for performance issues
    - Resource usage monitoring
    """
    
    def __init__(self, config: MonitoringConfig = None):
        self.config = config or MonitoringConfig()
        
        # Initialize components
        self.metrics_collector = MetricsCollector(self.config)
        self.dashboard = MonitoringDashboard(self.config, self.metrics_collector)
        
        logger.info("Initialized real-time monitor")
    
    def start(self):
        """Start monitoring system"""
        # Start metrics collection
        self.metrics_collector.start_collection()
        
        # Start dashboard
        logger.info(f"Starting dashboard at http://{self.config.dashboard_host}:{self.config.dashboard_port}")
        self.dashboard.run()
    
    def stop(self):
        """Stop monitoring system"""
        self.metrics_collector.stop_collection()
        logger.info("Stopped monitoring system")
    
    def add_metric(self, metric: MetricSnapshot):
        """Add metric to monitoring system"""
        self.metrics_collector.add_metric(metric)
    
    def create_metric_snapshot(self, agent_id: str, episode: int, **kwargs) -> MetricSnapshot:
        """Helper to create metric snapshot"""
        return MetricSnapshot(
            timestamp=datetime.now(),
            agent_id=agent_id,
            episode=episode,
            **kwargs
        )


# Factory functions
def create_monitor(config: Dict[str, Any] = None) -> RealTimeMonitor:
    """Create real-time monitor"""
    monitoring_config = MonitoringConfig(**(config or {}))
    return RealTimeMonitor(monitoring_config)


def start_monitoring_dashboard(port: int = 8050, **kwargs) -> RealTimeMonitor:
    """Start monitoring dashboard with simple configuration"""
    config = MonitoringConfig(dashboard_port=port, **kwargs)
    monitor = RealTimeMonitor(config)
    monitor.start()
    return monitor


# Export classes and functions
__all__ = [
    'MonitoringConfig',
    'MetricSnapshot',
    'MetricsDatabase',
    'MetricsCollector',
    'MonitoringDashboard',
    'RealTimeMonitor',
    'create_monitor',
    'start_monitoring_dashboard'
]