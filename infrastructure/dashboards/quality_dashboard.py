"""
Data Quality Dashboard
Real-time web dashboard for monitoring data quality metrics and alerts
"""

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import asyncio
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any
import uvicorn

from ..data_validators.monitoring import DataQualityMonitor, AlertSeverity
from utils.logger import get_logger

logger = get_logger(__name__)


class QualityDashboard:
    """Web dashboard for data quality monitoring"""
    
    def __init__(self, monitor: DataQualityMonitor, host: str = "localhost", port: int = 8000):
        self.monitor = monitor
        self.host = host
        self.port = port
        self.app = FastAPI(title="Data Quality Dashboard", version="1.0.0")
        
        # WebSocket connections for real-time updates
        self.active_connections: List[WebSocket] = []
        
        # Setup routes
        self._setup_routes()
        
        # Setup templates (in production, you'd have actual template files)
        self.templates = Jinja2Templates(directory="templates")
    
    def _setup_routes(self):
        """Setup FastAPI routes"""
        
        @self.app.get("/", response_class=HTMLResponse)
        async def dashboard_home(request: Request):
            return HTMLResponse(self._get_dashboard_html())
        
        @self.app.get("/api/dashboard-data")
        async def get_dashboard_data():
            return JSONResponse(self.monitor.get_dashboard_data())
        
        @self.app.get("/api/metrics")
        async def get_metrics(dataset: str = None, hours: int = 24):
            metrics = self.monitor.get_metrics_history(dataset, hours)
            return JSONResponse([
                {
                    'timestamp': m.timestamp.isoformat(),
                    'dataset_name': m.dataset_name,
                    'quality_score': m.quality_score,
                    'total_records': m.total_records,
                    'valid_records': m.valid_records,
                    'missing_value_rate': m.missing_value_rate,
                    'outlier_rate': m.outlier_rate,
                    'duplicate_rate': m.duplicate_rate,
                    'issues_count': m.issues_count,
                    'critical_issues_count': m.critical_issues_count
                }
                for m in metrics
            ])
        
        @self.app.post("/api/acknowledge-alert/{alert_id}")
        async def acknowledge_alert(alert_id: str):
            success = self.monitor.acknowledge_alert(alert_id)
            return JSONResponse({"success": success})
        
        @self.app.get("/api/alert-rules")
        async def get_alert_rules():
            return JSONResponse([
                {
                    'name': rule.name,
                    'condition': rule.condition,
                    'severity': rule.severity.value,
                    'message_template': rule.message_template,
                    'cooldown_minutes': rule.cooldown_minutes,
                    'enabled': rule.enabled
                }
                for rule in self.monitor.alert_rules
            ])
        
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await self._handle_websocket(websocket)
    
    async def _handle_websocket(self, websocket: WebSocket):
        """Handle WebSocket connections for real-time updates"""
        await websocket.accept()
        self.active_connections.append(websocket)
        
        try:
            while True:
                # Send periodic updates
                data = self.monitor.get_dashboard_data()
                await websocket.send_text(json.dumps(data))
                await asyncio.sleep(30)  # Update every 30 seconds
        except WebSocketDisconnect:
            self.active_connections.remove(websocket)
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
            if websocket in self.active_connections:
                self.active_connections.remove(websocket)
    
    async def broadcast_update(self, data: Dict[str, Any]):
        """Broadcast updates to all connected clients"""
        if self.active_connections:
            message = json.dumps(data)
            for connection in self.active_connections.copy():
                try:
                    await connection.send_text(message)
                except:
                    self.active_connections.remove(connection)
    
    def _get_dashboard_html(self) -> str:
        """Generate dashboard HTML (in production, use proper templates)"""
        return '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Data Quality Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        .header h1 {
            margin: 0;
            font-size: 2.5em;
        }
        .header p {
            margin: 5px 0 0 0;
            opacity: 0.9;
        }
        .dashboard-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }
        .card {
            background: white;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            border-left: 4px solid #667eea;
        }
        .card h3 {
            margin-top: 0;
            color: #333;
        }
        .metric-value {
            font-size: 2em;
            font-weight: bold;
            color: #667eea;
        }
        .metric-label {
            color: #666;
            font-size: 0.9em;
        }
        .alert {
            padding: 10px;
            margin: 5px 0;
            border-radius: 5px;
            border-left: 4px solid;
        }
        .alert.critical {
            background-color: #ffeaea;
            border-left-color: #dc3545;
            color: #721c24;
        }
        .alert.error {
            background-color: #fff3cd;
            border-left-color: #fd7e14;
            color: #856404;
        }
        .alert.warning {
            background-color: #fff3cd;
            border-left-color: #ffc107;
            color: #856404;
        }
        .alert.info {
            background-color: #d1ecf1;
            border-left-color: #17a2b8;
            color: #0c5460;
        }
        .chart-container {
            height: 400px;
            margin-top: 20px;
        }
        .status-indicator {
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 8px;
        }
        .status-active {
            background-color: #28a745;
        }
        .status-inactive {
            background-color: #dc3545;
        }
        .btn {
            background-color: #667eea;
            color: white;
            padding: 8px 16px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
        }
        .btn:hover {
            background-color: #5a6fd8;
        }
        #lastUpdate {
            color: #666;
            font-size: 0.8em;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🔍 Data Quality Dashboard</h1>
        <p>Real-time monitoring and validation of data quality metrics</p>
        <span id="lastUpdate">Loading...</span>
    </div>

    <div class="dashboard-grid">
        <div class="card">
            <h3>
                <span id="statusIndicator" class="status-indicator status-inactive"></span>
                System Status
            </h3>
            <div class="metric-value" id="monitoringStatus">Checking...</div>
            <div class="metric-label">Monitoring Status</div>
        </div>

        <div class="card">
            <h3>📊 Quality Score</h3>
            <div class="metric-value" id="avgQualityScore">-</div>
            <div class="metric-label">Average Quality Score</div>
        </div>

        <div class="card">
            <h3>📁 Datasets</h3>
            <div class="metric-value" id="datasetsMonitored">-</div>
            <div class="metric-label">Datasets Monitored</div>
        </div>

        <div class="card">
            <h3>📈 Records Processed</h3>
            <div class="metric-value" id="recordsProcessed">-</div>
            <div class="metric-label">Total Records (24h)</div>
        </div>

        <div class="card">
            <h3>🚨 Critical Alerts</h3>
            <div class="metric-value" id="criticalAlerts">-</div>
            <div class="metric-label">Requiring Attention</div>
        </div>

        <div class="card">
            <h3>⚠️ Warnings</h3>
            <div class="metric-value" id="warningAlerts">-</div>
            <div class="metric-label">Active Warnings</div>
        </div>
    </div>

    <div class="dashboard-grid">
        <div class="card">
            <h3>📈 Quality Trends</h3>
            <div class="chart-container">
                <canvas id="qualityTrendChart"></canvas>
            </div>
        </div>

        <div class="card">
            <h3>🔔 Recent Alerts</h3>
            <div id="recentAlerts">
                <p>Loading alerts...</p>
            </div>
        </div>
    </div>

    <div class="card">
        <h3>📋 Dataset Metrics</h3>
        <div id="datasetMetrics">
            <p>Loading dataset metrics...</p>
        </div>
    </div>

    <script>
        let chart = null;
        
        // WebSocket connection for real-time updates
        const ws = new WebSocket(`ws://${window.location.host}/ws`);
        
        ws.onmessage = function(event) {
            const data = JSON.parse(event.data);
            updateDashboard(data);
        };
        
        ws.onopen = function() {
            console.log('WebSocket connected');
        };
        
        ws.onclose = function() {
            console.log('WebSocket disconnected');
            // Try to reconnect after 5 seconds
            setTimeout(() => {
                location.reload();
            }, 5000);
        };

        function updateDashboard(data) {
            // Update summary metrics
            const summary = data.summary;
            document.getElementById('monitoringStatus').textContent = 
                summary.monitoring_status === 'active' ? 'Active' : 'Inactive';
            
            const statusIndicator = document.getElementById('statusIndicator');
            if (summary.monitoring_status === 'active') {
                statusIndicator.className = 'status-indicator status-active';
            } else {
                statusIndicator.className = 'status-indicator status-inactive';
            }
            
            document.getElementById('avgQualityScore').textContent = summary.avg_quality_score.toFixed(3);
            document.getElementById('datasetsMonitored').textContent = summary.datasets_monitored;
            document.getElementById('recordsProcessed').textContent = summary.total_records_processed.toLocaleString();
            document.getElementById('criticalAlerts').textContent = data.alerts.counts.critical;
            document.getElementById('warningAlerts').textContent = data.alerts.counts.warning;
            document.getElementById('lastUpdate').textContent = `Last updated: ${new Date(summary.last_update).toLocaleTimeString()}`;
            
            // Update quality trend chart
            updateQualityChart(data.quality_trends);
            
            // Update recent alerts
            updateRecentAlerts(data.alerts.recent_alerts);
            
            // Update dataset metrics
            updateDatasetMetrics(data.dataset_metrics);
        }

        function updateQualityChart(trends) {
            const ctx = document.getElementById('qualityTrendChart').getContext('2d');
            
            if (chart) {
                chart.destroy();
            }
            
            chart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: trends.map(t => new Date(t.date).toLocaleDateString()),
                    datasets: [{
                        label: 'Quality Score',
                        data: trends.map(t => t.quality_score),
                        borderColor: '#667eea',
                        backgroundColor: 'rgba(102, 126, 234, 0.1)',
                        fill: true,
                        tension: 0.4
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            beginAtZero: true,
                            max: 1.0
                        }
                    },
                    plugins: {
                        legend: {
                            display: false
                        }
                    }
                }
            });
        }

        function updateRecentAlerts(alerts) {
            const container = document.getElementById('recentAlerts');
            
            if (alerts.length === 0) {
                container.innerHTML = '<p style="color: #28a745;">✅ No recent alerts</p>';
                return;
            }
            
            container.innerHTML = alerts.map(alert => `
                <div class="alert ${alert.severity}">
                    <strong>${alert.severity.toUpperCase()}</strong>
                    <p>${alert.message}</p>
                    <small>${new Date(alert.timestamp).toLocaleString()}</small>
                    ${!alert.acknowledged ? 
                        `<button class="btn" onclick="acknowledgeAlert('${alert.id}')">Acknowledge</button>` : 
                        '<span style="color: green;">✓ Acknowledged</span>'
                    }
                </div>
            `).join('');
        }

        function updateDatasetMetrics(metrics) {
            const container = document.getElementById('datasetMetrics');
            
            if (metrics.length === 0) {
                container.innerHTML = '<p>No recent dataset metrics available</p>';
                return;
            }
            
            container.innerHTML = `
                <table style="width: 100%; border-collapse: collapse;">
                    <thead>
                        <tr style="border-bottom: 2px solid #eee;">
                            <th style="text-align: left; padding: 10px;">Dataset</th>
                            <th style="text-align: left; padding: 10px;">Quality Score</th>
                            <th style="text-align: left; padding: 10px;">Records</th>
                            <th style="text-align: left; padding: 10px;">Issues</th>
                            <th style="text-align: left; padding: 10px;">Last Updated</th>
                        </tr>
                    </thead>
                    <tbody>
                        ${metrics.map(m => `
                            <tr style="border-bottom: 1px solid #eee;">
                                <td style="padding: 10px;">${m.dataset_name}</td>
                                <td style="padding: 10px;">
                                    <span style="color: ${m.quality_score > 0.8 ? '#28a745' : m.quality_score > 0.6 ? '#ffc107' : '#dc3545'}">
                                        ${m.quality_score.toFixed(3)}
                                    </span>
                                </td>
                                <td style="padding: 10px;">${m.total_records.toLocaleString()}</td>
                                <td style="padding: 10px;">
                                    ${m.critical_issues > 0 ? `<span style="color: #dc3545;">🔴 ${m.critical_issues}</span>` : ''}
                                    ${m.issues_count - m.critical_issues > 0 ? `⚠️ ${m.issues_count - m.critical_issues}` : ''}
                                    ${m.issues_count === 0 ? '✅' : ''}
                                </td>
                                <td style="padding: 10px;">${new Date(m.timestamp).toLocaleTimeString()}</td>
                            </tr>
                        `).join('')}
                    </tbody>
                </table>
            `;
        }

        function acknowledgeAlert(alertId) {
            fetch(`/api/acknowledge-alert/${alertId}`, { method: 'POST' })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        // Refresh the alerts display
                        fetch('/api/dashboard-data')
                            .then(response => response.json())
                            .then(data => updateRecentAlerts(data.alerts.recent_alerts));
                    }
                });
        }

        // Initial load
        fetch('/api/dashboard-data')
            .then(response => response.json())
            .then(data => updateDashboard(data));
    </script>
</body>
</html>
        '''
    
    async def start_server(self):
        """Start the dashboard server"""
        logger.info(f"Starting quality dashboard at http://{self.host}:{self.port}")
        
        config = uvicorn.Config(
            app=self.app,
            host=self.host,
            port=self.port,
            log_level="info"
        )
        server = uvicorn.Server(config)
        await server.serve()
    
    def run(self):
        """Run the dashboard server (blocking)"""
        uvicorn.run(self.app, host=self.host, port=self.port)