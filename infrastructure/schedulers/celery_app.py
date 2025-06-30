"""
Celery application for automated data collection and processing
Handles scheduled tasks for market data, news collection, and analytics
"""

import os
from datetime import timedelta
from celery import Celery
from celery.schedules import crontab
from kombu import Queue, Exchange

# Celery configuration
BROKER_URL = os.getenv('CELERY_BROKER_URL', 'redis://localhost:6379/0')
RESULT_BACKEND = os.getenv('CELERY_RESULT_BACKEND', 'redis://localhost:6379/1')

# Create Celery app
app = Celery('trading_system')

# Configure Celery
app.conf.update(
    broker_url=BROKER_URL,
    result_backend=RESULT_BACKEND,
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='Asia/Seoul',
    enable_utc=True,
    
    # Task routing
    task_routes={
        'schedulers.data_tasks.collect_market_data': {'queue': 'market_data'},
        'schedulers.data_tasks.collect_news_data': {'queue': 'news'},
        'schedulers.data_tasks.process_analytics': {'queue': 'analytics'},
        'schedulers.data_tasks.health_check': {'queue': 'monitoring'},
    },
    
    # Queue configuration
    task_default_queue='default',
    task_queues=(
        Queue('default', Exchange('default'), routing_key='default'),
        Queue('market_data', Exchange('market_data'), routing_key='market_data'),
        Queue('news', Exchange('news'), routing_key='news'),
        Queue('analytics', Exchange('analytics'), routing_key='analytics'),
        Queue('monitoring', Exchange('monitoring'), routing_key='monitoring'),
    ),
    
    # Worker configuration
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    worker_max_tasks_per_child=1000,
    
    # Task execution settings
    task_soft_time_limit=300,  # 5 minutes
    task_time_limit=600,       # 10 minutes
    task_reject_on_worker_lost=True,
    
    # Result backend settings
    result_expires=3600,  # 1 hour
    result_persistent=True,
    
    # Error handling
    task_retry_max=3,
    task_retry_delay=60,  # 1 minute
)

# Beat schedule for periodic tasks
app.conf.beat_schedule = {
    # Market data collection
    'collect-crypto-1min': {
        'task': 'schedulers.data_tasks.collect_crypto_data',
        'schedule': timedelta(minutes=1),
        'options': {'queue': 'market_data'}
    },
    
    'collect-crypto-5min': {
        'task': 'schedulers.data_tasks.collect_crypto_data',
        'schedule': timedelta(minutes=5),
        'args': (['KRW-BTC', 'KRW-ETH', 'KRW-XRP'], ['5m']),
        'options': {'queue': 'market_data'}
    },
    
    'collect-stocks-korea': {
        'task': 'schedulers.data_tasks.collect_korean_stocks',
        'schedule': crontab(minute='*/15', hour='9-15', day_of_week='1-5'),  # Trading hours
        'options': {'queue': 'market_data'}
    },
    
    'collect-stocks-us': {
        'task': 'schedulers.data_tasks.collect_us_stocks',
        'schedule': crontab(minute='*/15', hour='23-6', day_of_week='0-4'),  # US trading hours in KST
        'options': {'queue': 'market_data'}
    },
    
    # News collection
    'collect-news-frequent': {
        'task': 'schedulers.data_tasks.collect_news_data',
        'schedule': timedelta(minutes=30),
        'options': {'queue': 'news'}
    },
    
    'collect-korean-news': {
        'task': 'schedulers.data_tasks.collect_korean_news',
        'schedule': timedelta(hours=1),
        'options': {'queue': 'news'}
    },
    
    # Analytics and processing
    'calculate-technical-indicators': {
        'task': 'schedulers.data_tasks.calculate_technical_indicators',
        'schedule': timedelta(minutes=5),
        'options': {'queue': 'analytics'}
    },
    
    'update-volume-profiles': {
        'task': 'schedulers.data_tasks.update_volume_profiles',
        'schedule': timedelta(minutes=15),
        'options': {'queue': 'analytics'}
    },
    
    'sentiment-analysis': {
        'task': 'schedulers.data_tasks.analyze_news_sentiment',
        'schedule': timedelta(hours=2),
        'options': {'queue': 'analytics'}
    },
    
    # Daily tasks
    'daily-portfolio-snapshot': {
        'task': 'schedulers.data_tasks.create_portfolio_snapshot',
        'schedule': crontab(hour=0, minute=0),  # Midnight KST
        'options': {'queue': 'analytics'}
    },
    
    'daily-performance-report': {
        'task': 'schedulers.data_tasks.generate_performance_report',
        'schedule': crontab(hour=1, minute=0),  # 1 AM KST
        'options': {'queue': 'analytics'}
    },
    
    'daily-data-cleanup': {
        'task': 'schedulers.data_tasks.cleanup_old_data',
        'schedule': crontab(hour=2, minute=0),  # 2 AM KST
        'options': {'queue': 'monitoring'}
    },
    
    # Monitoring and health checks
    'system-health-check': {
        'task': 'schedulers.data_tasks.system_health_check',
        'schedule': timedelta(minutes=10),
        'options': {'queue': 'monitoring'}
    },
    
    'api-connection-check': {
        'task': 'schedulers.data_tasks.check_api_connections',
        'schedule': timedelta(minutes=5),
        'options': {'queue': 'monitoring'}
    },
    
    # Weekly tasks
    'weekly-model-retrain': {
        'task': 'schedulers.data_tasks.retrain_ml_models',
        'schedule': crontab(hour=3, minute=0, day_of_week=0),  # Sunday 3 AM
        'options': {'queue': 'analytics'}
    },
    
    'weekly-backup': {
        'task': 'schedulers.data_tasks.backup_database',
        'schedule': crontab(hour=4, minute=0, day_of_week=0),  # Sunday 4 AM
        'options': {'queue': 'monitoring'}
    },
}

# Task configuration for beat scheduler
app.conf.beat_scheduler = 'django_celery_beat.schedulers:DatabaseScheduler'

# Import tasks to register them
from . import data_tasks

if __name__ == '__main__':
    app.start()