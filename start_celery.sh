#!/bin/bash

# Celery Worker and Beat Startup Script for AI Trading System

set -e

echo "=== Starting AI Trading System Celery Services ==="

# Check if Redis is running
if ! redis-cli ping > /dev/null 2>&1; then
    echo "Error: Redis is not running. Please start Redis first:"
    echo "  brew services start redis  # macOS"
    echo "  sudo systemctl start redis  # Linux"
    exit 1
fi

echo "✓ Redis is running"

# Set environment variables
export CELERY_BROKER_URL="${CELERY_BROKER_URL:-redis://localhost:6379/0}"
export CELERY_RESULT_BACKEND="${CELERY_RESULT_BACKEND:-redis://localhost:6379/1}"

# Create logs directory
mkdir -p logs/celery

echo "Starting Celery services..."

# Start Celery Worker
echo "Starting Celery Worker..."
celery -A infrastructure.schedulers.celery_app worker \
    --loglevel=info \
    --concurrency=4 \
    --queues=default,market_data,news,analytics,monitoring \
    --logfile=logs/celery/worker.log \
    --detach

# Start Celery Beat (scheduler)
echo "Starting Celery Beat..."
celery -A infrastructure.schedulers.celery_app beat \
    --loglevel=info \
    --logfile=logs/celery/beat.log \
    --detach

# Start Celery Flower (monitoring) - optional
if command -v celery &> /dev/null; then
    echo "Starting Celery Flower (monitoring)..."
    celery -A infrastructure.schedulers.celery_app flower \
        --port=5555 \
        --logfile=logs/celery/flower.log \
        --detach || echo "Flower not available, skipping..."
fi

echo ""
echo "=== Celery Services Started ==="
echo "Worker: Running with 4 concurrent processes"
echo "Beat: Scheduled tasks active"
echo "Flower: Available at http://localhost:5555 (if installed)"
echo ""
echo "Logs:"
echo "  Worker: logs/celery/worker.log"
echo "  Beat: logs/celery/beat.log"
echo "  Flower: logs/celery/flower.log"
echo ""
echo "To stop services:"
echo "  ./stop_celery.sh"
echo ""
echo "To view logs:"
echo "  tail -f logs/celery/worker.log"
echo "  tail -f logs/celery/beat.log"