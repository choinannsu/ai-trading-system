#!/bin/bash

# Stop Celery Services Script

echo "=== Stopping Celery Services ==="

# Stop Celery processes
echo "Stopping Celery Worker..."
pkill -f "celery.*worker" || echo "No worker processes found"

echo "Stopping Celery Beat..."
pkill -f "celery.*beat" || echo "No beat processes found"

echo "Stopping Celery Flower..."
pkill -f "celery.*flower" || echo "No flower processes found"

# Clean up any remaining celery processes
echo "Cleaning up remaining processes..."
pkill -f "celery" || echo "No remaining celery processes"

echo ""
echo "=== Celery Services Stopped ==="