#!/bin/bash

cd /app/src

# Run FastAPI in background
python app.py &

# Run Celery worker
celery -A tasks.celery_app worker --loglevel=info

