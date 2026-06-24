#!/bin/bash
DATA_DIR="data-0418"
INCIDENT_SCHEDULE="schedules/incident_schedule_share7d_n6_10day.json"
export TZ="America/New_York"

mkdir -p logs

uv run python run_incident_schedule.py \
  --schedule $INCIDENT_SCHEDULE \
  --data-dir $DATA_DIR \
  --snapshot-interval 5 \
  >> logs/run_scheduled_$(date +%Y-%m-%d_%H-%M-%S).log 2>&1