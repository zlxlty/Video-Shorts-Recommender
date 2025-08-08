#!/bin/bash

# Incremental data generation script
# This script generates the full dataset in manageable chunks

echo "Starting incremental data generation..."
echo "This will generate 10,000 users, 50,000 videos, and 1,000,000 interactions"
echo ""

# Check if state file exists to determine if we're resuming
STATE_FILE="data/raw/.generation_state.pkl"

if [ -f "$STATE_FILE" ]; then
    echo "Found existing state file. Resuming generation..."
else
    echo "Starting fresh generation..."
fi

# Run the batch generator with a timeout
# It will automatically save state and can be resumed
while [ -f "$STATE_FILE" ] || [ ! -f "data/raw/interactions.parquet" ]; do
    echo ""
    echo "Running generation batch (will timeout after 30 seconds)..."
    
    # Run with timeout - it will save state automatically
    timeout 30 uv run python src/data/generate_synthetic_batch.py --batch-size 2000
    
    EXIT_CODE=$?
    
    if [ $EXIT_CODE -eq 124 ]; then
        echo "Batch timed out (expected). Continuing..."
    elif [ $EXIT_CODE -eq 0 ]; then
        echo "Generation completed successfully!"
        break
    else
        echo "Error occurred. Exit code: $EXIT_CODE"
        exit 1
    fi
    
    # Small pause before next batch
    sleep 1
done

echo ""
echo "Data generation complete!"
echo ""

# Show statistics
if [ -f "data/raw/data_statistics.yaml" ]; then
    echo "Dataset statistics:"
    cat data/raw/data_statistics.yaml
fi