#!/bin/bash

INPUT_FILE="input_list.txt"

LOG_DIR="/home/ubuntu/isl-dataset-prep/logs"
SUCCESS_LOG="$LOG_DIR/success.log"
FAIL_LOG="$LOG_DIR/fail.log"

mkdir -p "$LOG_DIR"
: > "$SUCCESS_LOG"  # Clear old logs
: > "$FAIL_LOG"

# Export needed for GNU parallel to use these variables
export SUCCESS_LOG FAIL_LOG

run_job() {
    VIDEO_ID="$1"

    if python3 scripts/dwpose_processing.py "$VIDEO_ID"; then
        echo -e "$VIDEO_ID" >> "$SUCCESS_LOG"
    else
        echo -e "$VIDEO_ID" >> "$FAIL_LOG"
    fi
}

export -f run_job

# Run in parallel
parallel -j 3 run_job {1} :::: "$INPUT_FILE"


if [ -s /home/ubuntu/isl-dataset-prep/logs/fail.log ]; then
    echo "Retrying failed jobs..."
    cp /home/ubuntu/isl-dataset-prep/logs/fail.log retry.txt
    > /home/ubuntu/isl-dataset-prep/logs/fail.log  # Clear old failures

    parallel -j 3 run_job {1} :::: retry.txt
fi
