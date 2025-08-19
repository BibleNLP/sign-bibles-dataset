#!/bin/bash

INPUT_FILE="input_list.txt"

LOG_DIR="/my_logs/logs"
SUCCESS_LOG="$LOG_DIR/success.log"
FAIL_LOG="$LOG_DIR/fail.log"

mkdir -p "$LOG_DIR"
: > "$SUCCESS_LOG"  # Clear old logs
: > "$FAIL_LOG"

# Export needed for GNU parallel to use these variables
export SUCCESS_LOG FAIL_LOG

run_job() {
    VIDEO_ID="$1"

    if python3 dwpose_processing.py "$VIDEO_ID"; then
        echo -e "$VIDEO_ID" >> "$SUCCESS_LOG"
    else
        echo -e "$VIDEO_ID" >> "$FAIL_LOG"
    fi
}

export -f run_job

# Run in parallel
parallel -j 3 run_job {1} :::: "$INPUT_FILE"


if [ -s /my_logs/logs/fail.log ]; then
    echo "Retrying failed jobs..."
    cp /my_logs/logs/fail.log retry.txt
    > /my_logs/logs/fail.log  # Clear old failures

    parallel -j 3 run_job {1} :::: retry.txt
fi
