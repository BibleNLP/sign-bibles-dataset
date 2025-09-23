#!/bin/bash

INPUT_FILE="/content/input_list.txt"
DATA_PATH="/content/isl_gospel_videos"

LOG_DIR="/content/logs"
SUCCESS_LOG="$LOG_DIR/success.log"
FAIL_LOG="$LOG_DIR/fail.log"

mkdir -p "$LOG_DIR"
: > "$SUCCESS_LOG"  # Clear old logs
: > "$FAIL_LOG"

# Export needed for GNU parallel to use these variables
export SUCCESS_LOG FAIL_LOG

run_job() {
    VIDEO_ID="$1"
    DATA_PATH="$2"
    echo "Processing $VIDEO_ID in $DATA_PATH";

    if python3 mp_processing.py "$VIDEO_ID" "$DATA_PATH"; then
        echo -e "$VIDEO_ID\t$DATA_PATH" >> "$SUCCESS_LOG"
    else
        echo -e "$VIDEO_ID\t$DATA_PATH" >> "$FAIL_LOG"
    fi
}

export -f run_job

# Run in parallel
parallel -j 40 run_job {1} "$DATA_PATH" :::: "$INPUT_FILE"


if [ -s /content/logs/fail.log ]; then
    echo "Retrying failed jobs..."
    cp /content/logs/fail.log retry.txt
    > /content/logs/fail.log  # Clear old failures

    parallel -j 40 run_job {1} "$DATA_PATH" :::: retry.txt
fi
