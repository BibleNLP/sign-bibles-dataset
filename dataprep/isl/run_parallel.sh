#!/bin/bash

INPUT_FILE="mat_list.txt"
LOG_DIR="logs"
SUCCESS_LOG="$LOG_DIR/success.log"
FAIL_LOG="$LOG_DIR/fail.log"

mkdir -p "$LOG_DIR"
: > "$SUCCESS_LOG"  # Clear old logs
: > "$FAIL_LOG"

# Export needed for GNU parallel to use these variables
export SUCCESS_LOG FAIL_LOG

run_job() {
    VIDEO_ID="$1"
    VIDEO_PATH="$2"

    if python3 isl_gospel_processing.py "$VIDEO_ID" "$VIDEO_PATH"; then
        echo -e "$VIDEO_ID\t$VIDEO_PATH" >> "$SUCCESS_LOG"
    else
        echo -e "$VIDEO_ID\t$VIDEO_PATH" >> "$FAIL_LOG"
    fi
}

export -f run_job

# Run in parallel
parallel -j 4 --colsep '\t' run_job {1} {2} :::: "$INPUT_FILE"


if [ -s logs/fail.log ]; then
    echo "Retrying failed jobs..."
    cp logs/fail.log retry.txt
    > logs/fail.log  # Clear old failures

    parallel -j 4 --colsep '\t' run_job {1} {2} :::: retry.txt
fi
