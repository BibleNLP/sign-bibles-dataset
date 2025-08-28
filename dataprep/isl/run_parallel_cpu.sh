#!/bin/bash

INPUT_FILE="gospel_list.txt"
OUT_PATH="/mnt/share/ISLGospel_processed/"

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
    PROCCESSED_PATH="$3"

    if python3 isl_gospel_processing.py "$VIDEO_ID" "$VIDEO_PATH" "$PROCCESSED_PATH"; then
        echo -e "$VIDEO_ID\t$VIDEO_PATH" >> "$SUCCESS_LOG"
    else
        echo -e "$VIDEO_ID\t$VIDEO_PATH" >> "$FAIL_LOG"
    fi
}

export -f run_job

# Run in parallel
parallel --tmpdir /mnt/share/temp -j 16 --colsep '\t' run_job {1} {2} "$OUT_PATH" :::: "$INPUT_FILE"


if [ -s logs/fail.log ]; then
    echo "Retrying failed jobs..."
    cp logs/fail.log retry.txt
    > logs/fail.log  # Clear old failures

    parallel --tmpdir /mnt/share/temp -j 16 --colsep '\t' run_job {1} {2} "$OUT_PATH" :::: retry.txt
fi
