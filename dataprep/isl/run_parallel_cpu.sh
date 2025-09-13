#!/bin/bash

INPUT_FILE="/content/input_list.txt"
OUT_PATH="/content/dataprep_output"

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
    VIDEO_PATH="$2"
    PROCCESSED_PATH="$3"

    if python3 isl_bible_processing.py "$VIDEO_ID" "$VIDEO_PATH" "$PROCCESSED_PATH"; then
        echo -e "$VIDEO_ID\t$VIDEO_PATH" >> "$SUCCESS_LOG"
    else
        echo -e "$VIDEO_ID\t$VIDEO_PATH" >> "$FAIL_LOG"
    fi
}

export -f run_job

# Run in parallel
parallel --tmpdir /content/temp -j 90 --colsep '\t' run_job {1} {2} "$OUT_PATH" :::: "$INPUT_FILE"


if [ -s /content/logs/fail.log ]; then
    echo "Retrying failed jobs..."
    cp /content/logs/fail.log retry.txt
    > /content/logs/fail.log  # Clear old failures

    parallel --tmpdir /mnt/share/temp -j 50 --colsep '\t' run_job {1} {2} "$OUT_PATH" :::: retry.txt
fi
