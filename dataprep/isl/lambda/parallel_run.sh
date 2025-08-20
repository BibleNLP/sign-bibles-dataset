#!/bin/bash

INPUT_FILE="input_list.txt"

LOG_DIR="/mnt/share/logs"
SUCCESS_LOG="$LOG_DIR/success.log"
FAIL_LOG="$LOG_DIR/fail.log"

mkdir -p "$LOG_DIR"
: > "$SUCCESS_LOG"  # Clear old logs
: > "$FAIL_LOG"


# Function to run a job and log result
run_job() {
    VIDEO_ID="$1"
    if python3 scripts/dwpose_processing.py "$VIDEO_ID"; then
        echo -e "$VIDEO_ID" >> "$SUCCESS_LOG"
    else
        echo -e "$VIDEO_ID" >> "$FAIL_LOG"
    fi
}

# Run jobs sequentially from input file
while IFS= read -r VIDEO_ID || [ -n "$VIDEO_ID" ]; do
    run_job "$VIDEO_ID"
done < "$INPUT_FILE"



# Retry failed jobs if any
if [ -s "$FAIL_LOG" ]; then
    echo "Retrying failed jobs..."
    cp "$FAIL_LOG" retry.txt
    > "$FAIL_LOG"  # Clear old failures

    while IFS= read -r VIDEO_ID || [ -n "$VIDEO_ID" ]; do
        run_job "$VIDEO_ID"
    done < retry.txt
fi
