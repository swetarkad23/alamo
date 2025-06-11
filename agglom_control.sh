#!/bin/bash

# Gamma values (must match the sbatch script)
GAMMA_VALUES=(1e10 9e9 8e9 7e9 6e9 5e9 4e9 3e9 2e9 1e9)

# File to store job ID
JOB_ID_FILE=".agglom_job_id"

# Function to submit jobs
submit_jobs() {
    # Submit the job array
    echo "Submitting job array..."
    JOB_ID=$(sbatch --parsable agglom_run.sbatch)
    
    if [ $? -eq 0 ]; then
        echo "Successfully submitted job array with ID: $JOB_ID"
        echo $JOB_ID > $JOB_ID_FILE
        echo ""
        echo "To check job status: squeue -j $JOB_ID"
        echo "To clean up logs after completion: $0 --clean-up"
    else
        echo "Error: Failed to submit jobs"
        exit 1
    fi
}

# Function to clean up logs
cleanup_logs() {
    # Check if job ID file exists
    if [ ! -f "$JOB_ID_FILE" ]; then
        echo "Error: No job ID found. Did you submit jobs first?"
        exit 1
    fi
    
    JOB_ID=$(cat $JOB_ID_FILE)
    echo "Cleaning up logs for job array: $JOB_ID"
    
    # Move logs
    MOVED_COUNT=0
    for i in {0..9}; do
        GAMMA=${GAMMA_VALUES[$i]}
        OLD_LOG="tests/Agglomeration/${JOB_ID}-${i}.log"
        NEW_LOG="tests/Agglomeration/output-${GAMMA}/agglom-alamo-${GAMMA}-${JOB_ID}-${i}.log"
        
        if [ -f "$OLD_LOG" ]; then
            mv "$OLD_LOG" "$NEW_LOG"
            echo "Moved: gamma=${GAMMA} log to $NEW_LOG"
            ((MOVED_COUNT++))
        else
            echo "Warning: Log for gamma=${GAMMA} not found at $OLD_LOG"
        fi
    done
    
    echo ""
    echo "Moved $MOVED_COUNT log files"
    
    # Remove job ID file
    rm -f $JOB_ID_FILE
}

# Main script logic
case "$1" in
    --clean-up|--cleanup|-c)
        cleanup_logs
        ;;
    --help|-h)
        echo "Usage: $0 [OPTION]"
        echo ""
        echo "Options:"
        echo "  (no option)     Submit the job array"
        echo "  --clean-up, -c  Move log files to appropriate directories"
        echo "  --help, -h      Show this help message"
        ;;
    "")
        submit_jobs
        ;;
    *)
        echo "Error: Unknown option '$1'"
        echo "Use '$0 --help' for usage information"
        exit 1
        ;;
esac
