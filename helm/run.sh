#!/bin/bash -l
#SBATCH --job-name=HELM_training
#SBATCH --output=script_logs/%x_%j_output.log
#SBATCH --error=script_logs/%x_%j_error.log
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sj1030@rutgers.edu
#SBATCH -G 4 --mem=80G


echo "Starting job $SLURM_JOB_ID"
set -e

cd ~/Projects/helm

source /common/home/sj1030/miniconda3/etc/profile.d/conda.sh

# keep-job 80

# echo "Starting job $SLURM_JOB_ID"



# conda init bash

source ~/.bashrc


# if [ "$SLURM_JOB_NAME" == "MemoryMaze" ]; then
#     echo "Activating conda env gym for MemoryMaze"
#     conda activate gym || { echo "Failed to activate conda environment"; exit 1; }
# else
#     echo "Activating conda env memory-gym"
#     conda activate memory-gym || { echo "Failed to activate conda environment"; exit 1; }
# fi

conda activate helm || { echo "Failed to activate conda environment"; exit 1; }

gpustat

# echo "Starting job $SLURM_JOB_ID"
# Store the start time
START_TIME=$(date +%s)

# Function to print time elapsed every 5 minutes
print_time_elapsed() {
    while :
    do
        # Get the current time
        CURRENT_TIME=$(date +%s)
        
        # Calculate the time elapsed
        TIME_ELAPSED=$((CURRENT_TIME - START_TIME))
        
        # Convert the time elapsed to hours, minutes, and seconds
        HOURS=$((TIME_ELAPSED / 3600))
        MINUTES=$(( (TIME_ELAPSED % 3600) / 60 ))
        SECONDS=$((TIME_ELAPSED % 60))
        
        # Print the time elapsed
        echo "Time elapsed: $HOURS hours, $MINUTES minutes, $SECONDS seconds"
        
        # Print GPU usage
        gpustat
        
        # Sleep for 5 minutes
        sleep 120
    done
}

# Start the time elapsed function in the background
print_time_elapsed &

# Save the process ID of the background function
TIME_ELAPSED_PID=$!

CUDA_VISIBLE_DEVICES=0 python main.py --var env=MemoryMaze> script_logs/MemoryMaze_${SLURM_JOB_ID}_python.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 python main.py --var env=MortarMayhem> script_logs/MortarMayhem_${SLURM_JOB_ID}_python.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 python main.py --var env=SearingSpotlights> script_logs/SearingSpotlights_${SLURM_JOB_ID}_python.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 python main.py --var env=MysteryPath> script_logs/MysteryPath_${SLURM_JOB_ID}_python.log 2>&1 &
# script -f script_logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_python.log -c "python -m sb3.main ${SLURM_JOB_NAME}" > /dev/null 2>&1 &


echo $!
wait $!

gpustat

# Kill the time elapsed function
kill $TIME_ELAPSED_PID