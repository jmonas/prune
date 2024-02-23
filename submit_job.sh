#!/bin/bash

# Usage: ./submit_job.sh job_name model_class model dataset pruner compression

# Assign arguments to variables
JOB_NAME=$1
MODEL_CLASS=$2
MODEL=$3
DATASET=$4
PRUNER=$5
COMPRESSION=$6

# Create a temporary job script file
TMP_JOB_SCRIPT="temp_job_$JOB_NAME.slurm"

# Replace placeholders in the job script template and write to temporary file
sed "s/{job_name}/$JOB_NAME/;s/{model_class}/$MODEL_CLASS/;s/{model}/$MODEL/;s/{dataset}/$DATASET/;s/{pruner}/$PRUNER/;s/{compression}/$COMPRESSION/" job.slurm > $TMP_JOB_SCRIPT

# Submit the job
sbatch $TMP_JOB_SCRIPT

# Optional: remove the temporary job script if not needed
rm $TMP_JOB_SCRIPT
