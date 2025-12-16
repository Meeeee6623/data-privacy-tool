#!/bin/bash

# Script to submit Slurm jobs for cleaning up OCR JSON outputs for each folder

for dir in /usr/project/xtmp/bsc32/droid_yoloe/*/; do
    if [ -d "$dir" ]; then
        folder_name=$(basename "$dir")
        sbatch --job-name="cleanup_ocr_${folder_name}" \
               --output="slurm_${folder_name}_cleanup_%j.out" \
               --error="slurm_${folder_name}_cleanup_%j.err" \
               --cpus-per-task=8 \
               --mem=16G \
               --wrap="/home/users/bsc32/dataset-mining/.venv/bin/python /home/users/bsc32/dataset-mining/ocr/cleanup_ocr.py \"$folder_name\""
    fi
done
