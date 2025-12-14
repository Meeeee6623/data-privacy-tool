#!/bin/bash

# Script to submit Slurm jobs for each predictions file

for file in /usr/project/xtmp/bsc32/droid_yoloe/*/predictions.txt; do
    sbatch --job-name="gemini_ocr_$(basename "$file" .txt)" \
           --output="slurm_%j.out" \
           --error="slurm_%j.err" \
           --cpus-per-task=16 \
           --mem=20G \
           --wrap="/home/users/bsc32/dataset-mining/.venv/bin/python /home/users/bsc32/dataset-mining/ocr/make_gemini_ocr_batch.py \"$file\""
done