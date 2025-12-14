#!/bin/bash
#SBATCH --job-name=yoloe_batch_classify_${base}
#SBATCH --output=logs/yoloe_single_%j.out
#SBATCH --error=logs/yoloe_single_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=compsci-gpu
#SBATCH --constraint=a5000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

cd ~/dataset-mining
source .venv/bin/activate
python predict_yoloe.py
