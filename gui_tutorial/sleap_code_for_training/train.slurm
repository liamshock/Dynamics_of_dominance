#!/bin/bash -l
#SBATCH -t 72:00:00
#SBATCH -c 36
#SBATCH --output=out/out-%j.out
#SBATCH --error=out/err-%j.out
#SBATCH --partition=gpu
#SBATCH --gres=gpu:V100:1

conda activate sleap_env
export QT_QPA_PLATFORM=minimal
export QT_DEBUG_PLUGINS=1
python TEMP_fish_pipeline/training/train_sleap.py "$1" "$2" "$3"