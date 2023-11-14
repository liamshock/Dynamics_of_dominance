#!/bin/bash

# This is a script for the transition matrix, eigvals, and performing infomap,
# and the thetaW_thetaL_dpp data, as a slurm array over tau values.
#
# See "tau_list" inside transMat_and_infomap.py, to see how tau is parsed
# from the slurm_array_task_id

#SBATCH --job-name=tetW_tetL_dpp_bins_c2
#SBATCH --partition=compute
#SBATCH --array=1-30
#SBATCH --time=0-05:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=30G
#SBATCH --output=/flash/StephensU/Liam/tausweep_20220404/logs/%A-%a.out

loadpath="/flash/StephensU/Liam/tausweep_20220404/tetW_tetL_dpp_master_tseries.h5"
numInfoTrials=50
k=20
bin_choice=2

# load anaconda to get our desired python
source ~/anaconda3/etc/profile.d/conda.sh
conda activate analysis_env

# main code
python transMat_and_infomap.py ${loadpath} ${numInfoTrials} ${k} ${bin_choice} ${SLURM_ARRAY_TASK_ID}

