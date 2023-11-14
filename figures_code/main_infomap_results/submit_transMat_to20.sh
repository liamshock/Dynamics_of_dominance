#!/bin/bash

loadpath="/path/to/tetW_tetL_dpp_master_tseries.h5"
numInfoTrials=50
k=20
dppMax=20

# load anaconda to get our desired python
source ~/anaconda3/etc/profile.d/conda.sh
conda activate analysis

# main code
python transMat_and_infomap.py ${loadpath} ${numInfoTrials} ${k} ${dppMax}

