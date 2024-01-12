#!/bin/bash

## Make a movie from a single camera view with bodypoints drawn on


# ---------------------------------------------------------------------------------------------------------#
#                      Inputs
# ---------------------------------------------------------------------------------------------------------#

# h5 file containing "tracks_3D_smooth" dataset
trackPath="/media/liam/hd1/fighting_data/tracking_results/FishTank20200130_153857_tracking_results.h5"

# the winner gets red, the loser gets blue
winnerIdx=1

# the top directory containing the exp videos from all camera views
experiment_path="/media/liam/hd1/labelling/complete_shortfin_experiment/FishTank20200130_153857/"

# the frame-range and cam view
f0=0
fE=1000
camIdx=0

# save folder for the output
saveFolder="/media/liam/hd1/making_animations/v1"

# save name of the output movie
movName="camview.mp4"

# params of the output movie
fps=100
movHeight=480
movWidth=640



# ---------------------------------------------------------------------------------------------------------#
# ---------------------------------------------------------------------------------------------------------#


source ~/anaconda3/etc/profile.d/conda.sh
conda activate analysis

echo "Making the movie ..."

python draw_imageCoords.py $trackPath $winnerIdx $experiment_path \
                           $f0 $fE $camIdx \
                           $saveFolder $movName $fps \
                           $movHeight $movWidth > /dev/null

echo "Done!"
echo "See" $savefolderPath
