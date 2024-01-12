#!/bin/bash

## Make an animation of the tracking results


# ---------------------------------------------------------------------------------------------------------#
#                      Inputs
# ---------------------------------------------------------------------------------------------------------#

# h5 file containing "tracks_3D_smooth" dataset
loadpath="/media/liam/hd1/fighting_data/tracking_results/FishTank20200130_153857_tracking_results.h5" 

# the winner gets red, the loser gets blue
winnerIdx=1

# save folder for the output
savefolderPath="/media/liam/hd1/making_animations/v1/"
movieName="3D.mp4"

# bool to make the combined movie (1=True, anything else is False),
# and bool to delete all the short movies after gluing together, if we glued together
make_concatenation_movie=1
delete_individual_movies=1

# for parsing frames up into chunks
parstartFrame=0    # the start frame for making the movie
parstopFrame=1000    # the stop frame for the movie, -1 means up to last frame
step=100          # Each parallel job will make a movie with this many frames

# the number of processors to use for the parallel step
numProcessors=40

# the output movie frame rate
outmovie_fps=100


# ---------------------------------------------------------------------------------------------------------#
# ---------------------------------------------------------------------------------------------------------#


source ~/anaconda3/etc/profile.d/conda.sh
conda activate analysis

echo "Making the movie ..."

python animate_trajectory.py $loadpath $winnerIdx $savefolderPath $movieName \
                             $make_concatenation_movie $delete_individual_movies \
                             $parstartFrame $parstopFrame $step \
                             $numProcessors $outmovie_fps > log.txt #/dev/null

echo "Done!"
echo "See" $savefolderPath
