#!/bin/bash

## Make a movie from a single camera view with bodypoints drawn on

# -------------------------------------------------------------------#
#                      Inputs
## ------------------------------------------------------------------#

# h5 file containing "tracks_3D_smooth" & "tracks_imCoords_raw" datasets
loadPath="/media/liam/hd1/fighting_data/tracking_results/FishTank20200130_153857_tracking_results.h5"

# the winner gets red, the loser gets blue
winnerIdx=1

# save folder for the output
saveFolder="/media/liam/hd1/making_animations/v1_3"

# the frame-range and cam view
f0=169400
fE=181400
camIdx=0

# ----------------------------------------------#

# the save file names
movName3D="3D.mp4"
movNameCam="camview.mp4"
movNameComb="3D_and_camview.mp4"


# the number of frames to each job in the parallel processing in animate_trajectory
step=100

# the number of processors to use for the parallel step
numProcessors=40

# params of the output movie (this has to be the 3D animator output size)
fps=100
movHeight=480
movWidth=640

# bool to make the combined movie (1=True, anything else is False),
# and bool to delete all the short movies after gluing together, if we glued together
make_concatenation_movie=1
delete_individual_movies=1

# the top directory containing the exp videos from all camera views
experiment_path="/media/liam/hd1/labelling/complete_shortfin_experiment/FishTank20200130_153857/"






# -----------------------------------------------------------------------#
# -----------------------------------------------------------------------#

source ~/anaconda3/etc/profile.d/conda.sh
conda activate analysis


echo "Making the 3D movie ..."
python animate_trajectory.py $loadPath $winnerIdx $saveFolder $movName3D \
                             $make_concatenation_movie $delete_individual_movies \
                             $f0 $fE $step \
                             $numProcessors $fps > /dev/null
echo "Done!"
echo ""


echo "Making the camview movie ..."
python draw_imageCoords.py $loadPath $winnerIdx $experiment_path \
                           $f0 $fE $camIdx \
                           $saveFolder $movNameCam $fps \
                           $movHeight $movWidth 
echo "Done!"
echo ""


echo "Combining the 2 movies ..."
path3DMov=$saveFolder/$movName3D
pathCamMov=$saveFolder/$movNameCam
outMovPath=$saveFolder/$movNameComb
ffmpeg -i $path3DMov -i $pathCamMov -filter_complex "hstack,format=yuv420p" -c:v libx264 -crf 18 $outMovPath
echo "Done!"
echo ""











