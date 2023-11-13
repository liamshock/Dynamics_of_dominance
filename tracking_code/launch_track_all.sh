#!/bin/bash

## This is a script for launching the 3D tracking of the paper dataset


parStep=1000             # the number of frames we want to process in each parallel chunk
numProcessors=38         # the number of processors we want to use for our parallel calculation

mean_reg_skel_thresh=10   # the threshold on accepting cross-camera registrations
head_pec_thresh=1.2              # the threshold on the head_pec distance
pec_tail_thresh=2.5              # the threshold on the pec_tail distance
idtracks_to_sleap_thresh=12.5    # the threshold for registering xy_pecs with idtracks

interp_polyOrd=1  # the order of the polynomial used for interpolation
interp_limit=5    # the maximum number of frames to interpolate over
savgol_win=9      # the number of frames for the Savitzky-Golay filter
savgol_ord=2      # the polynomial order for the Savitzky-Golay filter
dt=0.01           # the frame rate of the recording

source ~/anaconda3/etc/profile.d/conda.sh
conda activate analysis

echo " Starting tracking the 22 experiments ... "
echo


input_path="/media/liam/hd1/fighting_data/FishTank20200127_143538/FishTank20200127_143538_sLEAP_and_idtracks.h5"
expName="FishTank20200127_143538"
output_h5_path="/media/liam/hd1/fighting_data/tracking_results/FishTank20200127_143538_tracking_results.h5"
output_csv_path="/media/liam/hd1/fighting_data/csv_trajectories/FishTank20200127_143538.csv"
calibrationFolderPath="./20200120_calibration/"
python track_experiment.py $input_path $expName $output_h5_path $output_csv_path \
                           $calibrationFolderPath \
                           $parStep $numProcessors \
                           $mean_reg_skel_thresh $head_pec_thresh $pec_tail_thresh \
                           $idtracks_to_sleap_thresh $interp_polyOrd $interp_limit \
                           $savgol_win $savgol_ord $dt >> log_track.txt
echo $expName "done"
echo


input_path="/media/liam/hd1/fighting_data/FishTank20200129_140656/FishTank20200129_140656_sLEAP_and_idtracks.h5"
expName="FishTank20200129_140656"
output_h5_path="/media/liam/hd1/fighting_data/tracking_results/FishTank20200129_140656_tracking_results.h5"
output_csv_path="/media/liam/hd1/fighting_data/csv_trajectories/FishTank20200129_140656.csv"
calibrationFolderPath="./20200120_calibration/"
python track_experiment.py $input_path $expName $output_h5_path $output_csv_path \
                           $calibrationFolderPath \
                           $parStep $numProcessors \
                           $mean_reg_skel_thresh $head_pec_thresh $pec_tail_thresh \
                           $idtracks_to_sleap_thresh $interp_polyOrd $interp_limit \
                           $savgol_win $savgol_ord $dt >> log_track.txt
echo $expName "done"
echo



input_path="/media/liam/hd1/fighting_data/FishTank20200130_153857/FishTank20200130_153857_sLEAP_and_idtracks.h5"
expName="FishTank20200130_153857"
output_h5_path="/media/liam/hd1/fighting_data/tracking_results/FishTank20200130_153857_tracking_results.h5"
output_csv_path="/media/liam/hd1/fighting_data/csv_trajectories/FishTank20200130_153857.csv"
calibrationFolderPath="./20200120_calibration/"
python track_experiment.py $input_path $expName $output_h5_path $output_csv_path \
                           $calibrationFolderPath \
                           $parStep $numProcessors \
                           $mean_reg_skel_thresh $head_pec_thresh $pec_tail_thresh \
                           $idtracks_to_sleap_thresh $interp_polyOrd $interp_limit \
                           $savgol_win $savgol_ord $dt >> log_track.txt
echo $expName "done"
echo

input_path="/media/liam/hd1/fighting_data/FishTank20200130_181614/FishTank20200130_181614_sLEAP_and_idtracks.h5"
expName="FishTank20200130_181614"
output_h5_path="/media/liam/hd1/fighting_data/tracking_results/FishTank20200130_181614_tracking_results.h5"
output_csv_path="/media/liam/hd1/fighting_data/csv_trajectories/FishTank20200130_181614.csv"
calibrationFolderPath="./20200120_calibration/"
python track_experiment.py $input_path $expName $output_h5_path $output_csv_path \
                           $calibrationFolderPath \
                           $parStep $numProcessors \
                           $mean_reg_skel_thresh $head_pec_thresh $pec_tail_thresh \
                           $idtracks_to_sleap_thresh $interp_polyOrd $interp_limit \
                           $savgol_win $savgol_ord $dt >> log_track.txt
echo $expName "done"
echo

input_path="/media/liam/hd1/fighting_data/FishTank20200207_161445/FishTank20200207_161445_sLEAP_and_idtracks.h5"
expName="FishTank20200207_161445"
output_h5_path="/media/liam/hd1/fighting_data/tracking_results/FishTank20200207_161445_tracking_results.h5"
output_csv_path="/media/liam/hd1/fighting_data/csv_trajectories/FishTank20200207_161445.csv"
calibrationFolderPath="./20200120_calibration/"
python track_experiment.py $input_path $expName $output_h5_path $output_csv_path \
                           $calibrationFolderPath \
                           $parStep $numProcessors \
                           $mean_reg_skel_thresh $head_pec_thresh $pec_tail_thresh \
                           $idtracks_to_sleap_thresh $interp_polyOrd $interp_limit \
                           $savgol_win $savgol_ord $dt >> log_track.txt
echo $expName "done"
echo



input_path="/media/liam/hd1/fighting_data/FishTank20200213_154940/FishTank20200213_154940_sLEAP_and_idtracks.h5"
expName="FishTank20200213_154940"
output_h5_path="/media/liam/hd1/fighting_data/tracking_results/FishTank20200213_154940_tracking_results.h5"
output_csv_path="/media/liam/hd1/fighting_data/csv_trajectories/FishTank20200213_154940.csv"
calibrationFolderPath="./20200120_calibration/"
python track_experiment.py $input_path $expName $output_h5_path $output_csv_path \
                           $calibrationFolderPath \
                           $parStep $numProcessors \
                           $mean_reg_skel_thresh $head_pec_thresh $pec_tail_thresh \
                           $idtracks_to_sleap_thresh $interp_polyOrd $interp_limit \
                           $savgol_win $savgol_ord $dt >> log_track.txt
echo $expName "done"
echo



input_path="/media/liam/hd1/fighting_data/FishTank20200214_153519/FishTank20200214_153519_sLEAP_and_idtracks.h5"
expName="FishTank20200214_153519"
output_h5_path="/media/liam/hd1/fighting_data/tracking_results/FishTank20200214_153519_tracking_results.h5"
output_csv_path="/media/liam/hd1/fighting_data/csv_trajectories/FishTank20200214_153519.csv"
calibrationFolderPath="./20200120_calibration/"
python track_experiment.py $input_path $expName $output_h5_path $output_csv_path \
                           $calibrationFolderPath \
                           $parStep $numProcessors \
                           $mean_reg_skel_thresh $head_pec_thresh $pec_tail_thresh \
                           $idtracks_to_sleap_thresh $interp_polyOrd $interp_limit \
                           $savgol_win $savgol_ord $dt >> log_track.txt
echo $expName "done"
echo



input_path="/media/liam/hd1/fighting_data/FishTank20200217_160052/FishTank20200217_160052_sLEAP_and_idtracks.h5"
expName="FishTank20200217_160052"
output_h5_path="/media/liam/hd1/fighting_data/tracking_results/FishTank20200217_160052_tracking_results.h5"
output_csv_path="/media/liam/hd1/fighting_data/csv_trajectories/FishTank20200217_160052.csv"
calibrationFolderPath="./20200120_calibration/"
python track_experiment.py $input_path $expName $output_h5_path $output_csv_path \
                           $calibrationFolderPath \
                           $parStep $numProcessors \
                           $mean_reg_skel_thresh $head_pec_thresh $pec_tail_thresh \
                           $idtracks_to_sleap_thresh $interp_polyOrd $interp_limit \
                           $savgol_win $savgol_ord $dt >> log_track.txt
echo $expName "done"
echo

input_path="/media/liam/hd1/fighting_data/FishTank20200218_153008/FishTank20200218_153008_sLEAP_and_idtracks.h5"
expName="FishTank20200218_153008"
output_h5_path="/media/liam/hd1/fighting_data/tracking_results/FishTank20200218_153008_tracking_results.h5"
output_csv_path="/media/liam/hd1/fighting_data/csv_trajectories/FishTank20200218_153008.csv"
calibrationFolderPath="./20200120_calibration/"
python track_experiment.py $input_path $expName $output_h5_path $output_csv_path \
                           $calibrationFolderPath \
                           $parStep $numProcessors \
                           $mean_reg_skel_thresh $head_pec_thresh $pec_tail_thresh \
                           $idtracks_to_sleap_thresh $interp_polyOrd $interp_limit \
                           $savgol_win $savgol_ord $dt >> log_track.txt
echo $expName "done"
echo

input_path="/media/liam/hd1/fighting_data/FishTank20200316_163320/FishTank20200316_163320_sLEAP_and_idtracks.h5"
expName="FishTank20200316_163320"
output_h5_path="/media/liam/hd1/fighting_data/tracking_results/FishTank20200316_163320_tracking_results.h5"
output_csv_path="/media/liam/hd1/fighting_data/csv_trajectories/FishTank20200316_163320.csv"
calibrationFolderPath="./20200120_calibration/"
python track_experiment.py $input_path $expName $output_h5_path $output_csv_path \
                           $calibrationFolderPath \
                           $parStep $numProcessors \
                           $mean_reg_skel_thresh $head_pec_thresh $pec_tail_thresh \
                           $idtracks_to_sleap_thresh $interp_polyOrd $interp_limit \
                           $savgol_win $savgol_ord $dt >> log_track.txt
echo $expName "done"
echo



input_path="/media/liam/hd1/fighting_data/FishTank20200327_154737/FishTank20200327_154737_sLEAP_and_idtracks.h5"
expName="FishTank20200327_154737"
output_h5_path="/media/liam/hd1/fighting_data/tracking_results/FishTank20200327_154737_tracking_results.h5"
output_csv_path="/media/liam/hd1/fighting_data/csv_trajectories/FishTank20200327_154737.csv"
calibrationFolderPath="./20200325_calibration/"
python track_experiment.py $input_path $expName $output_h5_path $output_csv_path \
                           $calibrationFolderPath \
                           $parStep $numProcessors \
                           $mean_reg_skel_thresh $head_pec_thresh $pec_tail_thresh \
                           $idtracks_to_sleap_thresh $interp_polyOrd $interp_limit \
                           $savgol_win $savgol_ord $dt >> log_track.txt
echo $expName "done"
echo



input_path="/media/liam/hd1/fighting_data/FishTank20200330_161100/FishTank20200330_161100_sLEAP_and_idtracks.h5"
expName="FishTank20200330_161100"
output_h5_path="/media/liam/hd1/fighting_data/tracking_results/FishTank20200330_161100_tracking_results.h5"
output_csv_path="/media/liam/hd1/fighting_data/csv_trajectories/FishTank20200330_161100.csv"
calibrationFolderPath="./20200325_calibration/"
python track_experiment.py $input_path $expName $output_h5_path $output_csv_path \
                           $calibrationFolderPath \
                           $parStep $numProcessors \
                           $mean_reg_skel_thresh $head_pec_thresh $pec_tail_thresh \
                           $idtracks_to_sleap_thresh $interp_polyOrd $interp_limit \
                           $savgol_win $savgol_ord $dt >> log_track.txt
echo $expName "done"
echo



input_path="/media/liam/hd1/fighting_data/FishTank20200331_162136/FishTank20200331_162136_sLEAP_and_idtracks.h5"
expName="FishTank20200331_162136"
output_h5_path="/media/liam/hd1/fighting_data/tracking_results/FishTank20200331_162136_tracking_results.h5"
output_csv_path="/media/liam/hd1/fighting_data/csv_trajectories/FishTank20200331_162136.csv"
calibrationFolderPath="./20200325_calibration/"
python track_experiment.py $input_path $expName $output_h5_path $output_csv_path \
                           $calibrationFolderPath \
                           $parStep $numProcessors \
                           $mean_reg_skel_thresh $head_pec_thresh $pec_tail_thresh \
                           $idtracks_to_sleap_thresh $interp_polyOrd $interp_limit \
                           $savgol_win $savgol_ord $dt >> log_track.txt
echo $expName "done"
echo

input_path="/media/liam/hd1/fighting_data/FishTank20200520_152810/FishTank20200520_152810_sLEAP_and_idtracks.h5"
expName="FishTank20200520_152810"
output_h5_path="/media/liam/hd1/fighting_data/tracking_results/FishTank20200520_152810_tracking_results.h5"
output_csv_path="/media/liam/hd1/fighting_data/csv_trajectories/FishTank20200520_152810.csv"
calibrationFolderPath="./20200325_calibration/"
python track_experiment.py $input_path $expName $output_h5_path $output_csv_path \
                           $calibrationFolderPath \
                           $parStep $numProcessors \
                           $mean_reg_skel_thresh $head_pec_thresh $pec_tail_thresh \
                           $idtracks_to_sleap_thresh $interp_polyOrd $interp_limit \
                           $savgol_win $savgol_ord $dt >> log_track.txt
echo $expName "done"
echo

input_path="/media/liam/hd1/fighting_data/FishTank20200521_154541/FishTank20200521_154541_sLEAP_and_idtracks.h5"
expName="FishTank20200521_154541"
output_h5_path="/media/liam/hd1/fighting_data/tracking_results/FishTank20200521_154541_tracking_results.h5"
output_csv_path="/media/liam/hd1/fighting_data/csv_trajectories/FishTank20200521_154541.csv"
calibrationFolderPath="./20200325_calibration/"
python track_experiment.py $input_path $expName $output_h5_path $output_csv_path \
                           $calibrationFolderPath \
                           $parStep $numProcessors \
                           $mean_reg_skel_thresh $head_pec_thresh $pec_tail_thresh \
                           $idtracks_to_sleap_thresh $interp_polyOrd $interp_limit \
                           $savgol_win $savgol_ord $dt >> log_track.txt
echo $expName "done"
echo


input_path="/media/liam/hd1/fighting_data/FishTank20200525_161602/FishTank20200525_161602_sLEAP_and_idtracks.h5"
expName="FishTank20200525_161602"
output_h5_path="/media/liam/hd1/fighting_data/tracking_results/FishTank20200525_161602_tracking_results.h5"
output_csv_path="/media/liam/hd1/fighting_data/csv_trajectories/FishTank20200525_161602.csv"
calibrationFolderPath="./20200325_calibration/"
python track_experiment.py $input_path $expName $output_h5_path $output_csv_path \
                           $calibrationFolderPath \
                           $parStep $numProcessors \
                           $mean_reg_skel_thresh $head_pec_thresh $pec_tail_thresh \
                           $idtracks_to_sleap_thresh $interp_polyOrd $interp_limit \
                           $savgol_win $savgol_ord $dt >> log_track.txt
echo $expName "done"
echo



input_path="/media/liam/hd1/fighting_data/FishTank20200526_160100/FishTank20200526_160100_sLEAP_and_idtracks.h5"
expName="FishTank20200526_160100"
output_h5_path="/media/liam/hd1/fighting_data/tracking_results/FishTank20200526_160100_tracking_results.h5"
output_csv_path="/media/liam/hd1/fighting_data/csv_trajectories/FishTank20200526_160100.csv"
calibrationFolderPath="./20200325_calibration/"
python track_experiment.py $input_path $expName $output_h5_path $output_csv_path \
                           $calibrationFolderPath \
                           $parStep $numProcessors \
                           $mean_reg_skel_thresh $head_pec_thresh $pec_tail_thresh \
                           $idtracks_to_sleap_thresh $interp_polyOrd $interp_limit \
                           $savgol_win $savgol_ord $dt >> log_track.txt
echo $expName "done"
echo



input_path="/media/liam/hd1/fighting_data/FishTank20200527_152401/FishTank20200527_152401_sLEAP_and_idtracks.h5"
expName="FishTank20200527_152401"
output_h5_path="/media/liam/hd1/fighting_data/tracking_results/FishTank20200527_152401_tracking_results.h5"
output_csv_path="/media/liam/hd1/fighting_data/csv_trajectories/FishTank20200527_152401.csv"
calibrationFolderPath="./20200325_calibration/"
python track_experiment.py $input_path $expName $output_h5_path $output_csv_path \
                           $calibrationFolderPath \
                           $parStep $numProcessors \
                           $mean_reg_skel_thresh $head_pec_thresh $pec_tail_thresh \
                           $idtracks_to_sleap_thresh $interp_polyOrd $interp_limit \
                           $savgol_win $savgol_ord $dt >> log_track.txt
echo $expName "done"
echo

input_path="/media/liam/hd1/fighting_data/FishTank20200824_151740/FishTank20200824_151740_sLEAP_and_idtracks.h5"
expName="FishTank20200824_151740"
output_h5_path="/media/liam/hd1/fighting_data/tracking_results/FishTank20200824_151740_tracking_results.h5"
output_csv_path="/media/liam/hd1/fighting_data/csv_trajectories/FishTank20200824_151740.csv"
calibrationFolderPath="./20200325_calibration/"
python track_experiment.py $input_path $expName $output_h5_path $output_csv_path \
                           $calibrationFolderPath \
                           $parStep $numProcessors \
                           $mean_reg_skel_thresh $head_pec_thresh $pec_tail_thresh \
                           $idtracks_to_sleap_thresh $interp_polyOrd $interp_limit \
                           $savgol_win $savgol_ord $dt >> log_track.txt
echo $expName "done"
echo

input_path="/media/liam/hd1/fighting_data/FishTank20200828_155504/FishTank20200828_155504_sLEAP_and_idtracks.h5"
expName="FishTank20200828_155504"
output_h5_path="/media/liam/hd1/fighting_data/tracking_results/FishTank20200828_155504_tracking_results.h5"
output_csv_path="/media/liam/hd1/fighting_data/csv_trajectories/FishTank20200828_155504.csv"
calibrationFolderPath="./20200325_calibration/"
python track_experiment.py $input_path $expName $output_h5_path $output_csv_path \
                           $calibrationFolderPath \
                           $parStep $numProcessors \
                           $mean_reg_skel_thresh $head_pec_thresh $pec_tail_thresh \
                           $idtracks_to_sleap_thresh $interp_polyOrd $interp_limit \
                           $savgol_win $savgol_ord $dt >> log_track.txt
echo $expName "done"
echo


input_path="/media/liam/hd1/fighting_data/FishTank20200902_160124/FishTank20200902_160124_sLEAP_and_idtracks.h5"
expName="FishTank20200902_160124"
output_h5_path="/media/liam/hd1/fighting_data/tracking_results/FishTank20200902_160124_tracking_results.h5"
output_csv_path="/media/liam/hd1/fighting_data/csv_trajectories/FishTank20200902_160124.csv"
calibrationFolderPath="./20200325_calibration/"
python track_experiment.py $input_path $expName $output_h5_path $output_csv_path \
                           $calibrationFolderPath \
                           $parStep $numProcessors \
                           $mean_reg_skel_thresh $head_pec_thresh $pec_tail_thresh \
                           $idtracks_to_sleap_thresh $interp_polyOrd $interp_limit \
                           $savgol_win $savgol_ord $dt >> log_track.txt
echo $expName "done"
echo



input_path="/media/liam/hd1/fighting_data/FishTank20200903_160946/FishTank20200903_160946_sLEAP_and_idtracks.h5"
expName="FishTank20200903_160946"
output_h5_path="/media/liam/hd1/fighting_data/tracking_results/FishTank20200903_160946_tracking_results.h5"
output_csv_path="/media/liam/hd1/fighting_data/csv_trajectories/FishTank20200903_160946.csv"
calibrationFolderPath="./20200325_calibration/"
python track_experiment.py $input_path $expName $output_h5_path $output_csv_path \
                           $calibrationFolderPath \
                           $parStep $numProcessors \
                           $mean_reg_skel_thresh $head_pec_thresh $pec_tail_thresh \
                           $idtracks_to_sleap_thresh $interp_polyOrd $interp_limit \
                           $savgol_win $savgol_ord $dt >> log_track.txt
echo $expName "done"
echo

























