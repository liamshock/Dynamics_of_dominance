import numpy as np
np.set_printoptions(suppress=True)
import h5py
import time
import cv2
import glob
import os
import argparse

t0 = time.perf_counter()


# ------------------------------- args ------------------------------- #

parser = argparse.ArgumentParser()
parser.add_argument("trackPath", type=str, help='path to h5 file tracking results')
parser.add_argument("winnerIdx", type=int)
parser.add_argument("experiment_path", type=str, help='directory containing vids for exp')
parser.add_argument("F0", type=int)
parser.add_argument("FE", type=int)
parser.add_argument("camIdx", type=int)
parser.add_argument("saveFolder", type=str, help='the folder to save output movie in')
parser.add_argument("output_movie_name", type=str, help='the name of the output movie')
parser.add_argument("fps", type=int, help='output movie fps')
parser.add_argument("nrows", type=int, help='the height of the output movie')
parser.add_argument("ncols", type=int, help='the width of the output movie')
args = parser.parse_args()

fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
splitdata_mov_with_bodypoints_path = os.path.join(args.saveFolder,
                                                  args.output_movie_name)


# ----------------------- Functions and classes --------------------------- #


def draw_skeleton_on_camera_frame(frame, frame_image_coordinates):
    ''' Draw the image coordinates for the fish on the frame.

    --- Inputs ---
    frame: array, an image frame taken from a .mp4
    frame_image_coordinates: shape (numFish, numBodyPoints, 2)

    --- Returns ---
    frame: in input frame with the skeleton drawn
    '''
    # parse shapes from the input
    numFish, numBodyPoints, _ = frame_image_coordinates.shape

    # how bodypoints are joined
    edges = [[0,1], [1,2]]

    # colors of the fish
    fish_colors = [(255,0,0), (0,0,255)]

    for fishIdx in range(numFish):

        # -- draw the bodypoints -- #
        for bpIdx in range(numBodyPoints):
            # get the data
            imCoord = frame_image_coordinates[fishIdx, bpIdx]
            # get the color
            fcolor = fish_colors[fishIdx]
            color = ( int (fcolor [ 0 ]), int(fcolor [ 1 ]), int(fcolor [ 2 ]) )
            # if we don't have a NaN, draw the point
            if np.all(np.isnan(imCoord)):
                continue
            cv2.circle(frame, (int(imCoord[0]), int(imCoord[1])), radius=2, color=color, thickness=-1)

        # --- draw the lines between bodypoints -- #
        for edge in edges:
            pt1_bpIdx = edge[0]
            pt2_bpIdx = edge[1]

            pt1_imCoord_x = frame_image_coordinates[fishIdx][pt1_bpIdx][0]
            pt1_imCoord_y = frame_image_coordinates[fishIdx][pt1_bpIdx][1]
            pt1_imCoord = np.array([pt1_imCoord_x, pt1_imCoord_y])

            pt2_imCoord_x = frame_image_coordinates[fishIdx][pt2_bpIdx][0]
            pt2_imCoord_y = frame_image_coordinates[fishIdx][pt2_bpIdx][1]
            pt2_imCoord = np.array([pt2_imCoord_x, pt2_imCoord_y])

            # parse the points for format and emptiness
            if np.all(np.isnan(pt1_imCoord)):
                pt1 =tuple([np.NaN, np.NaN])
            else:
                pt1 = tuple([int(x) for x in pt1_imCoord])
            if np.all(np.isnan(pt2_imCoord)):
                pt2 =tuple([np.NaN, np.NaN])
            else:
                pt2 = tuple([int(x) for x in pt2_imCoord])

            # if all is good, draw the lines
            if ~np.all(np.isnan(pt1)) and ~np.all(np.isnan(pt2)):
                cv2.line(frame, pt1=pt1,  pt2=pt2, color=color, thickness=1, lineType=cv2.LINE_AA)

    return frame

def get_video_params_from_mp4(mp4path):
    ''' Given a filepath of a .mp4, return the number of frames in the movie,
        the framerate of the movie, and the width and height of each frame.

    --- requirments ---
    cv2

    --- args ---
    mp4path:

    --- returns ---
    nfs: the number of frames in the movie
    fps: the framerate of the movie
    width: width of each frame.
    height:  height of each frame.

    '''
    cap = cv2.VideoCapture(mp4path)

    if not cap.isOpened():
        print("could not open : {0}".format(mp4path))
        return

    nfs = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    return nfs, fps, width, height



def get_video_params_for_all_movies_in_list(movpath_list):
    ''' Given a list of filepath for .mp4s, return arrays for the number of frames
        in each movie, the framerate of each movie, the width and height of each movie.

    --- see also ---
    get_video_params_from_mp4()

    --- args ---
    movpath_list: a list of .mp4 filepaths

    --- returns ---
    vids_nfs: array over movies, containing numframes in each
    vids_fps: array over movies, containing fps of each
    vids_widths: array over movies, containing frame width of each
    vids_heights: array over movies, containing frame height of each
    '''

    vids_nfs = []
    vids_fps = []
    vids_widths = []
    vids_heights = []

    for path in movpath_list:
        nfs, fps, width, height = get_video_params_from_mp4(path)
        vids_nfs.append(nfs)
        vids_fps.append(fps)
        vids_widths.append(width)
        vids_heights.append(height)

    vids_nfs = np.array(vids_nfs)
    vids_fps = np.array(vids_fps)
    vids_widths = np.array(vids_widths)
    vids_heights = np.array(vids_heights)

    return vids_nfs, vids_fps, vids_widths, vids_heights





class SplitdataManager(object):
    ''' A simple class for dealing with all of the splitdata folders for an
        experiment.

    -- Use cases --
    (a) find the filepaths for all camera views for a splitdata
    (b) find the global frame idxs of each splitdata
    (c) Given a global idx, find the splitdata folder and local index

    -- Attributes --
    main_path                      : the directory of the main experimental folder
    splitdata_paths                : a list over splitdatas, where each element is a
                                     list to the splitdata folder for each of the 3 cams
    start_stop_frames_for_splitdata: an array of shape (numSplitdatas, 2),
                                     where the 2nd dim gives the global frame idx for the
                                     first and last frame of each splitdata

    -- Public Methods --
    return_splitdata_folder_and_local_idx_for_global_frameIdx

    '''

    def __init__(self, main_folder_path):
        ''' Instantiate the object

        -- Args --
        main_folder_path: the 'top-level' folder in an experiment
                          e.g. '/path/to/Data/FishTank20191211_115245'
        '''
        self.main_path = main_folder_path

        # get the filepaths of all splitdata folders in experiment via the .mp4 names
        #print('--------')
        folderPaths = []
        for filepath in glob.glob(self.main_path + '**/*.mp4'):
            if '3panel' in filepath:
                continue
            #print(filepath)
            folderPath = filepath.split('.')[0]
            folderPaths.append(folderPath)
        folderPaths.sort()
        self._folderPaths = folderPaths


        # find the 3 filepaths (one for each cam) for each splitdata
        first_splitdata_num = int(self._folderPaths[0].split(sep='/')[-1][-4:])
        last_splitdata_num = int(self._folderPaths[-1].split(sep='/')[-1][-4:])
        splitdata_paths = []
        # plus 1 to include last folder
        for idx in range(first_splitdata_num, last_splitdata_num+1):
            splitdata_idx_paths = []
            splitdata_idx_paths.append(self.main_path + 'D_xz' + '/' + 'splitdata' + str(idx).zfill(4))
            splitdata_idx_paths.append(self.main_path + 'E_xy' + '/' + 'splitdata' + str(idx).zfill(4))
            splitdata_idx_paths.append(self.main_path + 'F_yz' + '/' + 'splitdata' + str(idx).zfill(4))
            splitdata_paths.append(splitdata_idx_paths)
        self.splitdata_paths = splitdata_paths
        self._num_splitdatas = len(self.splitdata_paths)

        # get the global start and stop frame idx for each splitdata
        self.start_stop_frames_for_splitdata = self._get_start_stop_frames_for_splitdata_folders()


    def return_splitdata_folder_and_local_idx_for_global_frameIdx(self, global_frameIdx,
                                                                  return_splitdataIdx=False):
        ''' Given the index of a frame, return the name of the splitdata folder that it came from

        -- see also --
        get_start_stop_frames_for_splitdata_folders

        -- EX1 --
        If we started on splitdata0000, and record 6000 frames in each,

        splitdata_folder,local_idx = return_splitdata_folder_for_global_frameIdx(splitdata_paths,
                                                                                start_stop_frames_for_splitdata,
                                                                                10000)
        print(splitdata_folder)
        >> ['/work/StephensU/liam/experiments/1_male/D_xz/splitdata0001',
            '/work/StephensU/liam/experiments/1_male/E_xy/splitdata0001',
            '/work/StephensU/liam/experiments/1_male/F_yz/splitdata0001']
        print(local_idx)
        >> 4000
        '''
        # find the folder index
        for fld_idx in range(self._num_splitdatas):
            if global_frameIdx >= self.start_stop_frames_for_splitdata[fld_idx, 1]:
                continue
            elif global_frameIdx >= (self.start_stop_frames_for_splitdata[fld_idx, 0] and
                 global_frameIdx < self.start_stop_frames_for_splitdata[fld_idx, 1]):
                splitdata_folder_idx = fld_idx
                break
            else:
                continue

        # find the local frame number
        local_idx = global_frameIdx - self.start_stop_frames_for_splitdata[splitdata_folder_idx, 0]

        if return_splitdataIdx == True:
            return splitdata_folder_idx, local_idx
        else:
            return self.splitdata_paths[splitdata_folder_idx], local_idx


    def _get_start_stop_frames_for_splitdata_folders(self):
        ''' Return an array containing the global start and stop frame index for each
            splitdata folder.
        '''
        # initialize an array for global start and stop idxs for each splitdata folder
        start_stop_frames_for_splitdata = np.zeros((self._num_splitdatas, 2), dtype=int)

        # set a counter for global frame index, updated as we move through the folders
        running_total_idx = 0

        # loop over splitdata folders getting the indices
        #total_numFrames = []
        for splitdata_idx, splitdata_folder in enumerate(self.splitdata_paths):

            # first get the number of frames in this folder by examining frameCropType array shape
            folder = splitdata_folder[0] # XZ, XY or YZ - doesn't matter, we just want to find the number of frames
            video_capture = cv2.VideoCapture(folder + '.mp4')
            splitdata_numFrames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
            video_capture.release()

            # now get the frame indices
            splitdata_start = running_total_idx
            splitdata_stop = splitdata_start + splitdata_numFrames
            start_stop_frames_for_splitdata[splitdata_idx, 0] = splitdata_start
            start_stop_frames_for_splitdata[splitdata_idx, 1] = splitdata_stop

            # update the counter
            running_total_idx = splitdata_stop

        return start_stop_frames_for_splitdata





# ------------------------------- loading and preparing ------------------------------- #

# load the tracking results
with h5py.File(args.trackPath, 'r') as hf:
    tracks_imCoords_raw = hf['tracks_imCoords_raw'][:]

# put winner first
if args.winnerIdx == 0:
    pass
elif args.winnerIdx == 1:
    tracks_imCoords_raw = np.copy(tracks_imCoords_raw[:, :, ::-1])
else:
    raise ValueError('winnerIdx should be 0 or 1')

# make sure the output folder exists
if not os.path.exists(args.saveFolder):
    os.makedirs(args.saveFolder)


# we will use this object to easily find the paths of the splitdata.mp4s, that we will want to read-in
splitman = SplitdataManager(args.experiment_path)
splitdata_names = [triplet[0].split('/')[-1] for triplet in splitman.splitdata_paths]
splitdata_idxs = [i for i in range(len(splitdata_names))]
splitdata_name_to_idx_dict = dict(zip(splitdata_names, splitdata_idxs))
splitdata_name_to_paths = dict(zip(splitdata_names, splitman.splitdata_paths))


#  get the splitdata index for all frames
fIdx_splitdataIdx_localIdx_arr = np.zeros((args.FE-args.F0,3), dtype=int)
for fIdx in range(args.F0, args.FE):
    fIdx_splitdataIdx_localIdx_arr[fIdx-args.F0, 0] = fIdx
    fIdx_splitdataIdx_localIdx_arr[fIdx-args.F0, 1:] = splitman.return_splitdata_folder_and_local_idx_for_global_frameIdx(fIdx, return_splitdataIdx=True)
# all the splitdatas used
splitdata_idxs_used = np.unique(fIdx_splitdataIdx_localIdx_arr[:,1])

# get the input movie paths for all the splitdatas
splitdata_movie_paths = []
for ii,splitdata_idx in enumerate(splitdata_idxs_used):
    splitdata_movie_path = splitman.splitdata_paths[splitdata_idx][args.camIdx] + '.mp4'
    splitdata_movie_paths.append(splitdata_movie_path)

# get the tracking results for each splitdata we use
used_splitdatas_imageCoords = []
for ii,splitdata_idx in enumerate(splitdata_idxs_used):
    splitdata_frame_range = splitman.start_stop_frames_for_splitdata[splitdata_idx]
    splitdata_raw_imageCoords = np.copy(tracks_imCoords_raw[:, splitdata_frame_range[0]:splitdata_frame_range[1]])
    used_splitdatas_imageCoords.append(splitdata_raw_imageCoords)





# ------------------------------- write the movie ------------------------------- #


# set other parameters for the movie
resolution = (args.ncols, args.nrows)
out = cv2.VideoWriter(splitdata_mov_with_bodypoints_path, fourcc, args.fps, resolution)

# open up the videos we want to read frames from
caps = []
for path in splitdata_movie_paths:
    caps.append( cv2.VideoCapture(path) )

# loop over the frames that we want,
# grabbing each frame, drawing the bodypoints on it, and then saving the edited frame to the new movie
numFramesToWrite = fIdx_splitdataIdx_localIdx_arr.shape[0]
for ii in range(numFramesToWrite):

    # get the associated splitdata_idx and local frame
    fIdx = fIdx_splitdataIdx_localIdx_arr[ii, 0]
    splitdata_idx = fIdx_splitdataIdx_localIdx_arr[ii, 1]
    localIdx =  fIdx_splitdataIdx_localIdx_arr[ii, 2]

    # print an update
    #if np.mod(fIdx, 10) == 0:
    #    print(fIdx, splitdata_idx, localIdx)

    # read the correct frame from the correct cap
    correct_cap_index = np.where(splitdata_idxs_used == splitdata_idx)[0][0]
    cap = caps[correct_cap_index]
    cap.set(1,localIdx)
    ret, frame = cap.read()
    if not ret:
        raise TypeError('problem with splitdata_movie_path in reading the frames')

    # get the correct imagecoords
    correct_splitdatas_imageCoords_index = np.where(splitdata_idxs_used == splitdata_idx)[0][0]
    splitdata_raw_imageCoords = used_splitdatas_imageCoords[correct_splitdatas_imageCoords_index]

    # Draw the skeletons
    frame_image_coordinates = np.copy(splitdata_raw_imageCoords[args.camIdx, localIdx])
    frame = draw_skeleton_on_camera_frame(frame, frame_image_coordinates)

    # crop and resize the frame
    # crop until resolution is
    frame = cv2.resize(frame[64:], (args.ncols, args.nrows), interpolation=cv2.INTER_AREA)

    # convert to BGR before writing the frame, otherwise the colors will be swapped
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # write the frame
    out.write(frame)


# close the writer and reader
for cap in caps:
    cap.release()
out.release()


# ----------------------------------------------------------------------------------------------------------#
# ----------------------------------------------------------------------------------------------------------#




























