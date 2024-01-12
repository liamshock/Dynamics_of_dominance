import numpy as np
np.set_printoptions(suppress=True)
import h5py
import time
import glob
import argparse
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from multiprocessing import Pool, RawArray
import os
import subprocess


t0 = time.time()

# --------------------------------------------------#





# ------ Parse args ------- #
parser = argparse.ArgumentParser()
parser.add_argument("loadpath", type=str)
parser.add_argument("winnerIdx", type=int)
parser.add_argument("savefolderPath", type=str)
parser.add_argument("movieName", type=str)
parser.add_argument("make_concatenation_movie", type=int)
parser.add_argument("delete_individual_movies", type=int)
parser.add_argument("parstartFrame", type=int)
parser.add_argument("parstopFrame", type=int)
parser.add_argument("step", type=int)
parser.add_argument("numProcessors", type=int)
parser.add_argument("outmovie_fps", type=int)
args = parser.parse_args()






def make_3D_trajectories_video_PARALLEL(i):
    """ NB: this is an intrinscially parallel version of make_3D_trajectories_video(),
            see that for explanations.
    """

    # parse what we need for var_dict
    complete_tracks_3D_data = np.frombuffer(var_dict['tracks_3D']).reshape(var_dict['tracks_3D_shape'])
    savefolderPath = var_dict['savefolderPath']
    parstartFrame = var_dict['parstartFrame']
    parstopFrame = var_dict['parstopFrame']
    step = var_dict['step']
    outmovie_fps = var_dict['outmovie_fps']


    # get the start/stop for this movie
    parallelization_start_stop_frms = create_array_of_start_stop_frames_for_movie_parallelization(parstartFrame,
                                                                                                  parstopFrame,
                                                                                                  step=step)
    f0, fE = parallelization_start_stop_frms[i]

    # get the savepath for this movie
    savePath = os.path.join(savefolderPath, 'mov_'+str(i).zfill(5)+'.mp4')

    # parse the data for this segment
    tracks_3D = complete_tracks_3D_data[f0:fE]

    # parse the input shapes
    numFrames, numFish, numBodyPoints, _ = tracks_3D.shape
    fish_colors = ['red', 'blue']

    # Attaching 3D axis to the figure
    fig = plt.figure()
    fig.tight_layout()
    ax = p3.Axes3D(fig, [0.06, -0.4, 0.92, 1.9], auto_add_to_figure=False)
    fig.add_axes(ax)

    # Initialize scatters (list over fish, list over bps)
    symbols = ['o', 's', 'x']
    sizes=12

    # Main scatters
    scatters = []
    for fishIdx in range(numFish):
        col = fish_colors[fishIdx]
        fish_scatters = []
        for bpIdx in range(numBodyPoints):
            fish_scatters.append(ax.scatter(tracks_3D[0,fishIdx,bpIdx,0:1],
                                            tracks_3D[0,fishIdx,bpIdx,1:2],
                                            tracks_3D[0,fishIdx,bpIdx,2:],
                                            c=col, s=sizes, marker=symbols[bpIdx]))
        scatters.append(fish_scatters)

    # Main lines
    lines = []
    for fishIdx in range(numFish):
        col = fish_colors[fishIdx]
        line = ax.plot(tracks_3D[0, fishIdx, :, 0],
                       tracks_3D[0, fishIdx, :, 1],
                       tracks_3D[0, fishIdx, :, 2], c=col)[0]
        lines.append(line)

    # ----------------
    # projections
    sizes=5

    scatter_projections = []
    lines_projections = []
    for projectionIdx in range(3):
        proj_scatters = []
        proj_lines = []
        for fishIdx in range(numFish):
            col = fish_colors[fishIdx]
            # scatter plots
            fish_scatters = []
            for bpIdx in range(numBodyPoints):
                #print(symbols[bpIdx])
                fish_scatters.append(ax.scatter(tracks_3D[0,fishIdx,bpIdx,0:1],
                                                tracks_3D[0,fishIdx,bpIdx,1:2],
                                                tracks_3D[0,fishIdx,bpIdx,2:],
                                                c=col, s=sizes, marker=symbols[bpIdx], alpha=0.1))
            # lines
            line = ax.plot(tracks_3D[0, fishIdx, :, 0],
                           tracks_3D[0, fishIdx, :, 1],
                           tracks_3D[0, fishIdx, :, 2], c=col, alpha=0.1)[0]
            proj_lines.append(line)
            proj_scatters.append(fish_scatters)

        scatter_projections.append(proj_scatters)
        lines_projections.append(proj_lines)
    # ----------------

    # the text
    all_timestamps = ['frame: '+ str(i).zfill(7) for i in range(f0, fE)]
    time_text = ax.text2D(-0.05, 0.07, all_timestamps[0], fontsize=7, transform=ax.transAxes)

    # Setting the axes properties
    ax.set_xlim3d([0, 40])
    ax.set_xlabel('X')

    ax.set_ylim3d([0, 40])
    ax.set_ylabel('Y')

    ax.set_zlim3d([0, 40])
    ax.set_zlabel('Z')

    # Provide starting angle for the view
    elevation_angle = 10
    azimuthal_angle = 45
    ax.view_init(elevation_angle, azimuthal_angle)

    ani = animation.FuncAnimation(fig, _animate_scatters, numFrames, fargs=(tracks_3D, scatters, lines,
                                                                           scatter_projections, lines_projections,
                                                                           time_text, all_timestamps),
                                       interval=10, blit=False, repeat=True)

    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=outmovie_fps, metadata=dict(artist='LOS'), bitrate=-1)
    ani.save(savePath, writer=writer)
    print(i, ' finished')
    return

def _animate_scatters(iteration, tracks_3D, scatters, lines, scatter_projections, lines_projections, text, all_timestamps):
    """
    Update the tracks_3D held by the scatter plot and therefore animates it.

    Args:
        iteration (int): Current iteration of the animation
        tracks_3D (list): List of the data positions at each iteration.
        scatters (list): List of all the scatters (One per element)

    Returns:
        list: List of scatters (One per element) with new coordinates
    """
    # parse the input shapes
    numFrames, numFish, numBodyPoints, _ = tracks_3D.shape

    # update the scatters
    for fishIdx in range(numFish):
        for bpIdx in range(numBodyPoints):
            scatters[fishIdx][bpIdx]._offsets3d = (tracks_3D[iteration,fishIdx,bpIdx,0:1],
                                                   tracks_3D[iteration,fishIdx,bpIdx,1:2],
                                                   tracks_3D[iteration,fishIdx,bpIdx,2:])

    # update the lines
    for fishIdx,line in enumerate(lines):
        line.set_data(tracks_3D[iteration, fishIdx, :, :2].swapaxes(0,1))
        line.set_3d_properties(tracks_3D[iteration, fishIdx, :, 2])


    # update the text
    text.set_text(all_timestamps[iteration])

    # -------------- #
    for projectionIdx in range(3):

        # XZ => Y=0
        if projectionIdx == 0:
            projectionData = np.copy(tracks_3D[iteration])
            projectionData[:, :, 1] = 0
        # XY => Z=0
        elif projectionIdx == 1:
            projectionData = np.copy(tracks_3D[iteration])
            projectionData[:, :, 2] = 0
        # YZ => X=0
        elif projectionIdx == 2:
            projectionData = np.copy(tracks_3D[iteration])
            projectionData[:, :, 0] = 0

        proj_scatters = scatter_projections[projectionIdx]
        proj_lines = lines_projections[projectionIdx]

        # update the scatters
        for fishIdx in range(numFish):
            for bpIdx in range(numBodyPoints):
                proj_scatters[fishIdx][bpIdx]._offsets3d = (projectionData[fishIdx,bpIdx,0:1],
                                                            projectionData[fishIdx,bpIdx,1:2],
                                                            projectionData[fishIdx,bpIdx,2:])
        # update the lines
        for fishIdx,line in enumerate(proj_lines):
            line.set_data(projectionData[fishIdx, :, :2].swapaxes(0,1))
            line.set_3d_properties(projectionData[fishIdx, :, 2])

    return


def create_array_of_start_stop_frames_for_movie_parallelization(startFrame, stopFrame, step=1000):
    ''' Return an array that we can index easily to divide the frames into chunks
        for processing in parallel.

    --- args ---
    startFrame: the frame index to start on
    stopFrame: the frame index to stop on
    step=1000: the size (in frames) of each chunk

    --- returns ---
    start_stop_frms: array of shape (numChunks, 2), where
                     each row contains the start frame and stop frame
                     for that chunk.

    --- example ---
    >> nfs = 501943
    >> start_stop_frms = create_array_of_start_stop_frames_for_parallelization(nfs, step=1000)
    >> start_stop_frms
    array([[     0,   1000],
           [  1000,   2000],
           [  2000,   3000],
           ...,
           [499000, 500000],
           [500000, 501000],
           [501000, 501943]])
    '''
    start_frms = np.arange(startFrame, stopFrame, step)
    stop_frms = np.append(np.arange(startFrame+step, stopFrame, step), stopFrame)
    start_stop_frms = np.stack([start_frms, stop_frms], axis=1)
    return start_stop_frms










# --------------------------------------------------#
#     Preparation
# --------------------------------------------------#

# load the trajectory data
with h5py.File(args.loadpath, 'r') as hf:
    tracks_3D = hf['tracks_3D_smooth'][:]

# put winner first
if args.winnerIdx == 0:
    pass
elif args.winnerIdx == 1:
    tracks_3D = np.copy(tracks_3D[:, ::-1])
else:
    raise ValueError('winnerIdx should be 0 or 1')

# check the frame limits
numFrames = tracks_3D.shape[0]
if args.parstopFrame == - 1:
    parstopFrame = numFrames
else:
    parstopFrame = args.parstopFrame

# create the savefolder if it doesnt already exist
if not os.path.exists(args.savefolderPath):
    os.makedirs(args.savefolderPath)

# parse the other options
if args.make_concatenation_movie == 1:
    make_concatenation_movie = True
else:
    make_concatenation_movie = False

if args.delete_individual_movies == 1:
    delete_individual_movies = True
else:
    delete_individual_movies = False


# make the array of frame ranges for each job
parallelization_start_stop_frms = create_array_of_start_stop_frames_for_movie_parallelization(args.parstartFrame,
                                                                                              parstopFrame,
                                                                                              step=args.step)

# the list of jobIdxs to map over
job_idxs = [i for i in range(parallelization_start_stop_frms.shape[0])]


# the make filepath of the concatenation movie
movsavepath = os.path.join(args.savefolderPath, args.movieName)


# ---- parallization boilerplate ---- #

# This is a formality to allow multiple processes to share access to arrays
tracks_3D_RARR = RawArray('d', int(np.prod(tracks_3D.shape)))
tracks_3D_np = np.frombuffer(tracks_3D_RARR).reshape(tracks_3D.shape)
np.copyto(tracks_3D_np, tracks_3D)

# A global dictionary storing the variables passed from the initializer
# This dictionary holds variables that each process will share access to, instead of making copies
var_dict = {}

# This function initializes the shared data in each job process
def init_worker(tracks_3D, tracks_3D_shape, savefolderPath, parstartFrame, parstopFrame, step, outmovie_fps):
    var_dict['tracks_3D'] = tracks_3D
    var_dict['tracks_3D_shape'] = tracks_3D_shape
    var_dict['savefolderPath'] = savefolderPath
    var_dict['parstartFrame'] = parstartFrame
    var_dict['parstopFrame'] = parstopFrame
    var_dict['step'] = step
    var_dict['outmovie_fps'] = outmovie_fps




# --------------------------------------------------#
#     Make the individual movies in parallel
# --------------------------------------------------#

print('Launching movie making ...')
print()


# map the function
with Pool(processes=args.numProcessors, initializer=init_worker,
          initargs=(tracks_3D_RARR,
                    tracks_3D.shape,
                    args.savefolderPath,
                    args.parstartFrame,
                    parstopFrame,
                    args.step,
                    args.outmovie_fps)) as pool:
    outputs = pool.map(make_3D_trajectories_video_PARALLEL, job_idxs)


print('Individual movie making finished: ', time.time()-t0)
print()
print()
print()




# --------------------------------------------------#
#     Make the concatenation movie
# --------------------------------------------------#

if make_concatenation_movie:

    # Use a subprocess bash command, to use ffmpeg, to concatenate the movies
    ffmpeg_concat_bash_string = ('for f in *.mp4 ; '
                                  'do echo file \"$f\" >> list.txt;'
                                  ' done && '
                                  'ffmpeg -f concat -safe 0 -i list.txt '
                                  '-c copy {0} && rm list.txt'.format(args.movieName)
                                 )
    full_bash_string = 'cd {0}; {1}'.format(args.savefolderPath, ffmpeg_concat_bash_string)

    process_result = subprocess.run(full_bash_string, shell=True, capture_output=True)


    # delete the individual movies if we want to
    movpaths = glob.glob(os.path.join(args.savefolderPath, "mov*.mp4"))
    if delete_individual_movies:
        for path in movpaths:
            os.remove(path)

# Do nothing else if we don't want the concatenation movie
else:
    pass




# --------------------------------------------------#
#     Finish up
# --------------------------------------------------#

print()
print()
print('-----')
print('Finished: ', time.time()-t0, ' s')





















