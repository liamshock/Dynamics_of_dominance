''' This file contains function for the post-processing of data
'''
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter as sgf


def contiguous_regions(bool_array):
    """Finds contiguous True regions of the boolean array.

    Returns a 2D array where the first column is the start index
    of the region and the second column is the end index.

    Thanks: https://stackoverflow.com/a/4495197
    """

    # Find the indicies of changes in "condition"
    d = np.diff(bool_array)
    idx, = d.nonzero()

    # We need to start things after the change in "condition". Therefore,
    # we'll shift the index by 1 to the right.
    idx += 1

    if bool_array[0]:
        # If the start of condition is True prepend a 0
        idx = np.r_[0, idx]

    if bool_array[-1]:
        # If the end of condition is True, append the length of the array
        idx = np.r_[idx, bool_array.size]

    # Reshape the result into two columns
    idx.shape = (-1,2)
    return idx




def interpolate_over_small_gaps(tracks_3D_raw, limit=5, polyord=1):
    ''' Return an interpolated version of the raw trajectories.
        These will still contain NaNs in big gaps

    -- args --
    tracks_3D_raw: array, shape (numFrames,numFish,numBodyPoints,3),
                   the trajectories before post-processing
    limit: int, the maximum size of gaps to interpolate over
    polyord: the order of the polynomial to use for the interpolation.
             (normally set to 1, as we will later smooth with a higher order poly)

    -- Returns --
    tracks_3D_interpd: array, shape (numFrames,numFish,numBodyPoints,3),
                        the trajectories after interpolation over small gaps
    '''
    # parse shapes
    print(tracks_3D_raw.shape)
    numFrames,numFish,numBodyPoints,_ = tracks_3D_raw.shape

    tracks_3D_interpd = np.copy(tracks_3D_raw)
    # loop to grab a 1D timeseries
    for fishIdx in range(numFish):
        for bpIdx in range(numBodyPoints):
            for dimIdx in range(3):
                # interpolate the data in the most basic way
                data_1D = tracks_3D_raw[:, fishIdx, bpIdx, dimIdx]
                df = pd.DataFrame(data_1D)
                interpd_df = df.interpolate(method='polynomial', order=1, limit_direction='both', limit=5, inplace=False)
                # record
                tracks_3D_interpd[:, fishIdx, bpIdx, dimIdx] = interpd_df.values[:, 0]

    return tracks_3D_interpd




def get_smooth_timeseries_and_derivatives_using_savgol(tracks_3D_interpd, win_len=9, polyOrd=2, dt=0.01):
    ''' Given trajectories, which should be, not not necessarily are,
        interpolated, return smoothes versions of the data and derivative info

    --- Method ---
    Apply a savitzky-golay filter to each component of the trajectory
    vector separately

    --- args ---
    tracks_3D_interpd: array, shape (numFrames,numFish,numBodyPoints,3),
                       the trajectories after interpolation over small gaps

    -- kwargs --
    win_len=9: the window size in frames for the savitzky-golay filter
    polyOrd=2: the polynomial order for the savitzky-golay filter
    dt=0.01: the time interval between frames (1/fps). Used for computing derivatives.

    --- Returns ---
    tracks_3D_smooth:      array, shape (numFrames,numFish,numBodyPoints,3),
                           the trajectories after smoothing
    tracks_3D_vel_smooth:  array, shape (numFrames,numFish,numBodyPoints,3),
                           the first derivatrive of the trajectories after smoothing
    tracks_3D_speed_smooth: array, shape (numFrames,numFish,numBodyPoints),
                            the norm of the vel_smooth
    tracks_3D_accvec_smooth: array, shape (numFrames,numFish,numBodyPoints,3),
                             the second derivatrive of the trajectories after smoothing
    tracks_3D_accmag_smooth: array, shape (numFrames,numFish,numBodyPoints),
                             the norm of the accvec_smooth
    '''
    # parse shapes
    numFrames,numFish,numBodyPoints,_ = tracks_3D_interpd.shape

    # preallocate
    tracks_3D_smooth = np.copy(tracks_3D_interpd)
    tracks_3D_vel_smooth = np.copy(tracks_3D_interpd)
    tracks_3D_speed_smooth = np.ones((numFrames,numFish,numBodyPoints))*np.NaN
    tracks_3D_accvec_smooth = np.copy(tracks_3D_interpd)
    tracks_3D_accmag_smooth = np.ones((numFrames,numFish,numBodyPoints))*np.NaN

    # smoothed position
    for fishIdx in range(numFish):
        for bpIdx in range(numBodyPoints):
            for dimIdx in range(3):
                raw_data_1D = np.copy(tracks_3D_interpd[:, fishIdx, bpIdx, dimIdx])
                tracks_3D_smooth[:, fishIdx, bpIdx, dimIdx] = sgf(raw_data_1D, window_length=win_len, polyorder=polyOrd)

    # smooth velocity and speed
    for fishIdx in range(numFish):
        for bpIdx in range(numBodyPoints):
            for dimIdx in range(3):
                raw_data_1D = np.copy(tracks_3D_interpd[:, fishIdx, bpIdx, dimIdx])
                tracks_3D_vel_smooth[:, fishIdx, bpIdx, dimIdx] = sgf(raw_data_1D, window_length=win_len, deriv=1, polyorder=polyOrd) / dt
    for fishIdx in range(numFish):
        for bpIdx in range(numBodyPoints):
            for dimIdx in range(3):
                tracks_3D_speed_smooth[:, fishIdx, bpIdx] = np.linalg.norm(tracks_3D_vel_smooth[:, fishIdx, bpIdx, :], axis=1)

    # smooth acceleration vector and magnitude
    for fishIdx in range(numFish):
        for bpIdx in range(numBodyPoints):
            for dimIdx in range(3):
                raw_data_1D = np.copy(tracks_3D_interpd[:, fishIdx, bpIdx, dimIdx])
                tracks_3D_accvec_smooth[:, fishIdx, bpIdx, dimIdx] = sgf(raw_data_1D, window_length=win_len, deriv=2, polyorder=polyOrd) / dt
    for fishIdx in range(numFish):
        for bpIdx in range(numBodyPoints):
            for dimIdx in range(3):
                tracks_3D_accmag_smooth[:, fishIdx, bpIdx] = np.linalg.norm(tracks_3D_accvec_smooth[:, fishIdx, bpIdx, :], axis=1)

    outs = [tracks_3D_smooth,
            tracks_3D_vel_smooth,
            tracks_3D_speed_smooth,
            tracks_3D_accvec_smooth,
            tracks_3D_accmag_smooth]
    return outs


def process_1D_timeseries(ts, interp_max_gap=10, interp_polyOrd=2, smooth_polyOrd=2,
                          smooth_winSize=10, deriv=0, dt=0.01):
    ''' Process the 1D timeseries ts by interpolation and smoothing

    -- args --
    ts: a one-dimensional numpy array

    -- kwargs --
    interp_max_gap: the largest NaN gap we will interpolate over
    interp_polyOrd: the order of the polynomial we will use
    smooth_polyOrd: the order of the sav-gol polynomial
    smooth_winSize: the window size of the sav-gol
    '''

    # first find where we originally had data
    raw_mask = np.ma.masked_invalid(ts).mask

    # find contiguous regions bigger than the threshold
    raw_NaN_regions = contiguous_regions(raw_mask)

    # find the lengths of these contiguous regions
    region_lengths = np.diff(raw_NaN_regions, axis=1)[:,0]

    # find the big regions - bigger than threshold of interpolation
    # These big regions will ultimately be NaNd
    big_raw_region_idxs = np.where(region_lengths > interp_max_gap)[0]
    big_regions = np.array([raw_NaN_regions[i, :] for i in big_raw_region_idxs])

    # interpolate the arrays
    df = pd.DataFrame(ts)
    interpd_df = df.interpolate(method='polynomial', order=interp_polyOrd, limit_direction='both', limit=interp_max_gap, inplace=False)
    ts_interpd = interpd_df.values[:,0]

    # now re_NaN the large gaps which will have had their edges fleshed out by pd
    for i,bigReg in enumerate(big_regions):
        f0,fE = bigReg
        ts_interpd[f0:fE] = np.NaN

    # now smooth the timeseries
    # smoothing with sgf will not fill-in NaN gaps, so no need for additional masking
    ts_smooth = sgf(ts_interpd, deriv=deriv, delta=dt, window_length=smooth_winSize, polyorder=smooth_polyOrd, mode='nearest')
    return ts_smooth




def get_smooth_trajectories_from_raw_trajectories(raw_trajectories, interp_max_gap, interp_polyOrd, smooth_polyOrd, smooth_winSize, dt=0.01):
    ''' Interpolate over gaps, and use a savitsky-golay filter to smooth,
        the supplied (numFrames, numFish, numBodyPoints, 3) shaped raw trajectories.

    -- args --
    raw_trajectories: (numFrames, numFish, numBodyPoints, 3) shaped array of raw tracking results
    interp_max_gap: the maximum Nan-gap size to interpolate over
    interp_polyOrd: the order of the interpolation
    smooth_polyOrd: the smooth polynomial order
    smooth_winSize: the window size over which to fit the smoothing polynomial

    -- returns --
    smooth_trajectories: (numFrames, numFish, numBodyPoints, 3) shaped array of post-processed tracking results
    '''
    # parse some shapes
    _, numFish, numBodyPoints, _ = raw_trajectories.shape
    # preallocate the output
    tracks_3D_smooth = np.ones_like(raw_trajectories)*np.NaN
    # post-process 1D tseries
    for fishIdx in range(numFish):
        for bpIdx in range(numBodyPoints):
            for dimIdx in range(3):
                tracks_3D_smooth[:,fishIdx,bpIdx,dimIdx] = process_1D_timeseries(raw_trajectories[:,fishIdx,bpIdx,dimIdx],
                                                                                 interp_max_gap=interp_max_gap,
                                                                                 interp_polyOrd=interp_polyOrd,
                                                                                 smooth_polyOrd=smooth_polyOrd,
                                                                                 smooth_winSize=smooth_winSize,
                                                                                 dt=dt)
    return tracks_3D_smooth
