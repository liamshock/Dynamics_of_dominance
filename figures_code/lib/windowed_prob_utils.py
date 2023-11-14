import numpy as np


def compute_windowed_distribution_array_from_1D_tseries(tseries, time_windows, spatial_bins):
    ''' Create the x=time, y=var, windowed distribution array for the given tseries.

    -- args --
    tseries: a 1D timeseries
    time_windows: a (N,2) array of start-stop frames i.e the xbins
    spatial_bins: a 1D array of bins in normal format (np.arange(start,stop,step)),
                  i.e. the ybins

    -- returns --
    ts_windowed_data: shape(numBins, numWins)


    -- Note --
    the sns.heatmap outlook indexing is like

    0 1 2 ..
    1
    2
   ...

   But we want y to run the opposite way, like a traditional graph ymin to ymax.
   So what we will do is reverse the histogram values for each window as we record,
   and also swap the ylabels to match everything up.

   '''

    # -- parse some shapes -- #
    numWins = time_windows.shape[0]
    # bins array is edges, so 1 longer than values array, hence the -1
    numBins = spatial_bins.shape[0] - 1

    # -- compute the distribution --#
    # np.histogram returns bins, which are bin edges, and values in the bins
    # so bins array is 1 longer than values array, hence the -1 in ts_windowed_data shape
    ts_windowed_data = np.zeros((numBins, numWins))
    for winIdx in range(numWins):
        f0, fE = time_windows[winIdx]
        windowed_data = tseries[f0:fE]
        histvals, _ = np.histogram(windowed_data, bins=spatial_bins)
        # now record the histvals backwards, so binmax appears at the top of the figure
        ts_windowed_data[:, winIdx] = histvals[::-1]

    return ts_windowed_data


def return_overlapping_windows_for_timeframes(numFrames, window_size=200, window_step=50):
    ''' Given a number of frames, return an 2D array of window start-stop frames.
    '''
    # define, for clarity, the first window
    win0_end = int(window_size)

    # find numWindows, by adding incrementally and watching the last frame
    last_frame_in_windows = win0_end
    numWindows = 1
    while last_frame_in_windows < (numFrames - window_step):
        numWindows += 1
        last_frame_in_windows = win0_end + (numWindows-1)*window_step

    # now fill-in the windows array of frame indices
    windows = np.zeros((numWindows, 2))
    windows[0, 0] = 0
    windows[0, 1] = win0_end
    for winIdx in range(1, numWindows):
        w0 = winIdx*window_step
        wF = w0 + window_size
        windows[winIdx, 0] = w0
        windows[winIdx, 1] = wF
    return windows.astype(int)


def get_fightbout_rectangle_info(fight_f0, fight_fE, time_windows, spatial_bins):
    ''' Return the (left, bottom, width, height) needed to draw  a rectangle
        representing a fight bout
    '''
    # get the timebin idxs for these frame numbers
    fightbout_start_window = time_windows[time_windows[:,0]>fight_f0][0]
    fight_start_winIdx = np.where(time_windows==fightbout_start_window)[0][0]
    
    # get the stop window,
    # if we hit the end of the experiment, use the last window
    fightbout_stop_window_candidates = time_windows[time_windows[:,0]>fight_fE]
    if len(fightbout_stop_window_candidates) == 0:
        fightbout_stop_window = time_windows[-1]
    else:
        fightbout_stop_window = time_windows[time_windows[:,0]>fight_fE][0]
    fight_stop_winIdx = np.where(time_windows==fightbout_stop_window)[0][0]
    
    # define the shapes we want
    left = fight_start_winIdx
    width = fight_stop_winIdx - left
    bottom = 0
    height = len(spatial_bins)-1
    return left, bottom, width, height

