import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


def compute_dpp_tet_tet_distribution_plot_with_fight_bouts(trajectory, expName, exp_fight_bout_info, figsavepath, dpp_bins, tet_bins,
                                                           winnerIdx=0, loserIdx=1, has_clear_winner=True, window_size=6000, window_step=100,
                                                           dpp_vmax=150, tet_vmax=150, num_xticks=10, cmap='Blues',
                                                           fps=100, use_cbar=False):
    ''' Save a figure showing the windowed distributions of Dpp,tetW,tetL for the provided trajectory, with fight bouts drawn.

    --- args ---
    trajectory         : array shaped (numFrames, numFish=2, numBodyPoints=3, 3), the tracking results for an experiment
    expName            : string, for the plot title
    exp_fight_bout_info: array shaped (numBouts,2), containing the (startFrame,stopFrame) for the bouts
    figsavepath        : the filepath to save to
    dpp_bins           : the bins used for Dpp, these do not have to match the bins used in clustering to find the fight-bouts.
                         These are for visualization only.
    tet_bins           : the bins used for theta variables, these do not have to match the bins used in clustering to find the fight-bouts.
                         These are for visualization only.

    --- kwargs ---
    winnerIdx=0            : the index of the winner
    loserIdx=1             : the index of the loser
    has_clear_winner=True  : A bool to show if we Trust winnner and loser or not (theta_W/L vs theta_?/?)
    window_size=6000       : The window size for the windowing
    window_step=100        : The number of frames between window start points for the windowing
    dpp_vmax=150           : The max color threshold for plotting dpp intensities
    tet_vmax=150           : The max color threshold for plotting theta intensities
    num_xticks=10          : The number of time ticks that we want
    cmap='Blues'           : The colorscheme
    fps=100                : The frame-rate of the recording, for converting frames to minutes.
    use_cbar=False         : Bool to show the colorbar or not.

    --- returns ---
    None: a .png file is saved at figsavepath.
    '''
    # compute state variables
    dpp_ts = _compute_pec_pec_distance(trajectory)
    tetW_ts, tetL_ts = _compute_thetaW_and_thetaL(trajectory, winnerIdx, loserIdx)


    # get the time-windows
    expNfs = dpp_ts.shape[0]
    exp_time_wins = _return_overlapping_windows_for_timeframes(expNfs,
                                                              window_size=window_size,
                                                              window_step=window_step)


    # compute the windowed distributions
    dpp_heatmap = _compute_windowed_distribution_array_from_1D_tseries(dpp_ts,
                                                                      exp_time_wins,
                                                                      dpp_bins)
    tetW_heatmap = _compute_windowed_distribution_array_from_1D_tseries(tetW_ts,
                                                                       exp_time_wins,
                                                                       tet_bins)
    tetL_heatmap = _compute_windowed_distribution_array_from_1D_tseries(tetL_ts,
                                                                       exp_time_wins,
                                                                       tet_bins)


    # turn the array over bouts into a list over bouts
    fight_bout_info_list = [exp_fight_bout_info[i] for i in range(exp_fight_bout_info.shape[0])]


    # make the figure
    fig, axs =  _make_dpp_tetW_tetL_windowed_dist_figure(expName,
                                                        dpp_heatmap, tetW_heatmap, tetL_heatmap,
                                                        dpp_bins, tet_bins, exp_time_wins,
                                                        conclusive_winner=has_clear_winner,
                                                        dpp_vmax=dpp_vmax, tet_vmax=tet_vmax, num_xticks=num_xticks,
                                                        cmap=cmap, fps=fps, use_cbar=use_cbar,
                                                        fight_bout_info_list=fight_bout_info_list)
    # save the figure
    fig.savefig(figsavepath, dpi=300, transparent=False, bbox_inches='tight', pad_inches=0.1)
    return




def _make_dpp_tetW_tetL_windowed_dist_figure(expName, dpp_win_dist, tetW_win_dist, tetL_win_dist, dpp_bins,
                                            tet_bins, exp_time_wins, conclusive_winner=True,
                                            dpp_vmax=150, tet_vmax=150, num_xticks=10, cmap='Blues',
                                            fps=100, use_cbar=False, fight_bout_info_list=None):
    ''' Return the fig, axs values for a figure containing the windowed distributions of dpp,tetW,tetL for each experiment.
    '''

    # -----------------------------------------------------------------#
    #    Preparation
    # -----------------------------------------------------------------#

    plt.ioff()
    nrows=3
    fig, axs = plt.subplots(nrows=nrows, ncols=1, figsize=(10,6), sharex=False)
    fig.suptitle(f"{expName}", fontsize=14)

    # prepare the drawing of fight-bout regions
    if fight_bout_info_list is not None:
        do_draw_fight_boundaries = True
        # count the number of fights
        numFightBouts = len(fight_bout_info_list)

    # prepare the parsing params for each row
    ts_windowed_data_list = [dpp_win_dist, tetW_win_dist, tetL_win_dist]
    data_bins_list = [dpp_bins, tet_bins, tet_bins]

    if conclusive_winner:
        side_title_list = [r'$D_{PP}$ [cm]', r'$\theta_{W}$ [rad]', r'$\theta_{L}$ [rad]']
    else:
        side_title_list = [r'$D_{PP}$ [cm]', r'$\theta_{?}$ [rad]', r'$\theta_{?}$ [rad]']


    ytick_label_list = [[int(np.round(dpp_bins[0])), int(np.round(dpp_bins[-1]))][::-1],
                        [r'$-\pi$', '0', r'$\pi$'][::-1],
                        [r'$-\pi$', '0', r'$\pi$'][::-1]]

    vmax_list = [dpp_vmax,  tet_vmax,  tet_vmax]

    # -----------------------------------------------------------------#
    #    Main plotting
    # -----------------------------------------------------------------#


    # plot each var in turn
    for axIdx in range(3):

        ax = axs[axIdx]
        ts_windowed_data = ts_windowed_data_list[axIdx]
        data_bins = data_bins_list[axIdx]
        side_title = side_title_list[axIdx]
        plot_vmax = vmax_list[axIdx]

        # make the heatmap
        heatmap = sns.heatmap(ts_windowed_data, vmax=plot_vmax, cbar=use_cbar, cmap=cmap, ax=ax);
        if use_cbar == True:
            cbar = heatmap.collections[0].colorbar
            cbar.set_ticks([])
            cbar.set_ticklabels([])

        # prepare the labels
        ax.set_title('')
        ax.set_ylabel(side_title, fontsize=12)

        # prepare the xticks
        xticks = np.round(np.linspace(0, len(exp_time_wins), num_xticks)).astype(int)
        ax.set_xticks(xticks);

        # plot the xtick labels if we are the last plot
        xtick_labels = (np.linspace(exp_time_wins[0,0], exp_time_wins[-1,1], len(xticks)) / (fps*60)).astype(int)
        if axIdx == nrows-1:
            ax.set_xticklabels(xtick_labels);
            ax.set_xlabel('time [min]', fontsize=12)
        else:
            ax.set_xticklabels([]);
            ax.set_xlabel('')

        # prepare the yticks and labels
        ytick_labels = ytick_label_list[axIdx]
        num_yticks = len(ytick_labels)
        if num_yticks == 2:
            yticks = [0, len(data_bins)]
        elif num_yticks == 3:
            yticks = [0, len(data_bins)/2, len(data_bins)]
        else:
            raise TypeError('yticks error: Use only 2 or 3 yticks')

        ax.set_yticks(yticks);
        ax.set_yticklabels(ytick_labels, rotation=0);

        # make frame visible
        for _, spine in heatmap.spines.items():
            spine.set_visible(True)

        # make the frames thicker
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(2)
        ax.xaxis.set_tick_params(width=2, length=8)
        ax.yaxis.set_tick_params(width=2, length=8)


        # draw the fight-bouts if you want to
        if do_draw_fight_boundaries:
            for boutIdx in range(numFightBouts):
                if np.mod(boutIdx, 2) == 0:
                    chosen_color='purple'
                else:
                    chosen_color='green'
                fight_f0, fight_fE = fight_bout_info_list[boutIdx]
                left, bottom, width, height = _get_fightbout_rectangle_info(fight_f0, fight_fE,
                                                                           exp_time_wins, data_bins)
                rect=mpatches.Rectangle((left,bottom),width,height,
                                        fill=False,
                                        alpha=1,
                                        color=chosen_color)
                ax.add_patch(rect)


    # -----------------------------------------------------------------#
    #    wrap up
    # -----------------------------------------------------------------#

    fig.tight_layout()
    plt.ion()
    return fig, axs




def _get_fightbout_rectangle_info(fight_f0, fight_fE, time_windows, spatial_bins):
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


def _compute_windowed_distribution_array_from_1D_tseries(tseries, time_windows, spatial_bins):
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





def _return_overlapping_windows_for_timeframes(numFrames, window_size=200, window_step=50):
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


def _compute_pec_pec_distance(smooth_trajectories):
    ''' Compute the timeseries of pec-pec distances for the two fish.

    -- args --
    smooth_trajectories:

    -- returns --
    pec_pec_distances: (numFrames,)

    -- Notes --
    We dont care about winner and loser here, the measure is invariant
    under fish-swap.

    '''
    fish1_pec_tseries = np.copy(smooth_trajectories[:,0,1,:])
    fish2_pec_tseries = np.copy(smooth_trajectories[:,1,1,:])
    pec_pec_distances = np.linalg.norm(fish1_pec_tseries - fish2_pec_tseries, axis=1)
    return pec_pec_distances



def _compute_thetaW_and_thetaL(smooth_trajectories, winnerIdx, loserIdx):
    ''' Return relative headings of the winner and the loser.

    theta_w is defined as the angle needed to rotate the winner XY heading
    to point directly towards the loser.

    theta_l is defined as the angle need to rotate the loser XY heading to
    point direectly towards the winner.

    Positive angles are counterclockwise.

    -- args --
    smooth_trajectories:

    -- returns --
    theta_w: timeseries in range (-pi,pi)
    theta_l: timeseries in range (-pi,pi)
    '''
    # winner-pec to winner_head vector
    winner_heading_XY =  (smooth_trajectories[:, winnerIdx, 0] - smooth_trajectories[:, winnerIdx, 1])[:, :2]
    winner_heading_XY = _normalize_2D_vector_timeseries(winner_heading_XY)

    # loser-pec to loser_head vector
    loser_heading_XY =  (smooth_trajectories[:, loserIdx, 0] - smooth_trajectories[:, loserIdx, 1])[:, :2]
    loser_heading_XY = _normalize_2D_vector_timeseries(loser_heading_XY)

    # get winner_pec to loser_pec vector
    winner_to_loser = (smooth_trajectories[:, loserIdx, 1] - smooth_trajectories[:, winnerIdx, 1])[:, :2]
    winner_to_loser = _normalize_2D_vector_timeseries(winner_to_loser)

    # get loser_pec to winner_pec
    loser_to_winner = (smooth_trajectories[:, winnerIdx, 1] - smooth_trajectories[:, loserIdx, 1])[:, :2]
    loser_to_winner = _normalize_2D_vector_timeseries(loser_to_winner)

    # compute the thetas
    vec1 = winner_heading_XY
    vec2 = winner_to_loser
    theta_w = _compute_angle_between_2D_vectors(vec1[:,0], vec1[:,1], vec2[:,0], vec2[:,1])
    theta_w = _map_02Pi_array_to_minusPiPi_array(theta_w)

    vec1 = loser_heading_XY
    vec2 = loser_to_winner
    theta_l = _compute_angle_between_2D_vectors(vec1[:,0], vec1[:,1], vec2[:,0], vec2[:,1])
    theta_l = _map_02Pi_array_to_minusPiPi_array(theta_l)

    return theta_w, theta_l



def _compute_angle_between_2D_vectors(vec1_x, vec1_y, vec2_x, vec2_y):
    ''' Compute the angle needed to change direction from the heading of vec1
        to the heading of vec2
    '''
    return np.arctan2(vec2_y, vec2_x) - np.arctan2(vec1_y, vec1_x)


def _normalize_2D_vector_timeseries(vec_ts):
    ''' vec_ts: (N,2) - shaped timeseries
    '''
    row_norms = np.linalg.norm(vec_ts, axis=1)
    vec_ts = vec_ts / row_norms[:, np.newaxis]
    return vec_ts

def _map_02Pi_array_to_minusPiPi_array(arr):
    ''' Given a timeseries of angles in the range (0, 2pi),
        return the same timeseries in the range (-pi, pi)
    '''
    out_arr = np.copy(arr)
    too_big_idxs = np.where(arr > np.pi)[0]
    too_neg_idxs = np.where(arr < -np.pi)[0]
    out_arr[too_big_idxs] -= 2*np.pi
    out_arr[too_neg_idxs] += 2*np.pi
    return out_arr





