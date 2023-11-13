import numpy as np
from scipy.spatial import distance

class FightDetector(object):
    ''' A class for detecting bouts of fighting in new experiments.

    -- Methods --
    detect_fight_bouts(trajectory): detect the fight bouts in the passed
                                    experiment tracking results array.

    '''

    def __init__(self, dpp_bins, g1_bins, g2_bins, window_size, window_step,
                       initial_skip_size, prob_vectors_clusterable_data,
                       cluster_labels, fight_label_number, refining_skip_size=2,
                       merge_gap_size_in_windows=5, min_region_window_size=5):
        ''' Instantiate the object

        -- args --
        dpp_bins: the bin_edges used to discretize dpp
        g1_bins: the bin_edges used to discretize g1
        g2_bins: the bin_edges used to discretize g2
        window_size: the size of the windows, in frames, used in the clustering.
        window_step: the number of frames between the centers of windows in the clustering.
        initial_skip_size: the number of windows skipped between chosen sample windows for
                           the first pass at detecting fights, as used in the clustering.
        prob_vectors_clusterable_data: the (numSamples, numStates) prob vecs that were clustered.
        cluster_labels: the labels applied to each of the numSample number of clustered vecs.
        fight_label_number: the element of cluster_labels which means "fight"
        refining_skip_size=2: the number of windows skipped between chosen sample windows for
                              the second pass at fight-bouts, refining the boarders.
        merge_gap_size_in_windows=5: combine fight-bouts separated by less than this many windows.
        min_region_window_size=5 remove fight-bouts smaller than this.

        '''
        self.dpp_bins = dpp_bins
        self.g1_bins= g1_bins
        self.g2_bins = g2_bins
        self.window_size = window_size
        self.window_step = window_step
        self.prob_vectors_clusterable_data = prob_vectors_clusterable_data
        self.cluster_labels = cluster_labels
        self.fight_label_number = fight_label_number
        self.initial_skip_size = initial_skip_size
        self.refining_skip_size = refining_skip_size
        self.merge_gap_size_in_windows = merge_gap_size_in_windows
        self.min_region_window_size = min_region_window_size


    def detect_fight_bouts(self, trajectory):
        ''' Return the frame ranges of detected fight bouts for this experiment.

        --- args ---
        trajectory: array, shape (numFrames, numFish=2, numBodyPoints=3, 3),
                    the tracking results for a fish fighting experiment.

        --- returns ---
        refined_fight_info: array, shape (numBouts, 2), containing the start
                            and stop frame for each detected bout.

        '''
        dpp_ts, g1_ts, g2_ts = self._compute_dpp_g1_g2_from_trajectory(trajectory)

        exp_time_windows = self._prepare_time_windows(trajectory.shape[0])

        exp_prob_vecs, exp_prob_vec_wins = self._compute_dpp_g1_g2_windowed_prob_vecs(dpp_ts,
                                                                                      g1_ts,
                                                                                      g2_ts,
                                                                                      exp_time_windows)

        exp_prob_vecs_labels = self._find_cluster_labels_of_exp_prob_vecs(exp_prob_vecs)

        fight_region_window_indices = self._find_contiguous_regions_of_fight_vecs(exp_prob_vecs_labels)

        merged_frwis = self._merge_close_contiguous_fight_regions(fight_region_window_indices)

        merged_threshed_frwis = self._remove_small_contiguous_fight_regions(merged_frwis)

        fight_info = self._return_fight_boundaries_in_frames(merged_threshed_frwis, exp_prob_vec_wins)
        if fight_info.shape[0] == 0:
            return fight_info
        else:
            refined_fight_info = self._refine_the_boarders_of_fight_bouts(fight_info, dpp_ts, g1_ts, g2_ts)
            return refined_fight_info




    # ------------------------------------------------------------------#

    def _compute_dpp_g1_g2_from_trajectory(self, trajectory):
        # we will use default Idxs here, since identity will be destroyed anyway
        dumWinIdx=0
        dumLosIdx=1
        dpp_ts = self._compute_pec_pec_distance(trajectory)
        tet1, tet2 = self._compute_thetaW_and_thetaL(trajectory, dumWinIdx, dumLosIdx)
        g1_ts, g2_ts = self._symmetrize_tet1_tet2(tet1, tet2)
        return dpp_ts, g1_ts, g2_ts


    def _prepare_time_windows(self, expNumFrames):
        # get the windows for the experiment at full resolution
        time_windows = self._return_overlapping_windows_for_timeframes(expNumFrames)
        # downsample the windows to the level that we want
        decimated_time_windows = time_windows[::self.initial_skip_size, :]
        return decimated_time_windows


    def _compute_dpp_g1_g2_windowed_prob_vecs(self, dpp_ts, g1_ts, g2_ts, time_wins):
        ''' Return the 1D vectors from binning (Dpp,g1,g2) in windows, and return
            the window start/stops also.

        --- returns ---
        exp_prob_vecs: shape (numWins, numStatesInVecs) - the distributions in each window
        exp_prob_vec_wins: shape (numWins,2) - the start/stop frames of the windows.
        '''
        # compute the vectors
        prob_vectors = self._get_1D_prob_vectors_from_dpp_g1_g2_in_timeWins(dpp_ts, g1_ts, g2_ts,
                                                                            time_wins)
        # now remove any NaN distributions, and the associated time_wins
        exp_prob_vecs = prob_vectors[ np.where(~np.isnan(prob_vectors).any(axis=1))[0] ]
        exp_prob_vec_wins = time_wins[ np.where(~np.isnan(prob_vectors).any(axis=1))[0] ]
        return exp_prob_vecs, exp_prob_vec_wins


    def _find_cluster_labels_of_exp_prob_vecs(self, exp_prob_vecs):
        ''' Find the nearest neighbour of each prob-vec when compared to clustered vecs,
            and hence find the cluster label of each prob-vec.
        '''
        exp_prob_vecs_labels = self._find_cluster_labels_for_input_samples_via_NN_search_of_clustered_data(exp_prob_vecs)

        return exp_prob_vecs_labels


    def _find_contiguous_regions_of_fight_vecs(self, exp_prob_vecs_labels):
        # find a binary timeseries of fight cluster membership
        # Vec=0 means no fight, Vec=1 means fight
        fight_bin_tseries = np.zeros_like(exp_prob_vecs_labels)
        fight_bin_tseries[exp_prob_vecs_labels==self.fight_label_number] = 1
        # get the fight regions - contiguous regions of fight cluster membership
        fight_region_window_indices = self._contiguous_regions(fight_bin_tseries)
        return fight_region_window_indices


    def _merge_close_contiguous_fight_regions(self, fight_region_window_indices):
        ''' Combine together fight bouts that are closer than "merge_gap_size_in_windows"
            number of windows apart.
        '''
        num_fight_regions = fight_region_window_indices.shape[0]
        # prepare the list to hold the results
        merged_fight_region_window_indices = []
        # start the variable which will update as we step through the list
        # this contains the start and stop window idxs for each region
        current_region_info = fight_region_window_indices[0]

        # step through, merging as far into the future as we can
        for ii in range(num_fight_regions-1):
            # can i merge current and next bout?
            curr_region_end_idx = current_region_info[1]
            next_region_start_idx = fight_region_window_indices[ii+1][0]
            num_indexs_between_regions = next_region_start_idx - curr_region_end_idx
            if num_indexs_between_regions < self.merge_gap_size_in_windows:
                do_merge_bouts = True
            else:
                do_merge_bouts = False

            if do_merge_bouts:
                current_region_start = current_region_info[0]
                next_region_end = fight_region_window_indices[ii+1][1]
                current_region_info = np.array([current_region_start, next_region_end])
                continue
            else:
                # record what we have
                merged_fight_region_window_indices.append(current_region_info)
                # move current info to next bout
                current_region_info = fight_region_window_indices[ii+1]
                continue

        # record the last region
        merged_fight_region_window_indices.append(current_region_info)
        # convert to array and record results for this experiment
        merged_fight_region_window_indices = np.array(merged_fight_region_window_indices)
        return merged_fight_region_window_indices



    def _remove_small_contiguous_fight_regions(self, fight_region_window_indices):
        ''' Remove any contiguous regions of fighting that are smaller than
            "min_region_window_size" in duration.
        '''
        # get the region sizes
        region_sizes_in_windows = fight_region_window_indices[:,1] - fight_region_window_indices[:,0]
        # get the regIdxs to remove, and remove them
        remove_reg_idxs = np.where(region_sizes_in_windows < self.min_region_window_size)[0]
        threshed_fight_region_window_indices = np.delete(fight_region_window_indices, remove_reg_idxs, 0)
        return threshed_fight_region_window_indices


    def _return_fight_boundaries_in_frames(self, fight_region_window_indices, exp_prob_vec_wins):
        ''' Return fight_info, an array of shape (numBouts,2) containing the start&stop frames of each bout.
            The start is the mean of the first time window, and the stop is the mean of the last time window.
        '''
        numBouts = fight_region_window_indices.shape[0]
        bouts_startStops = []
        for boutIdx in range(numBouts):
            bout_all_window_frames = exp_prob_vec_wins[fight_region_window_indices[boutIdx,0]:fight_region_window_indices[boutIdx,1]]
            bout_startStop_frames = np.array([np.mean(bout_all_window_frames[0]), np.mean(bout_all_window_frames[-1])])
            bouts_startStops.append(bout_startStop_frames)
        # convert to array and return
        fight_info = np.array(bouts_startStops)
        return fight_info


    def _refine_the_boarders_of_fight_bouts(self, fight_info, dpp_ts, g1_ts, g2_ts):
        ''' Given the fight_info, shape (numBouts,2), the start and stop frame for a number of bouts,
            and given dpp,g1,g2 timeseries for the assoicated data,
            return a refined version of fight_info, where by clustering distributions around the start
            and stop of bouts we determine the boards to greater accuracy than in the first pass.

        --- args ---
        fight_info: shape (numBouts,2), the start/stop frames for a number of fight-bouts
        dpp_ts: the dpp timeseries for the experiment fight_info refers to
        g1_ts: the g1 timeseries for the experiment fight_info refers to
        g2_ts: the g2 timeseries for the experiment fight_info refers to

        --- returns ---
        refined_fight_info: shape (numBouts,2), the start/stop frames for a number of fight-bouts,
                            where start/stop is more accurate than in fight_info

        '''
        refined_fight_info = []

        numBouts = fight_info.shape[0]
        exp_nfs = dpp_ts.shape[0]
        for boutIdx in range(numBouts):
            old_bout_startFrame, old_bout_stopFrame = fight_info[boutIdx]

            # create padding windows around the bout, then downsample as you wish
            bout_padding_windows = self._create_sliding_windows_around_detected_bout(old_bout_startFrame,
                                                                                     old_bout_stopFrame,
                                                                                     exp_nfs)
            bout_padding_windows = bout_padding_windows[::self.refining_skip_size]


            # compute the vectors
            prob_vectors_for_padding = self._get_1D_prob_vectors_from_dpp_g1_g2_in_timeWins(dpp_ts, g1_ts, g2_ts,
                                                                                            bout_padding_windows)
            # now remove any NaN distributions, and the associated time_wins
            prob_vectors_for_padding_nanless = prob_vectors_for_padding[ np.where(~np.isnan(prob_vectors_for_padding).any(axis=1))[0] ]
            padding_windows_nanless = bout_padding_windows[ np.where(~np.isnan(prob_vectors_for_padding).any(axis=1))[0] ]

            # cluster the prob vecs
            prob_vec_labels = self._find_cluster_labels_for_input_samples_via_NN_search_of_clustered_data(prob_vectors_for_padding_nanless)

            # the first window labelled fight is the new start, and the last window labelled fight is new stop
            first_fight_index = np.where(prob_vec_labels==self.fight_label_number)[0][0]
            last_fight_index = np.where(prob_vec_labels==self.fight_label_number)[0][-1]
            first_win = padding_windows_nanless[first_fight_index]
            last_win = padding_windows_nanless[last_fight_index]
            bout_start = np.mean(first_win)
            bout_stop = np.mean(last_win)
            bout_info = np.array([bout_start, bout_stop])

            # record
            refined_fight_info.append(bout_info)

        # combine results for all bouts and return
        refined_fight_info = np.stack(refined_fight_info, axis=0)
        return refined_fight_info



    # ---------------------------------------------------------------#

    def _compute_pec_pec_distance(self, trajectory):
        ''' Compute the timeseries of pec-pec distances for the two fish.

        -- args --
        trajectory: (numFrames,numFish=2,numBodyPoints=3,3)

        -- returns --
        pec_pec_distances: (numFrames,)

        -- Notes --
        We dont care about winner and loser here, the measure is invariant
        under fish-swap.

        '''
        fish1_pec_tseries = np.copy(trajectory[:,0,1,:])
        fish2_pec_tseries = np.copy(trajectory[:,1,1,:])
        pec_pec_distances = np.linalg.norm(fish1_pec_tseries - fish2_pec_tseries, axis=1)
        return pec_pec_distances



    def _compute_thetaW_and_thetaL(self, trajectory, winnerIdx, loserIdx):
        ''' Return relative headings of the winner and the loser.

        theta_w is defined as the angle needed to rotate the winner XY heading
        to point directly towards the loser.

        theta_l is defined as the angle need to rotate the loser XY heading to
        point direectly towards the winner.

        Positive angles are counterclockwise.

        -- args --
        trajectory: (numFrames,numFish=2,numBodyPoints=3,3)

        -- returns --
        theta_w: timeseries in range (-pi,pi)
        theta_l: timeseries in range (-pi,pi)
        '''
        # winner-pec to winner_head vector
        winner_heading_XY =  (trajectory[:, winnerIdx, 0] - trajectory[:, winnerIdx, 1])[:, :2]
        winner_heading_XY = self._normalize_2D_vector_timeseries(winner_heading_XY)

        # loser-pec to loser_head vector
        loser_heading_XY =  (trajectory[:, loserIdx, 0] - trajectory[:, loserIdx, 1])[:, :2]
        loser_heading_XY = self._normalize_2D_vector_timeseries(loser_heading_XY)


        # get winner_pec to loser_pec vector
        winner_to_loser = (trajectory[:, loserIdx, 1] - trajectory[:, winnerIdx, 1])[:, :2]
        winner_to_loser = self._normalize_2D_vector_timeseries(winner_to_loser)

        # get loser_pec to winner_pec
        loser_to_winner = (trajectory[:, winnerIdx, 1] - trajectory[:, loserIdx, 1])[:, :2]
        loser_to_winner = self._normalize_2D_vector_timeseries(loser_to_winner)

        # compute the thetas
        vec1 = winner_heading_XY
        vec2 = winner_to_loser
        theta_w = self._compute_angle_between_2D_vectors(vec1[:,0], vec1[:,1], vec2[:,0], vec2[:,1])
        theta_w = self._map_02Pi_array_to_minusPiPi_array(theta_w)

        vec1 = loser_heading_XY
        vec2 = loser_to_winner
        theta_l = self._compute_angle_between_2D_vectors(vec1[:,0], vec1[:,1], vec2[:,0], vec2[:,1])
        theta_l = self._map_02Pi_array_to_minusPiPi_array(theta_l)

        return theta_w, theta_l


    def _symmetrize_tet1_tet2(self, tet1, tet2):
        ''' Use a symmetric function to transform tet1 into tet2 into variables
            where the order of tet1 and tet2 doesn't matter.

        -- args --
        tet1: arr shape (numFrames,)
        tet2: arr shape (numFrames,)

        -- returns --
        g1_arr: arr shape (numFrames,) ->  tet1 + tet2
        g2_arr: arr shape (numFrames,) -> |tet1 - tet2|
        '''
        g1_arr =  self._sum_theta_angles(tet1, tet2)
        g2_arr =  self._abs_diff_theta_angles(tet1, tet2)
        return g1_arr, g2_arr


    def _normalize_2D_vector_timeseries(self, vec_ts):
        ''' vec_ts: (N,2) - shaped timeseries
        '''
        row_norms = np.linalg.norm(vec_ts, axis=1)
        vec_ts = vec_ts / row_norms[:, np.newaxis]
        return vec_ts

    def _map_02Pi_array_to_minusPiPi_array(self, arr):
        ''' Given a timeseries of angles in the range (0, 2pi),
            return the same timeseries in the range (-pi, pi)
        '''
        out_arr = np.copy(arr)
        too_big_idxs = np.where(arr > np.pi)[0]
        too_neg_idxs = np.where(arr < -np.pi)[0]
        out_arr[too_big_idxs] -= 2*np.pi
        out_arr[too_neg_idxs] += 2*np.pi
        return out_arr


    def _compute_angle_between_2D_vectors(self, vec1_x, vec1_y, vec2_x, vec2_y):
        ''' Compute the angle needed to change direction from the heading of vec1
            to the heading of vec2
        '''
        return np.arctan2(vec2_y, vec2_x) - np.arctan2(vec1_y, vec1_x)


    def _sum_theta_angles(self, tet1, tet2):
        ''' Add the two arrays of theta angles, keeping the (-pi,pi) range.

        -- args --
        tet1: arr shape (numFrames,)
        tet2: arr shape (numFrames,)

        -- returns --
        sum_thetas: arr shape (numFrames)
        '''
        # map to (0,2pi) range
        alpha1 = tet1 + np.pi
        alpha2 = tet2 + np.pi
        # sum
        alpha = alpha1 + alpha2
        #  apply boundary
        beta = np.mod(alpha, 2*np.pi)
        # map back to -pi, pi
        sum_thetas = self._map_02Pi_array_to_minusPiPi_array(beta)
        return sum_thetas

    def _abs_diff_theta_angles(self, tet1, tet2):
        ''' Subtract the two arrays of theta angles, keeping the (-pi,pi) range,
            then take the absolute value.

        -- args --
        tet1: arr shape (numFrames,)
        tet2: arr shape (numFrames,)

        -- returns --
        abs_diff_thetas: arr shape (numFrames)
        '''
        # map to (0,2pi) range
        alpha1 = tet1 + np.pi
        alpha2 = tet2 + np.pi
        alpha = alpha1 - alpha2
        # sum and apply boundary
        beta = np.mod(alpha, 2*np.pi)
        # map back to -pi, pi, then take the absolute value
        abs_diff_thetas = np.abs(self._map_02Pi_array_to_minusPiPi_array(beta))
        return abs_diff_thetas


    def _return_overlapping_windows_for_timeframes(self, numFrames):
        ''' Given a number of frames, return an 2D array of window start-stop frames.
        '''
        # define, for clarity, the first window
        win0_end = int(self.window_size)

        # find numWindows, by adding incrementally and watching the last frame
        last_frame_in_windows = win0_end
        numWindows = 1
        while last_frame_in_windows < (numFrames - self.window_step):
            numWindows += 1
            last_frame_in_windows = win0_end + (numWindows-1)*self.window_step

        # now fill-in the windows array of frame indices
        windows = np.zeros((numWindows, 2))
        windows[0, 0] = 0
        windows[0, 1] = win0_end
        for winIdx in range(1, numWindows):
            w0 = winIdx*self.window_step
            wF = w0 + self.window_size
            windows[winIdx, 0] = w0
            windows[winIdx, 1] = wF
        return windows.astype(int)



    def _get_1D_prob_vectors_from_dpp_g1_g2_in_timeWins(self, dpp, g1, g2, timeWins):
        ''' Return the 1D probability vector from the 3D histogramming of (dpp,g1,g2),
            using the spatial bins provided, and the timeWins provided. timeWins may be a decimated
            time_windows array.

        These 1D probability vectors represenent the probability of being in each of the microstates
        defined by the spatial binning, in that particular window of time.

        --- args ---
        dpp: (numFrames,)
        g1: (numFrames,)
        g2: (numFrames,)
        timeWins: (numWindows, win_start_and_stop)

        --- returns ---
        prob_vectors_for_wins: (numWindows, num_dpp_bins*num_g1_bins*num_g2_bins),

        '''

        # parse info
        hist_bins = [self.dpp_bins, self.g1_bins, self.g2_bins]
        if ~ dpp.shape[0] == g1.shape[0] == g2.shape[0]:
            raise TypeError('dpp, g1, g2 are not the same length')
        numWindows = timeWins.shape[0]

        # --------- get the prob vectors for all windows ------ #
        probs = []
        for winIdx in range(numWindows):
            # parse the data for this windows
            f0,fE = timeWins[winIdx]
            # grab the data
            win_g1 = g1[f0:fE]
            win_g2 = g2[f0:fE]
            win_dpp = dpp[f0:fE]
            # Make (N,D) data
            win_data = np.stack([win_dpp, win_g1, win_g2], axis=1)
            # get the probs
            hist_counts, hist_edges = np.histogramdd(win_data, bins=hist_bins, density=True)
            hist_probs = hist_counts / np.sum(hist_counts)
            probs.append(hist_probs)
        # convert probs list to array
        prob_vectors_for_wins = np.array(probs)
        # cast dpp-theta-theta dist into 1D form (first dim runs along windows)
        prob_vectors_for_wins = prob_vectors_for_wins.reshape(prob_vectors_for_wins.shape[0], -1)
        return prob_vectors_for_wins


    def _contiguous_regions(self, bool_array):
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


    def _create_sliding_windows_around_detected_bout(self, bout_detected_start_frame,
                                                     bout_detected_end_frame,
                                                     exp_nfs):
        ''' Given the start and end frame of a fight bout detected by the clustering (decimated),
            create some sliding windows either side of the fight, to probe if the bout can be
            extended futher backwards of forwards in time (to refine the start and stop frames).
        '''
        bout_start_padding_windows = self._create_padding_windows_before_bout_start_frame(bout_detected_start_frame,
                                                                                          exp_nfs)
        bout_stop_padding_windows = self._create_padding_windows_before_bout_stop_frame(bout_detected_end_frame,
                                                                                        exp_nfs)

        bout_padding_windows = np.concatenate([bout_start_padding_windows, bout_stop_padding_windows])
        return bout_padding_windows


    def _create_padding_windows_before_bout_start_frame(self, start_frame, exp_nfs):
        ''' Given a frame index, create an array of sliding windows before the frame.
        '''
        # create the padding windows
        padding_starts = np.arange(start_frame-(self.window_size)+self.window_step, start_frame, self.window_step)
        padding_stops = padding_starts + self.window_size
        padding_windows = np.stack([padding_starts, padding_stops], axis=1)

        # remove any problematic windows
        bad_idxs = []
        good_idxs = []
        for winIdx in range(padding_windows.shape[0]):
            winF0, winFE = padding_windows[winIdx]
            if winF0 < 0:
                bad_idxs.append(winIdx)
            elif winFE < 0:
                bad_idxs.append(winIdx)
            elif winFE >= exp_nfs:
                bad_idxs.append(winIdx)
            else:
                good_idxs.append(winIdx)
        padding_windows_parsed = []
        for idx in good_idxs:
            padding_windows_parsed.append(padding_windows[idx])
        padding_windows_parsed = np.array(padding_windows_parsed)

        # finally, return a simply padding without to avoid problems if empty
        if padding_windows_parsed.shape[0] == 0:
            padding_windows_parsed = np.array([start_frame, start_frame+self.window_size]).reshape(1,2)

        return padding_windows_parsed.astype(int)


    def _create_padding_windows_before_bout_stop_frame(self, stop_frame, exp_nfs):
        ''' Given a frame index, create an array of sliding windows after the frame.
        '''
        # create the padding windows
        padding_starts = np.arange(stop_frame-self.window_size+self.window_step, stop_frame, self.window_step)
        padding_stops = padding_starts + self.window_size
        padding_windows = np.stack([padding_starts, padding_stops], axis=1)

        # remove any problematic windows
        bad_idxs = []
        good_idxs = []
        for winIdx in range(padding_windows.shape[0]):
            winF0, winFE = padding_windows[winIdx]
            if winF0 < 0:
                bad_idxs.append(winIdx)
            elif winFE < 0:
                bad_idxs.append(winIdx)
            elif winFE >= exp_nfs:
                bad_idxs.append(winIdx)
            else:
                good_idxs.append(winIdx)
        padding_windows_parsed = []
        for idx in good_idxs:
            padding_windows_parsed.append(padding_windows[idx])
        padding_windows_parsed = np.array(padding_windows_parsed)

        # finally, return a simply padding without to avoid problems if empty
        if padding_windows_parsed.shape[0] == 0:
            padding_windows_parsed = np.array([stop_frame, stop_frame+self.window_size]).reshape(1,2)

        return padding_windows_parsed.astype(int)


    def _find_cluster_labels_for_input_samples_via_NN_search_of_clustered_data(self, sample_arr):
        ''' Given an array of shape ((numSamples,numStates),), containing data we want to cluster,
            find cluster labels for each sample by finding the NN to each sample in the clustered data,
            and using label of the NN as the label of the sample.

        --- args ---
        sample_arr1: shape (numSamples=g, States=N). An array containing 'g' samples, each of shape (N,)

        --- returns ---
        sample_arr1_labels: shape (numSamples=g,) -> the cluster labels for sample_arr

        '''
        # compute the distance matrix between prob_vectors_for_padding_nanless and clustered data
        distmat = self._compute_pairwise_distance_matrix_from_two_sample_arrays_using_JSD(sample_arr,
                                                                                          self.prob_vectors_clusterable_data)

        # find the cluster labels of the prob_vectors_for_padding_nanless
        sample_arr1_labels = self._find_NN_for_newData_oldData_distance_matrix(distmat,
                                                                               self.cluster_labels)
        return sample_arr1_labels


    def _compute_pairwise_distance_matrix_from_two_sample_arrays_using_JSD(self, sample_arr1, sample_arr2):
        ''' Given two arrays of shape (numSamples,numStates), concatenate the arrays into a master set,
            and compute the all-to-all distance matrix using the scipy.distance.jensenshannon metric

        --- args ---
        sample_arr1: shape (numSamples=g, States=N)
        sample_arr2: shape (numSamples=f, States=N)

        --- returns ---
        distmat: shape (g, f)

        '''
        numArr1Vecs = sample_arr1.shape[0]
        numArr2Vecs = sample_arr2.shape[0]
        distmat = np.zeros((numArr1Vecs, numArr2Vecs))
        for vec1_idx in range(numArr1Vecs):
            for vec2_idx in range(numArr2Vecs):
                vec_vec_dist  = distance.jensenshannon(sample_arr1[vec1_idx], sample_arr2[vec2_idx], base=2)
                distmat[vec1_idx, vec2_idx] = vec_vec_dist
        return distmat

    def _find_NN_for_newData_oldData_distance_matrix(self, dmat, cluster_labels):
        ''' Given a distance matrix of shape (g, f), find the NN of each of the 'g' elements,
            and hence a cluster label.

        --- args ---
        dmat: shape (g, f)
        cluster_labels: shape (f,) - labelled for each of the 'f' elements.

        --- returns ---
        new_labels: shape (g,) - the labels assigned to each of the 'g' elements

        '''
        NN_idxs = np.argmin(dmat, axis=1)
        new_labels = cluster_labels[NN_idxs]
        return new_labels
