''' In this library, we have functions for computing kinematic variables from trajectories,
    and functions for computing windowed distributions to these variables.
'''
import numpy as np
import warnings
import sys
#sys.path.append('./')
from post_processing import process_1D_timeseries


# --- utility functions ---- #

def compute_angle_between_2D_vectors(vec1_x, vec1_y, vec2_x, vec2_y):
    ''' Compute the angle needed to change direction from the heading of vec1
        to the heading of vec2
    '''
    return np.arctan2(vec2_y, vec2_x) - np.arctan2(vec1_y, vec1_x)


def normalize_2D_vector_timeseries(vec_ts):
    ''' vec_ts: (N,2) - shaped timeseries
    '''
    row_norms = np.linalg.norm(vec_ts, axis=1)
    vec_ts = vec_ts / row_norms[:, np.newaxis]
    return vec_ts

def map_02Pi_array_to_minusPiPi_array(arr):
    ''' Given a timeseries of angles in the range (0, 2pi),
        return the same timeseries in the range (-pi, pi)
    '''
    out_arr = np.copy(arr)
    too_big_idxs = np.where(arr > np.pi)[0]
    too_neg_idxs = np.where(arr < -np.pi)[0]
    out_arr[too_big_idxs] -= 2*np.pi
    out_arr[too_neg_idxs] += 2*np.pi
    return out_arr

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





# ------ Computing Variables ----- #

def compute_pec_pec_distance(smooth_trajectories):
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


def compute_pec_pec_XY_distance(smooth_trajectories):
    ''' Compute the distance between the pec points in the XY plane.
        This is also the X' pec-pec distance in the coordinate system.

    -- returns --
    pec_pec_xy_distances: (numFrames,)

   -- Notes --
    We dont care about winner and loser here, the measure is invariant
    under fish-swap.
    '''
    fish1_pec_xy_tseries = np.copy(smooth_trajectories[:,0,1,:2])
    fish2_pec_xy_tseries = np.copy(smooth_trajectories[:,1,1,:2])
    pec_pec_xy_distances = np.linalg.norm(fish1_pec_xy_tseries - fish2_pec_xy_tseries, axis=1)
    return pec_pec_xy_distances


def compute_pec_pec_Z_distance(smooth_trajectories):
    ''' Compute the Z distance between the pec points.
        This is also the Z' pec-pec distance in the coordinate system.

    -- returns --
    pec_pec_z_distances: (numFrames,)

    -- Notes --
    We dont care about winner and loser here, the measure is invariant
    under fish-swap.
    '''
    fish1_pec_z_tseries = np.copy(smooth_trajectories[:,0,1,2])
    fish2_pec_z_tseries = np.copy(smooth_trajectories[:,1,1,2])
    pec_pec_z_distances = np.abs(fish1_pec_z_tseries - fish2_pec_z_tseries)
    return pec_pec_z_distances


def compute_pec_pec_distance_dot(raw_trajectories,  dt=0.01, interp_max_gap=7, interp_polyOrd=1, smooth_polyOrd=2, smooth_winSize=11):
    ''' Compute the time derivative of the pec-pec distances for the two fish.

    -- args --
    raw_trajectories:

    -- returns --
    pec_pec_distances_dot

    -- Notes --
    We dont care about winner and loser here, the measure is invariant
    under fish-swap.

    Use raw trajectories, to avoid using the filter twice.
    '''
    fish1_pec_tseries = np.copy(raw_trajectories[:,0,1,:])
    fish2_pec_tseries = np.copy(raw_trajectories[:,1,1,:])
    pec_pec_distances_raw = np.linalg.norm(fish1_pec_tseries - fish2_pec_tseries, axis=1)

    pec_pec_distances_dot = process_1D_timeseries(pec_pec_distances_raw,
                                                  interp_max_gap=interp_max_gap,
                                                  interp_polyOrd=interp_polyOrd,
                                                  smooth_polyOrd=smooth_polyOrd,
                                                  smooth_winSize=smooth_winSize,
                                                  deriv=1,
                                                  dt=dt)

    return pec_pec_distances_dot

def compute_pec_pec_distance_dot_dot(raw_trajectories,  dt=0.01, interp_max_gap=7, interp_polyOrd=1, smooth_polyOrd=2, smooth_winSize=11):
    ''' Compute the 2nd time derivative of the pec-pec distances for the two fish.

    -- args --
    raw_trajectories:

    -- returns --
    pec_pec_distances_dot

    -- Notes --
    We dont care about winner and loser here, the measure is invariant
    under fish-swap.

    Use raw trajectories, to avoid using the filter twice.
    '''
    fish1_pec_tseries = np.copy(raw_trajectories[:,0,1,:])
    fish2_pec_tseries = np.copy(raw_trajectories[:,1,1,:])
    pec_pec_distances_raw = np.linalg.norm(fish1_pec_tseries - fish2_pec_tseries, axis=1)

    pec_pec_distances_dot_dot = process_1D_timeseries(pec_pec_distances_raw,
                                                      interp_max_gap=interp_max_gap,
                                                      interp_polyOrd=interp_polyOrd,
                                                      smooth_polyOrd=smooth_polyOrd,
                                                      smooth_winSize=smooth_winSize,
                                                      deriv=2,
                                                      dt=dt)
    return pec_pec_distances_dot_dot



def get_winner_and_loser_pec_Z_positions(smooth_trajectories, winnerIdx, loserIdx):
    ''' Return the lab-Z position of the pec of the winner and the pec of the loser
    '''
    winner_pec_z = smooth_trajectories[:,winnerIdx,1,2]
    loser_pec_z = smooth_trajectories[:,loserIdx,1,2]
    return winner_pec_z, loser_pec_z


def compute_signed_pec_z_difference(smooth_trajectories, winnerIdx, loserIdx):
    ''' Compute the pecZ coord of the winner minus the pecZ coord of the loser
        (The signed pec difference)

    -- returns --
    signed_pec_pec_z_difference: (numFrames,). Winner above loser is positive.

    -- see also --
    get_winner_and_loser_pec_Z_positions()
    '''
    winner_pec_z, loser_pec_z = get_winner_and_loser_pec_Z_positions(smooth_trajectories, winnerIdx, loserIdx)
    signed_pec_pec_z_difference = winner_pec_z - loser_pec_z
    return signed_pec_pec_z_difference


def compute_phi_from_trajectories(trajectories, winnerIdx, loserIdx):
    ''' Compute phi, the the orientation of the pec-pec line in lab coords,
        from winner to loser.

    -- returns --
    phi: [-pi, pi] range 1D tseries
    '''
    # winner and lose pec positions in XY
    winner_pec_ts = np.copy(trajectories[:, winnerIdx, 1, :2])
    loser_pec_ts = np.copy(trajectories[:, loserIdx, 1, :2])

    # get the unit vector tseries of winner_pec to loser_pec headings
    winnerPec_to_loserPec = loser_pec_ts - winner_pec_ts
    row_norms = np.linalg.norm(winnerPec_to_loserPec, axis=1)
    winnerPec_to_loserPec = winnerPec_to_loserPec / row_norms[:, np.newaxis]

    # compute and return phi
    phi = np.arctan2(winnerPec_to_loserPec[:,1], winnerPec_to_loserPec[:,0])
    return phi


def compute_phi_dot_from_raw_trajectories(raw_trajectories, winnerIdx, loserIdx, dt=0.01,
                                          interp_max_gap=7, interp_polyOrd=1, smooth_polyOrd=2, smooth_winSize=11):
    ''' Compute the phi-dot timeseries, by first computing phi_raw from raw_trajectories,
        then unwrapping, and taking the first derivative with a savgol filter.

    -- see also --
    compute_phi_from_trajectories()

    -- returns --
    phi_dot:
    '''
    phi_raw = compute_phi_from_trajectories(raw_trajectories, winnerIdx, loserIdx)

    # Before I compute phi-dot, i need to unwrap
    phi_raw_unwrapped = np.copy(phi_raw)
    phi_raw_unwrapped[~np.isnan(phi_raw_unwrapped)] = np.unwrap(phi_raw_unwrapped[~np.isnan(phi_raw_unwrapped)])

    # compute phi-dot
    phi_dot = process_1D_timeseries(phi_raw_unwrapped,
                                   interp_max_gap=interp_max_gap,
                                   interp_polyOrd=interp_polyOrd,
                                   smooth_polyOrd=smooth_polyOrd,
                                   smooth_winSize=smooth_winSize,
                                   deriv=1,
                                   dt=dt)
    return phi_dot



def compute_thetaW_and_thetaL(smooth_trajectories, winnerIdx, loserIdx):
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
    winner_heading_XY = normalize_2D_vector_timeseries(winner_heading_XY)

    # loser-pec to loser_head vector
    loser_heading_XY =  (smooth_trajectories[:, loserIdx, 0] - smooth_trajectories[:, loserIdx, 1])[:, :2]
    loser_heading_XY = normalize_2D_vector_timeseries(loser_heading_XY)


    # get winner_pec to loser_pec vector
    winner_to_loser = (smooth_trajectories[:, loserIdx, 1] - smooth_trajectories[:, winnerIdx, 1])[:, :2]
    winner_to_loser = normalize_2D_vector_timeseries(winner_to_loser)

    # get loser_pec to winner_pec
    loser_to_winner = (smooth_trajectories[:, winnerIdx, 1] - smooth_trajectories[:, loserIdx, 1])[:, :2]
    loser_to_winner = normalize_2D_vector_timeseries(loser_to_winner)

    # compute the thetas
    vec1 = winner_heading_XY
    vec2 = winner_to_loser
    theta_w = compute_angle_between_2D_vectors(vec1[:,0], vec1[:,1], vec2[:,0], vec2[:,1])
    theta_w = map_02Pi_array_to_minusPiPi_array(theta_w)

    vec1 = loser_heading_XY
    vec2 = loser_to_winner
    theta_l = compute_angle_between_2D_vectors(vec1[:,0], vec1[:,1], vec2[:,0], vec2[:,1])
    theta_l = map_02Pi_array_to_minusPiPi_array(theta_l)

    return theta_w, theta_l


def compute_coordinate_origin(smooth_trajectories, winnerIdx, loserIdx):
    ''' Compute the timeseries of the origin of the coordinate system,
        namely the centroid average of the winner and loser pec locations.

    -- returns --
    origin_timeseries: (numFrames,3) the XYZ timeseries of the
                       coordinate system origin location.

    -- Notes --
    We dont care about winner and loser here, the measure is invariant
    under fish-swap.

    '''
    bpIdx = 1 #pec
    fish1_pec_tseries = np.copy(smooth_trajectories[:, 0, bpIdx, :])
    fish2_pec_tseries = np.copy(smooth_trajectories[:, 1, bpIdx, :])

    # make (numFrames,2) shaped tseries of the x,y,z coordinates of both fish
    xs_tseries = np.stack([fish1_pec_tseries[:,0], fish2_pec_tseries[:,0]], axis=1)
    ys_tseries = np.stack([fish1_pec_tseries[:,1], fish2_pec_tseries[:,1]], axis=1)
    zs_tseries = np.stack([fish1_pec_tseries[:,2], fish2_pec_tseries[:,2]], axis=1)

    # now average across the 2 fish, for each component
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        fishmean_xs_tseries = np.nanmean(xs_tseries, axis=1)
        fishmean_ys_tseries = np.nanmean(ys_tseries, axis=1)
        fishmean_zs_tseries = np.nanmean(zs_tseries, axis=1)

    # now combine the components in the final (numFrames,3) shaped array
    origin_timeseries = np.stack([fishmean_xs_tseries, fishmean_ys_tseries, fishmean_zs_tseries], axis=1)
    return origin_timeseries



def compute_coordinate_origin_z(smooth_trajectories, winnerIdx, loserIdx):
    ''' Compute the timeseries of the Z-component of the origin of the
        coordinate system, namely the Z-component of centroid average
        of the winner and loser pec locations.

    --- returns ---
    origin_z_timeseries: (numFrames,) the timeseries of the Z location
                         of the coordinates system origin.

    --- see also ---
    compute_coordinate_origin
    '''
    origin_timeseries = compute_coordinate_origin(smooth_trajectories, winnerIdx, loserIdx)
    origin_z_timeseries = origin_timeseries[:,2]
    return origin_z_timeseries


def compute_coordinate_origin_z_dot(raw_trajectories, winnerIdx, loserIdx, dt=0.01,
                                    interp_max_gap=7, interp_polyOrd=1,
                                    smooth_polyOrd=2, smooth_winSize=11):
    ''' Compute the time derivative of timeseries of the Z-component of
        the origin of the  coordinate system, namely the Z-component of
        centroid average  of the winner and loser pec locations.

    --- Approach ---
    We are going to follow the same approach as we do to compute
    origin_z_timeseries, but this time we will use raw trajectories
    instead of smooth trajectories, and then once we are finished
    we will take a first derivative while post-processing.

    --- returns ---
    origin_z_timeseries_dot: (numFrames,) the derivative of timeseries
                             of the Z location of the coordinates system origin.

    --- see also ---
    compute_coordinate_origin
    compute_coordinate_origin_z
    '''
    origin_z_ts_raw = compute_coordinate_origin_z(raw_trajectories, winnerIdx, loserIdx)
    origin_z_timeseries_dot = process_1D_timeseries(origin_z_ts_raw,
                                                    interp_max_gap=interp_max_gap,
                                                    interp_polyOrd=interp_polyOrd,
                                                    smooth_polyOrd=smooth_polyOrd,
                                                    smooth_winSize=smooth_winSize,
                                                    deriv=1,
                                                    dt=dt)
    return origin_z_timeseries_dot



def compute_pitch_angles(smooth_trajectories, winnerIdx, loserIdx):
    ''' Compute the pitch angles of the winner and loser, defined as the angle
        between the pec-head vector and the z-level plane at the pec.

    --- Convention ---
    Pointing up is positive (head above pec), pointing down is negative.
    '''
    # compute the rise from pec to head (signed Z distance)
    w_headZ = smooth_trajectories[:, winnerIdx, 0, 2]
    w_pecZ = smooth_trajectories[:, winnerIdx, 1, 2]
    w_head_Z_rise = w_headZ - w_pecZ
    # compute the run from pec to head (unsigned XY distance)
    w_headXY = smooth_trajectories[:, winnerIdx, 1, :2]
    w_pecXY = smooth_trajectories[:, winnerIdx, 0, :2]
    w_pec_head_XY_dist = np.linalg.norm(w_headXY - w_pecXY, axis=1)
    # compute the winner pitch angle
    winner_pitch_tseries = np.arctan2(w_head_Z_rise, w_pec_head_XY_dist)

    # compute the rise from pec to head (signed Z distance)
    l_headZ = smooth_trajectories[:, loserIdx, 0, 2]
    l_pecZ = smooth_trajectories[:, loserIdx, 1, 2]
    l_head_Z_rise = l_headZ - l_pecZ
    # compute the run from pec to head (unsigned XY distance)
    l_headXY = smooth_trajectories[:, loserIdx, 1, :2]
    l_pecXY = smooth_trajectories[:, loserIdx, 0, :2]
    l_pec_head_XY_dist = np.linalg.norm(l_headXY - l_pecXY, axis=1)
    # compute the winner pitch angle
    loser_pitch_tseries = np.arctan2(l_head_Z_rise, l_pec_head_XY_dist)

    return winner_pitch_tseries, loser_pitch_tseries






# --- computing variables: old functions that aren't as useful anymore ---- #


def compute_alignment(smooth_trajectories):
    ''' Return the alignment of the two fish, defined as the dot product
        of the XY heading angles (the pec-head XY angles).
        1 is parallel, 0 is perpendicular, -1 is anti-parallel

    -- args --
    smooth_trajectories:

    -- returns --
    alignment: the XY alignment timeseries (dot product of xy headings)
    '''
    fishIdx = 0
    fish1_heading_XY =  (smooth_trajectories[:, fishIdx, 0] - smooth_trajectories[:, fishIdx, 1])[:, :2]
    fish1_heading_XY = normalize_2D_vector_timeseries(fish1_heading_XY)

    fishIdx = 1
    fish2_heading_XY =  (smooth_trajectories[:, fishIdx, 0] - smooth_trajectories[:, fishIdx, 1])[:, :2]
    fish2_heading_XY = normalize_2D_vector_timeseries(fish2_heading_XY)

    # preallocate the output
    expNfs = smooth_trajectories.shape[0]
    alignment = np.zeros((expNfs,))

    # take the dot-product of the headings for all frames
    for fIdx in range(expNfs):
        alignment[fIdx] = np.dot(fish1_heading_XY[fIdx], fish2_heading_XY[fIdx])
    return alignment



def compute_alignment_from_raw_trajectories(raw_trajectories, dt=0.01, interp_max_gap=7, interp_polyOrd=1, smooth_polyOrd=2, smooth_winSize=11):
    ''' Return the alignment of the two fish,
        defined as the dot product of the XY heading angles (the pec-head XY angles).
        1 is parallel, 0 is perpendicular, -1 is anti-parallel

    -- note --
    We compute the alignment from the raw trajectories, then interpolate and smooth.

    -- args --
    raw_trajectories:

    -- returns --
    alignment: the time derivative of XY alignment timeseries (dot product of xy headings)
    '''
    fishIdx = 0
    fish1_heading_XY =  (raw_trajectories[:, fishIdx, 0] - raw_trajectories[:, fishIdx, 1])[:, :2]
    fish1_heading_XY = normalize_2D_vector_timeseries(fish1_heading_XY)

    fishIdx = 1
    fish2_heading_XY =  (raw_trajectories[:, fishIdx, 0] - raw_trajectories[:, fishIdx, 1])[:, :2]
    fish2_heading_XY = normalize_2D_vector_timeseries(fish2_heading_XY)

    expNfs = raw_trajectories.shape[0]
    raw_alignment = np.zeros((expNfs,))

    # take the dot-product of the headings for all frames
    for fIdx in range(expNfs):
        raw_alignment[fIdx] = np.dot(fish1_heading_XY[fIdx], fish2_heading_XY[fIdx])

    # now process the raw alignment timeseries and take the derivative
    alignment = process_1D_timeseries(raw_alignment,
                                      interp_max_gap=interp_max_gap,
                                      interp_polyOrd=interp_polyOrd,
                                      smooth_polyOrd=smooth_polyOrd,
                                      smooth_winSize=smooth_winSize,
                                      deriv=0,
                                      dt=dt)
    return alignment


def compute_alignment_dot_from_raw_trajectories(raw_trajectories, dt=0.01, interp_max_gap=7, interp_polyOrd=1, smooth_polyOrd=2, smooth_winSize=11):
    ''' Return the time derivative of the alignment of the two fish,
        defined as the dot product of the XY heading angles (the pec-head XY angles).
        1 is parallel, 0 is perpendicular, -1 is anti-parallel

    -- note --
    pass raw trajectories, to avoid using the filter twice.

    -- args --
    raw_trajectories:

    -- returns --
    alignment_dot: the time derivative of XY alignment timeseries (dot product of xy headings)
    '''
    fishIdx = 0
    fish1_heading_XY =  (raw_trajectories[:, fishIdx, 0] - raw_trajectories[:, fishIdx, 1])[:, :2]
    fish1_heading_XY = normalize_2D_vector_timeseries(fish1_heading_XY)

    fishIdx = 1
    fish2_heading_XY =  (raw_trajectories[:, fishIdx, 0] - raw_trajectories[:, fishIdx, 1])[:, :2]
    fish2_heading_XY = normalize_2D_vector_timeseries(fish2_heading_XY)

    expNfs = raw_trajectories.shape[0]
    raw_alignment = np.zeros((expNfs,))

    # take the dot-product of the headings for all frames
    for fIdx in range(expNfs):
        raw_alignment[fIdx] = np.dot(fish1_heading_XY[fIdx], fish2_heading_XY[fIdx])

    # now process the raw alignment timeseries and take the derivative
    alignment_dot = process_1D_timeseries(raw_alignment,
                                          interp_max_gap=interp_max_gap,
                                          interp_polyOrd=interp_polyOrd,
                                          smooth_polyOrd=smooth_polyOrd,
                                          smooth_winSize=smooth_winSize,
                                          deriv=1,
                                          dt=dt)
    return alignment_dot





def compute_winner_and_loser_tail_elevations(smooth_trajectories, winnerIdx, loserIdx):
    ''' Compute the winner and loser tail elevation angles.

    -- TODO --
    Clean up variable naming here

    -- Approach --
    First compute the pec-head elevation in lab coords; alpha
    Then compute pec-tail elevation in lab coords; phi
    The instrinsic tail elevation is then the difference
    '''

    # -- compute the winner tail elevation -- #
    # get alpha: the head-to-pec elevation (positive if pec above head)
    # pecZ - headZ
    w_front_rise = smooth_trajectories[:, winnerIdx, 1, 2] - smooth_trajectories[:, winnerIdx, 0, 2]
    # XY distance between pec and head
    w_front_run = np.linalg.norm(smooth_trajectories[:, winnerIdx, 1, :2] - smooth_trajectories[:, winnerIdx, 0, :2], axis=1)
    alpha = np.arctan2(w_front_rise, w_front_run)

    # get phi: the pec-to-tail elevation (positive if tail above pec)
    # tailZ - pecZ
    w_back_rise = smooth_trajectories[:, winnerIdx, 2, 2] - smooth_trajectories[:, winnerIdx, 1, 2]
    # XY distance between pec and tail
    w_back_run = np.linalg.norm(smooth_trajectories[:, winnerIdx, 1, :2] - smooth_trajectories[:, winnerIdx, 2, :2], axis=1)
    phi = np.arctan2(w_back_rise, w_back_run)
    # get the tail elevation angle
    w_tail_elev = phi - alpha


    # -- compute the loser tail elevation -- #
    # get alpha: the head-to-pec elevation (positive if pec above head)
    # pecZ - headZ
    l_front_rise = smooth_trajectories[:, loserIdx, 1, 2] - smooth_trajectories[:, loserIdx, 0, 2]
    # XY distance between pec and head
    l_front_run = np.linalg.norm(smooth_trajectories[:, loserIdx, 1, :2] - smooth_trajectories[:, loserIdx, 0, :2], axis=1)
    alpha = np.arctan2(l_front_rise, l_front_run)

    # get phi: the pec-to-tail elevation (positive if tail above pec)
    # tailZ - pecZ
    l_back_rise = smooth_trajectories[:, loserIdx, 2, 2] - smooth_trajectories[:, loserIdx, 1, 2]
    # XY distance between pec and tail
    l_back_run = np.linalg.norm(smooth_trajectories[:, loserIdx, 1, :2] - smooth_trajectories[:, loserIdx, 2, :2], axis=1)
    phi = np.arctan2(l_back_rise, l_back_run)
    # get the tail elevation angle
    l_tail_elev = phi - alpha

    return w_tail_elev, l_tail_elev



def compute_alphaW_and_alphaL(smooth_trajectories, winnerIdx, loserIdx):
    ''' Return alphas, the XY tail bending angles of the winner and the loser.

    alpha is the angle between the head-pec vector and the pec-to-tail vector.
    A straight fish is defined to have alpha=0.
    Tail sweeping to the right (counterclockwise) are alpha > 0

    -- args --
    smooth_trajectories:

    -- returns --
    winner_alpha: timeseries in range (-pi,pi)
    loser_alpha: timeseries in range (-pi,pi)
    '''
    # winner-pec to winner_head vector
    winner_pecHead_XY =  (smooth_trajectories[:, winnerIdx, 0] - smooth_trajectories[:, winnerIdx, 1])[:, :2]
    winner_pecHead_XY = normalize_2D_vector_timeseries(winner_pecHead_XY)
    # winner-tail to winner_pec vector
    winner_tailPec_XY =  (smooth_trajectories[:, winnerIdx, 1] - smooth_trajectories[:, winnerIdx, 2])[:, :2]
    winner_tailPec_XY = normalize_2D_vector_timeseries(winner_tailPec_XY)
    # compute winner alpha
    winner_alpha = -compute_angle_between_2D_vectors(winner_tailPec_XY[:,0], winner_tailPec_XY[:,1], winner_pecHead_XY[:,0], winner_pecHead_XY[:,1])
    winner_alpha = map_02Pi_array_to_minusPiPi_array(winner_alpha)

    # loser-pec to loser_head vector
    loser_pecHead_XY =  (smooth_trajectories[:, loserIdx, 0] - smooth_trajectories[:, loserIdx, 1])[:, :2]
    loser_pecHead_XY = normalize_2D_vector_timeseries(loser_pecHead_XY)
    # winner-tail to winner_pec vector
    loser_tailPec_XY =  (smooth_trajectories[:, loserIdx, 1] - smooth_trajectories[:, loserIdx, 2])[:, :2]
    loser_tailPec_XY = normalize_2D_vector_timeseries(loser_tailPec_XY)
    # compute winner alpha
    loser_alpha = -compute_angle_between_2D_vectors(loser_tailPec_XY[:,0], loser_tailPec_XY[:,1], loser_pecHead_XY[:,0], loser_pecHead_XY[:,1])
    loser_alpha = map_02Pi_array_to_minusPiPi_array(loser_alpha)

    return winner_alpha, loser_alpha




