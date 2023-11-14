''' This file contains the following classes

    CoordTransformer_14D_HH

    CoordTransformer_14D_PP


'''
import numpy as np
import pandas as pd
#import h5py
#import os
#import sys
#sys.path.append('../../calibration/')






class CoordTransformer_14D_HH(object):
    ''' A class to generate the 14D coordinate system centered on the 2 heads.
    '''


    def __init__(self, trajectories, winnerIdx, numProjectionModes=14, rep_14D_tseries=None):
        ''' Class constructor:

        -- args --
        trajectories: a (numFrames, numFish=2, numBodyPoints=3, 3) array.
                      Post-processed tracking results. NaNs allowed
        winnerIdx   : the index of the winner fish (0 or 1)

        -- kwargs --
        numProjectionModes: the number of PCA modes to project the 14D onto

        '''
        # parse inputs
        self.trajectories = trajectories
        self.numFrames, self.numFish, self.numBodyPoints, _ = self.trajectories.shape
        self.winnerIdx = winnerIdx
        self.loserIdx = np.mod(self.winnerIdx+1, 2)
        self.numProjectionModes = numProjectionModes
        # perform PCA calculations
        self.data_order = self._set_data_ordering()
        if rep_14D_tseries is not None:
            self.rep_14D_tseries = rep_14D_tseries
        else:
            self.rep_14D_tseries = self._compute_14D_rep()
        self.covmat = self._compute_covariance_matrix_of_14D_rep()
        self.mean_rep = self._compute_mean_14D_rep()
        self.eigvals, self.eigvecs = self._eigendecompose_14D_covariance_matrix()
        self.cum_var_explained = self._compute_variance_explained()
        self.pca_tseries = self._project_14D_rep_onto_numProjectionModes()


    # ----- Public Methods ------- #

    def convert_mode_weights_to_14D_rep(self, mode_weights):
        ''' Convert these mode weight to a 14D representation using the eigenvectors
        '''
        return mode_weights.dot(self.eigvecs[:, :len(mode_weights)].T) + self.mean_rep

    def convert_rep14D_to_mode_weights(self, single_frame_14D_rep, numModes=14):
        ''' Project this 14D rep onto the modes.
        '''
        mode_weights = np.dot( (single_frame_14D_rep-self.mean_rep), self.eigvecs[:, :numModes])
        return mode_weights

    def convert_rep14D_to_plotable_positions(self, single_frame_14D_rep):
        ''' Given the 14 representation, return the bodypoints in this coordinate system
            in the more traditional format, for ease of plotting

        -- args --
        single_frame_14D_rep

        -- returns --
        frame_positions: (numFish=2, numBodyPoints=3, 3)
        '''
        # extract the 6 points
        winner_head = np.array([single_frame_14D_rep[0], 0, single_frame_14D_rep[1]])
        loser_head = np.array([single_frame_14D_rep[0]*-1, 0, single_frame_14D_rep[1]*-1])
        winner_pec = single_frame_14D_rep[2:5]
        winner_tail = single_frame_14D_rep[5:8]
        loser_pec = single_frame_14D_rep[8:11]
        loser_tail = single_frame_14D_rep[11:14]

        # combine into the output array
        winner_points = np.vstack([winner_head, winner_pec, winner_tail])
        loser_points = np.vstack([loser_head, loser_pec, loser_tail])
        frame_positions = np.stack((winner_points, loser_points))
        return frame_positions



    # ----- Private methods -------#



    # __init__ convenience functions

    def _compute_14D_rep(self):
        ''' Return the single-frame 14D representation of these trajectories.

        -- returns --
        rep_14D_tseries: shape (numFrame, 14)

        -- see also --
        self.data_order
        self._convert_single_frame_tracks_into_14D_rep()
        '''
        rep_14D_tseries = np.zeros((self.numFrames, 14))
        for fIdx in range(self.numFrames):
            frame_data = self.trajectories[fIdx]
            rep_14D_tseries[fIdx] = self._convert_single_frame_tracks_into_14D_rep(frame_data)
        return rep_14D_tseries

    def _convert_single_frame_tracks_into_14D_rep(self, frame_data):
        ''' Given the trajectories for a single frame, frame_data,
            and the index of the winner fish, winnerIdx,
            return the 14D representation
        '''

        # get the winner and loser positions
        winner_lab = frame_data[self.winnerIdx]
        loser_lab = frame_data[self.loserIdx]

        # get the origin as the average position of the two heads
        bpIdx = 0 # head
        origin = np.average([winner_lab[bpIdx], loser_lab[bpIdx]], axis=0)

        # get the x_hat direction
        # This runs from the head of the winner to the origin, and has 0 lab z component
        x_hat = np.zeros((3,))
        winnerHead_to_origin = (origin - winner_lab[0])
        x_hat[:2] = winnerHead_to_origin[:2]  # take XY components
        x_hat = x_hat / np.linalg.norm(x_hat) # normalize

        # now define z_hat along z_lab, and hence define y_hat
        z_hat = np.array([0, 0, 1])
        y_hat = np.cross(z_hat, x_hat)

        # determine the Q matrix
        Q = np.array([x_hat, y_hat, z_hat]).T

        # get each of the non-head bodypoints in this coordinate system
        origin_to_winner_tail = winner_lab[2] - origin
        origin_to_winner_pec = winner_lab[1] - origin
        origin_to_loser_tail = loser_lab[2] - origin
        origin_to_loser_pec = loser_lab[1] - origin

        winner_tail = np.linalg.solve(Q, origin_to_winner_tail)
        winner_pec = np.linalg.solve(Q, origin_to_winner_pec)
        loser_tail = np.linalg.solve(Q, origin_to_loser_tail)
        loser_pec  = np.linalg.solve(Q, origin_to_loser_pec)

        # now get the last two DOF
        # the of the winner and loser in this coordinate system, due to choice of origin,
        # will have no y_hat components, and the same x_hat and z_hat components, save for a sign flip
        # So we really only have 2 independent degrees of freedom
        # By convetion we can take the winner's values
        origin_to_winner_head = winner_lab[0] - origin
        winner_head = np.linalg.solve(Q, origin_to_winner_head)
        winner_head_x_hat_comp = winner_head[0]
        winner_head_z_hat_comp = winner_head[2]

        # create the final 14 D representation
        rep_14D = np.zeros((14,))
        rep_14D[0] = winner_head_x_hat_comp
        rep_14D[1] = winner_head_z_hat_comp
        rep_14D[2:5] = winner_pec
        rep_14D[5:8] = winner_tail
        rep_14D[8:11] = loser_pec
        rep_14D[11:] = loser_tail
        return rep_14D


    def _set_data_ordering(self):
        ''' Instantiate a list containing the explaination of
            each of the 14 DOF
        '''
        data_order = [ 'winner_head_x', 'winner_head_z',
                       'winner_pec_x', 'winner_pec_y', 'winner_pec_z',
                       'winner_tail_x', 'winner_tail_y', 'winner_tail_z',
                       'loser_pec_x', 'loser_pec_y', 'loser_pec_z',
                       'loser_tail_x', 'loser_tail_y', 'loser_tail_z']
        return data_order


    def _compute_covariance_matrix_of_14D_rep(self):
        ''' Compute the cov matrix of the timeseries data
        '''
        covmat = np.ma.cov(np.ma.masked_invalid(self.rep_14D_tseries), rowvar=False)
        return covmat


    def _compute_mean_14D_rep(self):
        ''' Compute the mean configuration
        '''
        return np.nanmean(self.rep_14D_tseries, axis=0)


    def _eigendecompose_14D_covariance_matrix(self):
        ''' Compute the eigenspectrum, with eigenvectors sorted by
            eigvals highest to lowest
        '''
        # get the sorted eigenspectrum
        eig_vals, eig_vecs = np.linalg.eig(self.covmat)
        idxs = np.argsort(eig_vals)
        idxs = idxs[::-1]
        sorted_eig_vals = eig_vals[idxs]
        sorted_eig_vecs = eig_vecs[:, idxs]
        sorted_eig_vecs = sorted_eig_vecs.data
        return sorted_eig_vals, sorted_eig_vecs

    def _compute_variance_explained(self):
        ''' Return a timeseries of the variance explained by successive modes
        '''
        variance_explained = self.eigvals/ np.sum(self.eigvals)
        cum_variance_explained = np.cumsum(variance_explained)
        return cum_variance_explained

    def _project_14D_rep_onto_numProjectionModes(self):
        ''' Project 14D rep onto pca space, using the number of modes
            set by self.numProjectionModes
        '''
        projection_eig_vecs = np.copy(self.eigvecs[:, :self.numProjectionModes])
        mean_subtraced_14D_tseries = self.rep_14D_tseries - self.mean_rep
        pca_tseries = np.dot(mean_subtraced_14D_tseries, projection_eig_vecs)
        return pca_tseries










class CoordTransformer_14D_PP(object):
    ''' A class to generate the 14D coordinate system centered on the 2 pec.
    '''


    def __init__(self, trajectories, winnerIdx, numProjectionModes=14, rep_14D_tseries=None):
        ''' Class constructor:

        -- args --
        trajectories: a (numFrames, numFish=2, numBodyPoints=3, 3) array.
                      Post-processed tracking results. NaNs allowed
        winnerIdx   : the index of the winner fish (0 or 1)

        -- kwargs --
        numProjectionModes: the number of PCA modes to project the 14D onto

        '''
        # parse inputs
        self.trajectories = trajectories
        self.numFrames, self.numFish, self.numBodyPoints, _ = self.trajectories.shape
        self.winnerIdx = winnerIdx
        self.loserIdx = np.mod(self.winnerIdx+1, 2)
        self.numProjectionModes = numProjectionModes
        # perform PCA calculations
        self.data_order = self._set_data_ordering()
        if rep_14D_tseries is not None:
            self.rep_14D_tseries = rep_14D_tseries
        else:
            self.rep_14D_tseries = self._compute_14D_rep()
        self.covmat = self._compute_covariance_matrix_of_14D_rep()
        self.mean_rep = self._compute_mean_14D_rep()
        self.eigvals, self.eigvecs = self._eigendecompose_14D_covariance_matrix()
        self.cum_var_explained = self._compute_variance_explained()
        self.pca_tseries = self._project_14D_rep_onto_numProjectionModes()


    # ----- Public Methods ------- #

    def convert_mode_weights_to_14D_rep(self, mode_weights):
        ''' Convert these mode weight to a 14D representation using the eigenvectors
        '''
        return mode_weights.dot(self.eigvecs[:, :len(mode_weights)].T) + self.mean_rep

    def convert_rep14D_to_mode_weights(self, single_frame_14D_rep, numModes=14):
        ''' Project this 14D rep onto the modes.
        '''
        mode_weights = np.dot( (single_frame_14D_rep-self.mean_rep), self.eigvecs[:, :numModes])
        return mode_weights

    def convert_rep14D_to_plotable_positions(self, single_frame_14D_rep):
        ''' Given the 14 representation, return the bodypoints in this coordinate system
            in the more traditional format, for ease of plotting

        -- args --
        single_frame_14D_rep

        -- returns --
        frame_positions: (numFish=2, numBodyPoints=3, 3)
        '''
        # extract the 6 points
        winner_pec = np.array([single_frame_14D_rep[0], 0, single_frame_14D_rep[1]])
        loser_pec = np.array([single_frame_14D_rep[0]*-1, 0, single_frame_14D_rep[1]*-1])
        winner_head = single_frame_14D_rep[2:5]
        winner_tail = single_frame_14D_rep[5:8]
        loser_head = single_frame_14D_rep[8:11]
        loser_tail = single_frame_14D_rep[11:14]

        # combine into the output array
        winner_points = np.vstack([winner_head, winner_pec, winner_tail])
        loser_points = np.vstack([loser_head, loser_pec, loser_tail])
        frame_positions = np.stack((winner_points, loser_points))

        return frame_positions



    # ----- Private methods -------#



    # __init__ convenience functions

    def _compute_14D_rep(self):
        ''' Return the single-frame 14D representation of these trajectories.

        -- returns --
        rep_14D_tseries: shape (numFrame, 14)

        -- see also --
        self.data_order
        self._convert_single_frame_tracks_into_14D_rep()
        '''
        rep_14D_tseries = np.zeros((self.numFrames, 14))
        for fIdx in range(self.numFrames):
            frame_data = self.trajectories[fIdx]
            rep_14D_tseries[fIdx] = self._convert_single_frame_tracks_into_14D_rep(frame_data)
        return rep_14D_tseries

    def _convert_single_frame_tracks_into_14D_rep(self, frame_data):
        ''' Given the trajectories for a single frame, frame_data,
            and the index of the winner fish, winnerIdx,
            return the 14D representation
        '''

        # get the winner and loser positions
        winner_lab = frame_data[self.winnerIdx]
        loser_lab = frame_data[self.loserIdx]

        # get the origin as the average position of the two pecs
        bpIdx = 1 # pec
        origin = np.average([winner_lab[bpIdx], loser_lab[bpIdx]], axis=0)

        # get the x_hat direction
        # This runs from the pec of the winner to the origin, and has 0 lab z component
        x_hat = np.zeros((3,))
        winnerPec_to_origin = (origin - winner_lab[1])
        x_hat[:2] = winnerPec_to_origin[:2]  # take XY components
        x_hat = x_hat / np.linalg.norm(x_hat) # normalize

        # now define z_hat along z_lab, and hence define y_hat
        z_hat = np.array([0, 0, 1])
        y_hat = np.cross(z_hat, x_hat)

        # determine the Q matrix
        Q = np.array([x_hat, y_hat, z_hat]).T

        # get each of the non-pec bodypoints in this coordinate system
        origin_to_winner_tail = winner_lab[2] - origin
        origin_to_winner_head = winner_lab[0] - origin
        origin_to_loser_tail = loser_lab[2] - origin
        origin_to_loser_head = loser_lab[0] - origin

        winner_tail = np.linalg.solve(Q, origin_to_winner_tail)
        winner_head = np.linalg.solve(Q, origin_to_winner_head)
        loser_tail = np.linalg.solve(Q, origin_to_loser_tail)
        loser_head  = np.linalg.solve(Q, origin_to_loser_head)

        # now get the last two DOF
        # the of the winner and loser in this coordinate system, due to choice of origin,
        # will have no y_hat components, and the same x_hat and z_hat components, save for a sign flip
        # So we really only have 2 independent degrees of freedom
        # By convetion we can take the winner's values
        origin_to_winner_pec = winner_lab[1] - origin
        winner_pec = np.linalg.solve(Q, origin_to_winner_pec)
        winner_pec_x_hat_comp = winner_pec[0]
        winner_pec_z_hat_comp = winner_pec[2]

        # create the final 14 D representation
        rep_14D = np.zeros((14,))
        rep_14D[0] = winner_pec_x_hat_comp
        rep_14D[1] = winner_pec_z_hat_comp
        rep_14D[2:5] = winner_head
        rep_14D[5:8] = winner_tail
        rep_14D[8:11] = loser_head
        rep_14D[11:] = loser_tail
        return rep_14D


    def _set_data_ordering(self):
        ''' Instantiate a list containing the explaination of
            each of the 14 DOF
        '''
        data_order = [ 'winner_pec_x', 'winner_pec_z',
                       'winner_head_x', 'winner_head_y', 'winner_head_z',
                       'winner_tail_x', 'winner_tail_y', 'winner_tail_z',
                       'loser_head_x', 'loser_head_y', 'loser_head_z',
                       'loser_tail_x', 'loser_tail_y', 'loser_tail_z']
        return data_order


    def _compute_covariance_matrix_of_14D_rep(self):
        ''' Compute the cov matrix of the timeseries data
        '''
        covmat = np.ma.cov(np.ma.masked_invalid(self.rep_14D_tseries), rowvar=False)
        return covmat


    def _compute_mean_14D_rep(self):
        ''' Compute the mean configuration
        '''
        return np.nanmean(self.rep_14D_tseries, axis=0)


    def _eigendecompose_14D_covariance_matrix(self):
        ''' Compute the eigenspectrum, with eigenvectors sorted by
            eigvals highest to lowest
        '''
        # get the sorted eigenspectrum
        eig_vals, eig_vecs = np.linalg.eig(self.covmat)
        idxs = np.argsort(eig_vals)
        idxs = idxs[::-1]
        sorted_eig_vals = eig_vals[idxs]
        sorted_eig_vecs = eig_vecs[:, idxs]
        sorted_eig_vecs = sorted_eig_vecs.data
        return sorted_eig_vals, sorted_eig_vecs

    def _compute_variance_explained(self):
        ''' Return a timeseries of the variance explained by successive modes
        '''
        variance_explained = self.eigvals/ np.sum(self.eigvals)
        cum_variance_explained = np.cumsum(variance_explained)
        return cum_variance_explained

    def _project_14D_rep_onto_numProjectionModes(self):
        ''' Project 14D rep onto pca space, using the number of modes
            set by self.numProjectionModes
        '''
        projection_eig_vecs = np.copy(self.eigvecs[:, :self.numProjectionModes])
        mean_subtraced_14D_tseries = self.rep_14D_tseries - self.mean_rep
        pca_tseries = np.dot(mean_subtraced_14D_tseries, projection_eig_vecs)
        return pca_tseries


    
    
    
# -------- Utility functions ---- #

def project_14D_rep_onto_modes(rep_14D_tseries, rep_14D_eigvecs, numProjectionModes=14):
    ''' Project 14D rep onto pca space
    '''
    projection_eig_vecs = np.copy(rep_14D_eigvecs[:, :numProjectionModes])
    mean_subtraced_14D_tseries = rep_14D_tseries - np.nanmean(rep_14D_tseries, axis=0)
    pca_tseries = np.dot(mean_subtraced_14D_tseries, projection_eig_vecs)
    return pca_tseries





def compute_14D_rep_from_trajectories(trajectory, winnerIdx):
    ''' Return the single-frame 14D representation of these trajectories.
    
    -- args --
    trajectory: (numFrames, numFish, numBodyPoints, 3)
                The tracking results
    winnerIdx: the fishIdx of the winner

    -- returns --
    rep_14D_tseries: shape (numFrame, 14)
    
    -- see also --
    _convert_single_frame_tracks_into_14D_rep()

    '''
    numFrames, numFish, numBodyPoints, _ = trajectory.shape
    rep_14D_tseries = np.zeros((numFrames, 14))
    for fIdx in range(numFrames):
        frame_data = trajectory[fIdx]
        rep_14D_tseries[fIdx] = _convert_single_frame_tracks_into_14D_rep(frame_data, winnerIdx)
    return rep_14D_tseries

def _convert_single_frame_tracks_into_14D_rep(frame_data, winnerIdx):
    ''' Given the trajectories for a single frame, frame_data,
        and the index of the winner fish, winnerIdx,
        return the 14D representation
    '''
    loserIdx = np.mod(winnerIdx+1, 2)

    # get the winner and loser positions
    winner_lab = frame_data[winnerIdx]
    loser_lab = frame_data[loserIdx]

    # get the origin as the average position of the two pecs
    bpIdx = 1 # pec
    origin = np.average([winner_lab[bpIdx], loser_lab[bpIdx]], axis=0)

    # get the x_hat direction
    # This runs from the pec of the winner to the origin, and has 0 lab z component
    x_hat = np.zeros((3,))
    winnerPec_to_origin = (origin - winner_lab[1])
    x_hat[:2] = winnerPec_to_origin[:2]  # take XY components
    x_hat = x_hat / np.linalg.norm(x_hat) # normalize

    # now define z_hat along z_lab, and hence define y_hat
    z_hat = np.array([0, 0, 1])
    y_hat = np.cross(z_hat, x_hat)

    # determine the Q matrix
    Q = np.array([x_hat, y_hat, z_hat]).T

    # get each of the non-pec bodypoints in this coordinate system
    origin_to_winner_tail = winner_lab[2] - origin
    origin_to_winner_head = winner_lab[0] - origin
    origin_to_loser_tail = loser_lab[2] - origin
    origin_to_loser_head = loser_lab[0] - origin

    winner_tail = np.linalg.solve(Q, origin_to_winner_tail)
    winner_head = np.linalg.solve(Q, origin_to_winner_head)
    loser_tail = np.linalg.solve(Q, origin_to_loser_tail)
    loser_head  = np.linalg.solve(Q, origin_to_loser_head)

    # now get the last two DOF
    # the of the winner and loser in this coordinate system, due to choice of origin,
    # will have no y_hat components, and the same x_hat and z_hat components, save for a sign flip
    # So we really only have 2 independent degrees of freedom
    # By convetion we can take the winner's values
    origin_to_winner_pec = winner_lab[1] - origin
    winner_pec = np.linalg.solve(Q, origin_to_winner_pec)
    winner_pec_x_hat_comp = winner_pec[0]
    winner_pec_z_hat_comp = winner_pec[2]

    # create the final 14 D representation
    rep_14D = np.zeros((14,))
    rep_14D[0] = winner_pec_x_hat_comp
    rep_14D[1] = winner_pec_z_hat_comp
    rep_14D[2:5] = winner_head
    rep_14D[5:8] = winner_tail
    rep_14D[8:11] = loser_head
    rep_14D[11:] = loser_tail
    return rep_14D














# ------------------ Extracting the 10D coordinate system ----------------------- #

def compute_11D_rep_from_trajectory(trajectory_data, winnerIdx):
    ''' Returnt the 11D representation for this trajectory data
    '''
    numFrames, numFish, numBodyPoints, _ = trajectory_data.shape
    rep10D = np.zeros((numFrames, 11))
    for fIdx in range(numFrames):
        whas, wtas, lhas, ltas, wpecx, wpecz, phi = return_thetas_psis_pecInfo_phi_for_frame(trajectory_data[fIdx], winnerIdx)
        rep10D[fIdx,0:2] = whas
        rep10D[fIdx,2:4] = wtas
        rep10D[fIdx,4:6] = lhas
        rep10D[fIdx,6:8] = ltas
        rep10D[fIdx,8] = wpecx
        rep10D[fIdx,9] = wpecz
        rep10D[fIdx,10] = phi
    return rep10D





def return_thetas_psis_pecInfo_phi_for_frame(frame_data, winnerIdx):
    ''' Return the thetas & psis for each bodypoint, the pec x and z,
        and the x' heading phi, from the trajectory data for a single frame

    '''
    # get the winner and loser positions
    loserIdx = np.mod(winnerIdx+1, 2)
    winner_lab = frame_data[winnerIdx]
    loser_lab = frame_data[loserIdx]

    # ---- Find the origin and the new basis ---- #

    # get the origin as the average position of the two pecs
    bpIdx = 1 # pec
    origin = np.average([winner_lab[bpIdx], loser_lab[bpIdx]], axis=0)

    # get the x_hat direction
    # This runs from the pec of the winner to the origin, and has 0 lab z component
    x_hat = np.zeros((3,))
    winnerPec_to_origin = (origin - winner_lab[1])
    x_hat[:2] = winnerPec_to_origin[:2]  # take XY components
    x_hat = x_hat / np.linalg.norm(x_hat) # normalize

    # now define z_hat along z_lab, and hence define y_hat
    z_hat = np.array([0, 0, 1])
    y_hat = np.cross(z_hat, x_hat)

    # determine the Q matrix
    Q = np.array([x_hat, y_hat, z_hat]).T

    # --- Compute phi, the orientation of the pec-pec line in lab coords --- #
    phi = np.arctan2(x_hat[1], x_hat[0])


    # --- the two distance DOF ---- #
    # the of the winner and loser in this coordinate system, due to choice of origin,
    # will have no y_hat components, and the same x_hat and z_hat components, save for a sign flip
    # So we really only have 2 independent degrees of freedom
    # By convetion we can take the winner's values
    origin_to_winner_pec = frame_data[winnerIdx, 1] - origin
    origin_to_loser_pec = frame_data[loserIdx, 1] - origin
    winner_pec = np.linalg.solve(Q, origin_to_winner_pec)
    loser_pec = np.linalg.solve(Q, origin_to_loser_pec)
    winner_pec_x_hat_comp = winner_pec[0]
    winner_pec_z_hat_comp = winner_pec[2]


    # --- computing the angles ---- #
    # Now i want to write the winner_head, winner_tail, loser_head and loser_tail as angles,
    # and i will keep their lengths constant. 
    # Notation: w=winner, l=loser, p=pec, h=head
    w_ph_lab = frame_data[winnerIdx, 0] - frame_data[winnerIdx, 1]
    w_pt_lab = frame_data[winnerIdx, 2] - frame_data[winnerIdx, 1]
    l_ph_lab = frame_data[loserIdx, 0] - frame_data[loserIdx, 1]
    l_pt_lab = frame_data[loserIdx, 2] - frame_data[loserIdx, 1]

    # convert the bodypoint vectors to the new coordinate system
    w_ph = np.linalg.solve(Q, w_ph_lab)
    w_pt = np.linalg.solve(Q, w_pt_lab)
    l_ph = np.linalg.solve(Q, l_ph_lab)
    l_pt = np.linalg.solve(Q, l_pt_lab)

    # the Psis
    psi_w_h = get_psi_from_XYZ_vec(w_ph)
    psi_w_t = get_psi_from_XYZ_vec(w_pt)
    psi_l_h = get_psi_from_XYZ_vec(l_ph)
    psi_l_t = get_psi_from_XYZ_vec(l_pt)

    # the thetas
    tet_w_h = get_tet_from_XYZ_vec(w_ph)
    tet_w_t = get_tet_from_XYZ_vec(w_pt)
    tet_l_h = get_tet_from_XYZ_vec(l_ph)
    tet_l_t = get_tet_from_XYZ_vec(l_pt)

    winner_head_angles = np.array([tet_w_h, psi_w_h])
    winner_tail_angles = np.array([tet_w_t, psi_w_t])
    loser_head_angles = np.array([tet_l_h, psi_l_h])
    loser_tail_angles = np.array([tet_l_t, psi_l_t])
    return winner_head_angles, winner_tail_angles, loser_head_angles, loser_tail_angles, winner_pec_x_hat_comp, winner_pec_z_hat_comp, phi



def get_psi_from_XYZ_vec(vec):
    ''' Return the psi angle (elevation) for this vector.

    -- args --
    vec: a 3D vector in the coordinates system basis

    -- returns --
    psi: angle in rads in range [-pi/2, pi/2]
    '''
    z = vec[2]
    xy = np.sqrt( (vec[0])**2 + (vec[1])**2 )
    psi = np.arctan2(z, xy)
    return psi


def get_tet_from_XYZ_vec(vec):
    ''' Return the tet angle (heading relative to X') for this vector

    -- args --
    vec: a 3D vector in the coordinates system basis

    -- returns --
    tet: angle in rads in range [-pi, pi]
    '''
    y = vec[1]
    x = vec[0]
    tet = np.arctan2(y, x)
    return tet