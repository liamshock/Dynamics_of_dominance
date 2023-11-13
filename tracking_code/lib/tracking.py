import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
import os
from joblib import load





class Calibration(object):
    ''' A class for loading and using the calibrations for the cameras

    -- Methods --
    compute_imageCoord_triplet_from_XYZ
    compute_XYZ_from_imageCoord_triplet
    compute_XZ_imcoords_from_XY_YZ
    compute_XY_imcoords_from_XZ_YZ
    compute_YZ_imcoords_from_XZ_XY
    compute_point_correspondence_error

    '''

    def __init__(self, calibration_folder_path):
        ''' Instantiate the object

        -- args --
        calibration_folder_path: the path to a calibration folder, where the regressed functions
                                 have already been computed and saved

        '''
        # record the folder paths
        self.calibration_folder_path = calibration_folder_path
        self.python_calibration_folderPath = os.path.join(self.calibration_folder_path,
                                                          'python_calibration_models')

        # load the models and assign as attributes
        self._load_models()


    def _load_models(self):
        ''' Instantiate the regression object attributes:
            xyz_getter, imCoord_getter, xz_getter, xy_getter, yz_getter
        '''
        imCoords_to_XYZ_path = os.path.join(self.python_calibration_folderPath, 'imCoords_to_XYZ.joblib')
        # XYZ_to_imCoords_path = os.path.join(self.python_calibration_folderPath, 'XYZ_to_imCoords.joblib')
        xy_yz_to_xz_path = os.path.join(self.python_calibration_folderPath, 'xy_yz_to_xz.joblib')
        xz_yz_to_xy_path = os.path.join(self.python_calibration_folderPath, 'xz_yz_to_xy.joblib')
        xz_xy_to_yz_path = os.path.join(self.python_calibration_folderPath, 'xz_xy_to_yz.joblib')

        self.xyz_getter = load(imCoords_to_XYZ_path)
        # self.imCoord_getter = load(XYZ_to_imCoords_path)
        self.xz_getter = load(xy_yz_to_xz_path)
        self.xy_getter = load(xz_yz_to_xy_path)
        self.yz_getter = load(xz_xy_to_yz_path)
        return


    # ---- Main Methods ---- #

    def compute_imageCoord_triplet_from_XYZ(self, XYZ):
        ''' Predict the image coordinates in all 3 camera views of the
            3D point XYZ

        -- inputs --
        XYZ: array (3,), the position of a point in 3D

        -- returns --
        imCoords: array (3,2) of image coordinates in standard camera
                  order of XZ,XY,YZ
        '''
        imCoords = self.imCoord_getter.predict(XYZ.reshape(1,-1))
        imCoords = imCoords.reshape(3,2)
        return imCoords


    def compute_XYZ_from_imageCoord_triplet(self, imCoords):
        ''' Predict the XYZ position of the point given by the image
            coordinates from all 3 cameras

        -- Inputs --
        imCoords: array of shape (3,2)

        -- Outputs --
        XYZ: array of shape (3)

        '''
        XYZ = self.xyz_getter.predict(imCoords.reshape(-1,6))
        return XYZ


    def compute_XZ_imcoords_from_XY_YZ(self, xy_imCoord, yz_imCoord):
        ''' Given an image coordinate from both the XY and YZ views,
            compute the corresponding image coordinate from the XZ view

        -- args --
        xy_imCoord: image coordinate of shape (2,)
        yz_imCoord: image coordinate of shape (2,)

        -- returns --
        xz_imCoord: image coordinate of shape (2,)

        '''
        input_data = np.hstack((xy_imCoord, yz_imCoord)).reshape(1,4)
        xz_imCoord = self.xz_getter.predict(input_data)
        return xz_imCoord

    def compute_XY_imcoords_from_XZ_YZ(self, xz_imCoord, yz_imCoord):
        ''' Given an image coordinate from both the XZ and YZ views,
            compute the corresponding image coordinate from the XY view

        -- args --
        xz_imCoord: image coordinate of shape (2,)
        yz_imCoord: image coordinate of shape (2,)

        -- returns --
        xy_imCoord: image coordinate of shape (2,)
        '''
        # prepare the input for predictor, and predict the imcoord
        input_data = np.hstack((xz_imCoord, yz_imCoord)).reshape(1,4)
        xy_imCoord = self.xy_getter.predict(input_data)
        return xy_imCoord

    def compute_YZ_imcoords_from_XZ_XY(self, xz_imCoord, xy_imCoord):
        ''' Given an image coordinate from both the XY and YZ views,
            compute the corresponding image coordinate from the XZ view

        -- args --
        xz_imCoord: image coordinate of shape (2,)
        xy_imCoord: image coordinate of shape (2,)

        -- returns --
        yz_imCoord: image coordinate of shape (2,)

        '''
        # prepare the input for predictor, and predict the imcoord
        input_data = np.hstack((xz_imCoord, xy_imCoord)).reshape(1,4)
        yz_imCoord = self.yz_getter.predict(input_data)
        return yz_imCoord


    def compute_point_correspondence_error(self, camIdxs, imCoords_cam1, imCoords_cam2):
        ''' Compute the error of making a cross-camera association between these points

        -- args --
        camIdxs: a list denoting the cameras the imCoords args are coming from.
                 Has to be [0,1], [1,2], or [0, 2]
        imCoords_cam1: image coordinates from a camera
        imCoords_cam2: image coordinates from a different camera


        -- returns --
        error: a scalar error value for making this association
        '''
        # STEP 0: The error is NaN if either point is NaN
        if np.all(np.isnan(imCoords_cam1)) or np.all(np.isnan(imCoords_cam2)):
            return np.NaN

        # STEP 1: Compute the proposed image coordinate triplet
        if camIdxs == [0,1]:
            # derive YZ
            imCoords_cam3 = self.compute_YZ_imcoords_from_XZ_XY(imCoords_cam1, imCoords_cam2)
            proposed_imCoords = np.vstack((imCoords_cam1, imCoords_cam2, imCoords_cam3))
        elif camIdxs == [0, 2]:
            # derive XY
            imCoords_cam3 = self.compute_XY_imcoords_from_XZ_YZ(imCoords_cam1, imCoords_cam2)
            proposed_imCoords = np.vstack((imCoords_cam1, imCoords_cam3, imCoords_cam2))
        elif camIdxs == [1, 2]:
            # derive XZ
            imCoords_cam3 = self.compute_XZ_imcoords_from_XY_YZ(imCoords_cam1, imCoords_cam2)
            proposed_imCoords = np.vstack((imCoords_cam3, imCoords_cam1, imCoords_cam2))


        # STEP 2: Compute the errors

        # For each pairing of cameras, compute the 3rd cam image coordinate,
        # then compare this triplet to the proposed_imCoords, which act as truth
        # Note1: If this is a good pairing, then proposed_imCoords represent the same point in 3D
        # Note2: for one of these camera pairings test, we will get back an error of 0,
        #        since we did the same computation to compute proposed_coordinates.
        # Note3: to deal with note2, we define the error as the maximum of the 3 errors
        derived_xz = self.compute_XZ_imcoords_from_XY_YZ(proposed_imCoords[1], proposed_imCoords[2])
        image_coords_derXZ = np.vstack((derived_xz, proposed_imCoords[1], proposed_imCoords[2]))
        error_derXZ = np.linalg.norm(proposed_imCoords - image_coords_derXZ)

        derived_xy = self.compute_XY_imcoords_from_XZ_YZ(proposed_imCoords[0], proposed_imCoords[2])
        image_coords_derXY = np.vstack((proposed_imCoords[0], derived_xy, proposed_imCoords[2]))
        error_derXY = np.linalg.norm(proposed_imCoords - image_coords_derXY)

        derived_yz = self.compute_YZ_imcoords_from_XZ_XY(proposed_imCoords[0], proposed_imCoords[1])
        image_coords_derYZ = np.vstack((proposed_imCoords[0], proposed_imCoords[1], derived_yz))
        error_derYZ = np.linalg.norm(proposed_imCoords - image_coords_derYZ)

        errors = np.vstack((error_derXZ, error_derXY, error_derYZ))
        error = np.sum(errors)

        return error









def fill_in_bad_camera_view_image_coordinates(frame_instances, good_cams, calOb):
    ''' Given frame instances, and two chosen camera views, return a set of frame instances where
        the values of the image coordinates in the unchosen third camera are replaced by values
        calculated using the calibration object and the image coordinates from the chosen views.

    --- args ---
    frame_instances: (numCams=3, numFish=2, numBodyPoints=3, numImCoords=2)
    calOb: an instantiated calibration object
    good_cams: list of two idxs showing which cams to use.
               good_cam_pairings = [ [0,1], [0,2], [1,2] ]
               So, good_cams[0] = [0,1], meaning XZ and XY cameras

    --- returns ---
    filled_in_frame_instances: (numCams=3, numFish=2, numBodyPoints=3, numImCoords=2)
                               Frame instances with imagecoordinates for the unsed camera
                               view set by the calibration to be consistent with chosen views.

    '''
    # parse shapes
    _, numFish, numBodyPoints, _ = frame_instances.shape

    # preallocate an output array
    filled_in_frame_instances = np.copy(frame_instances)

    # find the bad cam idx
    if good_cams == [0,1]:
        bad_cam_idx = 2
    elif good_cams == [1,2]:
        bad_cam_idx = 0
    elif good_cams == [0, 2]:
        bad_cam_idx = 1

    if bad_cam_idx == 0:
        for fishIdx in range(numFish):
            for bpIdx in range(numBodyPoints):
                xy_imcoord = frame_instances[1, fishIdx, bpIdx]
                yz_imcoord = frame_instances[2, fishIdx, bpIdx]
                if ~np.all(np.isnan(xy_imcoord)) and ~np.all(np.isnan(yz_imcoord)):
                    xz_imcoord = calOb.compute_XZ_imcoords_from_XY_YZ(xy_imcoord, yz_imcoord)
                else:
                    xz_imcoord = np.array([np.NaN, np.NaN])
                filled_in_frame_instances[bad_cam_idx, fishIdx, bpIdx] = xz_imcoord

    elif bad_cam_idx == 1:
        for fishIdx in range(numFish):
            for bpIdx in range(numBodyPoints):
                xz_imcoord = frame_instances[0, fishIdx, bpIdx]
                yz_imcoord = frame_instances[2, fishIdx, bpIdx]
                if ~np.all(np.isnan(xz_imcoord)) and ~np.all(np.isnan(yz_imcoord)):
                    xy_imcoord = calOb.compute_XY_imcoords_from_XZ_YZ(xz_imcoord, yz_imcoord)
                else:
                    xy_imcoord = np.array([np.NaN, np.NaN])
                filled_in_frame_instances[bad_cam_idx, fishIdx, bpIdx] = xy_imcoord

    elif bad_cam_idx == 2:
        for fishIdx in range(numFish):
            for bpIdx in range(numBodyPoints):
                xz_imcoord = frame_instances[0, fishIdx, bpIdx]
                xy_imcoord = frame_instances[1, fishIdx, bpIdx]
                if ~np.all(np.isnan(xz_imcoord)) and ~np.all(np.isnan(xy_imcoord)):
                    yz_imcoord = calOb.compute_YZ_imcoords_from_XZ_XY(xz_imcoord, xy_imcoord)
                else:
                    yz_imcoord = np.array([np.NaN, np.NaN])
                filled_in_frame_instances[bad_cam_idx, fishIdx, bpIdx] = yz_imcoord

    return filled_in_frame_instances



def fill_in_bad_camera_view_image_coordinates_single_fish(fish_instances, good_cams, calOb):
    ''' Given frame instances for a single fish, and two chosen camera views, return a set of
        fish instances where the values of the image coordinates in the unchosen third camera
        are replaced by values calculated using the calibration object and the image coordinates
        from the chosen views.

    --- args ---
    fish_instances: (numCams=3, numBodyPoints=3, numImCoords=2)
    calOb: an instantiated calibration object
    good_cams: list of two idxs showing which cams to use.
               good_cam_pairings = [ [0,1], [0,2], [1,2] ]
               So, good_cams[0] = [0,1], meaning XZ and XY cameras

    --- returns ---
    filled_in_fish_instances: (numCams=3, numBodyPoints=3, numImCoords=2)
                               Frame instances for single fish with imagecoordinates for the
                               unsed camera view set by the calibration to be consistent with
                               chosen views.

    --- see also ---
    fill_in_bad_camera_view_image_coordinates() -> similar to this function, but it works on
                                                   the results for all fish from a particular
                                                   frame instead of just a single fish.
    '''
    # parse shapes
    _, numBodyPoints, _ = fish_instances.shape

    # preallocate an output array
    filled_in_fish_instances = np.copy(fish_instances)

    # find the bad cam idx
    if good_cams == [0,1]:
        bad_cam_idx = 2
    elif good_cams == [1,2]:
        bad_cam_idx = 0
    elif good_cams == [0, 2]:
        bad_cam_idx = 1

    if bad_cam_idx == 0:
        for bpIdx in range(numBodyPoints):
            xy_imcoord = fish_instances[1, bpIdx]
            yz_imcoord = fish_instances[2, bpIdx]
            if ~np.all(np.isnan(xy_imcoord)) and ~np.all(np.isnan(yz_imcoord)):
                xz_imcoord = calOb.compute_XZ_imcoords_from_XY_YZ(xy_imcoord, yz_imcoord)
            else:
                xz_imcoord = np.array([np.NaN, np.NaN])
            filled_in_fish_instances[bad_cam_idx, bpIdx] = xz_imcoord

    elif bad_cam_idx == 1:
        for bpIdx in range(numBodyPoints):
            xz_imcoord = fish_instances[0, bpIdx]
            yz_imcoord = fish_instances[2, bpIdx]
            if ~np.all(np.isnan(xz_imcoord)) and ~np.all(np.isnan(yz_imcoord)):
                xy_imcoord = calOb.compute_XY_imcoords_from_XZ_YZ(xz_imcoord, yz_imcoord)
            else:
                xy_imcoord = np.array([np.NaN, np.NaN])
            filled_in_fish_instances[bad_cam_idx, bpIdx] = xy_imcoord

    elif bad_cam_idx == 2:
        for bpIdx in range(numBodyPoints):
            xz_imcoord = fish_instances[0, bpIdx]
            xy_imcoord = fish_instances[1, bpIdx]
            if ~np.all(np.isnan(xz_imcoord)) and ~np.all(np.isnan(xy_imcoord)):
                yz_imcoord = calOb.compute_YZ_imcoords_from_XZ_XY(xz_imcoord, xy_imcoord)
            else:
                yz_imcoord = np.array([np.NaN, np.NaN])
            filled_in_fish_instances[bad_cam_idx, bpIdx] = yz_imcoord

    return filled_in_fish_instances



def compute_3D_positions_from_registered_frame_instances(registered_frame_instances, calOb):
    ''' Given registered_frame_instances obtain the corresponding 3D bodypoint positions,
        namely the positions_3D for this frame.

    --- args ---
    registered_frame_instances: (numCams=3, numFish=2, numBodyPoints=3, numImCoords=2)
                                These are the 3 camera instances this frame that have been
                                registered together. See functions on registering frame instances.
    calOb: an instantiated calibration object

    --- returns ---
    fish_3D_positions: shape (numFish, numBodyPoints, 3),
                       the 3D positions of the bodypoints of the fish for this frame
    '''
    _, numFish, numBodyPoints, _ = registered_frame_instances.shape

    # compute the 3D positions
    fish_3D_positions = np.zeros((numFish, numBodyPoints, 3))
    for fishIdx in range(numFish):
        for bpIdx in range(numBodyPoints):
            if np.any(np.isnan(registered_frame_instances[:, fishIdx, bpIdx])):
                XYZ = np.NaN
            else:
                XYZ = calOb.compute_XYZ_from_imageCoord_triplet(registered_frame_instances[:, fishIdx, bpIdx])
            fish_3D_positions[fishIdx, bpIdx, :] = XYZ
    return fish_3D_positions


def check_frame_for_nonzero_detects_in_at_least_2cams(frame_instances):
    ''' If 2 or more camera views have two non-zero detections, return True
        and the camera indices, otherwise return False

    -- Why --
    If we have two detections in two camera views, we can easily make the
    camera<->camera correspondences needed to get 3D bodypoint positions

    -- Inputs --
    frame_instances: the array of bodypoint image coordinates for both fish
                     for a single frame, shape=(numCams,numFish,numBodyPoints,2)

    -- Returns --
    True or False
    '''
    # first a warning about zeros
    if np.any(frame_instances==0):
        raise TypeError('frame_instances has zeros. Missing vals should be NaNs.')

    # check each camera view for 2 non-zero detections
    cam_view_two_nonzero_detections = [~(np.all(np.isnan(frame_instances[camIdx,0])) or
                                         np.all(np.isnan(frame_instances[camIdx,1])))
                                         for camIdx in range(3)]

    # if we have 2 or more trues, then this frame is a "2 in at least 2"
    if np.count_nonzero(cam_view_two_nonzero_detections) >= 2:
        # find the cams and return the vals list
        return True
    else:
        return False



def make_all_permutations(frame_instances):
    ''' Given the instances (or tracks) array, along with a frame idx and cam idx,
        return all possible permutations of the results for this frame
    '''
    # perm1 is as we got it
    perm1 = np.copy(frame_instances)

    # perm2: xz swap
    perm2 = np.copy(frame_instances)
    perm2[0, 0] = np.copy(frame_instances[0,1])
    perm2[0, 1] = np.copy(frame_instances[0,0])

    # perm3: xy swap
    perm3 = np.copy(frame_instances)
    perm3[1, 0] = np.copy(frame_instances[1,1])
    perm3[1, 1] = np.copy(frame_instances[1,0])

    # perm4: yz swap
    perm4 = np.copy(frame_instances)
    perm4[2, 0] = np.copy(frame_instances[2,1])
    perm4[2, 1] = np.copy(frame_instances[2,0])

    permutations = [perm1, perm2, perm3, perm4]
    permutations = np.stack(permutations, axis=0)
    return permutations


def make_permutation_costs_matrix(frame_permutations, calOb):
    ''' Give a list containing all permutations of the frame instances,
        return an array containing the registration costs of all possible options.

    --- args --
    frame_permutations: a list containing all permutations of the frame instances
    calOb:              an instantiated calibration object

    --- returns ---
    permutation_costs: shape (numPerms, numGoodCamPairings, numFish, numBodyPoints)
                       The cost of registering detected skeletons across camera views
                       for all permutations and all camera pairings.

    --- see also ---
    make_all_permutations()
    '''
    # parse input shapes
    numPerms = len(frame_permutations)
    _, numFish, numBodyPoints, _ = frame_permutations[0].shape

    # define some vars on camera views
    good_cam_pairings = [ [0,1], [0,2], [1,2] ]
    numGoodCamPairings = len(good_cam_pairings)

    # ----------------------------------------------------#
    # preallocate output array
    permutation_costs = np.zeros((numPerms, numGoodCamPairings, numFish, numBodyPoints))

    # loop over permutations
    for permIdx in range(numPerms):
        perm_instances = frame_permutations[permIdx]

        # loop over each good_cam_pairing
        for good_cams_idx, good_cams in enumerate(good_cam_pairings):

            # compute the error for each bp of each fish for this cam pairing in this permutation
            for fishIdx in range(numFish):
                for bpIdx in range(numBodyPoints):
                    cam1_bp = perm_instances[good_cams[0], fishIdx, bpIdx]
                    cam2_bp = perm_instances[good_cams[1], fishIdx, bpIdx]
                    error = calOb.compute_point_correspondence_error(good_cams, cam1_bp, cam2_bp)
                    permutation_costs[permIdx, good_cams_idx, fishIdx, bpIdx] = error
    return permutation_costs


def find_mean_bodypoint_costs_for_more_than_two_bodypoints_found(permutation_costs):
    ''' Return a version of the permutation_costs where the bodypoint dimension has
        been collapsed down to the mean of the bodypooint costs, but only for skeleton
        registration costs with 2 or more nonNan values for bodypoints. If only 1 bodypoint is
        registered, the skeleton registration cost is set to NaN.

    --- examples ---
    <<Skeleton registration costs>>  ---------------- > <<summed values>>

    [  0.4673873 , 0.45785861,   0.79911474] ---------> 0.57478688
    [       nan,   6.54993568,   4.82579788] ---------> 5.68786678
    [2.56936649,          nan,   2.36329278] ---------> 2.46632964
    [         nan,        nan,  53.06485773] ---------> nan

     --- args ---
    permutation_costs: shape (numPerms, numGoodCamPairings, numFish, numBodyPoints)

    --- returns ---
    permutation_costs_meanBp: shape (numPerms, numGoodCamPairings, numFish)
    '''
    numPerms, numGoodCamPairings, numFish, numBodyPoints = permutation_costs.shape

    # preallocate
    permutation_costs_meanBp = np.ones((numPerms,numGoodCamPairings,numFish))*np.NaN
    for permIdx in range(numPerms):
        for gcIdx in range(numGoodCamPairings):
            for fishIdx in range(numFish):
                bp_costs = permutation_costs[permIdx, gcIdx, fishIdx]
                numBps_used = np.count_nonzero(~np.isnan(bp_costs))
                if numBps_used >= 2:
                    permutation_costs_meanBp[permIdx, gcIdx, fishIdx] = np.nanmean(bp_costs)

    return permutation_costs_meanBp




def check_method1_on_permutation_costs_matrix(permutation_costs):
    ''' Check if register_frame_instances_method1() will work to register the
        camera view for this permutation costs matrix.

        Big Idea: Can this frame be solved without a threshold on registration costs?

        Idea: We want to find a permutation of the data, together with a pairing
              of two camera views, where we have a non-Nan cost for registering
              3D positions for at least 2 bodypoints on both fish.

        Process: First sum along bodypoint costs, casting to NaN if less than
                 two bodypoints are available.
                 Then sum along fish, to get the cost of registering for each
                 cam pairing in each permutation.

    --- see also ---
    register_frame_instances_method1()

    --- args ---
    permutation_costs: shape (numPerms, numGoodCamPairings, numFish, numBodyPoints)

    --- returns ---
    True of False: True to use method1, False for don't use method1
    '''
    numPerms, numGoodCamPairings, numFish, numBodyPoints = permutation_costs.shape

    # sum costs for bodypoints
    permutation_costs_meanBP = find_mean_bodypoint_costs_for_more_than_two_bodypoints_found(permutation_costs)

    # sum the costs for both fish
    # shape (numPerms, numCamPairings)
    twofish_permutation_costs_meanBp = np.sum(permutation_costs_meanBP, axis=-1)

    # if not all option costs are zero, then we have at least 1 option where we can register both fish
    if ~np.all(np.isnan(twofish_permutation_costs_meanBp)):
        return True
    else:
        return False


def register_frame_instances_method1(frame_permutations, permutation_costs, calOb):
    ''' Find the cam pairing with the lowest cost for registering both fish.
        This is method 1 of registering frame_permutations.

    --- args ---
    frame_permutations: a list containing all permutations of the frame instances
    permutation_costs: shape (numPerms, numGoodCamPairings, numFish, numBodyPoints).
                             Array containing the costs of all permutation registrations
    calOb: an instantiated calibration object

    --- returns ---
    positions_imageCoordinates:       (numCams, numFish, numBodyPoints, 2)
                                      The registered image coordinates for this frame.
    positions_3DBps:                  (numFish, numBodyPoints, 3)
                                      The associated 3D position of the registered
                                      imageCoordinates.
    positions_registration_costs:     (numFish, numBodyPoints)
                                      The registration costs of all bodypoints of all fish.
    usedCamIdxs:                      List of two ints. Contains the idxs of the cam pair used.
                                      i.e. [0,1], [0,2], or [1,2]
    permutation_costs_meanBP:         The permutation_costs array collapsed along bodypoints,
                                      so shape=(numPerms, numGoodCamPairings, numFish)
    twofish_permutation_costs_meanBP: The permutation_costs array collapsed along Fish,
                                      so shape=(numPerms, numGoodCamPairings)
    options_info_and_costs:           An array of shape (numPerms*numGoodCamPairings, 3),
                                      where the elements of the second dimension represent
                                      [permIdx, camPairingIdx, totalRegistrationCost],
                                      with totalRegistrationCost
    bestOptInfoIdx:                   The row index of the best option from the
                                      options_info_and_costs array.


    --- see also ---
    check_method1() -> a function for checking if we can use this function to solve the frame

    '''
    # define some vars on camera views
    good_cam_pairings = [ [0,1], [0,2], [1,2] ]
    numGoodCamPairings = len(good_cam_pairings)
    numPerms = len(frame_permutations)

    # sum costs for bodypoints
    permutation_costs_meanBP = find_mean_bodypoint_costs_for_more_than_two_bodypoints_found(permutation_costs)

    # sum the costs for both fish
    # shape (numPerms, numCamPairings)
    twofish_permutation_costs_meanBP = np.sum(permutation_costs_meanBP, axis=-1)

    # check if we can proceed:
    # if not all option costs are zero, then we have at least 1 option where we can register both fish
    if np.all(np.isnan(twofish_permutation_costs_meanBP)):
        raise TypeError('frame_permutations does not contain enough info to use used with this method')


    # make an array for (permIdx, campairIdx, cost) for each option from all perms
    options_info_and_costs = []
    for permIdx in range(numPerms):
        for camPairIdx in range(numGoodCamPairings):
            options_info_and_costs.append(np.array([permIdx, camPairIdx, twofish_permutation_costs_meanBP[permIdx, camPairIdx]]))
    options_info_and_costs = np.array(options_info_and_costs)

    # find the option with the lowest nonNan cost for registering both fish
    # Note: entries will come in pairs, we pick the first one, the order doesnt matter
    bestOptInfoIdx = np.argsort(options_info_and_costs[:,-1])[0]
    bestOpt_permIdx = int(options_info_and_costs[bestOptInfoIdx,0])
    bestOpt_camPairIdx = int(options_info_and_costs[bestOptInfoIdx,1])

    # now derivate the missing imagecoordinates, and get the 3D positions
    chosen_frame_coordinates = np.copy(frame_permutations[bestOpt_permIdx])
    chosen_camPairing_idx = good_cam_pairings[bestOpt_camPairIdx]
    positions_imageCoordinates = fill_in_bad_camera_view_image_coordinates(chosen_frame_coordinates,
                                                                           chosen_camPairing_idx,
                                                                           calOb)

    # now get the 3D positions
    positions_3DBps = compute_3D_positions_from_registered_frame_instances(positions_imageCoordinates, calOb)

    # get the registration costs for the option we used
    positions_registration_costs = permutation_costs[bestOpt_permIdx, bestOpt_camPairIdx, :, :]

    # for debugging purposes
    usedCamIdxs = good_cam_pairings[bestOpt_camPairIdx]

    # finish up
    outputs = [positions_imageCoordinates,
               positions_3DBps,
               positions_registration_costs,
               usedCamIdxs,
               permutation_costs_meanBP,
               twofish_permutation_costs_meanBP,
               options_info_and_costs,
               bestOptInfoIdx]
    return outputs





def register_frame_instances_method2(frame_permutations, permutation_costs, calOb, mean_reg_skel_thresh=2):
    ''' Find the cam pairing with the lowest cost for registering both fish.
        This is method 2 of registering frame_permutations.

        --- args ---
        frame_permutations: a list containing all permutations of the frame instances
        permutation_costs: shape (numPerms, numGoodCamPairings, numFish, numBodyPointss).
                                 Array containing the costs of all permutation registrations
        calOb: an instantiated calibration object

        --- kwargs ---
        mean_reg_skel_thresh=10 : a threshold on registration costs.
                                  This is a threshold on the bodypoint-mean cost of registering
                                  a sleap skeleton from one camera view with a sleap skeleton from
                                  a different camera view, with both skeletons having n>=2
                                  bodypoints in common.
                                  This parameter can be found by applying method1 to all
                                  approriate frames, and hence estimating the distribution of costs
                                  for correctly matched skeletons.

        --- returns ---
        positions_imageCoordinates:       (numCams, numFish, numBodyPoints, 2)
                                          The registered image coordinates for this frame.
        positions_3DBps:                  (numFish, numBodyPoints, 3)
                                          The associated 3D position of the registered
                                          imageCoordinates.
        positions_registration_costs:     (numFish, numBodyPoints)
                                          The registration costs of all bodypoints of all fish.

        --- see also ---
        check_method1() -> a function for checking if we can solve frame using method1.
                           If we can't, we have to use this function.

    '''
    # parse some shapes
    _, _, numFish, numBodyPoints = permutation_costs.shape
    numCams = 3

    # define some vars on camera views
    good_cam_pairings = [ [0,1], [0,2], [1,2] ]
    numPerms = len(frame_permutations)

    # sum costs for bodypoints
    permutation_costs_meanBP = find_mean_bodypoint_costs_for_more_than_two_bodypoints_found(permutation_costs)


    # ------- find the best permutation to use ----- #
    # these are list over permutations, with each element being an array of length 2,
    # with elements of the cost of registering a fish and the cam pairing used.
    # The threshed version just applies the mean_reg_skel_thresh
    perm_best_options_for_making_fish = []
    perm_best_options_for_making_fish_threshed = []

    for permIdx in range(numPerms):

        reg_costs = permutation_costs_meanBP[permIdx]

        best_pairings_for_fish = []
        best_pairings_for_fish_threshed = []

        for fishIdx in range(numFish):
            # find the info for this fish
            if np.any(~np.isnan(reg_costs[:, fishIdx])):
                fish_min_cost = np.nanmin(reg_costs[:, fishIdx])
                fish_min_camPairIdx = np.argsort(reg_costs[:, fishIdx])[0]
                best_pairings_for_fish.append(np.array([fish_min_cost, fish_min_camPairIdx]))
            else:
                fish_min_cost = np.NaN
                fish_min_camPairIdx = np.NaN
                best_pairings_for_fish.append(np.array([fish_min_cost, fish_min_camPairIdx]))

            # does it pass threshold?
            if ~np.isnan(fish_min_cost):
                if fish_min_cost < mean_reg_skel_thresh:
                    best_pairings_for_fish_threshed.append(np.array([fish_min_cost, fish_min_camPairIdx]))
                else:
                    best_pairings_for_fish_threshed.append(np.array([np.NaN, np.NaN]))
            else:
                best_pairings_for_fish_threshed.append(np.array([np.NaN, np.NaN]))


        perm_best_options_for_making_fish.append(best_pairings_for_fish)
        perm_best_options_for_making_fish_threshed.append(best_pairings_for_fish_threshed)



    # --- count the number of found fish in each permutation ----#
    # an array containing the number of fish we can register in each permutation
    perm_numFish_found = []
    for permIdx, permInfo in enumerate(perm_best_options_for_making_fish_threshed):
        permFishFound = 0
        for fishIdx in range(numFish):
            fishRegCost = permInfo[fishIdx][0]
            if ~np.isnan(fishRegCost):
                permFishFound += 1
        perm_numFish_found.append(permFishFound)
    perm_numFish_found = np.array(perm_numFish_found)





    # ---- the best permutation ---------#
    # The only with the most fish, past that, we don't care (since all costs passed threshold)
    best_permIdx = np.argmax(perm_numFish_found)
    best_options_for_making_fish = perm_best_options_for_making_fish_threshed[best_permIdx]
    permuted_frame_instances = np.copy(frame_permutations[best_permIdx])


    # --- find the final positions_registration_costs ---#
    # positions_registration_costs array of shape (numFish, numBodyPoints)
    camIdxs_used_for_fish = []
    for fishIdx in range(numFish):
        cam_used_for_fish = best_options_for_making_fish[fishIdx][1]
        camIdxs_used_for_fish.append(cam_used_for_fish)
    camIdxs_used_for_fish = np.array(camIdxs_used_for_fish)

    positions_registration_costs = []
    for fishIdx in range(numFish):
        camPairIdx_used = camIdxs_used_for_fish[fishIdx]
        if ~np.isnan(camPairIdx_used):
            camPairIdx_used  = int(camPairIdx_used )
            fish_reg_cost = permutation_costs[best_permIdx, camPairIdx_used, fishIdx]
        else:
            fish_reg_cost = np.ones((numBodyPoints,))*np.NaN
        positions_registration_costs.append(fish_reg_cost)
    positions_registration_costs = np.array(positions_registration_costs)


    # --- fill-in the coordinates of both fish using calibration ------ #
    # preallocate the output
    positions_imageCoordinates = np.ones((numCams, numFish, numBodyPoints, 2))*np.NaN
    for fishIdx in range(numFish):
        # parse the values of interest for this fish
        fish_bestOpt_cost, fish_bestOpt_camPairIdx = best_options_for_making_fish[fishIdx]
        fish_instances = permuted_frame_instances[:, fishIdx, : , :]

        # if we can register the fish, fill-in its 3rd cam image coordinates
        if ~np.isnan(fish_bestOpt_cost):
            fill_in_fish_instances = fill_in_bad_camera_view_image_coordinates_single_fish(fish_instances,
                                                                                           good_cam_pairings[int(fish_bestOpt_camPairIdx)],
                                                                                           calOb)
        # if we cant register the fish, return NaN image coordinates so we don't try to make this fish
        else:
            fill_in_fish_instances = np.ones_like(fish_instances)*np.NaN

        # record
        positions_imageCoordinates[:, fishIdx] = fill_in_fish_instances



    # ----- now get the 3D positions ---- #
    positions_3DBps = compute_3D_positions_from_registered_frame_instances(positions_imageCoordinates,
                                                         calOb)



    # --------- finish up ------------------#
    outputs = [positions_imageCoordinates,
               positions_3DBps,
               positions_registration_costs,
               perm_best_options_for_making_fish,
               perm_best_options_for_making_fish_threshed,
               perm_numFish_found,
               best_permIdx]
    return outputs






def get_3D_positions_from_sleap_imcoords(frame_instances, calOb, mean_reg_skel_thresh=2, debug=False):
    ''' Given the sLEAP results from all 3 camera views for a frame,
        return the 3D positions for fish bodypoints.

    -- inputs --
    frame_instances: (numCams=3, numFish=2, numBodyPoints=3, numCoords=2)
                     The results from the 3 sLEAP networks for a single frame
    calOb: an instantiated calibration object

    --- kwargs ---
    mean_reg_skel_thresh=2  : a threshold on registration costs.
                              This is a threshold on the bodypoint-mean cost of registering
                              a sleap skeleton from one camera view with a sleap skeleton from
                              a different camera view, with both skeletons having n>=2
                              bodypoints in common.
                              This parameter can be found by applying method1 to all
                              approriate frames, and hence estimating the distribution of costs
                              for correctly matched skeletons.
    debug=False             : if true, return dbug info

    -- returns ---
    methodIdx: the index of the method use to register across cameras.
    positions_imageCoordinates:   (numCams, numFish, numBodyPoints, 2)
                                  The registered image coordinates for this frame.
    positions_3DBps:              (numFish, numBodyPoints, 3)
                                  The associated 3D position of the registered imageCoordinates,
    positions_registration_costs: (numFish, numBodyPoints)
                                  The registration costs of all bodypoints of all fish.
    debug_vals:                   A list of internal variables that can be used for debugging.
    '''
    # parse some shapes
    numCams, numFish, numBodyPoints, _ = frame_instances.shape

    #  get the permutations
    frame_permutations = make_all_permutations(frame_instances)

    # compute the cost of each permutation
    permutation_costs =  make_permutation_costs_matrix(frame_permutations, calOb)

    # find the methodIdx
    do_use_method1 = check_method1_on_permutation_costs_matrix(permutation_costs)
    if do_use_method1:
        methodIdx = 0
    else:
        methodIdx = 1

    # solve the frame using the correct method
    if methodIdx == 0:
        positions_imageCoordinates, \
        positions_3DBps, \
        positions_registration_costs, \
        usedCamIdxs, \
        permutation_costs_meanBP, \
        twofish_permutation_costs_meanBP, \
        options_info_and_costs, \
        bestOptInfoIdx = register_frame_instances_method1(frame_permutations,
                                                          permutation_costs,
                                                          calOb)
    elif methodIdx == 1:
        positions_imageCoordinates, \
        positions_3DBps, \
        positions_registration_costs, \
        perm_best_options_for_making_fish, \
        perm_best_options_for_making_fish_threshed, \
        perm_numFish_found, \
        best_permIdx  = register_frame_instances_method2(frame_permutations,
                                                         permutation_costs, calOb,
                                                         mean_reg_skel_thresh=mean_reg_skel_thresh)
    else:
        raise TypeError('methodIdx is not 0 or 1')


    # ---- finish up ---- #
    if debug:
        if methodIdx == 0:
            debug_vals = [usedCamIdxs,
                          permutation_costs,
                          permutation_costs_meanBP,
                          twofish_permutation_costs_meanBP,
                          options_info_and_costs,
                          bestOptInfoIdx]
        elif methodIdx == 1:
            debug_vals = [permutation_costs,
                          perm_best_options_for_making_fish,
                          perm_best_options_for_making_fish_threshed,
                          perm_numFish_found,
                          best_permIdx]

        return methodIdx, positions_imageCoordinates, positions_3DBps, positions_registration_costs, debug_vals

    else:
        return methodIdx, positions_imageCoordinates, positions_3DBps, positions_registration_costs, []








def register_sleap_instances(sleap_instances, calOb, print_updates=False):
    ''' Given the array of sleap_instances containing the results of the three
        sleap inferences, and a calibration object,
        perform cross-camera registration to obtain 3D positions for individuals,
        but without temporal identity.

    --- args ---
    sleap_instances: shape=(numCams, numFrames, numFish, numBodyPoints, 2)
                     The sleap results for the three networks.
    calOb: an instantiated calibration object

    --- kwargs ---
    print_updates=False : whether or not you want to print to screen the frame
                          index every 100 frames.

    '''
    # parse shapes
    numCams, numFrames, numFish, numBodyPoints, _ = sleap_instances.shape

    # preallocate otuput arrays
    frame_methodIdxs = np.ones((numFrames,))*np.NaN
    frame_positions_imageCoordinates = np.ones((numCams, numFrames, numFish, numBodyPoints, 2))*np.NaN
    frame_positions_3DBps = np.ones((numFrames, numFish, numBodyPoints, 3))*np.NaN
    frame_registration_costs = np.ones((numFrames, numFish, numBodyPoints))*np.NaN

    for fIdx in range(numFrames):

        if print_updates:
            if np.mod(fIdx,100) == 0:
                print(fIdx)

        # get the frame loaded_imcoords
        frame_instances = sleap_instances[:,fIdx,:,:,:]

        # register the skeletons across camera views
        methodIdx, \
        positions_imageCoordinates, \
        positions_3DBps, \
        positions_registration_costs, \
        debug_vals = get_3D_positions_from_sleap_imcoords(frame_instances, calOb, debug=False)

        # record
        frame_methodIdxs[fIdx] = methodIdx
        frame_positions_imageCoordinates[:, fIdx] = positions_imageCoordinates
        frame_positions_3DBps[fIdx] = positions_3DBps
        frame_registration_costs[fIdx] = positions_registration_costs

    return frame_methodIdxs, frame_positions_imageCoordinates, frame_positions_3DBps, frame_registration_costs



def create_array_of_start_stop_frames_for_parallelization(numFrames, step=1000):
    ''' Return an array that we can index easily to divide the frames into chunks
        for processing in parallel.

    --- args ---
    numFrames: the number of frames we want to process
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
    start_frms = np.arange(0, numFrames, step)
    stop_frms = np.append(np.arange(step, numFrames, step), numFrames)
    start_stop_frms = np.stack([start_frms, stop_frms], axis=1)
    return start_stop_frms



def find_mean_bodypoint_registration_costs_array(registration_costs_array):
    ''' Return a version of the registration_costs_array where the bodypoint dimension has
        been collapsed down to the mean of the bodypooint costs, but only for skeleton
        registration costs with 2 or more nonNan values for bodypoints. If only 1 bodypoint is
        registered, the skeleton registration cost is set to NaN.

    --- examples ---
    <<Skeleton registration costs>>  ---------------- > <<summed values>>

    [  0.4673873 , 0.45785861,   0.79911474] ---------> 0.57478688
    [       nan,   6.54993568,   4.82579788] ---------> 5.68786678
    [2.56936649,          nan,   2.36329278] ---------> 2.46632964
    [         nan,        nan,  53.06485773] ---------> nan

     --- args ---
    registration_costs_array: shape (numFrames, numFish, numBodyPoints).
                              Comes from the cross-camera registration step.

    --- returns ---
    registration_costs_array_meanBp: shape (numFrames, numFish)
    '''

    numFrames, numFish, numBodyPoints = registration_costs_array.shape

    registration_costs_meanBp = []
    for fishIdx in range(numFish):

        fish_reg_data = registration_costs_array[:,fishIdx]

        # count the number of nonNaN bodypoint registrations in each frame
        numBps_registered_each_frame = np.count_nonzero(~np.isnan(fish_reg_data), axis=(1))

        # take the mean along bodypoints
        registration_costs_fish_meanBp = np.nanmean(fish_reg_data, axis=1)

        # now naN any frame results if we detected less than 2 bodypoints
        registration_costs_fish_meanBp[numBps_registered_each_frame < 2] = np.NaN

        # record
        registration_costs_meanBp.append(registration_costs_fish_meanBp)

    # finish up
    registration_costs_array_meanBp = np.stack(registration_costs_meanBp, axis=1)
    return registration_costs_array_meanBp



def save_tracks_3D_to_csv_and_return_dataFrame(tracks_3D, savepath):
    ''' Save the trajectories to a csv file. This only works for data
        of the form (numFrames, numFish=2, numBodyPoints=3, numDims=3),
        because we title each column. Saved to 2 decimal places of precision.

    --- args ---
    tracks_3D: tracking data, shape
              (numFrames, numFish=2, numBodyPoints=3, numDims=3)
    savepath: string, location to save to.
    '''
    numFrames, numFish, numBodyPoints, _ = tracks_3D.shape
    column_headings = ['fish1_head_x', 'fish1_head_y', 'fish1_head_z',
                       'fish1_pec_x', 'fish1_pec_y', 'fish1_pec_z',
                       'fish1_tail_x', 'fish1_tail_y', 'fish1_tail_z',
                       'fish2_head_x', 'fish2_head_y', 'fish2_head_z',
                       'fish2_pec_x', 'fish2_pec_y', 'fish2_pec_z',
                       'fish2_tail_x', 'fish2_tail_y', 'fish2_tail_z']

    column_data_list = []
    for fishIdx in range(numFish):
        for bpIdx in range(numBodyPoints):
            for dimIdx in range(3):
                col = np.copy(tracks_3D[:, fishIdx, bpIdx, dimIdx])
                column_data_list.append(col)
    column_data = np.stack(column_data_list, axis=1)

    tracks_3D_df = pd.DataFrame(data=column_data, columns=column_headings)

    tracks_3D_df.to_csv(savepath, index_label='frame_index', sep=',', mode='w', float_format='%.2f')

    return tracks_3D_df







def compute_distance_between_3D_fish(fish1_3D_bps, fish2_3D_bps):
    ''' Return the distance between the two fish using our chosen metric
    '''
    numBodyPoints = fish1_3D_bps.shape[0]
    bp_distances = np.ones((numBodyPoints))
    for bpIdx in range(numBodyPoints):
        bp_distances[bpIdx] = np.linalg.norm(fish1_3D_bps[bpIdx] - fish2_3D_bps[bpIdx])
    distance = np.nanmean(bp_distances)
    return distance



def track_segment_in_3D_if_possible(existing_tracks_3D_for_segment, existing_imCoords_3D_for_segment,
                                    positions_3D_processed_for_segment, positions_imageCoordinates_processed_for_segment,
                                    last_known_positions, final_known_positions,
                                    tracks_available_post_pass2_id_assignments_for_segment):
    ''' Try to track the passed segment of frames, progagating identity
        through the 3D skeletons.


    --- Idea ---
    Using tracks_3D, we have the location of both individuals before and after this segment.
    We try to propagate identity through this collection of frames by tracking the observed
    3D skeletons. We know we did this correctly, if at the end of our tracking, our identities
    match the identities for after this segment.

    --- schematic ----
    * = last_known_position
    & = final_known_position

    |___before__|_*__|<-----segment--|-&-->|_____after_____|
    time ->

    Segment runs from the first frame without locations for both, up to and including
    the first frame where we have idxs for both again.
    Using the last known position to initialize the tracking during the segment,
    we track all the way until final_known_positions (the next time we have identities for
    both fish). If final_known_positions is equal to where we ended up by tracking,
    we declare success.

    --- args ---
    existing_tracks_3D_for_segment:
    existing_imCoords_3D_for_segment:
    positions_3D_processed_for_segment:
    positions_imageCoordinates_processed_for_segment:
    last_known_positions:
    final_known_positions:
    tracks_available_post_pass2_id_assignments_for_segment:

    --- returns ---
    ouput = [
            was_successful: bool, True if we successfully tracked
            segment_tracks_3D: ((refFE-regF0)+1, numFish, numBodypoints, 3)
            segment_tracks_imCoords: (3, (refFE-regF0)+1, numFish, numBodypoints, 2)
            seg_track_method: ((refFE-regF0)+1,), entries={0,1,2,3,NaN}
            seg_data_available_array: ((refFE-regF0)+1,numFish), 0=no_data, 1=data
            ]
    '''
    # parse input shapes
    reg_nfs, numFish, numBodyPoints, _ = existing_tracks_3D_for_segment.shape
    numCams = 3

    # preallocate the outputs
    segment_tracks_3D = np.zeros_like(existing_tracks_3D_for_segment)*np.NaN
    segment_tracks_imCoords = np.zeros( (numCams,reg_nfs,numFish,numBodyPoints,2) )*np.NaN

    # a debugging array to keep track of which method we used for the frame
    seg_track_method = np.zeros((reg_nfs,))*np.NaN


    for ii in range(reg_nfs):

        # ---- case 1 ---- #
        # If we already entered a value in tracks_3D for either fish,
        # then both fish are already spoken for,
        # because if we found one in pass2 using idtracker comparison,
        # then we would have assigned the other detected skeleton and id if we have another
        # detected skeleton.
        # So test if we can resolve this frame by simply importing from tracks_3D,
        # unless this is the last frame, in which case we want to naturally land on
        # the right matching.

        # if we are at the last frame, skip this step
        if ii == reg_nfs-1:
            pass
        # otherwise see if we have already fill-in the tracks for this frame
        else:
            # if some fish already identified
            if np.sum(tracks_available_post_pass2_id_assignments_for_segment[ii]) > 0:
                # assign the identities
                segment_tracks_3D[ii] = np.copy(existing_tracks_3D_for_segment[ii])
                segment_tracks_imCoords[:,ii] = np.copy(existing_imCoords_3D_for_segment[:,ii])
                # record the method
                seg_track_method[ii] = 0
                # update the last known positions
                for fishIdx in range(numFish):
                    if tracks_available_post_pass2_id_assignments_for_segment[ii, fishIdx] == 1:
                        last_known_positions[fishIdx] = segment_tracks_3D[ii, fishIdx]
                continue



        # --- case 2 ------ #
        # We are not referencing tracks_3D at all,
        # and will simply try to match onservations to last known positions

        # get the observed 3D skeletons for this frame
        frame_positions = np.copy(positions_3D_processed_for_segment[ii])

        # count the number of obsercations available
        # Do we have 0,1,2 skeletons available
        fish_have_some_bps = np.zeros((numFish,), dtype=bool)
        for fishIdx in range(numFish):
            fish_have_some_bps[fishIdx] = ~np.all(np.isnan(frame_positions[fishIdx]))
        numFishFound = np.sum(fish_have_some_bps)

        # deal with 2 observation case
        if numFishFound == numFish:
            frame_cost_mat = np.zeros((numFish, numFish)) # the cost array
            current_fish = []
            last_fish = []
            for fishIdx in range(numFish):
                current_fish.append(frame_positions[fishIdx])
                last_fish.append(last_known_positions[fishIdx])
            # fill-in the cost matrix
            for curr_idx in range(numFish):
                for prev_idx in range(numFish):
                    frame_cost_mat[curr_idx, prev_idx] = compute_distance_between_3D_fish(current_fish[curr_idx],
                                                                                          last_fish[prev_idx])
            # solve the optimal assigning of identities between frames
            row_ind, col_ind = linear_sum_assignment(frame_cost_mat)
            # update the tracks arrays
            for fishIdx in range(numFish):
                segment_tracks_3D[ii, fishIdx] = np.copy(frame_positions[col_ind[fishIdx]])
                segment_tracks_imCoords[:,ii,fishIdx] = np.copy(positions_imageCoordinates_processed_for_segment[:,ii,col_ind[fishIdx]])
            # update the last known positions
            last_known_positions = np.copy(segment_tracks_3D[ii])
            # record what we did
            seg_track_method[ii] = 1

        # deal with 1 obsercation case
        elif numFishFound == (numFish-1):
            # find the last observation that this detection is closest to, and assign that identity
            # which fish is not missing
            detectionIdx = int(np.where(fish_have_some_bps)[0][0])
            detection_skeleton = frame_positions[detectionIdx]
            detection_imCoords = np.copy(positions_imageCoordinates_processed_for_segment[:,ii,detectionIdx])
            # find the distance to last known positions
            reg_costs = []
            for fishIdx in range(numFish):
                reg_costs.append( compute_distance_between_3D_fish(detection_skeleton, last_known_positions[fishIdx]) )
            reg_costs = np.array(reg_costs)
            # find the correct identity for this observation
            fishIdx_for_observation = np.argmin(reg_costs)
            # record
            segment_tracks_3D[ii, fishIdx_for_observation] = detection_skeleton
            segment_tracks_imCoords[:, ii, fishIdx_for_observation] = detection_imCoords
            last_known_positions[fishIdx_for_observation] = detection_skeleton
            # record what we did
            seg_track_method[ii] = 2

        # deal with 0 observation case
        elif numFishFound == 0:
            # Do nothing, and move to next frame
            # record what we did
            seg_track_method[ii] = 3
            continue

        # deal with 0 observation case
        else:
            raise TypeError('frame {0}: number of fish detected in not 0,1 or 2.'.format(fIdx))


    # ----- count frames where we have skeleton info, -----#
    seg_data_available_array = np.zeros((reg_nfs,numFish))*np.NaN
    for fidx in range(segment_tracks_3D.shape[0]):
        for fishIdx in range(numFish):
            if ~np.all(np.isnan(segment_tracks_3D[fidx,fishIdx])):
                seg_data_available_array[fidx, fishIdx] = 1
            else:
                seg_data_available_array[fidx, fishIdx] = 0


    # ---- if the end of the segment_tracks_3D matches the next known positions --- #
    # declare success
    if np.all(segment_tracks_3D[-1][~np.isnan(segment_tracks_3D[-1])] == final_known_positions[~np.isnan(final_known_positions)]):
        was_successful = True
    else:
        was_successful = False


    # finish up
    return [was_successful, segment_tracks_3D, segment_tracks_imCoords, seg_track_method, seg_data_available_array]



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

