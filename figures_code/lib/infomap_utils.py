''' This file contains functions for
    -> building transition matrices
    -> putting data into bins
    -> performing infomap
    -> visualising infomap results
'''
import numpy as np
import itertools
from igraph import Graph


def find_binIdxs_for_timeseries(tseries, tseries_bins):
    '''
    --- args ---
    tseries: (numFrames,) - a 1D timeseries
    tseries_bins: (numBins+1,) - the bin edges for a discretization of tseries

    --- returns ---
    bin_idxs: the indices of the bins tseries fall into.
              Values run from 0 to numBins-1.
              -1 is returned for NaN data which does not lie in a bin
    --- Notes ---
    (1) np.digitize assigns NaNs to the rightmost index i.e. numBin_edges
        For more clarity, we will convert NaN entries to binIdx=-1

        -- illustration ---
    >> numBins = 10
    >> numBin_edges = numBins + 1
    >> tet_bins = np.linspace(-np.pi, np.pi, numBin_edges)
    >> tet_bins
       array([-3.14159265, -2.51327412, -1.88495559, -1.25663706, -0.62831853,
              0.        ,  0.62831853,  1.25663706,  1.88495559,
              2.51327412, 3.14159265]

    >> tet_w = np.nan
       => tet_bin_idxs = array([-1,])

    >> tet_w = 0.77828154,
       => tet_tet_bin_idxs = array([6])

    >> tet_w = 2.81087403
       => tet_tet_bin_idxs = array([9])
    '''
    numBin_edges = tseries_bins.shape[0]
    # get the binIdxs from 1 to numBins (numBins for NaN data)
    bin_idxs = np.digitize(tseries, tseries_bins)
    # Nan vals are set to numBin_edges, set them to Nans instead
    bin_idxs[bin_idxs == numBin_edges] = -1
    # subtract 1 from every non -1 value,
    # to move from 1-indexing to standard 0 indexing
    bin_idxs[bin_idxs != -1] -= 1
    return bin_idxs


def make_dictionary_of_state_numberings(num_bins_per_var):
    ''' Given a list containing the number of bins for each of K variables,
        return a dictionary mapping each the possible states defined using this
        binnning to a number.

    --- args ---
    state_var_bin_list: a list of length K, containing the number of bins
                        you will use for each of the K variables.

    --- example ---
    >> num_tetW_bins = 2
    >> num_tetL_bins = 2
    >> num_bpp_bins = 2
    >> num_bins_per_var = [num_tetW_bins, num_tetL_bins, num_bpp_bins]
    >> state_numberings = state_names(num_bins_per_var)
    >> state_numberings
        {(0, 0, 0): 0,
         (0, 0, 1): 1,
         (0, 1, 0): 2,
         (0, 1, 1): 3,
         (1, 0, 0): 4,
         (1, 0, 1): 5,
         (1, 1, 0): 6,
         (1, 1, 1): 7}

    --- see also ---
    find_binIdxs_for_theta_timeseries
    '''
    # get the number of possible states
    num_states = np.product(num_bins_per_var)
    # now create an array of the numberings for states defined by the bins
    # i.e. the values for the dictionary
    state_name_values = np.arange(num_states)
    # make a list over variables,
    # containing a list of the bin indices for each variable
    list_of_each_var_bin_idxs = [list(range(num_var_bins)) for
                                 num_var_bins in num_bins_per_var]
    # the keys are all possible products of state bin idxs
    state_name_keys = [x for x in
                       itertools.product(*list_of_each_var_bin_idxs)]
    # make the dictionary
    state_numberings = dict(zip(state_name_keys, state_name_values))
    return state_numberings


def build_transition_matrix(statespace_bin_idx_timeseries, tau, state_names):
    ''' Compute and return the transition matrix.

    --- args ---
    statespace_bin_idx_timeseries: (numFrames,numStateVars)
                                   The timeseries of bin indices.
    tau: The waiting time sampling the future state (in frames)
    state_name: the dictionary mapping bintuples and state indices

    --- returns ---
    transition_matrix: a row normalised transition matrix (rows sum to 1).
                       Rows represent current state,columns represent nex state
    '''
    numStates = len(state_names)
    numFrames = statespace_bin_idx_timeseries.shape[0]

    # preallocate the transition matrix
    trans_mat_counts = np.zeros((numStates, numStates))

    # loop over timesteps
    startframe = 0
    for fIdx in range(startframe, numFrames-tau):

        # find the bin values of the current and next state
        cur_state_bin_idxs = statespace_bin_idx_timeseries[fIdx]
        nex_state_bin_idxs = statespace_bin_idx_timeseries[fIdx+tau]

        # Leave early if we have NaNs in this state or next state
        if np.any(cur_state_bin_idxs == -1) or np.any(nex_state_bin_idxs == -1):
            continue
        else:
            # if we have no NaNs, keep going,
            # and get the state index from the bins values
            cur_state_index = state_names[tuple(cur_state_bin_idxs)]
            nex_state_index = state_names[tuple(nex_state_bin_idxs)]

        # enter this transition count
        trans_mat_counts[cur_state_index, nex_state_index] += 1

    # --- wrapping up ---- #

    # normalize the rows of the transition matrix, so probs sum to 1
    row_sums = trans_mat_counts.sum(axis=1)
    transition_matrix = trans_mat_counts / row_sums[:, np.newaxis]
    return transition_matrix


# ------- Removing the empty states ----------- #

def remove_empty_states(transMat, state_numberings):
    ''' Process the transition matrix, removing rows and columns that correspond
        to states that are never visited in the data, detectable as all-NaN
        rows of the transition matrix.

    --- args --
    transMat: the original transition matrix
    state_numberings: the dictionary mapping the bintuples of the rows
                      of the original transition matrix to integers.

    --- returns ---
    edited_transition_matrix: the new transition matrix with the empty
                              states removed
    edited_state_numberings: an updated version of state_numberings,
                             for the new transition matrix.
    transMat_row_bin_tup_arr: the bintuples for the rows of the new
                              transition matrix in array form
    NaN_rows_idxs: the row indices of the states we removed from
                   the original transition matrix.
    NaN_row_bin_tup_arr: the bintuples for the states that were removed
                        from the original transition matrix in array
                        form.

    --- see also ---
    -> For the args
    build_transition_matrix()
    make_dictionary_of_state_numberings()
    -> internally used
    find_NaN_rows_of_transition_matrix()
    make_edited_transition_matrix()
    make_edited_state_numberings()

    '''
    # --- step 1: find the row indices of empty states (NaN rows in mat) --- #
    NaN_rows_idxs = find_NaN_rows_of_transition_matrix(transMat)

    # --- step 2: make an edited_transition_matrix without the bad states --- #
    edited_transition_matrix = make_edited_transition_matrix(transMat,
                                                             NaN_rows_idxs)

    # --- step 3: make an edited_state_numberings  ---- #
    #             for the edited transition matrix
    outs = make_edited_state_numberings(state_numberings, NaN_rows_idxs)
    edited_state_numberings, transMat_row_bin_tup_arr, NaN_row_bin_tup_arr = outs

    return (edited_transition_matrix, edited_state_numberings,
            transMat_row_bin_tup_arr,
            NaN_rows_idxs, NaN_row_bin_tup_arr)


def make_edited_transition_matrix(tmat, remove_idxs):
    ''' Return an edited version of tmat, where the rows and columns
        with indices in remove_idxs have been removed.
        remove_idxs are intended to be the indices of rows of
        tmat that are all NaNs.

    --- Idea ---
    We have identified states in the transition matrix which are never
    visited, so we want to remove these states before subsequent analysis.
    '''
    edited_tmat = np.delete(np.delete(tmat, remove_idxs, axis=0),
                            remove_idxs, axis=1)
    return edited_tmat


def find_NaN_rows_of_transition_matrix(tmat):
    ''' Return the indices of rows of tmat, where all elements of the
        row are NaN (meaning this state never occurs).
    '''
    # first sum along rows,
    row_totals = np.nansum(tmat, axis=1)
    # now any row whose sum is 0, has no non-zero elements,
    # which means that all row entries are NaNs (because
    # we divided by the sum of this row when creating tmat)
    NaN_rows_idxs = np.where(row_totals == 0)[0]
    return NaN_rows_idxs


def make_edited_state_numberings(state_numberings, NaN_rows_idxs):
    ''' Given the original state_numberings, and the array of row indices
        we removed from the original transition matrix,
        return an edited_state_numbering dictionary mapping bintuples
        corresponding to rows of the edited transition matirx to integers,
        go given a bintuple we know what row of the transition matrix it comes
        from.

    --- args ---
    state_numberings: the state_numberings dictionary for the original
                      transition matrix
                      (see make_dictionary_of_state_numberings() for where
                      this dictionary comes from)
    NaN_rows_idxs: an array of rowIdxs that we want to remove from the original
                   transition matrix, because the correspond to states which
                   never occur in the data

    --- returns ---
    edited_state_numberings: dictionary mapping the bintuples (rows of the
                             edited transition matrix) to integers
    transMat_row_bin_tup_arr: a (N, len(tuples)) array, containing the
                              binindices for each row of the edited NxN
                              shaped transition matrix, for saving purposes.
    NaN_row_bin_tup_arr: a (k,len(tuples)) array, containing the binindices
                         of the rows we removed fromt the original transition
                         matrix, again for saving purposes.
    '''
    # get the keys of original as array
    original_keys = np.array(list(state_numberings.keys()))

    # get the bintuples corresponding to NaN rows
    NaN_row_bin_tup_arr = original_keys[list(NaN_rows_idxs)]

    # get the non NaN row idxs
    original_numStates = len(state_numberings)
    non_nan_row_idxs = np.setdiff1d(np.arange(original_numStates),
                                    NaN_rows_idxs)

    # get the bintuples corresponing to non_NaN rows
    transMat_row_bin_tup_arr = original_keys[list(non_nan_row_idxs)]

    # use transMat_row_bin_tup_arr to get the keys for the new dictionary
    new_keys = [tuple(transMat_row_bin_tup_arr[i])
                for i in range(transMat_row_bin_tup_arr.shape[0])]

    # create the new dictionary
    edited_state_numberings = dict(zip(new_keys,
                                   list(np.arange(len(new_keys)))))

    return edited_state_numberings, transMat_row_bin_tup_arr, NaN_row_bin_tup_arr





# ------- Infomap functions ----------- #

def apply_infomap_clustering_to_transition_matrix(transition_matrix,
                                                  state_names,
                                                  numInfoMapTrials=10):
    ''' Apply infomap to the transition matrix.

    --- args ---
    transition_matrix: (numStates,numStates) the row normalized transition
                        matrix between states.
    state_names: the dictionary matching tuples of bin indices to a state index
    numInfoMapTrials: the number of trials for infomap

    --- Returns ---
    comms: the infomap object
    numClusters: the number of communities found by infomap
    cluster_state_idxs: a list over clusters, containing the state indices
                        of all states in this cluster.
                        i.e. row indices of the transition matrix
    cluster_state_tuples: contains the bintuples corresponding to the
                          data in cluster_state_idxs (made using state_names)


    '''
    # Parse some shapes
    numStates = transition_matrix.shape[0]

    # ---- make the graph ---- #
    # make a list of strings to name each vertex
    # e.g. stateIdx=0 -> 'state1'
    vertices = []
    for state_idx in range(numStates):
        vertices.append('state{0}'.format(state_idx+1))

    # make the tuple list
    tuple_list = []
    # loop over the rows of the transition matrix
    for row_state_idx in range(numStates):
        # create a tuple of (from,to,weight) for each transition
        from_state = vertices[row_state_idx]
        for col_state_idx in range(numStates):
            to_state = vertices[col_state_idx]
            transition_weight = transition_matrix[row_state_idx, col_state_idx]
            if transition_weight == 0:
                continue
            else:
                tup = tuple([from_state, to_state, transition_weight])
                tuple_list.append(tup)

    g = Graph.TupleList(tuple_list, weights=True, directed=True)

    # ---- find the communities with infomap ---- #
    comms = g.community_infomap(edge_weights='weight',
                                trials=numInfoMapTrials)
    subgraphs = comms.subgraphs()
    numClusters = len(subgraphs)

    # --- parse the output of infomap ------- #

    # cluster_state_idxs is a list over clusters, containing the state indices
    # of all states in this cluster. i.e. row indices of the transition matrix
    cluster_state_names = []
    cluster_state_idxs = []
    for clusterIdx in range(numClusters):
        stnames = subgraphs[clusterIdx].vs['name']
        idx_list = []
        for stnam in stnames:
            stidx = int(stnam[5:])-1  # -1 to move, for example, state1 -> 0
            idx_list.append(stidx)
        cluster_state_names.append(stnames)
        cluster_state_idxs.append(idx_list)

    # cluster_state_tuples contains the bintuples corresponding to the
    # data in cluster_state_idxs.
    cluster_state_tuples = []
    bin_tuple_list = list(state_names.keys())
    for clusterIdx in range(numClusters):
        cluster_tups = []
        for idx in cluster_state_idxs[clusterIdx]:
            tup = bin_tuple_list[idx]
            cluster_tups.append(tup)
        cluster_state_tuples.append(cluster_tups)

    return comms, numClusters, cluster_state_idxs, cluster_state_tuples


def make_binTuple_to_clusterIdx_dict(cluster_state_tuples):
    ''' Return a dictionary mapping tuples of binIdxs to clusterIdxs.

    --- args ---
    cluster_state_tuples: a list of infomap clusters results.
                          The list is of length K (for K clusters).
                          Each element is an array of shape (Ki,3),
                          where Ki denote the number of states in that cluster.
                          And each "3" corresponds to a tuple of binIdxs
                          defining each state.

    --- returns ---
    binTuple_to_clusterIdx_dict: a dictionary which matches state 3-tuples to
                                 an infomap cluster index.
                                 Use this to turn states into their cluster
                                 index.

    --- see also ---
    apply_infomap_clustering_to_transition_matrix() -> this function outputs
                                                       cluster_state_tuples
    '''
    numClusters = len(cluster_state_tuples)

    keys_list_over_clusters = []
    vals_list_over_clusters = []

    for clusterIdx in range(numClusters):
        # get the array of tuples for this cluster index
        cls_tups = cluster_state_tuples[clusterIdx]
        # get the number of tuples for this cluster index
        num_tups_for_cluster = cls_tups.shape[0]
        # prepare a list of the same length, containing just this clusterIdx
        # at each position
        cls_clusterIdx_list = [clusterIdx]*num_tups_for_cluster
        # transform the array of cluster tuples into a dictionary friendly list
        cls_bintups_list = list(list(cls_tups[i]) for i in
                                range(num_tups_for_cluster))
        # record the two lists
        keys_list_over_clusters.append(cls_bintups_list)
        vals_list_over_clusters.append(cls_clusterIdx_list)

    # concatenate the lists
    binTuple_to_clusterIdx_dict_keys = [tuple(item) for clstlist
                                        in keys_list_over_clusters
                                        for item in clstlist]
    binTuple_to_clusterIdx_dict_vals = [item for clstlist in
                                        vals_list_over_clusters
                                        for item in clstlist]

    binTuple_to_clusterIdx_dict = dict(zip(binTuple_to_clusterIdx_dict_keys,
                                           binTuple_to_clusterIdx_dict_vals))
    return binTuple_to_clusterIdx_dict


