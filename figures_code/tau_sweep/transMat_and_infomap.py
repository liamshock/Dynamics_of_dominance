# This is a script to load the master thetaW_thetaL_dpp data,
# generate the transition matrix,
# compute the eigvals,
# perform infomap


import numpy as np
import h5py
import os
import time
from scipy.sparse.linalg import eigs as sparse_top_eigs
# import itertools
# import infomap
import argparse
# from igraph import *
import sys
# adding the infomap libs
# sys.path.append('/home/liam/code/FishTank_Analysis/lib/')
sys.path.append('/home/l/l-oshaughnessy/code/FishTank_Analysis/lib/')

from infomap_utils import make_dictionary_of_state_numberings
from infomap_utils import find_binIdxs_for_timeseries
from infomap_utils import build_transition_matrix
from infomap_utils import apply_infomap_clustering_to_transition_matrix
from infomap_utils import remove_empty_states

parser = argparse.ArgumentParser()
parser.add_argument("loadpath", type=str,
                    help='the filepath h5 containing the timeseries of thetas')
parser.add_argument("numInfoTrials", type=int,
                    help='the number of trials for the infomap calculation')
parser.add_argument("k", type=int,
                    help='the number of top eigvals to compute')
parser.add_argument("bin_choice", type=int,
                    help='an integer to decide which set of dpp_bins to use')
parser.add_argument('jobID', metavar='jobID', type=int,
                    help='the index of this job in the job array, for tau')
args = parser.parse_args()

# rename some args for ease
loadpath = args.loadpath
numInfoTrials = int(args.numInfoTrials)
k = int(args.k)
jobID = int(args.jobID)
bin_choice = int(args.bin_choice)

# use jobID to find the tau for this job
tau_list = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95]
#tau_list = [2,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95]
tau = tau_list[jobID-1]  # -1 moving from slurm 1-indexing to python 0-indexing

# derive the name of the folder to save in, and create it, and the savefile
mainsavefolder = os.path.join(os.path.dirname(loadpath),
                              'dpp_choice_{0}'.format(bin_choice),
                              'tau_'+str(tau).zfill(3))
if not os.path.exists(mainsavefolder):
    os.makedirs(mainsavefolder, exist_ok=True)
savepath = os.path.join(mainsavefolder, 'transmat_and_infomap.h5')

# print a summary of this job
t0 = time.time()
print('---- args ----')
print('loadpath: {0}'.format(loadpath))
print('savepath: {0}'.format(savepath))
print('tau = {0}'.format(tau))
print('numInfoTrials = {0}'.format(numInfoTrials))
print('bin_choice = {0}'.format(bin_choice))
print()


# ---------------------------------------------------------------#
#   Prepare the bins and other parameters
# ----------------------------------------------------------------#

# ---------- dpp bins ----------#

# a list of possible bin choices
dpp_bin_possibilities = [np.arange(0, 14+1, 1),
                         np.arange(0, 14+0.5, 0.5),
                         np.arange(0, 21+1, 1)]
# pick the one we want here
dpp_bins = dpp_bin_possibilities[bin_choice]
num_dpp_bin_edges = len(dpp_bins)
num_dpp_bins = num_dpp_bin_edges - 1

# ---------- theta bins ----------#

# set the number of bins for each variable
num_tetW_bins = 20
num_tetL_bins = 20

# derive the number of bin edges
num_tetW_bin_edges = num_tetW_bins + 1
num_tetL_bin_edges = num_tetL_bins + 1

# make the bins
tet_w_bins = np.linspace(-np.pi, np.pi, num_tetW_bin_edges)
tet_l_bins = np.linspace(-np.pi, np.pi, num_tetL_bin_edges)

# -------- other params -------- #


# make a dictionary of state numberings for looking up the correct
# state index after placing tet_tet_pairs into bins
num_bins_per_var = [num_tetW_bins, num_tetL_bins, num_dpp_bins]
state_numberings = make_dictionary_of_state_numberings(num_bins_per_var)


# ---------------------------------------------------------------#
#   Load the timeseries
# ----------------------------------------------------------------#

with h5py.File(loadpath, 'r') as hf:
    tseries = hf['tseries'][:]


# ---------------------------------------------------------------#
#  Build the transition matrix
# ---------------------------------------------------------------#

# find the binIdxs of each timepoint
tetW_bin_idxs = find_binIdxs_for_timeseries(tseries[:, 0], tet_w_bins)
tetL_bin_idxs = find_binIdxs_for_timeseries(tseries[:, 1], tet_l_bins)
dpp_bin_idxs = find_binIdxs_for_timeseries(tseries[:, 2], dpp_bins)
bin_idxs = np.stack([tetW_bin_idxs, tetL_bin_idxs, dpp_bin_idxs], axis=1)

transition_matrix = build_transition_matrix(bin_idxs, tau, state_numberings)
print()
print('Transition matrix made: {0} s'.format(time.time()-t0))

# ---------------------------------------------------------------#
#  Edit the transition matrix
# ---------------------------------------------------------------#
outs = remove_empty_states(transition_matrix, state_numberings)
edited_tmat, edited_state_numberings, transMat_row_bin_tup_arr, NaN_row_idxs, NaN_row_bin_tup_arr = outs
print()
print('Transition matrix edited: {0} s'.format(time.time()-t0))


# ---------------------------------------------------------------#
#  Compute the eigenspectrum
# ---------------------------------------------------------------#

def compute_top_K_eigs_sorted_by_largest_mag(matrix, k=10):
    ''' Compute the top K eigenvectors and values of the matrix,
        where top is defined as largest euclidean norm of the
        complex numbers,
        using a sparse matrix library,
        returning the eigs sorted by euclidean norm.

    --- args --
    matrix: a square matrix of real of complex values,
            can be sparse.
    k: the number of eigs we want to compute

    --- library used ---
    from scipy.sparse.linalg import eigs as sparse_top_eigs
    '''
    eigvals, eigvecs = sparse_top_eigs(matrix, k=k, which='LM')
    idxs = np.argsort(np.abs(eigvals))
    idxs = idxs[::-1]
    sorted_eig_vals = eigvals[idxs]
    sorted_eig_vecs = eigvecs[:, idxs]
    return sorted_eig_vals, sorted_eig_vecs


# compute the eigenvalues
sorted_eig_vals, sorted_eig_vecs = compute_top_K_eigs_sorted_by_largest_mag(transition_matrix,
                                                                            k=k)

print()
print('Eigenspectrum finished: {0} s'.format(time.time()-t0))

# ---------------------------------------------------------------#
#  Infomap
# ---------------------------------------------------------------#
infoouts = apply_infomap_clustering_to_transition_matrix(edited_tmat,
                                                         edited_state_numberings,
                                               numInfoMapTrials=numInfoTrials)
comms, numClusters, cluster_state_idxs, cluster_state_tuples = infoouts
print()
print('Infomap finished: {0} s'.format(time.time()-t0))


# ---------------------------------------------------------------#
#  Save
# ---------------------------------------------------------------#
with h5py.File(savepath, 'w') as hf:
    hf.create_dataset('edited_transition_matrix',
                      data=edited_tmat)
    hf.create_dataset('tseries', data=tseries)
    hf.create_dataset('tau', data=tau)
    hf.create_dataset('numInfoTrials', data=numInfoTrials)
    hf.create_dataset('tet_w_bins', data=tet_w_bins)
    hf.create_dataset('tet_l_bins', data=tet_l_bins)
    hf.create_dataset('dpp_bins', data=dpp_bins)
    hf.create_dataset('numClusters', data=numClusters)
    hf.create_dataset('transMat_row_bin_tup_arr',
                      data=transMat_row_bin_tup_arr)
    hf.create_dataset('NaN_row_idxs', data=NaN_row_idxs)
    hf.create_dataset('NaN_row_bin_tup_arr', data=NaN_row_bin_tup_arr)
    hf.create_dataset('sorted_eig_vals', data=sorted_eig_vals)
    hf.create_dataset('sorted_eig_vecs', data=sorted_eig_vecs)
    for clusterIdx in range(numClusters):
        hf.create_dataset('cluster_state_idxs/cluster{0}'.format(clusterIdx),
                          data=np.array(cluster_state_idxs[clusterIdx]))
        hf.create_dataset('cluster_state_tuples/cluster{0}'.format(clusterIdx),
                          data=np.array(cluster_state_tuples[clusterIdx]))

# -----------------------------------------------------------------#
print()
print('Finished: {0} s'.format(time.time()-t0))
