import numpy as np

def xcorr(x, y, normed=True, mean_subtract=True, maxlags=10):
    ''' Cross correlation of two signals of equal length,
        designed to match MATLABS xcorr

    The correlation function is np.correlate
    The sliding inner-product
    c_{av}[k] = sum_n a[n+k] * conj(v[n])

    -- Returns --
    Returns the coefficients when normed=True
    Returns inner products when normed=False

    --- Thanks ---
    # https://github.com/colizoli/xcorr_python
    # https://github.com/colizoli/xcorr_python/blob/master/xcorr.py

    '''
    # Cross correlation of two signals of equal length
    # Returns the coefficients when normed=True
    # Returns inner products when normed=False
    # Usage: lags, c = xcorr(x,y,maxlags=len(x)-1)

    Nx = len(x)
    if Nx != len(y):
        raise ValueError('x and y must be equal length')

    if mean_subtract:
        x = x - np.nanmean(x)
        y = y - np.nanmean(y)

    c = np.correlate(x, y, mode='full')

    if normed:
        n = np.sqrt(np.dot(x, x) * np.dot(y, y)) # this is the transformation function
        c = np.true_divide(c,n)

    if maxlags is None:
        maxlags = Nx - 1

    if maxlags >= Nx or maxlags < 1:
        raise ValueError('maglags must be None or strictly '
                         'positive < %d' % Nx)

    lags = np.arange(-maxlags, maxlags + 1)
    c = c[Nx - 1 - maxlags:Nx + maxlags]
    return lags, c




def cross_correlation(ts1, ts2, maxLag=10, numSamples=1000, min_numSamples_allowed=900):
    ''' Compute the cross correlation, define as the pearson correlation
        coefficient computed at various lags, for two 1D timeseries of equal length

    -- args --
    ts1: shape (numFrames,)
    ts2: shape (numFrames,)
    maxLag: the highest lag you want to go to
    numSamples: the number of samples drawn to estimate the correlation
    min_numSamples_allowed: we discard samples with NaNs, so this allows you
                            to ensure we have enough usable samples.

    -- Refs --
    https://numpy.org/doc/stable/reference/generated/numpy.correlate.html
    https://en.wikipedia.org/wiki/Cross-correlation
    '''
    
    # -------------- tests -------#
    if ts1.shape[0] != ts2.shape[0]:
        raise ValueError('ts1 and ts2 must have the same shape')
    numFrames = ts1.shape[0]
    
    if maxLag > numFrames:
        raise ValueError('maxLag is bigger than the number of frames')
    # -----------------------------#  
        
    
    # make an array of lagsizes, up to maxLag on positive and negative
    lag_sizes = np.arange(-maxLag, maxLag+1)
    numLags = lag_sizes.shape[0] 

    # make an array to hold the correlation coefficient for each lag
    lag_results = np.zeros((numLags,))
    
    
    # loop over lags, do numSamples for each
    for ii,lagsize in enumerate(lag_sizes):

        # get the first frame number, and last frame number,
        # that we can use for this lag and still get an element for ts2
        if lagsize < 0:
            ts1_first_possible_frame = np.abs(lagsize)
            ts1_last_possible_frame = numFrames
        elif lagsize >= 0:
            ts1_first_possible_frame = 0
            ts1_last_possible_frame = numFrames - lagsize

        # get pairs, shape (numSamples,2),
        # the indices to use to draw samples from ts1 and ts2 for this lag
        ts1_idxs = np.random.randint(low=ts1_first_possible_frame, high=ts1_last_possible_frame, size=(numSamples,))
        ts2_plag_idxs = ts1_idxs + lagsize
        pairs = np.stack((ts1_idxs, ts2_plag_idxs)).T
        
        # using the indices, grab the data for all the samples
        ts1_vals = ts1[pairs[:,0]]
        ts2_vals = ts2[pairs[:,1]]

        # get samples without either being NaN
        #discard_pairIdxs = np.union1d( np.where(np.isnan(ts1_vals))[0], np.where(np.isnan(ts2_vals))[0] )
        use_pairIdxs = np.intersect1d( np.where(~np.isnan(ts1_vals))[0], np.where(~np.isnan(ts2_vals))[0] )
        if (use_pairIdxs.shape[0]) < min_numSamples_allowed:
            raise ValueError('cant get enough samples, is min_numSamples_allowed set ok?')

        # set the values to use
        ts1_values = ts1_vals[use_pairIdxs]
        ts2_values = ts2_vals[use_pairIdxs]

        # estimate the sample means
        ts1_mean = np.mean(ts1_values)
        ts2_mean = np.mean(ts2_values)

        # mean subtract the timeseries
        ts1_values = ts1_values - ts1_mean
        ts2_values = ts2_values - ts2_mean

        # find the correlation coefficient for this lag
        # the normalized covariance
        pr = np.sum(ts1_values*ts2_values) / np.sqrt(np.dot(ts1_values, ts1_values) * np.dot(ts2_values, ts2_values))
        lag_results[ii] = pr
        
        
    return lag_sizes, lag_results





def angular_cross_correlation(ts1, ts2, maxLag=10, numSamples=1000, min_numSamples_allowed=900):
    ''' Compute the angular correlation function for the given number of lags,
        for two 1D timeseries of the same length, using the circular correlation
        coefficient.

    -- args --
    ts1:
    ts2:
    maxLag: the highest lag you want to go to
    numSamples: the number of samples drawn to estimate the correlation
    min_numSamples_allowed: we discard samples with NaNs, so this allows you
                            to ensure we have enough usable samples.

    -- Refs --
    https://docs.astropy.org/en/stable/_modules/astropy/stats/circstats.html

    S. R. Jammalamadaka, A. SenGupta. "Topics in Circular Statistics".
       Series on Multivariate Analysis, Vol. 5, 2001.

    C. Agostinelli, U. Lund. "Circular Statistics from 'Topics in
       Circular Statistics (2001)'". 2015.
    '''
    # -------------- tests -------#
    if ts1.shape[0] != ts2.shape[0]:
        raise ValueError('ts1 and ts2 must have the same shape')
    numFrames = ts1.shape[0]
    
    if maxLag > numFrames:
        raise ValueError('maxLag is bigger than the number of frames')
    # -----------------------------#  

    # make an array of lagsizes, up to maxLag on positive and negative
    lag_sizes = np.arange(-maxLag, maxLag+1)
    numLags = (2*maxLag)+1 # = lag_sizes.shape[0]

    # make an array to hold the correlation coefficient for each lag
    lag_results = np.zeros((numLags,))
    
    # loop over lags, do numSamples for each
    for ii,lagsize in enumerate(lag_sizes):

        # get the first frame number, and last frame number,
        # that we can use for this lag and still get an element for ts2
        if lagsize < 0:
            ts1_first_possible_frame = np.abs(lagsize)
            ts1_last_possible_frame = numFrames
        elif lagsize >= 0:
            ts1_first_possible_frame = 0
            ts1_last_possible_frame = numFrames - lagsize

        # get pairs, shape (numSamples,2),
        # the indices to use to draw samples from ts1 and ts2 for this lag
        ts1_idxs = np.random.randint(low=ts1_first_possible_frame, high=ts1_last_possible_frame, size=(numSamples,))
        ts2_plag_idxs = ts1_idxs + lagsize
        pairs = np.stack((ts1_idxs, ts2_plag_idxs)).T
        
        # using the indices, grab the data for all the samples
        ts1_vals = ts1[pairs[:,0]]
        ts2_vals = ts2[pairs[:,1]]

        # get samples without either being NaN
        #discard_pairIdxs = np.union1d( np.where(np.isnan(ts1_vals))[0], np.where(np.isnan(ts2_vals))[0] )
        use_pairIdxs = np.intersect1d( np.where(~np.isnan(ts1_vals))[0], np.where(~np.isnan(ts2_vals))[0] )
        if (use_pairIdxs.shape[0]) < min_numSamples_allowed:
            raise ValueError('cant get enough samples, is min_numSamples_allowed set ok?')

        # set the values to use
        ts1_values = ts1_vals[use_pairIdxs]
        ts2_values = ts2_vals[use_pairIdxs]

        # subtract the sample means
        ts1_mean = mean_of_angle_timeseries(ts1_values)
        ts2_mean = mean_of_angle_timeseries(ts2_values)
        ts1_less_mean = ts1_values - ts1_mean
        ts2_less_mean = ts2_values - ts2_mean

        # find the circular correlation coefficient
        sin_a = np.sin(ts1_less_mean)
        sin_b = np.sin(ts2_less_mean)
        r = np.sum(sin_a*sin_b)/np.sqrt(np.sum(sin_a*sin_a)*np.sum(sin_b*sin_b))
        lag_results[ii] = r

    return lag_sizes, lag_results


def mean_of_angle_timeseries(angle_ts):
    ''' Compute the mean angle from the timeseries of angles
    '''
    theta_mean = np.arctan2(np.nansum(np.sin(angle_ts)), np.nansum(np.cos(angle_ts)))
    return theta_mean