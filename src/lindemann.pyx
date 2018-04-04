# cython: nonecheck=True
# cython: boundscheck=False
# cython: wraparound=False

import numpy as np
cimport numpy as np
cimport cython

from scipy.spatial.distance import cdist

def lindemann_process_frames(frames, nframes):
    """
    Parameters
    ----------
    frames: a numpy ndarray. shape = (natoms, 3)
    nframes: the maxium frames will be processed

    Return
    ------
    will return lindemann indices in numpy array

    Notes
    -----
    the matrix to be updated:
     array_mean: the mean values of rij averaged over frames. shape = (natoms, natoms)
     array_var: the variance of rij over frames. shape = (natoms, natoms)

    References
    ----------
    http://math.stackexchange.com/a/116344
    """

    natoms  = len(frames[0])

    # global array for incremental calculation
    # array_mean:
    cdef np.ndarray[np.float64_t, ndim=2] array_mean = np.zeros((natoms, natoms))
    # array_var:
    cdef np.ndarray[np.float64_t, ndim=2] array_var  = np.zeros((natoms, natoms))
    # distance matrix for different atom pairs in a single frame
    # array_distance:
    # initialized for estimating memory usage
    cdef np.ndarray[np.float64_t, ndim=2] array_distance = np.zeros((natoms, natoms))

    ##
    # esitmate memory usage
    # -----------------------------------------------------------------------------
    mem_nbytes = array_mean.nbytes + array_var.nbytes + array_distance.nbytes
    mem_gb = mem_nbytes/(1024**3)
    print("estimated memory usage: {:.2f} GB".format(mem_gb))

    ##
    # main loop
    # -----------------------------------------------------------------------------
    cdef long iframe = 1
    cdef long i, j
    cdef double xn, mean, delta
    cdef np.ndarray[np.float64_t, ndim=2] coords

    for coords in frames:
        print("processing frame {}/{}".format(iframe, nframes))
        # distance matrix for all atoms in current frame. shape = (natoms, natoms)
        array_distance = cdist(coords, coords)

        #################################################################################
        # update mean and var arrays based on Welford algorithm suggested by Donald Knuth
        #################################################################################
        for i in range(natoms):
            for j in range(i+1, natoms):
                xn = array_distance[i, j]
                mean = array_mean[i, j]
                var = array_var[i, j]
                delta = xn - mean
                # update mean
                array_mean[i, j] = mean + delta / iframe
                # update variance
                array_var[i, j] = var + delta * (xn - array_mean[i, j])
        iframe += 1
        if iframe > nframes: break

    # update elements in symmetry
    for i in range(natoms):
        for j in range(i+1, natoms):
            array_mean[j, i] = array_mean[i, j]
            array_var[j, i]  = array_var[i, j]

    print(array_mean)
    print(array_var/nframes)
    # drop nan values in diagonals created by 0.0/0.0
    lindemann_indices = np.nanmean(np.sqrt(array_var/nframes)/array_mean, axis=1)

    return lindemann_indices
