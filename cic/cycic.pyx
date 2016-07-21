from __future__ import division, print_function
import math
import numpy as np
cimport numpy as cnp
cimport cython

# Define types to be used as array dtypes
FLOAT = np.float
INT = np.int_
UINT = np.uint
ctypedef cnp.float_t FLOAT_t
ctypedef cnp.int_t INT_t
ctypedef cnp.uint_t UINT_t

@cython.wraparound(False)
@cython.boundscheck(False)
def cic_3d(cnp.ndarray[FLOAT_t, ndim=2] points, cnp.ndarray[INT_t, ndim=1] ndims, cnp.ndarray[FLOAT_t] weights=np.array((-1.,))):
    """ A Cython-based 3D cloud-in-cell algorithm.

    Parameters
    ----------
    points : np.ndarray[FLOAT_t, ndim=2]
        Must be non-negative. `points.shape[1]` must be 3.
    ndims : np.ndarray[FLOAT_t, ndim=2]
        Desired number of cells per dimension. Must have length 3.
    weights : np.ndarray[FLOAT_t, ndim=1], optional
        Weight for each point. There must be a weight for each point.

    Returns
    -------
    ndensity : np.ndarray[FLOAT_t, ndim=2]
        A 2D array of the number density per cell. It has shape (ndims + 2).

    Raises
    ------
    ValueError
        If any of the following conditions are not met:
        points.shape[1] == ndims.shape[0]
        points.shape[1] == 3
        points.shape[1] == ndims.shape[0]
        (ndims > 0).all()
        (points <= ndims).all() and (points > 0).all()
    """
    cic_sanitize(points, ndims)
    if ndims.shape[0] != 3:
        raise ValueError("Argument `ndims` must be of shape (3,).")
    cdef UINT_t space = 3
    # TODO: Check weights sanity: non-negative & share shape[0] with points
    # TODO: Allow weights to be 2 dimensional
    cdef FLOAT_t[:] weights_view
    if weights[0] < 0:
        weights_view = np.ones(points.shape[0])
    else:
        weights_view = weights

    # Pre-allocate variables to be used in the main loop
    # ndensity has shape ndims + 2 in case points are within 0.5 of the box edge.
    ndensity = np.zeros(ndims + 2, dtype=FLOAT)
    cdef FLOAT_t[:, :, :] ndensity_view = ndensity
    cdef UINT_t[:] cell = np.zeros(space, dtype=UINT)
    cdef FLOAT_t[:, :] delta = np.zeros((2, space), dtype=FLOAT)
    cdef FLOAT_t celldelta = 0
    cdef UINT_t i, j, k
    cdef size_t n

    # Loop over the number of points
    for n in range(points.shape[0]):
        # Loop over each dimension of each point
        for i in range(space):
            cell[i] = <UINT_t>(points[n, i] + 0.5)
            celldelta = cell[i] - points[n, i]
            delta[0, i] = 0.5 + celldelta
            delta[1, i] = 0.5 - celldelta
        # Loop over the 8 neighbors of each point
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    ndensity_view[cell[0]+i, cell[1]+j, cell[2]+k] += delta[i,0] * delta[j,1] * delta[k,2] * weights_view[n]
    return ndensity

@cython.wraparound(False)
@cython.boundscheck(False)
cpdef int cic_sanitize(FLOAT_t[:,:] points, INT_t[:] ndims):
    """Checks inputs to cloud-in-cell functions for sanity.

    Parameters
    ----------
    points : a 2D array of floats.
    ndims : a 1D array of integers.

    Returns
    -------
    int
        True if all test passes

    Raises
    ------
    ValueError
        If any of the following conditions are not met:
        points.shape[1] == ndims.shape[0]
        (ndims > 0).all()
        (points <= ndims).all() and (points > 0).all()
    """
    if points.shape[1] != ndims.shape[0]:
        raise ValueError("Dimension mismatch between arguements `points` and `ndims`.")

    cdef size_t i, j
    for i in range(ndims.shape[0]):
        if ndims[i] <= 0:
            raise ValueError("Argument `ndims` must be positive.")
    for i in range(points.shape[0]):
        for j in range(points.shape[1]):
            if points[i,j] > ndims[j]:
                raise ValueError("Argument `ndims` too small for the range of `points`")
            elif points[i,j] < 0:
                raise ValueError("Argument `points` must be non-negative.")
    return True
