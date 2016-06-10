from __future__ import division, print_function
import math
import numpy as np
cimport numpy as np
cimport cython

FLOAT = np.float
INT = np.int_
UINT = np.uint
ctypedef np.float_t FLOAT_t
ctypedef np.int_t INT_t
ctypedef np.uint_t UINT_t

@cython.wraparound(False)
@cython.boundscheck(False)
def cic_3d(np.ndarray[FLOAT_t, ndim=2] points, np.ndarray[INT_t, ndim=1] ndims):
    """ A Cython-based 3D cloud-in-cell algorithm.

    Parameters
    ----------
    np.ndarray points : (n,3) array of points.
    np.ndarray ndims : (3,) vector for the shape of the return value.
    """
    cic_sanitize(points, ndims)
    if ndims.shape[0] != 3:
        raise ValueError("Argument `ndims` must be of shape (3,).")

    ndensity = np.zeros(ndims + 2, dtype=FLOAT)
    cdef FLOAT_t[:, :, :] ndensity_view = ndensity
    cdef UINT_t[:] cell = np.zeros(3, dtype=UINT)
    cdef FLOAT_t[:, :] delta = np.zeros((2,3), dtype=FLOAT)
    cdef FLOAT_t celldelta = 0
    cdef np.ndarray[FLOAT_t, ndim=1] p
    cdef UINT_t i, j, k
    cdef size_t n
    for n in range(points.shape[0]):
        # Find the cell nearest to p
        for i in range(3):
            cell[i] = <UINT_t>(points[n, i] + 0.5)
            celldelta = cell[i] - points[n, i]
            delta[0, i] = 0.5 + celldelta
            delta[1, i] = 0.5 - celldelta
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    ndensity_view[<UINT_t>(cell[0]+i), <UINT_t>(cell[1]+j), <UINT_t>(cell[2]+k)] += delta[i,0] * delta[j,1] * delta[k,2]
    return ndensity

@cython.wraparound(False)
@cython.boundscheck(False)
cdef int cic_sanitize(FLOAT_t[:,:] points, INT_t[:] ndims):
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
                raise ValueError("Argument `points` must be nonnegative.")
    return True
