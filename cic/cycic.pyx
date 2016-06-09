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
    if points.shape[1] != 3:
        raise ValueError("Argument `points` must of shape (:,3).")
    elif (points < 0).any():
        raise ValueError("Argument `points` must be positive.")
    if ndims.shape[0] != 3:
        raise ValueError("Argument `ndims` must have shape (3,).")
    if (points >= ndims).any():
        raise ValueError("Argument `ndims` too small to contain the range of `points`.")
#    assert points.dtype == FLOAT and ndims.dtype == FLOAT

    ndensity = np.zeros(ndims + 2, dtype=FLOAT)
    cdef FLOAT_t[:, :, :] ndensity_view = ndensity
#    cdef FLOAT_t[:, :] points_view = points
    cdef np.ndarray[FLOAT_t, ndim=1] cell# = np.zeros(3, dtype=UINT)
    cdef np.ndarray[FLOAT_t, ndim=2] delta = np.zeros((2,3), dtype=FLOAT)
    cdef np.ndarray[FLOAT_t, ndim=1] celldelta# = np.zeros(3, dtype=FLOAT)
    cdef np.ndarray[FLOAT_t, ndim=1] p
    cdef UINT_t i, j, k
    cdef unsigned long n
#    for n in range(points.shape[0]):
    for p in points:
        # Find the cell nearest to p
#        p = points[n]
        cell = np.rint(p)
        celldelta = cell - p
        delta[0] = 0.5 + celldelta
        delta[1] = 0.5 - celldelta
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    ndensity_view[<UINT_t>(cell[0]+i), <UINT_t>(cell[1]+j), <UINT_t>(cell[2]+k)] += delta[i,0] * delta[j,1] * delta[k,2]
    return ndensity
