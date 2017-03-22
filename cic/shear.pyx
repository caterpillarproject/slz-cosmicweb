from __future__ import division, print_function
import math
import numpy as np
import scipy.fftpack as fftpack
cimport numpy as cnp
cimport cython
from libc.math cimport M_PI

# Define types to be used as array dtypes
FLOAT = np.float
INT = np.int_
UINT = np.uint
ctypedef cnp.float_t FLOAT_t
ctypedef cnp.int_t INT_t
ctypedef cnp.uint_t UINT_t

@cython.wraparound(False)
@cython.boundscheck(False)
cpdef inline void shifty(FLOAT_t[:] y):
    """Shift elements of y around as if all the complex frequencies have been multiplied by 1j"""
    cdef size_t i, j
    cdef size_t n = y.shape[0]
    # loop should be 1, 3, 5, ..., N-1 if odd, 1, 3, 5, ..., N-2 if even.
    for i in range(1, n-1, 2):
        j = i + 1
        y[i], y[j] = -y[j], y[i]

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cpdef inline void freqy(FLOAT_t[:] y):
    """Multiply elements of y with their corresponding factors"""
    cdef size_t i
    cdef size_t n = y.shape[0]
    y[0] == 0
    for i in range(n):
        # In the original complex version, y[k] should be multiplied by 2j*pi*k/P
        # now, i + i % 2 happens to equal 2k
        # so we then multiply by pi and divide by period
        y[i] *= (i + (i % 2)) * M_PI
    # The Nyquist term, if it exists, must be murdered
    if n % 2 == 0:
        y[n-1] = 0

@cython.wraparound(False)
@cython.boundscheck(False)
cpdef cnp.ndarray[FLOAT_t, ndim=1] fft_deriv(cnp.ndarray[FLOAT_t, ndim=1] y):
    cdef cnp.ndarray[FLOAT_t, ndim=1] coeffs = fftpack.rfft(y)
    cdef size_t n = coeffs.shape[0]
    cdef size_t i
    shifty(coeffs)
    freqy(coeffs)
    return fftpack.irfft(coeffs)

@cython.wraparound(False)
@cython.boundscheck(False)
cpdef cnp.ndarray[FLOAT_t, ndim=5] calc_jacobian(FLOAT_t[:,:,:,:] vel_density):
    cdef size_t x_len = vel_density.shape[0]
    cdef size_t y_len = vel_density.shape[1]
    cdef size_t z_len = vel_density.shape[2]
    cdef size_t ndim = vel_density.shape[3]
    if ndim != 3:
        raise ValueError("vel_density.shape[3] must be 3")
    cdef cnp.ndarray[FLOAT_t, ndim=5] jacobian = np.zeros((x_len, y_len, z_len, ndim, ndim), dtype=FLOAT)
#    cdef FLOAT_t[:,:,:,:,:] jacobian_view = shear_tensor
    cdef size_t i, j, x, y, z
    cdef FLOAT_t[:] result
    cdef cnp.ndarray[FLOAT_t, ndim=1] vel_slice
    cdef FLOAT_t[:] vel_slice_view
    # Iterate over v_i
    for i in range(ndim):
        # iterate over x_j
        j = 0 # fft along x axis
        vel_slice = np.zeros(x_len)
        for y in range(y_len):
            for z in range(z_len):
                for x in range(x_len):
                    vel_slice[x] = vel_density[x,y,z,i]
                result = fft_deriv(vel_slice)
                # Pack result into shear tensor
                for x in range(x_len):
                    jacobian[x,y,z,i,j] = result[x]
        j = 1 # fft along y axis
        vel_slice = np.zeros(y_len)
        for x in range(y_len):
            for z in range(z_len):
                for y in range(y_len):
                    vel_slice[y] = vel_density[x,y,z,i]
                result = fft_deriv(vel_slice)
                # Pack result into shear tensor
                for y in range(y_len):
                    jacobian[x,y,z,i,j] = result[y]
        j = 2 # fft along z axis
        vel_slice = np.zeros(z_len)
        for x in range(y_len):
            for y in range(z_len):
                for z in range(z_len):
                    vel_slice[z] = vel_density[x,y,z,i]
                result = fft_deriv(vel_slice)
                # Pack result into shear tensor
                for z in range(z_len):
                    jacobian[x,y,z,i,j] = result[z]
    return jacobian

@cython.wraparound(False)
@cython.boundscheck(False)
cpdef void calc_shear_tensor_in_place(cnp.ndarray[FLOAT_t, ndim=5] jacobian):
    cdef size_t x_len = jacobian.shape[0]
    cdef size_t y_len = jacobian.shape[1]
    cdef size_t z_len = jacobian.shape[2]
    cdef size_t i_len = jacobian.shape[3]
    cdef size_t j_len = jacobian.shape[4]
    for x in range(x_len):
        for y in range(y_len):
            for z in range(z_len):
                for i in range(i_len):
                    jacobian[x,y,z,i,i] *= 2
                    for j in range(i):
                        jacobian[x,y,z,i,j] += jacobian[x,y,z,j,i]
                        jacobian[x,y,z,j,i] = jacobian[x,y,z,i,j]

@cython.wraparound(False)
@cython.boundscheck(False)
def eig_shear_tensor(cnp.ndarray[FLOAT_t, ndim=5] jacobian):
    cdef size_t x_len = jacobian.shape[0]
    cdef size_t y_len = jacobian.shape[1]
    cdef size_t z_len = jacobian.shape[2]
    cdef size_t ndim = jacobian.shape[3]
    cdef cnp.ndarray[FLOAT_t, ndim=4] eigvals = np.zeros((x_len, y_len, z_len, ndim), dtype=FLOAT)
    for x in range(x_len):
        for y in range(y_len):
            for z in range(z_len):
                pass
