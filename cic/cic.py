#!/usr/bin/env python

from __future__ import division, print_function
#import pyximport; pyximport.install()
#import cycic
import math
import numpy as np
import matplotlib.pyplot as plt
import itertools
import readsnapshots.readsnapHDF5 as rs
import h5py

# Width of simulation box
BOXWIDTH = 25.0 # Mpc
# Number of cells
NDIM = 256
CELLWIDTH = BOXWIDTH / NDIM
# Spatial dimensions
SPACE = 3
BOXWIDTHS = np.array((BOXWIDTH,) * SPACE)
NDIMS = (NDIM,) * SPACE
# Unit conversions
MPCTOCM = 3.08567758e24
SOLTOGRAM = 1.9891e33
# Baryonic density
OMEGAB = 0.18
# Particle mass
MPART = OMEGAB * 8.72e6 * SOLTOGRAM
# Number of particles for random data generation
NPART = 512

# File location
SNAPPREFIX = "/home/slz/Dropbox (MIT)/urop/2016-c/data/parent/snapdir_127/snap_127"

def cic(points, ndims):
    """A basic cloud-in-cell algorithm for arbitrary spatial dimensions.
    
    Parameters
    ----------
    points : iterable of points
    ndims : number of cells, per side"""
    # spatial dimentions
    space = len(ndims)
    points = np.array(points, copy=False)
    ndims = np.array(ndims, copy=False)
    assert space == points.shape[1]
    assert (points <= ndims).all()
    # Initialize number density field
    ndensity = np.zeros(ndims + 2)

    for p in points:
        # Find the cell nearest to p
        cell = np.array(np.floor(p - 0.5) + 1, dtype=int)
        # Displacement of p to its nearest and farthest cells
        delta = np.array((0.5 + cell - p, 0.5 - cell + p))
        # Split the particle into 2**space pieces
        numbers = np.prod([i for i in itertools.product(*delta.T)], axis=1).reshape((2,)*space)
        # Add the pieces to overall number density
        for dcell in itertools.product(range(2), repeat=space):
            ndensity[tuple(cell + dcell)] += numbers[dcell]
    return ndensity

def normalize_position(points, boxwidths, ndims):
    cellwidths = boxwidths / ndims
    return points / cellwidths

def normalize_density(density, cellwidths, mpart):
    assert density.ndim == len(cellwidths)
    density /= mpart / np.prod(cellwidths)

def plot_density(density, points=None):
    if density.ndim > 2:
        image_array_2d_mesh = np.sum(density, axis=2)
    elif density.ndim == 2:
        image_array_2d_mesh = density
    else:
        raise ValueError("2D or above density only")
    plt.figure()
    plt.pcolor(image_array_2d_mesh.T, cmap=plt.cm.jet)
    plt.colorbar()
    if points != None:
        assert points.shape[1] == density.ndim
        plot_points = points.T + 1
        plt.plot(plot_points[0], plot_points[1], "ro")
    xlim, ylim = image_array_2d_mesh.shape
    plt.xlim(1,xlim - 1)
    plt.ylim(1,ylim - 1)
    plt.show()

def write_density(density, filename):
    with h5py.File(filename, "w") as f:
        density_dset = f.create_dataset("DENSITY", density.shape, dtype=density.dtype)
        density_dset[...] = density

def read_density(filename):
    with h5py.File(filename, "r") as f:
        return f["DENSITY"][()]

def random_data_demo():
    points = np.random.random_sample((NPART,SPACE)) * NDIM
    density = cic(points, NDIMS)
    plot_density(density, points)

def read_position_data():
    positions = rs.read_block(SNAPPREFIX, "POS ", parttype=1)
    return positions

def calculate_params(points, ndim=NDIM):
    space = points.shape[1]
    boxwidths = np.array((math.ceil(points.max()),) * space)
    ndims = np.array((ndim,) * space)
    points = normalize_position(points, boxwidths, ndims)
    return points, boxwidths, ndims

def run_cic(points, boxwidths, ndims, mpart=MPART):
    density = cic(points, ndims)
    normalize_density(density, boxwidths/ndims, mpart)
    return density

if __name__ == "__main__":
    pass

