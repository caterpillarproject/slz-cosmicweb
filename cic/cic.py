#!/usr/bin/env python

from __future__ import division, print_function
import math
import numpy as np
import matplotlib.pyplot as plt
import itertools
import random

# width of simulation box
BOXWIDTH = 25.0 # Mpc
# number of cells
NDIM = 15
CELLWIDTH = BOXWIDTH / NDIM
MPART = 1.9891e33 * 0.18 * 8.72e6
# spatial dimensions
SPACE = 3
NDIMS = (NDIM,) * SPACE
# number of particles
NPART = 512

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

def cloud_in_cell(points, boxwidths, ndims, mpart):
    cellwidth = boxwidths / ndims
    return cic(points/cellwidth, ndims) * mpart / cellwidth ** 3

def normalize_density(density, mass_scale, length_scale):
    return density * mass_scale / length_scale ** 3

def plot_density(density, points=None):
    if density.ndim > 2:
        image_array_2d_mesh = np.sum(density, axis=2)
    elif density.ndim == 2:
        image_array_2d_mesh = density
    else:
        raise ValueError("2D or above density only")
    plt.figure()
    plt.pcolor(image_array_2d_mesh.T, cmap=plt.cm.jet)
    if points != None:
        assert points.shape[1] == density.ndim
        plt.colorbar()
        plot_points = points.T + 1
        plt.plot(plot_points[0], plot_points[1], "ro")
    xlim, ylim = image_array_2d_mesh.shape
    plt.xlim(1,xlim - 1)
    plt.ylim(1,ylim - 1)
    plt.show()

if __name__ == "__main__":
    points = np.random.random_sample((NPART,SPACE)) * NDIM
    density = cic(points, NDIMS)
    plot_density(density, points)

