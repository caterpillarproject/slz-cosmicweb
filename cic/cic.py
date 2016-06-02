# cloud-in-cell algorithm for tak.ndim distribution of particles (x,y,z) 
# and turn.ndim them o a regularly space mesh of dimension ndim.
# Author: Brendan F. Griffen
# Last Updated: 23-Nov-2012
# Note: Can take sustantial time if loading entire snapshot.

import readsnap as rs
import numpy as np
import matplotlib.pyplot as plt
import random
import mpl_toolkits.mplot3d.axes3d as p3
import math
#import time
#tic = time.clock()

# width of simulation box
boxwidth = 25.0   # Mpc

# number of cells
ndim = 512.0

# conversions
Mpctocm = 3.08567758*10**24
Soltogram = 1.9891*10**33

volcell = (boxwidth*Mpctocm/ndim)**3

# baryon density
Omegab = 0.18

# particle mass
mpart = Soltogram*Omegab*8.72*10**6

print 'Reading snapshot...'

# read Gadget file
pos = rs.read_block('snap_064',"POS ",parttype=1)

# select range of particles to use
#xpos = pos[:,0]
#ypos = pos[:,1]
#zpos = pos[:,2]


npoints = len(pos)

print "Number of particles to be put in mesh: ", npoints
#pos = np.ones((npoints,3))

#pos[:,0] = np.array([(boxwidth*random.random()) for i in xrange(npoints)])
#pos[:,1] = np.array([(boxwidth*random.random()) for i in xrange(npoints)])
#pos[:,2] = np.array([(boxwidth*random.random()) for i in xrange(npoints)])

x_min = 0.0
x_max = boxwidth

y_min = 0.0
y_max = boxwidth

z_min = 0.0
z_max = boxwidth

dense = np.zeros((ndim+2,ndim+2,ndim+2))

print "Populating Mesh:"

for i in range(0,npoints):
    x1 = np.floor((pos[i,0] - x_min)*(ndim/boxwidth) - 0.5) + 1
    x2 = x1 + 1

    y1 = np.floor((pos[i,1] - y_min)*(ndim/boxwidth) - 0.5) + 1
    y2 = y1 + 1
      
    z1 = np.floor((pos[i,2] - z_min)*(ndim/boxwidth) - 0.5) + 1
    z2 = z1 + 1

    dx1 = 1 + x1 - 0.5 + ((ndim/boxwidth)*(x_min - pos[i,0])) 
    dy1 = 1 + y1 - 0.5 + ((ndim/boxwidth)*(y_min - pos[i,1]))
    dz1 = 1 + z1 - 0.5 + ((ndim/boxwidth)*(z_min - pos[i,2]))

    dx2 = 1 - dx1
    dy2 = 1 - dy1
    dz2 = 1 - dz1
    
    #print x1, x2, y1, y2, z1, z2
    #print dx1, dx2, dy1, dy2, dz1, dz2

    dense[int(x1),int(y1),int(z1)] = dense[int(x1),int(y1),int(z1)] + dx1 * dy1 * dz1 * mpart
    dense[int(x2),int(y1),int(z1)] = dense[int(x2),int(y1),int(z1)] + dx2 * dy1 * dz1 * mpart
    dense[int(x1),int(y2),int(z1)] = dense[int(x1),int(y2),int(z1)] + dx1 * dy2 * dz1 * mpart
    dense[int(x2),int(y2),int(z1)] = dense[int(x2),int(y2),int(z1)] + dx2 * dy2 * dz1 * mpart
    dense[int(x1),int(y1),int(z2)] = dense[int(x1),int(y1),int(z2)] + dx1 * dy1 * dz2 * mpart
    dense[int(x2),int(y1),int(z2)] = dense[int(x2),int(y1),int(z2)] + dx2 * dy1 * dz2 * mpart
    dense[int(x1),int(y2),int(z2)] = dense[int(x1),int(y2),int(z2)] + dx1 * dy2 * dz2 * mpart
    dense[int(x2),int(y2),int(z2)] = dense[int(x2),int(y2),int(z2)] + dx2 * dy2 * dz2 * mpart

    #if (x1 > 0.0) & (y1 > 0.0) & (z1 > 0.0):
    #    dense[int(x1),int(y1),int(z1)] = dense[int(x1),int(y1),int(z1)] + dx1 * dy1 * dz1 * mpart
    #if (x2 <= ndim) & (y1 > 0.0) & (z1 > 0.0):
    #    dense[int(x2),int(y1),int(z1)] = dense[int(x2),int(y1),int(z1)] + dx2 * dy1 * dz1 * mpart

    #if (x1 > 0.0) & (y2 <= ndim) & (z1 > 0.0):
    #    dense[int(x1),int(y2),int(z1)] = dense[int(x1),int(y2),int(z1)] + dx1 * dy2 * dz1 * mpart
    #if (x2 > 0.0) & (y2 > 0.0) & (z1 > 0.0):
    #    dense[int(x2),int(y2),int(z1)] = dense[int(x2),int(y2),int(z1)] + dx2 * dy2 * dz1 * mpart

    #if (x1 > 0.0) & (y1 > 0.0) & (z2 > 0.0):
    #    dense[int(x1),int(y1),int(z2)] = dense[int(x1),int(y1),int(z2)] + dx1 * dy1 * dz2 * mpart
    #if (x2 <= ndim) & (y1 > 0.0) & (z2 <= ndim):
    #    dense[int(x2),int(y1),int(z2)] = dense[int(x2),int(y1),int(z2)] + dx2 * dy1 * dz2 * mpart

    #if (x1 > 0.0) & (y2 <= ndim) & (z2 <= ndim):
    #    dense[int(x1),int(y2),int(z2)] = dense[int(x1),int(y2),int(z2)] + dx1 * dy2 * dz2 * mpart
    #if (x2 <= ndim) & (y2 <= ndim) & (z2 <= ndim):
    #    dense[int(x2),int(y2),int(z2)] = dense[int(x2),int(y2),int(z2)] + dx2 * dy2 * dz2 * mpart

# # calculate density [g/cm^3]
#toc = time.clock()
#tottime = toc - tic
print "Finished mesh!"
#print "Only took...", tottime

dense = dense/volcell

# # plot
imageArray2Dmesh = np.mean(dense**2, axis=2);
dense = 0.0;
plt.figure()
plt.pcolor(imageArray2Dmesh, cmap=plt.cm.jet)
plt.colorbar()
plt.xlabel('x-cell')
plt.ylabel('y-cell')
plt.xlim((1,ndim))
plt.ylim((1,ndim))
plt.show()
plt.savefig("viz.pdf",format='pdf')

# print "min x1:",min(x1)
# print "max x1:",max(x1)
# print "min y1:",min(y1)
# print "max y1:",max(y1)
# print "min z1:",min(z1)
# print "max z1:",max(z1)

# print "min x2:",min(x2)
# print "max x2:",max(x2)
# print "min y2:",min(y2)
# print "max y2:",max(y2)
# print "min z2:",min(z2)
# print "max z2:",max(z2)

# print "min dx1:",min(dx1)
# print "max dx1:",max(dx1)
# print "min dy1:",min(dy1)
# print "max dy1:",max(dy1)
# print "min dz1:",min(dz1)
# print "max dz1:",max(dz1)

# print "min dx2:",min(dx2)
# print "max dx2:",max(dx2)
# print "min dy2:",min(dy2)
# print "max dx2:",max(dy2)
# print "min dz2:",min(dz2)
# print "max dx2:",max(dz2)