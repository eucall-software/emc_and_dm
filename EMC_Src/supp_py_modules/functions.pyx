import numpy as N
from scipy import signal
import math as M
# "cimport" is used to import special compile-time information
# about the numpy module (this is stored in a file numpy.pxd which is
# currently part of the Cython distribution).
cimport numpy as N
# We now need to fix a datatype for our arrays. I've used the variable
# DTYPE for this, which is assigned to the usual NumPy runtime
# type info object.
DTYPE = N.double
#DTYPE = N.float
#DTYPE = N.int
# "ctypedef" assigns a corresponding compile-time type to DTYPE_t. For
# every type in the numpy module there's a corresponding compile-time
# type with a _t-suffix.
ctypedef N.double_t DTYPE_t
cdef extern from "math.h":
    double expf(double x)
    double sqrtf(double x)
    double logf(double x)
    double cosf(double x)
    double fabs(double x)

cimport cython
#from cython.parallel import parallel, prange
def createRotationFromQuat(N.ndarray[N.double_t, ndim=1] quat):
    # Create rotation matrix from quaternion 4-tuple
    pass

def compareOrientations(N.ndarray[N.double_t, ndim=1] x, N.ndarray[N.double_t, ndim=1] y, N.ndarray[N.double_t, ndim=1] data, double pixSize):

    # Input quaternions, intensities1, intensities2, nx3 spatial frequency
    # locations qPos.
    # Interpolate intensities1 at qPos.
    #
    # For each quaternion find rotation matrix, and rotate qPos using this
    # matrix -> qPos'.
    # Interpolate qPos' from intensities2.
    # Score intensities1[qPos] and intensities2[qPos2], and report score.

    cdef double i, j, m, xlow, xhi, ylow, yhi, xmin, ymin
    cdef int xpos, ypos, r, c, l, xlen, x0, x1, y0, y1
    cdef int xLen = int(N.ceil((x.max() - x.min()) / pixSize))+5
    cdef int yLen = int(N.ceil((y.max() - y.min()) / pixSize))+5
    cdef N.ndarray[N.double_t, ndim=2] arr = N.zeros([xLen, yLen], dtype=N.double)
    cdef N.ndarray[N.double_t, ndim=2] wts = N.zeros([xLen, yLen], dtype=N.double)

    xlen = x.shape[0]
    xmin = x.min()
    ymin = y.min()

    for l from 0 <= l < xlen:
        m = data[l]

        ii = (x[l]-xmin)/pixSize
        jj = (y[l]-ymin)/pixSize
        x0 = ii
        xpos = x0
        x1 = x0+1
        y0 = jj
        y1 = y0+1
        ypos = y0

        xlow = ii - x0
        #xhi = x1 - ii
        xhi = 1. - xlow
        ylow = jj - y0
        #yhi = y1 - jj
        yhi = 1. - ylow

        arr[xpos, ypos]      += xhi  *yhi  *m
        wts[xpos, ypos]      += xhi  *yhi
        arr[xpos, ypos+1]    += xhi  *ylow *m
        wts[xpos, ypos+1]    += xhi  *ylow
        arr[xpos+1, ypos]    += xlow *yhi  *m
        wts[xpos+1, ypos]    += xlow *yhi
        arr[xpos+1, ypos+1]  += xlow *ylow *m
        wts[xpos+1, ypos+1]  += xlow *ylow

    for r from 0 <= r < xLen:
        for c from 0 <= c < yLen:
                if (wts[r,c] > 0.):
                    arr[r,c] /= wts[r,c]

    return arr
