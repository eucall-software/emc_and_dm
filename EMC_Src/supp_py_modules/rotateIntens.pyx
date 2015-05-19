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
DTYPE = N.float
#DTYPE = N.float
#DTYPE = N.int
# "ctypedef" assigns a corresponding compile-time type to DTYPE_t. For
# every type in the numpy module there's a corresponding compile-time
# type with a _t-suffix.
ctypedef N.float_t DTYPE_t
cdef extern from "math.h":
    float expf(float x)
    float sqrtf(float x)
    float logf(float x)
    float cosf(float x)
    float sinf(float x)
    float fabs(float x)

cimport cython
#from cython.parallel import parallel, prange
@cython.boundscheck(False) # turn of bounds-checking for entire function

def d3_to_d1(int i, int j, int k, int d0, int d1):
    cdef int pos
    pos = ((i*d0 + j)*d1+ k)
    return pos

def make_rot(float q0, float q1, float q2, float q3, N.ndarray[N.float_t, ndim=2] curr_rot):
    cdef float q01, q02, q03, q11, q12, q13, q22, q23, q33
    q01 = q0*q1
    q02 = q0*q2
    q03 = q0*q3
    q11 = q1*q1
    q12 = q1*q2
    q13 = q1*q3
    q22 = q2*q2
    q23 = q2*q3
    q33 = q3*q3

    curr_rot[0][0] = 1. - 2.*(q22 + q33)
    curr_rot[0][1] = 2.*(q12 + q03)
    curr_rot[0][2] = 2.*(q13 - q02)
    curr_rot[1][0] = 2.*(q12 - q03)
    curr_rot[1][1] = 1. - 2.*(q11 + q33)
    curr_rot[1][2] = 2.*(q01 + q23)
    curr_rot[2][0] = 2.*(q02 + q13)
    curr_rot[2][1] = 2.*(q23 - q01)
    curr_rot[2][2] = 1. - 2.*(q11 + q22)

def orient_two_intensities(N.ndarray[N.float_t, ndim=1] den1, N.ndarray[N.float_t, ndim=1] den2, N.ndarray[N.float_t, ndim=1] qPos, N.ndarray[N.float_t, ndim=1] quats, int len):
    cdef int i, j, k, r, t, it, ijk, num_quat, num_qPos
    cdef int x, y, z, qmax1, qmax, min_quat
    cdef float tx, ty, tz, fx, fy, fz, cx, cy, cz, ii, jj, kk
    cdef float v0, v1, v2, v3, v4, v5, v6, v7
    cdef float den1_tot, min_score
    num_quat = quats.shape[0]/5
    num_qPos = qPos.shape[0]/3

    cdef N.ndarray[N.float_t, ndim=1] scores    = N.zeros([num_quat], dtype=N.float)
    cdef N.ndarray[N.float_t, ndim=1] ref_vals  = N.zeros([num_qPos], dtype=N.float)
    cdef N.ndarray[N.float_t, ndim=1] temp_vals = N.zeros([num_qPos], dtype=N.float)
    cdef N.ndarray[N.float_t, ndim=1] rot_pix   = N.zeros([3], dtype=N.float)
    cdef N.ndarray[N.float_t, ndim=2] curr_rot  = N.zeros([3, 3], dtype=N.float)

    qmax1     = len/2
    den1_tot  = 0.
    min_score = 1.E27
    min_quat  = 0

    for t from 0 <= t < num_qPos:
        if sqrtf(qPos[t*3]*qPos[t*3] + qPos[t*3+1]*qPos[t*3+1] + qPos[t*3+2]*qPos[t*3+2]) > 0.95*(float(qmax1)):
            continue
        tx = qPos[t*3]   + qmax1
        ty = qPos[t*3+1] + qmax1
        tz = qPos[t*3+2] + qmax1
        x  = int(tx)
        y  = int(ty)
        z  = int(tz)
        fx = tx - x
        fy = ty - y
        fz = tz - z
        cx = 1. - fx
        cy = 1. - fy
        cz = 1. - fz
        v0 = den1[((x*len + y)*len + z)]
        v1 = den1[((x*len + y)*len + (z + 1))]
        v2 = den1[((x*len + (y + 1))*len + z)]
        v3 = den1[((x*len + (y + 1))*len + (z + 1))]
        v4 = den1[(((x + 1)*len + y)*len + z)]
        v5 = den1[(((x + 1)*len + y)*len + (z + 1))]
        v6 = den1[(((x + 1)*len + (y + 1))*len + z)]
        v7 = den1[(((x + 1)*len + (y + 1))*len + (z + 1))]
        ref_vals[t] = cx*(cy*(cz*v0 + fz*v1) + fy*(cz*v2 + fz*v3)) + \
                      fx*(cy*(cz*v4 + fz*v5) + fy*(cz*v6 + fz*v7))
    for r from 0 <= r < num_quat:
        scores[r] = 0.
        make_rot(quats[5*r], quats[5*r+1], quats[5*r+2], quats[5*r+3], curr_rot)
        for t from 0 <= t < num_qPos:
            for i from 0 <= i < 3:
                rot_pix[i] = 0.
                for j from 0 <= j < 3:
                    rot_pix[i] += curr_rot[i, j]*qPos[t*3+j]
            if sqrtf(rot_pix[0]*rot_pix[0] + rot_pix[1]*rot_pix[1] + rot_pix[2]*rot_pix[2]) > 0.99*(float(qmax1)):
                continue
            tx = rot_pix[0] + qmax1
            ty = rot_pix[1] + qmax1
            tz = rot_pix[2] + qmax1
            x  = int(tx)
            y  = int(ty)
            z  = int(tz)
            fx = tx - x
            fy = ty - y
            fz = tz - z
            cx = 1. - fx
            cy = 1. - fy
            cz = 1. - fz
            v0 = den2[((x*len + y)*len + z)]
            v1 = den2[((x*len + y)*len + (z + 1))]
            v2 = den2[((x*len + (y + 1))*len + z)]
            v3 = den2[((x*len + (y + 1))*len + (z + 1))]
            v4 = den2[(((x + 1)*len + y)*len + z)]
            v5 = den2[(((x + 1)*len + y)*len + (z + 1))]
            v6 = den2[(((x + 1)*len + (y + 1))*len + z)]
            v7 = den2[(((x + 1)*len + (y + 1))*len + (z + 1))]
            temp_vals[t] = cx*(cy*(cz*v0 + fz*v1) + fy*(cz*v2 + fz*v3)) + \
                           fx*(cy*(cz*v4 + fz*v5) + fy*(cz*v6 + fz*v7))
            scores[r] += fabs(ref_vals[t] - temp_vals[t])

        if(scores[r] <= min_score):
            minScore  = scores[r]
            minquat   = r

    return scores
def interp_intensities(N.ndarray[N.float_t, ndim=1] den1, N.ndarray[N.float_t, ndim=1] den2, N.ndarray[N.float_t, ndim=1] qPos, N.ndarray[N.float_t, ndim=1] quat, int len):
    cdef int i, j, k, r, t, it, ijk, num_quat, num_qPos
    cdef int x, y, z, qmax1, qmax, min_quat
    cdef float tx, ty, tz, fx, fy, fz, cx, cy, cz, ii, jj, kk
    cdef float v0, v1, v2, v3, v4, v5, v6, v7
    cdef float den1_tot, min_score
    num_qPos = qPos.shape[0]/3

    cdef N.ndarray[N.float_t, ndim=1] rot_pix   = N.zeros([3], dtype=N.float)
    cdef N.ndarray[N.float_t, ndim=2] curr_rot  = N.zeros([3, 3], dtype=N.float)

    qmax1     = len/2
    den1_tot  = 0.
    min_score = 1.E27
    min_quat  = 0

    make_rot(quat[0], quat[1], quat[2], quat[3], curr_rot)
    for t from 0 <= t < num_qPos:
        for i from 0 <= i < 3:
            rot_pix[i] = 0.
            for j from 0 <= j < 3:
                rot_pix[i] += curr_rot[i, j]*qPos[t*3+j]
        if sqrtf(rot_pix[0]*rot_pix[0] + rot_pix[1]*rot_pix[1] + rot_pix[2]*rot_pix[2]) > 0.99*(float(qmax1)):
            continue
        tx = rot_pix[0] + qmax1
        ty = rot_pix[1] + qmax1
        tz = rot_pix[2] + qmax1
        x  = int(tx)
        y  = int(ty)
        z  = int(tz)
        fx = tx - x
        fy = ty - y
        fz = tz - z
        cx = 1. - fx
        cy = 1. - fy
        cz = 1. - fz
        v0 = den1[((x*len + y)*len + z)]
        v1 = den1[((x*len + y)*len + (z + 1))]
        v2 = den1[((x*len + (y + 1))*len + z)]
        v3 = den1[((x*len + (y + 1))*len + (z + 1))]
        v4 = den1[(((x + 1)*len + y)*len + z)]
        v5 = den1[(((x + 1)*len + y)*len + (z + 1))]
        v6 = den1[(((x + 1)*len + (y + 1))*len + z)]
        v7 = den1[(((x + 1)*len + (y + 1))*len + (z + 1))]
        # need to modify this to extract to den2!!!
        den2[t] = cx*(cy*(cz*v0 + fz*v1) + fy*(cz*v2 + fz*v3)) + \
                        fx*(cy*(cz*v4 + fz*v5) + fy*(cz*v6 + fz*v7))
