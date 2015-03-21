import os
import glob
import numpy as N
import time
from supp_py_modules import viewRecon as VR
from supp_py_modules import rotateIntens as rI
from supp_py_modules import read_results as read
from matplotlib import pyplot as plt
import matplotlib as M
from mpl_toolkits.axes_grid1 import make_axes_locatable

#TODO: This module has to be modified before it can work
#TODO: Functions in the current viewRecon.py has to be simplified and placed into this master script.

dirs = glob.glob("20*/")
cwd = os.getcwd()
outputFN = "orient.h5"
#for dir in dirs:
#    os.chdir(dir)
#    print "="*80
#    print "Making images for %s "%dir + "."*20
#    VR.make_panel_of_intensity_slices(outputFN, c_n=16)
#    VR.make_error_time_plot(outputFN)
#    VR.make_mutual_info_plot(outputFN)
#    print "="*80
#    os.chdir(cwd)

# tarBallName = (os.getcwd().split('/')[-1])+".tgz"
# os.system("tar -czf  %s  "%tarBallName + ''.join([s+"*.pdf " for s in dirs]))
# print "Images saved in %s" % tarBallName

curr_dir = dirs[0]
curr_file = os.path.join(curr_dir, outputFN)
t_intens = (read.extract_final_arr_from_h5(curr_file, "/history/intensities")).astype("float")
intens_len = len(t_intens)
qmax = intens_len/2
(q_low, q_high) = (6, int(0.4*qmax))
qRange1 = N.arange(-q_high, q_high + 1)
qRange2 = N.arange(-qmax, qmax + 1)
qPos0   = N.array([[i,j,0] for i in qRange1 for j in qRange1 if N.sqrt(i*i+j*j) > q_low]).astype("float")
qPos1   = N.array([[i,0,j] for i in qRange1 for j in qRange1 if N.sqrt(i*i+j*j) > q_low]).astype("float")
qPos2   = N.array([[0,i,j] for i in qRange1 for j in qRange1 if N.sqrt(i*i+j*j) > q_low]).astype("float")
qPos    = N.concatenate((qPos0, qPos1, qPos2))
print len(qPos)
qPos_full = N.array([[i,j,k] for i in qRange2 for j in qRange2 for k in qRange2]).astype("float")
quat_file = os.path.join(curr_dir, "quaternion.dat")
quats = (N.fromfile(quat_file, sep=" ")[1:]).reshape(-1,5).astype("float")

num_dirs = len(dirs)
intens_stack = N.zeros((num_dirs, intens_len, intens_len, intens_len))
intens_stack[0] = t_intens.copy()
intens_counter = 1
tt_intens = (t_intens>0.)*t_intens
tt_intens /= tt_intens.max()
(rows, cols) = (3, num_dirs/3+1)
M.rcParams.update({'font.size': 13})
fig, ax = plt.subplots(3, num_dirs/3 + 1, sharex=True, sharey=True, figsize=(2.5*cols, 2.5*rows))
for r in range(rows):
    for c in range(cols):
        if intens_counter >= num_dirs:
           break
        dir = dirs[intens_counter]
        t0 = time.time()
        curr_file = os.path.join(dir, outputFN)
        c_intens  = (read.extract_final_arr_from_h5(curr_file, "/history/intensities")).astype("float")
        cc_intens = (c_intens>0.)*c_intens
        cc_intens /= cc_intens.max()
        scores = rI.orient_two_intensities(t_intens.ravel(), c_intens.ravel(), qPos.ravel(), quats.ravel(), intens_len)
        ml_quat = quats[(scores.argsort())[0]]
        out_intens = N.zeros_like(cc_intens)
        rI.interp_intensities(c_intens.ravel(), out_intens.ravel(), qPos_full.ravel(), ml_quat, intens_len)
        t1 = time.time()
        im = ax[r, c].imshow(N.log(N.abs(out_intens[qmax])+1.E-7), cmap=plt.cm.coolwarm, aspect='auto')
        intens_stack[intens_counter] = out_intens.copy()
        print "Done orienting intensity %d of %d. Took %lf s."%(intens_counter, num_dirs, t1-t0)
        plt.draw()
        intens_counter += 1
im = ax[rows-1, cols-1].imshow(N.log(N.abs(t_intens[qmax])+1.E-7), cmap=plt.cm.coolwarm, aspect='auto')
fig.subplots_adjust(wspace=0.01)
cbar_ax = fig.add_axes([0.8, 0.1, 0.025, 0.8])
fig.colorbar(im, cax=cbar_ax, label="log10(intensities)")
(shx, shy) = tt_intens[qmax].shape
(h_shx, h_shy) = (shx/2, shy/2)
xt = N.linspace(0.5*h_shx, shx-.5*h_shx-1, 3).astype('int')
xt_l = N.linspace(-0.5*h_shx, 0.5*h_shx, 3).astype('int')
yt = N.linspace(0, shy-1, 3).astype('int')
yt_l = N.linspace(-1*h_shy, h_shy, 3).astype('int')
plt.setp(ax, xticks=xt, xticklabels=xt_l, yticks=yt, yticklabels=yt_l)
plt.show()

fig2, ax2 = plt.subplots(1, 1)
im = ax2.imshow(N.log(N.abs((N.median(intens_stack, axis=0))[qmax])+1.E-7))
fig2.subplots_adjust(wspace=0.01)
cbar_ax2 = fig2.add_axes([0.9, 0.1, 0.025, 0.8])
fig2.colorbar(im, cax=cbar_ax2, label="log10(intensities)")
plt.setp(ax2, xticks=xt, xticklabels=xt_l, yticks=yt, yticklabels=yt_l)
plt.show()
