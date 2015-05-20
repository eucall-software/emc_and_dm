import numpy as N
import h5py
import os
import re
import time
import sys
import glob
from optparse import OptionParser
import matplotlib.pyplot as plt
import matplotlib.ticker as Tick
import matplotlib as M
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D

###############################################################
# Argument parser for important input
# Specify data input and output directories, as well as timeStamps
# Seek and associate data in input directory
###############################################################

cwd = os.getcwd()+"/"

parser = OptionParser()

parser.add_option("-i", "--inDir", action="store", type="string", dest="inputDir", help="absolute path to input intensities (orient_out.h5)", metavar="", default=cwd)
parser.add_option("-s", "--srcDir", action="store", type="string", dest="srcDir", help="absolute path to source files for executables", metavar="", default=cwd)
parser.add_option("-T", "--tmpOutDir", action="store", type="string", dest="tmpOutDir", help="temporary directory to store intermediate states of calculation", metavar="", default=cwd)
parser.add_option("-o", "--outDir", action="store", type="string", dest="outDir", help="absolute path to output", metavar="", default=cwd)

parser.add_option("-t", "--trials", action="store", type="int", dest="numTrials", help="", metavar="", default=100)
parser.add_option("-a", "--start_ave", action="store", type="int", dest="startAve", help="", metavar="", default=15)
parser.add_option("-n", "--iter", action="store", type="int", dest="numIter", help="", metavar="", default=50)
parser.add_option("-l", "--leash", action="store", type="float", dest="leash", help="", metavar="", default=0.2)
parser.add_option("-c", "--shrinkCycles", action="store", type="int", dest="shrinkCycles", help="", metavar="", default=10)

ct = time.localtime()
currTimeStamp = "%04d_%02d_%02d_%02d_%02d_%02d"%(ct.tm_year, ct.tm_mon, ct.tm_mday, ct.tm_hour, ct.tm_min, ct.tm_sec)
parser.add_option("-t", "--timeStamp", action="store", type="string", dest="timeStamp", help="time stamp to use for output", metavar="", default=currTimeStamp)

(op, args) = parser.parse_args()
runLogFile = os.path.join(op.tmpOutDir, op.timeStamp + ".log")

################################################################################
# Convenience functions for this script
################################################################################

def print_to_log(msg, log_file=runLogFile):
    fp = open(log_file, "a")
    t_msg = time.asctime() + ":: " + msg
    fp.write(t_msg)
    fp.write("\n")
    fp.close()

def create_directory(dir_name, log_file=runLogFile, err_msg=""):
    if os.path.exists(dir_name):
        print_to_log(dir_name + " exists! " + err_msg, log_file=log_file)
    else:
        print_to_log("Creating " + dir_name, log_file=log_file)
        os.makedirs(dir_name)

def load_intensities(ref_file):
    fp      = h5py.File(ref_file, 'r')
    t_intens = (fp["data/data"].value()).astype("float")
    fp.close()
    intens_len = len(t_intens)
    qmax    = intens_len/2
    (q_low, q_high) = (15, int(0.9*qmax))
    qRange1 = N.arange(-q_high, q_high + 1)
    qRange2 = N.arange(-qmax, qmax + 1)
    qPos0   = N.array([[i,j,0] for i in qRange1 for j in qRange1 if N.sqrt(i*i+j*j) > q_low]).astype("float")
    qPos1   = N.array([[i,0,j] for i in qRange1 for j in qRange1 if N.sqrt(i*i+j*j) > q_low]).astype("float")
    qPos2   = N.array([[0,i,j] for i in qRange1 for j in qRange1 if N.sqrt(i*i+j*j) > q_low]).astype("float")
    qPos    = N.concatenate((qPos0, qPos1, qPos2))
    qPos_full = N.array([[i,j,k] for i in qRange2 for j in qRange2 for k in qRange2]).astype("float")
    return (qmax, t_intens, intens_len, qPos, qPos_full)

def zero_neg(x):
    return 0. if x<=0. else x
v_zero_neg  = N.vectorize(zero_neg)

def find_two_means(vals, v0, v1):
    v0_t    = 0.
    v0_t_n  = 0.
    v1_t    = 0.
    v1_t_n  = 0.
    for vv in vals:
        if (N.abs(vv-v0) > abs(vv-v1)):
            v1_t    += vv
            v1_t_n  += 1.
        else:
            v0_t    += vv
            v0_t_n  += 1.
    return (v0_t/v0_t_n, v1_t/v1_t_n)

def cluster_two_means(vals):
    (v0,v1)     = (0.,0.1)
    (v00, v11)  = find_two_means(vals, v0, v1)
    err = 0.5*(N.abs(v00-v0)+N.abs(v11-v1))
    while(err > 1.E-5):
        (v00, v11)  = find_two_means(vals, v0, v1)
        err         = 0.5*(N.abs(v00-v0)+N.abs(v11-v1))
        (v0, v1)    = (v00, v11)
    return (v0, v1)

def support_from_autocorr(auto, qmax, thr_0, thr_1, supp_file, kl=1, write=True):
    pos     = N.argwhere(N.abs(auto-thr_0) > N.abs(auto-thr_1))
    pos_set = set()
    pos_list= []
    kerl    = range(-kl,kl+1)
    ker     = [[i,j,k] for i in kerl for j in kerl for k in kerl]

    def trun(v):
        return int(N.ceil(0.5*v))

    v_trun = N.vectorize(trun)

    for (pi, pj, pk) in pos:
        for (ci, cj, ck) in ker:
            pos_set.add((pi+ci, pj+cj, pk+ck))
    for s in pos_set:
        pos_list.append([s[0], s[1], s[2]])

    pos_array = N.array(pos_list)
    pos_array -= [a.min() for a in pos_array.transpose()]
    pos_array = N.array([v_trun(a) for a in pos_array])

    if write:
        fp  = open(supp_file, "w")
        fp.write("%d %d\n"%(qmax, len(pos_array)))
        for p in pos_array:
            fp.write("%d %d %d\n" % (p[0], p[1], p[2]))
        fp.close()

    return pos_array

def show_support(support):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    (x,y,z) = support.transpose()
    ax.scatter(x, y, z, c='r', marker='s')
    plt.show()


def parse_shrinkwrap_log(shrinkwrap_fn):
    fp = open(shrinkwrap_fn, "r")
    lines = fp.readlines()
    fp.close()
    lst = []
    for ll in lines:
        m = re.match("supp_vox = (\d+)\s", ll)
        if m:
            (supp_size) = m.groups()
            lst.append(int(supp_size))
    return N.array(lst)

def parse_error_log(err_fn):
    fp = open(err_fn, "r")
    lines = fp.readlines()[2:]
    fp.close()
    lst = []
    for ll in lines:
        m = re.match("iter = (\d+)\s+error = (\d+\.\d+)", ll)
        if m:
            (iter, err) = m.groups()
            lst.append(float(err))
    return N.array(lst)

def extract_object(object_fn):
    tmp = N.fromfile(object_fn, sep=" ")
    s = tmp.shape[0]
    l = int(round(s**(1./3.)))
    return tmp.reshape(l,l,l)

create_directory(op.tmpOutDir)
create_directory(op.outDir)
runInstanceDir = os.path.join(op.tmpOutDir, "phase_out_" + op.timeStamp + "/")
create_directory(runInstanceDir, err_msg=" Assuming that you are continuing a previous reconstruction.")

outputLog           = os.path.join(runInstanceDir, "phasing.log")
supportFile         = os.path.join(runInstanceDir, "support.dat")
inputIntensityFile  = glob.glob(os.path.join(op.inputDir, "orient_out*.h5"))[0]
intensityTmpFile    = os.path.join(runInstanceDir, "object_intensity.dat")
outputFile          = os.path.join(op.outDir, "phase_out_" + op.timeStamp + ".h5")

#Read intensity and translate into ASCII *.dat format
(qmax, t_intens, intens_len, qPos, qPos_full) = load_intensities(inputIntensityFile)
input_intens = t_intens
input_intens.tofile(intensityTmpFile, sep=" ")

# Compute autocorrelation and support
print "Computing autocorrelation..."
input_intens  = v_zero_neg(input_intens.ravel()).reshape(input_intens.shape)
auto        = N.fft.fftshift(N.abs(N.fft.fftn(N.fft.ifftshift(input_intens))))
print "Using 2-means clustering to determine significant voxels in autocorrelation..."
(a_0, a_1)  = cluster_two_means(auto.ravel())
print "Determining support from autocorrelation (will write to support.dat by default)..."
support     = support_from_autocorr(auto, qmax, a_0, a_1, supportFile)

#Start phasing
#Store parameters into phase_out.h5.
#Link executable from compiled version in srcDir to tmpDir
os.chdir(op.tmpOutDir)
inputOptions = (op.numTrials, op.numIter, op.startAve, op.leash, op.shrinkCycles)
os.system("./object_recon %d %d %d %lf %d&"%inputOptions)

min_objects     = glob.glob("finish_min_object*.dat")
logFiles        = glob.glob("object*.log")
shrinkWrapFile  = "shrinkwrap.log"
fin_object      = "finish_object.dat"

print "Done with reconstructions, now saving output from final shrink_cycle to h5 file"
fp          = h5py.File(outputFile, "w")
g_data      = fp.create_group("data")
g_params    = fp.create_group("params")
g_supp      = fp.create_group("/history/support")
g_err       = fp.create_group("/history/error")
g_hist_obj  = fp.create_group("/history/object")
for n, mo in enumerate(logFiles):
    err = parse_error_log(mo)
    g_err.create_dataset("%0.4d"%(n+1), data=err, compression="gzip")
    os.remove(mo)

for n, ob_fn in enumerate(min_objects):
    obj = extract_object(ob_fn)
    g_hist_obj.create_dataset("%0.4d"%(n+1), data=obj, compression="gzip")
    os.remove(mo)

finish_object = extract_object("finish_object.dat")
g_data.create_dataset("electronDensity", data=finish_object, compression="gzip")
os.system("cp finish_object.dat start_object.dat")

g_params.create_dataset("DM_support",           data=support, compression="gzip")
g_params.create_dataset("DM_numTrials",         data=op.numTrials)
g_params.create_dataset("DM_numIterPerTrial",   data=op.numIter)
g_params.create_dataset("DM_startAvePerIter",   data=op.startAve)
g_params.create_dataset("DM_leashParameter",    data=op.leash)
g_params.create_dataset("DM_shrinkwrapCycles",  data=op.shrinkCycles)

shrinkWrap = parse_shrinkwrap_log(shrinkWrapFile)
fp.create_dataset("/history/shrinkwrap", data=shrinkWrap, compression="gzip")
fp.create_dataset("version", data=h5py.version.hdf5_version)

fp.close()
