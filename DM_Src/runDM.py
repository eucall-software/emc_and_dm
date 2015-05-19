import numpy as N
import h5py
import os
import time
import sys
import glob
from optparse import OptionParser
import matplotlib.pyplot as plt
import matplotlib.ticker as Tick
import matplotlib as M
#from mpl_toolkits.axes_grid1 import make_axes_locatable
#from mpl_toolkits.mplot3d import Axes3D

###############################################################
# Argument parser for important input
# Specify data input and output directories, as well as timeStamps
# Seek and associate data in input directory
###############################################################

cwd = os.getcwd()+"/"

srcDir = os.path.join(cwd, "../diffr")

parser = OptionParser()

parser.add_option("-i", "--inDir", action="store", type="string", dest="inputDir", help="absolute path to input diffraction frames (diffr*.h5)", metavar="", default=srcDir)

parser.add_option("-s", "--srcDir", action="store", type="string", dest="srcDir", help="absolute path to source files for executables", metavar="", default=cwd)

parser.add_option("-T", "--tmpOutDir", action="store", type="string", dest="tmpOutDir", help="temporary directory to store intermediate states of calculation", metavar="", default=cwd)

parser.add_option("-o", "--outDir", action="store", type="string", dest="outDir", help="absolute path to output", metavar="", default=cwd)

parser.add_option("-q", "--initialQuaternion", action="store", type="int", dest="initialQuat", help="", metavar="", default=5)

parser.add_option("-Q", "--maxQuaternion", action="store", type="int", dest="maxQuat", help="", metavar="", default=9)

parser.add_option("-m", "--maxIterations", action="store", type="int", dest="maxIter", help="", metavar="", default=200)

parser.add_option("-e", "--minError", action="store", type="float", dest="minError", help="minimum error for terminating iterative intensity reconstructions", metavar="", default=4.E-8)

parser.add_option("-p", action="store_true", dest="plot", default=True)

parser.add_option("-d", action="store_true", dest="detailed", default=False)

ct = time.localtime()
currTimeStamp = "%04d_%02d_%02d_%02d_%02d_%02d"%(ct.tm_year, ct.tm_mon, ct.tm_mday, ct.tm_hour, ct.tm_min, ct.tm_sec)
parser.add_option("-t", "--timeStamp", action="store", type="string", dest="timeStamp", help="time stamp to use for output", metavar="", default=currTimeStamp)

(op, args) = parser.parse_args()
runLogFile = os.path.join(op.tmpOutDir, op.timeStamp + ".log")

