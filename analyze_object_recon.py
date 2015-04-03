import re
from matplotlib import pyplot as P
import sys
import numpy as N

#fn = sys.argv[1]

def parse_error(fn):
    fp = open(fn, "r")
    lines = fp.readlines()
    fp.close()
    pa = re.compile("iter = (\d+)\s+error = (\d+.\d+)")
    err_list = []
    for l in lines:
        m = pa.search(l)
        if m is not None:
            err_list.append([float(m.group(1)), float(m.group(2))])
    return N.array(err_list)

def plot_error(err_list):
    fig, ax = P.subplots(1,2, figsize=(5,5))
    ax[0].plot(err_list[:,0], err_list[:,1])
    ax[1].plot(err_list[:,0], N.log(err_list[:,1]))
    P.show()
