import h5py
import os
import numpy as N
import pylab as P
import matplotlib as M

def create_interval_labels(l, n):
    sep         = l / (n - 1.)
    intervals   = [1] + list(N.arange(sep+1, l-1, sep).astype(int)) + [l]
    return intervals

def extract_arr_from_h5(fn, tag, n=9):
    fp          = h5py.File(fn, 'r')
    keys        = fp[tag].keys()
    if (len(keys) < n) or n ==-1:
        num_keys    = (len(keys),)
        vals        = [fp[tag+"/"+k] for k in keys]
        size_arr    = (vals[0].value).shape
        temp        = N.zeros(num_keys + size_arr)
        for k,v in zip(keys, vals):
            loc         = int(k) - 1
            temp[loc]   = v.value
    else:
        intervals   = create_interval_labels(len(keys), n)
        c_vals      = [fp[tag+"/"+"%0.4d"%ll] for ll in intervals]
        temp        = []
        for v in c_vals:
            temp.append(v.value)
    fp.close()
    return N.array(temp)

def extract_final_arr_from_h5(fn, tag):
    fp          = h5py.File(fn, 'r')
    keys        = fp[tag].keys()
    final_key   = tag + ("/%0.4d"%len(keys))
    temp        = fp[final_key].value
    return temp
