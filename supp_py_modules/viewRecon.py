import h5py
import os
import numpy as N
import pylab as P
import matplotlib as M
import read_results as read
from mpl_toolkits.axes_grid1 import make_axes_locatable

# This backend does not display plots to screen.

quatSize ={1:60, 2:420, 3:1380, 4:3240, 5:6300, 6:10860, 7:17220, 8:25680, 9:36540}

def make_panel_of_intensity_slices(fn, c_n=9):
    M.rcParams.update({'font.size': 13})
    intensList = read.extract_arr_from_h5(fn, "/history/intensities", n=c_n)
    quatList = read.extract_arr_from_h5(fn, "/history/quaternion", n=-1)
    P.ioff()
    intens_len  = len(intensList)
    sqrt_len    = int(N.sqrt(intens_len))
    intens_sh   = intensList[0].shape
    iter_labels = read.create_interval_labels(len(quatList), c_n)[:intens_len]
    to_plot     = intensList[:intens_len]
    quat_label  = quatList[N.array(iter_labels)-1][:intens_len]
    plot_titles = ["iter_%d, quat_%d"%(ii,jj) for ii,jj in zip(iter_labels, quat_label)]
    fig, ax     = P.subplots(sqrt_len, sqrt_len, sharex=True, sharey=True, figsize=(1.8*sqrt_len, 2.*sqrt_len))
    plt_counter = 0
    for r in range(sqrt_len):
        for c in range(sqrt_len):
            ax[r,c].set_title(plot_titles[plt_counter])
            curr_slice = to_plot[plt_counter][intens_sh[0]/2]
            curr_slice = curr_slice*(curr_slice>0.) + 1.E-8*(curr_slice<=0.)
            ax[r,c].set_title(plot_titles[plt_counter], fontsize=11.5)
            im = ax[r,c].imshow(N.log10(curr_slice), vmin=-6.5, vmax=-3.5, aspect='auto', cmap=P.cm.coolwarm)
            plt_counter += 1
    fig.subplots_adjust(wspace=0.01)
    (shx, shy) = curr_slice.shape
    (h_shx, h_shy) = (shx/2, shy/2)
    xt = N.linspace(0.5*h_shx, shx-.5*h_shx-1, 3).astype('int')
    xt_l = N.linspace(-0.5*h_shx, 0.5*h_shx, 3).astype('int')
    yt = N.linspace(0, shy-1, 3).astype('int')
    yt_l = N.linspace(-1*h_shy, h_shy, 3).astype('int')
    P.setp(ax, xticks=xt, xticklabels=xt_l, yticks=yt, yticklabels=yt_l)
    cbar_ax = fig.add_axes([0.9, 0.1, 0.025, 0.8])
    fig.colorbar(im, cax=cbar_ax, label="log10(intensities)")
    img_name = "recon_series.pdf"
    P.savefig(img_name, bbox_inches='tight')
    print("Image has been saved as %s" % img_name)
    P.close(fig)

def make_error_time_plot(fn):
    M.rcParams.update({'font.size': 12})
    errList = read.extract_arr_from_h5(fn, "/history/error", n=-1)
    timesList = read.extract_arr_from_h5(fn, "/history/time", n=-1)
    quatList = read.extract_arr_from_h5(fn, "/history/quaternion", n=-1)
    quatSwitchPos = N.where(quatList[:-1]-quatList[1:] != 0)[0] + 1
    P.ioff()
    fig, ax = P.subplots(2, 1, sharex=True, figsize=(6,6))
    fig.subplots_adjust(hspace=0.1)

    iters = range(1, len(errList)+1)
    ax[0].set_title("model change vs iterations")
    #ax[0].set_xlabel("iteration")
    ax[0].set_ylabel("log10(rms diffraction \nvolume change per voxel)")
    err_to_plot = N.log10(errList)
    ax[0].plot(iters, err_to_plot, 'k-')
    ax[0].plot(iters, err_to_plot, 'ko')
    (e_min, e_max) = (err_to_plot.min()-0.3, err_to_plot.max())
    e_int = 0.1*(e_max-e_min)
    ax[0].plot([1, 1], [e_min, e_max + e_int], 'k-')
    ax[0].text(2, e_max+e_int, "quat%d"%quatList[0], size=8, rotation=-0, ha='left', va='center', color='w', bbox=dict(boxstyle="larrow,pad=0.1",facecolor='0.1') )
    for n,qs in enumerate(quatSwitchPos):
        ax[0].plot([qs+1, qs+1], [e_min, e_max + e_int], 'k-')
        ax[0].text(qs, e_max+(1-n)*e_int, "quat%d"%quatList[qs], size=8, rotation=-0, ha='right', va='center', color='w', bbox=dict(boxstyle="rarrow,pad=0.1",facecolor='0.1') )

    ax[1].set_xlabel("iteration")
    ax[1].set_ylabel("time per iteration (s)")
    ax[1].plot(iters, timesList, 'k-')
    ax[1].plot(iters, timesList, 'ko')
    (t_min, t_max) = (timesList.min()-100, timesList.max())
    t_int = 0.1*(t_max-t_min)
    ax[1].plot([1, 1], [t_min, t_max + t_int], 'k-')
    ax[1].text(2, t_max+t_int, "quat%d"%quatList[0], size=8, rotation=-0, ha='left', va='center', color='w', bbox=dict(boxstyle="larrow,pad=0.1",facecolor='0.1') )
    for n,qs in enumerate(quatSwitchPos):
        ax[1].plot([qs+1, qs+1], [t_min, t_max+t_int], 'k-')
        ax[1].text(qs+0.5, t_min, "quat%d"%quatList[qs], size=8, rotation=45, ha='right', va='center', color='w', bbox=dict(boxstyle="rarrow,pad=0.1",facecolor='0.1'))
    img_name = "time_and_error_plot.pdf"
    P.savefig(img_name, bbox_inches='tight')
    print("Image has been saved as %s" % img_name)
    P.close(fig)

def make_mutual_info_plot(fn):
    M.rcParams.update({'font.size': 11})
    angleList = N.array([f/f.max() for f in read.extract_arr_from_h5(fn, "/history/angle", n=-1)])
    mutualInfoList = read.extract_arr_from_h5(fn, "/history/mutual_info", n=-1)
    quatList = read.extract_arr_from_h5(fn, "/history/quaternion", n=-1)
    quatSwitchPos = N.where(quatList[:-1]-quatList[1:] != 0)[0] + 1
    angsort = N.argsort(angleList[-1])
    misort = N.argsort(mutualInfoList.mean(axis=0))
    blkPositions = [0] + list(quatSwitchPos) + [-1]
    for bp in range(len(blkPositions)-1):
        (start, end) = (blkPositions[bp], blkPositions[bp+1])
        curr_blk = angleList[start:end]
        curr_blk2 = mutualInfoList[start:end] / N.log(quatSize[quatList[0]])
        # curr_blk2 = mutualInfoList[start:end] / N.log(quatSize[quatList[bp]])
        if len(curr_blk) == 0:
            pass
        else:
            angsort = N.argsort(curr_blk[-1])
            angleList[start:end] = curr_blk[:,angsort]
            for n,l in enumerate(curr_blk2):
                misort = N.argsort(l)
                mutualInfoList[start+n] = l[misort]

    P.ioff()
    fig, ax = P.subplots(2, 1, sharex=True, figsize=(7, 10))
    fig.subplots_adjust(hspace=0.1)
    im0 = ax[0].imshow(angleList.transpose(), aspect='auto', interpolation=None, cmap=P.cm.OrRd)
    ax[0].set_xlabel("iteration")
    ax[0].set_ylabel("each pattern's most likely orientation\n(sorted by final orientation in each block)")
    (e_min, e_max) = (1, len(angleList[0]))
    e_int = 0.1*(e_max-e_min)
    ax[0].plot([0, 0], [e_min, e_max], 'k-')
    ax[0].text(1, e_max-e_int, "quat%d"%quatList[0], size=8, rotation=-0, ha='left', va='center', color='w', bbox=dict(boxstyle="larrow,pad=0.1",facecolor='0.1') )
    for n,qs in enumerate(quatSwitchPos):
        ax[0].plot([qs, qs], [e_min, e_max], 'k-')
        ax[0].text(qs-1, e_max+(-n-1)*e_int, "quat%d"%quatList[qs], size=8, rotation=-0, ha='right', va='center', color='w', bbox=dict(boxstyle="rarrow,pad=0.1",facecolor='0.1') )
    div0 = make_axes_locatable(ax[0])
    cax0 = div0.append_axes("right", size="5%", pad=0.05)
    cbar0 = P.colorbar(im0, cax=cax0)
    ax[0].set_ylim(e_min, e_max)
    ax[0].set_xlim(0, len(angleList)-1)

    (e_min, e_max) = (1, len(mutualInfoList[0]))
    e_int = 0.1*(e_max-e_min)
    im1 = ax[1].imshow(mutualInfoList.transpose(), vmax=.2, aspect='auto', cmap=P.cm.YlGnBu)
    ax[1].set_xlabel("iteration")
    ax[1].set_ylabel("average mutual-information per dataset\n(sorted by average information)")
    ax[1].plot([0, 0], [e_min, e_max], 'k-')
    ax[1].text(1, e_max-e_int, "quat%d"%quatList[0], size=8, rotation=-0, ha='left', va='center', color='w', bbox=dict(boxstyle="larrow,pad=0.1",facecolor='0.1') )
    for n,qs in enumerate(quatSwitchPos):
        ax[1].plot([qs, qs], [e_min, e_max], 'k-')
        ax[1].text(qs-1, e_max+(-n-1)*e_int, "quat%d"%quatList[qs], size=8, rotation=-0, ha='right', va='center', color='w', bbox=dict(boxstyle="rarrow,pad=0.1",facecolor='0.1') )
    div1 = make_axes_locatable(ax[1])
    cax1 = div1.append_axes("right", size="5%", pad=0.05)
    cbar1 = P.colorbar(im1, cax=cax1)
    ax[1].set_ylim(e_min, e_max)
    ax[1].set_xlim(0, len(mutualInfoList)-1)
    img_name = "mutual_info_plot.pdf"
    P.savefig(img_name, bbox_inches='tight')
    print("Image has been saved as %s" % img_name)
    P.close(fig)
