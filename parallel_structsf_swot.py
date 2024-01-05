import collections
import glob
import pickle
import multiprocessing
import torch.multiprocessing as mp
import torch
import os
import itertools
import re
import sys
import time
import traceback
import warnings
from datetime import datetime

import h5py
import matplotlib as mpl
import matplotlib_inline.backend_inline
import numpy as np
import xarray as xr
import pandas as pd
import scipy
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter
from astropy import units as u
import seaborn as sns
import oceans_sf as ocsf
import warnings
from swot_analysis import geostrophic_velocity
from swot_analysis import rotational_parameters


@torch.compile
def advection_sf(filename):

    warnings.filterwarnings("ignore")

    # For Pacific
    # latmin = 40
    # latmax = 50
    # lonmin = 0
    # lonmax = 360

    # For ACC
    # latmin = -70
    # latmax = -50
    # lonmin = 0
    # lonmax = 360

    # For Southern Ocean
    latmin = -57.5
    latmax = -53
    lonmin = 0
    lonmax = 360

    dx = 2000
    dy = 2000

    ds = xr.open_dataset(filename)

    ds_cut = ds.where(
        (ds.latitude > latmin)
        & (ds.latitude < latmax)
        & (ds.longitude < lonmax)
        & (ds.longitude > lonmin),
        drop=True,
    )

    ds.close()

    ds_vels = geostrophic_velocity(ds_cut, dx=dx, dy=dy, flip=False)
    ds_cut.close()

    ds_rot = rotational_parameters(ds_vels, dx=dx, dy=dy)
    ds_vels.close()

    uvals = ds_rot.u.values
    vvals = ds_rot.v.values
    wvals = ds_rot.w.values

    x = dx * ds_rot.num_lines.values
    y = dy * ds_rot.num_pixels.values

    ds_rot.close()

    sfs_v = ocsf.advection_velocity(
        uvals, vvals, x, y, even=False, boundary=None, grid_type=None, nbins=len(y))
    sfs_zeta = ocsf.advection_scalar(
        wvals, uvals, vvals, x, y, even=False, boundary=None, grid_type=None, nbins=len(y))

    df = pd.DataFrame(
        {'SF_velocity': sfs_v, 'SF_vorticity': sfs_zeta, 'filename': os.path.basename(filename)})

    return (df)


if __name__ == '__main__':

    chunk = 2
    # filepath = '/Users/cassswagner/Library/CloudStorage/OneDrive-OregonStateUniversity/oceans-research/swot_sf_analysis/data/pacific_data/SWOT_L2_LR_SSH_Expert_001_1*.nc'
    filepath = '/Volumes/Promise Disk/data/simulated_swot/acc_data/*.nc'

    filenames = [filename for filename in glob.glob(filepath)]

    with mp.Pool(processes=(mp.cpu_count() - 1)) as p:
        sfs_adv_multi = [p.map(advection_sf, filenames, chunksize=chunk)]

    with open('/Volumes/Promise Disk/data_analysis/simulated_swot/acc_data/sfs_pickles/structsfs_acc_simswot_data%s.pickle' %
              (datetime.today().strftime('%Y-%m-%d_%H%M%S')), 'wb') as handle:
        pickle.dump(sfs_adv_multi[0], handle,
                    protocol=pickle.HIGHEST_PROTOCOL)
