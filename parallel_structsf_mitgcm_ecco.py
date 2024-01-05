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
from astropy import constants as c
import seaborn as sns
import oceans_sf as ocsf
import warnings
from swot_analysis import geostrophic_velocity
from swot_analysis import rotational_parameters
from geopy.distance import great_circle


# @torch.compile
def advection_sf(params):

    filename = params[0]
    t = params[1]

    warnings.filterwarnings("ignore")

    ds = xr.open_dataset(filename)

    ssa = ds.Eta.isel(time=t).values

    xd = np.asarray([np.abs(great_circle((ds.YC[:, 0].values[i+1], ds.XC[:, 0].values[i+1]),
                    (ds.YC[:, 0].values[i], ds.XC[:, 0].values[i])).meters) for i in range(-1, len(ds.XC[:, 0].values)-1)])
    yd = np.asarray([np.abs(great_circle((ds.YC[0, :].values[i+1], ds.XC[0, :].values[i+1]),
                    (ds.YC[0, :].values[i], ds.XC[0, :].values[i])).meters) for i in range(-1, len(ds.YC[0, :].values)-1)])

    xd[0] = 0
    yd[0] = 0

    omega = 7.2921e-5 / u.s
    detadx, detady = np.gradient(
        ssa, xd.cumsum(), yd.cumsum(), axis=(0, 1))
    U = -(c.g0.si / (2 * omega * np.sin(ds.YC * np.pi / 180))) * detady
    V = (c.g0.si / (2 * omega * np.sin(ds.YC * np.pi / 180))) * detadx

    dudx, dudy = np.gradient(U, xd.cumsum(), yd.cumsum(), axis=(0, 1))
    dvdx, dvdy = np.gradient(V, xd.cumsum(), yd.cumsum(), axis=(0, 1))

    W = dvdx - dudy

    lons = ds.XC.values[0, :]
    lats = ds.YC.values[:, 0]

    sfs_v = ocsf.advection_velocity(
        U, V, lats, lons, xd, yd,
        even=False, boundary=None, grid_type='latlon', nbins=len(lons))

    sfs_zeta = ocsf.advection_scalar(
        W, U, V, lats, lons, xd, yd,
        even=False, boundary=None, grid_type='latlon', nbins=len(lons))

    df = pd.DataFrame(
        {'SF_velocity': sfs_v,
         'SF_vorticity': sfs_zeta,
         'filename': os.path.basename(filename),
         'time': t})

    return (df)


if __name__ == '__main__':

    chunk = 2
    filepath = '/Volumes/Promise Disk/data/mitgcm/acc_data/*.nc'
    # filename = [
    #     '/Volumes/Promise Disk/data/mitgcm/acc_data/LLC4320_pre-SWOT_ACC_SMST_20111113.nc']

    filenames = [filename for filename in glob.glob(filepath)]
    times = np.arange(0, 2)

    paramlist = list(itertools.product(filenames, times))

    with mp.Pool(processes=(mp.cpu_count() - 1)) as p:
        sfs_adv_multi = [p.map(advection_sf, paramlist, chunksize=chunk)]

    with open('/Volumes/Promise Disk/data_analysis/mitgcm/acc_data/sfs_pickles/structsfs_acc_mitgcm_data%s.pickle' %
              (datetime.today().strftime('%Y-%m-%d_%H%M%S')), 'wb') as handle:
        pickle.dump(sfs_adv_multi[0], handle,
                    protocol=pickle.HIGHEST_PROTOCOL)
