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
from scipy.ndimage import gaussian_filter, uniform_filter
# from scipy.signal import convolve
from scipy import interpolate
from astropy import units as u
from astropy import constants as c
from astropy.convolution import convolve, convolve_fft, Tophat2DKernel
import seaborn as sns
import warnings
from swot_analysis import geostrophic_velocity
from swot_analysis import rotational_parameters
from pympler.tracker import SummaryTracker
from geopy.distance import great_circle

# @torch.compile


def sfs_func(params):

    # tracker = SummaryTracker()

    warnings.filterwarnings("ignore")

    l = params[0]
    t = params[1]
    filename = params[2]
    ds = xr.open_dataset(filename, engine='h5netcdf')

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
    U = -((c.g0.si / (2 * omega * np.sin(ds.YC * np.pi / 180))) * detady)
    V = ((c.g0.si / (2 * omega * np.sin(ds.YC * np.pi / 180))) * detadx)

    dudx, dudy = np.gradient(U, xd.cumsum(), yd.cumsum(), axis=(0, 1))
    dvdx, dvdy = np.gradient(V, xd.cumsum(), yd.cumsum(), axis=(0, 1))

    W = dvdx - dudy

    tophat = Tophat2DKernel(l)

    ul = convolve_fft(U, tophat)
    vl = convolve_fft(V, tophat)
    # For energy flux
    uul = convolve_fft(U * U, tophat)
    vvl = convolve_fft(V * V, tophat)
    uvl = convolve_fft(U * V, tophat)

    tau_uu = uul - (ul * ul)
    tau_uv = uvl - (ul * vl)
    tau_vv = vvl - (vl * vl)
    duldx, duldy = np.gradient(ul, xd.cumsum(), yd.cumsum(), axis=(0, 1))
    dvldx, dvldy = np.gradient(vl, xd.cumsum(), yd.cumsum(), axis=(0, 1))
    SFS_energy_flux = - np.nanmean(tau_uu * duldx + tau_uv *
                                   (dvldx + duldy) + tau_vv * dvldy)

    # For enstrophy flux
    wl = convolve(W, tophat)

    uwl = convolve(U * W, tophat)
    vwl = convolve(V * W, tophat)

    tau_uw = uwl - (ul * wl)
    tau_vw = vwl - (vl * wl)

    dwldx, dwldy = np.gradient(wl, xd.cumsum(), yd.cumsum(), axis=(0, 1))

    SFS_enstrophy_flux = - np.nanmean(dwldx * tau_uw + dwldy * tau_vw)

    df = pd.DataFrame({'SFS_energy_flux': SFS_energy_flux,
                       'SFS_enstrophy_flux': SFS_enstrophy_flux,
                       'filename': os.path.basename(filename),
                       'time': t,
                      'l': l}, index=[0])

    return (df)


if __name__ == '__main__':

    # tracker = SummaryTracker()

    num_its = int(1e1)
    chunk = 2

    filepath = '/Volumes/Promise Disk/data/mitgcm/acc_data/*.nc'
    # filename = [
    #     '/Volumes/Promise Disk/data/mitgcm/acc_data/LLC4320_pre-SWOT_ACC_SMST_20111113.nc']

    filenames = [filename for filename in glob.glob(filepath)]
    times = np.arange(0, 2)

    l_all = np.logspace(0, 3, num_its)

    # ds_list = [xr.load_dataset(filename).attrs.clear()
    #            for filename in glob.glob(filepath)]

    paramlist = list(itertools.product(l_all, times, filenames))

    with mp.Pool(processes=(mp.cpu_count() - 1)) as p:
        sfs_fluxes_multi = [p.map(sfs_func, paramlist, chunksize=chunk)]

    # sfs_fluxes_multi = list(map(sfs_func, paramlist))

    with open('/Volumes/Promise Disk/data_analysis/mitgcm/acc_data/sfs_pickles/cgfluxes_acc_mitgcm_data%s.pickle' %
              (datetime.today().strftime('%Y-%m-%d_%H%M%S')), 'wb') as handle:
        pickle.dump(sfs_fluxes_multi[0], handle,
                    protocol=pickle.HIGHEST_PROTOCOL)
    # tracker.print_diff()
