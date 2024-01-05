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
from astropy.convolution import convolve_fft, Tophat2DKernel
import seaborn as sns
import warnings
from swot_analysis import geostrophic_velocity
from swot_analysis import rotational_parameters


# class sfs:
#     pass


# @torch.compile
def sfs_func(params):

    warnings.filterwarnings("ignore")

    l = params[0]
    # s1 = params[0]
    # s2 = params[1]
    filename = params[1]

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

    ds_rot.close()

    ut = [uvals, vvals]

    tophat = Tophat2DKernel(l)

    ul = convolve_fft(ut[0], tophat)
    vl = convolve_fft(ut[1], tophat)

    # For energy flux
    uul = convolve_fft(ut[0] * ut[0], tophat)
    vvl = convolve_fft(ut[1] * ut[1], tophat)
    uvl = convolve_fft(ut[0] * ut[1], tophat)

    tau_uu = uul - (ul * ul)
    tau_uv = uvl - (ul * vl)
    tau_vv = vvl - (vl * vl)

    duldx, duldy = np.gradient(ul, dx, dy, axis=(0, 1))
    dvldx, dvldy = np.gradient(vl, dx, dy, axis=(0, 1))

    SFS_energy_flux = - np.nanmean(tau_uu * duldx + tau_uv *
                                   (dvldx + duldy) + tau_vv * dvldy)

    # For enstrophy flux
    wl = convolve_fft(wvals, tophat)

    uwl = convolve_fft(ut[0] * wvals, tophat)
    vwl = convolve_fft(ut[1] * wvals, tophat)

    tau_uw = uwl - (ul * wl)
    tau_vw = vwl - (vl * wl)

    dwldx, dwldy = np.gradient(wl, dx, dy, axis=(0, 1))

    SFS_enstrophy_flux = - np.nanmean(dwldx * tau_uw + dwldy * tau_vw)

    df = pd.DataFrame({'SFS_energy_flux': SFS_energy_flux, 'SFS_enstrophy_flux': SFS_enstrophy_flux, 'filename': os.path.basename(filename),
                      'l': l}, index=[0])

    # df = pd.DataFrame({'SFS_energy_flux': SFS_energy_flux, 'SFS_enstrophy_flux': SFS_enstrophy_flux, 'filename': os.path.basename(filename),
    #                   's1': s1, 's2': s2}, index=[0])

    return (df)


if __name__ == '__main__':

    num_its = int(1e1)
    chunk = 2
    # filepath = '/Users/cassswagner/Library/CloudStorage/OneDrive-OregonStateUniversity/oceans-research/swot_sf_analysis/data/acc_data/*.nc'
    # filepath = '/Users/cassswagner/Library/CloudStorage/OneDrive-OregonStateUniversity/oceans-research/swot_sf_analysis/data/acc_data/SWOT_L2_LR_SSH_Expert_001_1*.nc'
    # filepath = '/Users/cassswagner/Library/CloudStorage/OneDrive-OregonStateUniversity/oceans-research/swot_sf_analysis/data/pacific_data/*.nc'
    # filepath = '/Users/cassswagner/Library/CloudStorage/OneDrive-OregonStateUniversity/oceans-research/swot_sf_analysis/data/pacific_data/SWOT_L2_LR_SSH_Expert_001_1*.nc'
    filepath = '/Volumes/Promise Disk/data/simulated_swot/acc_data/*.nc'

    l_all = np.logspace(0, 3, num_its)
    filenames = [filename for filename in glob.glob(filepath)]

    paramlist = list(itertools.product(l_all, filenames))

    with mp.Pool(processes=(mp.cpu_count() - 1)) as p:
        sfs_fluxes_multi = [p.map(sfs_func, paramlist, chunksize=chunk)]

    # sfs_fluxes_multi = list(map(sfs_func, paramlist))

    with open('/Volumes/Promise Disk/data_analysis/simulated_swot/acc_data/sfs_pickles/cgfluxes_acc_simswot_data%s.pickle' %
              (datetime.today().strftime('%Y-%m-%d_%H%M%S')), 'wb') as handle:
        pickle.dump(sfs_fluxes_multi[0], handle,
                    protocol=pickle.HIGHEST_PROTOCOL)