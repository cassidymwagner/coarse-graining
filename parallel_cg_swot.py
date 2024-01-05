import collections
import glob
import pickle
import multiprocessing
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
from scipy import interpolate
from astropy import units as u
import seaborn as sns
import warnings
from swot_analysis import geostrophic_velocity
from swot_analysis import rotational_parameters


def gauss_sfs(params):

    warnings.filterwarnings("ignore")

    l = params[0]
    # s1 = params[0]
    # s2 = params[1]
    filename = params[1]

    # For Pacific
    latmin = 40
    latmax = 50
    lonmin = 0
    lonmax = 360

    # For ACC
    # latmin = -70
    # latmax = -50
    # lonmin = 0
    # lonmax = 360

    dx = 2000
    dy = 2000

    ds = xr.open_dataset(filename)

    ds.simulated_true_ssh_karin.attrs["units"] = "cm"
    ds_cut = ds.where(
        (ds.latitude > latmin)
        & (ds.latitude < latmax)
        & (ds.longitude < lonmax)
        & (ds.longitude > lonmin),
        drop=True,
    )

    ds.close()

    ds_vels = geostrophic_velocity(ds_cut, dx=dx, dy=dy)
    ds_cut.close()

    ds_rot = rotational_parameters(ds_vels, dx=dx, dy=dy)
    ds_vels.close()

    uvals = ((ds_rot.u.values * (u.cm / u.s)).si).value
    vvals = ((ds_rot.v.values * (u.cm / u.s)).si).value
    wvals = ((ds_rot.w.values * (1 / u.s)).si).value

    ds_rot.close()

    # xtmp = np.arange(0, uvals.shape[1])
    # ytmp = np.arange(0, uvals.shape[0])
    # masked_u = np.ma.masked_invalid(uvals)
    # masked_v = np.ma.masked_invalid(vvals)
    # masked_w = np.ma.masked_invalid(wvals)
    # xxtmp, yytmp = np.meshgrid(xtmp, ytmp)

    # x1tmp_u = xxtmp[~masked_u.mask]
    # y1tmp_u = yytmp[~masked_u.mask]
    # newarr_u = masked_u[~masked_u.mask]
    # uvals = interpolate.griddata((x1tmp_u, y1tmp_u), newarr_u.ravel(),
    #                              (xxtmp, yytmp),
    #                              method='cubic')

    # x1tmp_v = xxtmp[~masked_v.mask]
    # y1tmp_v = yytmp[~masked_v.mask]
    # newarr_v = masked_v[~masked_v.mask]
    # vvals = interpolate.griddata((x1tmp_v, y1tmp_v), newarr_v.ravel(),
    #                              (xxtmp, yytmp),
    #                              method='cubic')

    # x1tmp_w = xxtmp[~masked_w.mask]
    # y1tmp_w = yytmp[~masked_w.mask]
    # newarr_w = masked_w[~masked_w.mask]
    # wvals = interpolate.griddata((x1tmp_w, y1tmp_w), newarr_w.ravel(),
    #                              (xxtmp, yytmp),
    #                              method='cubic')

    # top_nanmask_u = ~np.any(np.isnan(uvals), axis=1)
    # top_nanmask_v = ~np.any(np.isnan(vvals), axis=1)
    # top_nanmask_w = ~np.any(np.isnan(wvals), axis=1)

    # mask_ = min([top_nanmask_u, top_nanmask_v, top_nanmask_w], key=len)

    # uvals = uvals[(top_nanmask_u & top_nanmask_v & top_nanmask_w)]
    # vvals = vvals[(top_nanmask_u & top_nanmask_v & top_nanmask_w)]
    # wvals = wvals[(top_nanmask_u & top_nanmask_v & top_nanmask_w)]

    ut = [uvals, vvals]

    ul = gaussian_filter(ut[0], sigma=l)
    vl = gaussian_filter(ut[1], sigma=l)

    # ul = uniform_filter(ut[0], size=(s1, s2))
    # vl = uniform_filter(ut[1], size=(s1, s2))

    # For energy flux
    uul = gaussian_filter(ut[0] * ut[0], sigma=l)
    vvl = gaussian_filter(ut[1] * ut[1], sigma=l)
    uvl = gaussian_filter(ut[0] * ut[1], sigma=l)

    # uul = uniform_filter(ut[0] * ut[0], size=(s1, s2))
    # vvl = uniform_filter(ut[1] * ut[1], size=(s1, s2))
    # uvl = uniform_filter(ut[0] * ut[1], size=(s1, s2))

    tau_uu = uul - (ul * ul)
    tau_uv = uvl - (ul * vl)
    tau_vv = vvl - (vl * vl)

    duldx, duldy = np.gradient(ul, dx, dy, axis=(1, 0))
    dvldx, dvldy = np.gradient(vl, dx, dy, axis=(1, 0))

    SFS_energy_flux = - np.nanmean(tau_uu * duldx + tau_uv *
                                   (dvldx + duldy) + tau_vv * dvldy)

    # For enstrophy flux
    wl = gaussian_filter(wvals, sigma=l)
    # wl = uniform_filter(wvals, size=(s1, s2))

    uwl = gaussian_filter(ut[0] * wvals, sigma=l)
    vwl = gaussian_filter(ut[1] * wvals, sigma=l)
    # uwl = uniform_filter(ut[0] * wvals, size=(s1, s2))
    # vwl = uniform_filter(ut[1] * wvals, size=(s1, s2))

    tau_uw = uwl - (ul * wl)
    tau_vw = vwl - (vl * wl)

    dwldx, dwldy = np.gradient(wl, dx, dy, axis=(1, 0))

    SFS_enstrophy_flux = - np.nanmean(dwldx * tau_uw + dwldy * tau_vw)

    df = pd.DataFrame({'SFS_energy_flux': SFS_energy_flux, 'SFS_enstrophy_flux': SFS_enstrophy_flux, 'filename': os.path.basename(filename),
                      'sigma': l}, index=[0])

    # df = pd.DataFrame({'SFS_energy_flux': SFS_energy_flux, 'SFS_enstrophy_flux': SFS_enstrophy_flux, 'filename': os.path.basename(filename),
    #                   's1': s1, 's2': s2}, index=[0])

    return (df)


if __name__ == '__main__':

    num_its = int(1e2)
    chunk = 2
    # filepath = '/Users/cassswagner/Library/CloudStorage/OneDrive-OregonStateUniversity/oceans-research/swot_sf_analysis/data/acc_data/*.nc'
    # filepath = '/Users/cassswagner/Library/CloudStorage/OneDrive-OregonStateUniversity/oceans-research/swot_sf_analysis/data/acc_data/SWOT_L2_LR_SSH_Expert_001_1*.nc'
    # filepath = '/Users/cassswagner/Library/CloudStorage/OneDrive-OregonStateUniversity/oceans-research/swot_sf_analysis/data/pacific_data/*.nc'
    filepath = '/Users/cassswagner/Library/CloudStorage/OneDrive-OregonStateUniversity/oceans-research/swot_sf_analysis/data/pacific_data/SWOT_L2_LR_SSH_Expert_001_1*.nc'

    sigma_all = np.logspace(np.log10(1e-3), np.log10(1e2), num_its)
    # size1_all = np.logspace(np.log10(1), np.log10(3e3), num_its)
    # size2_all = np.logspace(np.log10(1), np.log10(3e3), num_its)
    filenames = [filename for filename in glob.glob(filepath)]

    paramlist = list(itertools.product(sigma_all, filenames))

    with multiprocessing.Pool() as p:
        sfs_fluxes_multi = [p.map(gauss_sfs, paramlist, chunksize=chunk)]

    with open('sfs_pickles/sfs_flux_pacific_swot_data%s.pickle' %
              (datetime.today().strftime('%Y-%m-%d_%H%M%S')), 'wb') as handle:
        pickle.dump(sfs_fluxes_multi[0], handle,
                    protocol=pickle.HIGHEST_PROTOCOL)
