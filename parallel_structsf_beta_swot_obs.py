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

import swot_ssh_utils as swot


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
    lonmin = 148
    lonmax = 158

    ds = xr.open_dataset(filename)

    try:
        tmp_cut = ds.where(
            (ds.latitude > latmin)
            & (ds.latitude < latmax)
            & (ds.longitude < lonmax)
            & (ds.longitude > lonmin),
            drop=True,
        )
        tmp_cut.close()

        dx = 2000
        dy = 2000

        data = swot.SSH_L2()
        data.load_data(filename)

        ssha = data.Expert.ssha_karin_2
        flag = data.Expert.ancillary_surface_classification_flag
        ssha = np.where(flag == 0, ssha, np.nan)
        lon_1 = data.Expert.longitude.values
        lat_1 = data.Expert.latitude.values
        distance = data.Expert.cross_track_distance.values

        del data

        ds = xr.open_dataset(filename)

        # Bias correction (optional)
        ssha_1 = swot.fit_bias(
            ssha, distance,
            check_bad_point_threshold=0.1,
            remove_along_track_polynomial=True
        )

        # mask out data in nadir and outside of 60km swath width
        distance = np.nanmean(distance, axis=0)
        msk = (np.abs(distance) < 60e3) & (np.abs(distance) > 10e3)
        lon_1[:, ~msk] = np.nan
        lat_1[:, ~msk] = np.nan
        ssha_1[:, ~msk] = np.nan

        ds['ssha_bias_corrected'] = ds.ssh_karin_2
        ds.ssha_bias_corrected.data = ssha_1

        ds_cut = ds.where(
            (ds.latitude > latmin)
            & (ds.latitude < latmax)
            & (ds.longitude < lonmax)
            & (ds.longitude > lonmin),
            drop=True,
        )
        ds.close()

        omega = 7.2921e-5 / u.s
        detadx, detady = np.gradient(
            ds_cut.ssha_bias_corrected, dx, dy, axis=(0, 1))

        ug = -(c.g0.si / (2 * omega *
                          np.sin(ds_cut.latitude.data * np.pi / 180))) * detady
        vg = (c.g0.si / (2 * omega * np.sin(ds_cut.latitude.data * np.pi / 180))) * detadx

        del detadx, detady

        dudx, dudy = np.gradient(ug.value, dx, dy, axis=(0, 1))
        dvdx, dvdy = np.gradient(vg.value, dx, dy, axis=(0, 1))

        wg = dvdx - dudy

        del dudx, dudy, dvdx, dvdy

        x = dx * ds_cut.num_lines.values
        y = dy * ds_cut.num_pixels.values

        ds_cut.close()

        sfs_v = ocsf.advection_velocity(
            ug.value, vg.value, x, y, even=False, boundary=None, grid_type=None, nbins=len(y))
        sfs_zeta = ocsf.advection_scalar(
            wg, ug.value, vg.value, x, y, even=False, boundary=None, nbins=len(y))

        del ug, vg, wg

        df = pd.DataFrame(
            {'SF_velocity': sfs_v,
                'SF_vorticity': sfs_zeta,
                'filename': os.path.basename(filename)})

        return (df)

    except ValueError:
        ds.close()
        return (None)


if __name__ == '__main__':

    chunk = 2
    # filepath = '/Volumes/Promise Disk/data/beta_prevalidated_swot/20210329-20230710/SWOT_L2_LR_SSH_1.0/SWOT_L2_LR_SSH_Expert_005_*.nc'
    filepath = '/Volumes/Promise Disk/data/beta_prevalidated_swot/SWOT_L2_LR_SSH_1.0_acc_cut/*Expert*.nc'

    filenames = [filename for filename in glob.glob(filepath)]

    with mp.Pool(processes=(mp.cpu_count() - 1)) as p:
        sfs_adv_multi = [p.map(advection_sf, filenames, chunksize=chunk)]

    with open('/Volumes/Promise Disk/data_analysis/beta_prevalidated_swot/acc_data/sfs_pickles/structsfs_acc_beta_swot_obs_data%s.pickle' %
              (datetime.today().strftime('%Y-%m-%d_%H%M%S')), 'wb') as handle:
        pickle.dump(sfs_adv_multi[0], handle,
                    protocol=pickle.HIGHEST_PROTOCOL)
