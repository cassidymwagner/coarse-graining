import collections
import glob
import pickle
import multiprocess
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
import pandas as pd
import scipy
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter
import seaborn as sns


def gauss_sfs(params):

    l = params[0]
    filename = params[1]
    time = float(re.search('betax_0.0_(.+?).mat', filename).group(1))

    f = h5py.File(filename, "r")
    u_tensor = np.array([f['u'][()], f['v'][()]])
    w = np.array(f['zeta'][()])

    dx = f['dx'][0]
    dy = f['dy'][0]

    f.close()

    ut = u_tensor

    ul = gaussian_filter(ut[0], sigma=l, truncate=3)
    vl = gaussian_filter(ut[1], sigma=l, truncate=2)

    # For energy flux
    uul = gaussian_filter(ut[0] * ut[0], sigma=l, truncate=1)
    vvl = gaussian_filter(ut[1] * ut[1], sigma=l)

    uvl = gaussian_filter(ut[0] * ut[1], sigma=l)

    tau_uu = uul - (ul * ul)
    tau_uv = uvl - (ul * vl)
    tau_vv = vvl - (vl * vl)

    duldx, duldy = np.gradient(ul, dx, dy, axis=(1, 0))
    dvldx, dvldy = np.gradient(vl, dx, dy, axis=(1, 0))

    SFS_energy_flux = - np.mean(tau_uu * duldx + tau_uv *
                                (dvldx + duldy) + tau_vv * dvldy)

    # For enstrophy flux
    wl = gaussian_filter(w, sigma=l)

    uwl = gaussian_filter(ut[0] * w, sigma=l)
    vwl = gaussian_filter(ut[1] * w, sigma=l)

    tau_uw = uwl - (ul * wl)
    tau_vw = vwl - (vl * wl)

    dwldx, dwldy = np.gradient(wl, dx, dy, axis=(1, 0))

    SFS_enstrophy_flux = - np.nanmean(dwldx * tau_uw + dwldy * tau_vw)

    df = pd.DataFrame({'SFS_energy_flux': SFS_energy_flux, 'SFS_enstrophy_flux': SFS_enstrophy_flux, 'time': time,
                      'sigma': l}, index=[int(time)])

    return (df)


if __name__ == '__main__':

    num_its = int(1e2)
    chunk = 2
    filepath = '/Users/cassswagner/Library/CloudStorage/Box-Box/2D_Data_For_Cassidy/Aniso*.mat'

    sigma_all = np.logspace(np.log10(1e-3), np.log10(1e2), num_its)
    filenames = [filename for filename in glob.glob(filepath) if float(
        re.search('betay_(.+?)_betax', filename).group(1)) == 10.0]

    paramlist = list(itertools.product(sigma_all, filenames))

    with multiprocess.Pool() as p:
        sfs_fluxes_multi = [p.map(gauss_sfs, paramlist, chunksize=chunk)]

    with open('sfs_pickles/sfs_flux_all_data%s.pickle' %
              (datetime.today().strftime('%Y-%m-%d_%H%M%S')), 'wb') as handle:
        pickle.dump(sfs_fluxes_multi[0], handle,
                    protocol=pickle.HIGHEST_PROTOCOL)
