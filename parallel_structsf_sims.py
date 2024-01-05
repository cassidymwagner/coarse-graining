import collections
import glob
import pickle
import multiprocessing
import torch.multiprocessing as mp
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


def advection_sf(filename):

    warnings.filterwarnings("ignore")

    betay = float(re.search('betay_(.+?)_betax', filename).group(1))
    time = float(re.search('betax_0.0_(.+?).mat', filename).group(1))

    f = h5py.File(filename, "r")
    timestep = float(re.search('_0.0_(.+?).mat', filename).group(1))

    x = np.linspace(-f['Lx'][0], f['Lx'][0], f['nx'][0])
    y = np.linspace(-f['Ly'][0], f['Ly'][0], f['ny'][0])

    sfs_v = ocsf.advection_velocity(f['u'], f['v'], x, y, boundary='Periodic')
    sfs_zeta = ocsf.advection_scalar(
        f['zeta'], f['u'], f['v'], x, y, boundary='Periodic')

    df = pd.DataFrame(
        {'SF_velocity': sfs_v, 'SF_vorticity': sfs_zeta, 'beta_y': betay, 't': timestep})

    return (df)


if __name__ == '__main__':

    chunk = 2
    filepath = '/Users/cassswagner/Library/CloudStorage/Box-Box/2D_Data_For_Cassidy/Aniso*.mat'

    filenames = [filename for filename in glob.glob(filepath) if float(
        re.search('betay_(.+?)_betax', filename).group(1)) == 10.0]

    with mp.Pool(processes=(mp.cpu_count() - 2)) as p:
        sfs_adv_multi = [p.map(advection_sf, filenames, chunksize=chunk)]

    with open('/Users/cassswagner/Library/CloudStorage/OneDrive-OregonStateUniversity/proposals/swot_dec23/data/structsfs_sims_data%s.pickle' %
              (datetime.today().strftime('%Y-%m-%d_%H%M%S')), 'wb') as handle:
        pickle.dump(sfs_adv_multi[0], handle,
                    protocol=pickle.HIGHEST_PROTOCOL)
