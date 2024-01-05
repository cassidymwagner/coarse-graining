import re
from datetime import datetime
from astropy import constants as c
from astropy import units as u
import warnings
from geopy import distance as gd
import matplotlib as mpl
import os
import matplotlib_inline.backend_inline
import seaborn as sns
import glob
import cartopy.crs as ccrs
import xarray as xr
import time
from matplotlib import pyplot as plt
import collections
import oceans_sf as ocsf
import pickle
import h5py
import numpy as np
import pandas as pd
import traceback
import scipy
from scipy import fft
from scipy import signal
from scipy.ndimage import gaussian_filter
from scipy import ndimage
from scipy.stats import bootstrap
from scipy.io import loadmat
import sys

sys.path.append(
    "/Users/cassswagner/Library/CloudStorage/OneDrive-OregonStateUniversity/oceans-research/analysis_scripts/"
)

# from calculate_spectral_fluxes import SpectralFlux
# from calculate_sfs import StructureFunctions
# from flux_sf_figures import *
# import flux_sf_figures
# import swot_analysis as swotan


sns.set_style(style="white")
sns.set_context("talk")

plt.rcParams["figure.figsize"] = [9, 6]
# plt.rcParams['figure.dpi'] = 100

matplotlib_inline.backend_inline.set_matplotlib_formats("retina")
# %config InlineBackend.figure_format = 'svg'


os.environ["PATH"] = os.environ["PATH"] + ":/Library/TeX/texbin"

mpl.rcParams["text.usetex"] = True
plt.rcParams["xtick.bottom"] = True
plt.rcParams["ytick.left"] = True
plt.rcParams["xtick.top"] = True
plt.rcParams["ytick.right"] = True


warnings.filterwarnings("ignore")


data_dir = '/Volumes/Promise Disk/data/mitgcm/acc_data/'
ds = xr.open_dataset(
    data_dir+'LLC4320_pre-SWOT_ACC_SMST_20111213.nc')

sfs_v = ocsf.advection_velocity(
    ds.U.isel(time=0, k=0).values, ds.V.isel(time=0, k=0).values, ds.i.values, ds.j.values, even=False, boundary=None, grid_type=None, nbins=len(ds.j.values))

print(sfs_v)
