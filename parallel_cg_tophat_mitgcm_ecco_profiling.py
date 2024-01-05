import os
import numpy as np
import xarray as xr
import pandas as pd
from astropy.convolution import convolve_fft, Tophat2DKernel
import glob
import itertools


def sfs_func(params):

    l = params[0]
    t = params[1]
    filename = params[2]

    ds = xr.open_dataset(filename)

    dx = 1
    dy = 1

    U = np.copy(ds.U.isel(time=t, k=0).values)
    V = np.copy(ds.V.isel(time=t, k=0).values)
    ds.close()

    tophat = Tophat2DKernel(l)

    ul = convolve_fft(U, tophat)
    vl = convolve_fft(V, tophat)
    uul = convolve_fft(U * U, tophat)
    vvl = convolve_fft(V * V, tophat)
    uvl = convolve_fft(U * V, tophat)

    tau_uu = uul - (ul * ul)
    tau_uv = uvl - (ul * vl)
    tau_vv = vvl - (vl * vl)
    duldx, duldy = np.gradient(ul, dx, dy, axis=(1, 0))
    dvldx, dvldy = np.gradient(vl, dx, dy, axis=(1, 0))
    SFS_energy_flux = - np.nanmean(tau_uu * duldx + tau_uv *
                                   (dvldx + duldy) + tau_vv * dvldy)

    df = pd.DataFrame({'SFS_energy_flux': SFS_energy_flux, 'filename': os.path.basename(filename), 'time': t,
                       'l': l}, index=[0])

    return (df)


num_its = int(1e1)
chunk = 2

filepath = '/Volumes/Promise Disk/data/mitgcm/acc_data/*.nc'
filename = [
    '/Volumes/Promise Disk/data/mitgcm/acc_data/LLC4320_pre-SWOT_ACC_SMST_20111113.nc']

filenames = [filename for filename in glob.glob(filepath)]
times = np.arange(0, 2)

l_all = np.linspace(1, 1e2, num_its)
filenames = [filename for filename in glob.glob(filepath)]

paramlist = list(itertools.product(l_all, times, filename))

sfs_fluxes_multi = list(map(sfs_func, paramlist))
