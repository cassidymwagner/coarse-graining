import numpy as np
from astropy import units as u
from astropy import constants as c
from scipy import signal
from scipy.ndimage import gaussian_filter


def tophat_kernel(r, x, y, l, A):
    """
    r: length vector
    l: separation distance (one value)
    A: surface area array dependent on r and l
    """

    Hl = np.zeros(np.shape(r)) / u.m**2

    r_mag = np.sqrt(np.einsum('i,j', x, y))

    Hl_new = np.where(r_mag < (l/2), 1/A, Hl)

    return (Hl_new.value)


def gaussian_kernel(x, y):

    Gl = np.exp(-(x**2+y**2) / 2) / (np.sqrt(2*np.pi))

    return (Gl)


def surface_area_earth(l):
    """
    l: separation distance (one value)
    """

    l = l.si

    A = 2 * np.pi * c.R_earth.si**2 * \
        (1 - np.cos((l / (2 * c.R_earth.si)).value))
    return (A)


def surface_area(l):
    l = l.si
    A = np.pi * l**2 / 4
    return (A)


def coarse_grain(f, Gl):
    """
    f: vector you want to create a coarse-grained version of
    Gl: kernel for convolving
    """

    fl = signal.convolve(f, Gl, mode='same')

    return (fl)


def SFS_flux(ul, uul, dx, dy, rho0):
    """
    ul: coarse-grained u
    uul: coarse-grained uu
    rho0: density (of water)
    """

    duldx, duldy = np.gradient(ul, dx, dy, axis=(1, 0))
    dutdx, dutdy = np.gradient(ul.T, dx, dy, axis=(1, 0))

    Sl = 0.5 * (duldy + dutdx)

    taul = uul - (ul @ ul)

    SFS_flux = -rho0 * np.einsum('ij,ij', Sl, taul)

    return (SFS_flux)


def coarse_grain_flux_method(u_, l, x, y, xx, yy, dx, dy, rho0, kernel='tophat'):

    A = surface_area_earth(l*u.m)

    if kernel == 'tophat':
        kern = tophat_kernel(xx, x, y, l, A)

    else:
        kern = gaussian_kernel(xx, yy)
    ul = coarse_grain(u_, kern)
    uul = coarse_grain(u_ @ u_, kern)
    sfs_flux = SFS_flux(ul, uul, dx, dy, rho0)

    sfs_dict = {'flux': sfs_flux, 'surface_area': A,
                'kernel': kern, 'u_l': ul, 'uu_l': uul}

    return (sfs_dict)


def gauss_sfs(u_tensor, l, dx, dy):

    ut = u_tensor

    ul = gaussian_filter(ut, sigma=l)
    uul = gaussian_filter(np.stack((ut, ut)), sigma=l)

    grad_ul = np.gradient(ul, dx, dy, axis=(1, 0))

    Sl = (grad_ul + np.transpose(grad_ul, (0, 1, 3, 2)))

    taul = uul - (np.stack((ul, ul)))
    # this is a tensor inner product that yields a scalar
    SFS_flux = - np.einsum('ijkl,ijkl', Sl, taul)

    return (SFS_flux)
