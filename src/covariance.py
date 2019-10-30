import numpy
import sys
from matplotlib import pyplot
from scipy.constants import c
from scipy import signal

from .radiotelescope import beam_width
from .radiotelescope import mwa_dipole_locations
sys.path.append("../../beam_perturbations/code/tile_beam_perturbations/")

from analytic_covariance import moment_returner


def position_covariance(nu, u, v, delta_u, gamma = 0.8, mode = "frequency", nu_0 = 150e6):
    mu_1 = moment_returner(n_order = 1)
    mu_2 = moment_returner(n_order = 2)

    if mode == "frequency":
        nn1, nn2 = numpy.meshgrid(nu, nu)
        vv1 = v
        vv2 = v
        uu1 = u
        uu2 = u
    else:
        nn1 = nu
        nn2 = nu
        vv1, vv2 = numpy.meshgrid(v, v)
        uu1, uu2 = numpy.meshgrid(u, u)

    beamwidth1 = beam_width(nn1)
    beamwidth2 = beam_width(nn2)

    sigma_nu = beamwidth1**2*beamwidth2**2/(beamwidth1**2 + beamwidth2**2)
    kernel = -2*numpy.pi**2*sigma_nu*((uu1*nn1 - uu2*nn2)**2 + (vv1*nn1 - vv2*nn2)**2 )/nu_0**2
    a = (2*numpy.pi)**5*mu_2*(nn1*nn2/nu_0**2)**(-gamma)*delta_u**2*sigma_nu*numpy.exp(kernel)*(2+2*kernel)
    #b = mu_1**2*(nn1*nn2)**(-gamma)*delta_u**2
    covariance = a
    return covariance


def beam_covariance(nu, u, v, dx = 1.1, gamma= 0.8, mode = 'frequency'):
    x_offsets, y_offsets = mwa_dipole_locations(dx)
    mu_2 = moment_returner(n_order = 2)

    if mode == "frequency":
        nn1, nn2, xx = numpy.meshgrid(nu, nu, x_offsets )
        nn1, nn2, yy = numpy.meshgrid(nu, nu, y_offsets)
        vv1 = v
        vv2 = v
        uu1 = u
        uu2 = u
    else:
        nn1 = nu
        nn2 = nu
        vv1, vv2, yy = numpy.meshgrid(v, v, y_offsets)
        uu1, uu2, xx = numpy.meshgrid(u, u, x_offsets)

    width_1_tile = numpy.sqrt(2) * beam_width(nn1)
    width_2_tile = numpy.sqrt(2) * beam_width(nn2)
    width_1_dipole = numpy.sqrt(2) * beam_width(nn1, diameter=1)
    width_2_dipole = numpy.sqrt(2) * beam_width(nn2, diameter=1)

    sigma_a = (width_1_tile * width_2_tile * width_1_dipole * width_2_dipole) ** 2 / (
            width_2_tile ** 2 * width_1_dipole ** 2 * width_2_dipole ** 2 +
            width_1_tile ** 2 * width_1_dipole ** 2 * width_2_dipole ** 2 +
            width_1_tile ** 2 * width_2_tile ** 2 * width_1_dipole ** 2 +
            width_1_tile ** 2 * width_2_tile ** 2 * width_2_dipole ** 2)

    a = 2*numpy.pi*mu_2*(nn1*nn2)**(-gamma) / len(y_offsets) ** 3 * numpy.sum(
        sigma_a * numpy.exp(-2 * numpy.pi ** 2 * sigma_a * ((uu1*nn1 - uu2*nn2 + xx*(nn1 - nn2) / c) ** 2 +
                                                            (vv1*nn1 - vv2*nn2 + yy*(nn1 - nn2) / c) ** 2)), axis=-1)
    b = 1

    covariance = a

    return covariance


def sky_covariance(u, v, nu, S_low=0.1, S_mid=1, S_high=1, gamma = 0.8, mode = 'frequency'):
    mu_2 = moment_returner(2, S_low=S_low, S_mid=S_mid, S_high=S_high)

    if mode == "frequency":
        nn1, nn2 = numpy.meshgrid(nu, nu)
        uu1 = u
        uu2 = u
        vv1 = v
        vv2 = v
    else:
        nn1 = nu
        nn2 = nu
        uu1, uu2 = numpy.meshgrid(u, u)
        vv1, vv2 = numpy.meshgrid(v, v)

    width_tile1 = beam_width(nn1)
    width_tile2 = beam_width(nn2)
    sigma_nu = width_tile1**2*width_tile2**2/(width_tile1**2 + width_tile2**2)

    covariance = 2 * numpy.pi * mu_2* sigma_nu * numpy.exp(-numpy.pi ** 2 * sigma_nu *
                                                           ((uu1*nn1 - uu2*nn2) ** 2 + (vv1*nn1 - vv2*nn2) ** 2)/nu[0])

    return covariance


def dft_matrix(nu):
    dft = numpy.exp(-2 * numpy.pi * 1j / len(nu)) ** numpy.arange(0, len(nu), 1)
    dftmatrix = numpy.vander(dft, increasing=True) / numpy.sqrt(len(nu))

    eta = numpy.arange(0, len(nu), 1) / (nu.max() - nu.min())
    return dftmatrix, eta


def blackman_harris_taper(frequency_range):
    window = signal.blackmanharris(len(frequency_range))
    return window


def compute_ps_variance(taper1, taper2, covariance, dft_matrix):
    tapered_cov = covariance * taper1 * taper2
    eta_cov = numpy.dot(numpy.dot(dft_matrix.conj().T, tapered_cov), dft_matrix)
    variance = numpy.diag(numpy.real(eta_cov))

    return variance