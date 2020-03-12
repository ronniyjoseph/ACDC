import numpy
from scipy.constants import c
from scipy import signal

from .radiotelescope import beam_width
from .radiotelescope import mwa_dipole_locations

from .skymodel import sky_moment_returner


def position_covariance(nu, u, v, position_precision = 1e-2, gamma = 0.8, mode = "frequency", nu_0 = 150e6,
                        tile_diameter = 4, s_high = 10):
    mu_1 = sky_moment_returner(n_order = 1, s_high= s_high)
    mu_2 = sky_moment_returner(n_order = 2, s_high= s_high)

    if mode == "frequency":
        nn1, nn2 = numpy.meshgrid(nu, nu)
        vv1 = v
        vv2 = v
        uu1 = u
        uu2 = u
        delta_u = position_precision*nu[0] / c
    else:
        nn1 = nu
        nn2 = nu
        vv1, vv2 = numpy.meshgrid(v, v)
        uu1, uu2 = numpy.meshgrid(u, u)
        delta_u = position_precision*nu / c

    beamwidth1 = beam_width(nn1, diameter=tile_diameter)
    beamwidth2 = beam_width(nn2, diameter=tile_diameter)

    sigma = beamwidth1**2*beamwidth2**2/(beamwidth1**2 + beamwidth2**2)

    kernel = -2*numpy.pi**2*sigma*((uu1*nn1 - uu2*nn2)**2 + (vv1*nn1 - vv2*nn2)**2 )/nu_0**2
    a = 16*numpy.pi**3*mu_2*(nn1*nn2/nu_0**2)**(1-gamma)*delta_u**2*sigma*numpy.exp(kernel)*(1+2*kernel)
    b = mu_1**2*(nn1*nn2)**(-gamma)*delta_u**2
    covariance = a
    return covariance


def beam_covariance(nu, u, v, dx = 1.1, gamma= 0.8, mode = 'frequency', broken_tile_fraction = 1.0, nu_0 = 150e6):
    x_offsets, y_offsets = mwa_dipole_locations(dx)
    mu_2 = sky_moment_returner(n_order = 2)
    if mode == "frequency":
        nn1, nn2, xx = numpy.meshgrid(nu, nu, x_offsets)
        nn1, nn2, yy = numpy.meshgrid(nu, nu, y_offsets)
        vv1 = v
        vv2 = v
        uu1 = u
        uu2 = u
        frequency_scaling = nn1[..., 0]*nn2[..., 0]/nu_0**2
    else:
        nn1 = nu
        nn2 = nu
        vv1, vv2, yy = numpy.meshgrid(v, v, y_offsets)
        uu1, uu2, xx = numpy.meshgrid(u, u, x_offsets)
        frequency_scaling = nu**2/nu_0**2

    width_1_tile = numpy.sqrt(2) * beam_width(frequency=nn1)
    width_2_tile = numpy.sqrt(2) * beam_width(frequency=nn2)
    width_1_dipole = numpy.sqrt(2) * beam_width(frequency=nn1, diameter=1)
    width_2_dipole = numpy.sqrt(2) * beam_width(frequency=nn2, diameter=1)

    sigma_a = (width_1_tile * width_2_tile * width_1_dipole * width_2_dipole) ** 2 / (
            width_2_tile ** 2 * width_1_dipole ** 2 * width_2_dipole ** 2 +
            width_1_tile ** 2 * width_1_dipole ** 2 * width_2_dipole ** 2 +
            width_1_tile ** 2 * width_2_tile ** 2 * width_1_dipole ** 2 +
            width_1_tile ** 2 * width_2_tile ** 2 * width_2_dipole ** 2)

    kernel = -2 * numpy.pi ** 2 * sigma_a * ((uu1*nn1 - uu2*nn2 + xx*(nn1 - nn2) / c) ** 2 +
                                             (vv1*nn1 - vv2*nn2 + yy*(nn1 - nn2) / c) ** 2)/nu_0**2
    a = 2*numpy.pi*mu_2*frequency_scaling**(-gamma) / len(y_offsets) ** 3 * numpy.sum(sigma_a * numpy.exp(kernel),
                                                                                        axis=-1)
    b = 1

    covariance = a

    return broken_tile_fraction*covariance


def sky_covariance(nu, u, v, S_low=1e-5, S_mid=1, S_high=1, gamma=0.8, mode = 'frequency', nu_0 = 150e6):
    mu_2 = sky_moment_returner(2, s_low=S_low, s_mid=S_mid, s_high=S_high)
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

    kernel = -2*numpy.pi ** 2 * sigma_nu * ((uu1*nn1 - uu2*nn2) ** 2 + (vv1*nn1 - vv2*nn2) ** 2)/nu_0**2
    covariance = 2 * numpy.pi * mu_2 * sigma_nu * (nn1*nn2/nu_0**2)**(-gamma)*numpy.exp(kernel)

    return covariance


def thermal_variance(sefd=20e3, bandwidth=40e3, t_integrate=120):
    variance = (sefd / numpy.sqrt(bandwidth * t_integrate))**2

    return variance
