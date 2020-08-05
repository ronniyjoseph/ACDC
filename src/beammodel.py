from mwa_pb import config
from mwa_pb.beam_full_EE import ApertureArray
from mwa_pb.beam_full_EE import Beam
from pyrem.radiotelescope import ideal_gaussian_beam

import numpy as np
from scipy.constants import c

def mwa_fee_model(theta, phi, nu = 150e6):
    h5filepath = config.h5file  # recent version was MWA_embedded_element_pattern_V02.h5
    tile = ApertureArray(h5filepath, nu)
    my_Astro_Az = 0
    my_ZA = 0
    delays = np.zeros([2, 16])  # Dual-pol.
    amps = np.ones([2, 16])

    tile_beam = Beam(tile, delays, amps=amps)
    jones = tile_beam.get_response(phi, theta)
    power = jones[0, 0] * jones[0, 0].conjugate() + jones[0, 1] * jones[0, 1].conjugate()
    return power/power.max()


def simple_mwa_tile(l, m, frequency=150e6, weights=1, normalisation_only=False,
                    dipole_sep=1.1):
    # meters
    x_offsets = np.array([-1.5, -0.5, 0.5, 1.5, -1.5, -0.5, 0.5, 1.5, -1.5,
                             -0.5, 0.5, 1.5, -1.5, -0.5, 0.5, 1.5], dtype=np.float32) * dipole_sep
    y_offsets = np.array([1.5, 1.5, 1.5, 1.5, 0.5, 0.5, 0.5, 0.5, -0.5, -0.5,
                             -0.5, -0.5, -1.5, -1.5, -1.5, -1.5], dtype=np.float32) * dipole_sep
    z_offsets = np.zeros(x_offsets.shape)

    weights += np.zeros(x_offsets.shape)

    dipole_jones_matrix = ideal_gaussian_beam(l, 0, nu=frequency, diameter=1)
    array_factor = get_array_factor(x_offsets, y_offsets,  weights, l, m, frequency)

    tile_response = array_factor * dipole_jones_matrix
    normalisation = tile_response[0]
    tile_response /= normalisation

    if not normalisation_only:
        output = tile_response
    if normalisation_only:
        output = normalisation

    return output


def get_array_factor(x, y, weights, l, m, l0=0, m0=0, frequency=150e6):
    wavelength = c / frequency
    number_dipoles = len(x)

    k_x = (2. * np.pi / wavelength) * l
    k_y = (2. * np.pi / wavelength) * m

    k_x0 = (2. * np.pi / wavelength) * l0
    k_y0 = (2. * np.pi / wavelength) * m0
    array_factor_map = np.zeros(l.shape, dtype=complex)

    for i in range(number_dipoles):
        complex_exponent = -1j * ((k_x - k_x0) * x[i] + (k_y - k_y0) * y[i])

        # !This step takes a long time, look into optimisation through vectorisation/clever np usage
        dipole_factor = weights[i] * np.exp(complex_exponent)

        array_factor_map += dipole_factor

    # filter all NaN
    array_factor_map[np.isnan(array_factor_map)] = 0
    array_factor_map = array_factor_map / np.sum(weights)

    return array_factor_map