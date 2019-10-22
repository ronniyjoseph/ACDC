import os
import sys
import numpy
from numba import prange, njit
from matplotlib import pyplot
from src.util import hexagonal_array

sys.path.append("../../beam_perturbations/code/tile_beam_perturbations/")

from radiotelescope import RadioTelescope
from radiotelescope import AntennaPositions
from radiotelescope import BaselineTable
from radiotelescope import broken_mwa_beam_loader
from skymodel import SkyRealisation
from skymodel import create_visibilities_analytic
from generaltools import from_lm_to_theta_phi
from cramer_rao_bound import redundant_baseline_finder
import time


def beam_covariance_simulation(hex_array=True, array_size=4, create_signal=True,
                               load=False, compute_covariance=False, plot_covariance=True):
    output_path = "/data/rjoseph/Hybrid_Calibration/numerical_simulations/"
    project_path = "redundant_based_beam_covariance"
    n_realisations = 1

    if not os.path.exists(output_path + project_path + "/"):
        print("Creating Project folder at output destination!")
        os.makedirs(output_path + project_path)

    if create_signal:
        hex_telescope = create_hex_telescope(4)
        create_visibility_data(hex_telescope, n_realisations, output_path + project_path, output_data=True)

    if compute_covariance:
        compute_covariance(baseline_table, frequency_range, output_path + project_path, n_realisations)

    #
    # if plot_covariance:
    #     #gain_covariance_impact(baseline_table, frequency_range, output_path + project_path)
    #     residual_PS_error(baseline_table, frequency_range, output_path + project_path)
    #

    return


def broken_tiles(telescope, fraction=25 / 128, seed=None, number_dipoles=16):
    if seed is not None:
        numpy.random.seed((seed))
    number_antennas = len(telescope.antenna_positions.x_coordinates)
    # Determine number of broken tiles
    number_broken_tiles = numpy.random.binomial(n=number_antennas, p=fraction, size=1)
    # Select which tiles are broken
    broken_flags = numpy.zeros(number_antennas, dtype=int)
    broken_tile_indices = numpy.random.randint(0, number_antennas, number_broken_tiles)
    broken_dipole_indices = numpy.random.randint(0, number_dipoles, number_broken_tiles)
    broken_flags[broken_tile_indices] = broken_dipole_indices

    return broken_flags


def create_visibility_data(telescope_object, n_realisations, path, output_data=False):
    print("Creating Signal Realisations")
    if not os.path.exists(path + "/" + "Simulated_Visibilities") and output_data:
        print("Creating realisation folder in Project path")
        os.makedirs(path + "/" + "Simulated_Visibilities")

    for i in range(n_realisations):
        print(f"Realisation {i}")
        broken_flags = broken_tiles(telescope_object, seed=i)
        source_population = SkyRealisation(sky_type='random', flux_high=1)
        original_table = telescope_object.baseline_table
        redundant_baselines = redundant_baseline_finder(original_table.antenna_id1, original_table.antenna_id2,
                                                        original_table.u_coordinates, original_table.v_coordinates,
                                                        original_table.w_coordinates, verbose=False)
        redundant_table = BaselineTable()
        redundant_table.antenna_id1 = redundant_baselines[:, 0]
        redundant_table.antenna_id2 = redundant_baselines[:, 1]
        redundant_table.u_coordinates= redundant_baselines[:, 2]
        redundant_table.v_coordinates= redundant_baselines[:, 3]
        redundant_table.w_coordinates= redundant_baselines[:, 4]
        redundant_table.reference_frequency = 150e6
        redundant_table.number_of_baselines = len(redundant_baselines[:, 0])

        model_visibilities = create_visibilities_analytic(source_population, redundant_table,
                                                           frequency_range = numpy.array([150e6]))
        t1 = time.perf_counter()
        perturbed_visibilities = create_perturbed_visibilities(source_population, redundant_table, broken_flags)
        t2 = time.perf_counter()
        print(f"{t1 - t2} total perturb vis time")
        residual_visibilities = model_visibilities - perturbed_visibilities

        numpy.save(path +  "/" + "Simulated_Visibilities/" + f"model_realisation_{i}", model_visibilities)
        numpy.save(path +  "/" + "Simulated_Visibilities/" + f"perturbed_realisation_{i}", perturbed_visibilities)
        numpy.save(path + "/" + "Simulated_Visibilities/" + f"residual_realisation_{i}", residual_visibilities)
    return


def create_hex_telescope(size):
    hex_telescope = RadioTelescope(load=False)

    antenna_positions = hexagonal_array(size)
    antenna_table = AntennaPositions(load=False)
    antenna_table.antenna_ids = numpy.arange(0, antenna_positions.shape[0], 1)
    antenna_table.x_coordinates = antenna_positions[:, 0]
    antenna_table.y_coordinates = antenna_positions[:, 1]
    antenna_table.z_coordinates = antenna_positions[:, 2]

    hex_telescope.antenna_positions = antenna_table
    hex_telescope.baseline_table = BaselineTable(position_table=antenna_table)

    return hex_telescope


def apparent_flux_possibilities(source_population, number_of_dipoles=16, nu=150e6):
    number_of_sources = len(source_population.fluxes)
    theta, phi = from_lm_to_theta_phi(source_population.l_coordinates, source_population.m_coordinates)

    beam_response = numpy.zeros((number_of_sources, number_of_dipoles + 1), dtype=complex)
    flux_beam_product = numpy.zeros_like(beam_response)
    for i in range(number_of_dipoles + 1):
        if i == 0:
            faulty_dipole_i = None
        else:
            faulty_dipole_i = i - 1

        beam_response[:, i] = broken_mwa_beam_loader(theta, phi, frequency=nu, faulty_dipole=faulty_dipole_i,
                                                     load=False)
        flux_beam_product[:, i] = beam_response[:, i] * source_population.fluxes

    apparent_fluxes = numpy.einsum('ij,ik->ijk', flux_beam_product, numpy.conj(beam_response))
    return apparent_fluxes


def create_perturbed_visibilities(source_population, baseline_table, broken_flags, frequency = 150e6):
    observations = numpy.zeros(baseline_table.number_of_baselines, dtype = complex)
    t1 = time.perf_counter()
    apparent_fluxes = apparent_flux_possibilities(source_population, nu = frequency)
    t2 = time.perf_counter()
    print(f"{t2-t1} Time fo generate fluxes ")


    flags_antenna1 = broken_flags[baseline_table.antenna_id1.astype(int)]
    flags_antenna2 = broken_flags[baseline_table.antenna_id2.astype(int)]
    t3 = time.perf_counter()
    numba_perturbed_loop(observations, apparent_fluxes, source_population.l_coordinates, source_population.m_coordinates
                         , baseline_table.u(frequency), baseline_table.v(frequency), flags_antenna1, flags_antenna2)
    t4 = time.perf_counter()
    print(t4-t3)
    return observations


@njit(parallel=True)
def numba_perturbed_loop(observations, fluxes, l_source, m_source, u_baselines, v_baselines, broken_flags1,
                         broken_flags2):
    for source_index in prange(len(fluxes)):
        for baseline_index in range(u_baselines.shape[0]):
            kernel = numpy.exp(-2j * numpy.pi * (u_baselines[baseline_index] * l_source[source_index] +
                                                 v_baselines[baseline_index] * m_source[source_index]))

            observations[baseline_index] += fluxes[source_index, broken_flags1[baseline_index],
                                                   broken_flags2[baseline_index]] * kernel


def compute_frequency_frequency_covariance_serial(baseline_table, frequency_range, path, n_realisations):
    if not os.path.exists(path + "/" + "Simulated_Covariance"):
        print("Creating realisation folder in Project path")
        os.makedirs(path + "/" + "Simulated_Covariance")

    baseline_frequency_covariance = numpy.zeros((baseline_table.number_of_baselines, len(frequency_range),
                                                 len(frequency_range)), dtype=complex)

    for j in range(baseline_table.number_of_baselines):
        noise_signal_ratio = numpy.zeros((len(frequency_range), n_realisations), dtype = complex)

        if not j%100 and j != 0 or j == 1:
            print(f"Estimated time to finish re-processing = {delta_t*(baseline_table.number_of_baselines - j)}")

        t0 = time.perf_counter()
        for i in range(n_realisations):
            model_signal = numpy.load(path + f"model_realisation_{i}.npy")
            residual_signal = numpy.load(path + f"residual_realisation_{i}.npy")
            noise_signal_ratio[:, i] = residual_signal[j, :] / model_signal[j, :]

        baseline_frequency_covariance[j, ...] = numpy.cov(noise_signal_ratio)
        t1 = time.perf_counter()
        delta_t = t1 - t0

    numpy.save(path + f"frequency_frequency_covariance", baseline_frequency_covariance)

    return baseline_frequency_covariance



if __name__ == "__main__":
    beam_covariance_simulation()
