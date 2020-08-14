import numpy as np
import multiprocessing
import copy

from functools import partial
from scipy.constants import c
from scipy import signal

from pyrem.skymodel import sky_moment_returner
from pyrem.radiotelescope import mwa_dipole_locations
from pyrem.radiotelescope import beam_width
from pyrem.powerspectrum import compute_power


class Covariance:
    def __init__(self, s_low=1e-5, s_mid=1., s_high=10., gamma=0.8, alpha1 = 1.59, alpha2 = 2.5,
                 k1 = 4100, k2 = 4100, calibration_type = None):
        self.matrix = None                      #Place Holder for Covariance Computation
        self.nu = None                          #Place Holder for Frequency array
        self.u = None                           #Place Holder for baseline lengths
        self.gamma = gamma                      #Radio Spectral Energy Distribution Power Law Index

        self.s_low = s_low                      #Lowest Brightness sources
        self.s_mid = s_mid                      #Midpoint Brightness Source Counts
        self.s_high = s_high                    #Maximum Brightness Source Counts
        self.alpha1 = alpha1                    #Source Count Slope
        self.alpha2 = alpha2                    #Source Count Slope
        self.k1 = k1                            #Source Count Normalisation
        self.k2 = k2                            #Source count normalisation
        self.calibration_type = calibration_type
        return


    def __add__(self, other):
        # check whether all parameters are the same
        total_covariance = Covariance()
        for k, v in self.__dict__.items():
            if k != "matrix":
                assert np.array_equal(self.__dict__[k],other.__dict__[k]), f"Cannot add matrices because {k} is not the same"
            total_covariance.__dict__[k] = copy.deepcopy(v)
        total_covariance.matrix = self.matrix + other.matrix

        return total_covariance


    def compute_power(self):
        print("Computing Power Spectrum Contamination")

        n_u_scales = self.matrix.shape[0]
        n_nu_channels = self.matrix.shape[1]
        variance = np.zeros((n_u_scales, int(n_nu_channels/2)))

        for i in range(n_u_scales):
            variance[i, :] = compute_power(self.nu, self.matrix[i,...])

        return variance


    def setup_computation(self):
        return


class SkyCovariance(Covariance):

    def __init__(self, model_depth = None, **kwargs):
        super(SkyCovariance, self).__init__(**kwargs)
        self.calibration_type = "sky"
        self.model_depth = model_depth
        assert self.model_depth is not None, "Specify a sky model catalogue depth by setting 'model_depth'"


    def compute_covariance(self, u, v, nu):
        print("Computing Sky Covariance Matrix")

        mu_2 = sky_moment_returner(n_order=2, s_low=self.s_low, s_mid=self.s_mid, s_high=self.model_depth, k1=self.k1,
                                   gamma1=self.alpha1, k2=self.k2, gamma2=self.alpha2)
        x, y = mwa_dipole_locations(dx=1.1)

        nn1, nn2 = np.meshgrid(nu, nu)
        xx = (np.meshgrid(x, x, x, x, indexing="ij"))
        yy = (np.meshgrid(y, y, y, y, indexing="ij"))

        dxx = (xx[0] - xx[1], xx[2] - xx[3])
        dyy = (yy[0] - yy[1], yy[2] - yy[3])
        index, i_index, j_index = covariance_indexing(nu)

        self.matrix = np.zeros((len(u), len(nu), len(nu)))
        self.u = u
        self.nu = nu

        for k in range(len(u)):
            pool = multiprocessing.Pool(4)
            output = np.array(
            pool.map(partial(covariance_kernels, u[k], v, nn1.flatten(), nn2.flatten(), dxx, dyy, self.gamma), index))
            pool.close()

            self.matrix[k, i_index[index], j_index[index]] = 2 * np.pi * mu_2 * output / dxx[0].shape[0] ** 4
            self.matrix[k, j_index[index], i_index[index]] = self.matrix[k, i_index[index], j_index[index]]

        return


class BeamCovariance(Covariance):

    def __init__(self, model_depth = None, calibration_type=None, broken_fraction=1, **kwargs):
        super(BeamCovariance, self).__init__(**kwargs)

        self.model_depth = model_depth
        self.calibration_type = calibration_type
        self.broken_fraction = broken_fraction
        assert self.model_depth is not None, "Specify a sky model catalogue depth by setting 'model_depth'"
        assert self.calibration_type is not None, "Specify a sky model catalogue depth by setting 'calibration_type' to" \
                                                  "'sky' or 'redundant'"


    def compute_covariance(self, u, v, nu):
        print("Computing Beam Covariance Matrix")
        mu_1_r = sky_moment_returner(1, s_high=self.model_depth, s_low=self.s_low, s_mid=self.s_mid, k1=self.k1,
                                   gamma1=self.alpha1, k2=self.k2, gamma2=self.alpha2)
        mu_2_r = sky_moment_returner(2, s_high=self.model_depth, s_low=self.s_low, s_mid=self.s_mid, k1=self.k1,
                                   gamma1=self.alpha1, k2=self.k2, gamma2=self.alpha2)

        mu_1_m = sky_moment_returner(1, s_low=self.model_depth, s_mid = self.s_mid, s_high=self.s_high, k1 = self.k1,
        gamma1 = self.alpha1, k2 = self.k2, gamma2 = self.alpha2)
        mu_2_m = sky_moment_returner(2, s_low=self.model_depth, s_mid = self.s_mid, s_high=self.s_high, k1 = self.k1,
        gamma1 = self.alpha1, k2 = self.k2, gamma2 = self.alpha2)

        x, y = mwa_dipole_locations(dx=1.1)
        nn1, nn2 = np.meshgrid(nu, nu)
        index, i_index, j_index = covariance_indexing(nu)

        xx = (np.meshgrid(x, x, x, indexing="ij"))
        yy = (np.meshgrid(y, y, y, indexing="ij"))
        dxx = (xx[1] - xx[0], xx[0] - xx[2])
        dyy = (yy[1] - yy[0], yy[0] - yy[2])

        self.matrix = np.zeros((len(u), len(nu), len(nu)))
        self.u = u
        self.nu = nu

        for k in range(len(u)):
            pool = multiprocessing.Pool(4)
            kernel_A = np.array(
                pool.map(partial(covariance_kernels, u[k], v, nn1.flatten(), nn2.flatten(), dxx, dyy, self.gamma), index))
            self.matrix[k, i_index[index], j_index[index]] = 2 * np.pi * (mu_2_m + mu_2_r) * kernel_A / dxx[0].shape[0] ** 5
            pool.close()

            if self.calibration_type == 'sky':
                xx = (np.meshgrid(x, x, x, x, indexing="ij"))
                yy = (np.meshgrid(y, y, y, y, indexing="ij"))
                dxx = (xx[2] - xx[0], xx[1] - xx[3])
                dyy = (yy[2] - yy[0], yy[1] - yy[3])
                pool = multiprocessing.Pool(4)
                kernel_B = np.array(
                    pool.map(partial(covariance_kernels, u[k], v, nn1.flatten(), nn2.flatten(), dxx, dyy, self.gamma), index))
                self.matrix[k, i_index[index], j_index[index]] += -4 * np.pi * mu_2_r * kernel_B / dxx[0].shape[0] ** 5
                pool.close()

            self.matrix[k, j_index[index], i_index[index]] = self.matrix[k, i_index[index], j_index[index]]
        self.matrix *= self.broken_fraction**2
        return self.matrix


class PositionCovariance(Covariance):
    def __init__(self, position_precision = None, **kwargs):
        super(PositionCovariance, self).__init__(**kwargs)
        self.calibration_type = "relative"
        self.position_precision = position_precision

        assert self.position_precision is not None, "Specify a antenna position precision by setting 'position_precision'"


    def compute_covariance(self, u, v, nu):
        print("Computing Position Covariance Matrix")

        mu_2 = sky_moment_returner(2, s_low=self.s_low, s_mid=self.s_mid, s_high=self.s_high, k1 = self.k1,
                                   gamma1 = self.alpha1, k2 = self.k2, gamma2 = self.alpha2)
        delta_u = self.position_precision * nu[0] / c

        x, y = mwa_dipole_locations(dx=1.1)

        nn1, nn2 = np.meshgrid(nu, nu)
        xx = (np.meshgrid(x, x, x, x, indexing="ij"))
        yy = (np.meshgrid(y, y, y, y, indexing="ij"))

        dxx = (xx[0] - xx[1], xx[2] - xx[3])
        dyy = (yy[0] - yy[1], yy[2] - yy[3])
        index, i_index, j_index = covariance_indexing(nu)

        self.matrix = np.zeros((len(u), len(nu), len(nu)))
        self.u = u
        self.nu = nu
        for k in range(len(u)):
            pool = multiprocessing.Pool(4)
            output = np.array(
                pool.map(partial(derivative_kernels, u[k], v, nn1.flatten(), nn2.flatten(), dxx, dyy, self.gamma), index))
            pool.close()

            self.matrix[k,i_index[index], j_index[index]] = 16 * delta_u ** 2 * np.pi ** 3 * mu_2 * output / \
                                                            dxx[0].shape[0] ** 4
            self.matrix[k, j_index[index], i_index[index]] = self.matrix[k, i_index[index], j_index[index]]
        return self.matrix


class GainCovariance(Covariance):

    def __init__(self, residual_covariance, calibration_type = None, baseline_table = None):
        super().__init__(self)
        for k, v in residual_covariance.__dict__.items():
            self.__dict__[k] = copy.deepcopy(v)

        self.residual_matrix = residual_covariance.matrix                      #Place Holder for Covariance Computation

        self.compute_covariance(calibration_type = calibration_type, baseline_table=baseline_table)
        return

    def compute_covariance(self, calibration_type = None, baseline_table=None):

        if calibration_type == "sky" or calibration_type == 'absolute':
            model_variance = sky_moment_returner(2, s_low=self.s_low, s_mid=self.s_mid, s_high=self.model_depth,
                                                 k1 = self.k1, gamma1 = self.alpha1, k2 = self.k2, gamma2 = self.alpha2)
        elif calibration_type == "relative":
            model_variance = sky_moment_returner(2, s_low=self.s_low, s_mid=self.s_mid, s_high=self.s_high,
                                                 k1 = self.k1, gamma1 = self.alpha1, k2 = self.k2, gamma2 = self.alpha2)
        else:
            raise ValueError("Specify calibration_type= to 'sky' or 'relative")

        self.residual_matrix /= model_variance

        if baseline_table is None:
            n_parameters = calibration_parameter_number(baseline_table, calibration_type)
            self.matrix = np.sum(self.residual_matrix, axis=0) * (1 / (n_parameters * len(self.u))) ** 2
        else:
            weights = compute_weights(self.u, baseline_table, calibration_type)
            self.matrix = np.zeros_like(self.residual_matrix)
            for k in range(len(self.u)):
                u_weight_reshaped = np.tile(weights[k, :].flatten(), (len(self.nu), len(self.nu), 1)).T
                self.matrix[k, ...] = np.sum(self.residual_matrix * u_weight_reshaped, axis=0)

        if calibration_type == "absolute":
            absolute_averaged_covariance = np.zeros_like(self.matrix)
            for i in range(self.matrix.shape[0]):
                absolute_averaged_covariance[i, ...] = np.mean(self.matrix, axis=0)
            self.matrix = absolute_averaged_covariance
        return


class CalibratedResiduals(Covariance):

    def __init__(self, gaincovariance):

        return

def covariance_kernels(u, v, nn1, nn2, dxx, dyy, gamma, i):
    datatype = np.float64
    nu0 = nn2[0].astype(dtype=datatype)
    nu1 = nn1[i].astype(dtype=datatype)
    nu2 = nn2[i].astype(dtype=datatype)

    width_tile1 = beam_width(nu1, diameter=1.0)
    width_tile2 = beam_width(nu2, diameter=1.0)

    sigma_nu = width_tile1 ** 2 * width_tile2 ** 2 / (width_tile1 ** 2 + width_tile2 ** 2)
    taper_size = 0.8**2/2
    sigma_taper = taper_size**2*sigma_nu/(taper_size**2 + sigma_nu)

    a = u * (nu1 - nu2) / nu0 + dxx[0].astype(dtype=datatype) * nu1 / c + dxx[1].astype(dtype=datatype) * nu2 / c
    b = v * (nu1 - nu2) / nu0 + dyy[0].astype(dtype=datatype) * nu1 / c + dyy[1].astype(dtype=datatype) * nu2 / c

    kernels =  np.exp(-2 * np.pi**2 * sigma_taper * (a**2 + b**2))
    covariance = np.sum(sigma_nu*(nu1 * nu2/nu0 ** 2) ** (-gamma) * kernels)

    return covariance


def derivative_kernels(u, v, nn1, nn2, dxx, dyy, gamma, i):
    datatype = np.float64
    nu0 = nn2[0].astype(dtype=datatype)
    nu1 = nn1[i].astype(dtype=datatype)
    nu2 = nn2[i].astype(dtype=datatype)

    width_tile1 = beam_width(nu1, diameter=1.0)
    width_tile2 = beam_width(nu2, diameter=1.0)

    sigma_nu = width_tile1 ** 2 * width_tile2 ** 2 / (width_tile1 ** 2 + width_tile2 ** 2)
    taper_size = 0.8**2/2
    sigma_taper = taper_size**2*sigma_nu/(taper_size**2 + sigma_nu)

    a = u * (nu1 - nu2) / nu0 + dxx[0].astype(dtype=datatype) * nu1 / c + dxx[1].astype(dtype=datatype) * nu2 / c
    b = v * (nu1 - nu2) / nu0 + dyy[0].astype(dtype=datatype) * nu1 / c + dyy[1].astype(dtype=datatype) * nu2 / c

    kernels =  (1- 2*np.pi**2 * sigma_taper * (a**2 + b**2))*np.exp(-2 * np.pi**2 * sigma_taper * (a**2 + b**2))
    covariance = np.sum(sigma_nu*(nu1 * nu2/nu0 ** 2) ** (1-gamma) * kernels)

    return covariance


def covariance_indexing(nu):
    #set up matrix indexing arrays for efficient computation and mapping
    i_index, j_index = np.meshgrid(np.arange(0, len(nu), 1), np.arange(0, len(nu), 1))
    i_index = i_index.flatten()
    j_index = j_index.flatten()

    index = np.arange(0, int(len(nu)**2), 1)
    index = index.reshape((len(nu), len(nu)))
    index = np.triu(index, k =0)
    index = index.flatten()
    index = index[index > 0]
    index = np.concatenate((np.array([0]), index))
    return index, i_index, j_index


def kernel_index(xx):
    length = xx.shape[0]
    dimensions = len(xx.shape)
    index = np.arange(0, int(length**(dimensions)), 1)
    index = index.reshape(xx.shape)
    index = np.triu(index, k = 1)
    index = index.flatten()
    index = index[index > 0]
    index = np.concatenate((np.array([0]), index))
    return index, i_index, j_index


def compute_weights(u_bins, baseline_table, calibration_type = None):
    n_parameters = calibration_parameter_number(baseline_table, calibration_type)
    u_bin_edges = np.zeros(len(u_bins) + 1)
    baseline_lengths = np.sqrt(baseline_table.u_coordinates**2 + baseline_table.v_coordinates**2)
    log_steps = np.diff(np.log10(u_bins))
    u_bin_edges[1:] = 10**(np.log10(u_bins) + 0.5*log_steps[0])
    u_bin_edges[0] = 10**(np.log10(u_bins[0] - 0.5*log_steps[0]))
    counts, bin_edges = np.histogram(baseline_lengths, bins=u_bin_edges)

    weight_approx = n_parameters / len(baseline_lengths) / counts
    prime, unprime = np.meshgrid(weight_approx, weight_approx)
    weights = prime * unprime
    return weights


def calibration_parameter_number(baseline_table, calibration_type):
    calibration_param_number= len(np.unique([baseline_table.antenna_id1, baseline_table.antenna_id2]))
    if calibration_type == 'relative':
        calibration_param_number += len(np.unique(baseline_table.group_indices))

    return calibration_param_number