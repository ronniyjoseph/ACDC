import numpy
import multiprocessing
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

        return


    def __add__(self, other):
        # check whether all parameters are the same
        assert self.matrix is not None, "Please compute a covariance matrix using compute_covariance"
        assert other.matrix is not None, "Please compute a covariance matrix using compute covariance"

        assert numpy.array_equal(self.nu, other.nu)
        assert numpy.array_equal(self.u, other.u)
        assert self.gamma == other.gamma
        assert self.s_low == other.s_low                      #Lowest Brightness sources
        assert self.s_mid == other.s_mid                      #Midpoint Brightness Source Counts
        assert self.s_high == other.s_high                    #Maximum Brightness Source Counts
        assert self.alpha1 == other.alpha1                    #Source Count Slope
        assert self.alpha2 == other.alpha2                    #Source Count Slope
        assert self.k1 == other.k1                            #Source Count Normalisation
        assert self.k2 == other.k2

        total_covariance = Covariance()
        total_covariance.matrix = self.matrix + other.matrix
        total_covariance.nu = self.nu
        total_covariance.u = self.u
        return total_covariance


    def compute_power(self):
        print("Computing Power Spectrum Contamination")

        n_u_scales = self.matrix.shape[0]
        n_nu_channels = self.matrix.shape[1]
        variance = numpy.zeros((n_u_scales, int(n_nu_channels/2)))

        for i in range(n_u_scales):
            variance[i, :] = compute_power(self.nu, self.matrix[i,...])

        return variance


    def setup_computation(self):
        return


class SkyCovariance(Covariance):

    def __init__(self, model_depth = None, **kwargs):
        self.model_depth = model_depth

        assert self.model_depth is not None, "Specify a sky model catalogue depth by setting 'model_depth'"


        super(SkyCovariance, self).__init__(**kwargs)


    def compute_covariance(self, u, v, nu):
        print("Computing Sky Covariance Matrix")

        mu_2 = sky_moment_returner(n_order=2, s_low=self.s_low, s_mid=self.s_mid, s_high=self.model_depth, k1=self.k1,
                                   gamma1=self.alpha1, k2=self.k2, gamma2=self.alpha2)
        x, y = mwa_dipole_locations(dx=1.1)

        nn1, nn2 = numpy.meshgrid(nu, nu)
        xx = (numpy.meshgrid(x, x, x, x, indexing="ij"))
        yy = (numpy.meshgrid(y, y, y, y, indexing="ij"))

        dxx = (xx[0] - xx[1], xx[2] - xx[3])
        dyy = (yy[0] - yy[1], yy[2] - yy[3])
        index, i_index, j_index = covariance_indexing(nu)

        self.matrix = numpy.zeros((len(u), len(nu), len(nu)))
        self.u = u
        self.nu = nu

        for k in range(len(u)):
            pool = multiprocessing.Pool(4)
            output = numpy.array(
            pool.map(partial(covariance_kernels, u[k], v, nn1.flatten(), nn2.flatten(), dxx, dyy, self.gamma), index))
            pool.close()

            self.matrix[k, i_index[index], j_index[index]] = 2 * numpy.pi * mu_2 * output / dxx[0].shape[0] ** 4
            self.matrix[k, j_index[index], i_index[index]] = self.matrix[k, i_index[index], j_index[index]]

        return


class BeamCovariance(Covariance):

    def __init__(self, model_depth = None, calibration_type=None, broken_fraction=1, **kwargs):
        self.model_depth = model_depth
        self.calibration_type = calibration_type
        self.broken_fraction = broken_fraction
        assert self.model_depth is not None, "Specify a sky model catalogue depth by setting 'model_depth'"
        assert self.calibration_type is not None, "Specify a sky model catalogue depth by setting 'calibration_type' to" \
                                                  "'sky' or 'redundant'"
        super(BeamCovariance, self).__init__(**kwargs)


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
        nn1, nn2 = numpy.meshgrid(nu, nu)
        index, i_index, j_index = covariance_indexing(nu)

        xx = (numpy.meshgrid(x, x, x, indexing="ij"))
        yy = (numpy.meshgrid(y, y, y, indexing="ij"))
        dxx = (xx[1] - xx[0], xx[0] - xx[2])
        dyy = (yy[1] - yy[0], yy[0] - yy[2])

        self.matrix = numpy.zeros((len(u), len(nu), len(nu)))
        self.u = u
        self.nu = nu

        for k in range(len(u)):
            pool = multiprocessing.Pool(4)
            kernel_A = numpy.array(
                pool.map(partial(covariance_kernels, u[k], v, nn1.flatten(), nn2.flatten(), dxx, dyy, self.gamma), index))
            pool.close()
            self.matrix[k, i_index[index], j_index[index]] = 2 * numpy.pi * (mu_2_m + mu_2_r) * kernel_A / dxx[0].shape[0] ** 5

            if self.calibration_type == 'sky':
                xx = (numpy.meshgrid(x, x, x, x, indexing="ij"))
                yy = (numpy.meshgrid(y, y, y, y, indexing="ij"))
                dxx = (xx[2] - xx[0], xx[1] - xx[3])
                dyy = (yy[2] - yy[0], yy[1] - yy[3])
                pool = multiprocessing.Pool(4)
                kernel_B = numpy.array(
                    pool.map(partial(covariance_kernels, u[k], v, nn1.flatten(), nn2.flatten(), dxx, dyy, self.gamma), index))
                self.matrix[k,i_index[index], j_index[index]] += -4 * numpy.pi * mu_2_r * kernel_B / dxx[0].shape[0] ** 5
                pool.close()
            self.matrix[k, j_index[index], i_index[index]] = self.matrix[k,i_index[index], j_index[index]]
            self.matrix *= self.broken_fraction**2
        return self.matrix


class PositionCovariance(Covariance):
    def __init__(self, position_precision = None, **kwargs):
        self.position_precision = position_precision

        assert self.position_precision is not None, "Specify a antenna position precision by setting 'position_precision'"
        super(PositionCovariance, self).__init__(**kwargs)


    def compute_covariance(self, u, v, nu):
        print("Computing Position Covariance Matrix")

        mu_2 = sky_moment_returner(2, s_low=self.s_low, s_mid=self.s_mid, s_high=self.s_high, k1 = self.k1,
                                   gamma1 = self.alpha1, k2 = self.k2, gamma2 = self.alpha2)
        delta_u = self.position_precision * nu[0] / c

        x, y = mwa_dipole_locations(dx=1.1)

        nn1, nn2 = numpy.meshgrid(nu, nu)
        xx = (numpy.meshgrid(x, x, x, x, indexing="ij"))
        yy = (numpy.meshgrid(y, y, y, y, indexing="ij"))

        dxx = (xx[0] - xx[1], xx[2] - xx[3])
        dyy = (yy[0] - yy[1], yy[2] - yy[3])
        index, i_index, j_index = covariance_indexing(nu)

        self.matrix = numpy.zeros((len(u), len(nu), len(nu)))
        self.u = u
        self.nu = nu
        for k in range(len(u)):
            pool = multiprocessing.Pool(4)
            output = numpy.array(
                pool.map(partial(derivative_kernels, u[k], v, nn1.flatten(), nn2.flatten(), dxx, dyy, self.gamma), index))
            pool.close()

            self.matrix[k,i_index[index], j_index[index]] = 16 * delta_u ** 2 * numpy.pi ** 3 * mu_2 * output / \
                                                            dxx[0].shape[0] ** 4
            self.matrix[k, j_index[index], i_index[index]] = self.matrix[k, i_index[index], j_index[index]]
            self.matrix[k,...] *= (nn1 * nn2 / nu[0] ** 2)
        return self.matrix


def covariance_kernels(u, v, nn1, nn2, dxx, dyy, gamma, i):
    datatype = numpy.float64
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

    kernels =  numpy.exp(-2 * numpy.pi**2 * sigma_taper * (a**2 + b**2))
    covariance = numpy.sum(sigma_nu*(nu1 * nu2/nu0 ** 2) ** (-gamma) * kernels)

    return covariance


def derivative_kernels(u, v, nn1, nn2, dxx, dyy, gamma, i):
    datatype = numpy.float64
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

    kernels =  (1- 2*numpy.pi**2 * sigma_taper * (a**2 + b**2))*numpy.exp(-2 * numpy.pi**2 * sigma_taper * (a**2 + b**2))
    covariance = numpy.sum(sigma_nu*(nu1 * nu2/nu0 ** 2) ** (-gamma) * kernels)

    return covariance


def covariance_indexing(nu):
    #set up matrix indexing arrays for efficient computation and mapping
    i_index, j_index = numpy.meshgrid(numpy.arange(0, len(nu), 1), numpy.arange(0, len(nu), 1))
    i_index = i_index.flatten()
    j_index = j_index.flatten()

    index = numpy.arange(0, int(len(nu)**2), 1)
    index = index.reshape((len(nu), len(nu)))
    index = numpy.triu(index, k =0)
    index = index.flatten()
    index = index[index > 0]
    index = numpy.concatenate((numpy.array([0]), index))
    return index, i_index, j_index


def kernel_index(xx):
    length = xx.shape[0]
    dimensions = len(xx.shape)
    index = numpy.arange(0, int(length**(dimensions)), 1)
    index = index.reshape(xx.shape)
    index = numpy.triu(index, k = 1)
    index = index.flatten()
    index = index[index > 0]
    index = numpy.concatenate((numpy.array([0]), index))
    return index, i_index, j_index