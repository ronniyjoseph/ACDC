import numpy
from scipy import signal
from scipy.interpolate import interp1d

from matplotlib import pyplot

def discrete_fourier_transform_matrix(nu):
    dft = numpy.exp(-2 * numpy.pi * 1j / len(nu)) ** numpy.arange(0, len(nu), 1)
    dftmatrix = numpy.vander(dft, increasing=True) / numpy.sqrt(len(nu))

    return dftmatrix


def from_frequency_to_eta(nu):
    eta = numpy.arange(0, len(nu), 1) / (nu.max() - nu.min())
    return eta[:int(len(nu) / 2)]


def blackman_harris_taper(frequency_range):
    window = signal.blackmanharris(len(frequency_range))
    return window


def compute_power(nu, covariance):
    dft_matrix = discrete_fourier_transform_matrix(nu)
    window = blackman_harris_taper(nu)
    taper1, taper2 = numpy.meshgrid(window, window)

    tapered_cov = covariance * taper1 * taper2

    eta_cov = numpy.dot(numpy.dot(dft_matrix.conj().T, tapered_cov), dft_matrix)
    power = numpy.diag(numpy.real(eta_cov))[:int(len(nu) / 2)]

    return power


def read_data(ps_data_path):

    z_fiducial = 8
    bandwidth_fiducial = 30e6
    central_frequency_fiducial = 1.42e9 / (z_fiducial + 1)

    power_spectrum_fiducial_eor = numpy.loadtxt(ps_data_path, delimiter=',')
    u_range_fiducial = numpy.linspace(0, 500, 100)
    frequency_range_fiducial = numpy.linspace(central_frequency_fiducial - bandwidth_fiducial / 2,
                                              central_frequency_fiducial + bandwidth_fiducial / 2, 251)

    eta_fiducial = from_frequency_to_eta(frequency_range_fiducial)
    # hist, eta_fiducial = numpy.histogram(eta_fiducial[:int(len(eta_fiducial) / 2)], bins=251//2)

    return u_range_fiducial, eta_fiducial, power_spectrum_fiducial_eor.T


def interpolate(u, eta, u_original, eta_original, ps_data):

    eta_interpolated = numpy.zeros((len(u_original), len(eta)))
    for i in range(len(u_original)):
        eta_1d_interp = interp1d(eta_original, ps_data[i, :], kind='cubic')
        eta_interpolated[i, :] = eta_1d_interp(eta)

    fully_interpolated = numpy.zeros((len(u), len(eta)))
    for i in range(len(eta)):
        u_1d_interp = interp1d(u_original, eta_interpolated[:, i], kind='cubic')
        fully_interpolated[:, i] = u_1d_interp(u)

    return fully_interpolated


def fiducial_eor_power_spectrum(u, eta, path = "./data/"
                 , file = "redshift8.csv"):
    u_fiducial, eta_fiducial, ps_fiducial = read_data(path + file)
    ps_interpolated = interpolate(u, eta, u_fiducial, eta_fiducial, ps_fiducial)
    return ps_interpolated