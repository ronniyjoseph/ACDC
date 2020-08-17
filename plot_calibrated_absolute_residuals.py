import numpy
import argparse
import copy
import time
from matplotlib import colors

from src.covariance import SkyCovariance
from src.covariance import BeamCovariance
from src.covariance import GainCovariance
from src.covariance import CalibratedResiduals
from src.covariance import compute_weights
from pyrem.skymodel import sky_moment_returner
from pyrem.powerspectrum import from_frequency_to_eta
from pyrem.plottools import plot_1dpower_spectrum

from pyrem.powerspectrum import from_frequency_to_eta
from src.powerspectrum import fiducial_eor_power_spectrum

from pyrem.radiotelescope import RadioTelescope

from pyrem.plottools import plot_2dpower_spectrum
from pyrem.plottools import plot_power_contours
from pyrem.generaltools import from_jansky_to_milikelvin

from pyrem.covariance import calibrated_residual_error

from pyrem.util import redundant_baseline_finder


def main(labelfontsize = 20, ticksize= 15):
    model_limit = 100e-3
    broken_fraction = 0.25
    u_plot = 7

    telescope_position_path = "./data/MWA_Compact_Coordinates.txt"

    u_range = numpy.logspace(-1, numpy.log10(500), 100)
    frequency_range = numpy.linspace(135, 165,251 )* 1e6
    eta = from_frequency_to_eta(frequency_range)

    eor_power_spectrum = fiducial_eor_power_spectrum(u_range, eta)

    telescope = RadioTelescope(load=True, path=telescope_position_path)
    redundant_table = redundant_baseline_finder(telescope.baseline_table)

    sky_covariance = SkyCovariance(model_depth=model_limit)
    sky_covariance.compute_covariance(u=u_range, v = 0, nu=frequency_range)

    model_sky = copy.deepcopy(sky_covariance)
    unmodeled_mu = sky_moment_returner(2, s_low=sky_covariance.s_low, s_mid=sky_covariance.s_mid,
                                       s_high=sky_covariance.model_depth, k1=sky_covariance.k1,
                                       gamma1=sky_covariance.alpha1, k2=sky_covariance.k2, gamma2=sky_covariance.alpha2)
    modeled_mu = sky_moment_returner(2, s_low=sky_covariance.model_depth, s_mid=sky_covariance.s_mid,
                                       s_high=sky_covariance.s_high, k1=sky_covariance.k1,
                                       gamma1=sky_covariance.alpha1, k2=sky_covariance.k2, gamma2=sky_covariance.alpha2)

    model_sky.matrix *= modeled_mu / unmodeled_mu

    beam_covariance = BeamCovariance(model_depth=model_limit, calibration_type='sky', broken_fraction=broken_fraction)
    beam_covariance.compute_covariance(u=u_range, v = 0, nu=frequency_range)
    total_covariance = sky_covariance + beam_covariance

    sky_gain = GainCovariance(sky_covariance, calibration_type='absolute', baseline_table=redundant_table)
    beam_gain = GainCovariance(beam_covariance, calibration_type='absolute', baseline_table=redundant_table)
    total_gain= GainCovariance(total_covariance, calibration_type='absolute', baseline_table=redundant_table)

    sky_residuals = CalibratedResiduals(sky_gain, model_matrix=model_sky, residual_matrix=sky_covariance)
    beam_residuals = CalibratedResiduals(beam_gain, model_matrix=model_sky, residual_matrix=sky_covariance)
    total_residuals = CalibratedResiduals(total_gain, model_matrix=model_sky, residual_matrix=sky_covariance)

    sky_power = sky_residuals.compute_power()
    beam_power = beam_residuals.compute_power()
    total_power = total_residuals.compute_power()

    index = numpy.where(numpy.abs(u_range - u_plot) == numpy.min(numpy.abs(u_range - u_plot)))[0]

    sky_calibrated = sky_power[index, ...]
    beam_calibrated= beam_power[index,...]
    total_calibrated= total_power[index, ...]
    eor_power_spectrum = eor_power_spectrum[index,...]

    figure, axes = pyplot.subplots(1, 3, figsize=(15, 5))

    ps_norm = colors.LogNorm(vmin=1e3, vmax=1e15)

    plot_1dpower_spectrum(eta, frequency_range, sky_calibrated[0,:], title="Sky Error", axes=axes[0],
                        axes_label_font=labelfontsize, tickfontsize=ticksize, xlabel_show=True, ylabel_show=True)

    plot_1dpower_spectrum(eta, frequency_range, beam_calibrated[0,:], title="Beam Variations", axes=axes[1],
                        axes_label_font=labelfontsize, tickfontsize=ticksize, xlabel_show=True, ylabel_show=False)

    plot_1dpower_spectrum(eta, frequency_range, total_calibrated[0,:], title="Total Error", axes=axes[2],
                        axes_label_font=labelfontsize, tickfontsize=ticksize, xlabel_show=True, ylabel_show=False)


    plot_1dpower_spectrum(eta, frequency_range, eor_power_spectrum[0, :], axes=axes[0], ratio=True, color = 'k')
    plot_1dpower_spectrum(eta, frequency_range, eor_power_spectrum[0, :], axes=axes[1], ratio=True, color = 'k')
    plot_1dpower_spectrum(eta, frequency_range, eor_power_spectrum[0, :], axes=axes[2], ratio=True, color = 'k')

    # plot_power_contours(eta, frequency_range, from_jansky_to_milikelvin(position_calibrated, frequency_range)/eor_power_spectrum,
    #                     axes=axes[0], ratio=True, axes_label_font=labelfontsize, tickfontsize=ticksize, xlabel_show=True, norm=ps_norm, ylabel_show=False, contour_levels=contour_levels)
    #
    # plot_power_contours(u_range, eta, frequency_range, from_jansky_to_milikelvin(beam_calibrated, frequency_range)/eor_power_spectrum,
    #                     axes=axes[1], ratio=True, axes_label_font=labelfontsize, tickfontsize=ticksize, xlabel_show=True, norm=ps_norm, ylabel_show=False, contour_levels=contour_levels)
    #
    # plot_power_contours(u_range, eta, frequency_range, from_jansky_to_milikelvin(total_calibrated, frequency_range)/eor_power_spectrum,
    #                     axes=axes[2], ratio=True, axes_label_font=labelfontsize, tickfontsize=ticksize, xlabel_show=True, norm=ps_norm, ylabel_show=False, contour_levels=contour_levels)
    pyplot.tight_layout()
    pyplot.savefig("../plots/Calibrated_Residuals_Absolute_MWA.pdf")
    pyplot.show()
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ssh", action="store_true", dest="ssh_key", default=False)
    params = parser.parse_args()

    import matplotlib

    if params.ssh_key:
        matplotlib.use("Agg")
    from matplotlib import pyplot

    main()
