import numpy
import argparse
from matplotlib import colors

from src.covariance import PositionCovariance
from src.covariance import BeamCovariance
from pyrem.powerspectrum import from_frequency_to_eta
from pyrem.plottools import plot_2dpower_spectrum


def main(labelfontsize = 20, ticksize= 15):
    tile_precision = 1e-2
    broken_fraction = 0.25
    k_perp_range = numpy.array([1e-4, 1.1e-1])
    u_range = numpy.logspace(-1, numpy.log10(500), 100)
    frequency_range = numpy.linspace(135, 165, 251) * 1e6
    eta = from_frequency_to_eta(frequency_range)

    position_covariance = PositionCovariance(position_precision=tile_precision)
    position_covariance.compute_covariance(u=u_range, v=0, nu=frequency_range)
    beam_covariance = BeamCovariance(model_depth=10, calibration_type='redundant', broken_fraction=broken_fraction)
    beam_covariance.compute_covariance(u=u_range, v=0, nu = frequency_range)

    position_error_power = position_covariance.compute_power()
    beam_error_power = beam_covariance.compute_power()
    total_error_power = position_error_power + beam_error_power

    figure, axes = pyplot.subplots(1, 3, figsize=(15, 5))

    ps_norm = colors.LogNorm(vmin=1e3, vmax=1e15)

    plot_2dpower_spectrum(u_range, eta, frequency_range, position_error_power, title="Position Error", axes=axes[0],
                        axes_label_font=labelfontsize, tickfontsize=ticksize, colorbar_show=True,
                        xlabel_show=True, norm=ps_norm, ylabel_show=True)

    plot_2dpower_spectrum(u_range, eta, frequency_range, beam_error_power, title="Beam Model Error", axes=axes[1],
                        axes_label_font=labelfontsize, tickfontsize=ticksize, colorbar_show=True,
                        xlabel_show=True, norm=ps_norm)

    plot_2dpower_spectrum(u_range, eta, frequency_range, total_error_power, title="Total Error", axes=axes[2],
                        axes_label_font=labelfontsize, tickfontsize=ticksize, colorbar_show=True,
                        xlabel_show=True, norm=ps_norm, zlabel_show=True)

    figure.tight_layout()
    figure.savefig("../plots/Uncalibrated_redundant_residuals.pdf")


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
