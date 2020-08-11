import numpy
import argparse
from matplotlib import colors

from src.covariance import position_covariance
from src.covariance import beam_covariance
from src.powerspectrum import compute_power
from src.powerspectrum import from_frequency_to_eta
from src.plottools import plot_2dpower_spectrum


def main(labelfontsize = 20, ticksize= 15):
    k_perp_range = numpy.array([1e-4, 1.1e-1])
    u_range = numpy.logspace(-1, numpy.log10(500), 100)
    frequency_range = numpy.linspace(135, 165, 251) * 1e6
    eta = from_frequency_to_eta(frequency_range)

    position_error_power = calculate_position_power_spectrum(u=u_range, nu=frequency_range)
    beam_error_power = calculate_beam_power_spectrum(u=u_range, nu=frequency_range)
    total_error_power = calculate_total_power_spectrum(u=u_range, nu=frequency_range)

    figure, axes = pyplot.subplots(1, 3, figsize=(15, 5))

    ps_norm = colors.LogNorm(vmin=1e3, vmax=1e15)

    plot_2dpower_spectrum(u_range, eta, frequency_range, position_error_power, title="Position Error", axes=axes[0],
                        axes_label_font=labelfontsize, tickfontsize=ticksize, colorbar_show=True,
                        xlabel_show=True, norm=ps_norm, ylabel_show=True)

    plot_2dpower_spectrum(u_range, eta, frequency_range, beam_error_power, title="Beam Variations", axes=axes[1],
                        axes_label_font=labelfontsize, tickfontsize=ticksize, colorbar_show=True,
                        xlabel_show=True, norm=ps_norm)

    plot_2dpower_spectrum(u_range, eta, frequency_range, total_error_power, title="Total Noise", axes=axes[2],
                        axes_label_font=labelfontsize, tickfontsize=ticksize, colorbar_show=True,
                        xlabel_show=True, norm=ps_norm, zlabel_show=True)

    figure.tight_layout()

    pyplot.show()

    return


def calculate_position_power_spectrum(u, nu):
    variance = numpy.zeros((len(u), int(len(nu) / 2)))

    print(f"Calculating covariances for all baselines")
    for i in range(len(u)):
        nu_cov = position_covariance(u[i], v=0, nu=nu)
        variance[i, :] = compute_power(nu, nu_cov)

    return variance


def calculate_beam_power_spectrum(u, nu):
    variance = numpy.zeros((len(u), int(len(nu) / 2)))

    print(f"Calculating covariances for all baselines")
    for i in range(len(u)):
        nu_cov = beam_covariance(u[i], v=0, nu=nu, calibration_type="redundant")
        variance[i, :] = compute_power(nu, nu_cov)

    return variance


def calculate_total_power_spectrum(u, nu):
    variance = numpy.zeros((len(u), int(len(nu) / 2)))

    print(f"Calculating covariances for all baselines")
    for i in range(len(u)):
        nu_cov = beam_covariance(u= u[i], v=0, nu=nu, calibration_type="redundant") + \
                 position_covariance(u=u[i], v=0, nu=nu)
        variance[i, :] = compute_power(nu, nu_cov)

    return variance


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ssh", action="store_true", dest="ssh_key", default=False)
    params = parser.parse_args()

    import matplotlib

    if params.ssh_key:
        matplotlib.use("Agg")
    from matplotlib import pyplot

    main()
