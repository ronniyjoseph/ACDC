import numpy
import argparse
from matplotlib import colors

from src.covariance import sky_covariance_pab
from src.covariance import SkyCovariance
from pyrem.powerspectrum import compute_power
from pyrem.powerspectrum import from_frequency_to_eta
from pyrem.plottools import plot_2dpower_spectrum

import time

def main(labelfontsize = 16, ticksize= 11):
    k_perp_range = numpy.array([1e-4, 1.1e-1])
    u_range = numpy.logspace(-1, numpy.log10(500), 100)
    frequency_range = numpy.linspace(135, 165, 251) * 1e6
    eta = from_frequency_to_eta(frequency_range)

    start = time.time()
    sky_covariance = SkyCovariance(model_depth=100e-3)
    sky_covariance.compute_covariance(u=u_range, v= 0, nu=frequency_range)
    sky_error_power = sky_covariance.compute_power()

    # sky_error_power = calculate_sky_power_spectrum(u=u_range, nu=frequency_range)
    lapse = time.time() - start
    print(f"It took {lapse}")
    figure, axes = pyplot.subplots(1, 1, figsize=(5, 5))

    ps_norm = colors.LogNorm(vmin=1e3, vmax=1e15)

    plot_2dpower_spectrum(u_range, eta, frequency_range, sky_error_power, title="Sky Model Error", axes=axes,
                        axes_label_font=labelfontsize, tickfontsize=ticksize, colorbar_show=True,
                        xlabel_show=True, norm=ps_norm, ylabel_show=True, zlabel_show=True)

    figure.tight_layout()
    # pyplot.show()
    figure.savefig("../plots/Sky_Class_Test.pdf")

    return


def calculate_sky_power_spectrum(u, nu):
    variance = numpy.zeros((len(u), int(len(nu) / 2)))

    print(f"Calculating covariances for all baselines")
    for i in range(len(u)):
        nu_cov = sky_covariance_pab(u[i], v=0, nu=nu, S_high=100e-3, gamma=0.8)
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
