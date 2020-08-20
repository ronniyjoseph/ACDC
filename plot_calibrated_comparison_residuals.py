import numpy
import argparse
import copy
import time
import pickle
from matplotlib import colors
from scipy.interpolate import RectBivariateSpline

from src.covariance import SkyCovariance
from src.covariance import PositionCovariance
from src.covariance import BeamCovariance
from src.covariance import GainCovariance
from src.covariance import CalibratedResiduals

from pyrem.skymodel import sky_moment_returner
from pyrem.plottools import plot_2dpower_spectrum
from pyrem.plottools import plot_power_contours
from pyrem.util import redundant_baseline_finder
from pyrem.powerspectrum import from_frequency_to_eta
from src.powerspectrum import fiducial_eor_power_spectrum

from pyrem.radiotelescope import RadioTelescope
from pyrem.generaltools import from_jansky_to_milikelvin

def main(labelfontsize = 20, ticksize= 20):
    model_limit = 100e-3
    position_error = 0.01
    broken_fraction = 0.25
    compute = False
    save = False
    load = True
    telescope_position_path = "./data/MWA_Compact_Coordinates.txt"

    u_range = numpy.logspace(-1, numpy.log10(500), 100)
    frequency_range = numpy.linspace(135, 165, 251 )* 1e6
    eta = from_frequency_to_eta(frequency_range)
    print(f"{eta.max()} blaah")
    eor_power_spectrum = fiducial_eor_power_spectrum(u_range, eta)

    telescope = RadioTelescope(load=True, path=telescope_position_path)
    redundant_table = redundant_baseline_finder(telescope.baseline_table)

    contour_levels = numpy.array([1e-1, 1e0, 1e2])
    sky_clocations = None#[(6e-2, 0.21), (4e-2, 0.25), (1.1e-2, 0.35 )]
    beam_clocations = None #[(6e-2, 0.21), (8e-2, 0.5), (1.1e-2, 0.35 )]
    linestyles = ['dotted','dashed','solid' ]


    if compute:
    #### Initialise Sky Errors ###
        sky_error = SkyCovariance(model_depth=model_limit)
        sky_error.compute_covariance(u=u_range, v = 0, nu=frequency_range)
        sky_beam_covariance = BeamCovariance(model_depth=model_limit, calibration_type='sky', broken_fraction=broken_fraction)
        sky_beam_covariance.compute_covariance(u=u_range, v = 0, nu=frequency_range)
        sky_total = sky_error + sky_beam_covariance

    if load:
        with open('unmodeled_sky.obj', 'rb') as f:
            sky_error = pickle.load(f)
        with open('sky_total.obj','rb') as f:
            sky_total = pickle.load(f)

    sky_gain = GainCovariance(sky_total, calibration_type='sky', baseline_table=redundant_table)

    if compute:
        ###### Initialise Redundant Errors
        position_covariance = PositionCovariance(position_precision=position_error)
        position_covariance.compute_covariance(u=u_range, v = 0, nu=frequency_range)
        beam_covariance = BeamCovariance(model_depth=model_limit, calibration_type='relative', broken_fraction=broken_fraction)
        beam_covariance.compute_covariance(u=u_range, v = 0, nu=frequency_range)
        redundant_total = position_covariance + beam_covariance
    if load:
        with open('redundant_residuals.obj','rb') as f:
           redundant_total = pickle.load(f)

    relative_gain= GainCovariance(redundant_total, calibration_type='relative', baseline_table=redundant_table)
    absolute_gain= GainCovariance(sky_total, calibration_type='absolute', baseline_table=redundant_table)
    redundant_gain = relative_gain + absolute_gain


    ##### Rescale unmodeled sky covariance to model (avoiding heavy computation)
    model_sky = residual_to_model_rescale(sky_error)

    ######### Compute residuals
    redundant_residuals = CalibratedResiduals(redundant_gain, model_matrix=model_sky, residual_matrix=sky_error)
    sky_residuals = CalibratedResiduals(sky_gain, model_matrix=model_sky, residual_matrix=sky_error)

    ###### Compute PS #########
    redundant_power = redundant_residuals.compute_power()
    sky_power = sky_residuals.compute_power()

    if save:
    ###### Save covariance matrices
        sky_error.save("unmodeled_sky")
        sky_beam_covariance.save("sky_beam_error")
        sky_total.save("sky_total")

        position_covariance.save("position_error")
        beam_covariance.save("redundant_beam")
        redundant_total.save("redundant_residuals")

    figure, axes = pyplot.subplots(1, 2, figsize=(11, 5))
    ps_norm = colors.LogNorm(vmin=1e3, vmax=1e15)

    plot_2dpower_spectrum(u_range, eta, frequency_range, sky_power, title="Sky Based", axes=axes[0],
                        axes_label_font=labelfontsize, tickfontsize=ticksize, colorbar_show=False,
                        xlabel_show=True, norm=ps_norm, ylabel_show=True)

    plot_2dpower_spectrum(u_range, eta, frequency_range, redundant_power, title="Redundancy Based", axes=axes[1],
                        axes_label_font=labelfontsize, tickfontsize=ticksize, colorbar_show=True, zlabel_show=True,
                        xlabel_show=True, norm=ps_norm, ylabel_show=False)


    plot_power_contours(u_range, eta, frequency_range, from_jansky_to_milikelvin(sky_power, frequency_range)/eor_power_spectrum,
                        axes=axes[0], ratio=True, axes_label_font=labelfontsize, tickfontsize=ticksize, xlabel_show=True,
                        norm=ps_norm, ylabel_show=False, contour_levels=contour_levels, contour_label_locs=sky_clocations,
                        smooth = 3, contour_styles = linestyles)

    plot_power_contours(u_range, eta, frequency_range, from_jansky_to_milikelvin(redundant_power, frequency_range)/eor_power_spectrum,
                        axes=axes[1], ratio=True, axes_label_font=labelfontsize, tickfontsize=ticksize, xlabel_show=True,
                        norm=ps_norm, ylabel_show=False, contour_levels=contour_levels, contour_label_locs=beam_clocations,
                        smooth =3, contour_styles = linestyles)



    pyplot.tight_layout()
    pyplot.savefig("../plots/Calibrated_Residuals_Comparison_MWA.pdf")
    pyplot.show()
    return


def residual_to_model_rescale(sky_error):

    model_sky = copy.deepcopy(sky_error)
    unmodeled_mu = sky_moment_returner(2, s_low=sky_error.s_low, s_mid=sky_error.s_mid,
                                       s_high=sky_error.model_depth, k1=sky_error.k1,
                                       gamma1=sky_error.alpha1, k2=sky_error.k2, gamma2=sky_error.alpha2)
    modeled_mu = sky_moment_returner(2, s_low=sky_error.model_depth, s_mid=sky_error.s_mid,
                                     s_high=sky_error.s_high, k1=sky_error.k1,
                                     gamma1=sky_error.alpha1, k2=sky_error.k2, gamma2=sky_error.alpha2)

    model_sky.matrix *= modeled_mu / unmodeled_mu

    return model_sky



def coarse_band_taper(nu):
    taper = numpy.zeros_like(nu) + 1
    for i in range(24+1):
        taper[i*128:i*128+14] = 0
    tt1, tt2 = numpy.meshgrid(taper, taper)
    taper = tt1*tt2
    return taper


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ssh", action="store_true", dest="ssh_key", default=False)
    params = parser.parse_args()

    import matplotlib

    if params.ssh_key:
        matplotlib.use("Agg")
    from matplotlib import pyplot

    main()
