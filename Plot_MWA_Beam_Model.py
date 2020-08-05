import numpy as np
import argparse
from matplotlib import colors

from src.radiotelescope import mwa_tile_beam
from src.radiotelescope import ideal_gaussian_beam
from src.radiotelescope import airy_beam
from src.radiotelescope import simple_mwa_tile

from pyuvdata import UVBeam
from pyuvdata.data import DATA_PATH

from mwa_pb import config
from mwa_pb.beam_full_EE import ApertureArray
from mwa_pb.beam_full_EE import Beam

from mpl_toolkits.axes_grid1 import make_axes_locatable


def main(labelfontsize = 15, tickfontsize= 15):
    theta = np.linspace(0, np.pi, 300)
    phi = np.zeros_like(theta)

    mwa_tile_size = 4

    mwa_model = mwa_fee_model(theta, phi)
    # hera_model = hera_fee_model(theta, phi)


    mwa_gaussian = ideal_gaussian_beam(np.sin(theta), 0, nu=150e6, diameter=mwa_tile_size, epsilon=0.42)
    mwa_airy = airy_beam(np.sin(theta), diameter=mwa_tile_size*0.7)
    mwa_simple = simple_mwa_tile(theta, 0)

    # hera_gaussian = ideal_gaussian_beam(np.sin(theta), 0, nu=150e6, diameter=hera_dish_size, epsilon=0.42)
    # hera_airy = airy_beam(theta, diameter=hera_dish_size)

    figure, axes = plt.subplots(1,1, figsize = (5,5), subplot_kw = {"yscale": "log",
                                                                     "ylim": (5e-5, 2)})
    model_line_width = 5
    model_line_alpha = 0.4
    model_line_color = 'k'

    axes.plot(np.degrees(theta), np.sqrt(np.abs(mwa_model)), linewidth = model_line_width, alpha=model_line_alpha,
                 color=model_line_color, label = "FEE")
    axes.plot(np.degrees(theta), np.abs(mwa_gaussian), label = "Gaussian")
    axes.plot(np.degrees(theta), np.abs(mwa_airy), label = "Airy")
    axes.plot(np.degrees(theta), np.abs(mwa_simple), label = "Multi-Gaussian")

    axes.tick_params(axis='both', which='major', labelsize=tickfontsize)

    axes.set_xlabel(r"Zenith Angle [$^\circ$]", fontsize = labelfontsize)

    axes.set_ylabel("Normalised Response", fontsize = labelfontsize)
    axes.legend()

    plt.show()


    return


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


def hera_fee_model(theta, phi, nu = 150e6):
    path = "data/HERA_beam/"
    beam = UVBeam()
    # settings_file = os.path.join(DATA_PATH, path + 'HERA_Vivaldi_CST_beams.yaml')
    beam.read_beamfits(path + "NF_HERA_Dipole_power_beam.fits")
    beam.peak_normalize()
    beam.interpolation_function = 'az_za_simple'
    beam.freq_interp_kind = 'linear'
    response = beam.interp(az_array = phi, za_array = theta, freq_array=np.array([nu]))
    return response[0][0,0,0,0,:]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ssh", action="store_true", dest="ssh_key", default=False)
    params = parser.parse_args()

    import matplotlib

    if params.ssh_key:
        matplotlib.use("Agg")
    from matplotlib import pyplot as plt


    main()