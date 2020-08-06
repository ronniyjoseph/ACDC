import numpy as np
import argparse
from matplotlib import colors

from pyrem.radiotelescope import ideal_gaussian_beam
from pyrem.radiotelescope import airy_beam
from pyrem.radiotelescope import simple_mwa_tile
from src.beammodel import simple_mwa_tile as simple_beam

from src.beammodel import mwa_fee_model

def main(labelfontsize = 15, tickfontsize= 15):
    theta = np.linspace(0, np.pi/2, 300)
    phi = np.zeros_like(theta)

    mwa_tile_size = 4

    mwa_model = mwa_fee_model(theta, phi)

    mwa_gaussian = ideal_gaussian_beam(np.sin(theta), 0, nu=150e6, diameter=mwa_tile_size, epsilon=0.42)
    mwa_airy = airy_beam(np.sin(theta), diameter=mwa_tile_size*0.7)
    # mwa_simple = simple_mwa_tile(theta, 0,)

    l = np.linspace(0, 10, 1000)
    mwa_simple = simple_mwa_tile(theta, 0)
    mwa_l = simple_beam(l, 0)
    taper = ideal_gaussian_beam(np.sin(theta), 0, nu=150e6, diameter=1.0, epsilon=1)



    figure, axes = plt.subplots(1,1, figsize = (6,5))
    model_line_width = 5
    model_line_alpha = 0.4
    model_line_color = 'k'

    # axes.plot(np.degrees(theta), np.abs(mwa_model), linewidth = model_line_width, alpha=model_line_alpha,
    #              color=model_line_color, label = "FEE")
    # axes.plot(np.degrees(theta), np.abs(mwa_gaussian)**2, label = "Gaussian")
    # axes.plot(np.degrees(theta), np.abs(mwa_airy)**2, label = "Airy")
    # axes.plot(np.degrees(theta), np.abs(mwa_simple)**2, label = "Multi-Gaussian")
    # axes.plot(np.degrees(theta), np.abs(mwa_simple*taper)**2, label = "taper")
    axes.plot(np.sin(theta), (mwa_simple)**2)
    axes.plot(l, mwa_l**2)

    axes.tick_params(axis='both', which='major', labelsize=tickfontsize)
    axes.set_yscale('log')
    axes.set_ylim([1e-6, 5])
    axes.set_xlabel(r"Zenith Angle [$^\circ$]", fontsize = labelfontsize)
    axes.set_ylabel("Normalised Response", fontsize = labelfontsize)

    axes.legend()
    figure.tight_layout()
    plt.show()


    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ssh", action="store_true", dest="ssh_key", default=False)
    params = parser.parse_args()

    import matplotlib

    if params.ssh_key:
        matplotlib.use("Agg")
    from matplotlib import pyplot as plt


    main()