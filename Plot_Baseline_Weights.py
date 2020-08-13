import sys
import numpy
import powerbox
import argparse
import matplotlib.colors as colors

from pyrem.radiotelescope import RadioTelescope
from pyrem.radiotelescope import ideal_gaussian_beam
from pyrem.util import redundant_baseline_finder

# from src.skymodel import SkyRealisation
# from src.generaltools import from_lm_to_theta_phi
# from src.plottools import colorbar
#
# from scipy.signal import convolve2d

sys.path.append("../")

def main(plot_u_dist = False ,plot_array_matrix = False, plot_inverse_matrix = False, plot_weights = False ,
         grid_weights = True, binned_weights = True):
    plot_blaah = True
    show_plot = True
    save_plot = False

    path = "./Data/MWA_Compact_Coordinates.txt"
    plot_folder = "../Plots/Analytic_Covariance/"

    telescope = RadioTelescope(load=True, path=path)

    if plot_u_dist:
        make_plot_uv_distribution(telescope, show_plot=show_plot, save_plot=save_plot, plot_folder=plot_folder)

    sky_matrix = sky_matrix_constructor(telescope)
    inverse_redundant_matrix = redundant_matrix_constructor_alt(telescope)

    # print(redundant_matrix.shape)
    print(sky_matrix.shape)
    print(f"Sky Calibration uses {len(sky_matrix[:, 3])/2} baselines")
    # print(f"Redundant Calibration uses {len(redundant_matrix[:, 3])/2} baselines")

    if plot_array_matrix:
        make_plot_array_matrix(sky_matrix, show_plot=show_plot, save_plot=save_plot, plot_folder=plot_folder)
        make_plot_array_matrix(redundant_matrix, show_plot=show_plot, save_plot=save_plot, plot_folder=plot_folder)

    print(numpy.linalg.cond(sky_matrix))
    # print(numpy.linalg.cond(redundant_matrix))

    inverse_sky_matrix = numpy.linalg.pinv(sky_matrix)
    print("MAXIMUM", numpy.abs(inverse_redundant_matrix).max())
    # inverse_redundant_matrix = numpy.linalg.pinv(redundant_matrix)

    if plot_inverse_matrix:
        make_plot_array_matrix(inverse_sky_matrix.T, show_plot=show_plot, save_plot=save_plot, plot_folder=plot_folder)
        make_plot_array_matrix(inverse_redundant_matrix.T, show_plot=show_plot, save_plot=save_plot, plot_folder=plot_folder)

    sky_weights = numpy.sqrt((numpy.abs(inverse_sky_matrix[::2, ::2])**2 +
                                   numpy.abs(inverse_sky_matrix[1::2, 1::2])**2))

    redundant_weights = numpy.sqrt((numpy.abs(inverse_redundant_matrix[::2, ::2])**2 +
                                   numpy.abs(inverse_redundant_matrix[1::2, 1::2])**2))

    print(f"Every Sky calibrated Tile sees {len(sky_weights[0,:][sky_weights[0, :] > 1e-4])}")
    print(f"Every redundant Tile sees {len(redundant_weights[0,:][redundant_weights[0, :] > 1e-4])}")

    u_bins_sky = numpy.linspace(0, 375, 50)
    u_bins_red = numpy.linspace(0, 100, 50)
    redundant_bins, redundant_uu_weights, red_counts = compute_binned_weights(redundant_baseline_finder(telescope.baseline_table),
                                                                  redundant_weights, binned=True, u_bins=u_bins_red)
    sky_bins, sky_uu_weights, sky_counts = compute_binned_weights(telescope.baseline_table, sky_weights, binned=True, u_bins=u_bins_sky)
    if plot_blaah:
        figure, axes = pyplot.subplots(2,3, figsize=(6, 4))

        sky_bins, sky_approx = baseline_hist(sky_bins, telescope.baseline_table)
        redundant_bins, redundant_approx = baseline_hist(redundant_bins, redundant_baseline_finder(telescope.baseline_table))

        norm = colors.LogNorm()
        redplot = axes[1, 0].pcolor(redundant_bins, redundant_bins, redundant_uu_weights, norm = norm)
        # redplot = axes[1, 0].semilogy(redundant_bins[:len(redundant_bins)-1],redundant_uu_weights, 'k', alpha = 0.3)

        norm = colors.LogNorm()

        skyplot = axes[0, 0].pcolor(sky_bins, sky_bins, sky_uu_weights, norm=norm)
        # skyplot = axes[0, 0].semilogy(sky_bins[:len(sky_bins)-1],sky_uu_weights, 'k', alpha = 0.3)


        norm = colors.LogNorm()
        countsredplot = axes[1, 1].pcolor(redundant_bins, redundant_bins, red_counts, norm = norm)
        # countsredplot = axes[1, 1].semilogy(redundant_bins[:len(redundant_bins)-1],red_counts, 'k', alpha = 0.3)

        norm = colors.LogNorm()

        countskyplot = axes[0, 1].pcolor(sky_bins, sky_bins, sky_counts, norm = norm)
        # countskyplot = axes[0, 1].semilogy(sky_bins[:len(sky_bins)-1], sky_counts, 'k', alpha = 0.3)


        norm = colors.LogNorm()
        # normredplot = axes[1, 2].pcolor(redundant_bins, redundant_bins, redundant_uu_weights/red_counts,
        #                                  norm = norm)
        normredplot = axes[1, 2].semilogy(redundant_bins[:len(redundant_bins)-1], (redundant_uu_weights/red_counts).T, 'k', alpha = 0.3)
        # normredplot = axes[1, 2].semilogy(redundant_bins[:len(redundant_bins)-1], redundant_approx, 'C0')
        normredplot = axes[1, 2].semilogy(redundant_bins[:len(redundant_bins)-1], redundant_approx*(245/2018), 'C1')

        # normskyplot = axes[0, 2].pcolor(sky_bins, sky_bins, sky_uu_weights/sky_counts, norm = norm)
        normskyplot = axes[0, 2].semilogy(sky_bins[:len(sky_bins)-1],(sky_uu_weights/sky_counts).T, 'k', alpha = 0.3)
        # normskyplot = axes[0, 2].semilogy(sky_bins[:len(sky_bins)-1], sky_approx, 'C0')
        normskyplot = axes[0, 2].semilogy(sky_bins[:len(sky_bins)-1], sky_approx*(127/8128), 'C1')

        axes[1, 0].set_xlabel(r"$u\,[\lambda]$")
        axes[1, 1].set_xlabel(r"$u\,[\lambda]$")
        axes[1, 2].set_xlabel(r"$u\,[\lambda]$")

        axes[0, 0].set_ylabel(r"$u^{\prime}\,[\lambda]$")
        axes[1, 0].set_ylabel(r"$u^{\prime}\,[\lambda]$")

        axes[0, 2].set_ylabel(r"weight")
        axes[1, 2].set_ylabel(r"weight")


        axes[0, 0].set_title("Sky Weights")
        axes[1, 0].set_title("Redundant Weights")

        axes[0, 1].set_title("Sky Normalisation")
        axes[1, 1].set_title("Redundant Normalisation")

        axes[0, 2].set_title("Sky Normalised Weights")
        axes[1, 2].set_title("Redundant Normalised Weights")

        # colorbar(redplot)
        # colorbar(skyplot)
        # colorbar(countsredplot)
        # colorbar(countskyplot)
        # colorbar(normredplot)
        # colorbar(normskyplot)

        pyplot.tight_layout()
        if show_plot:
            pyplot.show()

    return


def compute_binned_weights(baseline_table, baseline_weights, binned=True, u_bins = None):
    antennas = numpy.unique([baseline_table.antenna_id1, baseline_table.antenna_id2] )
    baseline_lengths = numpy.sqrt(baseline_table.u_coordinates**2 + baseline_table.v_coordinates**2)
    uu_weights = numpy.zeros((len(baseline_lengths), len(baseline_lengths)))

    for i in range(len(baseline_lengths)):
        index1 = numpy.where(antennas == baseline_table.antenna_id1[i])[0]
        index2 = numpy.where(antennas == baseline_table.antenna_id2[i])[0]
        if index1 == 0:
            baseline_weights1 = 0
        else:
            baseline_weights1 = baseline_weights[index1 - 1, :]

        if index2 == 0:
            baseline_weights2 = 0
        else:
            baseline_weights2 = baseline_weights[index2 - 1, :]
        weights = numpy.sqrt((baseline_weights1**2 + baseline_weights2**2))
        weights[weights < 1e-4] = 0
        uu_weights[i, :] = weights

    sorted_indices = numpy.argsort(baseline_lengths)
    sorted_weights = uu_weights[sorted_indices, :][:, sorted_indices]

    if binned:
        if u_bins is None:
            u_bins = numpy.linspace(0, numpy.max(baseline_lengths), 101)
        bin_counter = numpy.zeros_like(uu_weights)
        bin_counter[uu_weights > 0] = 1

        uu1, uu2 = numpy.meshgrid(baseline_lengths, baseline_lengths)


        flattened_uu1 = uu1.flatten()
        flattened_uu2 = uu2.flatten()
        computed_weights = numpy.histogram2d(flattened_uu1, flattened_uu2, bins=u_bins,
                                         weights=uu_weights.flatten())
        computed_counts = numpy.histogram2d(flattened_uu1, flattened_uu2, bins=u_bins, normed=False)
    return u_bins, computed_weights[0], computed_counts[0]


def baseline_hist(u, baseline_table):
    baseline_lengths = numpy.sqrt(baseline_table.u_coordinates**2 + baseline_table.v_coordinates**2)
    hist, edges = numpy.histogram(baseline_lengths, u)
    return u, 1/(hist)


def make_plot_uv_distribution(telescope, show_plot = True, save_plot = False, plot_folder = "./"):
    baseline_lengths = numpy.sqrt(telescope.baseline_table.u_coordinates**2 + telescope.baseline_table.v_coordinates**2)
    figure_u, axes_u = pyplot.subplots(1, 1)
    axes_u.hist(baseline_lengths, density=True, bins=100, label="MWA Phase II Compact")
    axes_u.set_xlabel(r"$u\,[\lambda]$")
    axes_u.set_ylabel("Baseline PDF")
    axes_u.legend()

    if save_plot:
        figure_u.savefig(plot_folder + "MWA_Phase_II_Baseline_PDF.pdf")
    if show_plot:
        pyplot.show()
    return


def make_plot_array_matrix(array_matrix, show_plot = True, save_plot = False, plot_folder = "./"):
    figure_amatrix = pyplot.figure(figsize=(10, 5))

    axes_amatrix = figure_amatrix.add_subplot(111)
    absmatrix = numpy.abs(array_matrix)
    absmatrix[absmatrix <= 1e-4] = 0
    absmatrix[absmatrix > 1e-4] = 1
    # blaah, counts = numpy.unique(absmatrix, axis=0, return_counts=True)
    counts  = (absmatrix[::2, ::2] == 1).sum(axis=0)
    pyplot.hist(counts[0:128], bins=30)
    pyplot.xlabel("Number of Baselines", fontsize =20)
    pyplot.ylabel("Number of Antennas", fontsize =20)
    pyplot.show()
    #plot_amatrix = axes_amatrix.plot(array_matrix.T)
    #colorbar(plot_amatrix)
    axes_amatrix.set_xlabel("Baseline Number", fontsize = 15)
    axes_amatrix.set_ylabel("Antenna Number", fontsize = 15)
    if save_plot:
        figure_amatrix.savefig(plot_folder + "Array_Matrix_Double.pdf")
    if show_plot:
        pyplot.show()
    return


def redundant_matrix_constructor_alt(telescope):
    redundant_table = redundant_baseline_finder(telescope.baseline_table)
    redundant_group_ids = numpy.unique(redundant_table.group_indices)
    antennas = numpy.unique([redundant_table.antenna_id1, redundant_table.antenna_id2] )
    x_positions, y_positions = redundant_positions(telescope, antennas)


    number_antennas = len(antennas)
    number_groups = len(redundant_group_ids)
    array_matrix = numpy.zeros((2 * redundant_table.number_of_baselines, 2 * number_antennas + 2*number_groups))

    for i in range(redundant_table.number_of_baselines):
        index1 = numpy.where(antennas == redundant_table.antenna_id1[i])[0]
        index2 = numpy.where(antennas == redundant_table.antenna_id2[i])[0]
        index3 = numpy.where(redundant_group_ids == redundant_table.group_indices[i])[0]
        # Fill in the real rows
        array_matrix[2 * i, 2 * index1] = 1
        array_matrix[2 * i, 2 * index2] = 1
        array_matrix[2 * i, 2*number_antennas + 2 * index3] = 1


        # Fill in the imaginary rows
        array_matrix[2 * i + 1, 2 * index1 + 1] = 1
        array_matrix[2 * i + 1, 2 * index2 + 1] = -1
        array_matrix[2 * i + 1, 2*number_antennas + 2 * index3 + 1] = 1
    constraints = numpy.zeros((3,2 * number_antennas + 2*number_groups))
    constraints[0, 1: 2* number_antennas +1:2] = 1
    constraints[1, 1: 2* number_antennas +1:2] = x_positions
    constraints[2, 1: 2 * number_antennas + 1:2] = y_positions

    constrained_matrix = numpy.vstack((array_matrix, constraints))
    constrained_matrix = constrained_matrix[:, 2:]
    inv = numpy.linalg.pinv(constrained_matrix)
    print("bfdg")
    print(array_matrix.shape)
    print(inv.shape)
    print(inv[: , :2 * redundant_table.number_of_baselines].shape)
    return inv[:, :2 * redundant_table.number_of_baselines]


def redundant_matrix_constructor(telescope):
    redundant_table = redundant_baseline_finder(telescope.baseline_table)
    redundant_group_ids = numpy.unique(redundant_table.group_indices)
    antennas = numpy.unique([redundant_table.antenna_id1, redundant_table.antenna_id2] )
    x_positions, y_positions = redundant_positions(telescope, antennas)


    number_antennas = len(antennas)
    number_groups = len(redundant_group_ids)
    array_matrix = numpy.zeros((2 * redundant_table.number_of_baselines, 2 * number_antennas + 2*number_groups))

    for i in range(redundant_table.number_of_baselines):
        index1 = numpy.where(antennas == redundant_table.antenna_id1[i])[0]
        index2 = numpy.where(antennas == redundant_table.antenna_id2[i])[0]
        index3 = numpy.where(redundant_group_ids == redundant_table.group_indices[i])[0]
        # Fill in the real rows
        array_matrix[2 * i, 2 * index1] = 1
        array_matrix[2 * i, 2 * index2] = 1
        array_matrix[2 * i, 2*number_antennas + 2 * index3] = 1


        # Fill in the imaginary rows
        array_matrix[2 * i + 1, 2 * index1 + 1] = 1
        array_matrix[2 * i + 1, 2 * index2 + 1] = -1
        array_matrix[2 * i + 1, 2*number_antennas + 2 * index3 + 1] = 1

    # constraints = numpy.zeros((2,2 * number_antennas + 2*number_groups))
    # constraints[0, 1: 2* number_antennas +1:2] = x_positions
    # constraints[1, 1: 2 * number_antennas + 1:2] = y_positions
    #
    # constrained_matrix = numpy.vstack((array_matrix, constraints))
    constrained_matrix = array_matrix[:, 2:]
    return constrained_matrix


def sky_matrix_constructor(telescope):
    antennas = telescope.antenna_positions.antenna_ids
    baselines = telescope.baseline_table
    array_matrix = numpy.zeros((2 * baselines.number_of_baselines, 2 * len(antennas)))

    for i in range(baselines.number_of_baselines):
        index1 = numpy.where(antennas == baselines.antenna_id1[i])[0]
        index2 = numpy.where(antennas == baselines.antenna_id2[i])[0]

        # Fill in the real rows
        array_matrix[2 * i, 2 * index1] = 1
        array_matrix[2 * i, 2 * index2] = 1

        # Fill in the imaginary rows

        array_matrix[2 * i + 1, 2 * index1 + 1] = 1
        array_matrix[2 * i + 1, 2 * index2 + 1] = -1

    constrained_matrix = array_matrix[:, 2:]
    return constrained_matrix


def redundant_positions(telescope, antennas):
    antenna_table = telescope.antenna_positions
    x_positions = numpy.zeros(len(antennas))
    y_positions = x_positions.copy()

    for i in range(len(antennas)):
        index = numpy.where(antenna_table.antenna_ids == antennas[i])[0]
        x_positions[i] = antenna_table.x_coordinates[index]
        y_positions[i] = antenna_table.y_coordinates[index]
    return x_positions, y_positions


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ssh", action="store_true", dest="ssh_key", default=False)
    params = parser.parse_args()

    import matplotlib

    if params.ssh_key:
        matplotlib.use("Agg")
    from matplotlib import pyplot

    main()