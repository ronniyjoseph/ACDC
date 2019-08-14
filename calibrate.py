import sys
import numpy
from scipy.optimize import fmin_cg

sys.path.append("../corrcal2")
from corrcal2 import sparse_2level
from corrcal2 import get_chisq
from corrcal2 import get_gradient


def calibrate(data, noise_variance,covariance_vectors, model_vectors, edges, antenna1_indices, antenna2_indices,
              gain_guess = None, scale_factor = 1000):
    if gain_guess is None:
        #compute number of antennas based on number of baselines
        n_antennas = int((1 + numpy.sqrt(1 + 4*len(noise_variance)))/2)
        gain_guess = numpy.zeros(2 * n_antennas)
        gain_guess[::2] = 1

    sparse_matrix_object = sparse_2level(noise_variance, covariance_vectors, model_vectors, edges)

    gain_solutions_split = 1/scale_factor*fmin_cg(get_chisq, gain_guess * scale_factor, get_gradient,
                                   (data, sparse_matrix_object, antenna1_indices, antenna2_indices, scale_factor))
    gain_solutions_complex = gain_solutions_split[::2] + 1j*gain_solutions_split[1::2]

    return gain_solutions_complex
