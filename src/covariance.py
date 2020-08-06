def sky_covariance_pab(u, v, nu, S_low=1e-3, S_mid=1., S_high=10., gamma=0.8, mode = 'frequency', nu_0 = 150e6,

                   dx=1.1):
    mu_2 = sky_moment_returner(2, s_low=S_low, s_mid=S_mid, s_high=S_high)
    x, y = mwa_dipole_locations(dx=dx)

    nn1, nn2 = numpy.meshgrid(nu, nu)
    xx = (numpy.meshgrid(x, x, x, x, indexing = "ij"))
    yy = (numpy.meshgrid(y, y, y, y, indexing = "ij"))

    dxx = (xx[0] - xx[1], xx[2] - xx[3])
    dyy = (yy[0] - yy[1], yy[2] - yy[3])

    #set up matrix indexing arrays for efficient computation and mapping
    x_index, y_index = numpy.meshgrid(numpy.arange(0, len(nu), 1), numpy.arange(0, len(nu), 1))
    x_index = x_index.flatten()
    y_index = y_index.flatten()

    index = numpy.arange(0, int(len(nu)**2), 1)
    index = index.reshape((len(nu), len(nu)))
    index = numpy.triu(index, k =0)
    index = index.flatten()
    index = index[index > 0]
    index = numpy.concatenate((numpy.array([0]), index))

    pool = multiprocessing.Pool(4)
    output = numpy.array(pool.map(partial(parallel, u, v, nn1.flatten(), nn2.flatten(), dxx, dyy, gamma), index))
    covariance = numpy.zeros((len(nu), len(nu)))
    covariance[x_index[index], y_index[index]] = 2 * numpy.pi * mu_2 * output
    covariance[y_index[index], x_index[index]] = 2 * numpy.pi * mu_2 * output
    return covariance


def parallel(u, v, nn1, nn2, dxx, dyy, gamma, i):

    datatype = numpy.float64
    nu0 = nn2[0].astype(dtype=datatype)
    nu1 = nn1[i].astype(dtype=datatype)
    nu2 = nn2[i].astype(dtype=datatype)

    width_tile1 = beam_width(nu1, diameter=1.0)
    width_tile2 = beam_width(nu2, diameter=1.0)

    sigma_nu = width_tile1 ** 2 * width_tile2 ** 2 / (width_tile1 ** 2 + width_tile2 ** 2)
    sigma_taper = sigma_nu/(1 + sigma_nu)

    a = u * (nu1 - nu2) / nu0 + dxx[0].astype(dtype=datatype) * nu1 / c + dxx[1].astype(dtype=datatype) * nu2 / c
    b = v * (nu1 - nu2) / nu0 + dyy[0].astype(dtype=datatype) * nu1 / c + dyy[1].astype(dtype=datatype) * nu2 / c

    kernels =  numpy.exp(-2 * numpy.pi**2 * sigma_taper * (a**2 + b**2))
    covariance = numpy.sum(sigma_nu*(nu1 * nu2/nu0 ** 2) ** (-gamma) * kernels)/dxx[0].shape[0]**4

    return covariance