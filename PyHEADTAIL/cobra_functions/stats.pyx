"""
@author Kevin Li, Michael Schenk, Stefan Hegglin
@date 31. January 2014
@brief Collection of cython functions to calculate statistics
       of bunch and slice_set data.
@copyright CERN
"""
import numpy as np
cimport numpy as np
cimport libc.math as cmath

cimport cython.boundscheck
cimport cython.cdivision
cimport cython.wraparound



@cython.boundscheck(False)
@cython.cdivision(True)
cpdef double cov(double[::1] a, double[::1] b):
    """ Cython function which calculates the covariance
    (not the covariance matrix!) of two data sets
    a and b using a shifted single pass algorithm
    definition: cov(a,b) = E[(A - E[A])*(B - E[B])]
    implementation: cov(a,b) = 1/(n-1) \sum_{i=1}^n (a_i -<a>)*(b_i -<b>)
    a and b do not necessarily have to reference different data -> var
    shifts are for makeing the algorithm more stable against cancellation
    Args:
        a: numpy array, a.shape[0] defines n. n must be > 1
        b: numpy array, at least with length a.shape[0]

    ~ 3 times faster than cov() for n > 1e5 (timed using %timeit)
    Faster than np.cov() because it does not compute the whole cov matrix
    """
    cdef unsigned int n = a.shape[0]
    if n < 2:
        return 0.
    cdef double shift_a = a[0]
    cdef double shift_b = b[0]
    cdef double a_sum = 0.
    cdef double b_sum = 0.
    cdef double ab_sum = 0.
    cdef unsigned int i
    for i in xrange(n):
        a_sum += a[i] - shift_a
        b_sum += b[i] - shift_b
        ab_sum += (a[i] - shift_a) * (b[i] - shift_b)
    return (ab_sum - a_sum * b_sum / n) / (n - 1)

@cython.boundscheck(False)
@cython.cdivision(True)
cpdef double std(double[::1] u):
    """ Cython function to calculate the standard deviation of
    dataset u. The dataset must consist of at least 2 samples
    """
    return cmath.sqrt(cov(u, u))

@cython.boundscheck(False)
@cython.cdivision(True)
cpdef double dispersion(double[::1] u, double[::1] dp):
    """Cython function to compute the statistial dispersion:
    disp = <u*dp>/<dp**2>
    Args:
        u a coordinate array, typically x or y spatial coordinates
          it is also possible to pass xp or yp
    """
    cdef double mean_u_dp = np.mean(np.multiply(u, dp))
    cdef double mean_dp2 = np.mean(np.multiply(dp, dp))
    if mean_dp2 > 0: # can never be smaller than 0
        return mean_u_dp / mean_dp2
    else:
        return 0

@cython.boundscheck(False)
@cython.cdivision(True)
cdef double _det_beam_matrix(double sigma11, double sigma12, double sigma22):
    """Function which computes the determinant of the 2x2 beam matrix as defined
    in F.Loehl 2005
    Args:
        sigma11: cov(u,u)- disp(u)**2*cov(dp,dp)
        sigma12: cov(u,up) - disp(u)*disp(up)*cov(dp,dp)
    """
    return sigma11 * sigma22 - sigma12 * sigma12

@cython.boundscheck(False)
@cython.cdivision(True)
cpdef double emittance(double[::1] u, double[::1] up, double[::1] dp):
    """ Cython function to calculate the effective (neglecting dispersion)
    emittance of datasets u and up, i.e. a coordinate-momentum pair.
    To calculate the emittance, one needs the mean values of quantities u and
    up.
    Args:
        u spatial coordinate array
        up momentum coordinate array
        dp momentum deviation array: (p-p_0)/p_0. If None, the effective
           emittance is computed instead (dispersion is set to 0)
    """
    covariance = cov
    cdef double sigma11 = 0.
    cdef double sigma12 = 0.
    cdef double sigma22 = 0.
    cdef double cov_u2 = covariance(u,u)
    cdef double cov_up2 = covariance(up, up)
    cdef double cov_u_up = covariance(up, u)
    cdef double cov_u_dp = 0.
    cdef double cov_up_dp = 0.
    cdef double cov_dp2 = 1.

    if dp != None: #if not None, assign values to variables involving dp
        cov_u_dp = covariance(u, dp)
        cov_up_dp = covariance(up,dp)
        cov_dp2 = covariance(dp,dp)

    sigma11 = cov_u2 - cov_u_dp*cov_u_dp/cov_dp2
    sigma12 = cov_u_up - cov_u_dp*cov_up_dp/cov_dp2
    sigma22 = cov_up2 - cov_up_dp*cov_up_dp/cov_dp2

    return cmath.sqrt(_det_beam_matrix(sigma11, sigma12, sigma22))

@cython.boundscheck(False)
@cython.cdivision(True)
cpdef double get_alpha(double[::1] u, double[::1] up, double[::1] dp):
    """Cython function to calculate the statistical alpha (Twiss)
    If dp=None, the effective alpha is computed
    Args:
        u: spatial coordinate array
        up: momentum coordinate array
        dp: (p-p0)/p0
    """
    covariance = cov
    cdef double cov_u_up = covariance(u, up)
    cdef double cov_dp2 = 1.
    cdef double cov_u_dp = 0.
    cdef double cov_up_dp = 0.
    if dp != None:
        cov_dp2 = covariance(dp, dp)
        cov_u_dp = covariance(u, dp)
        cov_up_dp = covariance(up, dp)
    cdef double sigma12 = cov_u_up - cov_u_dp * cov_up_dp / cov_dp2
    return - sigma12 / emittance(u, up, dp)

@cython.boundscheck(False)
@cython.cdivision(True)
cpdef double get_beta(double[::1] u, double[::1] up, double[::1] dp):
    """Cython function to calculate the statistical beta (Twiss)
    If dp=None, the effective beta is computed
    Args:
        u: spatial coordinate array
        up: momentum coordinate array
        dp: (p-p0)/p0
    """
    covariance = cov
    cdef double cov_u2 = covariance(u, u)
    cdef double cov_u_dp = 0.
    cdef double cov_dp2 = 1. # default initialization to 1 -> division if dp=0
    if dp != None:
        cov_u_dp = covariance(u, dp)
        cov_dp2 = covariance(dp, dp)
    cdef double sigma11 = cov_u2 - cov_u_dp * cov_u_dp / cov_dp2
    return sigma11 / emittance(u, up, dp)

@cython.boundscheck(False)
@cython.cdivision(True)
cpdef double get_gamma(double[::1] u, double[::1] up, double[::1] dp):
    """Cython function to calculate the statistical gamma (Twiss)
    If dp=None, the effective gamma is computed
    Args:
        u: spatial coordinate array
        up: momentum coordinate array
        dp: (p-p0)/p0
    """
    covariance = cov
    cdef double cov_up2 = covariance(up, up)
    cdef double cov_up_dp = 0.
    cdef double cov_dp2 = 1.
    if dp != None:
        cov_up_dp = covariance(up, dp)
        cov_dp2 = covariance(dp, dp)
    cdef double sigma22 = cov_up2 - cov_up_dp * cov_up_dp / cov_dp2
    return sigma22 / emittance(u, up, dp)

'''
Cython statistics functions for an instance of a SliceSet class.
'''

@cython.boundscheck(False)
@cython.cdivision(True)
cpdef count_macroparticles_per_slice(int[::1] slice_index_of_particle,
                                     int[::1] particles_within_cuts,
                                     int[::1] n_macroparticles):
    """ Cython function to count the number of macroparticles in
    each slice. """
    cdef unsigned int n_particles_within_cuts = particles_within_cuts.shape[0]
    cdef unsigned int s_idx, i

    for i in xrange(n_particles_within_cuts):
        s_idx = slice_index_of_particle[particles_within_cuts[i]]
        n_macroparticles[s_idx] += 1


@cython.boundscheck(False)
@cython.cdivision(True)
cpdef sort_particle_indices_by_slice(int[::1] slice_index_of_particle,
                                     int[::1] particles_within_cuts,
                                     int[::1] slice_positions,
                                     int[::1] particle_indices_by_slice):
    """ Iterate once through all the particles within the slicing
    region and assign their position in the bunch.z array to the
    respective slice they are in.
    This is to provide a method to the user that allows to see
    which particles are in a specific slice (see
    particle_indices_of_slice in SliceSet class). """
    cdef unsigned int n_part_in_cuts = particles_within_cuts.shape[0]
    cdef unsigned int n_slices = slice_positions.shape[0] - 1

    cdef unsigned int[::1] pos_ctr = np.zeros(n_slices, dtype=np.uint32)
    cdef unsigned int i, p_idx, s_idx
    cdef unsigned int pos

    for i in xrange(n_part_in_cuts):
        p_idx = particles_within_cuts[i]
        s_idx = slice_index_of_particle[p_idx]

        pos = slice_positions[s_idx] + pos_ctr[s_idx]
        particle_indices_by_slice[pos] = p_idx
        pos_ctr[s_idx] += 1


@cython.boundscheck(False)
@cython.cdivision(True)
cpdef mean_per_slice(int[::1] slice_index_of_particle,
                     int[::1] particles_within_cuts,
                     int[::1] n_macroparticles,
                     double[::1] u, double[::1] mean_u):
    """ Iterate once through all the particles within the
    slicing region and calculate simultaneously the mean
    value of quantity u for each slice separately. """
    cdef unsigned int n_part_in_cuts = particles_within_cuts.shape[0]
    cdef unsigned int n_slices = mean_u.shape[0]
    cdef unsigned int p_idx, s_idx, i

    for i in xrange(n_part_in_cuts):
        p_idx = particles_within_cuts[i]
        s_idx = slice_index_of_particle[p_idx]
        mean_u[s_idx] += u[p_idx]

    for i in xrange(n_slices):
        if n_macroparticles[i]:
            mean_u[i] /= n_macroparticles[i]


@cython.boundscheck(False)
@cython.cdivision(True)
cpdef std_per_slice(int[::1] slice_index_of_particle,
                    int[::1] particles_within_cuts,
                    int[::1] n_macroparticles,
                    double[::1] u, double[::1] std_u):
    """ Iterate once through all the particles within the
    slicing region and calculate simultaneously the
    standard deviation of quantity u for each slice
    separately. """
    cdef unsigned int n_part_in_cuts = particles_within_cuts.shape[0]
    cdef unsigned int n_slices = std_u.shape[0]
    cdef unsigned int p_idx, s_idx, i
    cdef double du

    cdef double[::1] mean_u = np.zeros(n_slices, dtype=np.double)
    mean_per_slice(slice_index_of_particle, particles_within_cuts,
                   n_macroparticles, u, mean_u)

    for i in xrange(n_part_in_cuts):
        p_idx = particles_within_cuts[i]
        s_idx = slice_index_of_particle[p_idx]

        du = u[p_idx] - mean_u[s_idx]
        std_u[s_idx] += du * du

    for i in xrange(n_slices):
        if n_macroparticles[i] > 1:
            std_u[i] /= (n_macroparticles[i] -1)

        std_u[i] = cmath.sqrt(std_u[i])

@cython.boundscheck(False)
@cython.cdivision(True)
cpdef cov_per_slice(int[::1] slice_index_of_particle,
                    int[::1] particles_within_cuts,
                    int[::1] n_macroparticles,
                    double[::1] a, double[::1] b, double[::1] result):
    """Cov per slice. Cannot make use of cov() because the particles
    per slice are not contiguous in memory"""
    #TODO: write single pass version of this algorithm
    cdef unsigned int n_part_in_cuts = particles_within_cuts.shape[0]
    cdef unsigned int n_slices = result.shape[0]
    cdef unsigned int p_idx, s_idx, i
    cdef double du

    cdef double[::1] mean_a = np.zeros(n_slices, dtype=np.double)
    cdef double[::1] mean_b = np.zeros(n_slices, dtype=np.double)
    cdef double[::1] cov_ab = np.zeros(n_slices, dtype=np.double)

    mean_per_slice(slice_index_of_particle, particles_within_cuts,
                   n_macroparticles, a, mean_a)
    mean_per_slice(slice_index_of_particle, particles_within_cuts,
                   n_macroparticles, b, mean_b)

    for i in xrange(n_part_in_cuts):
        p_idx = particles_within_cuts[i]
        s_idx = slice_index_of_particle[p_idx]
        cov_ab[s_idx] += (a[p_idx] - mean_a[s_idx])*(b[p_idx] - mean_b[s_idx])

    for i in xrange(n_slices):
        if n_macroparticles[i] > 1:
            cov_ab[i] /= (n_macroparticles[i] -1)
        result[i] = cov_ab[i]


@cython.boundscheck(False)
@cython.cdivision(True)
cpdef emittance_per_slice_old(int[::1] slice_index_of_particle,
                          int[::1] particles_within_cuts,
                          int[::1] n_macroparticles,
                          double[::1] u, double[::1] up,
                          double[::1] epsn_u):
    """ Iterate once through all the particles within the
    slicing region and calculate simultaneously the emittance
    of quantities u and up, i.e. a coordinate-momentum pair,
    for each slice separately. To calculate the emittance per
    slice, one needs the mean values of quantities u and up
    for each slice. """
    cdef unsigned int n_part_in_cuts = particles_within_cuts.shape[0]
    cdef unsigned int n_slices = epsn_u.shape[0]
    cdef unsigned int p_idx, s_idx, i

    # Determine mean values of u and up for each slice.
    cdef double[::1] mean_u = np.zeros(n_slices, dtype=np.double)
    cdef double[::1] mean_up = np.zeros(n_slices, dtype=np.double)
    mean_per_slice(slice_index_of_particle, particles_within_cuts,
                   n_macroparticles, u, mean_u)
    mean_per_slice(slice_index_of_particle, particles_within_cuts,
                   n_macroparticles, up, mean_up)

    cdef double du, dup
    cdef double[::1] u2 = np.zeros(n_slices, dtype=np.double)
    cdef double[::1] up2 = np.zeros(n_slices, dtype=np.double)
    cdef double[::1] uup = np.zeros(n_slices, dtype=np.double)

    for i in xrange(n_part_in_cuts):
        p_idx = particles_within_cuts[i]
        s_idx = slice_index_of_particle[p_idx]

        du = u[p_idx] - mean_u[s_idx]
        dup = up[p_idx] - mean_up[s_idx]

        u2[s_idx] += du * du
        up2[s_idx] += dup * dup
        uup[s_idx] += du * dup

    for i in xrange(n_slices):
        if n_macroparticles[i]:
            u2[i] /= n_macroparticles[i]
            up2[i] /= n_macroparticles[i]
            uup[i] /= n_macroparticles[i]

        epsn_u[i] = cmath.sqrt(u2[i]*up2[i] - uup[i]*uup[i])


@cython.boundscheck(False)
@cython.cdivision(True)
cpdef emittance_per_slice(int[::1] slice_index_of_particle,
                          int[::1] particles_within_cuts,
                          int[::1] n_macroparticles,
                          double[::1] u, double[::1] up, double[::1] dp,
                          double[::1] emittance):
    """ Iterate once through all the particles within the
    slicing region and calculate simultaneously the emittance
    of quantities u and up, i.e. a coordinate-momentum pair,
    for each slice separately. To calculate the emittance per
    slice, one needs the mean values of quantities u and up
    for each slice.
    """
    cdef unsigned int n_slices = emittance.shape[0]
    # allocate arrays for covariances
    cdef double[::1] cov_u2 = np.zeros(n_slices, dtype=np.double)
    cdef double[::1] cov_up2 = np.zeros(n_slices, dtype=np.double)
    cdef double[::1] cov_u_up = np.zeros(n_slices, dtype=np.double)
    cdef double[::1] cov_u_dp = np.zeros(n_slices, dtype=np.double)
    cdef double[::1] cov_up_dp = np.zeros(n_slices, dtype=np.double)
    cdef double[::1] cov_dp2 = np.ones(n_slices, dtype=np.double)

    # compute the covariances
    cov_per_slice(slice_index_of_particle, particles_within_cuts,
                  n_macroparticles, u, u, cov_u2)
    cov_per_slice(slice_index_of_particle, particles_within_cuts,
                  n_macroparticles, u, up, cov_u_up)
    cov_per_slice(slice_index_of_particle, particles_within_cuts,
                  n_macroparticles, up, up, cov_up2)
    if dp != None:
        cov_per_slice(slice_index_of_particle, particles_within_cuts,
                      n_macroparticles, u, dp, cov_u_dp)
        cov_per_slice(slice_index_of_particle, particles_within_cuts,
                      n_macroparticles, up, dp, cov_up_dp)
        cov_per_slice(slice_index_of_particle, particles_within_cuts,
                      n_macroparticles, dp, dp, cov_dp2)

    cdef double sigma11, sigma12, sigma22

    cdef unsigned int i
    for i in xrange(n_slices):
        if n_macroparticles[i] > 1:
            sigma11 = cov_u2[i] - cov_u_dp[i]*cov_u_dp[i]/cov_dp2[i]
            sigma12 = cov_u_up[i] - cov_u_dp[i]*cov_up_dp[i]/cov_dp2[i]
            sigma22 = cov_up2[i] - cov_up_dp[i]*cov_up_dp[i]/cov_dp2[i]
            emittance[i] = cmath.sqrt(_det_beam_matrix(sigma11, sigma12, sigma22))
        else:
            emittance[i] = 0.

@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
cpdef calc_cell_stats(
        double[::1] x, double[::1] xp, double[::1] y,
        double[::1] yp, double[::1] z, double[::1] dp,
        double beta_z, double radial_cut,
        int n_rings, int n_azim_slices):

    # Prepare arrays to store cell statistics.
    cdef int[:,::1] n_particles_cell = np.zeros((n_azim_slices, n_rings),
                                                dtype=np.int32)
    cdef double[:,::1] mean_x_cell = np.zeros((n_azim_slices, n_rings),
                                              dtype=np.double)
    cdef double[:,::1] mean_xp_cell = np.zeros((n_azim_slices, n_rings),
                                               dtype=np.double)
    cdef double[:,::1] mean_y_cell = np.zeros((n_azim_slices, n_rings),
                                              dtype=np.double)
    cdef double[:,::1] mean_yp_cell = np.zeros((n_azim_slices, n_rings),
                                               dtype=np.double)
    cdef double[:,::1] mean_z_cell = np.zeros((n_azim_slices, n_rings),
                                              dtype=np.double)
    cdef double[:,::1] mean_dp_cell = np.zeros((n_azim_slices, n_rings),
                                               dtype=np.double)

    # Declare datatypes of bunch coords.
    # cdef double[::1] x = bunch.x
    # cdef double[::1] xp = bunch.xp
    # cdef double[::1] y = bunch.y
    # cdef double[::1] yp = bunch.yp
    # cdef double[::1] z = bunch.z
    # cdef double[::1] dp = bunch.dp
    cdef unsigned int n_particles = x.shape[0]

    cdef double ring_width = radial_cut / <double>n_rings
    cdef double azim_width = 2.*cmath.M_PI / <double>n_azim_slices
    cdef double beta_z_square = beta_z*beta_z

    cdef double z_i, dp_i, long_action
    cdef unsigned int p_idx
    cdef int ring_idx, azim_idx
    for p_idx in xrange(n_particles):
        z_i = z[p_idx]
        dp_i = dp[p_idx]

        # Slice radially.
        long_action = cmath.sqrt(z_i*z_i + beta_z_square*dp_i*dp_i)
        ring_idx = <int>cmath.floor(long_action / ring_width)
        if ring_idx >= n_rings:
            continue

        # Slice azimuthally: atan 2 returns values in the interval [-pi, pi];
        # hence, in order to avoid negative indices, we need to add an offset of
        # +pi before performing the floor division (this consequently just adds
        # an offset to the slices indices). The one-to-one mapping between
        # angles and slice indices is retained as desired. In this
        # interpretation, slice index 0 corresponds to -pi (i.e. starting in 3rd
        # quadrant) and slice n-1 correspoinds to +pi (i.e. ending in 2nd
        # quadrant). This needs to be taken into account when interpreting and
        # plotting cell monitor data - for this, use
        # theta = np.linspace(-np.pi, np.pi, n_azim_slices)
        azim_idx = <int>cmath.floor(
            (cmath.M_PI + cmath.atan2(beta_z*dp_i, z_i)) / azim_width)

        n_particles_cell[azim_idx, ring_idx] += 1
        mean_x_cell[azim_idx, ring_idx] += x[p_idx]
        mean_xp_cell[azim_idx, ring_idx] += xp[p_idx]
        mean_y_cell[azim_idx, ring_idx] += y[p_idx]
        mean_yp_cell[azim_idx, ring_idx] += yp[p_idx]
        mean_z_cell[azim_idx, ring_idx] += z_i
        mean_dp_cell[azim_idx, ring_idx] += dp_i

    for azim_idx in xrange(n_azim_slices):
        for ring_idx in xrange(n_rings):
            if n_particles_cell[azim_idx, ring_idx] > 0:
                mean_x_cell[azim_idx, ring_idx] /= n_particles_cell[azim_idx, ring_idx]
                mean_xp_cell[azim_idx, ring_idx] /= n_particles_cell[azim_idx, ring_idx]
                mean_y_cell[azim_idx, ring_idx] /= n_particles_cell[azim_idx, ring_idx]
                mean_yp_cell[azim_idx, ring_idx] /= n_particles_cell[azim_idx, ring_idx]
                mean_z_cell[azim_idx, ring_idx] /= n_particles_cell[azim_idx, ring_idx]
                mean_dp_cell[azim_idx, ring_idx] /= n_particles_cell[azim_idx, ring_idx]

    return n_particles_cell, mean_x_cell, mean_xp_cell, mean_y_cell, mean_yp_cell, mean_z_cell, mean_dp_cell
