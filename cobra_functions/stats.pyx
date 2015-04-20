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


@cython.boundscheck(False)
@cython.cdivision(True)
cpdef double mean(double[::1] u):
    """ Cython function to calculate the mean value of dataset u. """
    cdef double mean_u = 0
    cdef unsigned int n = u.shape[0]

    cdef unsigned int i
    for i in xrange(n):
        mean_u += u[i]
    if n:
        mean_u /= n

    return mean_u

@cython.boundscheck(False)
@cython.cdivision(True)
cpdef double std(double[::1] u):
    """ Cython function to calculate the standard deviation of
    dataset u. The dataset must consist of at least 2 samples
    """
    cdef double mean_u = mean(u)
    cdef double std_u = 0
    cdef double du = 0

    cdef unsigned int n = u.shape[0]
    cdef unsigned int i
    for i in xrange(n):
        du = u[i] - mean_u
        std_u += du * du
    if n > 1:
        std_u /= (n-1) # divide by n-1: unbiased estimator for var

    return cmath.sqrt(std_u)

@cython.boundscheck(False)
@cython.cdivision(True)
cpdef double cov(double[::1] a, double[::1] b):
    """ Cython function which calculates the covariance
    (not the covariance matrix!) of two data sets
    a and b using a two pass algorithm (computing the means first)
    definition: cov(a,b) = E[(A - E[A])*(B - E[B])]
    implementation: cov(a,b) = 1/(n-1) \sum_{i=1}^n (a_i -<a>)*(b_i -<b>)
    a and b do not necessarily have to reference different data -> var
    Args:
        a: numpy array, a.shape[0] defines n. n must be > 1
        b: numpy array, at least with length a.shape[0]
    """
    cdef double mean_a = mean(a)
    cdef double mean_b = mean(b)

    cdef double cov = 0.
    cdef unsigned int n = a.shape[0]
    if n < 2:
        return 0.
    cdef unsigned int i
    for i in xrange(n):
        cov += (a[i] - mean_a)*(b[i] - mean_b)/(n-1)
    return cov

@cython.boundscheck(False)
@cython.cdivision(True)
cpdef double cov_onepass(double[::1] a, double[::1] b):
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
cpdef double emittance_old(double[::1] u, double[::1] up):
    """ Cython function to calculate the effective (neglecting dispersion)
    emittance of datasets u and up, i.e. a coordinate-momentum pair.
    To calculate the emittance, one needs the mean values of quantities u and
    up. """
    cdef double mean_u = mean(u)
    cdef double mean_up = mean(up)

    cdef double u2 = 0
    cdef double up2 = 0
    cdef double uup = 0
    cdef double du = 0
    cdef double dup = 0

    cdef unsigned int n = u.shape[0]
    cdef unsigned int i
    for i in xrange(n):
        du = u[i] - mean_u
        dup = up[i] - mean_up

        u2 += du * du
        up2 += dup * dup
        uup += du * dup
    if n:
        u2 /= n
        up2 /= n
        uup /= n

    return cmath.sqrt(u2*up2 - uup*uup)

@cython.boundscheck(False)
@cython.cdivision(True)
cpdef double dispersion(double[::1] u, double[::1] dp):
    """Cython function to compute the statistial dispersion:
    disp = <x*dp>/<dp**2>
    Args:
        u a coordinate array, typically x or y spatial coordinates
          it is also possible to pass xp or yp
    """
    cdef double mean_u_dp = mean(np.multiply(u, dp))
    cdef double mean_dp2 = mean(np.multiply(dp, dp))
    if mean_dp2 > 0: # can never be smaller than 0
        return mean_u_dp / mean_dp2
    else:
        return 0

@cython.boundscheck(False)
@cython.cdivision(True)
cdef double _det_beam_matrix(double u2, double u_up, double up2, double disp_u,
                             double disp_up, double mean_dp2):
    """Function which computes the determinant of the 2x2 beam matrix as defined
    in F.Loehl 2005
    Args:
        u2: cov(u,u)
        u_up: cov(u,up)
        up2: cov(up, up)
        disp_u: (statistical) dispersion of u
        disp_up: (statistical) dispersion uf up
        mean_dp2: <dp*dp>
    """
    return (((u2 - disp_u * disp_u * mean_dp2)
            *(up2 - disp_up * disp_up * mean_dp2))
            - (u_up - disp_u * disp_up * mean_dp2)
             *(u_up - disp_u * disp_up * mean_dp2))

@cython.boundscheck(False)
@cython.cdivision(True)
cpdef double effective_emittance(double[::1] u, double[::1] up):
    """Cython function to calculate the effective emittance
    (assuming zero dispersion) of the beam specified by the spatial and
    momentum coordinates u, up
    """
    covariance = cov_onepass
    cdef double cov_u2 = covariance(u,u)
    cdef double cov_u_up = covariance(u, up)
    cdef double cov_up2 = covariance(up, up)
    cdef double disp_u = 0.
    cdef double disp_up = 0.
    cdef double mean_dp2 = 0.
    cdef double result = _det_beam_matrix(cov_u2, cov_u_up, cov_up2, disp_u,
                                          disp_up, mean_dp2)
    return cmath.sqrt(result)



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

    covariance = cov_onepass
    cdef double cov_u2 = covariance(u, u)
    cdef double cov_u_up = covariance(u, up)
    cdef double cov_up2 = covariance(up, up)
    cdef double disp_u = 0.
    cdef double disp_up = 0.
    cdef double mean_dp2 = 0.
    if dp != None: # 'if dp:' doesn't work
        disp_u = dispersion(u, dp)
        disp_up = dispersion(up, dp)
        mean_dp2 = mean(np.multiply(dp, dp))

    # this can be optimized by not doing disp_u*disp_u*mean_dp2
    # but mean(u*dp)*mean(u*dp)*mean_dp2 inside of this function directly
    # currently mean_dp2 is computed here and in dispersion()
    cdef double result = _det_beam_matrix(cov_u2, cov_u_up, cov_up2, disp_u,
                                          disp_up, mean_dp2)
    return cmath.sqrt(result)


@cython.boundscheck(False)
@cython.cdivision(True)
cpdef double get_alpha(double[::1] u, double[::1] up, double[::1] dp):
    """Cython function to calculate the statistical alpha (Twiss) of
    the beam specified by the spatial coordinate u, momentum up and
    dp=(p-p0)/p0. Not optimized yet
    """
    covariance = cov_onepass
    cdef double cov_u_up = covariance(u, up)
    cdef double disp_u = dispersion(u, dp)
    cdef double disp_up = dispersion(up, dp)
    cdef double mean_dp2 = mean(np.multiply(dp, dp))
    return -(cov_u_up - disp_u*disp_up*mean_dp2) / emittance(u, up, dp)
    

@cython.boundscheck(False)
@cython.cdivision(True)
cpdef double get_alpha_effective(double[::1] u, double[::1] up):
    """Cython function to calculate the statistical effective alpha (Twiss) of
    the beam specified by the spatial coordinate u, momentum up
    Not optimized yet. Effective means dispersion is assumed to
    be 0.
    """
    covariance = cov_onepass
    cdef double cov_u_up = covariance(u,up)
    return -(cov_u_up) / effective_emittance(u, up)


@cython.boundscheck(False)
@cython.cdivision(True)
cpdef double get_beta(double[::1] u, double[::1] up, double[::1] dp):
    """Cython function to calculate the statistical beta (Twiss) of the
    beam specified by the spatial coordinate u, momentum up and
    dp = (p-p0)/p0. Not optimized yet
    """
    covariance = cov_onepass
    cdef double cov_u2 = covariance(u, u)
    cdef double disp_u = dispersion(u, dp)
    cdef double mean_dp2 = mean(np.multiply(dp, dp))
    return (cov_u2 - disp_u*disp_u*mean_dp2) / emittance(u, up, dp)

@cython.boundscheck(False)
@cython.cdivision(True)
cpdef double get_beta_effective(double[::1] u, double[::1] up):
    """Cython function to compute the effective statistical beta (Twiss) of the
    beam specified by the spatial coordinate u, momentum up
    Not optimized yet. Effective means the dispersion is assumed to be 0.
    """
    covariance = cov_onepass
    cdef double cov_u2 = covariance(u, u)
    return (cov_u2) / effective_emittance(u, up)

@cython.boundscheck(False)
@cython.cdivision(True)
cpdef double get_gamma(double[::1] u, double[::1] up, double[::1] dp):
    """ Cython function to calculate the statistial gamma (Twiss) of the
    beam specified by the spatial coordinate u, momentum up and
    dp = (p-p0)/p0. Not optimized yet
    """
    covariance = cov_onepass
    cdef double cov_up2 = covariance(up, up)
    cdef double disp_up = dispersion(up, dp)
    cdef double mean_dp2 = mean(np.multiply(dp, dp))
    return (cov_up2 - disp_up*disp_up*mean_dp2) / emittance(u, up, dp)

@cython.boundscheck(False)
@cython.cdivision(True)
cpdef double get_gamma_effective(double[::1] u, double[::1] up):
    """ Cython function to compute the effective statistical gamma (Twiss) of
    the beam specified by the spatial coordinate u, momentum up 
    Not optimized yet. effective means the dispersion is assumed to be 0
    """
    covariance = cov_onepass
    cdef double cov_up2 = covariance(up, up)
    return (cov_up2) / effective_emittance(u, up)




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
        if n_macroparticles[i]:
            std_u[i] /= n_macroparticles[i]

        std_u[i] = cmath.sqrt(std_u[i])

@cython.boundscheck(False)
@cython.cdivision(True)
cpdef cov_per_slice(int[::1] slice_index_of_particle,
                    int[::1] particles_within_cuts,
                    int[::1] n_macroparticles,
                    double[::1] a, double[::1] b, double[::1] cov_ab):
    """Cov per slice. Cannot make use of cov() because the particles
    per slice are not contiguous in memory 
    The result gets !added! to the cov_ab array"""
    #TODO: write single pass version of this algorithm
    cdef unsigned int n_part_in_cuts = particles_within_cuts.shape[0]
    cdef unsigned int n_slices = cov_ab.shape[0]
    cdef unsigned int p_idx, s_idx, i
    cdef double du

    cdef double[::1] mean_a = np.zeros(n_slices, dtype=np.double)
    cdef double[::1] mean_b = np.zeros(n_slices, dtype=np.double)

    mean_per_slice(slice_index_of_particle, particles_within_cuts,
                   n_macroparticles, a, mean_a)
    mean_per_slice(slice_index_of_particle, particles_within_cuts,
                   n_macroparticles, b, mean_b)

    for i in xrange(n_part_in_cuts):
        p_idx = particles_within_cuts[i]
        s_idx = slice_index_of_particle[p_idx]
        cov_ab[s_idx] += (a[p_idx] - mean_a[s_idx])*(b[p_idx] - mean_b[s_idx])

    for i in xrange(n_slices):
        if n_macroparticles[i]:
            cov_ab[i] /= n_macroparticles[i]
        cov_ab[i] = cmath.sqrt(cov_ab[i])
 

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
cpdef dispersion_per_slice(int[::1] slice_index_of_particle,
                           int[::1] particles_within_cuts,
                           int[::1] n_macroparticles,
                           double[::1] u, double[::1] dp, double[::1] disp):
    """ Compute the dispersion per slice via mean_per_slice
    The result gets stored in the disp array
    """
    cdef unsigned int n_slices = disp.shape[0]
    cdef double[::1] mean_u_dp = np.zeros(n_slices, dtype=np.double)
    cdef double[::1] mean_dp2 = np.zeros(n_slices, dtype=np.double)
    mean_per_slice(slice_index_of_particle, particles_within_cuts,
                   n_macroparticles, np.multiply(u, dp), mean_u_dp)
    mean_per_slice(slice_index_of_particle, particles_within_cuts,
                   n_macroparticles, np.multiply(dp, dp), mean_dp2)

    cdef unsigned int i
    for i in xrange(n_slices):
        if mean_dp2[i] > 0:
            disp[i] = mean_u_dp[i] / mean_dp2[i]
        else:
            disp[i] = 0.

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
    for each slice. """
    #TODO: Clean up, optimize (time & space) if necessary
    cdef unsigned int n_slices = emittance.shape[0]
    # allocate arrays for covariances, means and dispersions 
    cdef double[::1] u2 = np.zeros(n_slices, dtype=np.double)
    cdef double[::1] up2 = np.zeros(n_slices, dtype=np.double)
    cdef double[::1] uup = np.zeros(n_slices, dtype=np.double)
    cdef double[::1] disp_u = np.zeros(n_slices, dtype=np.double)
    cdef double[::1] disp_up = np.zeros(n_slices, dtype=np.double)
    cdef double[::1] mean_dp2 = np.zeros(n_slices, dtype=np.double)

    # compute the covariances
    cov_per_slice(slice_index_of_particle, particles_within_cuts,
                  n_macroparticles, u, u, u2)
    cov_per_slice(slice_index_of_particle, particles_within_cuts,
                  n_macroparticles, u, up, uup)
    cov_per_slice(slice_index_of_particle, particles_within_cuts,
                  n_macroparticles, up, up, up2)
    if dp != None:
        # compute the necessary params (mean of dp2, dispersions)
        mean_per_slice(slice_index_of_particle, particles_within_cuts,
                       n_macroparticles, np.multiply(dp,dp), mean_dp2)
        dispersion_per_slice(slice_index_of_particle, particles_within_cuts,
                             n_macroparticles, u, dp, disp_u)
        dispersion_per_slice(slice_index_of_particle, particles_within_cuts,
                             n_macroparticles, up, dp, disp_up)
    # compute the emittance per slice
    cdef unsigned int i
    for i in xrange(n_slices):
        emittance[i] = cmath.sqrt(_det_beam_matrix(u2[i], uup[i], up2[i],
                                  disp_u[i], disp_up[i], mean_dp2[i]))

