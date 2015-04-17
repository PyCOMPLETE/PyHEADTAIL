"""
@author Kevin Li, Michael Schenk
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
cpdef double emittance(double[::1] u, double[::1] up):
    """ Cython function to calculate the emittance of datasets
    u and up, i.e. a coordinate-momentum pair. To calculate the
    emittance, one needs the mean values of quantities u and
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
cpdef emittance_per_slice(int[::1] slice_index_of_particle,
                          int[::1] particles_within_cuts,
                          int[::1] n_macroparticles,
                          double[::1] u, double[::1] up, double[::1] epsn_u):
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
@cython.wraparound(False)
cpdef calc_cell_stats(bunch, double beta_z, double radial_cut,
                      int n_rings, int n_azim_slices):

    # Prepare arrays to store cell statistics.
    cdef int[:,::1] n_particles_cell = np.zeros((n_azim_slices, n_rings),
                                               dtype=np.int32)
    cdef double[:,::1] mean_x_cell = np.zeros((n_azim_slices, n_rings),
                                               dtype=np.double)
    cdef double[:,::1] mean_y_cell = np.zeros((n_azim_slices, n_rings),
                                               dtype=np.double)
    cdef double[:,::1] mean_z_cell = np.zeros((n_azim_slices, n_rings),
                                               dtype=np.double)
    cdef double[:,::1] mean_dp_cell = np.zeros((n_azim_slices, n_rings),
                                                dtype=np.double)

    # Declare datatypes of bunch coords.
    cdef double[::1] x = bunch.x
    cdef double[::1] y = bunch.y
    cdef double[::1] z = bunch.z
    cdef double[::1] dp = bunch.dp
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

        # Slice azimuthally.
        if (z_i > 0. and dp_i > 0.):
            azim_idx = <int>cmath.floor(
                cmath.atan(beta_z*dp_i / z_i) / azim_width)
        elif (z_i < 0. and dp_i > 0.):
            azim_idx = <int>cmath.floor(
                (cmath.M_PI - cmath.atan(-beta_z*dp_i / z_i)) / azim_width)
        elif (z_i < 0. and dp_i <= 0.):
            azim_idx = <int>cmath.floor(
                (cmath.M_PI + cmath.atan(beta_z*dp_i / z_i)) / azim_width)
        elif (z_i > 0. and dp_i < 0.):
            azim_idx = <int>cmath.floor(
                (2.*cmath.M_PI - cmath.atan(-beta_z*dp_i / z_i)) / azim_width)

        n_particles_cell[azim_idx, ring_idx] += 1
        mean_x_cell[azim_idx, ring_idx] += x[p_idx]
        mean_y_cell[azim_idx, ring_idx] += y[p_idx]
        mean_z_cell[azim_idx, ring_idx] += z_i
        mean_dp_cell[azim_idx, ring_idx] += dp_i

    for azim_idx in xrange(n_azim_slices):
        for ring_idx in xrange(n_rings):
            if n_particles_cell[azim_idx, ring_idx] > 0:
                mean_x_cell[azim_idx, ring_idx] /= n_particles_cell[azim_idx, ring_idx]
                mean_y_cell[azim_idx, ring_idx] /= n_particles_cell[azim_idx, ring_idx]
                mean_z_cell[azim_idx, ring_idx] /= n_particles_cell[azim_idx, ring_idx]
                mean_dp_cell[azim_idx, ring_idx] /= n_particles_cell[azim_idx, ring_idx]

    return n_particles_cell, mean_x_cell, mean_y_cell, mean_z_cell, mean_dp_cell
