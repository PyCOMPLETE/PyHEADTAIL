#!python
#cython: boundscheck=False

'''
Created on 31.01.2014

@author: Kevin Li, Michael Schenk
'''

import numpy as np
cimport numpy as np
cimport libc.math as cmath


cpdef double mean(double[::1] u):

    cdef double mean_u = 0
    cdef unsigned int n = u.shape[0]
    cdef unsigned int i
    for i in xrange(n):
        mean_u += u[i]
    if n:
        mean_u /= n

    return mean_u


cpdef double std(double[::1] u):

    cdef double mean_u = mean(u)
    cdef double std_u = 0
    cdef double du = 0

    cdef unsigned int n = u.shape[0]
    cdef unsigned int i
    for i in xrange(n):
        du = u[i] - mean_u
        std_u += du * du
    if n:
        std_u /= n

    return cmath.sqrt(std_u)


cpdef double emittance(double[::1] u, double[::1] v):

    cdef double mean_u = mean(u)
    cdef double mean_v = mean(v)

    cdef double u2 = 0
    cdef double v2 = 0
    cdef double uv = 0
    cdef double du = 0
    cdef double dv = 0

    cdef unsigned int n = u.shape[0]
    cdef unsigned int i
    for i in xrange(n):
        du = u[i] - mean_u
        dv = v[i] - mean_v

        u2 += du * du
        v2 += dv * dv
        uv += du * dv
    if n:
        u2 /= n
        v2 /= n
        uv /= n

    return cmath.sqrt(u2 * v2 - uv * uv)


# Statistics for SliceSet

cpdef count_macroparticles_per_slice(int[::1] slice_index_of_particle,
                                     int[::1] particles_within_cuts,
                                     int[::1] n_macroparticles):

    cdef unsigned int n_particles_within_cuts = particles_within_cuts.shape[0]
    cdef unsigned int s_idx, i

    for i in xrange(n_particles_within_cuts):
        s_idx = slice_index_of_particle[particles_within_cuts[i]]
        n_macroparticles[s_idx] += 1


cpdef sort_particle_indices_by_slice(int[::1] slice_index_of_particle,
                                     int[::1] particles_within_cuts,
                                     int[::1] slice_positions,
                                     int[::1] particle_indices_by_slice):

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


cpdef mean_per_slice(int[::1] slice_index_of_particle,
                     int[::1] particles_within_cuts,
                     int[::1] n_macroparticles,
                     double[::1] u, double[::1] mean_u):

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


cpdef std_per_slice(int[::1] slice_index_of_particle,
                    int[::1] particles_within_cuts,
                    int[::1] n_macroparticles,
                    double[::1] u, double[::1] std_u):

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


cpdef emittance_per_slice(int[::1] slice_index_of_particle,
                          int[::1] particles_within_cuts,
                          int[::1] n_macroparticles,
                          double[::1] u, double[::1] up, double[::1] epsn_u):

    cdef unsigned int n_part_in_cuts = particles_within_cuts.shape[0]
    cdef unsigned int n_slices = epsn_u.shape[0]
    cdef unsigned int p_idx, s_idx, i

    cdef double[::1] mean_u  = np.zeros(n_slices, dtype=np.double)
    cdef double[::1] mean_up = np.zeros(n_slices, dtype=np.double)
    mean_per_slice(slice_index_of_particle, particles_within_cuts,
                   n_macroparticles, u, mean_u)
    mean_per_slice(slice_index_of_particle, particles_within_cuts,
                   n_macroparticles, up, mean_up)

    cdef double du, dup
    cdef double[::1] u2  = np.zeros(n_slices, dtype=np.double)
    cdef double[::1] up2 = np.zeros(n_slices, dtype=np.double)
    cdef double[::1] uup = np.zeros(n_slices, dtype=np.double)

    for i in xrange(n_part_in_cuts):
        p_idx = particles_within_cuts[i]
        s_idx = slice_index_of_particle[p_idx]

        du  = u[p_idx] - mean_u[s_idx]
        dup = up[p_idx] - mean_up[s_idx]

        u2[s_idx]  += du * du
        up2[s_idx] += dup * dup
        uup[s_idx] += du * dup

    for i in xrange(n_slices):
        if n_macroparticles[i]:
            u2[i]  /= n_macroparticles[i]
            up2[i] /= n_macroparticles[i]
            uup[i] /= n_macroparticles[i]

        epsn_u[i] = cmath.sqrt(u2[i] * up2[i] - uup[i] * uup[i])
