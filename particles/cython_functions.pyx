'''
@date 01.12.2014
@author Michael Schenk

TODO:
 - Find fast sorting algorithm in C.
 - Possible to enable vectorization?!
'''
cimport cython
import numpy as np
cimport numpy as np

@cython.boundscheck(False)
@cython.wraparound(False)
def relocate_lost_particles(beam):
    ''' Memory efficient (and fast) cython function to relocate
    particles marked as lost to the end of the beam.u arrays (u = x, y,
    z, ...). Returns the number of alive particles n_alive_post after
    removal of those marked as lost.

    Description of the algorithm:
    (1) Starting from the end of the numpy array view beam.alive, find
        the index of the last particle in the array which is still
        alive. Store its array index in last_alive.
    (2) Loop through the alive array from there (continuing in reverse
        order). If a particle i is found for which alive[i] == 0, i.e.
        it is a lost one, swap its position (and data x, y, z, ...) with
        the one located at index last_alive.
    (3) Move last_alive by -1. Due to the chosen procedure, the particle
        located at the new last_alive index is known to be alive.
    (4) Repeat steps (2) and (3) until index i = 0 is reached.
    '''
    cdef double[::1] x = beam.x
    cdef double[::1] y = beam.y
    cdef double[::1] z = beam.z
    cdef double[::1] xp = beam.xp
    cdef double[::1] yp = beam.yp
    cdef double[::1] dp = beam.dp
    cdef unsigned int[::1] id = beam.id
    cdef unsigned int[::1] alive = beam.alive

    # Temporary variables for swapping entries.
    cdef double t_x, t_xp, t_y, t_yp, t_z, t_dp
    cdef unsigned int t_alive, t_id

    # Find last_alive index.
    cdef int n_alive_pri = alive.shape[0]
    cdef int last_alive = n_alive_pri - 1
    while not alive[last_alive]:
        last_alive -= 1

    # Identify particles marked as lost and relocate them.
    cdef int n_alive_post = last_alive + 1
    cdef int i
    for i in xrange(last_alive-1, -1, -1):
        if not alive[i]:
            # Swap lost particle coords with last_alive.
            t_x, t_y, t_z = x[i], y[i], z[i]
            t_xp, t_yp, t_dp = xp[i], yp[i], dp[i]
            t_id, t_alive = id[i], alive[i]

            x[i], y[i], z[i] = x[last_alive], y[last_alive], z[last_alive]
            xp[i], yp[i], dp[i] = xp[last_alive], yp[last_alive], dp[last_alive]
            id[i], alive[i] = id[last_alive], alive[last_alive]

            x[last_alive], y[last_alive], z[last_alive] = t_x, t_y, t_z
            xp[last_alive], yp[last_alive], dp[last_alive] = t_xp, t_yp, t_dp
            id[last_alive], alive[last_alive] = t_id, t_alive

            # Move last_alive pointer and update number of alive
            # particles.
            last_alive -= 1
            n_alive_post -= 1

    return n_alive_post
