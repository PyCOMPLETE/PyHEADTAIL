from __future__ import division
'''
Aperture module to manage particle losses. An aperture is
defined as a condition on the phase space coordinates. Particles
not fulfilling this condition are tagged as lost and are removed
from the beam. Parts of this module are implemented in cython for
better performance.

@date: Created on 04.12.2014
@author: Hannes Bartosik, Giovanni Iadarola, Kevin Li, Michael Schenk

TODO:
  - Store the information on a lost particle in the corresponding
    aperture element instance, incl. time / turn number.
'''
cimport cython
import numpy as np
cimport numpy as np

import aperture

class Aperture(aperture.Aperture):
    '''Pendant to aperture.Aperture with the relocate algorithm
    implemented in cython for more efficiency.
    '''
    @staticmethod
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def relocate_lost_particles(beam, int[::1] alive):
        ''' Memory efficient (and fast) cython function to relocate
        particles marked as lost to the end of the beam.u arrays (u = x, y,
        z, ...). Returns the number of alive particles n_alive_post after
        considering the losses.

        Precondition:
        - At least one particle must be tagged as alive, otherwise bad things
          might happen...

        Description of the algorithm:
        (1) Starting from the end of the numpy array 'alive', find the index
            of the last particle in the array which is still alive. Store its
            array index in last_alive.
        (2) Loop through the 'alive' array from there (continuing in reverse
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
        cdef int[::1] id = beam.id

        # Temporary variables for swapping entries.
        cdef double t_x, t_xp, t_y, t_yp, t_z, t_dp
        cdef int t_alive, t_id

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

class RectangularApertureX(Aperture):
    ''' Mark particles with transverse spatial coord (x) outside the
    interval (x_high, x_low) as lost and remove them from the beam.
    '''

    def __init__(self, x_low, x_high, *args, **kwargs):
        ''' The arguments x_low and x_high define the interval of
        horizontal spatial coordinates for which particles pass through
        the rectangular horizontal aperture. '''
        self.x_low = x_low
        self.x_high = x_high
        super(RectangularApertureX, self).__init__(*args, **kwargs)

    def tag_lost_particles(self, beam):
        ''' This method is called by Aperture.track(beam) to identify
        particles not passing through the aperture. The aperture condition
        on the phase space coordinates is defined by the given Aperture
        element. Returns a np.int32 array 'alive' which contains the
        information on whether a particle is lost (0) or not (1).
        '''
        return cytag_lost_rectangular(beam.x, self.x_low, self.x_high)


class RectangularApertureY(Aperture):
    ''' Mark particles with transverse spatial coord (y) outside the
    interval (y_high, y_low) as lost and remove them from the beam.
    '''

    def __init__(self, y_low, y_high, *args, **kwargs):
        ''' The arguments y_low and y_high define the interval of
        vertical spatial coordinates for which particles pass through
        the rectangular vertical aperture. '''
        self.y_low = y_low
        self.y_high = y_high
        super(RectangularApertureY, self).__init__(*args, **kwargs)

    def tag_lost_particles(self, beam):
        ''' This method is called by Aperture.track(beam) to identify
        particles not passing through the aperture. The aperture condition
        on the phase space coordinates is defined by the given Aperture
        element. Returns a np.int32 array 'alive' which contains the
        information on whether a particle is lost (0) or not (1).
        '''
        return cytag_lost_rectangular(beam.y, self.y_low, self.y_high)


class RectangularApertureZ(Aperture):
    ''' Mark particles with longitudinal spatial coord (z) outside the
    interval (z_high, z_low) as lost and remove them from the beam.
    '''

    def __init__(self, z_low, z_high, *args, **kwargs):
        ''' The arguments z_low and z_high define the interval of
        longitudinal spatial coordinates for which particles pass
        through the rectangular longitudinal aperture. '''
        self.z_low = z_low
        self.z_high = z_high
        super(RectangularApertureZ, self).__init__(*args, **kwargs)

    def tag_lost_particles(self, beam):
        ''' This method is called by Aperture.track(beam) to identify
        particles not passing through the aperture. The aperture condition
        on the phase space coordinates is defined by the given Aperture
        element. Returns a np.int32 array 'alive' which contains the
        information on whether a particle is lost (0) or not (1).
        '''
        return cytag_lost_rectangular(beam.z, self.z_low, self.z_high)


class CircularApertureXY(Aperture):
    ''' Mark particles with transverse spatial coords (x, y) outside a
    circle of specified radius, i.e. x**2 + y**2 > radius**2, as lost
    and remove them from the beam. '''

    def __init__(self, radius, *args, **kwargs):
        ''' The argument radius defines the radius of the circular
        (transverse) aperture. '''
        self.radius_square = radius * radius
        super(CircularApertureXY, self).__init__(*args, **kwargs)

    def tag_lost_particles(self, beam):
        ''' This method is called by Aperture.track(beam) to identify
        particles not passing through the aperture. The aperture condition
        on the phase space coordinates is defined by the given Aperture
        element. Returns a np.int32 array 'alive' which contains the
        information on whether a particle is lost (0) or not (1).
        '''
        return cytag_lost_circular(beam.x, beam.y, self.radius_square)


class EllipticalApertureXY(Aperture):
    ''' Mark particles with transverse spatial coords (x, y) outside a
    ellipse of specified radius, i.e. (x/x_aper)**2 + (y/y_aper)**2 > 1.,
    as lost and remove them from the beam. '''

    def __init__(self, x_aper, y_aper, *args, **kwargs):

        self.x_aper = x_aper
        self.y_aper = y_aper
        super(EllipticalApertureXY, self).__init__(*args, **kwargs)

    def tag_lost_particles(self, beam):
        ''' This method is called by Aperture.track(beam) to identify
        particles not passing through the aperture. The aperture condition
        on the phase space coordinates is defined by the given Aperture
        element. Returns a np.int32 array 'alive' which contains the
        information on whether a particle is lost (0) or not (1).
        '''
        return cytag_lost_ellipse(beam.x, beam.y,
                                  self.x_aper, self.y_aper)


''' Cython functions for fast id and tagging of lost particles. '''

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef cytag_lost_rectangular(double[::1] u,
                             double low_lim, double high_lim):
    ''' Cython function for fast identification and tagging of particles
    lost at a rectangular aperture element, i.e. it tags particles with
    a spatial coord u (beam.x, beam.y or beam.z) lying outside the
    interval (low_lim, high_lim) as lost. Returns a np array 'alive'
    containing the information of alive / lost for each particle in the
    beam after the aperture. '''
    cdef int n = u.shape[0]
    cdef int[::1] alive = np.ones(n, dtype=np.int32)

    cdef int i
    for i in xrange(n):
        if u[i] < low_lim or u[i] > high_lim:
            alive[i] = 0

    return alive

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef cytag_lost_circular(double[::1] u, double[::1] v,
                          double radius_square):
    ''' Cython function for fast identification and tagging of particles
    lost at a circular transverse aperture element of a given radius,
    i.e. it tags particles with spatial coords u, v (usually (beam.x,
    beam.y)) fulfilling u**2 + v**2 > radius_square as lost. Returns a
    np array 'alive' containing the information of alive / lost for
    each particle in the beam after the aperture. '''
    cdef int n = u.shape[0]
    cdef int[::1] alive = np.ones(n, dtype=np.int32)

    cdef int i
    for i in xrange(n):
        if (u[i]*u[i] + v[i]*v[i]) > radius_square:
            alive[i] = 0

    return alive

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef cytag_lost_ellipse(double[::1] u, double[::1] v,
                         double u_aper, double v_aper):
    ''' Cython function for fast identification and tagging of particles
    lost at a elliptical transverse aperture element. Returns a
    np array 'alive' containing the information of alive / lost for
    each particle in the beam after the aperture. '''
    cdef int n = u.shape[0]
    cdef int[::1] alive = np.ones(n, dtype=np.int32)
    cdef double u_aper_sq_rec = 1./(u_aper*u_aper)
    cdef double v_aper_sq_rec = 1./(v_aper*v_aper)

    cdef int i
    for i in xrange(n):
        if (u[i]*u[i]*u_aper_sq_rec + v[i]*v[i]*v_aper_sq_rec) > 1.:
            alive[i] = 0

    return alive
