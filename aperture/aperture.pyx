from __future__ import division
'''
@date: Created on 04.12.2014
@author: Hannes Bartosik, Michael Schenk

TODO:
 - Speed up RFBucketAperture. It is currently EXTREMELY slow. Use
   RectangularApertureZ as an alternative.
'''
cimport cython
import numpy as np
cimport numpy as np

from . import Element

from abc import ABCMeta, abstractmethod

class Aperture(Element):
    ''' Abstract base class for Aperture elements. An aperture is
    generally defined as a condition on the phase space coordinates.
    Particles not fulfilling this condition are tagged as lost. They
    will be removed from the tracked particles as soon as the
    Particles.update_losses() method is called. '''
    __metaclass__ = ABCMeta

    def __init__(self, *args, **kwargs):
        ''' The boolean apply_losses_here controls whether the
        Particles.update_losses(beam) method is called to relocate lost
        particles to the end of the bunch.u_all arrays (u = x, y, z,
        ...) and to actually remove them from the numpy array views
        beam.u and leave them in an untracked state. In case there are
        several Aperture elements placed at a segment boundary of the
        accelerator ring, apply_losses_here should only be set to True
        for the last one to increase performance. '''
        pass

    def track(self, beam):
        ''' Tag particles not passing through the aperture as lost. If
        there are any losses and the self.apply_losses_here is set to
        True, the method Particles.update_losses() is called to relocate
        lost particles to the end of the beam.u_all numpy arrays (u = x,
        y, z, ...) and to actually remove them from the numpy array
        views beam.u and leave them in an untracked state. Also, clean
        currently cached slice_sets of the bunch as losses change its
        state. '''
        alive = self.tag_lost_particles(beam)

        if not np.all(alive):
            n_alive_post = relocate_lost_particles(beam, alive)

            beam.macroparticlenumber = n_alive_post
            beam.x = beam.x[:n_alive_post]
            beam.y = beam.y[:n_alive_post]
            beam.z = beam.z[:n_alive_post]
            beam.xp = beam.xp[:n_alive_post]
            beam.yp = beam.yp[:n_alive_post]
            beam.dp = beam.dp[:n_alive_post]
            beam.id = beam.id[:n_alive_post]

            beam.clean_slices()

    @abstractmethod
    def tag_lost_particles(self, beam):
        ''' This method is called by Aperture.track(beam) to identify
        particles not passing through the aperture and set their
        bunch.alive state to 0 (false) to mark them as lost. Return
        whether or not any lost particles were found. '''
        pass


class RectangularApertureX(Aperture):
    ''' Mark particles with transverse spatial coord (x) outside the
    interval (x_high, x_low) as lost. '''

    def __init__(self, x_low, x_high, apply_losses_here=True,
                 *args, **kwargs):
        ''' The arguments x_low and x_high define the interval of
        horizontal spatial coordinates for which particles pass through
        the rectangular horizontal aperture. The boolean
        apply_losses_here specifies whether the
        Particles.update_losses(beam) method should be called after
        tagging lost particles to relocate them to the end of the
        bunch.u_all arrays (u = x, y, z, ...), remove them from the
        views bunch.u and leave them in an untracked state. In case
        there are several Aperture elements placed at a segment boundary
        of the accelerator ring, apply_losses_here should only be set to
        True for the last one to increase performance. '''
        self.x_low = x_low
        self.x_high = x_high
        super(RectangularApertureX, self).__init__(apply_losses_here)

    def tag_lost_particles(self, beam):
        ''' This method is called by Aperture.track(beam) to identify
        particles not passing through the aperture and set their
        bunch.alive state to 0 (false) to mark them as lost. The search
        for lost particles is done using a cython function. Return
        whether or not any lost particles were found. '''
        return cytag_lost_rectangular(
            beam.x, self.x_low, self.x_high)


class RectangularApertureY(Aperture):
    ''' Mark particles with transverse spatial coord (y) outside the
    interval (y_high, y_low) as lost. '''

    def __init__(self, y_low, y_high, apply_losses_here=True,
                 *args, **kwargs):
        ''' The arguments y_low and y_high define the interval of
        vertical spatial coordinates for which particles pass through
        the rectangular vertical aperture. The boolean apply_losses_here
        specifies whether the Particles.update_losses(beam) method
        should be called after tagging lost particles to relocate them
        to the end of the bunch.u_all arrays (u = x, y, z, ...), remove
        them from the views bunch.u and leave them in an untracked
        state. In case there are several Aperture elements placed at a
        segment boundary of the accelerator ring, apply_losses_here
        should only be set to True for the last one to increase
        performance. '''
        self.y_low = y_low
        self.y_high = y_high
        super(RectangularApertureY, self).__init__(apply_losses_here)

    def tag_lost_particles(self, beam):
        ''' This method is called by Aperture.track(beam) to identify
        particles not passing through the aperture and set their
        bunch.alive state to 0 (false) to mark them as lost. The search
        for lost particles is done using a cython function. Return
        whether or not any lost particles were found. '''
        return cytag_lost_rectangular(
            beam.y, self.y_low, self.y_high)


class RectangularApertureZ(Aperture):
    ''' Mark particles with longitudinal spatial coord (z) outside the
    interval (z_high, z_low) as lost. '''

    def __init__(self, z_low, z_high, apply_losses_here=True,
                 *args, **kwargs):
        ''' The arguments z_low and z_high define the interval of
        longitudinal spatial coordinates for which particles pass
        through the rectangular longitudinal aperture. The boolean
        apply_losses_here specifies whether the
        Particles.update_losses(beam) method should be called after
        tagging lost particles to relocate them to the end of the
        bunch.u_all arrays (u = x, y, z, ...), remove them from the
        views bunch.u and leave them in an untracked state. In case
        there are several Aperture elements placed at a segment boundary
        of the accelerator ring, apply_losses_here should only be set to
        True for the last one to increase performance. '''
        self.z_low = z_low
        self.z_high = z_high
        super(RectangularApertureZ, self).__init__(apply_losses_here)

    def tag_lost_particles(self, beam):
        ''' This method is called by Aperture.track(beam) to identify
        particles not passing through the aperture and set their
        bunch.alive state to 0 (false) to mark them as lost. The
        search for lost particles is done using a cython function.
        Return whether or not any lost particles were found. '''
        return cytag_lost_rectangular(
            beam.z, self.z_low, self.z_high)


class CircularApertureXY(Aperture):
    ''' Mark particles with transverse spatial coords (x, y) outside a
    circle of specified radius, i.e. x**2 + y**2 > radius**2, as lost.
    '''

    def __init__(self, radius, apply_losses_here=True, *args, **kwargs):
        ''' The argument radius defines the radius of the circular
        (transverse) aperture. The argument apply_losses_here specifies
        whether the Particles.update_losses(beam) method should be
        called after tagging lost particles to relocate them to the end
        of the bunch.u_all arrays (u = x, y, z, ...), remove them from
        the views bunch.u and leave them in an untracked state. In case
        there are several Aperture elements placed at a segment boundary
        of the accelerator ring, apply_losses_here should only be set to
        True for the last one to increase performance. '''
        self.radius_square = radius * radius
        super(CircularApertureXY, self).__init__(apply_losses_here)

    def tag_lost_particles(self, beam):
        ''' This method is called by Aperture.track(beam) to identify
        particles not passing through the aperture and set their
        bunch.alive state to 0 (false) to mark them as lost. The search
        for lost particles is done using a cython function. Return
        whether or not any lost particles were found. '''
        return cytag_lost_circular(
            beam.x, beam.y, beam.alive, self.radius_square)

class EllipticalApertureXY(Aperture):
    ''' Mark particles with transverse spatial coords (x, y) outside a
    ellipse of specified radius, i.e. (x/x_aper)**2 + (y/y_aper)**2 > 1., as lost.
    '''

    def __init__(self, x_aper, y_aper, apply_losses_here=True, *args, **kwargs):
        ''' The argument apply_losses_here specifies
        whether the Particles.update_losses(beam) method should be
        called after tagging lost particles to relocate them to the end
        of the bunch.u_all arrays (u = x, y, z, ...), remove them from
        the views bunch.u and leave them in an untracked state. In case
        there are several Aperture elements placed at a segment boundary
        of the accelerator ring, apply_losses_here should only be set to
        True for the last one to increase performance. '''
        self.x_aper = x_aper
        self.y_aper = y_aper
        super(EllipticalApertureXY, self).__init__(apply_losses_here)

    def tag_lost_particles(self, beam):
        ''' This method is called by Aperture.track(beam) to identify
        particles not passing through the aperture and set their
        bunch.alive state to 0 (false) to mark them as lost. The search
        for lost particles is done using a cython function. Return
        whether or not any lost particles were found. '''
        return cytag_lost_ellipse(
            beam.x, beam.y, beam.alive, self.x_aper, self.y_aper)


class RFBucketAperture(Aperture):
    ''' Mark particles with longitudinal phase space coords (z, dp)
    outside the accepted region as lost.

    NOTE: THE CURRENT IMPLEMENTATION IS EXTREMELY SLOW! For 1e6
    macroparticles, executing it once takes 120ms. Hence, as an
    alternative, one may use the RectangularApertureZ using the z-limits
    of the RFBucket. Like this, particles leaking from the RFBucket will
    at some point be marked lost and finally removed as well. '''

    def __init__(self, is_accepted, apply_losses_here=True,
                 *args, **kwargs):
        ''' The argument is_accepted takes a reference to a function of
        the form is_accepted(z, dp) returning a boolean array saying
        whether the pair (z, dp) is in- or outside the specified region
        of the longitudinal phase space. Usually, is_accepted can be
        generated either using the RFSystems.RFBucket.make_is_accepted
        or using directly RFSystems.RFBucket.is_in_separatrix. The
        argument apply_losses_here specifies whether the
        Particles.update_losses(beam) method should be called after
        tagging lost particles to relocate them to the end of the
        bunch.u_all arrays (u = x, y, z, ...), remove them from the
        views bunch.u and leave them in an untracked state. In case
        there are several Aperture elements placed at a segment boundary
        of the accelerator ring, apply_losses_here should only be set to
        True for the last one to increase performance. '''
        self.is_accepted = is_accepted
        super(RFBucketAperture, self).__init__(apply_losses_here)

    def tag_lost_particles(self, beam):
        ''' This method is called by Aperture.track(beam) to identify
        particles not passing through the aperture and set their
        bunch.alive state to 0 (false) to mark them as lost. The
        search for lost particles is done using a cython function.
        Return whether or not any lost particles were found. '''
        mask_lost = ~self.is_accepted(beam.z, beam.dp)
        beam.alive[mask_lost] = 0
        return np.sum(beam.alive)

@cython.boundscheck(False)
@cython.wraparound(False)
def relocate_lost_particles(beam, int[::1] alive):
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

''' Cython functions for fast id and tagging of lost particles. '''

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef cytag_lost_rectangular(double[::1] u,
                             double low_lim, double high_lim):
    ''' Cython function for fast identification and tagging of particles
    lost at a rectangular aperture element, i.e. it tags particles with
    a spatial coord u (beam.x, beam.y or beam.z) lying outside the
    interval (low_lim, high_lim) as lost. Returns whether or not any
    lost particles were found. '''
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
    beam.y)) fulfilling u**2 + v**2 > radius_square as lost. Returns
    whether or not any lost particles were found. '''
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
    lost at a elliptical transverse aperture element lost. Returns
    whether or not any lost particles were found. '''
    cdef int n = u.shape[0]
    cdef int[::1] alive = np.ones(n, dtype=np.int32)
    cdef int i
    cdef double u_aper_sq_rec = 1./(u_aper*u_aper)
    cdef double v_aper_sq_rec = 1./(v_aper*v_aper)
    for i in xrange(n):
        if (u[i]*u[i]*u_aper_sq_rec + v[i]*v[i]*v_aper_sq_rec) > 1.:
            alive[i] = 0

    return alive
