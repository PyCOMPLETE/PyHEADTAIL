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

    def __init__(self, apply_losses_here, *args, **kwargs):
        ''' The boolean apply_losses_here controls whether the
        Particles.update_losses(beam) method is called to relocate lost
        particles to the end of the bunch.u_all arrays (u = x, y, z,
        ...) and to actually remove them from the numpy array views
        beam.u and leave them in an untracked state. In case there are
        several Aperture elements placed at a segment boundary of the
        accelerator ring, apply_losses_here should only be set to True
        for the last one to increase performance. '''
        self.apply_losses_here = apply_losses_here

    def track(self, beam):
        ''' Tag particles not passing through the aperture as lost. If
        there are any losses and the self.apply_losses_here is set to
        True, the method Particles.update_losses() is called to relocate
        lost particles to the end of the beam.u_all numpy arrays (u = x,
        y, z, ...) and to actually remove them from the numpy array
        views beam.u and leave them in an untracked state. Also, clean
        currently cached slice_sets of the bunch as losses change its
        state. '''
        losses = self.tag_lost_particles(beam)

        if losses and self.apply_losses_here:
            beam.update_losses()
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
            beam.x, beam.alive, self.x_low, self.x_high)


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
            beam.y, beam.alive, self.y_low, self.y_high)


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
            beam.z, beam.alive, self.z_low, self.z_high)


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


''' Cython functions for fast id and tagging of lost particles. '''

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef cytag_lost_rectangular(double[::1] u, unsigned int[::1] alive,
                           double low_lim, double high_lim):
    ''' Cython function for fast identification and tagging of particles
    lost at a rectangular aperture element, i.e. it tags particles with
    a spatial coord u (beam.x, beam.y or beam.z) lying outside the
    interval (low_lim, high_lim) as lost. Returns whether or not any
    lost particles were found. '''
    cdef unsigned int n = alive.shape[0]
    cdef unsigned int losses = 0
    cdef unsigned int i
    for i in xrange(n):
        if u[i] < low_lim or u[i] > high_lim:
            alive[i] = 0
            losses = 1
    return losses

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef cytag_lost_circular(
    double[::1] u, double[::1] v, unsigned int[::1] alive,
    double radius_square):
    ''' Cython function for fast identification and tagging of particles
    lost at a circular transverse aperture element of a given radius,
    i.e. it tags particles with spatial coords u, v (usually (beam.x,
    beam.y)) fulfilling u**2 + v**2 > radius_square as lost. Returns
    whether or not any lost particles were found. '''
    cdef unsigned int n = alive.shape[0]
    cdef unsigned int losses = 0
    cdef unsigned int i
    for i in xrange(n):
        if (u[i]*u[i] + v[i]*v[i]) > radius_square:
            alive[i] = 0
            losses = 1
    return losses
