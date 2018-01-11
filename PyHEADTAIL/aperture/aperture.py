from __future__ import division
'''
Aperture module to manage particle losses. An aperture is
defined as a condition on the phase space coordinates. Particles
not fulfilling this condition are tagged as lost and are removed
from the beam. Parts of this module are implemented in cython
under aperture_cython.pyx for better performance.

@date: Created on 23.03.2016
@author: Hannes Bartosik, Giovanni Iadarola, Kevin Li, Adrian Oeftiger,
         Michael Schenk
'''

from . import Element, clean_slices

from abc import ABCMeta, abstractmethod

from ..general import pmath as pm
import numpy as np

def make_int32(array):
    # return np.array(array, dtype=np.int32)
    return array.astype(np.int32)


class Aperture(Element):
    '''Abstract base class for Aperture elements. An aperture is
    generally defined as a condition on the phase space coordinates.
    Particles not fulfilling this condition are tagged as lost and
    are removed from the beam directly after.
    '''

    __metaclass__ = ABCMeta

    @clean_slices
    def track(self, beam):
        '''Tag particles not passing through the aperture as lost. If
        there are any losses, the corresponding particles are removed
        from the beam by updating the beam.u arrays, s.t.
        beam.u = beam.u[:n_alive] after relocating lost particles to
        the end of these arrays. 'n_alive' denotes the number of alive
        particles after the given aperture element. In addition, the
        currently cached slice_sets of the beam are cleaned since losses
        change its (longitudinal) state.
        '''
        alive = self.tag_lost_particles(beam)

        if not pm.all(alive):
            # check whether all particles are lost, it's not safe to call
            # the cython version of relocate_all_particles in this case
            if not pm.any(alive):
                self.warns('ALL particles were lost')
                n_alive = 0
            else:
                # Move lost particles to the end of the beam.u arrays.
                n_alive = self.relocate_lost_particles(beam, alive)
            # Update beam.u arrays, i.e. remove lost particles.
            beam.macroparticlenumber = n_alive
            beam.x = beam.x[:n_alive]
            beam.y = beam.y[:n_alive]
            beam.z = beam.z[:n_alive]
            beam.xp = beam.xp[:n_alive]
            beam.yp = beam.yp[:n_alive]
            beam.dp = beam.dp[:n_alive]
            beam.id = beam.id[:n_alive]

    @abstractmethod
    def tag_lost_particles(self, beam):
        ''' This method is called by Aperture.track(beam) to identify
        particles not passing through the aperture. The aperture condition
        on the phase space coordinates is defined by the given Aperture
        element. Returns a np.int32 array 'alive' which contains the
        information on whether a particle is lost (0) or not (1).
        '''
        pass

    @staticmethod
    def relocate_lost_particles(beam, alive):
        '''Relocate particles marked as lost to the end of the beam.u arrays
        (u = x, y, z, ...). Return the number of alive particles
        n_alive_post after considering the losses.

        Arguments:
            - beam: Particles instance
            - alive: boolean mask with length n_particles where 1 means alive
        '''
        # descending sort to have alive particles (the 1 entries) in the front
        perm = pm.argsort(-alive)

        beam.reorder(perm)

        n_alive = make_int32(pm.sum(alive))
        return n_alive


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
        return tag_lost_rectangular(beam.x, self.x_low, self.x_high)


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
        return tag_lost_rectangular(beam.y, self.y_low, self.y_high)


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
        return tag_lost_rectangular(beam.z, self.z_low, self.z_high)


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
        return tag_lost_circular(beam.x, beam.y, self.radius_square)


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
        return tag_lost_ellipse(beam.x, beam.y,
                                  self.x_aper, self.y_aper)


def tag_lost_rectangular(u, low_lim, high_lim):
    '''Identify and tag particles lost at a rectangular aperture
    element, i.e. particles with
    a spatial coord u (beam.x, beam.y or beam.z) lying outside the
    interval (low_lim, high_lim). Return a np array mask 'alive'
    containing the information of alive / lost for each particle in the
    beam after the aperture.
    '''
    dead = (u < low_lim) + (u > high_lim)
    return make_int32(1 - dead)

def tag_lost_circular(u, v, radius_square):
    '''Identify and tag particles lost at a circular transverse aperture
    element of a given radius, i.e. particles with
    spatial coords u, v (usually (beam.x, beam.y)) fulfilling
    u**2 + v**2 > radius_square. Return a np array mask 'alive'
    containing the information of alive / lost for each particle in the
    beam after the aperture.
    '''
    alive = (u*u + v*v) <= radius_square
    return make_int32(alive)

def tag_lost_ellipse(u, v, u_aper, v_aper):
    '''Identify and tag particles lost at an elliptical transverse
    aperture element. Return a np array mask 'alive'
    containing the information of alive / lost for each particle in the
    beam after the aperture.
    '''
    alive = (u*u / (u_aper*u_aper) + v*v / (v_aper*v_aper)) <= 1.
    return make_int32(alive)
