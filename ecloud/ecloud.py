'''
Created on 06.01.2014

@author: Kevin Li
'''


import numpy as np


import copy, h5py, sys
from scipy.constants import c, e, epsilon_0, m_e, m_p, pi

from particles.particles import *
from particles.slicer import *
from solvers.poissonfft import *


# re = 1 / (4 * pi * epsilon_0) * e ** 2 / c ** 2 / m_e
rp = 1 / (4 * pi * epsilon_0) * e ** 2 / c ** 2 / m_p

class eletrack_forward_euler_drift(object):
    def __init__(self, charge, mass):
        self.q_over_m = charge/mass


    def step(self, dt, x, y, vx, vy, vz, ex, ey):
        vx +=  self.q_over_m*dt*ex
        vy +=  self.q_over_m*dt*ey
        x += dt * vx
        y += dt * vy


class Ecloud():
    '''
    ecloud (very comprehensive Gianni style documentation :-) )
    '''

    def __init__(self, particles, grid_extension_x, grid_extension_y, grid_nx, grid_ny, slicer):

        self.poisson = PoissonFFT(grid_extension_x, grid_extension_y, grid_nx, grid_ny)

        self.slicer = slicer

        self.ex1 = np.zeros((grid_ny, grid_nx))
        self.ey1 = np.zeros((grid_ny, grid_nx))
        self.phi1 = np.zeros((grid_ny, grid_nx))
        self.rho1 = np.zeros((grid_ny, grid_nx))


        self.particles = particles

        self.eletracker = eletrack_forward_euler_drift(particles.charge, particles.mass)

        self.particles.x_init = self.particles.x.copy()
        self.particles.xp_init = self.particles.xp.copy()

        self.particles.y_init = self.particles.y.copy()
        self.particles.yp_init = self.particles.yp.copy()

        self.particles.zp_init = self.particles.zp.copy()

        self.particles.ex = np.zeros(particles.n_macroparticles)
        self.particles.ey = np.zeros(particles.n_macroparticles)

        self.ex2 = np.zeros((grid_ny, grid_nx))
        self.ey2 = np.zeros((grid_ny, grid_nx))
        self.phi2 = np.zeros((grid_ny, grid_nx))
        self.rho2 = np.zeros((grid_ny, grid_nx))

        self.save_ele_distributions_last_track = False

        self.save_ele_potential_and_field = False

        self.save_ele_MP_position = False
        self.save_ele_MP_velocity = False
        self.save_ele_MP_size = False

    def reinit(self):
        self.particles.x[:]=self.particles.x_init #it is a mutation and not a binding (and we have tested it :-))
        self.particles.xp[:]=self.particles.xp_init

        self.particles.y[:]=self.particles.y_init #it is a mutation and not a binding (and we have tested it :-))
        self.particles.yp[:]=self.particles.yp_init


    # @profile
    def track(self, beam):

        if not beam.same_size_for_all_MPs:
            raise ValueError('ecloud module assumes same size for all beam MPs')

        beam_ex = np.zeros(beam.n_macroparticles)
        beam_ey = np.zeros(beam.n_macroparticles)

        # beam.compute_statistics() # need not be done here
        self.slicer.update_slices(beam)
        self.reinit()

        if self.save_ele_distributions_last_track:
            self.rho_ele_last_track = []

        if self.save_ele_potential_and_field:
            self.phi_ele_last_track = []
            self.Ex_ele_last_track = []
            self.Ey_ele_last_track = []

        if self.save_ele_MP_position:
            self.x_MP_last_track = []
            self.y_MP_last_track = []

        if self.save_ele_MP_velocity:
            self.vx_MP_last_track = []
            self.vy_MP_last_track = []

        if self.save_ele_MP_size:
            raise ValueError('Not implemented yet!')

        for i in xrange(self.slicer.n_slices-1, -1, -1):

            ix = np.s_[self.slicer.z_index[i]:self.slicer.z_index[i + 1]]

            dz = (self.slicer.z_bins[i + 1] - self.slicer.z_bins[i])
            dt = dz / (beam.beta * c)
            #print  'z_bins[i]', self.slicer.z_bins[i], 'dz', dz, 'dt', dt

            # beam fields:
            self.poisson.gather_from(beam.x[ix], beam.y[ix], self.rho1)
            self.poisson.compute_potential(self.rho1, self.phi1)
            self.poisson.compute_fields(self.phi1, self.ex1, self.ey1)
            #beam field in V/m:
            self.ex1 = 1./(2. * pi * epsilon_0)*e*beam.n_particles_per_mp/dz * self.ex1
            self.ey1 = 1./(2. * pi * epsilon_0)*e*beam.n_particles_per_mp/dz * self.ey1

            # scatter to the cloud
            self.poisson.scatter_to(self.ex1, self.ey1, self.particles.x, self.particles.y, self.particles.ex, self.particles.ey)

            # Beam pushes e-cloud
            self.eletracker.step(dt, self.particles.x, self.particles.y, self.particles.xp, self.particles.yp, self.particles.zp,
                                 self.particles.ex, self.particles.ey)
#             self.particles.xp +=  -e/m_e*dt*self.particles.ex
#             self.particles.yp +=  -e/m_e*dt*self.particles.ey
#             self.particles.x += dt * self.particles.xp
#             self.particles.y += dt * self.particles.yp


            # ecloud fields
            self.poisson.gather_from(self.particles.x, self.particles.y, self.rho2)
            self.poisson.compute_potential(self.rho2, self.phi2)
            self.poisson.compute_fields(self.phi2, self.ex2, self.ey2)

            # scatter to beam
            self.poisson.scatter_to(self.ex2, self.ey2, beam.x[ix], beam.y[ix], beam_ex[ix], beam_ey[ix])
            cf1 = -2 * 1 * rp * 1 / (beam.gamma * beam.beta ** 2) *self.particles.n_particles_per_mp
            #   = -2 * c ** 2 * rp  * ex / dL * dL * 1 / gamma / (beta * c) ** 2 # WATCH OUT HERE!!! dL / dz???

            # e-cloud pushes beam
            beam.xp[ix] += cf1 * beam_ex[ix]
            beam.yp[ix] += cf1 * beam_ey[ix]

            if self.save_ele_distributions_last_track:
                self.rho_ele_last_track.append(self.rho2.copy())
                #print 'Here'

            if self.save_ele_potential_and_field:
                self.phi_ele_last_track.append(self.phi2.copy())
                self.Ex_ele_last_track.append(self.ex2.copy())
                self.Ey_ele_last_track.append(self.ey2.copy())

            if self.save_ele_MP_position:
                self.x_MP_last_track.append(self.particles.x.copy())
                self.y_MP_last_track.append(self.particles.y.copy())

            if self.save_ele_MP_velocity:
                self.vx_MP_last_track.append(self.particles.xp.copy())
                self.vy_MP_last_track.append(self.particles.yp.copy())
                
        
        if self.save_ele_distributions_last_track:
            self.rho_ele_last_track = self.rho_ele_last_track[::-1]

        if self.save_ele_potential_and_field:
            self.phi_ele_last_track = self.phi_ele_last_track[::-1]
            self.Ex_ele_last_track = self.Ex_ele_last_track[::-1]
            self.Ey_ele_last_track = self.Ey_ele_last_track[::-1]

        if self.save_ele_MP_position:
            self.x_MP_last_track = self.x_MP_last_track[::-1]
            self.y_MP_last_track = self.y_MP_last_track[::-1]

        if self.save_ele_MP_velocity:
            self.vx_MP_last_track = self.vx_MP_last_track[::-1]
            self.vy_MP_last_track = self.vy_MP_last_track[::-1]

        if self.save_ele_MP_size:
            raise ValueError('Not implemented yet!')         
