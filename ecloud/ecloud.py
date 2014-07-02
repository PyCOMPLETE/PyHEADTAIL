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


re = 1 / (4 * pi * epsilon_0) * e ** 2 / c ** 2 / m_e
rp = 1 / (4 * pi * epsilon_0) * e ** 2 / c ** 2 / m_p


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

		self.particles.x_init = self.particles.x.copy()
		self.particles.xp_init = self.particles.xp.copy()

		self.particles.y_init = self.particles.y.copy()
		self.particles.yp_init = self.particles.yp.copy()

		self.particles.ex = np.zeros(particles.n_macroparticles)
		self.particles.ey = np.zeros(particles.n_macroparticles)

		self.ex2 = np.zeros((grid_ny, grid_nx))
		self.ey2 = np.zeros((grid_ny, grid_nx))
		self.phi2 = np.zeros((grid_ny, grid_nx))
		self.rho2 = np.zeros((grid_ny, grid_nx))

    def reinit(self):
		self.particles.x[:]=self.particles.x_init #it is a mutation and not a binding (and we have tested it :-))
		self.particles.xp[:]=self.particles.xp_init
		
		self.particles.y[:]=self.particles.y_init #it is a mutation and not a binding (and we have tested it :-))
		self.particles.yp[:]=self.particles.yp_init
        

    # @profile
    def track(self, beam):

        beam_ex = np.zeros(beam.n_macroparticles)
        beam_ey = np.zeros(beam.n_macroparticles)

        # beam.compute_statistics() # need not be done here
        self.slicer.update_slices(beam)
        self.reinit()


        for i in xrange(self.slicer.n_slices):

			ix = np.s_[self.slicer.z_index[i]:self.slicer.z_index[i + 1]]

			# beam fields:
			self.poisson.gather_from(beam.x[ix], beam.y[ix], self.rho1)
			self.poisson.compute_potential(self.rho1, self.phi1)
			self.poisson.compute_fields(self.phi1, self.ex1, self.ey1)

			# scatter to the cloud
			self.poisson.scatter_to(self.ex1, self.ey1, self.particles.x, self.particles.y, self.particles.ex, self.particles.ey)


			dt = (self.slicer.z_bins[i + 1] - self.slicer.z_bins[i]) / (beam.beta * c)

			cf1 = -2 * c * re * 1 / beam.beta * beam.intensity / beam.n_macroparticles
			
			#print i, (self.slicer.z_bins[i + 1] - self.slicer.z_bins[i])


			# Beam pushes e-cloud
			self.particles.xp += cf1 * self.particles.ex
			self.particles.yp += cf1 * self.particles.ey
			self.particles.x += dt * self.particles.xp
			self.particles.y += dt * self.particles.yp
			#print self.particles.x, self.particles.y

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
			
                

                
                
                
			# Visualisation for testing the electron cloud pinch
			# ==================================================
			if i == 0:
				fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 12))
			[ax.cla() for ax in (ax1, ax2, ax3, ax4)]
			# [ax.set_aspect('equal') for ax in (ax1, ax2, ax3, ax4)]
			ax1.contour(self.poisson.fgreen.T, 100)
			ax2.plot(self.phi1[self.poisson.ny / 2,:], '-g')
			# ax2.plot(phi1[bunch.poisson_other.ny / 2,:], '-r')
			# ax2.plot(phi2[bunch.poisson_other.ny / 2,:], '-', c='orange')
			# ax3.contourf(self.poisson_self.x, self.poisson_self.y, 10 * plt.log10(self.poisson_self.rho), 100)
			# print self.phi1 # 10 * plt.log10(self.rho2)
			ax3.imshow(10 * plt.log10(self.rho2), origin='lower', aspect='auto', vmin=50, vmax=1e2,
					   extent=(self.poisson.x[0,0], self.poisson.x[0,-1], self.poisson.y[0,0], self.poisson.y[-1,0]))
			# ax3.scatter(self.x[::20], self.y[::20], c='b', marker='.')
			# ax3.quiver(self.x[::50], self.y[::50], self.kx[::50], self.ky[::50], color='g')
			# ax3.contour(p.x, p.y, p.phi, 100, lw=2)
			# ax3.scatter(bunch.x[ix], bunch.y[ix], c='y', marker='.', alpha=0.8)
			ax4.imshow(plt.sqrt(self.ex1 ** 2 + self.ey1 ** 2), origin='lower', aspect='auto',
					   extent=(self.poisson.x[0,0], self.poisson.x[0,-1], self.poisson.y[0,0], self.poisson.y[-1,0]))

			plt.draw()
