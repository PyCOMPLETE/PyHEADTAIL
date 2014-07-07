from __future__ import division



import itertools, time

from ecloud.ecloud import *
import ecloud.ecloud as ec
from particles.particles import *
from particles.slicer import *
from trackers.transverse_tracker import *
from trackers.longitudinal_tracker import *

from scipy.constants import e, m_e

import pylab as pl
import time 

plt.ion()
# Parameters
# ==========
n_macroparticles = 100000

C = 6911.
R = C / (2 * np.pi)
gamma_tr = 18.
gamma = 27.7
eta = 1 / gamma_tr ** 2 - 1 / gamma ** 2
Qx = 20.13
Qy = 20.18
Qs = 0.017
beta_x = 54.6
beta_y = 54.6
beta_z = np.abs(eta) * R / Qs
epsn_x = 2.5
epsn_y = 2.5
epsn_z = 0.5

n_turns = 1

# Beam
bunch = Particles.as_gaussian(100000, e, gamma, 1.15e11, m_p, 0, beta_x, epsn_x, 0, beta_y, epsn_y, beta_z, epsn_z)

# Betatron
n_segments = 1
s = np.arange(n_segments + 1) * C / n_segments
ltm = TransverseTracker.from_copy(s,
    np.zeros(n_segments), np.ones(n_segments) * beta_x, np.zeros(n_segments),
    np.zeros(n_segments), np.ones(n_segments) * beta_y, np.zeros(n_segments),
    Qx, 0, 0, 0, Qy, 0, 0, 0)

# Synchrotron
alpha = 1 / gamma_tr ** 2
cavity = LinearMap(C, alpha, Qs)

# E-cloud
e_density = 2e11 # electrons per m^3
x_max = 20 * plt.std(bunch.x)
x_min = -x_max
y_max = 20 * plt.std(bunch.y)
y_min = -y_max

grid_extension_x = x_max
grid_extension_y = y_max
grid_nx = 128
grid_ny = 128

N_electrons = e_density * (x_max - x_min) * (y_max - y_min) * C / n_segments

particles = Particles.as_uniformXY(n_macroparticles, -e, 1., N_electrons, m_e, x_min, x_max, y_min, y_max)
#~ plt.plot(particles.x, particles.y, '.')
#~ plt.show()
slicer = Slicer(64, nsigmaz=3)
ecloud = ec.Ecloud(particles, grid_extension_x, grid_extension_y, grid_nx, grid_ny, slicer)
ecloud.save_ele_distributions_last_track = True 
ecloud.save_ele_potential_and_field = True
ecloud.save_ele_MP_size = False
ecloud.save_ele_MP_position = True
ecloud.save_ele_MP_velocity = True
#~ map_ = list(itertools.chain.from_iterable([[l] + [ecloud] for l in ltm] + [[cavity]]))

elements=[]
for l in ltm:
    elements+=[l, ecloud]
elements.append(cavity)


#~ r = 10 * plt.log10(bunch.z ** 2 + (beta_z * bunch.dp) ** 2)
for i in range(n_turns):
    t0 = time.clock()
    for ele in elements:
        ele.track(bunch)

    #~ bunchmonitor.dump(bunch)
    print '--> Elapsed time:', time.clock() - t0
        # plt.cla()
        # plt.scatter(bunch.z, bunch.dp, c=r, marker='o', lw=0)
        # plt.gca().set_xlim(-1, 1)
        # plt.gca().set_ylim(-1e-2, 1e-2)
        # plt.draw()
        
# Try to plot the pinch
from itertools import izip
import mystyle as ms

dec_fact = 10

x_obs = .001
y_obs = -.001 


i_obs = np.argmin((ecloud.x_MP_last_track[0]-x_obs)**2+(ecloud.y_MP_last_track[0]-y_obs)**2)


x_arr = np.array(ecloud.x_MP_last_track) 
y_arr = np.array(ecloud.y_MP_last_track)

vx_arr = np.array(ecloud.vx_MP_last_track) 
vy_arr = np.array(ecloud.vy_MP_last_track)

pl.close('all')
pl.figure(2)
pl.subplot(2,1,1)
pl.plot(ecloud.slicer.z_centers/3e8, x_arr[:, i_obs], '.-')
pl.plot(ecloud.slicer.z_centers/3e8, y_arr[:, i_obs], '.-r')

pl.subplot(2,1,2)
pl.plot(ecloud.slicer.z_centers/3e8, vx_arr[:, i_obs], '.-')
pl.plot(ecloud.slicer.z_centers/3e8, vy_arr[:, i_obs], '.-r')

pl.show()


pl.figure(1, figsize=(12, 12))
for ii in xrange(ecloud.slicer.n_slices):
    pl.clf()
    pl.subplot(2,2,1)
    pl.imshow(10. * plt.log10(ecloud.rho_ele_last_track[ii]), origin='lower', aspect='auto', vmin=50, vmax=1e2,
              extent=(ecloud.poisson.x[0,0], ecloud.poisson.x[0,-1], ecloud.poisson.y[0,0], ecloud.poisson.y[-1,0]))
    pl.colorbar()
    
#    pl.subplot(2,2,2)
#    pl.imshow(10. * plt.log10( ecloud.phi_ele_last_track[ii]), origin='lower', aspect='auto',
#              extent=(ecloud.poisson.x[0,0], ecloud.poisson.x[0,-1], ecloud.poisson.y[0,0], ecloud.poisson.y[-1,0]))

    pl.subplot(2,2,2)
    pl.plot(ecloud.x_MP_last_track[ii][::dec_fact],  ecloud.y_MP_last_track[ii][::dec_fact], '.')
       
    pl.subplot(2,2,3)
    pl.imshow(ecloud.Ex_ele_last_track[ii], origin='lower', aspect='auto',
              extent=(ecloud.poisson.x[0,0], ecloud.poisson.x[0,-1], ecloud.poisson.y[0,0], ecloud.poisson.y[-1,0]))
    pl.colorbar()
    
    pl.subplot(2,2,4)
    pl.imshow(ecloud.Ey_ele_last_track[ii], origin='lower', aspect='auto',
              extent=(ecloud.poisson.x[0,0], ecloud.poisson.x[0,-1], ecloud.poisson.y[0,0], ecloud.poisson.y[-1,0]))
    pl.colorbar()
    ms.sciy
    pl.draw()
    time.sleep(.01)
    
    



