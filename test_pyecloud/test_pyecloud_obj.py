import ecloud.PyECLOUD_for_PyHEADTAIL as pyecl
from particles.particles import *
from scipy.constants import e, m_e
import numpy as np
from particles.slicer import *


C = 6911.
R = C/(2.*np.pi)
gamma_tr = 18.
gamma = 27.7
eta = 1. / gamma_tr ** 2 - 1. / gamma ** 2
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
beamslicer = Slicer(64, nsigmaz=3)

# Beam
bunch = Particles.as_gaussian(100000, e, gamma, 1.15e11, m_p, 0, beta_x, epsn_x, 0, beta_y, epsn_y, beta_z, epsn_z)


ecloud = pyecl.Ecloud(beamslicer, Dt_ref = 25e-12)
ecloud.save_ele_distributions_last_track = True 
ecloud.save_ele_potential_and_field = True
ecloud.save_ele_MP_size = False
ecloud.save_ele_MP_position = True
ecloud.save_ele_MP_velocity = True


ecloud.track(bunch)


# Try to plot the pinch

import pylab as pl
from itertools import izip
import mystyle as ms

dec_fact = 10

x_obs = .001
y_obs = -.001 


i_obs = np.argmin((ecloud.x_MP_last_track[-1]-x_obs)**2+(ecloud.y_MP_last_track[-1]-y_obs)**2)


x_obs = map(lambda v:v[i_obs], ecloud.x_MP_last_track)
y_obs = map(lambda v:v[i_obs], ecloud.y_MP_last_track)

vx_obs = map(lambda v:v[i_obs], ecloud.vx_MP_last_track)
vy_obs = map(lambda v:v[i_obs], ecloud.vy_MP_last_track)

pl.close('all')
pl.figure(2)
pl.subplot(2,1,1)
pl.plot(-ecloud.slicer.z_centers/3e8, x_obs, '.-')
pl.plot(-ecloud.slicer.z_centers/3e8, y_obs, '.-r')

pl.subplot(2,1,2)
pl.plot(-ecloud.slicer.z_centers/3e8, vx_obs, '.-')
pl.plot(-ecloud.slicer.z_centers/3e8, vy_obs, '.-r')

pl.show()

import time
pl.figure(1, figsize=(12, 12))
for ii in xrange(ecloud.slicer.n_slices-1, -1, -4):
    pl.clf()
    pl.subplot(2,2,1)
    pl.imshow(10. * np.log10(-ecloud.rho_ele_last_track[ii]/e).T, origin='lower', aspect='auto', vmin=50, vmax=1e2,
              extent=(ecloud.spacech_ele.xg[0], ecloud.spacech_ele.xg[-1], ecloud.spacech_ele.yg[0], ecloud.spacech_ele.yg[-1]))
    pl.colorbar()
    
#    pl.subplot(2,2,2)
#    pl.imshow(10. * plt.log10( ecloud.phi_ele_last_track[ii]), origin='lower', aspect='auto',
#              extent=(ecloud.poisson.x[0,0], ecloud.poisson.x[0,-1], ecloud.poisson.y[0,0], ecloud.poisson.y[-1,0]))

    pl.subplot(2,2,2)
    pl.plot(ecloud.x_MP_last_track[ii][::dec_fact],  ecloud.y_MP_last_track[ii][::dec_fact], '.')
       
    pl.subplot(2,2,3)
    pl.imshow(ecloud.Ex_ele_last_track[ii].T, origin='lower', aspect='auto',
              extent=(ecloud.spacech_ele.xg[0], ecloud.spacech_ele.xg[-1], ecloud.spacech_ele.yg[0], ecloud.spacech_ele.yg[-1]))
    #~ pl.colorbar()
    
    pl.subplot(2,2,4)
    pl.imshow(ecloud.Ey_ele_last_track[ii].T, origin='lower', aspect='auto',
              extent=(ecloud.spacech_ele.xg[0], ecloud.spacech_ele.xg[-1], ecloud.spacech_ele.yg[0], ecloud.spacech_ele.yg[-1]))
    #~ pl.colorbar()
    ms.sciy
    pl.draw()
    pl.ion()
    time.sleep(.1)
