import numpy as np
import pylab as pl

pl.close('all')

import ecloud.PyECLOUD_for_PyHEADTAIL as pyecl
from particles.particles import *
from scipy.constants import e, m_e
import numpy as np
from particles.slicer import *

from itertools import izip
import mystyle as ms

n_part_per_turn=5000

device = 'drift_for_benchmark'
#~ device = 'MBB'

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
epsn_z = 0.5*(0.2/0.23)**2

n_turns = 1
beamslicer = Slicer(64, nsigmaz=4)

L_ecloud = C

# Beam
bunch = Particles.as_gaussian(100000, e, gamma, 1.15e11, m_p, 0, beta_x, epsn_x, 0, beta_y, epsn_y, beta_z, epsn_z)


ecloud = pyecl.Ecloud(L_ecloud, beamslicer, Dt_ref = 25e-12, pyecl_input_folder=device)
ecloud.save_ele_distributions_last_track = True 
ecloud.save_ele_potential_and_field = True
ecloud.save_ele_MP_size = True
ecloud.save_ele_MP_position = True
ecloud.save_ele_MP_velocity = True

id_before = bunch.id[bunch.id<=n_part_per_turn]
xp_before = bunch.xp[bunch.id<=n_part_per_turn]
yp_before = bunch.yp[bunch.id<=n_part_per_turn]

ecloud.track(bunch)
ecloud.track(bunch)

id_after = bunch.id[bunch.id<=n_part_per_turn]
xp_after = bunch.xp[bunch.id<=n_part_per_turn]
z_after = bunch.z[bunch.id<=n_part_per_turn]
yp_after = bunch.yp[bunch.id<=n_part_per_turn]


indsort = np.argsort(id_after)
id_after = np.take(id_after, indsort)
xp_after = np.take(xp_after, indsort)
yp_after = np.take(yp_after, indsort)
z_after = np.take(z_after, indsort)



pl.figure(106)
sp1 = pl.subplot(2,1,1)
pl.plot(z_after,  (xp_after-xp_before), '.b')
ms.sciy()
pl.grid('on')
pl.subplot(2,1,2, sharex=sp1)
pl.plot(z_after,  (yp_after-yp_before), '.b')
ms.sciy()
pl.grid('on')



pl.show()







def plot_trajectory( x_obs = 0., y_obs = -.005):
	 
	i_obs = np.argmin((ecloud.x_MP_last_track[-1]-x_obs)**2+(ecloud.y_MP_last_track[-1]-y_obs)**2)

	 
	
	x_obs = map(lambda v:v[i_obs], ecloud.x_MP_last_track)
	y_obs = map(lambda v:v[i_obs], ecloud.y_MP_last_track)


	 
	 
	pl.figure(2000)
	pl.clf()
	pl.subplot(2,1,1)
	pl.plot(-ecloud.slicer.z_centers/3e8, x_obs, '.-')

	pl.plot(-ecloud.slicer.z_centers/3e8, y_obs, '.-r')

	 
	 
	pl.subplot(2,1,2)
	pl.plot(ecloud.slicer.z_centers[::-1]/3e8, ecloud.slicer.n_particles/\
					(np.abs(ecloud.slicer.z_centers[1]-ecloud.slicer.z_centers[0])), '.-')

	 
	pl.show()

plot_trajectory(x_obs = 0., y_obs = -.005 ) 



dec_fact = 50
import time
pl.figure(3000, figsize=(12, 12))
for ii in xrange(ecloud.slicer.n_slices-1, -1, -4):
    pl.clf()
    pl.subplot(2,2,1)#vmin=1, vmax=15,
    pl.imshow(np.log10(-ecloud.rho_ele_last_track[ii]/e).T, origin='lower', aspect='auto', 
              extent=(ecloud.spacech_ele.xg[0], ecloud.spacech_ele.xg[-1], ecloud.spacech_ele.yg[0], ecloud.spacech_ele.yg[-1]))
    #~ pl.colorbar()
    pl.axis('equal')
    
#    pl.subplot(2,2,2)
#    pl.imshow(10. * plt.log10( ecloud.phi_ele_last_track[ii]), origin='lower', aspect='auto',
#              extent=(ecloud.poisson.x[0,0], ecloud.poisson.x[0,-1], ecloud.poisson.y[0,0], ecloud.poisson.y[-1,0]))

    pl.subplot(2,2,2)
    pl.plot(ecloud.x_MP_last_track[ii][::dec_fact],  ecloud.y_MP_last_track[ii][::dec_fact], '.')
    pl.axis('equal')
       
    pl.subplot(2,2,3)
    pl.imshow(ecloud.Ex_ele_last_track[ii].T, origin='lower', aspect='auto',
              extent=(ecloud.spacech_ele.xg[0], ecloud.spacech_ele.xg[-1], ecloud.spacech_ele.yg[0], ecloud.spacech_ele.yg[-1]))
    pl.axis('equal')
    #~ pl.colorbar()
    
    pl.subplot(2,2,4)
    pl.imshow(ecloud.Ey_ele_last_track[ii].T, origin='lower', aspect='auto',
              extent=(ecloud.spacech_ele.xg[0], ecloud.spacech_ele.xg[-1], ecloud.spacech_ele.yg[0], ecloud.spacech_ele.yg[-1]))
    #~ pl.colorbar()
    ms.sciy()
    pl.axis('equal')
    pl.draw()
    pl.ion()
    time.sleep(.1)

