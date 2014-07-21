import numpy as np
import pylab as pl

filename = 'headtail_for_test/test_protons/SPS_Q20_proton_check_prb.dat'


n_part_per_turn = 5000

appo = np.loadtxt(filename)

parid = np.reshape(appo[:,0], (-1, n_part_per_turn))
x = np.reshape(appo[:,1], (-1, n_part_per_turn))
xp = np.reshape(appo[:,2], (-1, n_part_per_turn))
y = np.reshape(appo[:,3], (-1, n_part_per_turn))
yp =np.reshape(appo[:,4], (-1, n_part_per_turn))
z = np.reshape(appo[:,5], (-1, n_part_per_turn))
zp = np.reshape(appo[:,6], (-1, n_part_per_turn))


pl.close('all')
pl.figure(1)
pl.plot(xp[0,:], '.-')
pl.plot(xp[1,:], 'r.-')



dxp = xp[1,:]-xp[0,:]
dyp = yp[1,:]-yp[0,:]

pl.figure(2)
sp1 = pl.subplot(2,1,1)
pl.plot(dxp, '.-')
sp2 = pl.subplot(2,1,2)
pl.plot(dyp, '.-')


pl.show()


import ecloud.PyECLOUD_for_PyHEADTAIL as pyecl
from particles.particles import *
from scipy.constants import e, m_e
import numpy as np
from particles.slicer import *


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
beamslicer = Slicer(64, nsigmaz=3)

L_ecloud = C

# Beam
bunch = Particles.as_gaussian(100000, e, gamma, 1.15e11, m_p, 0, beta_x, epsn_x, 0, beta_y, epsn_y, beta_z, epsn_z)
bunch.x[:n_part_per_turn] = x[0,:]
bunch.xp[:n_part_per_turn] = xp[0,:]
bunch.y[:n_part_per_turn] = y[0,:]
bunch.yp[:n_part_per_turn] = yp[0,:]
bunch.z[:n_part_per_turn] = z[0,:]
bunch.dp[:n_part_per_turn] =zp[0,:]

ecloud = pyecl.Ecloud(L_ecloud, beamslicer, Dt_ref = 25e-12, pyecl_input_folder='drift_for_benchmark')
ecloud.save_ele_distributions_last_track = True 
ecloud.save_ele_potential_and_field = True
ecloud.save_ele_MP_size = True
ecloud.save_ele_MP_position = True
ecloud.save_ele_MP_velocity = True

id_before = bunch.id[bunch.id<=n_part_per_turn]
xp_before = bunch.xp[bunch.id<=n_part_per_turn]
yp_before = bunch.yp[bunch.id<=n_part_per_turn]

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

pl.figure(100)
pl.plot(id_before-1, xp_before, 'r')
pl.plot(xp[0, :], 'ob', mfc=None)


pl.figure(101)
pl.plot(id_after-1, xp_after, '.r')
pl.plot(xp[1, :], 'ob', mfc=None)


pl.figure(103)
pl.plot(id_before-1, xp_after-xp_before, '.r')
pl.plot(xp[1, :]-xp[0, :], 'ob', mfc=None)


pl.figure(104)
sp1 = pl.subplot(2,1,1)
pl.plot(id_before-1, 100*np.abs((xp_after-xp_before)-(xp[1, :]-xp[0, :]))/np.abs((xp[1, :]-xp[0, :])), '.r')
pl.subplot(2,1,2, sharex=sp1)
pl.plot(id_before-1, np.abs((xp[1, :]-xp[0, :])), '.r')

mask_no_kick = np.abs((xp[1, :]-xp[0, :]))>0.5*np.max(np.abs((xp[1, :]-xp[0, :])))

pl.figure(105)
sp1 = pl.subplot(2,1,1)
pl.plot( (100*np.abs((xp_after-xp_before)-(xp[1, :]-xp[0, :]))/np.abs((xp[1, :]-xp[0, :])))[mask_no_kick], '.r')
pl.subplot(2,1,2, sharex=sp1)
pl.plot( (np.abs((xp[1, :]-xp[0, :])))[mask_no_kick], '.r')
pl.ylim(0, None)


pl.figure(106)
sp1 = pl.subplot(2,1,1)
pl.plot(z_after,  (100*np.abs((xp_after-xp_before)-(xp[1, :]-xp[0, :]))/np.abs((xp[1, :]-xp[0, :]))), '.r')
pl.grid('on')
pl.subplot(2,1,2, sharex=sp1)
pl.plot(z_after,  (xp[1, :]-xp[0, :]), '.r')
pl.plot(z_after,  (xp_after-xp_before), '.b')
pl.grid('on')

pl.suptitle('%.2f'%(100*pl.norm((xp_after-xp_before)-(xp[1, :]-xp[0, :]))/pl.norm((xp[1, :]-xp[0, :]))))


pl.figure(116)
sp1 = pl.subplot(2,1,1)
pl.plot(z_after,  (100*np.abs((yp_after-yp_before)-(yp[1, :]-yp[0, :]))/np.abs((yp[1, :]-yp[0, :]))), '.r')
pl.grid('on')
pl.subplot(2,1,2, sharex=sp1)
pl.plot(z_after,  (yp[1, :]-yp[0, :]), '.r')
pl.plot(z_after,  (yp_after-yp_before), '.b')
pl.grid('on')
pl.suptitle('%.2f'%(100*pl.norm((yp_after-yp_before)-(yp[1, :]-yp[0, :]))/pl.norm((yp[1, :]-yp[0, :]))))


pl.show()




# Try to plot the pinch
aaa = np.loadtxt(filename.split('_prb.dat')[0]+'_elec.dat');
hdtl_1st_turn = np.loadtxt(filename.split('_prb.dat')[0]+'_hdtl.dat');

z_hdtl = hdtl_1st_turn[:,0]
pp_bin_hdtl = hdtl_1st_turn[:,1]

#np.ascontiguousarray(
                     
sl_id = aaa[:,0]
x_e = aaa[:,1]
y_e = aaa[:,3]


x_hdtl = []
y_hdtl = []
for ii in xrange(int(np.max(sl_id)), -1, -1):
    mask_select = np.abs(sl_id-ii)<.1
    x_hdtl.append(x_e[mask_select])
    y_hdtl.append(y_e[mask_select])

z_hdtl = z_hdtl[:len(x_hdtl)]
pp_bin_hdtl = pp_bin_hdtl[:len(x_hdtl)]
from itertools import izip
import mystyle as ms



def plot_trajectory( x_obs = 0., y_obs = -.005):
	 


	i_obs_hdtl = np.argmin((x_hdtl[0]-x_obs)**2+(y_hdtl[0]-y_obs)**2)
	i_obs = np.argmin((ecloud.x_MP_last_track[-1]-x_hdtl[0][i_obs_hdtl])**2+(ecloud.y_MP_last_track[-1]-y_hdtl[0][i_obs_hdtl])**2)

	 
	 
	#x_arr = np.array(ecloud.x_MP_last_track) 
	#y_arr = np.array(ecloud.y_MP_last_track)
	 
	#x_arr_hdtl = np.array(x_hdtl) 
	#y_arr_hdtl = np.array(y_hdtl)
	 
	#vx_arr = np.array(ecloud.vx_MP_last_track) 
	#vy_arr = np.array(ecloud.vy_MP_last_track)
	x_obs = map(lambda v:v[i_obs], ecloud.x_MP_last_track)
	y_obs = map(lambda v:v[i_obs], ecloud.y_MP_last_track)

	x_obs_hdtl = map(lambda v:v[i_obs_hdtl], x_hdtl)
	y_obs_hdtl = map(lambda v:v[i_obs_hdtl], y_hdtl)

	 
	 
	pl.figure(2000)
	pl.clf()
	pl.subplot(2,1,1)
	pl.plot(-ecloud.slicer.z_centers/3e8, x_obs, '.-')
	pl.plot(-z_hdtl/3e8, x_obs_hdtl, 'o--')
	pl.plot(-ecloud.slicer.z_centers/3e8, y_obs, '.-r')
	pl.plot(-z_hdtl/3e8, y_obs_hdtl, 'o--r')
	 
	 
	pl.subplot(2,1,2)
	pl.plot(ecloud.slicer.z_centers[::-1]/3e8, ecloud.slicer.n_particles/\
					(np.abs(ecloud.slicer.z_centers[1]-ecloud.slicer.z_centers[0])), '.-')
	pl.plot(-z_hdtl/3e8, pp_bin_hdtl/np.abs(z_hdtl[1]-z_hdtl[0]), 'r.-')
	#pl.plot(ecloud.slicer.z_centers/3e8, vx_arr[:, i_obs], '.-')
	#pl.plot(ecloud.slicer.z_centers/3e8, vy_arr[:, i_obs], '.-r')
	 
	pl.show()

plot_trajectory(x_obs = 0., y_obs = -.005 ) 



dec_fact = 50
import time
pl.figure(3000, figsize=(12, 12))
for ii in xrange(ecloud.slicer.n_slices-1, -1, -4):
    pl.clf()
    pl.subplot(2,2,1)
    pl.imshow(np.log10(-ecloud.rho_ele_last_track[ii]/e).T, origin='lower', aspect='auto', vmin=1, vmax=15,
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

