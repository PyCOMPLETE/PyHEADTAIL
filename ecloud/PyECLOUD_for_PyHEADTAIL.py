# To do:
# 1 - now the pinch is not reinitialized (done!)
# 2 - introduce length of the interaction (done!)
# 3 - different input files (done!)
# 4 - define substeps consistently (done!)
# 5 - push the beam (done!)
# 6 - implement pinch saving (done!)



#import init as init 


#import beam_and_timing as beatim


from geom_impact_ellip import ellip_cham_geom_object
from geom_impact_poly import polyg_cham_geom_object
from sec_emission_model_ECLOUD import SEY_model_ECLOUD
from sec_emission_model_accurate_low_ene import SEY_model_acc_low_ene
from sec_emission_model_ECLOUD_nunif import SEY_model_ECLOUD_non_unif
from sec_emission_model_cos_low_ener import SEY_model_cos_le
from sec_emission_model_flat_low_ener import SEY_model_flat_le
import dynamics_dipole as dyndip
import dynamics_Boris_f2py as dynB
import dynamics_strong_B_generalized as dyngen

import MP_system as MPs
import space_charge_class as scc
import impact_management_class as imc
#import pyecloud_saver as pysav
import gas_ionization_class as gic
import gen_photoemission_class as gpc

#import parse_beam_file as pbf

import numpy as np
from scipy.constants import c, e


class MP_light(object):
	pass



class Ecloud(object):
	def __init__(self, L_ecloud, slicer, Dt_ref, pyecl_input_folder='./'):
		
		print 'PyECLOUD for PyHEADTAIL'
		print 'Initializing ecloud from folder: '+pyecl_input_folder
		self.slicer = slicer
		self.Dt_ref = Dt_ref
		self.L_ecloud = L_ecloud	
		
		b_par, x_aper, y_aper, B,\
		gas_ion_flag, P_nTorr, sigma_ion_MBarn, Temp_K, unif_frac, E_init_ion,\
		Emax, del_max, R0, E_th, sigmafit, mufit,\
		Dt, t_end, lam_th, t_ion, N_mp_max,\
		N_mp_regen, N_mp_after_regen, fact_split, fact_clean, nel_mp_ref_0,\
		Nx_regen, Ny_regen, Nvx_regen, Nvy_regen, Nvz_regen,regen_hist_cut,\
		N_mp_regen_low,\
		Dt_sc, Dh_sc, t_sc_ON,Dx_hist,r_center, scrub_en_th,\
		progress_path,  logfile_path, flag_movie, flag_sc_movie,\
		Dt_En_hist, Nbin_En_hist,En_hist_max, \
		photoem_flag, inv_CDF_refl_photoem_file, k_pe_st, refl_frac, alimit, e_pe_sigma,\
		e_pe_max,x0_refl, y0_refl, out_radius, \
		switch_model, switch_no_increase_energy, thresh_low_energy, save_mp_state_time_file, \
		init_unif_flag, Nel_init_unif, E_init_unif, x_max_init_unif, x_min_init_unif, y_max_init_unif, y_min_init_unif,\
		chamb_type, filename_chm, flag_detailed_MP_info, flag_hist_impact_seg,\
		track_method, B0x, B0y, B0z, B_map_file,  Bz_map_file, N_sub_steps, fact_Bmap, B_zero_thrhld,\
		N_mp_soft_regen, N_mp_after_soft_regen,\
		flag_verbose_file, flag_verbose_stdout,\
		flag_presence_sec_beams, sec_b_par_list, phem_resc_fac, dec_fac_secbeam_prof, el_density_probes, save_simulation_state_time_file,\
		x_min_hist_det, x_max_hist_det, y_min_hist_det, y_max_hist_det, Dx_hist_det, dec_fact_out, stopfile= \
		read_parameter_files_pyhdtl(pyecl_input_folder)
		
		#pyeclsaver=pysav.pyecloud_saver(logfile_path)
       
		if switch_model=='ECLOUD_nunif':
			flag_non_unif_sey = 1
		else:
			flag_non_unif_sey = 0
			
		if chamb_type=='ellip':
			chamb=ellip_cham_geom_object(x_aper, y_aper, flag_verbose_file=flag_verbose_file)
		elif chamb_type=='polyg':
			chamb=polyg_cham_geom_object(filename_chm, flag_non_unif_sey,
										 flag_verbose_file=flag_verbose_file, flag_verbose_stdout=flag_verbose_stdout)
		else:
			raise ValueError('Chamber type not recognized (choose: ellip/polyg)')
		
		
		MP_e=MPs.MP_system(N_mp_max, nel_mp_ref_0, fact_split, fact_clean, 
						   N_mp_regen_low, N_mp_regen, N_mp_after_regen,
						   Dx_hist, Nx_regen, Ny_regen, Nvx_regen, Nvy_regen, Nvz_regen, regen_hist_cut, chamb,
						   N_mp_soft_regen=N_mp_soft_regen, N_mp_after_soft_regen=N_mp_after_soft_regen)
		
	
		
		spacech_ele = scc.space_charge(chamb, Dh_sc, Dt_sc=Dt_sc)
		
		#~ sec_beams_list=[]
		#~ if flag_presence_sec_beams:
			#~ N_sec_beams = len(sec_b_par_list)
			#~ for ii in xrange(N_sec_beams):
				#~ print 'Initialize secondary beam %d/%d'%(ii+1, N_sec_beams)
				#~ sb_par = sec_b_par_list[ii]
				#~ sec_beams_list.append(beatim.beam_and_timing(sb_par.flag_bunched_beam, sb_par.fact_beam, sb_par.coast_dens, sb_par.beam_field_file,lam_th,
					 #~ b_spac=sb_par.b_spac, sigmaz=sb_par.sigmaz,t_offs=sb_par.t_offs, filling_pattern_file=sb_par.filling_pattern_file, Dt=Dt, t_end=t_end,
					 #~ beam_long_prof_file=sb_par.beam_long_prof_file, Dh_beam_field=sb_par.Dh_beam_field, chamb=chamb,  sigmax=sb_par.sigmax, sigmay=sb_par.sigmay,
					 #~ x_beam_pos = sb_par.x_beam_pos, y_beam_pos = sb_par.y_beam_pos, save_beam_field_file_as=sb_par.save_beam_field_file_as,
					 #~ flag_secodary_beam = True, t_primary_beam = beamtim.t,
					 #~ Nx=sb_par.Nx, Ny=sb_par.Ny, nimag=sb_par.nimag, progress_mapgen_file = (progress_path+('_mapgen_sec_%d'%ii))))
		
		
		
		if switch_model==0 or switch_model=='ECLOUD':
			sey_mod=SEY_model_ECLOUD(Emax,del_max,R0)
		elif switch_model==1 or switch_model=='ACC_LOW':
			sey_mod=SEY_model_acc_low_ene(Emax,del_max,R0)
		elif switch_model=='ECLOUD_nunif':
			sey_mod=SEY_model_ECLOUD_non_unif(chamb, Emax,del_max,R0)
		elif switch_model=='cos_low_ene':
			sey_mod=SEY_model_cos_le(Emax,del_max,R0)
		elif switch_model=='flat_low_ene':
			sey_mod=SEY_model_flat_le(Emax,del_max,R0)

		
		flag_seg = (flag_hist_impact_seg==1)
		   
		impact_man=imc.impact_management(switch_no_increase_energy, chamb, sey_mod, E_th, sigmafit, mufit,
					 Dx_hist, scrub_en_th, Nbin_En_hist, En_hist_max, thresh_low_energy=thresh_low_energy, flag_seg=flag_seg)
		

		if track_method == 'Boris':
			dynamics=dynB.pusher_Boris(Dt, B0x, B0y, B0z, \
					 B_map_file, fact_Bmap,  Bz_map_file,N_sub_steps=N_sub_steps)
		#~ elif track_method == 'StrongBdip':
			#~ dynamics=dyndip.pusher_dipole_magnet(Dt,B)  
		#~ elif track_method == 'StrongBgen':
			#~ dynamics=dyngen.pusher_strong_B_generalized(Dt, B0x, B0y,  \
					 #~ B_map_file, fact_Bmap, B_zero_thrhld) 
		else:
			raise ValueError("""track_method should be 'Boris' - others are not implemented in the PyHEADTAIL module""")
			   

			
		if init_unif_flag==1:
			MP_e.add_uniform_MP_distrib(Nel_init_unif, E_init_unif, x_max_init_unif, x_min_init_unif, y_max_init_unif, y_min_init_unif)
			
		spacech_ele.flag_decimate = False
			
		self.MP_e = MP_e
		self.dynamics = dynamics
		self.impact_man = impact_man
		self.spacech_ele = spacech_ele
		
		self.save_ele_distributions_last_track = False
		self.save_ele_potential_and_field = False
		self.save_ele_MP_position = False
		self.save_ele_MP_velocity = False
		self.save_ele_MP_size = False
		
		self.init_x = self.MP_e.x_mp.copy()
		self.init_y = self.MP_e.y_mp.copy()
		self.init_z = self.MP_e.z_mp.copy()
		self.init_vx = self.MP_e.vx_mp.copy()
		self.init_vy = self.MP_e.vy_mp.copy()
		self.init_vz = self.MP_e.vz_mp.copy()
		self.init_nel = self.MP_e.nel_mp.copy()
		self.init_N_mp = self.MP_e.N_mp
		
	def track(self, beam):
		
		#reinitialize
		self.MP_e.x_mp[:] = self.init_x #it is a mutation and not a binding (and we have tested it :-))
		self.MP_e.y_mp[:] = self.init_y
		self.MP_e.z_mp[:] = self.init_z
		self.MP_e.vx_mp[:] = self.init_vx
		self.MP_e.vy_mp[:] = self.init_vy
		self.MP_e.vz_mp[:] = self.init_vz
		self.MP_e.nel_mp[:] = self.init_nel
		self.MP_e.N_mp = self.init_N_mp
		
		MP_e = self.MP_e
		dynamics = self.dynamics
		impact_man = self.impact_man
		spacech_ele = self.spacech_ele

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
			self.nel_MP_last_track = []
			
		if self.save_ele_MP_position or save_ele_MP_velocity or self.save_ele_MP_size:
			self.N_MP_last_track = []
			
		
		if not beam.same_size_for_all_MPs:
			raise ValueError('ecloud module assumes same size for all beam MPs')

		self.slicer.update_slices(beam)
		
		
		for i in xrange(self.slicer.n_slices-1, -1, -1):

			# select particles in the slice
			ix = np.s_[self.slicer.z_index[i]:self.slicer.z_index[i + 1]]

			# slice size and time step
			dz = (self.slicer.z_bins[i + 1] - self.slicer.z_bins[i])
			dt = dz / (beam.beta * c)
			
			# define substep
			N_sub_steps = int(np.round(dt/self.Dt_ref))
			Dt_substep = dt/N_sub_steps

			# beam field 
			MP_p = MP_light()
			MP_p.x_mp = beam.x[ix]
			MP_p.y_mp = beam.y[ix]
			MP_p.nel_mp = beam.x*0.+beam.n_particles_per_mp/dz#they have to become cylinders
			MP_p.N_mp = self.slicer.n_macroparticles[i]
			#compute beam field (it assumes electrons)
			spacech_ele.recompute_spchg_efield(MP_p)
			#scatter to electrons
			Ex_n_beam, Ey_n_beam = spacech_ele.get_sc_eletric_field(MP_e)
			# go to actual beam particles 
			Ex_n_beam = -Ex_n_beam * beam.charge/e
			Ey_n_beam = -Ey_n_beam * beam.charge/e
			
			
			## compute electron field map
			spacech_ele.recompute_spchg_efield(MP_e)
			
			## compute electron field on electrons
			Ex_sc_n, Ey_sc_n = spacech_ele.get_sc_eletric_field(MP_e)
			
			## compute electron field on beam particles
			Ex_sc_p, Ey_sc_p = spacech_ele.get_sc_eletric_field(MP_p)
			
			## Total electric field on electrons
			Ex_n=Ex_sc_n+Ex_n_beam;
			Ey_n=Ey_sc_n+Ey_n_beam;
				
			## save position before motion step
			old_pos=MP_e.get_positions()
			
			## motion electrons
			MP_e = dynamics.stepcustomDt(MP_e, Ex_n,Ey_n, Dt_substep=Dt_substep, N_sub_steps=N_sub_steps)
			
			## impacts: backtracking and secondary emission
			MP_e = impact_man.backtrack_and_second_emiss(old_pos, MP_e)
			
			## kick beam particles
			fact_kick = beam.charge/(beam.mass*beam.beta*beam.beta*beam.gamma*c*c)*self.L_ecloud
			beam.xp[ix]+=fact_kick*Ex_sc_p
			beam.yp[ix]+=fact_kick*Ey_sc_p
			
			if self.save_ele_distributions_last_track:
				self.rho_ele_last_track.append(spacech_ele.rho.copy())
				#print 'Here'

			if self.save_ele_potential_and_field:
				self.phi_ele_last_track.append(spacech_ele.phi.copy())
				self.Ex_ele_last_track.append(spacech_ele.efx.copy())
				self.Ey_ele_last_track.append(spacech_ele.efy.copy())

			if self.save_ele_MP_position:
				self.x_MP_last_track.append(MP_e.x_mp.copy())
				self.y_MP_last_track.append(MP_e.y_mp.copy())

			if self.save_ele_MP_velocity:
				self.vx_MP_last_track.append(MP_e.vx_mp.copy())
				self.vy_MP_last_track.append(MP_e.vy_mp.copy())
				
			if self.save_ele_MP_size:
				self.nel_MP_last_track.append(MP_e.nel_mp.copy())
				
			if self.save_ele_MP_position or save_ele_MP_velocity or self.save_ele_MP_size:
				self.N_MP_last_track.append(MP_e.N_mp)
				
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
			self.nel_MP_last_track = self.nel_MP_last_track[::-1]
			
		if self.save_ele_MP_position or save_ele_MP_velocity or self.save_ele_MP_size:
				self.N_MP_last_track = self.N_MP_last_track[::-1]
			
			
		 
			
			
def read_parameter_files_pyhdtl(pyecl_input_folder):
    switch_model=0
    simulation_param_file=pyecl_input_folder+'/simulation_parameters.input'
    
    save_mp_state_time_file = -1
    
    stopfile = 'stop'
    
    dec_fact_out = 1
    
    init_unif_flag = 0
    Nel_init_unif = None
    E_init_unif = None
    x_max_init_unif = None
    x_min_init_unif = None
    y_max_init_unif = None
    y_min_init_unif = None
    
    chamb_type = 'ellip'
    filename_chm = None
    
    x_aper = None
    y_aper = None
    flag_detailed_MP_info=0
    flag_hist_impact_seg = 0
    
    track_method= 'StrongBdip'
    
    B = 0.   #Tesla (if B=-1 computed from energy and bending radius)
    bm_totlen= -1 #m
 
   
    B0x = 0.
    B0y = 0.
    B0z = 0.
    B_map_file = None
    Bz_map_file = None
    N_sub_steps = 1
    fact_Bmap = 1.
    B_zero_thrhld = None
    
    
    # photoemission parameters
    photoem_flag = 0
    inv_CDF_refl_photoem_file = -1
    k_pe_st = -1
    refl_frac = -1
    alimit= -1
    e_pe_sigma = -1
    e_pe_max = -1
    x0_refl = -1
    y0_refl = -1
    out_radius = -1
    
    # gas ionization parameters
    gas_ion_flag = 0
    P_nTorr=-1
    sigma_ion_MBarn=-1
    Temp_K=-1
    unif_frac=-1
    E_init_ion=-1
    
    N_mp_soft_regen = None
    N_mp_after_soft_regen = None
    Dx = 0.
    Dy = 0.
    betafx = None
    betafy = None

    
    flag_verbose_file=False
    flag_verbose_stdout=False
    
    secondary_beams_file_list = []
    
    phem_resc_fac = 0.9999 
    
    dec_fac_secbeam_prof=1
    
    el_density_probes=[]
    
    save_simulation_state_time_file = -1
    
    # detailed histogram
    x_min_hist_det=None
    x_max_hist_det=None
    y_min_hist_det=None
    y_max_hist_det=None
    Dx_hist_det=None
    
    
    f=open(simulation_param_file)
    exec(f.read())
    f.close()  
    
    
    f=open(pyecl_input_folder+'/'+machine_param_file)
    exec(f.read())
    f.close() 
    
    f=open(pyecl_input_folder+'/'+secondary_emission_parameters_file)
    exec(f.read())
    f.close()  
    
    b_par = None# = pbf.beam_descr_from_fil(beam_parameters_file, betafx, Dx, betafy, Dy)
    
    flag_presence_sec_beams = False
    #~ if len(secondary_beams_file_list)>0:
        #~ flag_presence_sec_beams = True
    
    sec_b_par_list=[]
    #~ if flag_presence_sec_beams:
        #~ for sec_b_file in secondary_beams_file_list:
            #~ sec_b_par_list.append(pbf.beam_descr_from_fil(sec_b_file, betafx, Dx, betafy, Dy))
        
    if B==-1:
        B   = 2*pi*b_par.beta_rel*b_par.energy_J/(c*qe*bm_totlen) 
        
    
    
    return b_par, x_aper, y_aper, B,\
    gas_ion_flag, P_nTorr, sigma_ion_MBarn, Temp_K, unif_frac, E_init_ion,\
    Emax, del_max, R0, E_th, sigmafit, mufit,\
    Dt, t_end, lam_th, t_ion, N_mp_max,\
    N_mp_regen, N_mp_after_regen, fact_split, fact_clean, nel_mp_ref_0,\
    Nx_regen, Ny_regen, Nvx_regen, Nvy_regen, Nvz_regen,regen_hist_cut,\
    N_mp_regen_low,\
    Dt_sc, Dh_sc, t_sc_ON,Dx_hist,r_center, scrub_en_th,\
    progress_path,  logfile_path, flag_movie, flag_sc_movie,\
    Dt_En_hist, Nbin_En_hist,En_hist_max, \
    photoem_flag, inv_CDF_refl_photoem_file, k_pe_st, refl_frac, alimit, e_pe_sigma,\
    e_pe_max,x0_refl, y0_refl, out_radius, \
    switch_model, switch_no_increase_energy, thresh_low_energy, save_mp_state_time_file, \
    init_unif_flag, Nel_init_unif, E_init_unif, x_max_init_unif, x_min_init_unif, y_max_init_unif, y_min_init_unif,\
    chamb_type, filename_chm, flag_detailed_MP_info, flag_hist_impact_seg,\
    track_method, B0x, B0y, B0z, B_map_file,  Bz_map_file, N_sub_steps, fact_Bmap, B_zero_thrhld,\
    N_mp_soft_regen, N_mp_after_soft_regen,\
    flag_verbose_file, flag_verbose_stdout,\
    flag_presence_sec_beams, sec_b_par_list, phem_resc_fac, dec_fac_secbeam_prof, el_density_probes, save_simulation_state_time_file,\
    x_min_hist_det, x_max_hist_det, y_min_hist_det, y_max_hist_det, Dx_hist_det, dec_fact_out, stopfile			
			
    
		
		
