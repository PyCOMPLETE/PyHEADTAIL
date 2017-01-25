from PyHEADTAIL.machines.synchrotron import BasicSynchrotron
import numpy as np
from scipy.constants import c, e, m_e

class CLIC_DR(BasicSynchrotron):

	def __init__(self, machine_configuration=None, optics_mode='smooth', **kwargs):
		
		 
		alpha		= 0.0001276102729
		h_RF       	= 1465
		mass 		= m_e
		charge		= e
		RF_at		= 'middle'
		
		if machine_configuration=='3TeV':
			longitudinal_mode = 'non-linear'				
			p0 			= 2.86e9 * e /c
			p_increment = 0.
			accQ_x		= 48.35
			accQ_y		= 10.4
			V_RF		= 5.1e6
			dphi_RF		= 0.
                        Q_s             = None
		elif machine_configuration=='3TeV_linear':
			Q_s		  = 0.0059	
			longitudinal_mode = 'linear'				
			p0 		  = 2.86e9 * e /c
			p_increment       = 0.
			accQ_x		  = 48.35
			accQ_y		  = 10.4
			V_RF		  = 5.1e6
			dphi_RF		  = 0.
		else:
			raise ValueError('machine_configuration not recognized!')
			
		
		
		if optics_mode=='smooth':
			if 's' in kwargs.keys(): raise ValueError('s vector cannot be provided if optics_mode = "smooth"')
			
			n_segments = kwargs['n_segments']
			circumference = 427.5
			
			name = None
			
			beta_x 		= 3.475 
			D_x 		= 0
			beta_y 		= 9.233
			D_y 		= 0 
			
			alpha_x 	= None
			alpha_y 	= None
			
			s = None
			
		elif optics_mode=='non-smooth':
			if 'n_segments' in kwargs.keys(): raise ValueError('n_segments cannot be provided if optics_mode = "non-smooth"')
			n_segments = None
			circumference = None
			
			name		= kwargs['name']
			
			beta_x 		= kwargs['beta_x']
			beta_y 		= kwargs['beta_y'] 

			try:
				D_x 		= kwargs['D_x']	
			except KeyError:
				D_x 		= 0*np.array(kwargs['s'])
			try:
				D_y 		= kwargs['D_y']	
			except KeyError:
				D_y 		= 0*np.array(kwargs['s'])			
					
			alpha_x 	= kwargs['alpha_x']
			alpha_y 	= kwargs['alpha_y']
			
			s = kwargs['s']
		
		else:
			raise ValueError('optics_mode not recognized!')		

		
		# detunings
		Qp_x		= 0
		Qp_y		= 0
		
		app_x		= 0
		app_y		= 0
		app_xy		= 0
		

		
		for attr in kwargs.keys():
			if kwargs[attr] is not None:
				if type(kwargs[attr]) is list or type(kwargs[attr]) is np.ndarray:
					str2print = '[%s ...]'%repr(kwargs[attr][0])
				else:
					str2print = repr(kwargs[attr])
				self.prints('Synchrotron init. From kwargs: %s = %s'
							% (attr, str2print))
				temp =  kwargs[attr]
				exec('%s = temp'%attr)
				
		
		
		super(CLIC_DR, self).__init__(optics_mode=optics_mode, circumference=circumference, n_segments=n_segments, s=s, name=name,
             alpha_x=alpha_x, beta_x=beta_x, D_x=D_x, alpha_y=alpha_y, beta_y=beta_y, D_y=D_y,
             accQ_x=accQ_x, accQ_y=accQ_y, Qp_x=Qp_x, Qp_y=Qp_y, app_x=app_x, app_y=app_y, app_xy=app_xy,
             alpha_mom_compaction=alpha, longitudinal_mode=longitudinal_mode,
             h_RF=np.atleast_1d(h_RF), V_RF=np.atleast_1d(V_RF), dphi_RF=np.atleast_1d(dphi_RF), p0=p0, p_increment=p_increment,
             charge=charge, mass=mass, RF_at=RF_at, Q_s=Q_s)
		

