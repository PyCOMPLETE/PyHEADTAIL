import numpy as np
from scipy.constants import c, e

from PyHEADTAIL.particles.slicing import UniformBinSlicer
from PyPIC.PyPIC_Scatter_Gather import PyPIC_Scatter_Gather



class Transverse_Efield_map(object):
	def __init__(self, xg, yg, Ex, Ey, n_slices, z_cut,
		L_interaction, flag_clean_slices = False, wrt_slice_centroid = False):

		self.slicer = UniformBinSlicer(n_slices = n_slices, z_cuts=(-z_cut, z_cut))
		self.L_interaction = L_interaction
		self.flag_clean_slices = flag_clean_slices
		self.wrt_slice_centroid = wrt_slice_centroid

		self.Ex = Ex
		self.Ey = Ey
		self.pic = PyPIC_Scatter_Gather(xg=xg, yg=yg)


	def get_beam_x(self, beam):
		return beam.x

	def get_beam_y(self, beam):
		return beam.y

#	@profile
	def track(self, beam):


		if self.flag_clean_slices:
			beam.clean_slices()

		slices = beam.get_slices(self.slicer)


		for i in xrange(slices.n_slices-1, -1, -1):

			# select particles in the slice
			ix = slices.particle_indices_of_slice(i)

			# slice size and time step
			dz = (slices.z_bins[i + 1] - slices.z_bins[i])
			#dt = dz / (beam.beta * c)

			x = self.get_beam_x(beam)[ix]
			y = self.get_beam_y(beam)[ix]

			self.pic.efx = np.squeeze(self.Ex[i,:,:])
			self.pic.efy = np.squeeze(self.Ey[i,:,:])
			if self.wrt_slice_centroid:
				Ex_sc_p, Ey_sc_p = self.pic.gather(x-np.mean(x), y-np.mean(y))
			else:
				Ex_sc_p, Ey_sc_p = self.pic.gather(x, y)

			## kick beam particles
			fact_kick = beam.charge/(beam.mass*beam.beta*beam.beta*beam.gamma*c*c)*self.L_interaction
			beam.xp[ix]+=fact_kick*Ex_sc_p
			beam.yp[ix]+=fact_kick*Ey_sc_p







