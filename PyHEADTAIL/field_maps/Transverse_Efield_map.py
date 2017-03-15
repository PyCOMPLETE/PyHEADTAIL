import numpy as np
from scipy.constants import c

from PyHEADTAIL.particles.slicing import UniformBinSlicer
from PyPIC.PyPIC_Scatter_Gather import PyPIC_Scatter_Gather

from . import Element

class Transverse_Efield_map(Element):
    def __init__(self, xg, yg, Ex, Ey, L_interaction, slicer,
        	flag_clean_slices=False, wrt_slice_centroid=False,
        	x_beam_offset=0., y_beam_offset=0., verbose=False,
            *args, **kwargs):

        self.slicer = slicer
        self.L_interaction = L_interaction
        self.flag_clean_slices = flag_clean_slices
        self.wrt_slice_centroid = wrt_slice_centroid

        self.Ex = Ex
        self.Ey = Ey
        self.pic = PyPIC_Scatter_Gather(xg=xg, yg=yg, verbose=verbose)

        self.x_beam_offset = x_beam_offset
        self.y_beam_offset = y_beam_offset

    def get_beam_x(self, beam):
        return beam.x

    def get_beam_y(self, beam):
        return beam.y

    def track(self, beam):
        if self.flag_clean_slices:
            beam.clean_slices()

        slices = beam.get_slices(self.slicer)

        for sid in xrange(slices.n_slices-1, -1, -1):

            # select particles in the slice
            pid = slices.particle_indices_of_slice(sid)

            # slice size
            dz = (slices.z_bins[sid + 1] - slices.z_bins[sid])

            x = self.get_beam_x(beam)[pid]
            y = self.get_beam_y(beam)[pid]

            self.pic.efx = np.squeeze(self.Ex[sid,:,:])
            self.pic.efy = np.squeeze(self.Ey[sid,:,:])

            centroid_x = 0
            centroid_y = 0
            if self.wrt_slice_centroid:
                centroid_x = np.mean(x)
                centroid_y = np.mean(y)

            Ex_sc_p, Ey_sc_p = self.pic.gather(
                x - centroid_x + self.x_beam_offset,
                y - centroid_y + self.y_beam_offset
            )

            # kick beam particles
            fact_kick = beam.charge / (beam.p0*beam.beta*c) * self.L_interaction
            beam.xp[pid] += fact_kick * Ex_sc_p
            beam.yp[pid] += fact_kick * Ey_sc_p
