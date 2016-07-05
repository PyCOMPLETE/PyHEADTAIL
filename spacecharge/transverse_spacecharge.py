from __future__ import division

from . import Element

import numpy as np
from scipy.constants import c

class TransverseSpaceCharge(Element):
    def __init__(self, L_interaction, slicer, pyPICsolver,
                 flag_clean_slices=False, *args, **kwargs):
        self.slicer = slicer
        self.L_interaction = L_interaction
        self.pyPICsolver = pyPICsolver

        self.save_distributions_last_track = False
        self.save_potential_and_field = False

        self.flag_clean_slices = flag_clean_slices

    def get_beam_x(self, beam):
        return beam.x

    def get_beam_y(self, beam):
        return beam.y

    def track(self, beam):
        if self.save_distributions_last_track:
            self.rho_last_track = []

        if self.save_potential_and_field:
            self.phi_last_track = []
            self.Ex_last_track = []
            self.Ey_last_track = []

        if hasattr(beam.particlenumber_per_mp, '__iter__'):
            raise ValueError('spacecharge module assumes same number of charges'
                             ' for all beam macroparticles!')

        if self.flag_clean_slices:
            beam.clean_slices()

        slices = beam.get_slices(self.slicer)

        for sid in xrange(slices.n_slices-1, -1, -1):

            # select particles in the slice
            pid = slices.particle_indices_of_slice(sid)

            # slice size
            dz = slices.slice_widths[sid]

            x = self.get_beam_x(beam)[pid]
            y = self.get_beam_y(beam)[pid]

            # the particles have to become cylinders:
            n_mp = np.zeros_like(x) + beam.particlenumber_per_mp / dz

            # compute beam field (it assumes electrons!)
            self.pyPICsolver.scatter_and_solve(x, y, n_mp, beam.charge)

            # interpolate beam field to particles
            Ex_n_beam, Ey_n_beam = self.pyPICsolver.gather(x, y)

            # go to actual beam particles and add relativistic (B field) factor
            Ex_n_beam = Ex_n_beam / beam.gamma / beam.gamma
            Ey_n_beam = Ey_n_beam / beam.gamma / beam.gamma

            # kick beam particles
            fact_kick = beam.charge / (beam.p0*beam.beta*c) * self.L_interaction
            beam.xp[pid] += fact_kick * Ex_n_beam
            beam.yp[pid] += fact_kick * Ey_n_beam

            if self.save_distributions_last_track:
                self.rho_last_track.append(self.pyPICsolver.rho.copy())
                #print 'Here'

            if self.save_potential_and_field:
                self.phi_last_track.append(self.pyPICsolver.phi.copy())
                self.Ex_last_track.append(
                    self.pyPICsolver.efx.copy()/beam.gamma/beam.gamma)
                self.Ey_last_track.append(
                    self.pyPICsolver.efy.copy()/beam.gamma/beam.gamma)

        if self.save_distributions_last_track:
            self.rho_last_track = np.array(self.rho_last_track[::-1])

        if self.save_potential_and_field:
            self.phi_last_track = np.array(self.phi_last_track[::-1])
            self.Ex_last_track = np.array(self.Ex_last_track[::-1])
            self.Ey_last_track = np.array(self.Ey_last_track[::-1])
