import numpy as np
from scipy.constants import c, e

class TransverseSpaceCharge(object):
    def __init__(self, L_interaction, slicer, pyPICsolver, flag_clean_slices = False):


        self.slicer = slicer
        self.L_interaction = L_interaction
        self.pyPICsolver = pyPICsolver

        self.save_distributions_last_track = False
        self.save_potential_and_field = False

        self.flag_clean_slices = flag_clean_slices

#	@profile
    def track(self, beam):


        if self.save_distributions_last_track:
            self.rho_last_track = []

        if self.save_potential_and_field:
            self.phi_last_track = []
            self.Ex_last_track = []
            self.Ey_last_track = []

        if hasattr(beam.particlenumber_per_mp, '__iter__'):
            raise ValueError('spacecharge module assumes same size for all beam MPs')

        if self.flag_clean_slices:
            beam.clean_slices()

        slices = beam.get_slices(self.slicer)

        for i in xrange(slices.n_slices-1, -1, -1):

            # select particles in the slice
            ix = slices.particle_indices_of_slice(i)

            # slice size and time step
            dz = (slices.z_bins[i + 1] - slices.z_bins[i])
            dt = dz / (beam.beta * c)

            # beam field
            x_mp = beam.x[ix]
            y_mp = beam.y[ix]
            n_mp = beam.x[ix]*0.+beam.particlenumber_per_mp/dz#they have to become cylinders
            N_mp = slices.n_macroparticles_per_slice[i]

            #compute beam field (it assumes electrons!)
            self.pyPICsolver.scatter_and_solve(x_mp, y_mp, n_mp, beam.charge)

            #gather to beam particles
            Ex_n_beam, Ey_n_beam = self.pyPICsolver.gather(x_mp, y_mp)


            # go to actual beam particles and add relativistic (B field) factor
            Ex_n_beam = Ex_n_beam/beam.gamma/beam.gamma
            Ey_n_beam = Ey_n_beam/beam.gamma/beam.gamma


            ## kick beam particles
            fact_kick = beam.charge/(beam.mass*beam.beta*beam.beta*beam.gamma*c*c)*self.L_interaction
            beam.xp[ix]+=fact_kick*Ex_n_beam
            beam.yp[ix]+=fact_kick*Ex_n_beam

            if self.save_distributions_last_track:
                self.rho_last_track.append(self.pyPICsolver.rho.copy())
                #print 'Here'

            if self.save_potential_and_field:
                self.phi_last_track.append(self.pyPICsolver.phi.copy())
                self.Ex_last_track.append(self.pyPICsolver.efx.copy()/beam.gamma/beam.gamma)
                self.Ey_last_track.append(self.pyPICsolver.efy.copy()/beam.gamma/beam.gamma)


        if self.save_distributions_last_track:
            self.rho_last_track = np.array(self.rho_last_track[::-1])

        if self.save_potential_and_field:
            self.phi_last_track = np.array(self.phi_last_track[::-1])
            self.Ex_last_track = np.array(self.Ex_last_track[::-1])
            self.Ey_last_track = np.array(self.Ey_last_track[::-1])
