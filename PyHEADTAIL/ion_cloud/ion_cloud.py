from PyHEADTAIL.field_maps import efields_funcs as efields
from PyHEADTAIL.general.element import Element
from PyHEADTAIL.particles import particles, generators
# from PyHEADTAIL.particles.slicing import UniformBinSlicer
from PyHEADTAIL.general import pmath as pm
from PyHEADTAIL.general.printers import SilentPrinter
from scipy.constants import m_p, e, c


CIRCUMFERENCE = 354
N_SEGMENTS = 500


class BeamIonElement(Element):
    def __init__(self,  sig_check=True, dist='GS', monitor_on=True, L_sep=0.85, n_macroparticles_max=int(1e3)):
        self.ion_beam = None
        self.dist = dist
        if dist == 'GS':
            self._efieldn = efields._efieldn_mit
            self.sig_check = sig_check
        elif dist == 'LN':
            self._efieldn = efields._efieldn_linearized
        self.L_sep = L_sep
        self.N_MACROPARTICLES = 50
        self.N_MACROPARTICLES_MAX = n_macroparticles_max
        self.CIRCUMFERENCE = 354
        self.N_SEGMENTS = 500
        if sig_check:
            self._efieldn = efields.add_sigma_check(
                self._efieldn, self.dist)
        self.n_g = 2.4e13
        self.sigma_i = 1.8e-22
        self.A = 28
        self.charge_state = 1
        self.ion_beam = particles.Particles(
            macroparticlenumber=1,
            particlenumber_per_mp=1,
            charge=self.charge_state*e,
            mass=self.A*m_p,
            circumference=self.CIRCUMFERENCE,
            gamma=1.0001,
            coords_n_momenta_dict={
                'x': [0, ],
                'xp': [0, ],
                'y': [0, ],
                'yp': [0, ],
                'z': [0, ],
                'dp': [0, ]
            }
        )

    def track(self, electron_bunch):
        '''Tracking method to track an interaction between an electron bunch
        and an ion beam (2D electromagnetic field).
        The kicks are performed both for electron beam slice and for an ion beam. 
        Ion beam is tracked in a drift/space-charge of electron bunch sections. 

        Interaction is computed via Eqs. (17, 18) of 

        Tian, S. K.; Wang, N. (2018). Ion instability in the HEPS storage ring.
        FLS 2018 - Proceedings of the 60th ICFA Advanced Beam Dynamics Workshop on Future Light Sources,
        34–38. https://doi.org/10.18429/JACoW-FLS2018-TUA2WB04
        '''
        if self.ion_beam.macroparticlenumber < self.N_MACROPARTICLES_MAX:
            new_particles = generators.ParticleGenerator(
                macroparticlenumber=self.N_MACROPARTICLES,
                intensity=electron_bunch.intensity*self.sigma_i *
                self.n_g*self.CIRCUMFERENCE/self.N_SEGMENTS,
                charge=self.charge_state*e,
                gamma=1.0001,
                mass=self.A*m_p,
                circumference=self.CIRCUMFERENCE,
                distribution_x=generators.uniform2D(
                    -2*electron_bunch.sigma_x(), 2*electron_bunch.sigma_x()),
                distribution_y=generators.uniform2D(
                    -2*electron_bunch.sigma_y(), 2*electron_bunch.sigma_y()),
                distribution_z=generators.uniform2D(
                    0, self.CIRCUMFERENCE/self.N_SEGMENTS),
                printer=SilentPrinter()
            ).generate()
            new_particles.x[:] += electron_bunch.mean_x()
            new_particles.y[:] += electron_bunch.mean_y()
            self.ion_beam += new_particles
        else:
            self.ion_beam.intensity += electron_bunch.intensity * \
                self.sigma_i*self.n_g*self.CIRCUMFERENCE/self.N_SEGMENTS
        prefactor_kick_ion_field = -(self.ion_beam.intensity *
                                     self.ion_beam.charge**2 /
                                     (electron_bunch.p0*electron_bunch.beta*c))
        prefactor_kick_electron_field = -(electron_bunch.intensity *
                                          electron_bunch.charge**2 /
                                          (c))
        p_id_electrons = electron_bunch.id-1
        p_id_ions = self.ion_beam.id-1
#         if len(p_id_electrons) == 0:
#                 continue
# Electric field of ions
        en_ions_x, en_ions_y = self.get_efieldn(
            pm.take(electron_bunch.x, p_id_electrons),
            pm.take(electron_bunch.y, p_id_electrons),
            self.ion_beam.mean_x(), self.ion_beam.mean_y(),
            self.ion_beam.sigma_x(), self.ion_beam.sigma_y())
# Electric field of electrons
        en_electrons_x, en_electrons_y = self.get_efieldn(
            pm.take(self.ion_beam.x, p_id_ions),
            pm.take(self.ion_beam.y, p_id_ions),
            electron_bunch.mean_x(),  electron_bunch.mean_y(),
            electron_bunch.sigma_x(), electron_bunch.sigma_y()
        )
        kicks_electrons_x = en_ions_x * prefactor_kick_ion_field
        kicks_electrons_y = en_ions_y * prefactor_kick_ion_field
        kicks_ions_x = en_electrons_x * prefactor_kick_electron_field
        kicks_ions_y = en_electrons_y * prefactor_kick_electron_field

        kicked_electrons_xp = pm.take(
            electron_bunch.xp, p_id_electrons) + kicks_electrons_x
        kicked_electrons_yp = pm.take(
            electron_bunch.yp, p_id_electrons) + kicks_electrons_y

        kicked_ions_xp = pm.take(self.ion_beam.xp, p_id_ions) + kicks_ions_x
        kicked_ions_yp = pm.take(self.ion_beam.yp, p_id_ions) + kicks_ions_y

        pm.put(electron_bunch.xp, p_id_electrons, kicked_electrons_xp)
        pm.put(electron_bunch.yp, p_id_electrons, kicked_electrons_yp)

        pm.put(self.ion_beam.xp, p_id_ions, kicked_ions_xp)
        pm.put(self.ion_beam.yp, p_id_ions, kicked_ions_yp)
        # Drift for the ions in one bucket
        drifted_ions_x = pm.take(self.ion_beam.xp, p_id_ions)*self.L_sep / \
            (self.ion_beam.mass*c)+pm.take(self.ion_beam.x, p_id_ions)
        drifted_ions_y = pm.take(self.ion_beam.yp, p_id_ions)*self.L_sep / \
            (self.ion_beam.mass*c)+pm.take(self.ion_beam.y, p_id_ions)
        pm.put(self.ion_beam.x, p_id_ions, drifted_ions_x)
        pm.put(self.ion_beam.y, p_id_ions, drifted_ions_y)

    def get_efieldn(self, xr, yr, mean_x, mean_y, sig_x, sig_y):
        '''The charge-normalised electric field components of a
        two-dimensional Gaussian charge distribution according to
        M. Bassetti and G. A. Erskine in CERN-ISR-TH/80-06.

        Return (E_x / Q, E_y / Q).
        '''
        x = xr - mean_x
        y = yr - mean_y

        # absolute values for convergence reasons of erfc
        en_x, en_y = self._efieldn(pm.abs(x), pm.abs(y), sig_x, sig_y)
        en_x = pm.abs(en_x) * pm.sign(x)
        en_y = pm.abs(en_y) * pm.sign(y)

        return en_x, en_y
