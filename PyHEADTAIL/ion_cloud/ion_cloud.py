import PyHEADTAIL.aperture.aperture as aperture
from PyHEADTAIL.field_maps import efields_funcs as efields
from PyHEADTAIL.general import pmath as pm
from PyHEADTAIL.general.element import Element
from PyHEADTAIL.general.printers import SilentPrinter
from PyHEADTAIL.monitors.monitors import BunchMonitor, ParticleMonitor
from PyHEADTAIL.particles import particles, generators
from scipy.constants import m_p, e, c
from numpy import linspace, int64

H_RF = 416
CIRCUMFERENCE = 354
N_SEGMENTS = 500
N_TURNS = 1000


class BeamIonElement(Element):
    def __init__(self,  sig_check=True,
                 dist_ions='GS',
                 monitor_name=None,
                 particle_monitor=False,
                 L_sep=0.85,
                 n_macroparticles_max=int(1e3),
                 set_aperture=True,
                 n_segments=500,
                 circumference=354,
                 n_steps=None,
                 interaction_model='weak-weak'
                 ):
        self.ion_beam = None
        self.dist = dist_ions
        self.dist_func_z = generators.uniform2D
        if self.dist == 'GS':
            self._efieldn = efields._efieldn_mit
            self.sig_check = sig_check
            self.dist_func_x = generators.gaussian2D_asymmetrical
            self.dist_func_y = generators.gaussian2D_asymmetrical
        elif self.dist == 'LN':
            self._efieldn = efields._efieldn_linearized
            self.dist_func_x = generators.uniform2D
            self.dist_func_y = generators.uniform2D
        else:
            print('Distribution given is not implemented')
        self.interaction_model = interaction_model
        self.set_aperture = set_aperture
        self.L_sep = L_sep
        self.N_MACROPARTICLES = 30
        self.N_MACROPARTICLES_MAX = n_macroparticles_max
        self.CIRCUMFERENCE = circumference
        self.N_SEGMENTS = n_segments
        self.L_SEG = self.CIRCUMFERENCE/self.N_SEGMENTS
        if sig_check:
            self._efieldn = efields.add_sigma_check(
                self._efieldn, self.dist)
        self.n_g = 2.4e13  # (m**-3)
        self.sigma_i = 1.8e-22  # (m**2)
        self.A = 28
        self.n_steps = n_steps
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
        if particle_monitor:
            self.particle_monitor = particle_monitor
            self.ions_monitor = ParticleMonitor(monitor_name,
                                                stride=1,
                                                parameters_dict=None
                                                )
        elif monitor_name is not None:
            self.monitor_name = monitor_name
            self.ions_monitor = BunchMonitor(monitor_name,
                                             n_steps=self.n_steps,
                                             parameters_dict=None,
                                             write_buffer_every=50,
                                             buffer_size=100,
                                             )
        else:
            self.ions_monitor = None

    def get_ion_beam(self):
        return self.ion_beam

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
        ION_INTENSITY_PER_ELECTRON_BUNCH = electron_bunch.intensity * \
            self.sigma_i*self.n_g*self.L_SEG
        self.N_MACROPARTICLES = 50  # int(ION_INTENSITY_PER_ELECTRON_BUNCH)
        assert (self.dist in ['LN', 'GS']), (
            'The implementation for required distribution {:} is not found'.format(self.dist))
        if self.dist == 'LN':
            a_x, b_x = -2*electron_bunch.sigma_x(), 2*electron_bunch.sigma_x()
            a_y, b_y = -2*electron_bunch.sigma_y(), 2*electron_bunch.sigma_y()
        elif self.dist == 'GS':
            a_x, b_x = electron_bunch.sigma_x(), electron_bunch.sigma_xp()
            a_y, b_y = electron_bunch.sigma_y(), electron_bunch.sigma_yp()
        '''
        Particles are generated in pairs -x, -y and +x, +y to avoid numerical noise.
        The idea came from Blaskiewicz, M. (2019) https://doi.org/10.18429/JACoW-NAPAC2019-TUPLM11
        '''
        if self.ion_beam.macroparticlenumber < self.N_MACROPARTICLES_MAX:
            new_particles = generators.ParticleGenerator(
                macroparticlenumber=self.N_MACROPARTICLES//2,
                intensity=ION_INTENSITY_PER_ELECTRON_BUNCH//2,
                charge=self.charge_state*e,
                gamma=1.0001,
                mass=self.A*m_p,
                circumference=self.CIRCUMFERENCE,
                distribution_x=self.dist_func_x(a_x, b_x),
                distribution_y=self.dist_func_y(a_y, b_y),
                distribution_z=self.dist_func_z(
                    0, self.L_SEG),
                limit_n_rms_x=3.,
                limit_n_rms_y=3.,
                printer=SilentPrinter()
            ).generate()
            new_particles_twin = particles.Particles(
                macroparticlenumber=self.N_MACROPARTICLES//2,
                particlenumber_per_mp=ION_INTENSITY_PER_ELECTRON_BUNCH/self.N_MACROPARTICLES,
                charge=self.charge_state*e,
                gamma=1.0001,
                mass=self.A*m_p,
                circumference=self.CIRCUMFERENCE,
                coords_n_momenta_dict={
                    'x': -new_particles.x,
                    'xp': -new_particles.xp,
                    'y': -new_particles.y,
                    'yp': -new_particles.yp,
                    'z': -new_particles.z,
                    'dp': -new_particles.dp
                },
                printer=SilentPrinter()
            )
            new_particles += new_particles_twin
            # Initial conditions
            new_particles.x[:] += electron_bunch.mean_x()
            new_particles.y[:] += electron_bunch.mean_y()
            new_particles.xp[:] = 0
            new_particles.yp[:] = 0
            self.ion_beam += new_particles
        else:
            self.ion_beam.intensity += ION_INTENSITY_PER_ELECTRON_BUNCH
        if self.ions_monitor is not None:
            self.ions_monitor.dump(self.ion_beam)
        prefactor_kick_ion_field = -(self.ion_beam.intensity *
                                     self.ion_beam.charge*electron_bunch.charge /
                                     (electron_bunch.p0*electron_bunch.beta*c))*self.L_SEG/self.L_sep
        prefactor_kick_electron_field = -(electron_bunch.intensity *
                                          electron_bunch.charge*self.ion_beam.charge /
                                          (self.ion_beam.mass*c**2))
        # Clearing ions that are far away
        apt_xy = aperture.EllipticalApertureXY(
            x_aper=5*electron_bunch.sigma_x(),
            y_aper=5*electron_bunch.sigma_y())
        if self.set_aperture == True:
            apt_xy.track(self.ion_beam)
        else:
            pass
        p_id_electrons = electron_bunch.id-1
        p_id_ions = linspace(
            0, self.ion_beam.y.shape[0]-1, self.ion_beam.y.shape[0], dtype=int64)
# Electric field of ions
        if self.interaction_model == 'weak-weak':
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
        elif self.interaction_model == 'weak-strong':
            en_ions_x, en_ions_y = self.get_efieldn(
                electron_bunch.mean_x(),
                electron_bunch.mean_y(),
                self.ion_beam.mean_x(), self.ion_beam.mean_y(),
                self.ion_beam.sigma_x(), self.ion_beam.sigma_y())
# Electric field of electrons
            en_electrons_x, en_electrons_y = self.get_efieldn(
                pm.take(self.ion_beam.x, p_id_ions),
                pm.take(self.ion_beam.y, p_id_ions),
                electron_bunch.mean_x(),  electron_bunch.mean_y(),
                electron_bunch.sigma_x(), electron_bunch.sigma_y()
            )
        elif self.interaction_model == 'strong-strong':
            en_ions_x, en_ions_y = self.get_efieldn(
                electron_bunch.mean_x(),
                electron_bunch.mean_y(),
                self.ion_beam.mean_x(), self.ion_beam.mean_y(),
                self.ion_beam.sigma_x(), self.ion_beam.sigma_y())
# Electric field of electrons
            en_electrons_x, en_electrons_y = self.get_efieldn(
                self.ion_beam.mean_x(),
                self.ion_beam.mean_x(),
                electron_bunch.mean_x(),  electron_bunch.mean_y(),
                electron_bunch.sigma_x(), electron_bunch.sigma_y()
            )
        else:
            pass
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
        drifted_ions_x = pm.take(
            self.ion_beam.xp, p_id_ions)*self.L_sep + pm.take(self.ion_beam.x, p_id_ions)
        drifted_ions_y = pm.take(
            self.ion_beam.yp, p_id_ions)*self.L_sep + pm.take(self.ion_beam.y, p_id_ions)
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
