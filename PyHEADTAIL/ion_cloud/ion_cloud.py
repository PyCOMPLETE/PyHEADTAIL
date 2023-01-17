import PyHEADTAIL.aperture.aperture as aperture
from PyHEADTAIL.field_maps import efields_funcs as efields
from PyHEADTAIL.general import pmath as pm
from PyHEADTAIL.general.element import Element
from PyHEADTAIL.general.printers import SilentPrinter
from PyHEADTAIL.monitors.monitors import BunchMonitor, ParticleMonitor
from PyHEADTAIL.particles import particles, generators
from scipy.constants import m_p, e, c
from numpy import linspace, int64
import PyPIC.geom_impact_ellip as ell
import PyPIC.FFT_OpenBoundary as PIC_FFT


H_RF = 416
CIRCUMFERENCE = 354
N_SEGMENTS = 500
N_TURNS = 1000


class BeamIonElement(Element):
    '''
    It has various attributes and methods that allow to initialize the ion beam properties,
    create an instance of `Particles` class which represent the ion beam, 
    create an instance of `BunchMonitor` or `ParticleMonitor` classes to monitor the ion beam,
    and track the interaction between the electron bunch and the ion beam.

    Attributes:
        ion_beam (Particles): An instance of `Particles` class that represents the ion beam
        dist (str): The distribution of ions in the beam (default is 'GS')
        dist_func_z (func): A function that generates the z distribution of ions
        _efieldn (func): A function that generates the electric field of the electron bunch
        sig_check (bool): A boolean to specify if sigma check for the electron field is activated
        dist_func_x (func): A function that generates the x distribution of ions
        dist_func_y (func): A function that generates the y distribution of ions
        interaction_model (str): A string that sets the interaction model between the electron and the ion bunches
        set_aperture (bool): A boolean to specify if the aperture should be set
        L_sep (float): A scalar value that gives the distance between the electron and the ion bunches
        N_MACROPARTICLES (int): The number of macroparticles in the ion beam
        N_MACROPARTICLES_MAX (int): The maximum number of macroparticles in the ion beam
        CIRCUMFERENCE (float): The circumference of the ion beam
        N_SEGMENTS (int): The number of segments for the ion beam
        L_SEG (float): The length of each segment of the ion beam
        n_g (float): The residual gas density in the vacuum chamber
        sigma_i (float): Ionization cross-section of ions
        A (float): The mass number of the ions
        n_steps (int): The number of tracking steps for the monitor
        charge_state (int): The charge state of the ions
        ions_monitor (Union[BunchMonitor, ParticleMonitor, None]): An instance of `BunchMonitor` or `ParticleMonitor`
            classes that monitor the ion beam
    '''

    def __init__(self,  sig_check=True,
                 dist_ions='GS',
                 monitor_name=None,
                 use_particle_monitor=False,
                 L_sep=0.85,
                 n_macroparticles_max=int(1e3),
                 set_aperture=True,
                 n_segments=500,
                 ring_circumference=354,
                 n_steps=None,
                 interaction_model='weak-weak'
                 ):
        self.use_particle_monitor = use_particle_monitor
        self.dist = dist_ions
        self.monitor_name = monitor_name
        self.L_sep = L_sep
        self.N_MACROPARTICLES_MAX = n_macroparticles_max
        self.set_aperture = set_aperture
        self.n_segments = n_segments
        self.ring_circumference = ring_circumference
        self.n_steps = n_steps
        self.interaction_model = interaction_model
        self._set_distribution_for_particle_generation()
        self.N_MACROPARTICLES = 30
        self.L_SEG = self.ring_circumference/self.n_segments
        self.n_g = 2.4e13  # (m**-3)
        self.sigma_i = 1.8e-22  # (m**2)
        self.A = 28
        self.charge_state = 1
        self.ion_beam = particles.Particles(
            macroparticlenumber=1,
            particlenumber_per_mp=1,
            charge=self.charge_state*e,
            mass=self.A*m_p,
            circumference=self.ring_circumference,
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
        self._add_monitors()

    def _set_distribution_for_particle_generation(self):
        self.dist_func_z = generators.uniform2D
        if self.dist == 'GS':
            self._efieldn = efields._efieldn_mit
            self.dist_func_x = generators.gaussian2D_asymmetrical
            self.dist_func_y = generators.gaussian2D_asymmetrical
        elif self.dist == 'LN':
            self._efieldn = efields._efieldn_linearized
            self.dist_func_x = generators.uniform2D
            self.dist_func_y = generators.uniform2D
        else:
            print('Distribution given is not implemented')
        self._efieldn = efields.add_sigma_check(
            self._efieldn, self.dist)

    def _add_monitors(self):
        if self.monitor_name is not None:
            if self.use_particle_monitor:
                self.ions_monitor = ParticleMonitor(self.monitor_name,
                                                    stride=1,
                                                    parameters_dict=None
                                                    )
            else:
                self.ions_monitor = BunchMonitor(self.monitor_name,
                                                 n_steps=self.n_steps,
                                                 parameters_dict=None,
                                                 write_buffer_every=50,
                                                 buffer_size=100,
                                                 )
        else:
            self.ions_monitor = None

    def get_ion_beam(self):
        """
        A method to access the ion beam object
        """
        return self.ion_beam

    def clear_ions(self):
        self.ion_beam = self.ion_beam = particles.Particles(
            macroparticlenumber=1,
            particlenumber_per_mp=1,
            charge=self.charge_state*e,
            mass=self.A*m_p,
            circumference=self.ring_circumference,
            gamma=1.0001,
            coords_n_momenta_dict={
                'x': [0, ],
                'xp': [0, ],
                'y': [0, ],
                'yp': [0, ],
                'z': [0, ],
                'dp': [0, ]
            })

    def _generate_ions(self, electron_bunch):
        '''
        Particles are generated in pairs -x, -y and +x, +y to avoid numerical noise.
        The idea came from Blaskiewicz, M. (2019) https://doi.org/10.18429/JACoW-NAPAC2019-TUPLM11
        '''
        assert (self.dist in ['LN', 'GS']), (
            'The implementation for required distribution {:} is not found'.format(self.dist))
        if self.dist == 'LN':
            a_x, b_x = -2*electron_bunch.sigma_x(), 2*electron_bunch.sigma_x()
            a_y, b_y = -2*electron_bunch.sigma_y(), 2*electron_bunch.sigma_y()
        elif self.dist == 'GS':
            a_x, b_x = electron_bunch.sigma_x(), electron_bunch.sigma_xp()
            a_y, b_y = electron_bunch.sigma_y(), electron_bunch.sigma_yp()
        new_particles = generators.ParticleGenerator(
            macroparticlenumber=self.N_MACROPARTICLES//2,
            intensity=self.ION_INTENSITY_PER_ELECTRON_BUNCH//2,
            charge=self.charge_state*e,
            gamma=1.0001,
            mass=self.A*m_p,
            circumference=self.ring_circumference,
            distribution_x=self.dist_func_x(a_x, b_x),
            distribution_y=self.dist_func_y(a_y, b_y),
            distribution_z=self.dist_func_z(
                -self.L_SEG/2, self.L_SEG/2),
            limit_n_rms_x=3.,
            limit_n_rms_y=3.,
            printer=SilentPrinter()
        ).generate()
        new_particles_twin = particles.Particles(
            macroparticlenumber=self.N_MACROPARTICLES//2,
            particlenumber_per_mp=self.ION_INTENSITY_PER_ELECTRON_BUNCH/self.N_MACROPARTICLES,
            charge=self.charge_state*e,
            gamma=1.0001,
            mass=self.A*m_p,
            circumference=self.ring_circumference,
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
        # Apply initial conditions
        new_particles.x[:] += electron_bunch.mean_x()
        new_particles.y[:] += electron_bunch.mean_y()
        new_particles.xp[:] = 0
        new_particles.yp[:] = 0
        self.ion_beam += new_particles
        self.ions_aperture = aperture.EllipticalApertureXY(
            x_aper=5*electron_bunch.sigma_x(),
            y_aper=5*electron_bunch.sigma_y())

    def track_ions_in_drift(self, p_id_ions):
        drifted_ions_x = pm.take(
            self.ion_beam.xp, p_id_ions)*self.L_sep + pm.take(self.ion_beam.x, p_id_ions)
        drifted_ions_y = pm.take(
            self.ion_beam.yp, p_id_ions)*self.L_sep + pm.take(self.ion_beam.y, p_id_ions)
        pm.put(self.ion_beam.x, p_id_ions, drifted_ions_x)
        pm.put(self.ion_beam.y, p_id_ions, drifted_ions_y)

    def get_updated_ion_positions(self, electron_bunch):
        pass

    def _get_efields(self, first_beam, second_beam, p_id_first_beam, interaction_model='weak'):
        assert (interaction_model in ['weak', 'strong', 'PIC']), ((
            'The implementation for required beam-ion interaction model {:} is not implemented'.format(self, interaction_model)))
        if interaction_model == 'weak':
            en_x, en_y = self.get_efieldn(
                pm.take(first_beam.x, p_id_first_beam),
                pm.take(first_beam.y, p_id_first_beam),
                second_beam.mean_x(), second_beam.mean_y(),
                second_beam.sigma_x(), second_beam.sigma_y())
        elif interaction_model == 'strong':
            en_x, en_y = self.get_efieldn(
                first_beam.mean_x(),
                first_beam.mean_y(),
                second_beam.mean_x(), second_beam.mean_y(),
                second_beam.sigma_x(), second_beam.sigma_y())
        if interaction_model == 'PIC':
            qe = 1.602176565e-19
            eps0 = 8.8541878176e-12
            Dx = 0.1*second_beam.sigma_x()
            Dy = 0.1*second_beam.sigma_y()
            x_aper = 10*second_beam.sigma_x()
            y_aper = 10*second_beam.sigma_y()
            chamber = ell.ellip_cham_geom_object(x_aper=x_aper, y_aper=y_aper)
            picFFT = PIC_FFT.FFT_OpenBoundary(
                x_aper=chamber.x_aper, y_aper=chamber.y_aper, dx=Dx, dy=Dy, fftlib='pyfftw')
            nel_part = 0*second_beam.x+1.
            picFFT.scatter(second_beam.x, second_beam.y, nel_part)
            picFFT.solve()
            en_x, en_y = picFFT.gather(
                first_beam.x, first_beam.y)/second_beam.x.shape[0]
        return en_x, en_y

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
        self.ION_INTENSITY_PER_ELECTRON_BUNCH = electron_bunch.intensity * \
            self.sigma_i*self.n_g*self.L_SEG

        if self.ion_beam.macroparticlenumber < self.N_MACROPARTICLES_MAX:
            self._generate_ions(electron_bunch)
        else:
            self.ion_beam.intensity += self.ION_INTENSITY_PER_ELECTRON_BUNCH

        if self.set_aperture == True:
            self.ions_aperture.track(self.ion_beam)

        if self.ions_monitor is not None:
            self.ions_monitor.dump(self.ion_beam)

        prefactor_kick_ion_field = -(self.ion_beam.intensity *
                                     self.ion_beam.charge*electron_bunch.charge /
                                     (electron_bunch.p0*electron_bunch.beta*c))
        prefactor_kick_electron_field = -(electron_bunch.intensity *
                                          electron_bunch.charge*self.ion_beam.charge /
                                          (self.ion_beam.mass*c**2))
        p_id_electrons = electron_bunch.id-1
        p_id_ions = linspace(
            0, self.ion_beam.y.shape[0]-1, self.ion_beam.y.shape[0], dtype=int64)
        en_ions_x, en_ions_y = self._get_efields(first_beam=electron_bunch,
                                                 second_beam=self.ion_beam,
                                                 p_id_first_beam=p_id_electrons,
                                                 interaction_model='weak')
        en_electrons_x, en_electrons_y = self._get_efields(first_beam=self.ion_beam,
                                                           second_beam=electron_bunch,
                                                           p_id_first_beam=p_id_ions,
                                                           interaction_model='weak')

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
        # drifted_ions_x = pm.take(
        #     self.ion_beam.xp, p_id_ions)*self.L_sep + pm.take(self.ion_beam.x, p_id_ions)
        # drifted_ions_y = pm.take(
        #     self.ion_beam.yp, p_id_ions)*self.L_sep + pm.take(self.ion_beam.y, p_id_ions)
        # pm.put(self.ion_beam.x, p_id_ions, drifted_ions_x)
        # pm.put(self.ion_beam.y, p_id_ions, drifted_ions_y)
        self.track_ions_in_drift(p_id_ions)

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
