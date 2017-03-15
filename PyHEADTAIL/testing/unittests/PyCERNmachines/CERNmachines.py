from __future__ import division

import numpy as np
from scipy.constants import c, e, m_p

from machines import Synchrotron
import SPS.SPSOctupoles as SPSOctupoles


class PSB(Synchrotron):

    def __init__(self, *args, **kwargs):

        if 'n_segments' not in kwargs.keys():
            raise ValueError('Number of segments must be specified')

        if 'machine_configuration' not in kwargs.keys():
            raise ValueError('machine_configuration must be specified')

        self.n_segments = kwargs['n_segments']
        self.machine_configuration = kwargs['machine_configuration']

        self.circumference  = 50*np.pi
        self.s = (np.arange(0, self.n_segments + 1)
                  * self.circumference / self.n_segments)

        if self.machine_configuration == '160MeV':
            self.charge = e
            self.mass = m_p

            self.gamma = 160e6*e/(self.mass*c**2) + 1

            self.Q_x     = 4.23
            self.Q_y     = 4.37

            self.Qp_x    = [-1*self.Q_x]
            self.Qp_y    = [-2*self.Q_y]

            self.app_x   = 0.0000e-9
            self.app_y   = 0.0000e-9
            self.app_xy  = 0

            self.alpha_x = 0 * np.ones(self.n_segments + 1)
            self.beta_x  = self.circumference/(2*np.pi*self.Q_x) * np.ones(self.n_segments + 1)
            self.D_x     = 0 * np.ones(self.n_segments + 1)
            self.alpha_y = 0 * np.ones(self.n_segments + 1)
            self.beta_y  = self.circumference/(2*np.pi*self.Q_y) * np.ones(self.n_segments + 1)
            self.D_y     = 0 * np.ones(self.n_segments + 1)

            self.alpha       = 0.06
            self.h1          = 1
            self.h2          = 2
            self.V1          = 8e3
            self.V2          = 0
            self.dphi1       = 0
            self.dphi2       = np.pi
            self.p_increment = 0 * e/c * self.circumference/(self.beta*c)

            self.longitudinal_focusing = 'non-linear'

        elif self.machine_configuration == '1GeV':
            self.charge = e
            self.mass = m_p

            self.gamma = 1e9*e/(self.mass*c**2) + 1

            self.Q_x     = 4.23
            self.Q_y     = 4.37

            self.Qp_x    = [-1*self.Q_x]
            self.Qp_y    = [-2*self.Q_y]

            self.app_x   = 0.0000e-9
            self.app_y   = 0.0000e-9
            self.app_xy  = 0

            self.alpha_x = 0 * np.ones(self.n_segments + 1)
            self.beta_x  = self.circumference/(2*np.pi*self.Q_x) * np.ones(self.n_segments + 1)
            self.D_x     = 0 * np.ones(self.n_segments + 1)
            self.alpha_y = 0 * np.ones(self.n_segments + 1)
            self.beta_y  = self.circumference/(2*np.pi*self.Q_y) * np.ones(self.n_segments + 1)
            self.D_y     = 0 * np.ones(self.n_segments + 1)

            self.alpha       = 0.06
            self.h1          = 1
            self.h2          = 2
            self.V1          = 8e3
            self.V2          = 0
            self.dphi1       = 0
            self.dphi2       = np.pi
            self.p_increment = 0 * e/c * self.circumference/(self.beta*c)

            self.longitudinal_focusing = 'non-linear'

        elif self.machine_configuration == '1.4GeV':
            self.charge = e
            self.mass = m_p

            self.gamma = 1.4e9*e/(self.mass*c**2) + 1

            self.Q_x     = 4.23
            self.Q_y     = 4.37

            self.Qp_x    = [-1*self.Q_x]
            self.Qp_y    = [-2*self.Q_y]

            self.app_x   = 0.0000e-9
            self.app_y   = 0.0000e-9
            self.app_xy  = 0

            self.alpha_x = 0 * np.ones(self.n_segments + 1)
            self.beta_x  = self.circumference/(2*np.pi*self.Q_x) * np.ones(self.n_segments + 1)
            self.D_x     = 0 * np.ones(self.n_segments + 1)
            self.alpha_y = 0 * np.ones(self.n_segments + 1)
            self.beta_y  = self.circumference/(2*np.pi*self.Q_y) * np.ones(self.n_segments + 1)
            self.D_y     = 0 * np.ones(self.n_segments + 1)

            self.alpha       = 0.06
            self.h1          = 1
            self.h2          = 2
            self.V1          = 8e3
            self.V2          = 0
            self.dphi1       = 0
            self.dphi2       = np.pi
            self.p_increment = 0 * e/c * self.circumference/(self.beta*c)

            self.longitudinal_focusing = 'non-linear'

        else:
            raise ValueError('ERROR: unknown machine configuration ' +
                             self.machine_configuration)

        super(PSB, self).__init__(*args, **kwargs)


class PS(Synchrotron):

    def __init__(self, *args, **kwargs):
        self.n_segments = kwargs['n_segments']
        self.gamma = kwargs['gamma'] #this should set also self.beta
        machine_configuration = kwargs.get('machine_configuration', 'LHCbeam_h7')
        self.circumference = 100*2*np.pi
        self.s = np.arange(0, self.n_segments + 1) * self.circumference / self.n_segments
        if machine_configuration == 'LHCbeam_h7':
            self.charge = e
            self.mass = m_p

            self.alpha_x        = 0 * np.ones(self.n_segments)
            self.beta_x         = 16 * np.ones(self.n_segments)
            self.D_x            = 0 * np.ones(self.n_segments)
            self.alpha_y        = 0 * np.ones(self.n_segments)
            self.beta_y         = 16 * np.ones(self.n_segments)
            self.D_y            = 0 * np.ones(self.n_segments)
            self.Q_x            = 6.27
            self.Q_y            = 6.23
            self.Qp_x           = 0
            self.Qp_y           = 0
            self.app_x          = 0.0000e-9
            self.app_y          = 0.0000e-9
            self.app_xy         = 0
            self.alpha     = 0.027
            self.h1, self.h2       = 7, 0
            self.V1, self.V2       = 24e3, 0
            self.dphi1, self.dphi2 = 0, 0
            self.p_increment       = 0 * e/c * self.circumference/(self.beta*c)
        elif machine_configuration =='TOFbeam_transition':
            self.charge = e
            self.mass = m_p

            self.alpha_x        = 0 * np.ones(self.n_segments)
            self.beta_x         = 16 * np.ones(self.n_segments)
            self.D_x            = 0 * np.ones(self.n_segments)
            self.alpha_y        = 0 * np.ones(self.n_segments)
            self.beta_y         = 16 * np.ones(self.n_segments)
            self.D_y            = 0 * np.ones(self.n_segments)

            self.Q_x            = 6.27
            self.Q_y            = 6.23

            self.Qp_x           = 0
            self.Qp_y           = 0

            self.app_x          = 0.0000e-9
            self.app_y          = 0.0000e-9
            self.app_xy         = 0

            self.alpha     = 0.027

            self.h1, self.h2       = 8, 8
            self.V1, self.V2       = 0.2e6, 0
            self.dphi1, self.dphi2 = np.pi, np.pi
            self.p_increment       = 46e9 * e/c * self.circumference/(self.beta*c)

        else:
            raise ValueError('ERROR: unknown machine configuration ' +
                             self.machine_configuration)
        super(PS, self).__init__(*args, **kwargs)


class SPS(Synchrotron):

    def __init__(self, *args, **kwargs):

        if 'n_segments' not in kwargs.keys():
            raise ValueError('Number of segments must be specified')

        if 'machine_configuration' not in kwargs.keys():
            raise ValueError('machine_configuration must be specified')

        if 'octupole_settings_dict' not in kwargs:
            kwargs['octupole_settings_dict'] = dict(
                KLOF=0.,
                KLOD=0.,
                dp_offset=0.
            )

        self.n_segments = kwargs['n_segments']
        self.machine_configuration = kwargs['machine_configuration']

        self.circumference = 1100*2*np.pi

        self.s = np.arange(0, self.n_segments + 1) * self.circumference / self.n_segments

        if self.machine_configuration == 'Q20-injection':
            self.charge = e
            self.mass = m_p

            self.gamma = 27.7

            self.alpha_x        = 0 * np.ones(self.n_segments + 1)
            self.beta_x         = 54.6 * np.ones(self.n_segments + 1)
            self.D_x            = 0 * np.ones(self.n_segments + 1)
            self.alpha_y        = 0 * np.ones(self.n_segments + 1)
            self.beta_y         = 54.6 * np.ones(self.n_segments + 1)
            self.D_y            = 0 * np.ones(self.n_segments + 1)

            self.Q_x            = 20.13
            self.Q_y            = 20.18

            self.Qp_x           = [ 0. ]
            self.Qp_y           = [ 0. ]

            self.app_x          = 0.0000e-9
            self.app_y          = 0.0000e-9
            self.app_xy         = 0

            self.alpha             = 0.00308
            self.h1, self.h2       = 4620, 4620*4
            self.V1, self.V2       = 5.75e6, 0
            self.dphi1, self.dphi2 = 0, np.pi
            self.p_increment       = 0 * e/c * self.circumference/(self.beta*c)

            self.longitudinal_focusing = 'non-linear'

            self.add_effect_of_octupoles(kwargs, optics='Q20')

        elif self.machine_configuration =='Q26-injection':
            self.charge = e
            self.mass = m_p

            self.gamma = 27.7

            self.alpha_x        = 0 * np.ones(self.n_segments + 1)
            self.beta_x         = 42. * np.ones(self.n_segments + 1)
            self.D_x            = 0 * np.ones(self.n_segments + 1)
            self.alpha_y        = 0 * np.ones(self.n_segments + 1)
            self.beta_y         = 42. * np.ones(self.n_segments + 1)
            self.D_y            = 0 * np.ones(self.n_segments + 1)

            self.Q_x            = 26.13
            self.Q_y            = 26.18

            self.Qp_x           = [ 0. ]
            self.Qp_y           = [ 0. ]

            self.app_x          = 0.0000e-9
            self.app_y          = 0.0000e-9
            self.app_xy         = 0

            self.alpha             = 0.00192
            self.h1, self.h2       = 4620, 4620*4
            self.V1, self.V2       = 2.e6, 0
            self.dphi1, self.dphi2 = 0, np.pi
            self.p_increment       = 0 * e/c * self.circumference/(self.beta*c)

            self.longitudinal_focusing = 'non-linear'

            self.add_effect_of_octupoles(kwargs, optics='Q26')

        elif self.machine_configuration == 'Q20-flattop':
            self.charge = e
            self.mass = m_p

            self.gamma = np.sqrt((450e9*e/(m_p*c**2))**2+1)

            self.alpha_x        = 0 * np.ones(self.n_segments + 1)
            self.beta_x         = 54.6 * np.ones(self.n_segments + 1)
            self.D_x            = 0 * np.ones(self.n_segments + 1)
            self.alpha_y        = 0 * np.ones(self.n_segments + 1)
            self.beta_y         = 54.6 * np.ones(self.n_segments + 1)
            self.D_y            = 0 * np.ones(self.n_segments + 1)

            self.Q_x            = 20.13
            self.Q_y            = 20.18

            self.Qp_x           = [ 0. ]
            self.Qp_y           = [ 0. ]

            self.app_x          = 0.0000e-9
            self.app_y          = 0.0000e-9
            self.app_xy         = 0

            self.alpha             = 0.00308
            self.h1, self.h2       = 4620, 4620*4
            self.V1, self.V2       = 10e6, 1e6
            self.dphi1, self.dphi2 = 0, 0
            self.p_increment       = 0 * e/c * self.circumference/(self.beta*c)

            self.longitudinal_focusing = 'non-linear'

            self.add_effect_of_octupoles(kwargs, optics='Q20')

        else:
            raise ValueError('ERROR: unknown machine configuration ' +
                             self.machine_configuration)

        for k in ['app_x', 'app_y', 'app_xy']:
            if k in kwargs.keys():
                kwargs[k] += getattr(self, k)

        super(SPS, self).__init__(*args, **kwargs)

    def add_effect_of_octupoles(self, kwargs, optics):
        if 'octupole_settings_dict' in kwargs.keys():
            octupoles = SPSOctupoles.SPSOctupoles(optics)
            KLOF = kwargs['octupole_settings_dict']['KLOF']
            KLOD = kwargs['octupole_settings_dict']['KLOD']
            dp_offset = kwargs['octupole_settings_dict']['dp_offset']
            octupoles.apply_to_machine(self, KLOF, KLOD, dp_offset)

class LHC(Synchrotron):

    def __init__(self, *args, **kwargs):

        if 'n_segments' not in kwargs.keys():
            raise ValueError('Number of segments must be specified')

        if 'machine_configuration' not in kwargs.keys():
            raise ValueError('machine_configuration must be specified')

        self.n_segments = kwargs['n_segments']
        self.machine_configuration = kwargs['machine_configuration']

        self.circumference  = 26658.883
        self.s = (np.arange(0, self.n_segments + 1)
                  * self.circumference / self.n_segments)

        if self.machine_configuration == '450GeV':
            self.charge = e
            self.mass = m_p

            self.gamma = np.sqrt( (450e9*e/(self.mass*c**2))**2 + 1 )

            self.Q_x     = 64.28
            self.Q_y     = 59.31

            self.alpha_x = 0 * np.ones(self.n_segments + 1)
            self.beta_x  = self.R/self.Q_x * np.ones(self.n_segments + 1)
            self.D_x     = 0 * np.ones(self.n_segments + 1)
            self.alpha_y = 0 * np.ones(self.n_segments + 1)
            self.beta_y  = self.R/self.Q_y * np.ones(self.n_segments + 1)
            self.D_y     = 0 * np.ones(self.n_segments + 1)

            self.Qp_x    = 0
            self.Qp_y    = 0

            self.app_x   = 0.0000e-9
            self.app_y   = 0.0000e-9
            self.app_xy  = 0

            self.alpha       = 3.225e-4
            self.h1          = 35640
            self.h2          = 71280
            self.V1          = 8e6
            self.V2          = 0
            self.dphi1       = 0
            self.dphi2       = np.pi
            self.p_increment = 0 * e/c * self.circumference/(self.beta*c)

            self.longitudinal_focusing = 'non-linear'

        elif self.machine_configuration == '3.5TeV':
            # as in 2010...
            self.charge = e
            self.mass = m_p

            self.gamma = np.sqrt( (3500e9*e/(self.mass*c**2))**2 + 1 )

            self.Q_x     = 64.28
            self.Q_y     = 59.31

            self.alpha_x = 0 * np.ones(self.n_segments + 1)
            self.beta_x  = self.R/self.Q_x * np.ones(self.n_segments + 1)
            self.D_x     = 0 * np.ones(self.n_segments + 1)
            self.alpha_y = 0 * np.ones(self.n_segments + 1)
            self.beta_y  = self.R/self.Q_y * np.ones(self.n_segments + 1)
            self.D_y     = 0 * np.ones(self.n_segments + 1)

            self.Qp_x    = 0
            self.Qp_y    = 0

            self.app_x   = 0.0000e-9
            self.app_y   = 0.0000e-9
            self.app_xy  = 0

            self.alpha       = 3.225e-4
            self.h1          = 35640
            self.h2          = 71280
            self.V1          = 8e6
            self.V2          = 0
            self.dphi1       = 0
            self.dphi2       = np.pi
            self.p_increment = 0 * e/c * self.circumference/(self.beta*c)

            self.longitudinal_focusing = 'non-linear'

        elif self.machine_configuration == '6.5TeV':
            self.charge = e
            self.mass = m_p

            self.gamma = np.sqrt( (6500e9*e/(self.mass*c**2))**2 + 1 )

            self.Q_x     = 64.31
            self.Q_y     = 59.32

            self.alpha_x = 0 * np.ones(self.n_segments + 1)
            self.beta_x  = self.R/self.Q_x * np.ones(self.n_segments + 1)
            self.D_x     = 0 * np.ones(self.n_segments + 1)
            self.alpha_y = 0 * np.ones(self.n_segments + 1)
            self.beta_y  = self.R/self.Q_y * np.ones(self.n_segments + 1)
            self.D_y     = 0 * np.ones(self.n_segments + 1)

            self.Qp_x    = 0
            self.Qp_y    = 0

            self.app_x   = 0.0000e-9
            self.app_y   = 0.0000e-9
            self.app_xy  = 0

            self.alpha       = 3.225e-4
            self.h1          = 35640
            self.h2          = 71280
            self.V1          = 10e6
            self.V2          = 0
            self.dphi1       = 0
            self.dphi2       = np.pi
            self.p_increment = 0 * e/c * self.circumference/(self.beta*c)

            self.longitudinal_focusing = 'non-linear'

        else:
            raise ValueError('ERROR: unknown machine configuration ' +
                             self.machine_configuration)

        i_focusing = kwargs.pop('i_focusing', False)
        i_defocusing = kwargs.pop('i_defocusing', False)
        if i_focusing or i_defocusing is True:
            print ('\n--> Powering LHC octupoles to {:g} A.\n'.format(i_focusing))
            self.app_x, self.app_y, self.app_xy = self.get_anharmonicities_from_octupole_currents_LHC(
                i_focusing, i_defocusing)

        super(LHC, self).__init__(*args, **kwargs)

    def get_anharmonicities_from_octupole_currents_LHC(cls, i_focusing, i_defocusing):
        """Calculate the constants of proportionality app_x, app_y and
        app_xy (== app_yx) for the amplitude detuning introduced by the
        LHC octupole magnets (aka. LHC Landau octupoles) from the
        electric currents i_focusing [A] and i_defocusing [A] flowing
        through the magnets. The maximum current is given by
        i_max = +/- 550 [A]. The values app_x, app_y, app_xy obtained
        from the formulae are proportional to the strength of detuning
        for one complete turn around the accelerator, i.e. one-turn
        values.

        The calculation is based on formulae (3.6) taken from 'The LHC
        transverse coupled-bunch instability' by N. Mounet, EPFL PhD
        Thesis, 2012. Values (hard-coded numbers below) are valid for
        LHC Landau octupoles before LS1. Beta functions in x and y are
        correctly taken into account. Note that here, the values of
        app_x, app_y and app_xy are not normalized to the reference
        momentum p0. This is done only during the calculation of the
        detuning in the corresponding detune method of the
        AmplitudeDetuningSegment.

        More detailed explanations and references on how the formulae
        were obtained are given in the PhD thesis (pg. 85ff) cited
        above.
        """
        i_max = 550.  # [A]
        E_max = 7000. # [GeV]

        app_x  = E_max * (267065. * i_focusing / i_max -
            7856. * i_defocusing / i_max)
        app_y  = E_max * (9789. * i_focusing / i_max -
            277203. * i_defocusing / i_max)
        app_xy = E_max * (-102261. * i_focusing / i_max +
            93331. * i_defocusing / i_max)

        # Convert to SI units.
        convert_to_SI = e / (1.e-9 * c)
        app_x *= convert_to_SI
        app_y *= convert_to_SI
        app_xy *= convert_to_SI

        return app_x, app_y, app_xy


class HLLHC(Synchrotron):

    def __init__(self, *args, **kwargs):

        if 'n_segments' not in kwargs.keys():
            raise ValueError('Number of segments must be specified')

        if 'machine_configuration' not in kwargs.keys():
            raise ValueError('machine_configuration must be specified')

        self.n_segments = kwargs['n_segments']
        self.machine_configuration = kwargs['machine_configuration']

        self.circumference  = 26658.883
        self.s = (np.arange(0, self.n_segments + 1)
                  * self.circumference / self.n_segments)

        if self.machine_configuration == '7TeV':
            self.charge = e
            self.mass = m_p

            self.gamma = np.sqrt( (7000e9*e/(self.mass*c**2))**2 + 1 )

            self.Q_x     = 62.31
            self.Q_y     = 60.32

            self.alpha_x = 0 * np.ones(self.n_segments + 1)
            self.beta_x  = self.R/self.Q_x * np.ones(self.n_segments + 1)
            self.D_x     = 0 * np.ones(self.n_segments + 1)
            self.alpha_y = 0 * np.ones(self.n_segments + 1)
            self.beta_y  = self.R/self.Q_y * np.ones(self.n_segments + 1)
            self.D_y     = 0 * np.ones(self.n_segments + 1)

            self.Qp_x    = 0
            self.Qp_y    = 0

            self.app_x   = 0.0000e-9
            self.app_y   = 0.0000e-9
            self.app_xy  = 0

            self.alpha       = 53.83**-2
            self.h1          = 35640
            self.h2          = 71280
            self.V1          = 16e6
            self.V2          = 0
            self.dphi1       = 0
            self.dphi2       = np.pi
            self.p_increment = 0 * e/c * self.circumference/(self.beta*c)

            self.longitudinal_focusing = 'non-linear'

        else:
            raise ValueError('ERROR: unknown machine configuration ' +
                             self.machine_configuration)

        i_focusing = kwargs.pop('i_focusing', False)
        i_defocusing = kwargs.pop('i_defocusing', False)
        if i_focusing or i_defocusing is True:
            print ('\n--> Powering LHC octupoles to {:g} A.\n'.format(i_focusing))
            self.app_x, self.app_y, self.app_xy = self.get_anharmonicities_from_octupole_currents_LHC(
                i_focusing, i_defocusing)

        super(HLLHC, self).__init__(*args, **kwargs)

    def get_anharmonicities_from_octupole_currents_LHC(cls, i_focusing, i_defocusing):
        """Calculate the constants of proportionality app_x, app_y and
        app_xy (== app_yx) for the amplitude detuning introduced by the
        LHC octupole magnets (aka. LHC Landau octupoles) from the
        electric currents i_focusing [A] and i_defocusing [A] flowing
        through the magnets. The maximum current is given by
        i_max = +/- 550 [A]. The values app_x, app_y, app_xy obtained
        from the formulae are proportional to the strength of detuning
        for one complete turn around the accelerator, i.e. one-turn
        values.

        The calculation is based on formulae (3.6) taken from 'The LHC
        transverse coupled-bunch instability' by N. Mounet, EPFL PhD
        Thesis, 2012. Values (hard-coded numbers below) are valid for
        LHC Landau octupoles before LS1. Beta functions in x and y are
        correctly taken into account. Note that here, the values of
        app_x, app_y and app_xy are not normalized to the reference
        momentum p0. This is done only during the calculation of the
        detuning in the corresponding detune method of the
        AmplitudeDetuningSegment.

        More detailed explanations and references on how the formulae
        were obtained are given in the PhD thesis (pg. 85ff) cited
        above.
        """
        i_max = 550.  # [A]
        E_max = 7000. # [GeV]

        app_x  = E_max * (267065. * i_focusing / i_max -
            7856. * i_defocusing / i_max)
        app_y  = E_max * (9789. * i_focusing / i_max -
            277203. * i_defocusing / i_max)
        app_xy = E_max * (-102261. * i_focusing / i_max +
            93331. * i_defocusing / i_max)

        # Convert to SI units.
        convert_to_SI = e / (1.e-9 * c)
        app_x *= convert_to_SI
        app_y *= convert_to_SI
        app_xy *= convert_to_SI

        return app_x, app_y, app_xy
