import numpy as np
from scipy.constants import c, e, m_p

from PyHEADTAIL.machines.synchrotron import Synchrotron


class HLLHC(Synchrotron):

    def __init__(self, machine_configuration=None,
                 optics_mode='smooth', longitudinal_mode='non-linear', h_RF = 35640,
                 circumference = 26658.883, **kwargs):

        charge = e
        mass = m_p
        alpha = 53.86**-2
#        h_RF = 35640
        self.h_RF = h_RF
        RF_at = 'middle'

        if machine_configuration == '7_TeV_collision_tunes':
            p0 = 7000e9 * e / c
            p_increment = 0.
            accQ_x = 62.31
            accQ_y = 60.32
            V_RF = 16e6
            dphi_RF = 0
        else:
            raise ValueError('machine_configuration not recognized!')

        if optics_mode == 'smooth':
            if 's' in kwargs.keys():
                raise ValueError('s vector cannot be provided if ' +
                                 'optics_mode == "smooth"')

            name = None
            n_segments = kwargs['n_segments']
#            circumference = 26658.883

            s = None
            alpha_x = None
            alpha_y = None
            beta_x = circumference / (2.*np.pi*accQ_x)
            beta_y = circumference / (2.*np.pi*accQ_y)
            self.beta_x = beta_x
            self.beta_y = beta_y
            D_x = 0
            D_y = 0

        elif optics_mode == 'non-smooth':
            if 'n_segments' in kwargs.keys():
                raise ValueError('n_segments cannot be provided if\
                optics_mode = "non-smooth"')

            circumference = None
            n_segments = None
            name = kwargs['name']

            s = kwargs['s']
            alpha_x = kwargs['alpha_x']
            alpha_y = kwargs['alpha_y']
            beta_x = kwargs['beta_x']
            beta_y = kwargs['beta_y']
            accQ_x = kwargs['accQ_x']
            accQ_y = kwargs['accQ_y']
            try:
                D_x = kwargs['D_x']
            except KeyError:
                D_x = 0*np.array(kwargs['s'])
            try:
                D_y = kwargs['D_y']
            except KeyError:
                D_y = 0*np.array(kwargs['s'])

        else:
            raise ValueError('optics_mode not recognized!')

        # detunings
        Qp_x = 0
        Qp_y = 0

        app_x = 0
        app_y = 0
        app_xy = 0

        i_octupole_focusing = None
        i_octupole_defocusing = None
        octupole_knob = None

        for attr in kwargs.keys():
            if kwargs[attr] is not None:
                if (type(kwargs[attr]) is list or
                        type(kwargs[attr]) is np.ndarray):
                    str2print = '[%s ...]' % repr(kwargs[attr][0])
                else:
                    str2print = repr(kwargs[attr])
                self.prints('Synchrotron init. From kwargs: %s = %s'
                            % (attr, str2print))
                temp = kwargs[attr]
                exec('%s = temp' % attr)

        if i_octupole_focusing is not None or i_octupole_defocusing is not None:
            if octupole_knob is not None:
                raise ValueError('octupole_knobs and octupole currents cannot' +
                                 ' be used at the same time!')
            app_x, app_y, app_xy = (
                self._anharmonicities_from_octupole_current_settings(
                    i_octupole_focusing, i_octupole_defocusing))
            self.i_octupole_focusing = i_octupole_focusing
            self.i_octupole_defocusing = i_octupole_defocusing

        if octupole_knob is not None:
            if (i_octupole_focusing is not None or
                    i_octupole_defocusing is not None):
                raise ValueError('octupole_knobs and octupole currents cannot' +
                                 ' be used at the same time!')
            i_octupole_focusing, i_octupole_defocusing = (
                self._octupole_currents_from_octupole_knobs(octupole_knob, p0))
            app_x, app_y, app_xy = (
                self._anharmonicities_from_octupole_current_settings(
                    i_octupole_focusing, i_octupole_defocusing))
            self.i_octupole_focusing = i_octupole_focusing
            self.i_octupole_defocusing = i_octupole_defocusing


        super(HLLHC, self).__init__(
            optics_mode=optics_mode, circumference=circumference,
            n_segments=n_segments, s=s, name=name,
            alpha_x=alpha_x, beta_x=beta_x, D_x=D_x,
            alpha_y=alpha_y, beta_y=beta_y, D_y=D_y,
            accQ_x=accQ_x, accQ_y=accQ_y, Qp_x=Qp_x, Qp_y=Qp_y,
            app_x=app_x, app_y=app_y, app_xy=app_xy,
            alpha_mom_compaction=alpha, longitudinal_mode=longitudinal_mode,
            h_RF=np.atleast_1d(h_RF), V_RF=np.atleast_1d(V_RF),
            dphi_RF=np.atleast_1d(dphi_RF), p0=p0, p_increment=p_increment,
            charge=charge, mass=mass, wrap_z=wrap_z)

    def _anharmonicities_from_octupole_current_settings(
            self, i_octupole_focusing, i_octupole_defocusing):
        """Calculate the constants of proportionality app_x, app_y
        and app_xy (== app_yx) for the amplitude detuning introduced
        by the LHC octupole magnets (aka. LHC Landau octupoles) from
        the electric currents i_octupole_focusing [A] and
        i_octupole_defocusing [A] flowing through the magnets. The
        maximum current is given by i_max = +/- 550 [A]. The values
        app_x, app_y, app_xy obtained from the formulae are
        proportional to the strength of detuning for one complete
        turn around the accelerator, i.e. one-turn values.
        The calculation is based on formulae (3.6) taken from 'The
        LHC transverse coupled-bunch instability' by N. Mounet, EPFL
        PhD Thesis, 2012. Values (hard-coded numbers below) are
        valid for LHC Landau octupoles before LS1. Beta functions
        in x and y are correctly taken into account. Note that here,
        the values of app_x, app_y and app_xy are not normalized to
        the reference momentum p0. This is done only during the
        calculation of the detuning in the corresponding detune
        method of the AmplitudeDetuningSegment.
        More detailed explanations and references on how the
        formulae were obtained are given in the PhD thesis (pg.
        85ff) cited above.
        """
        i_max = 550.  # [A]
        E_max = 7000. # [GeV]

        app_x  = E_max * (267065. * i_octupole_focusing / i_max -
                          7856. * i_octupole_defocusing / i_max)
        app_y  = E_max * (9789. * i_octupole_focusing / i_max -
                          277203. * i_octupole_defocusing / i_max)
        app_xy = E_max * (-102261. * i_octupole_focusing / i_max +
                          93331. * i_octupole_defocusing / i_max)

        # Convert to SI units.
        convert_to_SI = e / (1.e-9 * c)
        app_x *= convert_to_SI
        app_y *= convert_to_SI
        app_xy *= convert_to_SI

        return app_x, app_y, app_xy


    def _octupole_currents_from_octupole_knobs(self, octupole_knob, p0):

        i_octupole_focusing = (19.557 * octupole_knob / (-1.5) * p0 /
                               2.4049285931335872e-16)
        i_octupole_defocusing = - i_octupole_focusing

        return i_octupole_focusing, i_octupole_defocusing