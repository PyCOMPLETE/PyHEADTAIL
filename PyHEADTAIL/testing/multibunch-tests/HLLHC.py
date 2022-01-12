import numpy as np
from scipy.constants import c, e, m_p

from PyHEADTAIL.machines.synchrotron import Synchrotron


class EmptyObject(object):
    pass

class HLLHC(Synchrotron):

    def __init__(self, machine_configuration=None,
                 optics_mode='smooth', longitudinal_mode='non-linear', h_RF = 35640,
                 circumference = 26658.883, **kwargs):

        pp = EmptyObject()
        pp.machine_configuration = machine_configuration
        pp.optics_mode = optics_mode
        pp.longitudinal_mode = longitudinal_mode
        pp.h_RF = h_RF
        pp.circumference = circumference
    
        pp.charge = e
        pp.mass = m_p
        pp.alpha = 53.86**-2
        pp.RF_at = 'middle'
        pp.wrap_z = False
        
        if pp.machine_configuration == '7_TeV_collision_tunes':
            pp.p0 = 7000e9 * e / c
            pp.p_increment = 0.
            pp.accQ_x = 62.31
            pp.accQ_y = 60.32
            pp.V_RF = 16e6
            pp.dphi_RF = 0
        else:
            raise ValueError('machine_configuration not recognized!')

        if pp.optics_mode == 'smooth':
            if 's' in kwargs.keys():
                raise ValueError('s vector cannot be provided if ' +
                                 'optics_mode == "smooth"')

            pp.name = None
            pp.n_segments = kwargs['n_segments']
#            circumference = 26658.883

            pp.s = None
            pp.alpha_x = None
            pp.alpha_y = None
            pp.beta_x = pp.circumference / (2.*np.pi*pp.accQ_x)
            pp.beta_y = pp.circumference / (2.*np.pi*pp.accQ_y)
            pp.D_x = 0
            pp.D_y = 0

        elif pp.optics_mode == 'non-smooth':
            if 'n_segments' in kwargs.keys():
                raise ValueError('n_segments cannot be provided if\
                optics_mode = "non-smooth"')

            pp.circumference = None
            pp.n_segments = None
            pp.name = kwargs['name']

            pp.s = kwargs['s']
            pp.alpha_x = kwargs['alpha_x']
            pp.alpha_y = kwargs['alpha_y']
            pp.beta_x = kwargs['beta_x']
            pp.beta_y = kwargs['beta_y']
            pp.accQ_x = kwargs['accQ_x']
            pp.accQ_y = kwargs['accQ_y']
            try:
                pp.D_x = kwargs['D_x']
            except KeyError:
                pp.D_x = 0*np.array(kwargs['s'])
            try:
                pp.D_y = kwargs['D_y']
            except KeyError:
                pp.D_y = 0*np.array(kwargs['s'])

        else:
            raise ValueError('optics_mode not recognized!')

        # detunings
        pp.Qp_x = 0
        pp.Qp_y = 0

        pp.app_x = 0
        pp.app_y = 0
        pp.app_xy = 0

        pp.i_octupole_focusing = None
        pp.i_octupole_defocusing = None
        pp.octupole_knob = None

        for attr in list(kwargs.keys()):
            if kwargs[attr] is not None:
                if (type(kwargs[attr]) is list) or (
                        type(kwargs[attr]) is np.ndarray):
                    str2print = '[{:s} ...]'.format(repr(kwargs[attr][0]))
                else:
                    str2print = repr(kwargs[attr])
                self.prints('Synchrotron init. '
                            'From kwargs: {:s} = {:s}'.format(attr, str2print))
                if not hasattr(pp, attr):
                    raise NameError("I don't understand {:s}".format(attr))

                setattr(pp, attr, kwargs[attr])

        if pp.i_octupole_focusing is not None or pp.i_octupole_defocusing is not None:
            if pp.octupole_knob is not None:
                raise ValueError('octupole_knobs and octupole currents cannot' +
                                 ' be used at the same time!')
            pp.app_x, pp.app_y, pp.app_xy = (
                self._anharmonicities_from_octupole_current_settings(
                    pp.i_octupole_focusing, pp.i_octupole_defocusing))
            self.i_octupole_focusing = pp.i_octupole_focusing
            self.i_octupole_defocusing = pp.i_octupole_defocusing

        if pp.octupole_knob is not None:
            if (pp.i_octupole_focusing is not None or
                    pp.i_octupole_defocusing is not None):
                raise ValueError('octupole_knobs and octupole currents cannot' +
                                 ' be used at the same time!')
            pp.i_octupole_focusing, pp.i_octupole_defocusing = (
                self._octupole_currents_from_octupole_knobs(pp.octupole_knob, pp.p0))
            pp.app_x, pp.app_y, pp.app_xy = (
                self._anharmonicities_from_octupole_current_settings(
                    pp.i_octupole_focusing, pp.i_octupole_defocusing))
            self.i_octupole_focusing = pp.i_octupole_focusing
            self.i_octupole_defocusing = pp.i_octupole_defocusing


        super(HLLHC, self).__init__(
            optics_mode=pp.optics_mode, circumference=pp.circumference,
            n_segments=pp.n_segments, s=pp.s, name=pp.name,
            alpha_x=pp.alpha_x, beta_x=pp.beta_x, D_x=pp.D_x,
            alpha_y=pp.alpha_y, beta_y=pp.beta_y, D_y=pp.D_y,
            accQ_x=pp.accQ_x, accQ_y=pp.accQ_y, Qp_x=pp.Qp_x, Qp_y=pp.Qp_y,
            app_x=pp.app_x, app_y=pp.app_y, app_xy=pp.app_xy,
            alpha_mom_compaction=pp.alpha, longitudinal_mode=pp.longitudinal_mode,
            h_RF=np.atleast_1d(pp.h_RF), V_RF=np.atleast_1d(pp.V_RF),
            dphi_RF=np.atleast_1d(pp.dphi_RF), p0=pp.p0, p_increment=pp.p_increment,
            charge=pp.charge, mass=pp.mass, wrap_z=pp.wrap_z)

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
