'''
@author Kevin Li, Michael Schenk, Adrian Oeftiger
@date June, 23rd 2014
@brief Factory of detuners
@copyright CERN
'''
from __future__ import division
from abc import ABCMeta, abstractmethod
from scipy.constants import e, c
import numpy as np


class Detuner(object):
    """
    ABC for detuners.
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def detune(self, beam):
        """
        Calculates the detune caused by the corresponding detuner.
        """
        pass


"""
Some commonly used detuners. To be extended.
"""
class ChromaticitySegment(Detuner):

    def __init__(self, dQp_x, dQp_y):
        self.dQp_x = dQp_x
        self.dQp_y = dQp_y


    def detune(self, beam):
        # W/o factor 2 np.pi. See TransverseSegmentMap.track().
        dphi_x = self.dQp_x * beam.dp
        dphi_y = self.dQp_y * beam.dp

        return dphi_x, dphi_y


class AmplitudeDetuningSegment(Detuner):

    def __init__(self, beta_x, beta_y, dapp_x, dapp_y, dapp_xy):
        self.beta_x = beta_x
        self.beta_y = beta_y

        # For octupole magnets: dapp_xy == dapp_yx.
        self.dapp_x  = dapp_x
        self.dapp_y  = dapp_y
        self.dapp_xy = dapp_xy


    def detune(self, beam):
        Jx = (beam.x ** 2 + (self.beta_x * beam.xp) ** 2) / (2. * self.beta_x)
        Jy = (beam.y ** 2 + (self.beta_y * beam.yp) ** 2) / (2. * self.beta_y)

        # W/o factor 2 np.pi. See TransverseSegmentMap.track().
        dphi_x = self.dapp_x/beam.p0 * Jx + self.dapp_xy/beam.p0 * Jy
        dphi_y = self.dapp_y/beam.p0 * Jy + self.dapp_xy/beam.p0 * Jx

        return dphi_x, dphi_y


class LaslettSpaceChargeDetuner(Detuner):
    '''
    Calculate the Laslett direct space charge detuning.

    .. math::
        \Delta Q_{x,y}(z) = -r_p \lambda(z)/(2 \pi \beta^2 \gamma^3)
            \cdot \oint ds \frac{\beta_{x,y}(s)}{
                (\sigma_{x,y}(s) \cdot (\sigma_x(s) + \sigma_y(s)))
            }
    '''

    def __init__(self, slices, ointegral_x, ointegral_y):
        '''
        ointegral_x and ointegral_y are the integrals around the
        accelerator ring over the betatron function divided by the beam
        rms sizes sigma_x and sigma_y:

        .. math::
            \oint ds \frac{ \beta_{x,y}(s) }{
                (\sigma_{x,y}(s) \cdot (\sigma_x(s) + \sigma_y(s)))
            }
        '''
        self.slices = slices
        self.ointegral_x = ointegral_x
        self.ointegral_y = ointegral_y

    def detune(self, beam):
        lambda_ = self.slices.n_macroparticles * beam.n_particles_per_mp
        slice_tuneshift_x = self._prefactor(beam) * lambda_ * self.ointegral_x
        slice_tuneshift_y = self._prefactor(beam) * lambda_ * self.ointegral_y

        dQ_x = np.empty(beam.n_macroparticles)
        dQ_y = np.empty(beam.n_macroparticles)

        p_id = self.slices.particles_within_cuts
        s_id = self.slices.slice_index_of_particle.take(p_id)

        dQ_x[p_id] = slice_tuneshift_x[s_id]
        dQ_y[p_id] = slice_tuneshift_y[s_id]

        return dQ_x, dQ_y

    def _prefactor(self, beam):
        return (-beam.classical_particle_radius /
                (2 * np.pi * beam.beta**2 * beam.gamma**3))



class DetunerCollection(object):
    """
    Collection classes for each class of detuner. These are the classes
    instantiated explicitly by the user. They use 1-turn integrated values
    as input and instantiate detuners for each segment in 's' with a
    detuning proportional to the segment length.
    """
    __metaclass__ = ABCMeta
    segment_detuners = []

    @abstractmethod
    def generate_segment_detuner(self, relative_segment_length, beta_x, beta_y):
        pass

    def __len__(self):
        return len(self.segment_detuners)


    def __getitem__(self, key):
        return self.segment_detuners[key]


class AmplitudeDetuning(DetunerCollection):

    def __init__(self, app_x, app_y, app_xy):
        self.app_x  = app_x
        self.app_y  = app_y
        self.app_xy = app_xy


    @classmethod
    def from_octupole_currents_LHC(cls, i_focusing, i_defocusing):
        """
        Calculate app_x, app_y, app_xy == app_yx on the basis of formulae (3.6) in
        'THE LHC TRANSVERSE COUPLED-BUNCH INSTABILITY' (EPFL PhD Thesis), N. Mounet, 2012
        from LHC octupole currents i_focusing, i_defocusing [A].

        Measurement values (hard-coded numbers) were obtained before LS1.
        """
        i_max = 550.  # [A]
        E_max = 7000. # [GeV]

        app_x  = E_max * (267065. * i_focusing / i_max - 7856. * i_defocusing / i_max)
        app_y  = E_max * (9789. * i_focusing / i_max - 277203. * i_defocusing / i_max)
        app_xy = E_max * (-102261. * i_focusing / i_max + 93331. * i_defocusing / i_max)

        convert_to_SI_units = e/(1e-9*c)
        app_x  *= convert_to_SI_units
        app_y  *= convert_to_SI_units
        app_xy *= convert_to_SI_units

        return cls(app_x, app_y, app_xy)


    def generate_segment_detuner(self, relative_segment_length, beta_x, beta_y):
        dapp_x  = self.app_x * relative_segment_length
        dapp_y  = self.app_y * relative_segment_length
        dapp_xy = self.app_xy * relative_segment_length
        # For octupole magnets, app_xy == app_yx.

        self.segment_detuners.append(AmplitudeDetuningSegment(
            beta_x, beta_y, dapp_x, dapp_y, dapp_xy))


class Chromaticity(DetunerCollection):

    def __init__(self, Qp_x, Qp_y):
        self.Qp_x = Qp_x
        self.Qp_y = Qp_y


    def generate_segment_detuner(self, relative_segment_length, *args):
        dQp_x = self.Qp_x * relative_segment_length
        dQp_y = self.Qp_y * relative_segment_length

        self.segment_detuners.append(ChromaticitySegment(dQp_x, dQp_y))


class LaslettSpaceCharge(DetunerCollection):

    def __init__(self, slices, ointegral_x, ointegral_y):
        self.slices = slices
        self.ointegral_x = ointegral_x
        self.ointegral_y = ointegral_y

    def generate_segment_detuner(self, relative_segment_length, *args):
        ointegral_x = self.ointegral_x * relative_segment_length
        ointegral_y = self.ointegral_y * relative_segment_length

        self.segment_detuners.append(LaslettSpaceChargeDetuner(
            self.slices, ointegral_x, ointegral_y))
