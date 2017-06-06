"""
@author Kevin Li, Michael Schenk, Stefan Hegglin
@date 07. January 2014
@brief Description of the transport of transverse phase spaces.
@copyright CERN
"""
from __future__ import division

import numpy as np

from .. general import pmath as pm

from ..general.decorators import deprecated
from . import Element, Printing


class TransverseSegmentMap(Element):
    """ Class to transport/track the particles of the beam in the
    transverse plane through an accelerator ring segment defined by its
    boundaries [s0, s1]. To calculate the transverse linear transport
    matrix M that transports each particle's transverse phase space
    coordinates (x, xp, y, yp) from position s0 to position s1 in the
    accelerator, the TWISS parameters alpha and beta at positions s0
    and s1 must be provided. The betatron phase advance of each
    particle in the present segment is given by their betatron tune
    Q_{x,y} (phase advance) and possibly by an incoherent tune shift
    introduced e.g. by amplitude detuning or chromaticity effects
    (see trackers.detuners module).

    Dispersion is added in the horizontal and vertical planes. Care
    needs to be taken, that dispersive effects were taken into account
    upon beam creation. Then, before each linear tracking step, the
    dispersion is removed, linear tracking is performed via the linear
    periodic map and dispersion is added back so that any subsequent
    collective effect has dispersion taken into account.
    """
    def __init__(self,
                 alpha_x_s0, beta_x_s0, D_x_s0, alpha_x_s1, beta_x_s1, D_x_s1,
                 alpha_y_s0, beta_y_s0, D_y_s0, alpha_y_s1, beta_y_s1, D_y_s1,
                 dQ_x, dQ_y, *args, **kwargs):
        """ Return an instance of the TransverseSegmentMap class. The
        values of the TWISS parameters alpha_{x,y} and beta_{x,y} as
        well as of the dispersion coefficients D_{x,y} (not yet
        implemented) are given at the beginning s0 and at the end s1 of
        the corresponding segment. The dQ_{x,y} denote the betatron
        tune advance over the current segment (phase advance divided by
        2 \pi). The SegmentDetuner objects present in this segment are
        passed as a list via the keyword argument 'segment_detuners'.
        The matrices self.I and self.J are constant and are calculated
        only once at instantiation of the TransverseSegmentMap. """
        self.D_x_s0 = D_x_s0
        self.D_x_s1 = D_x_s1
        self.D_y_s0 = D_y_s0
        self.D_y_s1 = D_y_s1
        self.dQ_x = dQ_x
        self.dQ_y = dQ_y

        self._build_segment_map(alpha_x_s0, beta_x_s0, alpha_x_s1, beta_x_s1,
                                alpha_y_s0, beta_y_s0, alpha_y_s1, beta_y_s1)

        self.segment_detuners = kwargs.pop('segment_detuners', [])

        # bind the implementation of the tracking depending on whether
        # the all dispersion parameters are close to 0 or not
        if np.allclose([D_x_s0, D_x_s1, D_y_s0, D_y_s1],
                       np.zeros(4), atol=1e-3):
            self._track = self._track_without_dispersion
        else:
            self._track = self._track_with_dispersion

    def _build_segment_map(self, alpha_x_s0, beta_x_s0, alpha_x_s1, beta_x_s1,
                           alpha_y_s0, beta_y_s0, alpha_y_s1, beta_y_s1):
        """ Calculate matrices I and J which are decoupled from the
        phase advance dependency and only depend on the TWISS parameters
        at the boundaries of the accelerator segment. These matrices are
        constant and hence need to be calculated only once at
        instantiation of the TransverseSegmentMap. """
        self.I = np.zeros((4, 4))
        self.J = np.zeros((4, 4))

        # Sine component.
        self.I[0, 0] = np.sqrt(beta_x_s1 / beta_x_s0)
        self.I[0, 1] = 0.
        self.I[1, 0] = (np.sqrt(1. / (beta_x_s0 * beta_x_s1)) *
                        (alpha_x_s0 - alpha_x_s1))
        self.I[1, 1] = np.sqrt(beta_x_s0 / beta_x_s1)
        self.I[2, 2] = np.sqrt(beta_y_s1 / beta_y_s0)
        self.I[2, 3] = 0.
        self.I[3, 2] = (np.sqrt(1. / (beta_y_s0 * beta_y_s1)) *
                        (alpha_y_s0 - alpha_y_s1))
        self.I[3, 3] = np.sqrt(beta_y_s0 / beta_y_s1)

        # Cosine component.
        self.J[0, 0] = np.sqrt(beta_x_s1 / beta_x_s0) * alpha_x_s0
        self.J[0, 1] = np.sqrt(beta_x_s0 * beta_x_s1)
        self.J[1, 0] = -(np.sqrt(1. / (beta_x_s0 * beta_x_s1)) *
                         (1. + alpha_x_s0 * alpha_x_s1))
        self.J[1, 1] = -np.sqrt(beta_x_s0 / beta_x_s1) * alpha_x_s1
        self.J[2, 2] = np.sqrt(beta_y_s1 / beta_y_s0) * alpha_y_s0
        self.J[2, 3] = np.sqrt(beta_y_s0 * beta_y_s1)
        self.J[3, 2] = -(np.sqrt(1. / (beta_y_s0 * beta_y_s1)) *
                         (1. + alpha_y_s0 * alpha_y_s1))
        self.J[3, 3] = -np.sqrt(beta_y_s0 / beta_y_s1) * alpha_y_s1

    def _track_with_dispersion(self, beam, M00, M01, M10, M11, M22, M23,
                               M32, M33):
        """This method gets bound to the self._track() method if
        there are dispersion effects, i.e. any of the 4 dispersion parameters
        is != 0
        It computes the transverse tracking given the matrix elements Mij.
        1) Subtract the dispersion using dp
        2) Change the positions and momenta using the matrix elements
        3) Add the dispersion effects using dp
        """

        # subtract the dispersion
        beam.x -= self.D_x_s0 * beam.dp
        beam.y -= self.D_y_s0 * beam.dp

        beam.x, beam.xp = M00*beam.x + M01*beam.xp, M10*beam.x + M11*beam.xp
        beam.y, beam.yp = M22*beam.y + M23*beam.yp, M32*beam.y + M33*beam.yp

        # add the new dispersion effect
        beam.x += self.D_x_s1 * beam.dp
        beam.y += self.D_y_s1 * beam.dp

    def _track_without_dispersion(self, beam, M00, M01, M10, M11, M22, M23,
                                  M32, M33):
        """This method gets bound to the self._track() method if there are
        no dispersive effects, i.e. all of the 4 dispersion parameters
        are close to (1e-3) 0.
        It computes the transverse tracking given the matrix elements Mij
        """

        beam.x, beam.xp = M00*beam.x + M01*beam.xp, M10*beam.x + M11*beam.xp
        beam.y, beam.yp = M22*beam.y + M23*beam.yp, M32*beam.y + M33*beam.yp

    def track(self, beam):
        """ The dphi_{x,y} denote the phase advance in the horizontal
        and vertical plane respectively for the given accelerator
        segment. They are composed of the betatron tunes dQ_{x,y} and a
        possible incoherent tune shift introduced by detuner elements
        / effects defined in the list self.segment_detuners (they are
        all instances of the SegmentDetuner child classes).
        The transport matrix is defined by the coefficients M_{ij}. """

        # Calculate phase advance for this segment (betatron motion in
        # this segment and incoherent tune shifts introduced by detuning
        # effects).
        dphi_x = self.dQ_x
        dphi_y = self.dQ_y

        dphi_is_array = False

        for element in self.segment_detuners:
            detune_x, detune_y = element.detune(beam)
            dphi_x += detune_x
            dphi_y += detune_y
            dphi_is_array = True

        dphi_x *= 2.*np.pi
        dphi_y *= 2.*np.pi

        # needs to be pm.cos, cos alone not possible:
        # the change in the pm namespace has to be visible here
        # --> use of named vars better style anyway
        # another problem is that dphi_x can be either a scalar (no detuning)
        # or an array (with detuning): somehow discriminate between the two
        # bc. cumath.cos() can't handle scalars. For now simply put an if/else,
        # think about better solutions
        if dphi_is_array:
            s_dphi_x, c_dphi_x = pm.sincos(pm.atleast_1d(dphi_x))
            s_dphi_y, c_dphi_y = pm.sincos(pm.atleast_1d(dphi_y))
            # c_dphi_x = pm.cos(dphi_x)
            # c_dphi_y = pm.cos(dphi_y)
            # s_dphi_x = pm.sin(dphi_x)
            # s_dphi_y = pm.sin(dphi_y)
        else:
            c_dphi_x = np.cos(dphi_x)
            c_dphi_y = np.cos(dphi_y)
            s_dphi_x = np.sin(dphi_x)
            s_dphi_y = np.sin(dphi_y)

        # Calculate the matrix M and transport the transverse phase
        # spaces through the segment.
        M00 = self.I[0, 0] * c_dphi_x + self.J[0, 0] * s_dphi_x
        M01 = self.I[0, 1] * c_dphi_x + self.J[0, 1] * s_dphi_x
        M10 = self.I[1, 0] * c_dphi_x + self.J[1, 0] * s_dphi_x
        M11 = self.I[1, 1] * c_dphi_x + self.J[1, 1] * s_dphi_x
        M22 = self.I[2, 2] * c_dphi_y + self.J[2, 2] * s_dphi_y
        M23 = self.I[2, 3] * c_dphi_y + self.J[2, 3] * s_dphi_y
        M32 = self.I[3, 2] * c_dphi_y + self.J[3, 2] * s_dphi_y
        M33 = self.I[3, 3] * c_dphi_y + self.J[3, 3] * s_dphi_y

        # bound to _track_with_dispersion or _track_without_dispersion
        self._track(beam, M00, M01, M10, M11, M22, M23, M32, M33)


class TransverseMap(Printing):
    """ Collection class for TransverseSegmentMap objects. This class is
    used to define a one turn map for transverse particle tracking. An
    accelerator ring is divided into segments (1 or more). They are
    defined by the user with the array s containing the positions of
    all the segment boundaries. The TransverseMap stores all the
    relevant parameters (optics) at each segment boundary. The first
    boundary of the first segment is referred to as the injection
    point.
    At instantiation of the TransverseMap, a TransverseSegmentMap object
    is created for each segment of the accelerator and appended to the
    list self.segment_maps. When generating the TransverseSegmentMaps,
    the influence of incoherent detuning by effects defined in the
    trackers.detuners module is included and the corresponding
    SegmentDetuner objects are generated on the fly. Their strength of
    detuning is distributed proportionally along the accelerator
    circumference.
    Note that the TransverseMap only knows all the relevant optics
    parameters needed to generate the TransverseSegmentMaps. It is not
    capable of tracking particles. The transport mechanism of particles
    in the transverse plane is entirely implemented in the
    TransverseSegmentMap class.
    Since the TransverseMap is implemented to act as a sequence, the
    instances of the TransverseSegmentMap objects (stored in
    self.segment_maps) can be accessed using the notation
    TransverseMap(...)[i] (with i the index of the accelerator
    segment). """
    def __init__(self, s, alpha_x, beta_x, D_x, alpha_y, beta_y, D_y,
                 accQ_x, accQ_y, detuners=[], *args, **kwargs):
        """ Create a one-turn map that manages the transverse tracking
        for each of the accelerator segments defined by s.
          - s is the array of positions defining the boundaries of the
            segments for one turn. The first element in s must be zero
            and the last element must be equal to the accelerator
            circumference C.
          - accQ_{x,y} are arrays with the accumulating phase advance
            in units of 2 \pi (i.e. mu_{x,y} / 2 \pi) at each segment
            boundary. The respective last entry gives the betatron tune
            Q_{x,y} .
            Note: instead of arrays of length len(s) it is possible
            to provide solely the scalar one-turn betatron tune Q_{x,y}
            directly. Then the phase advances are smoothly distributed
            over the segments (proportional to the respective s length).
          - alpha_{x,y}, beta_{x,y} are the TWISS parameters alpha and
            beta. They are arrays of size len(s) as these parameters
            must be defined at every segment boundary of the
            accelerator.
          - D_{x,y} are the dispersion coefficients. They are arrays of
            size len(s) as these parameters must be defined at every
            segment boundary of the accelerator.
          - detuner_collections is a list of DetunerCollection objects
            that are present in the accelerator. Each DetunerCollection
            knows how to generate and store its SegmentDetuner objects
            to 'distribute' the detuning proportionally along the
            accelerator circumference. """

        self.s = s
        self.alpha_x = alpha_x
        self.beta_x = beta_x
        self.D_x = D_x
        self.alpha_y = alpha_y
        self.beta_y = beta_y
        self.D_y = D_y
        self.accQ_x = accQ_x
        self.accQ_y = accQ_y
        self.detuner_collections = detuners

        '''List to store TransverseSegmentMap instances.'''
        self.segment_maps = []
        self._generate_segment_maps()

        if self.D_x.any() or self.D_y.any():
            self.prints('Non-zero dispersion in tracking: '
                        'ensure the beam has been generated '
                        'being matched to the correct dispersion!')

    def _generate_segment_maps(self):
        """ This method is called at instantiation of a TransverseMap
        object. For each segment of the accelerator ring (defined by the
        array self.s), a TransverseSegmentMap object is instantiated and
        appended to the list self.segment_maps. The creation of the
        TransverseSegmentMaps includes the instantiation of the
        SegmentDetuner objects which is achieved by calling the
        self.detuner_collections.generate_segment_detuner(...) method.
        The detuning strength given in a DetunerCollection is valid for
        one complete turn around the accelerator. To determine the
        detuning strength of a SegmentDetuner, the one-turn detuning
        strength is scaled to the segment_length. Note that this
        quantity is given in relative units (i.e. it is normalized to
        the accelerator circumference s[-1]). """
        segment_length = pm.diff(self.s) / self.s[-1]

        if np.ndim(self.accQ_x) == 0:
            # smooth approximation for phase advance (proportional to s)
            dQ_x = self.accQ_x * segment_length
        else:
            dQ_x = pm.diff(self.accQ_x)
        if np.ndim(self.accQ_y) == 0:
            # smooth approximation for phase advance (proportional to s)
            dQ_y = self.accQ_y * segment_length
        else:
            dQ_y = pm.diff(self.accQ_y)

        n_segments = len(self.s) - 1
        # relative phase advances for detuners:
        dmu_x = dQ_x / pm.atleast_1d(self.accQ_x)[-1]
        dmu_y = dQ_y / pm.atleast_1d(self.accQ_y)[-1]

        for seg in xrange(n_segments):
            s0 = seg % n_segments
            s1 = (seg + 1) % n_segments

            # Instantiate SegmentDetuner objects.
            for detuner in self.detuner_collections:
                detuner.generate_segment_detuner(
                    dmu_x[s0], dmu_y[s0],
                    alpha_x=self.alpha_x[s0], beta_x=self.beta_x[s0],
                    alpha_y=self.alpha_y[s0], beta_y=self.beta_y[s0],
                    )

            # Instantiate TransverseSegmentMap objects.
            transverse_segment_map = TransverseSegmentMap(
                self.alpha_x[s0], self.beta_x[s0], self.D_x[s0],
                self.alpha_x[s1], self.beta_x[s1], self.D_x[s1],
                self.alpha_y[s0], self.beta_y[s0], self.D_y[s0],
                self.alpha_y[s1], self.beta_y[s1], self.D_y[s1],
                dQ_x[seg], dQ_y[seg],
                segment_detuners=[detuner[seg]
                                  for detuner in self.detuner_collections])

            self.segment_maps.append(transverse_segment_map)

    def get_injection_optics(self):
        """Return a dict with the transverse TWISS parameters
        alpha_x, beta_x, D_x, alpha_y, beta_y, D_y from the
        beginning of the first segment (injection point).
        """
        return {
            'alpha_x': self.alpha_x[0],
            'beta_x': self.beta_x[0],
            'D_x': self.D_x[0],
            'alpha_y': self.alpha_y[0],
            'beta_y': self.beta_y[0],
            'D_y': self.D_y[0]
        }

    def __len__(self):
        return len(self.segment_maps)

    def __getitem__(self, key):
        return self.segment_maps[key]
