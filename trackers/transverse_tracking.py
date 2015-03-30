"""
@author Kevin Li, Michael Schenk
@date 07. January 2014
@brief Description of the transport of transverse phase spaces.
@copyright CERN
"""
from __future__ import division
import numpy as np

from . import Element

sin = np.sin
cos = np.cos


class TransverseSegmentMap(Element):
    """ Class to transport/track the particles of the beam in the
    transverse plane through an accelerator ring segment defined by its
    boundaries [s0, s1]. To calculate the transverse linear transport
    matrix M that transports each particle's transverse phase space
    coordinates (x, xp, y, yp) from position s0 to position s1 in the
    accelerator, the TWISS parameters alpha and beta at positions s0
    and s1 must be provided. The betatron phase advance of each
    particle in the present segment is given by their betatron tune
    Q_{x,y} and possibly by an incoherent tune shift introduced e.g. by
    amplitude detuning or chromaticity effects (see trackers.detuners
    module).

    TODO Implement dispersion effects, i.e. the change of a particle's
    transverse phase space coordinates on its relative momentum offset.
    For the moment, a NotImplementedError is raised if dispersion
    coefficients are non-zero. """
    def __init__(self,
            alpha_x_s0, beta_x_s0, D_x_s0, alpha_x_s1, beta_x_s1, D_x_s1,
            alpha_y_s0, beta_y_s0, D_y_s0, alpha_y_s1, beta_y_s1, D_y_s1,
            dQ_x, dQ_y, *args, **kwargs):
        """ Return an instance of the TransverseSegmentMap class. The
        values of the TWISS parameters alpha_{x,y} and beta_{x,y} as
        well as of the dispersion coefficients D_{x,y} (not yet
        implemented) are given at the beginning s0 and at the end s1 of
        the corresponding segment. The dQ_{x,y} denote the betatron
        tunes normalized to the (relative) segment length. The
        SegmentDetuner objects present in this segment are passed as a
        list via the keyword argument 'segment_detuners'.
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
        self.I[0,0] = np.sqrt(beta_x_s1 / beta_x_s0)
        self.I[0,1] = 0.
        self.I[1,0] = (np.sqrt(1. / (beta_x_s0 * beta_x_s1)) *
                      (alpha_x_s0 - alpha_x_s1))
        self.I[1,1] = np.sqrt(beta_x_s0 / beta_x_s1)
        self.I[2,2] = np.sqrt(beta_y_s1 / beta_y_s0)
        self.I[2,3] = 0.
        self.I[3,2] = (np.sqrt(1. / (beta_y_s0 * beta_y_s1)) *
                      (alpha_y_s0 - alpha_y_s1))
        self.I[3,3] = np.sqrt(beta_y_s0 / beta_y_s1)

        # Cosine component.
        self.J[0,0] = np.sqrt(beta_x_s1 / beta_x_s0) * alpha_x_s0
        self.J[0,1] = np.sqrt(beta_x_s0 * beta_x_s1)
        self.J[1,0] = -(np.sqrt(1. / (beta_x_s0 * beta_x_s1)) *
                      (1. + alpha_x_s0 * alpha_x_s1))
        self.J[1,1] = -np.sqrt(beta_x_s0 / beta_x_s1) * alpha_x_s1
        self.J[2,2] = np.sqrt(beta_y_s1 / beta_y_s0) * alpha_y_s0
        self.J[2,3] = np.sqrt(beta_y_s0 * beta_y_s1)
        self.J[3,2] = -(np.sqrt(1. / (beta_y_s0 * beta_y_s1)) *
                      (1. + alpha_y_s0 * alpha_y_s1))
        self.J[3,3] = -np.sqrt(beta_y_s0 / beta_y_s1) * alpha_y_s1

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

        for element in self.segment_detuners:
            detune_x, detune_y = element.detune(beam)
            dphi_x += detune_x
            dphi_y += detune_y

        dphi_x *= 2.*np.pi
        dphi_y *= 2.*np.pi

        c_dphi_x = cos(dphi_x)
        c_dphi_y = cos(dphi_y)
        s_dphi_x = sin(dphi_x)
        s_dphi_y = sin(dphi_y)

        # Calculate the matrix M and transport the transverse phase
        # spaces through the segment.
        M00 = self.I[0,0] * c_dphi_x + self.J[0,0] * s_dphi_x
        M01 = self.I[0,1] * c_dphi_x + self.J[0,1] * s_dphi_x
        M10 = self.I[1,0] * c_dphi_x + self.J[1,0] * s_dphi_x
        M11 = self.I[1,1] * c_dphi_x + self.J[1,1] * s_dphi_x
        M22 = self.I[2,2] * c_dphi_y + self.J[2,2] * s_dphi_y
        M23 = self.I[2,3] * c_dphi_y + self.J[2,3] * s_dphi_y
        M32 = self.I[3,2] * c_dphi_y + self.J[3,2] * s_dphi_y
        M33 = self.I[3,3] * c_dphi_y + self.J[3,3] * s_dphi_y

        beam.x += -self.D_x_s0 * beam.dp
        beam.y += -self.D_y_s0 * beam.dp

        beam.x, beam.xp = M00*beam.x + M01*beam.xp, M10*beam.x + M11*beam.xp
        beam.y, beam.yp = M22*beam.y + M23*beam.yp, M32*beam.y + M33*beam.yp

        beam.x += self.D_x_s1 * beam.dp
        beam.y += self.D_y_s1 * beam.dp


class TransverseMap(object):
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
    def __init__(self, C, s, alpha_x, beta_x, D_x, alpha_y, beta_y, D_y,
                 Q_x, Q_y, *detuner_collections):
        """ Create a one-turn map that manages the transverse tracking
        for each of the accelerator segments defined by s.
          - s is the array of positions defining the boundaries of the
            segments for one turn. The first element in s must be zero
            and the last element must be equal to the accelerator
            circumference C.
          - alpha_{x,y}, beta_{x,y} are the TWISS parameters alpha and
            beta. They are arrays of size len(s) as these parameters
            must be defined at every segment boundary of the
            accelerator.
          - D_{x,y} are the dispersion coefficients. They are arrays of
            size len(s) as these parameters must be defined at every
            segment boundary of the accelerator.
            WARNING: Dispersion effects are not yet implemented.
          - Q_{x,y} are scalar values and define the betatron tunes
            (i.e. the number of betatron oscillations in one complete
            turn).
          - detuner_collections is a list of DetunerCollection objects
            that are present in the accelerator. Each DetunerCollection
            knows how to generate and store its SegmentDetuner objects
            to 'distribute' the detuning proportionally along the
            accelerator circumference. """
        if not np.allclose([s[0], s[-1]], [0., C]):
            raise ValueError('The first element of s must be zero \n' +
                'and the last element must be equal to the \n' +
                'accelerator circumference C. \n')
        self.s = s
        self.alpha_x = alpha_x
        self.beta_x = beta_x
        self.D_x = D_x
        self.alpha_y = alpha_y
        self.beta_y = beta_y
        self.D_y = D_y
        self.Q_x = Q_x
        self.Q_y = Q_y
        self.detuner_collections = detuner_collections

        '''List to store TransverseSegmentMap instances.'''
        self.segment_maps = []
        self._generate_segment_maps()

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
        segment_length = np.diff(self.s) / self.s[-1]

        # Betatron motion normalized to this particular segment.
        dQ_x = self.Q_x * segment_length
        dQ_y = self.Q_y * segment_length

        n_segments = len(self.s) - 1
        for seg in range(n_segments):
            s0 = seg % n_segments
            s1 = (seg + 1) % n_segments

            # Instantiate SegmentDetuner objects.
            for detuner in self.detuner_collections:
                detuner.generate_segment_detuner(segment_length[s0],
                    alpha_x=self.alpha_x[0], beta_x=self.beta_x[s0],
                    alpha_y=self.alpha_y[0], beta_y=self.beta_y[s0],
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
        """ Return a tuple with the transverse TWISS parameters
        (alpha_x, beta_x, alpha_y, beta_y) from the beginning of the
        first segment (injection point). """
        return (self.alpha_x[0], self.beta_x[0],
                self.alpha_y[0], self.beta_y[0])

    def __len__(self):
        return len(self.segment_maps)

    def __getitem__(self, key):
        return self.segment_maps[key]
