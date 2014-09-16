'''
Created on 07.01.2014

@author: Kevin Li, Michael Schenk
'''

from __future__ import division
import numpy as np
import cython_tracker as cytrack


class TransverseSegmentMap(object):
    """
    Transverse linear transport matrix for a segment [s0, s1].
    """
    def __init__(self, alpha_x_s0, beta_x_s0, D_x_s0, alpha_x_s1, beta_x_s1, D_x_s1,
                       alpha_y_s0, beta_y_s0, D_y_s0, alpha_y_s1, beta_y_s1, D_y_s1,
                       dQ_x, dQ_y, detuner_elements):

        self.dQ_x = dQ_x
        self.dQ_y = dQ_y

        if detuner_elements:
            self.detuner_elements = detuner_elements
            self.track = self.track_with_detuners
        else:
            self.track = self.track_without_detuners

        self._build_segment_map(alpha_x_s0, beta_x_s0, D_x_s0, alpha_x_s1, beta_x_s1, D_x_s1,
                                alpha_y_s0, beta_y_s0, D_y_s0, alpha_y_s1, beta_y_s1, D_y_s1)


    def _build_segment_map(self, alpha_x_s0, beta_x_s0, D_x_s0, alpha_x_s1, beta_x_s1, D_x_s1,
                                 alpha_y_s0, beta_y_s0, D_y_s0, alpha_y_s1, beta_y_s1, D_y_s):

        # Allocate coefficient matrices.
        I = np.zeros((4, 4))
        J = np.zeros((4, 4))

        # Sine component.
        I[0, 0] = np.sqrt(beta_x_s1 / beta_x_s0)
        I[0, 1] = 0
        I[1, 0] = np.sqrt(1 / (beta_x_s0 * beta_x_s1)) * (alpha_x_s0 - alpha_x_s1)
        I[1, 1] = np.sqrt(beta_x_s0 / beta_x_s1)
        I[2, 2] = np.sqrt(beta_y_s1 / beta_y_s0)
        I[2, 3] = 0
        I[3, 2] = np.sqrt(1 / (beta_y_s0 * beta_y_s1)) * (alpha_y_s0 - alpha_y_s1)
        I[3, 3] = np.sqrt(beta_y_s0 / beta_y_s1)

        # Cosine component.
        J[0, 0] = np.sqrt(beta_x_s1 / beta_x_s0) * alpha_x_s0
        J[0, 1] = np.sqrt(beta_x_s0 * beta_x_s1)
        J[1, 0] = -np.sqrt(1 / (beta_x_s0 * beta_x_s1)) * (1 + alpha_x_s0 * alpha_x_s1)
        J[1, 1] = -np.sqrt(beta_x_s0 / beta_x_s1) * alpha_x_s1
        J[2, 2] = np.sqrt(beta_y_s1 / beta_y_s0) * alpha_y_s0
        J[2, 3] = np.sqrt(beta_y_s0 * beta_y_s1)
        J[3, 2] = -np.sqrt(1 / (beta_y_s0 * beta_y_s1)) * (1 + alpha_y_s0 * alpha_y_s1)
        J[3, 3] = -np.sqrt(beta_y_s0 / beta_y_s1) * alpha_y_s1

        self.I = I
        self.J = J


    def track_without_detuners(self, beam):

        # Phase advance.
        self.dphi_x = self.dQ_x
        self.dphi_y = self.dQ_y

        cytrack.track_transverse_without_detuners(self.I, self.J, self.dphi_x, self.dphi_y,
                                                  beam.x, beam.xp, beam.y, beam.yp)


    def track_with_detuners(self, beam):

        try:
            dphi_x = self.dphi_x
            dphi_y = self.dphi_y
        except AttributeError:
            self.dphi_x = np.zeros(beam.n_macroparticles)
            self.dphi_y = np.zeros(beam.n_macroparticles)
            dphi_x = self.dphi_x
            dphi_y = self.dphi_y

        # Phase advance (w/o. factor 2 np.pi, multiplied after detuning loop).
        dphi_x[:] = self.dQ_x
        dphi_y[:] = self.dQ_y

        for detuner in self.detuner_elements:
           detuner.detune(beam, self.dphi_x, self.dphi_y)

        cytrack.track_transverse_with_detuners(self.I, self.J, self.dphi_x, self.dphi_y,
                                               beam.x, beam.xp, beam.y, beam.yp)


class TransverseMap(object):
    """
    Collection class for the transverse segment map objects. This is the class normally instantiated by a
    user. It generates a TransverseSegmentMap for each segment in 's'.
    """
    def __init__(self, s, alpha_x, beta_x, D_x, alpha_y, beta_y, D_y, Q_x, Q_y, *detuner_collections):

        self.s       = s
        self.alpha_x = alpha_x
        self.beta_x  = beta_x
        self.D_x     = D_x
        self.alpha_y = alpha_y
        self.beta_y  = beta_y
        self.D_y     = D_y

        self.Q_x = Q_x
        self.Q_y = Q_y

        self.detuner_collections = detuner_collections

        self._generate_segment_maps()


    def _generate_segment_maps(self):

        segment_maps = []

        relative_segment_length = np.diff(self.s) / self.s[-1]
        dQ_x = self.Q_x * relative_segment_length
        dQ_y = self.Q_y * relative_segment_length

        n_segments = len(self.s) - 1
        for seg in range(n_segments):
            s0 = seg % n_segments
            s1 = (seg + 1) % n_segments

            for detuner in self.detuner_collections:
                detuner.generate_segment_detuner(relative_segment_length[s0])

            transverse_segment_map = TransverseSegmentMap(self.alpha_x[s0], self.beta_x[s0], self.D_x[s0],
                                                          self.alpha_x[s1], self.beta_x[s1], self.D_x[s1],
                                                          self.alpha_y[s0], self.beta_y[s0], self.D_y[s0],
                                                          self.alpha_y[s1], self.beta_y[s1], self.D_y[s1],
                                                          dQ_x[seg], dQ_y[seg],
                                                          [detuner[seg] for detuner in self.detuner_collections])
            segment_maps.append(transverse_segment_map)

        self.segment_maps = segment_maps


    def __len__(self):

        return len(self.segment_maps)


    def __getitem__(self, key):

        return self.segment_maps[key]
