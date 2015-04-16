
# coding: utf-8

# In[1]:

import sys, os
BIN = os.path.expanduser("../../../../")
sys.path.append(BIN)


# In[2]:

import numpy as np
from scipy.constants import m_p, c, e

from PyHEADTAIL.trackers.transverse_tracking import TransverseMap
from PyHEADTAIL.trackers.detuners import Chromaticity, AmplitudeDetuning
from PyHEADTAIL.trackers.simple_long_tracking import LinearMap
from PyHEADTAIL.particles.particles import Particles
import PyHEADTAIL.particles.generators as generators


# In[3]:

# HELPERS
def run():
    def track_n_save(bunch, map_):
        mean_x = np.empty(n_turns)
        mean_y = np.empty(n_turns)
        sigma_z = np.empty(n_turns)

        for i in xrange(n_turns):
            mean_x[i] = bunch.mean_x()
            mean_y[i] = bunch.mean_y()
            sigma_z[i] = bunch.sigma_z()

            for m_ in map_:
                m_.track(bunch)

        return mean_x, mean_y, sigma_z

    def my_fft(data):
        t = np.arange(len(data))
        fft = np.fft.rfft(data)
        fft_freq = np.fft.rfftfreq(t.shape[-1])

        return fft_freq, np.abs(fft.real)

    def generate_bunch(n_macroparticles, alpha_x, alpha_y, beta_x, beta_y, linear_map):

        intensity = 1.05e11
        sigma_z = 0.059958
        gamma = 3730.26
        gamma_t = 1. / np.sqrt(alpha_0)
        p0 = np.sqrt(gamma**2 - 1) * m_p * c

        beta_z = (linear_map.eta(dp=0, gamma=gamma) * linear_map.circumference /
                  (2 * np.pi * linear_map.Qs))

        epsn_x = 3.75e-6 # [m rad]
        epsn_y = 3.75e-6 # [m rad]
        epsn_z = 4 * np.pi * sigma_z**2 * p0 / (beta_z * e)

        bunch = generators.Gaussian6DTwiss(
            macroparticlenumber=n_macroparticles, intensity=intensity, charge=e,
            gamma_reference=gamma, mass=m_p, circumference=C,
            alpha_x=alpha_x, beta_x=beta_x, epsn_x=epsn_x,
            alpha_y=alpha_y, beta_y=beta_y, epsn_y=epsn_y,
            beta_z=beta_z, epsn_z=epsn_z).generate()

        return bunch


    # In[4]:
        # Basic parameters.
    n_turns = 3
    n_segments = 5
    n_macroparticles = 10

    Q_x = 64.28
    Q_y = 59.31
    Q_s = 0.0020443

    C = 26658.883
    R = C / (2.*np.pi)

    alpha_x_inj = 0.
    alpha_y_inj = 0.
    beta_x_inj = 66.0064
    beta_y_inj = 71.5376
    alpha_0 = [0.0003225]


    # ##### Things tested:   - Instantiation of a TransverseMap (and therewith of several     TransverseSegmentMaps as we have more than 1 segment).   - With and without detuners, i.e. instantiation of Chromaticity and     AmplitudeDetuning DetunerCollections as well as the corresponding     SegmentDetuners.   - Are betatron tunes Q_{x,y} and detuning strengths correctly     scaled to segment lengths?   - If TransverseMap is a sequence.   - TransverseSegmentMap.track(beam) method.       - Check spectrum of beam centroid motion.       - Betatron tune (implicitly checks the scaling to segment lengths)       - If chromaticity and linear synchro motion are on: synchrotron sidebands?       - If amplitude detuning is on and there is initial kick: decoherence?   - Is exception risen when s[0] != 0 or s[-1] != C?   - Is exception risen when spec. D_{x,y}?   - Get optics at injection is tested in RFBucket matching     Particles generation).

    # In[5]:

    # Parameters for transverse map.
    s = np.arange(0, n_segments + 1) * C / n_segments

    alpha_x = alpha_x_inj * np.ones(n_segments)
    beta_x = beta_x_inj * np.ones(n_segments)
    D_x = np.zeros(n_segments)

    alpha_y = alpha_y_inj * np.ones(n_segments)
    beta_y = beta_y_inj * np.ones(n_segments)
    D_y = np.zeros(n_segments)


    # In[6]:
    # TEST CASE SETUP
    def gimme(*detuners):
        trans_map = TransverseMap(
            C, s, alpha_x, beta_x, D_x, alpha_y, beta_y, D_y, Q_x, Q_y, *detuners)
        long_map = LinearMap(alpha_0, C, Q_s)
        bunch = generate_bunch(
            n_macroparticles, alpha_x_inj, alpha_y_inj, beta_x_inj, beta_y_inj,
            long_map)
        return bunch, trans_map, long_map


    # In[7]:

    # CASE I
    # Pure transverse tracking. Without detuners.
    bunch, trans_map, _ = gimme()

    map_ = trans_map
    mean_x, mean_y, sigma_z = track_n_save(bunch, map_)


    # In[8]:

    # CASE II
    # Without detuners. With linear synchrotron motion.
    bunch, trans_map, long_map = gimme()

    # This tests if TransverseMap is actually a sequence.
    trans_one_turn = [ m for m in trans_map ]

    map_ = trans_one_turn + [long_map]
    mean_x, mean_y, sigma_z = track_n_save(bunch, map_)


    # In[9]:

    # CASE III
    # With chromaticity in horizontal and vertical. With linear synchrotron motion.

    chroma = Chromaticity(Qp_x=[6], Qp_y=[10])
    bunch, trans_map, long_map = gimme(chroma)

    trans_one_turn = [ m for m in trans_map ]

    map_ = trans_one_turn + [long_map]
    mean_x, mean_y, sigma_z = track_n_save(bunch, map_)


    # In[10]:

    # CASE IV
    # With amplitude detuning. With linear synchrotron motion. With initial kick.

    ampl_det = AmplitudeDetuning.from_octupole_currents_LHC(i_focusing=200, i_defocusing=-200)
    bunch, trans_map, long_map = gimme(ampl_det)

    trans_one_turn = [ m for m in trans_map ]

    map_ = trans_one_turn + [long_map]
    bunch.x += 0.0003
    bunch.y += 0.0005

    mean_x, mean_y, sigma_z = track_n_save(bunch, map_)


    # In[11]:

    # CASE V
    # With amplitude detuning and chromaticity. With linear synchrotron motion. With initial kick.

    ampl_det = AmplitudeDetuning.from_octupole_currents_LHC(i_focusing=200, i_defocusing=-200)
    chroma = Chromaticity(Qp_x=[10], Qp_y=[6])
    bunch, trans_map, long_map = gimme(ampl_det, chroma)

    trans_one_turn = [ m for m in trans_map ]

    map_ = trans_one_turn + [long_map]
    bunch.x += 0.0003
    bunch.y += 0.0005

    mean_x, mean_y, sigma_z = track_n_save(bunch, map_)


    # In[12]:

    # Test how detuning parameters and betatron tunes are scaled
    # for the TransverseSegmentMaps.
    Qp_x = [8.]
    Qp_y = [10.]
    chroma = Chromaticity(Qp_x, Qp_y)
    trans_map = TransverseMap(
        C, s, alpha_x, beta_x, D_x, alpha_y, beta_y, D_y, Q_x, Q_y,
        chroma)

    i = 1

    app_x = 20.
    app_y = 12.
    app_xy = 30.
    ampl_det = AmplitudeDetuning(app_x, app_y, app_xy)
    trans_map = TransverseMap(
        C, s, alpha_x, beta_x, D_x, alpha_y, beta_y, D_y, Q_x, Q_y,
        ampl_det)


    # Test if optics at injection are correctly returned.
    trans_map = TransverseMap(
        C, s, alpha_x, beta_x, D_x, alpha_y, beta_y, D_y, Q_x, Q_y)
    alpha_x_inj, beta_x_inj, alpha_y_inj, beta_y_inj = trans_map.get_injection_optics()


if __name__ == '__main__':
    run()

# In[14]:
