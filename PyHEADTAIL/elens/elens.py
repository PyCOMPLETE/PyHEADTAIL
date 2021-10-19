'''
@authors: Vadim Gubaidulin, Adrian Oeftiger
@date:    18.02.2020
'''
from __future__ import division

from PyHEADTAIL.general.element import Element
from PyHEADTAIL.particles import slicing
import numpy as np
from scipy.constants import c, m_e, e, pi
from scipy.interpolate import splrep, splev
from functools import wraps

from PyHEADTAIL.general import pmath as pm
from PyHEADTAIL.field_maps import efields_funcs as efields


class ElectronLens(Element):
    '''
    Contains implemenation of electron lens generated electromagnetic field acting on a particle collection.
    Acts as a localized kick of a thin element. This implementation assumes that an electromagnetic field of
    electron lens beam is not affected by interaction with a bunch. 
    '''
    '''
    Alfven current, used in the expression for the maximum tune shift from an electron lens
    as defined here: 
    Landau Damping of Beam Instabilities by Electron Lenses 
    V. Shiltsev, Y. Alexahin, A. Burov, and A. Valishev
    Phys. Rev. Lett. 119, 134802 (2017)
    '''
    I_a = 17e3
    '''Threshold for relative transverse bunch size difference
    below which the bunch is assumed to be round:
    abs(1 - sig_y / sig_x) < ratio_threshold ==> round bunch
    '''
    ratio_threshold = 1e-3
    '''Threshold for absolute transverse bunch size difference
    below which the bunch is assumed to be round:
    abs(sig_y - sig_x) < absolute_threshold ==> round bunch
    '''
    absolute_threshold = 1e-10

    def __init__(self,
                 L_e,
                 I_e,
                 sigma_x,
                 sigma_y,
                 beta_e,
                 dist,
                 offset_x=0, 
                 offset_y=0,
                 sig_check=True):
        '''Arguments:
        L_e: the length of an electron lens beam and bunch interaction region
        I_e: a list of floats defining slicewise electron lens current 
        sigma_x: transverse horizontal rms size of an electron beam
        sigma_y: transverse vertical rms size of an electron beam
        beta_e: relativistic beta of electron bunch. Negative value means electron beam is 
        counterpropagating the accelerator bunch
        dist: electron beam transverse distribution from a list ['GS', 'WB', 'KV']
        sig_check: exchanges x and y quantities for sigma_x < sigma_y
        and applies the round bunch formulae for sigma_x == sigma_y .
        sig_check defaults to True and should not usually be False.
        offset_x: Horizontal offset of an electron lens beam to the nominal beam.
        Defaults to zero, i.e. the electron lens is ideally matched to beam. 
        offset_y: Vertical offset of an electron lens beam to the nominal beam 
        Defaults to zero, i.e. the electron lens is ideally matched to beam. 
        '''
        self.slicer = slicing.UniformBinSlicer(n_slices=len(I_e), n_sigma_z=4)
        self.L_e = L_e
        self.I_e = I_e
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y
        self.beta_e = beta_e
        self.dist = dist
        self.offset_x = offset_x
        self.offset_y = offset_y

        assert dist in ['GS', 'WB', 'KV', 'LN'
                        ], ('The given distribution type is not understood.')
        if self.dist == 'GS':
            self._efieldn = efields._efieldn_mit
            if sig_check:
                self._efieldn = efields.add_sigma_check(
                    self._efieldn, self.dist)
        elif self.dist == 'WB':
            self._efieldn = efields._efieldn_wb
        elif self.dist == 'KV':
            self._efieldn = efields._efieldn_kv_b
            if sig_check:
                self._efieldn = efields.add_sigma_check(
                    self._efieldn, self.dist)
        elif self.dist=='LN':
            self._efieldn = efields._efieldn_linearized

    @classmethod
    def RoundDCElectronLens(cls, L_e, dQ_max, ratio, beta_e, dist, bunch):
        '''
        Returns an round electron lens matched with a given ratio of electron lens beam size to nominal beam size 
        Arguments:
        L_e: the length of an electron lens beam and bunch interaction region
        dQ_max: the maximum tune shift from an electron lens kick
        ratio: the ratio of electron lens beam size to nominal beam size
        beta_e: relativistic beta of electron bunch. Negative value means electron beam is 
        counterpropagating the accelerator bunch
        dist: electron beam transverse distribution from a list ['GS', 'WB', 'KV']
        bunch: the nominal bunch 
        '''
        absolute_threshold = 1e-10
        I_a = 17e3

        assert (bunch.sigma_y() - bunch.sigma_y()
                ) < absolute_threshold, ('The given bunch is not round')
        if dist == 'GS':
            I_e = e/bunch.charge*dQ_max * I_a * (bunch.mass / m_e) * (
                4 * pi *
                bunch.epsn_x()) / L_e * ratio**2 * beta_e * bunch.beta / (
                    1 + pm.abs(beta_e) * bunch.beta)
        elif dist == 'WB':
            I_e =  e/bunch.charge*3 / 4 * dQ_max * I_a * (bunch.mass / m_e) * (
                4 * pi *
                bunch.epsn_x()) / L_e * ratio**2 * beta_e * bunch.beta / (
                    1 + pm.abs(beta_e) * bunch.beta)
        elif dist == 'KV':
            I_e =  e/bunch.charge*4*dQ_max * I_a * (bunch.mass / m_e) * (
                4 * pi *
                bunch.epsn_x()) / L_e * ratio**2 * beta_e * bunch.beta / (
                    1 + np.abs(beta_e) * bunch.beta)
        elif dist == 'LN':
            I_e =  e/bunch.charge*4*dQ_max * I_a * (bunch.mass / m_e) * (
                4 * pi *
                bunch.epsn_x()) / L_e * ratio**2 * beta_e * bunch.beta / (
                    1 + np.abs(beta_e) * bunch.beta)
        else:
            I_e = 0
        return ElectronLens(L_e, [
            I_e,
        ],
            ratio * bunch.sigma_x(),
            ratio * bunch.sigma_x(),
            beta_e,
            offset_x=bunch.mean_x(),
            offset_y=bunch.mean_y(),
            dist=dist,
            sig_check=True)

    def get_max_tune_shift(self, bunch):
        '''
        '''
        if self.dist == 'GS':
            [
                dQmax,
            ] =  bunch.charge/e*self.I_e / self.I_a * m_e / bunch.mass * self.L_e / (
                4 * pi *
                bunch.epsn_x()) * (bunch.sigma_x() / self.sigma_x())**2 * (
                    1 + self.beta_e * bunch.beta) / (np.abs(self.beta_e) *
                                                     bunch.beta)
        elif self.dist == 'WB':
            [
                dQmax,
            ] = bunch.charge/e*4 * self.I_e / self.I_a * m_e / bunch.mass * self.L_e / (
                4 * pi * bunch.epsn_x()) * (1 / 3) * (
                    bunch.sigma_x() / self.sigma_x())**2 * (
                        1 + self.beta_e * bunch.beta) / (np.abs(self.beta_e) *
                                                         bunch.beta)
        elif self.dist == 'KV':
            [
                dQmax,
            ] = bunch.charge/e*self.I_e / self.I_a * m_e / bunch.mass * self.L_e / (
                4 * pi * bunch.epsn_x()) * (1 / 3) * (
                    bunch.sigma_x() / self.sigma_x())**2 * (
                        1 + self.beta_e * bunch.beta) / (np.abs(self.beta_e) *
                                                         bunch.beta)

        return dQmax

    def track(self, bunch):
        '''Add the kick from electron lens electromagnetic field to the bunch's
        transverse kicks.
        '''
        slices = bunch.get_slices(
            self.slicer, statistics=['mean_x', 'mean_y', 'sigma_x', 'sigma_y'])
        # Prefactor for round Gaussian bunch from theory:
        prefactor = -bunch.charge * self.L_e * (
            1 + self.beta_e * bunch.beta) * 1. / (bunch.gamma * bunch.mass *
                                                  (bunch.beta * c)**2)
        # Nlambda_i is the line density [Coul/m] for the current slice
        for s_i, I_i in enumerate(self.I_e):
            p_id = slices.particle_indices_of_slice(s_i)
            if len(p_id) == 0:
                continue
            Nlambda_i = I_i / (self.beta_e * c)
            # Offset for an electron lens
            en_x, en_y = self.get_efieldn(pm.take(bunch.x, p_id),
                                          pm.take(bunch.y, p_id), (self.offset_x), (self.offset_y),
                                          self.sigma_x, self.sigma_y)
            kicks_x = (en_x * Nlambda_i) * prefactor
            kicks_y = (en_y * Nlambda_i) * prefactor

            kicked_xp = pm.take(bunch.xp, p_id) + kicks_x
            kicked_yp = pm.take(bunch.yp, p_id) + kicks_y

            pm.put(bunch.xp, p_id, kicked_xp)
            pm.put(bunch.yp, p_id, kicked_yp)

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
