from __future__ import division
'''
@class Cavity
@author Kevin Li
@date 03.02.2014
@brief Class for creation and management of the synchrotron transport matrices
@copyright CERN
'''


import numpy as np
import sys


from beams.distributions import stationary_exponential
from scipy.integrate import quad, dblquad
from abc import ABCMeta, abstractmethod 
from scipy.constants import c, e



class LongitudinalTracker(object):

    __metaclass__ = ABCMeta

    @abstractmethod
    def hamiltonian():

        return None

    @abstractmethod
    def separatrix():

        return None

    @abstractmethod
    def isin_separatrix():

        return None


#~ # @profile
#~ def match_simple(bunch, cavity):
#~ 
    #~ p0 = bunch.mass * bunch.gamma * bunch.beta * c
    #~ sigma_dz = np.std(bunch.dz)
    #~ sigma_dp = np.std(bunch.dp)
    #~ epsn_z = 4 * np.pi * sigma_dz * sigma_dp * p0 / e
#~ 
    #~ n_particles = len(bunch.dz)
    #~ for i in xrange(n_particles):
        #~ if not cavity.isin_separatrix(bunch.dz[i], bunch.dp[i], bunch):
            #~ while not cavity.isin_separatrix(bunch.dz[i], bunch.dp[i], bunch):
                #~ bunch.dz[i] = sigma_dz * np.random.randn()
                #~ bunch.dp[i] = sigma_dp * np.random.randn()
#~ 
#~ def match_full(bunch, cavity):
#~ 
    #~ p0 = bunch.mass * bunch.gamma * bunch.beta * c
    #~ R = cavity.circumference / (2 * np.pi)
    #~ eta = cavity.eta(bunch)
    #~ Qs = cavity.Qs(bunch)
#~ 
    #~ zmax = np.pi * R / cavity.h
    #~ pmax = 2 * Qs / eta / cavity.h
    #~ Hmax1 = cavity.hamiltonian(zmax, 0, bunch)
    #~ Hmax2 = cavity.hamiltonian(0, pmax, bunch)
    #~ # print Hmax1 - Hmax2
    #~ # assert(Hmax1 == Hmax2)
    #~ Hmax = Hmax1
    #~ epsn_z = np.pi / 2 * zmax * pmax * p0 / e
    #~ print '\nStatistical parameters from RF bucket:'
    #~ print 'zmax:', zmax, 'pmax:', pmax, 'epsn_z:', epsn_z
#~ 
    #~ # Assuming a gaussian-type stationary distribution
    #~ sigma_dz = np.std(bunch.dz)
    #~ sigma_dp = sigma_dz * Qs / eta / R
    #~ H0 = eta * bunch.beta * c * sigma_dp ** 2
    #~ epsn_z = 4 * np.pi * Qs / eta / R * sigma_dz ** 2 * p0 / e
    #~ print '\nStatistical parameters from initialisation:'
    #~ print 'sigma_dz:', sigma_dz, 'sigma_dp:', sigma_dp, 'epsn_z:', epsn_z
    #~ 
    #~ print '\nBunchlength:'
    #~ sigma_dz = bunchlength(bunch, cavity, sigma_dz)
    #~ sigma_dp = sigma_dz * Qs / eta / R
    #~ H0 = eta * bunch.beta * c * sigma_dp ** 2
#~ 
    #~ psi = stationary_exponential(cavity.hamiltonian, Hmax, H0, bunch)
#~ 
    #~ zl = plt.linspace(-zmax, zmax, 1000) * 1.5
    #~ pl = plt.linspace(-pmax, pmax, 1000) * 1.5
    #~ zz, pp = plt.meshgrid(zl, pl)
    #~ HH = psi(zz, pp)
    #~ HHmax = np.amax(HH)
#~ 
    #~ n_particles = len(bunch.x)
    #~ for i in xrange(n_particles):
        #~ while True:
            #~ s = (np.random.rand() - 0.5) * 2 * zmax
            #~ t = (np.random.rand() - 0.5) * 2 * pmax
            #~ u = (np.random.rand()) * HHmax * 1.01
            #~ C = psi(s, t)
            #~ if u < C:
                #~ break
        #~ bunch.dz[i] = s
        #~ bunch.dp[i] = t
        #~ 
    #~ epsz = np.sqrt(np.mean(bunch.dz * bunch.dz) * np.mean(bunch.dp * bunch.dp)
                 #~ - np.mean(bunch.dz * bunch.dp) * np.mean(bunch.dz * bunch.dp))
    #~ print '\nStatistical parameters from distribution:'
    #~ print 'sigma_dz:', np.std(bunch.dz), 'sigma_dp:', np.std(bunch.dp), \
          #~ 'epsn_z:', 4*np.pi*np.std(bunch.dz)*np.std(bunch.dp)*p0/e, 4*np.pi*epsz*p0/e
#~ 
#~ def bunchlength(bunch, cavity, sigma_dz):
#~ 
    #~ print 'Iterative evaluation of bunch length...'
#~ 
    #~ counter = 0
    #~ eps = 1
#~ 
    #~ p0 = bunch.mass * bunch.gamma * bunch.beta * c
    #~ R = cavity.circumference / (2 * np.pi)
    #~ eta = cavity.eta(bunch)
    #~ Qs = cavity.Qs(bunch)
#~ 
    #~ zmax = np.pi * R / cavity.h
    #~ Hmax = cavity.hamiltonian(zmax, 0, bunch)
#~ 
    #~ # Initial values
    #~ z0 = sigma_dz
    #~ p0 = z0 * Qs / eta / R            #Matching condition
    #~ H0 = eta * bunch.beta * c * p0 ** 2
#~ 
    #~ z1 = z0
    #~ while abs(eps)>1e-6:
        #~ # cf1 = 2 * Qs ** 2 / (eta * h) ** 2
        #~ # dplim = lambda dz: np.sqrt(cf1 * (1 + np.cos(h / R * dz) + (h / R * dz - np.pi) * np.sin(cavity.phi_s)))
        #~ # dplim = lambda dz: np.sqrt(2) * Qs / (eta * h) * np.sqrt(np.cos(h / R * dz) - np.cos(h / R * zmax))
        #~ # Stationary distribution
        #~ # psi = lambda dz, dp: np.exp(cavity.hamiltonian(dz, dp, bunch) / H0) - np.exp(Hmax / H0)
#~ 
        #~ # zs = zmax / 2.
#~ 
        #~ psi = stationary_exponential(cavity.hamiltonian, Hmax, H0, bunch)
        #~ dplim = cavity.separatrix.__get__(cavity)
        #~ N = dblquad(lambda dp, dz: psi(dz, dp), -zmax, zmax,
                    #~ lambda dz: -dplim(dz, bunch), lambda dz: dplim(dz, bunch))
        #~ I = dblquad(lambda dp, dz: dz ** 2 * psi(dz, dp), -zmax, zmax,
                    #~ lambda dz: -dplim(dz, bunch), lambda dz: dplim(dz, bunch))
#~ 
        #~ # Second moment
        #~ z2 = np.sqrt(I[0] / N[0])
        #~ eps = z2 - z0
#~ 
        #~ # print z1, z2, eps
        #~ z1 -= eps
#~ 
        #~ p0 = z1 * Qs / eta / R
        #~ H0 = eta * bunch.beta * c * p0 ** 2
#~ 
        #~ counter += 1
        #~ if counter > 100:
            #~ print "\n*** WARNING: too many interation steps! There are several possible reasons for that:"
            #~ print "1. Is the Hamiltonian correct?"
            #~ print "2. Is the stationary distribution function convex around zero?"
            #~ print "3. Is the bunch too long to fit into the bucket?"
            #~ print "4. Is this algorithm not qualified?"
            #~ print "Aborting..."
            #~ sys.exit(-1)
#~ 
    #~ return z1


class RFCavity(LongitudinalTracker):
    '''
    classdocs
    '''

    def __init__(self, circumference, length, gamma_transition, frequency, voltage, phi_s, integrator='rk4'):
        '''
        Constructor
        '''
        self.integrator = integrator

        self.i_turn = 0
        self.time = 0

        self.circumference = circumference
        self.length = length
        self.gamma_transition = gamma_transition
        self.h = frequency
        self.voltage = voltage
        self.phi_s = phi_s

        # self._p0 = 0
        # self._R = 0
        # self._omega_0 = 0
        # self._omega_rf = 0
        # self._h = 0

        # self._eta = 0
        # self._Qs = 0

    def eta(self, bunch):

        eta = 1. / self.gamma_transition ** 2 - 1. / bunch.gamma ** 2

        return eta

    def Qs(self, bunch):
        '''
        Synchrotron tune derived from the linearized Hamiltonian

        .. math::
        H = -1 / 2 * eta * beta * c * delta ** 2
           - e * V / (p0 * 2 * np.pi * h) * (np.cos(phi) - np.cos(phi_s) + (phi - phi_s) * np.sin(phi_s))
        '''
        p0 = bunch.mass * bunch.gamma * bunch.beta * c
        Qs = np.sqrt(e * self.voltage * np.abs(self.eta(bunch)) * self.h / (2 * np.pi * p0 * bunch.beta * c))

        return Qs
    
    def hamiltonian(self, dz, dp, bunch):

        p0 = bunch.mass * bunch.gamma * bunch.beta * c
        R = self.circumference / (2 * np.pi)
        eta = self.eta(bunch)

        phi = self.h / R * dz + self.phi_s

        H = -1 / 2 * eta * bunch.beta * c * dp ** 2 \
           + e * self.voltage / (p0 * 2 * np.pi * self.h) \
           * (np.cos(phi) - np.cos(self.phi_s) + (phi - self.phi_s) * np.sin(self.phi_s))

        return H

    def separatrix(self, dz, bunch):
        '''
        Separatriox defined by

        .. math::
        p(dz): (H(dz, dp) == H(zmax, 0))
        '''
        # p0 = bunch.mass * bunch.gamma * bunch.beta * c
        R = self.circumference / (2 * np.pi) 
        eta = self.eta(bunch)
        Qs = self.Qs(bunch) # np.sqrt(e * self.voltage * np.abs(eta) * self.h / (2 * np.pi * p0 * bunch.beta * c))

        phi = self.h / R * dz + self.phi_s 
        cf1 = 2 * Qs ** 2 / (eta * self.h) ** 2

        p =  cf1 * (1 + np.cos(phi) + (phi - np.pi) * np.sin(self.phi_s))
        p = np.sqrt(p)

        return p

    def isin_separatrix(self, dz, dp, bunch):

        # p0 = bunch.mass * bunch.gamma * bunch.beta * c
        R = self.circumference / (2 * np.pi)
        eta = self.eta(bunch)
        Qs = self.Qs(bunch) #np.sqrt(e * self.voltage * np.abs(eta) * h / (2 * np.pi * p0 * bunch.beta * c))

        phi = self.h / R * dz + self.phi_s
        cf1 = 2 * Qs ** 2 / (eta * self.h) ** 2

        zmax = np.pi * R / self.h
        pmax = cf1 * (-1 - np.cos(phi) + (np.pi - phi) * np.sin(self.phi_s))

        isin = np.abs(dz) < zmax and dp ** 2 < np.abs(pmax)

        return isin

    def track(self, bunch):

        p0 = bunch.mass * bunch.gamma * bunch.beta * c
        R = self.circumference / (2 * np.pi)
        eta = self.eta(bunch)
         
        cf1 = self.h / R
        cf2 = np.sign(eta) * e * self.voltage / (p0 * bunch.beta * c)

        if self.integrator == 'rk4':
            # Initialize
            dz0 = bunch.dz
            dp0 = bunch.dp

            # Integration
            k1 = -eta * self.length * dp0
            kp1 = cf2 * np.sin(cf1 * dz0 + self.phi_s)
            k2 = -eta * self.length * (dp0 + kp1 / 2)
            kp2 = cf2 * np.sin(cf1 * (dz0 + k1 / 2) + self.phi_s)
            k3 = -eta * self.length * (dp0 + kp2 / 2)
            kp3 = cf2 * np.sin(cf1 * (dz0 + k2 / 2) + self.phi_s)
            k4 = -eta * self.length * (dp0 + kp3);
            kp4 = cf2 * np.sin(cf1 * (dz0 + k3) + self.phi_s)

            # Finalize
            bunch.dz = dz0 + k1 / 6 + k2 / 3 + k3 / 3 + k4 / 6
            bunch.dp = dp0 + kp1 / 6 + kp2 / 3 + kp3 / 3 + kp4 / 6

        elif self.integrator == 'euler-chromer':
            # Length L drift
            bunch.dz += - eta * self.length * bunch.dp
            # Full turn kick
            bunch.dp += cf2 * np.sin(cf1 * bunch.dz + self.phi_s)

        elif self.integrator == 'ruth4':
            alpha = 1 - 2 ** (1 / 3)
            d1 = 1 / (2 * (1 + alpha))
            d2 = alpha / (2 * (1 + alpha))
            d3 = alpha / (2 * (1 + alpha))
            d4 = 1 / (2 * (1 + alpha))
            c1 = 1 / (1 + alpha)
            c2 = (alpha - 1) / (1 + alpha)
            c3 = 1 / (1 + alpha)
            # c4 = 0;

            # Initialize
            dz0 = bunch.dz
            dp0 = bunch.dp

            dz1 = dz0
            dp1 = dp0
            # Drift
            dz1 += d1 * -eta * self.length * dp1
            # Kick
            dp1 += c1 * cf2 * np.sin(cf1 * dz1 + self.phi_s)

            dz2 = dz1
            dp2 = dp1
            # Drift
            dz2 += d2 * -eta * self.length * dp2
            # Kick
            dp2 += c2 * cf2 * np.sin(cf1 * dz2 + self.phi_s)

            dz3 = dz2
            dp3 = dp2
            # Drift
            dz3 += d3 * -eta * self.length * dp3
            # Kick
            dp3 += c3 * cf2 * np.sin(cf1 * dz3 + self.phi_s)

            dz4 = dz3
            dp4 = dp3
            # Drift
            dz4 += d4 * -eta * self.length * dp4

            # Finalize
            bunch.dz = dz4
            bunch.dp = dp4
            
        bunch.update_slices()


class CSCavity(object):
    '''
    classdocs
    '''

    def __init__(self, circumference, gamma_transition, Qs):

        self.circumference = circumference
        self.gamma_transition = gamma_transition
        self.Qs = Qs

    def track(self, bunch):

        p0 = bunch.mass * bunch.gamma * bunch.beta * c
        eta = 1 / self.gamma_transition ** 2 - 1 / bunch.gamma ** 2

        omega_0 = 2 * np.pi * bunch.beta * c / self.circumference
        omega_s = self.Qs * omega_0

        dQs = 2 * np.pi * self.Qs
        cosdQs = np.cos(dQs)
        sindQs = np.sin(dQs)

        dz0 = bunch.dz
        dp0 = bunch.dp
    
        bunch.dz = dz0 * cosdQs - eta * c / omega_s * dp0 * sindQs
        bunch.dp = dp0 * cosdQs + omega_s / eta / c * dz0 * sindQs
