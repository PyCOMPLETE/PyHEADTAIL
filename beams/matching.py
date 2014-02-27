#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  RFMatch_v1.py
#  
#  Copyright 2013 Kevin Li <kevin.shing.bruce.li@cern.ch>
#  
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#  
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#  
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#  MA 02110-1301, USA.
#  
#  


import sys
from scipy import special
from scipy.integrate import quad, dblquad
from pylab import *


close('all')


e = 1.602e-19
c = 2.997e8
mp = 938e6 * e / c / c

energy = 26.
gamma = energy / 0.938 + 1
beta = sqrt(1 - 1 / gamma ** 2)
p0 = mp * gamma * beta * c

V = 4.5e6
h = 4620.
R = 6911. / 2 / pi
gamma_tr = 18.
eta = 1 / gamma_tr ** 2 - 1 / gamma ** 2

Qs = sqrt(e * V * eta * h / (2 * pi * beta * c * p0))


def main():
    global dz, dp, H0

    sigma_dz = 0.2
    sigma_dp = sigma_dz * Qs / eta / R
    H0 = eta * beta * c * sigma_dp ** 2
    epsn_z = 4 * pi * Qs / eta / R * sigma_dz ** 2 * p0 / e
    print '\nStatistical parameters from initialisation:'
    print sigma_dz, sigma_dp, epsn_z, '\n'
    #~sdz = sqrt(H0 / (e / (p0 * 2 * pi * R) * V * h / R))
    #~sdp = sqrt(H0 / (eta * beta * c))
    #~print sdz, sdp, 4*pi*sdz*sdp*p0/e, '\n'

    n_particles = 100000
    
    # Select Hamiltonian
    zmax = pi * R / h
    pmax = Qs / eta / h * 2
    H = H1
    Hmax = H(zmax, 0)
    print Hmax
    Hmax = H(0, pmax)
    print Hmax
    
    #~print zmax, pmax, Qs, h*beta*c/6911, gamma, p0, eta, beta
    #~exit(-1)
    
    #~zmax = pi * R / h
    #~pmax = Qs / eta / h * pi
    #~zmax = sigma_dz * 2
    #~pmax = sigma_dp * 2
    #~H = H2
    #~Hmax = H(zmax, 0)
    #~Hmax = H(0, pmax)
    epsn_z = pi / 2 * zmax * pmax * p0 / e
    print '\nStatistical parameters from RF bucket:'
    print zmax, pmax, epsn_z, '\n'

    #~print '\nEnergy integral:'
    #~energy1(zmax)# * p0 / e
    
    print '\nBunchlength:'
    #~#print bunchlength(H1, zmax)[0] * p0 / e
    sigma_dz = bunchlength(H1, sigma_dz)
    sigma_dp = sigma_dz * Qs / eta / R
    H0 = eta * beta * c * sigma_dp ** 2
    #~exit(-1)

    figure(3)
    zl = linspace(-zmax, zmax, 1000) * 1.5
    pl = linspace(-pmax, pmax, 1000) * 1.5

    #~y = exp(H(zl, 0) / H0) - exp(Hmax / H0)
    #~plot(zl, y)
    #~show()
    #~exit(-1)

    zz, pp = meshgrid(zl, pl)
    HH = (exp(H(zz, pp) / H0) - exp(Hmax / H0))
    HHmax = amax(HH)
    contourf(zz, pp, HH, 20)
    #~plot(zl, HH.T)
    print HHmax, Hmax, sigma_dz, sigma_dp, H0
    #~exit(-1)

    dz = np.zeros(n_particles)
    dp = np.zeros(n_particles)
    for i in range(n_particles):
        while True:
            s = (rand() - 0.5) * 2 * zmax
            #~pmax = sqrt(2) * Qs / eta / h * sqrt(1 + cos(s * h / R))
            #~pmax = sqrt(e * V) / sqrt(c * h * p0 * pi * beta * eta) * sqrt(1 + cos(s * h / R))
            t = (rand() - 0.5) * 2 * pmax
            u = (rand()) * HHmax * 1.01
            #~C = exp(H(s, t) / H0)
            C = exp(H(s, t) / H0) - exp(Hmax / H0)
            if u < C:
                #~if i % 2e4 == 0:
                    #~print i, s, t, u, C
                break
        dz[i] = s
        dp[i] = t
        
    epsz = sqrt(mean(dz * dz) * mean(dp *dp) - mean(dz * dp) * mean(dz * dp))
    print '\nStatistical parameters from distribution:'
    print std(dz), std(dp), 4*pi*std(dz)*std(dp)*p0/e, 4*pi*epsz*p0/e
    
    scatter(dz, dp, c='r', marker='.')
    
    figure(1)
    pdf, bins, patches = hist(dz, 80, color='g')
    y1 = lz1(zl, sigma_dp)
    y1 *= max(pdf) / max(y1)
    y2 = lz2(zl, sigma_dp)
    y2 *= max(pdf) / max(y2)
    plot(zl, y1, c='purple', lw=2)
    plot(zl, y2, c='brown', lw=2)
    
    figure(2)
    pdf, bins, patches = hist(dp, 80, color='r')
    y = exp((-eta * beta * c * pl ** 2 / 2. - Hmax) * 1 / H0) - 1
    y1 = lp1(pl, sigma_dp)
    y1 *= max(pdf) / max(y1)
    y2 = lp2(pl, sigma_dp)
    y2 *= max(pdf) / max(y2)
    y3 = lp3(pl, sigma_dp)
    y3 *= max(pdf) / max(y3)
    plot(pl, y1, c='purple', lw=2)
    plot(pl, y2, c='yellow', lw=2)
    plot(pl, y3, c='brown', lw=2)
    grid()
    
    show()
    return 0


def H1(dz, dp):
    H = (-1 / 2. * eta * beta * c * dp ** 2
      - sign(eta) * e * V / (p0 * 2 * pi * h) * (1 - cos(h / R * dz)))
    
    return H


def H2(dz, dp):
    H = (-eta * beta * c * dp ** 2 / 2.
    - e * V * h / (p0 * 2 * pi * R ** 2) * dz ** 2 / 2.)
    
    return H


def lz1(dz, sigma_dp):
    lz = (1 / sqrt(pi * eta * beta * c * p0 * h) * exp(-2 * Qs ** 2 / (eta ** 2 * h ** 2 * sigma_dp ** 2))
       * (-2 * sqrt(e * V * (1 + cos(h / R * dz)))
       + sqrt(2 * pi) * exp(2 * Qs ** 2 * cos(h / 2 / R * dz) ** 2 / (eta ** 2 * h ** 2 * sigma_dp ** 2))
       * sqrt(pi * eta * beta * c * p0 * h) * sigma_dp *
       special.erf(sqrt(Qs ** 2 / (eta ** 2 * h ** 2) * (1 + cos(h / R * dz))) / sigma_dp)))
    
    return lz


def lz2(dz, sigma_dp):
    lz = (1 / sqrt(eta * beta * c) * sqrt(2 / pi)
       * (-1 / R * exp(-Qs ** 2 * pi ** 2 / (2 * h ** 2 * eta ** 2 * sigma_dp ** 2))
       * sqrt(abs(R ** 2 * pi ** 2 - h ** 2 * dz ** 2)) * sqrt(e * V / h / p0)
       + exp(-Qs ** 2 * dz ** 2 / 2 / R ** 2 / eta ** 2 / sigma_dp ** 2) * pi * sqrt(eta * beta * c)
       * sigma_dp * special.erf(sqrt(Qs ** 2 / h ** 2 / eta ** 2) / sqrt(2) / R / sigma_dp * sqrt(abs(pi ** 2 * R ** 2 - h ** 2 * dz ** 2)))))

    return lz


def lp1(dp, sigma_dp):
    c1 = Qs ** 2 / eta ** 2 / h ** 2
    lp = (2 * pi * R / h * exp(-2 * c1 / sigma_dp ** 2)
       * (exp((2 * c1 - dp ** 2) / 2 / sigma_dp ** 2) * special.iv(0, c1 / sigma_dp ** 2) - 1))

    return lp


def lp2(dp, sigma_dp):
    c1 = sqrt(e * V / beta / c / p0 / eta / h) / sigma_dp
    lp = (2 * pi * R / h * (-exp(-pi / 4 * c1 ** 2)
       + 1 / c1 * exp(-dp ** 2 / 2 / sigma_dp ** 2)
       * special.erf(sqrt(pi / 4 * c1 ** 2))))
       
    return lp


def lp3(dp, sigma_dp):
    c1 = sqrt(beta * c * p0 * eta * h * sigma_dp ** 2)
    c2 = sqrt(R ** 2 * abs(pi * e * V - 2 * c1 ** 2 * dp ** 2 / sigma_dp ** 2))
    lp = (sqrt(4 * pi / (e * V * h ** 2))
       * (-exp(-pi * e * V / (4 * c1 ** 2)) * c2
       + exp(-dp ** 2 / 2 / sigma_dp ** 2) * c1 * sqrt(pi * R ** 2)
       * special.erf(c2 / 2 / R / c1)))
     
    return lp


def bunchlength(H,  sigma_dz):
    print 'Iterative evaluation of bunch length...'

    counter = 0
    eps = 1

    zmax = pi * R / h
    Hmax = H(zmax, 0)

    # Initial values
    z0 = sigma_dz
    p0 = z0 * Qs / eta / R
    H0 = eta * beta * c * p0 ** 2

    z1 = z0
    while abs(eps)>1e-6:
        dplim = lambda dz: sqrt(2) * Qs / (eta * h) * sqrt(cos(h / R * dz) - cos(h / R * zmax))
        psi = lambda dz, dp: exp(H(dz, dp) / H0) - exp(Hmax / H0)

        N = dblquad(lambda dp, dz: psi(dz, dp), -zmax, zmax, lambda dz: -dplim(dz), lambda dz: dplim(dz))
        I = dblquad(lambda dp, dz: dz ** 2 * psi(dz, dp), -zmax, zmax, lambda dz: -dplim(dz), lambda dz: dplim(dz))

        z2 = sqrt(I[0] / N[0])
        eps = z2-z0

        #~sys.stdout.write('%g, %g, %g' % (z1, z2, eps))
        print z1, z2, eps
        z1 -= eps

        p0 = z1 * Qs / eta / R
        H0 = eta * beta * c * p0 ** 2
        
        counter += 1
        if counter > 10000:
            print '\n*** WARNING: too many interation steps! Target bunchlength seems to exceed bucket. Aborting...'
            exit(-1)

    return z1


def energy1(z0):
    I = quad(lambda dz: sqrt(2) * Qs / (eta * h) * sqrt(cos(h / R * dz) - cos(h / R * z0)), -z0, z0)
    I1 = quad(lambda dz: dz ** 2 * sqrt(2) * Qs / (eta * h) * sqrt(cos(h / R * dz) - cos(h / R * z0)), -z0, z0)
    
    print I[0] * 2
    print I1[0] * 2
    
    return I[0] * 2
    
    
def Htest():
    print 'I am in the python function! Yeah!'
    exit(-1)


if __name__ == '__main__':
	main()
