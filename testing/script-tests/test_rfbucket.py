from __future__ import division

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.constants import c, e, m_p

from PyHEADTAIL.trackers.rf_bucket import RFBucket
from PyHEADTAIL.particles.generators import ParticleGenerator
from PyHEADTAIL.particles.generators import gaussian2D, RF_bucket_distribution

plt.style.use('seaborn-talk')
plt.switch_backend('TkAgg')
sns.set_context('notebook', font_scale=1.5,
                rc={'lines.markeredgewidth': 1})
sns.set_style('darkgrid', {
        'axes.linewidth': 2,
        'legend.fancybox': True})


# machine = 'SPS'
machine = 'HL-LHC-200'


if machine is 'HL-LHC-200' or machine is 'HL-LHC-400':
    p0 = 7000e9 * e/c
    E0 = p0*c
    gamma = np.sqrt((p0/(m_p*c))**2 + 1)
    beta = np.sqrt(1 - gamma**-2)

    C = 26658.883
    R = C/(2*np.pi)
    T0 = C/(beta*c)
    omega0 = 2*np.pi/T0
    alpha = 53.86**-2
    eta = alpha - gamma**-2

    if machine == 'HL-LHC-200':
        V_RF = [6e6, -3e6]
        h_RF = [17820, 35640]
        dphi_RF = [0, 0*np.pi]
        epsn_z = 3.8
        zcut = 0.1500
    elif machine == 'HL-LHC-400':
        V_RF = [16e6, 0*8e6]
        h_RF = [35640, 71280]
        dphi_RF = [0, 0*np.pi]
        epsn_z = 2.5
        zcut = 0.0810

elif machine == 'LHC':
    p0 = 7000e9 * e/c
    E0 = p0*c
    gamma = np.sqrt((p0/(m_p*c))**2 + 1)
    beta = np.sqrt(1 - gamma**-2)

    C = 26658.883
    R = C/(2*np.pi)
    T0 = C/(beta*c)
    omega0 = 2*np.pi/T0
    alpha = 3.225e-4
    eta = alpha - gamma**-2

    V_RF = [16e6]
    h_RF = [35640]
    dphi_RF = [0]
    epsn_z = 2.5
    zcut = 0.0755

elif machine is 'SPS':
    p0 = 26e9 * e/c
    E0 = p0*c
    gamma = np.sqrt((p0/(m_p*c))**2 + 1)
    beta = np.sqrt(1 - gamma**-2)

    C = 50*11*4*np.pi
    R = C/(2*np.pi)
    T0 = C/(beta*c)
    omega0 = 2*np.pi/T0
    alpha = 18**-2
    eta = alpha - gamma**-2

    V_RF = [5.75e6]
    h_RF = [4620]
    dphi_RF = [0]
    epsn_z = 0.35
    zcut = 0.23


# RF bucket in RF conventions - single harmonic only
# ==================================================
V, h, phi = V_RF[0], h_RF[0], dphi_RF[0]
p_increment = 0 * e/c * C/(beta*c)
phi_c = h*2*zcut/R
phi_s = np.pi


def T(DE):
    return 1/2.*eta*h*omega0**2/(beta**2*E0) * (DE/omega0)**2


def U(phi):
    return (e/(2*np.pi)*V *
            (np.cos(phi) - np.cos(phi_s) + (phi - phi_s)*np.sin(phi_s)))


def hamiltonian(phi, DE):
    return -(T(DE) - U(phi))


def eqH(phi_c):
    def f(phi):
        Hc = hamiltonian(phi_c, 0)
        return (np.sqrt(2*beta**2*E0/(eta*h*omega0**2) *
                        (U(phi) - Hc)))
    return f


# PyHEADTAIL BUCKET
# =================
rfbucket = RFBucket(
    circumference=C, charge=e, mass=m_p, gamma=gamma,
    alpha_array=[alpha], p_increment=p_increment,
    harmonic_list=h_RF, voltage_list=V_RF, phi_offset_list=dphi_RF)
Qs = rfbucket.Qs
print Qs


# TEST DEPRECATIONS
# =================
f = rfbucket.make_singleharmonic_force(V_RF[0], h_RF[0], dphi_RF[0])
print f
f = rfbucket.make_total_potential()
print f
f = rfbucket.zleft
print f


# COMPARE BUCKETS
# ===============
A = quad(eqH(phi_c), -phi_c, +phi_c)
print "\nPyHEADTAIL values from single particle trajectories"
print "\n--> Full bucket area: {:g}".format(
    rfbucket.emittance_single_particle())
print "--> Bunch length: {:g}".format(zcut)
print "--> Emittance: {:g}".format(
    rfbucket.emittance_single_particle(z=zcut))
print "--> RF emittance: {:g}".format(
    2*A[0]/e/h)
print "--> Gauss approximation emittance: {:g}".format(
    4*np.pi * (zcut)**2 * Qs/eta/R * p0/e)

print "\nBunch length for an emittance of {:g} eVs: {:g} cm".format(
    epsn_z, rfbucket.bunchlength_single_particle(epsn_z))

print "\nCanonically conjugate pair 1: z={:1.2e}, dp/p={:1.2e}".format(
    zcut, rfbucket.dp_max(2*zcut)/2.)
print "Canonically conjugate pair 2: z={:1.2e}, dp/p={:1.2e}".format(
    zcut, eqH(phi_c)(0)*omega0/E0/beta**2/2.)


eps_array = np.linspace(0, rfbucket.emittance_single_particle(), 10)
sig_array = [rfbucket.bunchlength_single_particle(eps) for eps in eps_array]
fig, ax = plt.subplots(1, figsize=(12, 7))
ax.set_title(machine)
ax.plot(sig_array, eps_array, '-o')
ax.set_xlabel(r'Bunch length $\sigma_z$ [m]')
ax.set_ylabel(r'Emittance $\varepsilon_z$ [eV s]')
ax.set_ylim(0, 1.2*rfbucket.emittance_single_particle())
plt.show()


# PLOT BUCKETS
# ============
fig, (ax1, ax2) = plt.subplots(2, figsize=(12, 10))

zz = np.linspace(1.1*rfbucket.zleft, 1.1*rfbucket.zright, 200)
pp = np.linspace(-1.5*rfbucket.dp_max(rfbucket.zleft),
                 +1.5*rfbucket.dp_max(rfbucket.zleft), 200)
phi = h*zz/R
hh = rfbucket.hamiltonian

ZZ, PP = np.meshgrid(zz, pp)
PH, EE = h*ZZ/R, PP*E0*beta**2
# print "\nHmax 1: {:e}".format(np.max(hh(ZZ, PP)))
# print "\nHmax 2: {:e}".format(np.max(hamiltonian(PH, EE)*omega0))

ax1.set_title("RF convention")
ax1.contourf(ZZ, PP, hamiltonian(PH, EE), 10, cmap=plt.cm.viridis)
ax1.contour(ZZ, PP, hamiltonian(PH, EE),
            levels=sorted([hamiltonian(h*zcut/R, 0),
                           hamiltonian(h*rfbucket.zright/R, 0)]),
            colors='orange')
ax1.plot(zcut, 0, '*', ms=20)
ax1.plot(0, eqH(phi_c)(0)*omega0/E0/beta**2/2., '*', ms=20)

ax2.set_title('PyHEADTAIL')
ax2.contourf(ZZ, PP, hh(ZZ, PP), 10, cmap=plt.cm.viridis)
ax2.contour(ZZ, PP, hh(ZZ, PP), levels=[hh(rfbucket.zright, 0), hh(zcut, 0)],
            colors='orange')
ax2.plot(zcut, 0, '*', ms=20)
ax2.plot(0, rfbucket.dp_max(2*zcut)/2, '*', ms=20)

plt.show()
plt.close('all')


# PARTICLE DISTRIBUTION
# =====================
bunch = ParticleGenerator(
    macroparticlenumber=1e6, intensity=1e11,
    charge=e, mass=m_p, gamma=gamma,
    circumference=C,
    distribution_x=gaussian2D(2e-6),
    distribution_y=gaussian2D(2e-6),
    distribution_z=RF_bucket_distribution(rfbucket,
                                          sigma_z=zcut)).generate()

print "\nPyHEADTAIL values from bunch distribution"
print "--> Bunch length: {:g} instead of {:g}".format(
    bunch.sigma_z(), zcut)
print "--> Bunch emitance: {:g} instead of {:g}".format(
    bunch.epsn_z(), rfbucket.emittance_single_particle(z=zcut))
print "--> Bunch momentum spread: {:1.2e} instead of {:1.2e}".format(
    bunch.sigma_dp(), rfbucket.dp_max(2*zcut)/2.)


wurstel


eps_geo_z = epsn_z * e/(4*np.pi*p0)
bunch1 = ParticleGenerator(
    macroparticlenumber=2e6, intensity=1e11,
    charge=e, mass=m_p, gamma=gamma,
    circumference=C,
    distribution_x=gaussian2D(2e-6),
    distribution_y=gaussian2D(2e-6),
    distribution_z=gaussian2D(eps_geo_z), Qs=Qs, eta=eta).generate()

bunch2 = ParticleGenerator(
    macroparticlenumber=2e6, intensity=1e11,
    charge=e, mass=m_p, gamma=gamma,
    circumference=C,
    distribution_x=gaussian2D(2e-6),
    distribution_y=gaussian2D(2e-6),
    distribution_z=RF_bucket_distribution(rfbucket, epsn_z=epsn_z)).generate()

print "\n\nBunch1 length: {:g}, momentum spread: {:g}".format(bunch1.sigma_z(),
                                                          bunch1.sigma_dp())
print "Bunch2 length: {:g}, momentum spread: {:g}".format(bunch2.sigma_z(),
                                                          bunch2.sigma_dp())
print "Synchrotron tune {:g}".format(Qs)
print "\nBunch length ratios {:g}".format(bunch1.sigma_z()/bunch2.sigma_z())


def gauss(x, *p):
    A, mu, sigma = p
    return A*np.exp(-(x-mu)**2/(2.*sigma**2))


fig, (ax1, ax2) = plt.subplots(2, sharex=True)

hist1, bins1, p = ax1.hist(bunch1.z, 100, normed=True)
hist2, bins2, p = ax2.hist(bunch2.z, 100, normed=True)
bins1 = (bins1[:-1] + bins1[1:])/2.
bins2 = (bins2[:-1] + bins2[1:])/2.

p0 = [1., 0., 1.]
coeff1, var_matrix = curve_fit(gauss, bins1, hist1, p0=p0)
coeff2, var_matrix = curve_fit(gauss, bins2, hist2, p0=p0)

ax1.plot(bins1, gauss(bins1, *coeff1), '-')
ax2.plot(bins2, gauss(bins2, *coeff2), '-')

ax1.text(0.1, 0.9, "$\sigma$ fit: {:g}".format(coeff1[-1]),
         transform=ax1.transAxes, fontsize=20)
ax2.text(0.1, 0.9, "$\sigma$ fit: {:g}".format(coeff2[-1]),
         transform=ax2.transAxes, fontsize=20)

plt.show()
