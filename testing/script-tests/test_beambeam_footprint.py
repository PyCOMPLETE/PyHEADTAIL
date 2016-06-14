import sys, os
BIN=os.path.expanduser('../../../')
sys.path.append(BIN)

from LHC import LHC
from PyHEADTAIL.beambeam.beambeam import BeamBeam4DWeakStrong
from PyHEADTAIL.particles.particles import Particles
from PySUSSIX import Sussix

import numpy as np
from matplotlib import pyplot as plt

def create_phase_space(sigma,nSigma = 7,nAngle = 6,minAmpl = 0.05):
    x = []
    y = []
    x.append(minAmpl)
    y.append(minAmpl)
    x.append(minAmpl)
    y.append(minAmpl)
    for i in np.arange(1.0,nSigma):
        x.append(i)
        y.append(minAmpl)
        for j in range(1,nAngle):
            angle = 0.5*j*np.pi/nAngle
            x.append(i*np.cos(angle))
            y.append(i*np.sin(angle))
        x.append(minAmpl)
        y.append(i)
    return len(x),{'x': np.ascontiguousarray(x)*sigma,'xp': np.ascontiguousarray(np.zeros_like(x)),'y': np.ascontiguousarray(y)*sigma,'yp': np.ascontiguousarray(np.zeros_like(y)),'z':np.ascontiguousarray(np.zeros_like(y)),'dp':np.ascontiguousarray(np.zeros_like(y))}


if __name__ == '__main__':
    epsn_x  = 3.5e-6
    epsn_y  = 3.5e-6
    intensity = 1e11
    beta = 1.0

    nTurn = 1024

    machine = LHC(machine_configuration='6.5_TeV_collision_tunes',optics_mode='non-smooth',name=['START','END'],s=[0.0,26658.8832],beta_x=[beta,beta],beta_y=[beta,beta],alpha_x=[0.0,0.0],alpha_y=[0.0,0.0])
    sigma = np.sqrt(beta*epsn_x/machine.gamma)
    nPart,coords = create_phase_space(sigma)
    bunch = Particles(nPart,intensity/nPart,machine.charge, machine.mass, machine.circumference,machine.gamma,coords_n_momenta_dict=coords)
    bbElement = BeamBeam4DWeakStrong(intensity,sigma)
    machine.one_turn_map.append(bbElement)

    x = np.zeros((nPart,nTurn),dtype=float)
    xp = np.zeros((nPart,nTurn),dtype=float)
    y = np.zeros((nPart,nTurn),dtype=float)
    yp = np.zeros((nPart,nTurn),dtype=float)
    z = np.zeros((nPart,nTurn),dtype=float)
    dp = np.zeros((nPart,nTurn),dtype=float)
    for i in range(nTurn):
        machine.track(bunch)
        x[:,i] = bunch.x
        xp[:,i] = bunch.xp
        y[:,i] = bunch.y
        yp[:,i] = bunch.yp
        z[:,i] = bunch.z
        dp[:,i] = bunch.dp

    qxs = np.zeros(nPart,dtype=float)
    qys = np.zeros(nPart,dtype=float)
    sussix = Sussix()
    sussix.sussix_inp(nt1=1,nt2=nTurn,tunex=0.31,tuney=0.32);
    for i in range(nPart):
        sussix.sussix(x[i,:],xp[i,:],y[i,:],yp[i,:],z[i,:],dp[i,:])
        qxs[i] = sussix.ox[0]
        qys[i] = sussix.oy[0]
    plt.figure(1)
    plt.plot(qxs,qys,'x')
    plt.show()

