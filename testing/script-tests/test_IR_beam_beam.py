import sys, os
BIN=os.path.expanduser('../../../')
sys.path.append(BIN)

from LHC import LHC
from PyHEADTAIL.trackers.transverse_tracking import TransverseSegmentMap 

from PyHEADTAIL.beambeam.beambeam import BeamBeam4DWeakStrong
from PyHEADTAIL.general.element import Element
from PyHEADTAIL.particles.particles import Particles
from PySUSSIX import Sussix

import numpy as np
import pylab as plt


machine = LHC(machine_configuration='6.5_TeV_collision_tunes',n_segments=1)


inj_opt = machine.transverse_map.get_injection_optics()

betastar = 40e-2
full_Xing_angle_rad = 280e-6
gamma_both_beams = machine.gamma
epsn_strong_beam = 2.e-6
intensity_strong_beam = 1e11*0.9

enable_horizontal_LR = True
enable_vertical_LR = True
n_head_on = 2.
n_long_range_per_side = 16



removeDipole = True

########################################################

full_separation_at_LR = betastar*full_Xing_angle_rad



alpha_x_s0 = inj_opt['alpha_x']
beta_x_s0 = inj_opt['beta_x']
D_x_s0 = inj_opt['D_x']
alpha_x_s1 = inj_opt['alpha_x']
beta_x_s1 = inj_opt['beta_x']
D_x_s1 = inj_opt['D_x']
alpha_y_s0 = inj_opt['alpha_y']
beta_y_s0 = inj_opt['beta_y']
D_y_s0 = inj_opt['D_y']
alpha_y_s1 = inj_opt['alpha_y']
beta_y_s1 = inj_opt['beta_y']
D_y_s1 = inj_opt['D_y']
dQ_x=0.
dQ_y=0.

IR_ele_list = []

#matching to LR
IR_ele_list.append(TransverseSegmentMap(alpha_x_s0=alpha_x_s0, beta_x_s0=beta_x_s0, D_x_s0=D_x_s0, 
    alpha_x_s1=0., beta_x_s1=betastar, D_x_s1=0.,
    alpha_y_s0=alpha_y_s0, beta_y_s0=beta_y_s0, D_y_s0=D_y_s0, 
    alpha_y_s1=0., beta_y_s1=betastar, D_y_s1=0.,
    dQ_x=.25, dQ_y=.25))

#~ #LR
if enable_horizontal_LR:
    IR_ele_list.append(BeamBeam4DWeakStrong(intensity=intensity_strong_beam*n_long_range_per_side, 
        beamSizeX = np.sqrt(epsn_strong_beam*betastar/gamma_both_beams), 
        beamSizeY = None,
        offsetX = full_separation_at_LR,
        offsetY = 0.0, 
        removeDipole = removeDipole))
        
if enable_vertical_LR:
    IR_ele_list.append(BeamBeam4DWeakStrong(intensity=intensity_strong_beam*n_long_range_per_side, 
        beamSizeX = np.sqrt(epsn_strong_beam*betastar/gamma_both_beams), 
        beamSizeY = None,
        offsetX = 0.0,
        offsetY = full_separation_at_LR, 
        removeDipole = removeDipole))
#phase advance to IP
IR_ele_list.append(TransverseSegmentMap(alpha_x_s0=0.0, beta_x_s0=betastar, D_x_s0=0.0, 
    alpha_x_s1=0., beta_x_s1=betastar, D_x_s1=0.,
    alpha_y_s0=0.0, beta_y_s0=betastar, D_y_s0=0.0, 
    alpha_y_s1=0., beta_y_s1=betastar, D_y_s1=0.,
    dQ_x=.25, dQ_y=.25))
#~ #HO
IR_ele_list.append(BeamBeam4DWeakStrong(intensity=intensity_strong_beam*n_head_on, 
        beamSizeX = np.sqrt(epsn_strong_beam*betastar/gamma_both_beams), 
        beamSizeY = None,
        offsetX = 0.0,
        offsetY = 0.0, 
        removeDipole = removeDipole))
        
#phase advance to LR
IR_ele_list.append(TransverseSegmentMap(alpha_x_s0=0.0, beta_x_s0=betastar, D_x_s0=0.0, 
    alpha_x_s1=0., beta_x_s1=betastar, D_x_s1=0.,
    alpha_y_s0=0.0, beta_y_s0=betastar, D_y_s0=0.0, 
    alpha_y_s1=0., beta_y_s1=betastar, D_y_s1=0.,
    dQ_x=.25, dQ_y=.25))
#~ #LR    
if enable_horizontal_LR:
    IR_ele_list.append(BeamBeam4DWeakStrong(intensity=intensity_strong_beam*n_long_range_per_side, 
        beamSizeX = np.sqrt(epsn_strong_beam*betastar/gamma_both_beams), 
        beamSizeY = None,
        offsetX = -full_separation_at_LR,
        offsetY = 0.0, 
        removeDipole = removeDipole))
        
if enable_vertical_LR:
    IR_ele_list.append(BeamBeam4DWeakStrong(intensity=intensity_strong_beam*n_long_range_per_side, 
        beamSizeX = np.sqrt(epsn_strong_beam*betastar/gamma_both_beams), 
        beamSizeY = None,
        offsetX = 0.0,
        offsetY = -full_separation_at_LR, 
        removeDipole = removeDipole))
        
#matching to arc
IR_ele_list.append(TransverseSegmentMap(alpha_x_s1=alpha_x_s1, beta_x_s1=beta_x_s1, D_x_s1=D_x_s1, 
    alpha_x_s0=0., beta_x_s0=betastar, D_x_s0=0.,
    alpha_y_s1=alpha_y_s1, beta_y_s1=beta_y_s1, D_y_s1=D_y_s1, 
    alpha_y_s0=0., beta_y_s0=betastar, D_y_s0=0.,
    dQ_x=.25, dQ_y=.25))
    
class LHC_bemabeam_IRs(Element):
    def __init__(self, IR_ele_list):
        self.IR_ele_list = IR_ele_list
        
    def track(self, beam):
        for ele in self.IR_ele_list:
            ele.track(beam)
            

machine.one_turn_map.append(LHC_bemabeam_IRs(IR_ele_list))

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


nTurn = 1024

nPart,coords = create_phase_space(sigma = np.sqrt(beta_x_s0*epsn_strong_beam/machine.gamma))    
bunch = Particles(nPart,1.,machine.charge, machine.mass, machine.circumference,machine.gamma,coords_n_momenta_dict=coords)


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

plt.close('all')
plt.figure(1)
plt.plot(qxs,qys,'x')
plt.axis('equal')

#plot mad
from beambeam_Footprint import parseDynapTune, Footprint
stringRep = parseDynapTune('beambeam_dynaptune',7,7);
foot = Footprint(stringRep,dSigma=1.0);
foot.repair();
plottable = foot.getPlottable();
plt.plot(plottable[0],plottable[1],'r',label=r'2E-6');


plt.show()
