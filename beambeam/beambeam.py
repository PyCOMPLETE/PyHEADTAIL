'''
Beam-beam elements

@authors: Xavier Buffat
@date:    10/06/2016
'''

from PyHEADTAIL.general.element import Element
import numpy as np
from scipy import constants as cst

class BeamBeam4DWeakStrong(Element):
    '''Weak strong beam-beam element'''
    def __init__(self, intensity,beamSizeX,beamSizeY=None,offsetX=0.0,offsetY=0.0, removeDipole = True , *args, **kwargs):
        '''Arguments:
            - intensity : number of elementary charges
            - beamSizeX [m]
            - beamSizeY (using round beam approximation if not specified)
            - offsetX : horizontal offset [m]
            - offsetY : vertical offset [m]
            - removeDipole : remove dipole componenent to avoid change of closed orbit
        '''
        self.intensity = intensity
        self.beamSizeX = beamSizeX
        if beamSizeY == None or beamSizeX==beamSizeY:
            self.round = True
            self.beamSizeY = None
        else:
            self.round = False
            self.beamSizeY == beamSizeY

        self.offsetX = offsetX
        self.offsetY = offsetY
        if removeDipole and (self.offsetX!=0 or self.offsetY!=0):
            r2 = self.offsetX**2 + self.offsetY**2
            fullKick = (1.0-np.exp(-0.5*r2/self.beamSizeX**2))/r2
            self.dipoleKickX = fullKick*self.offsetX
            self.dipoleKickY = fullKick*self.offsetY
        else:
            self.dipoleKickX = 0.0
            self.dipoleKickY = 0.0

    def track(self, beam):
        fact = self.intensity*cst.e*beam.charge/(2.0*np.pi*cst.epsilon_0*cst.c**2*beam.mass*beam.gamma)
        if self.round:
            fullSepX = beam.x + self.offsetX
            fullSepY = beam.y + self.offsetY
            r2 = fullSepX**2 + fullSepY**2
            fullKick = np.where(np.abs(r2)>0.,(1.0-np.exp(-0.5*r2/self.beamSizeX**2))/r2,0.0)
            beam.xp += fact*(fullKick*fullSepX - self.dipoleKickX)
            beam.yp += fact*(fullKick*fullSepY - self.dipoleKickY)
        else:
            print('Beam-beam of non-round beams is not implemented : abort')
            exit()
