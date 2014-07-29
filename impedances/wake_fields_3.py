'''
@class Wakes
@author Hannes Bartosik & Kevin Li & Giovanni Rumolo & Michael Schenk
@date March 2014
@Class for creation and management of wakefields from impedance sources
@copyright CERN
'''
from __future__ import division
import numpy as np

import wake_sources as WakeSources


class Wakes(object):

    def __init__(self, wake_sources, slices):

        self.slices = slices

        self.wake_kicks = []
        for ws in wake_sources:
            for kick in ws.kicks:
                self.wake_kicks.append(kick)


    @classmethod
    def table(cls, wake_file, wake_components, slices):

        self = WakeSources.WakeTable(wake_file, wake_components, slices)
        return cls((self,), slices)


    @classmethod
    def resonator(cls, R_shunt, frequency, Q, Yokoya_X1, Yokoya_Y1, Yokoya_X2, Yokoya_Y2, Yokoya_Z, slices):

        self = WakeSources.Resonator(R_shunt, frequency, Q, Yokoya_X1, Yokoya_Y1, Yokoya_X2, Yokoya_Y2,
                                     Yokoya_Z, slices)
        return cls((self,), slices)


    @classmethod
    def resonator_circular(cls, R_shunt, frequency, Q, slices):

        Yokoya_X1 = 1
        Yokoya_Y1 = 1
        Yokoya_X2 = 0
        Yokoya_Y2 = 0
        Yokoya_Z  = 0
        self = WakeSources.Resonator(R_shunt, frequency, Q, Yokoya_X1, Yokoya_Y1, Yokoya_X2, Yokoya_Y2,
                                     Yokoya_Z, slices)
        return cls((self,), slices)


    @classmethod
    def resonator_parallel_plates(cls, R_shunt, frequency, Q, slices):

        Yokoya_X1 = np.pi**2/24
        Yokoya_Y1 = np.pi**2/12
        Yokoya_X2 = -np.pi**2/24
        Yokoya_Y2 = np.pi**2/24
        Yokoya_Z  = 0
        self = WakeSources.Resonator(R_shunt, frequency, Q, Yokoya_X1, Yokoya_Y1, Yokoya_X2, Yokoya_Y2,
                                     Yokoya_Z, slices)
        return cls((self,), slices)


    @classmethod
    def resistive_wall(cls, pipe_radius, resistive_wall_length, conductivity, dz_min, Yokoya_X1, Yokoya_Y1,
                       Yokoya_X2, Yokoya_Y2, slices):
        
        self = WakeSources.ResistiveWall(pipe_radius, resistive_wall_length, conductivity, dz_min,
                                         Yokoya_X1, Yokoya_Y1, Yokoya_X2, Yokoya_Y2, slices)
        return cls((self,), slices)


    @classmethod
    def resistive_wall_circular(cls, pipe_radius, resistive_wall_length, conductivity, dz_min, slices):

        Yokoya_X1 = 1
        Yokoya_Y1 = 1
        Yokoya_X2 = 0
        Yokoya_Y2 = 0
        self = WakeSources.ResistiveWall(pipe_radius, resistive_wall_length, conductivity, dz_min,
                                         Yokoya_X1, Yokoya_Y1, Yokoya_X2, Yokoya_Y2, slices)
        return cls((self,), slices)


    @classmethod
    def resistive_wall_parallel_plates(cls, pipe_radius, length_resistive_wall, conductivity, dz_min, slices):

        Yokoya_X1 = np.pi**2/24
        Yokoya_Y1 = np.pi**2/12
        Yokoya_X2 = -np.pi**2/24
        Yokoya_Y2 = np.pi**2/24
        self = WakeSources.ResistiveWall(pipe_radius, length_resistive_wall, conductivity, dz_min,
                                         Yokoya_X1, Yokoya_Y1, Yokoya_X2, Yokoya_Y2, slices)
        return cls((self,), slices)
        

    def track(self, bunch):

        for kick in self.wake_kicks:
            kick.apply(bunch, self.slices)
