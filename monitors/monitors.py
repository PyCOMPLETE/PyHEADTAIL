'''
@author: Kevin Li
@date: 11.02.2014
'''


import numpy as np


import h5py as hp
from abc import ABCMeta


class Monitor():

    @abstractmethod
    def dump(bunch):
        pass

class BunchMonitor(Monitor):

    def __init__(self, filename, n_steps):

        self.filename = filename + '.h5'
        self.n_steps = n_steps

    def dump(bunch):
        pass

class SliceMonitor(Monitor):

    def __init__(self, filename, n_steps):

        self.filename = filename + '.h5'
        self.n_steps = n_steps

    def dump(bunch):
        pass

class ParticleMonitor(Monitor):

    def __init__(self, filename, n_steps):

        self.filename = filename + '.h5'
        self.n_steps = n_steps

    def dump(bunch):
        pass
