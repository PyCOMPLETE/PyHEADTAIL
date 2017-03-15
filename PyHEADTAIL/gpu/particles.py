# default classes imports from modules as assigned in gpu/__init__.py
from .oldinit import def_particles

import numpy as np

from pycuda import gpuarray

import thrust_interface as thrust

class ParticlesGPU(def_particles.Particles):
    '''Implementation of the Particles with data stored on the GPU.

    The flag particles_gpu.i_data_on_gpu indicates whether the data
    arrays reside on the GPU. If it is False the data should have been
    transferred to the host RAM.
    '''

    def __init__(self, *args, **kwargs):
        super(ParticlesGPU, self).__init__(*args, **kwargs)
        self.transfer_to_gpu()

    def transfer_to_gpu(self):
        '''Transfer all host RAM data to the GPU device.
        Assumes all coodinates and momenta arrays including the id
        are on the host RAM.
        '''
        for coord in self.coords_n_momenta:
            device_array = gpuarray.to_gpu(getattr(self, coord))
            setattr(self, coord, device_array)
        self.id = gpuarray.to_gpu(self.id)
        self.i_data_on_gpu = True

    def transfer_to_host(self):
        '''Transfer all GPU device data back to the host RAM.'''
        for coord in self.coords_n_momenta:
            device_array = getattr(self, coord).get()
            setattr(self, coord, device_array)
        self.id = self.id.get()
        self.i_data_on_gpu = False

    def sort_for(self, attr):
        '''Sort the named particle attribute (coordinate / momentum)
        array and reorder all particles accordingly.
        '''
        permutation = gpuarray.empty(self.macroparticlenumber, dtype=np.int32)
        thrust.get_sort_perm_double(getattr(self, attr), permutation)
        self.reorder(permutation, except_for_attrs=[attr])

    def reorder(self, permutation, except_for_attrs=[]):
        '''Reorder all particle coordinate and momentum arrays
        (in self.coords_n_momenta) and ids except for except_for_attrs
        according to the given index array permutation.
        '''
        tmp = gpuarray.empty(self.macroparticlenumber, dtype=np.float64)
        for attr in self.coords_n_momenta:
            if attr in except_for_attrs:
                continue
            unordered = getattr(self, attr)
            thrust.apply_sort_perm_double(unordered, tmp, permutation)
            setattr(self, attr, tmp)
            tmp = unordered
        del tmp
        tmp = gpuarray.empty(self.macroparticlenumber, dtype=self.id.dtype)
        thrust.apply_sort_perm_int(self.id, tmp, permutation)
        self.id = tmp


