"""
@author Kevin Li, Michael Schenk
@date 11. February 2014
@brief Implementation of monitors to store bunch-, slice- or particle-
       specific data to a HDF5 file.
@copyright CERN
"""
from __future__ import division

import h5py as hp
import numpy as np
import sys

from abc import ABCMeta, abstractmethod


class Monitor(object):
    """ Abstract base class for monitors. A monitor can request
    statistics data such as mean value and standard deviation and store
    the results in an HDF5 file. """

    __metaclass__ = ABCMeta

    @abstractmethod
    def dump(bunch):
        """ Write particle data given by bunch (instance of Particles
        class) to buffer and/or file at the specific time the method is
        called. Data can e.g. be bunch-specific, slice_set-specific or
        particle-specific. """
        pass


class BunchMonitor(Monitor):
    """ Class to store bunch-specific data to a HDF5 file. This monitor
    uses a buffer (a shift register) to reduce the number of writing
    operations to file. This also helps to avoid IO errors and loss of
    data when writing to a file that may become temporarily unavailable
    (e.g. if file is located on network) during the simulation. """
    def __init__(self, filename, n_steps, parameters_dict=None,
                 write_buffer_every=512, buffer_size=4096):
        """ Create an instance of a BunchMonitor class. Apart from
        initializing the HDF5 file, a self.buffer dictionary is
        prepared to buffer the data before writing them to file.

          filename:           Path and name of HDF5 file. Without file
                              extension.
          n_steps:            Number of entries to be reserved for each
                              of the quantities in self.stats_to_store.
          parameters_dict:    Metadata for HDF5 file containing main
                              simulation parameters.
          write_buffer_every: Number of steps after which buffer
                              contents are actually written to file.
          buffer_size:        Number of steps to be buffered. """
        self.stats_to_store = [
            'mean_x', 'mean_xp', 'mean_y', 'mean_yp', 'mean_z', 'mean_dp',
            'sigma_x', 'sigma_y', 'sigma_z', 'sigma_dp', 'epsn_x', 'epsn_y',
            'epsn_z', 'macroparticlenumber' ]
        self.filename = filename
        self.n_steps = n_steps
        self.i_steps = 0

        self._create_file_structure(parameters_dict)

        # Prepare buffer.
        self.buffer_size = buffer_size
        self.write_buffer_every = write_buffer_every
        self.buffer = {}

        for stats in self.stats_to_store:
            self.buffer[stats] = np.zeros(self.buffer_size)

    def dump(self, bunch):
        """ Evaluate the statistics like mean and standard deviation for
        the given bunch and write the data to the HDF5 file. Make use of
        a buffer to reduce the number of writing operations to file.
        This helps to avoid IO errors and loss of data when writing data
        to a file that may become temporarily unavailable (e.g. if file
        is on network). during the simulation. Buffer contents are
        written to file only every self.write_buffer_every steps. """
        self._write_data_to_buffer(bunch)
        if ((self.i_steps + 1) % self.write_buffer_every == 0 or
                (self.i_steps + 1) == self.n_steps):
            self._write_buffer_to_file()

        self.i_steps += 1

    def _create_file_structure(self, parameters_dict):
        """ Initialize HDF5 file and create its basic structure (groups
        and datasets). One group is created for bunch-specific data.
        One dataset for each of the quantities defined in
        self.stats_to_store is generated.
        If specified by the user, write the contents of the
        parameters_dict as metadata (attributes) to the file.
        Maximum file compression is activated. """
        try:
            h5file = hp.File(self.filename + '.h5', 'w')
            if parameters_dict:
                for key in parameters_dict:
                    h5file.attrs[key] = parameters_dict[key]

            h5file.create_group('Bunch')
            h5group = h5file['Bunch']
            for stats in sorted(self.stats_to_store):
                h5group.create_dataset(stats, shape=(self.n_steps,),
                                       compression='gzip', compression_opts=9)
            h5file.close()
        except:
            print ('Creation of bunch monitor file ' + self.filename +
                   'failed. \n')
            raise

    def _write_data_to_buffer(self, bunch):
        """ Store the data in the self.buffer dictionary before writing
        them to file. The buffer is implemented as a shift register. """
        for stats in self.stats_to_store:
            evaluate_stats = getattr(bunch, stats)

            # Handle the different statistics quantities, which can
            # either be methods (like mean(), ...) or simply attributes
            # (macroparticlenumber) of the bunch.
            try:
                self.buffer[stats][0] = evaluate_stats()
            except TypeError:
                self.buffer[stats][0] = evaluate_stats

            self.buffer[stats] = np.roll(self.buffer[stats], shift=-1, axis=0)

    def _write_buffer_to_file(self):
        """ Write buffer contents to the HDF5 file. The file is opened and
        closed each time the buffer is written to file to prevent from
        loss of data in case of a crash. """
        # Keep track of where to read from buffer and where to store
        # data in file.
        n_entries_in_buffer = min(self.i_steps+1, self.buffer_size)
        low_pos_in_buffer = self.buffer_size - n_entries_in_buffer
        low_pos_in_file = self.i_steps + 1 - n_entries_in_buffer
        up_pos_in_file = self.i_steps + 1


        # Try to write data to file. If file is not available, skip this
        # step and repeat it again after self.write_buffer_every. As
        # long as self.buffer_size is not exceeded, no data are lost.
        try:
            h5file = hp.File(self.filename + '.h5', 'a')
            h5group = h5file['Bunch']
            for stats in self.stats_to_store:
                h5group[stats][low_pos_in_file:up_pos_in_file] = \
                    self.buffer[stats][low_pos_in_buffer:]
            h5file.close()
        except:
            print ('Bunch monitor file is temporarily unavailable. \n')


class SliceMonitor(Monitor):
    """ Class to store bunch- and slice_set-specific data to a HDF5
    file. This monitor uses two buffers (shift registers) to reduce the
    number of writing operations to file. This also helps to avoid IO
    errors and loss of data when writing to a file that may become
    temporarily unavailable (e.g. if file is located on network) during
    the simulation. """

    def __init__(self, filename, n_steps, slicer, parameters_dict=None,
                 write_buffer_every=512, buffer_size=4096):
        """ Create an instance of a SliceMonitor class. Apart from
        initializing the HDF5 file, two buffers self.buffer_bunch and
        self.buffer_slice are prepared to buffer the bunch-specific and
        slice_set-specific data before writing them to file.

          filename:           Path and name of HDF5 file. Without file
                              extension.
          n_steps:            Number of entries to be reserved for each
                              of the quantities in self.stats_to_store.
          slicer:             Instance of the Slicer class containing
                              the configuration defining a slice_set.
          parameters_dict:    Metadata for HDF5 file containing main
                              simulation parameters.
          write_buffer_every: Number of steps after which buffer
                              contents are actually written to file.
          buffer_size:        Number of steps to be buffered. """
        self.bunch_stats_to_store = [
            'mean_x', 'mean_xp', 'mean_y', 'mean_yp', 'mean_z', 'mean_dp',
            'sigma_x', 'sigma_y', 'sigma_z', 'sigma_dp', 'epsn_x', 'epsn_y',
            'epsn_z', 'macroparticlenumber' ]
        self.slice_stats_to_store = [
            'mean_x', 'mean_xp', 'mean_y', 'mean_yp', 'mean_z', 'mean_dp',
            'sigma_x', 'sigma_y', 'sigma_z', 'sigma_dp', 'epsn_x', 'epsn_y',
            'epsn_z', 'n_macroparticles_per_slice' ]

        self.filename = filename
        self.n_steps = n_steps
        self.i_steps = 0
        self.slicer = slicer

        # Prepare buffers.
        self.buffer_size = buffer_size
        self.write_buffer_every = write_buffer_every
        self.buffer_bunch = {}
        self.buffer_slice = {}

        for stats in self.bunch_stats_to_store:
            self.buffer_bunch[stats] = np.zeros(self.buffer_size)
        for stats in self.slice_stats_to_store:
            self.buffer_slice[stats] = np.zeros((self.slicer.n_slices,
                                                 self.buffer_size))
        self._create_file_structure(parameters_dict)

    def dump(self, bunch):
        """ Evaluate the statistics like mean and standard deviation for
        the given slice_set and the bunch and write the data to the
        buffers and/or to the HDF5 file. The buffers are used to reduce
        the number of writing operations to file. This helps to avoid IO
        errors and loss of data when writing data to a file that may
        become temporarily unavailable (e.g. if file is on network)
        during the simulation. Buffer contents are written to file only
        every self.write_buffer_every steps. """
        self._write_data_to_buffer(bunch)
        if ((self.i_steps + 1) % self.write_buffer_every == 0 or
                (self.i_steps + 1) == self.n_steps):
            self._write_buffer_to_file()

        self.i_steps += 1

    def _create_file_structure(self, parameters_dict):
        """ Initialize HDF5 file and create its basic structure (groups
        and datasets). Two groups are created, one for slice_set-
        specific and one for bunch-specific data. One dataset for each
        of the quantities defined in self.bunch_stats_to_store and
        self.slice_stats_to_store resp. is generated. If specified by
        the user, write the contents of the parameters_dict as metadata
        (attributes) to the file. Maximum file compression is
        activated. """
        try:
            h5file = hp.File(self.filename + '.h5', 'w')
            if parameters_dict:
                for key in parameters_dict:
                    h5file.attrs[key] = parameters_dict[key]

            h5file.create_group('Bunch')
            h5file.create_group('Slices')
            h5group_bunch = h5file['Bunch']
            h5group_slice = h5file['Slices']

            for stats in self.bunch_stats_to_store:
                h5group_bunch.create_dataset(stats, shape=(self.n_steps,),
                    compression='gzip', compression_opts=9)
            for stats in self.slice_stats_to_store:
                h5group_slice.create_dataset(stats, shape=(self.slicer.n_slices,
                    self.n_steps), compression='gzip', compression_opts=9)
            h5file.close()
        except:
            print ('Creation of slice monitor file ' + self.filename +
                   'failed. \n')
            raise

    def _write_data_to_buffer(self, bunch):
        """ Store the data in the self.buffer dictionary before writing
        them to file. The buffer is implemented as a shift register. To
        find the slice_set-specific data, a slice_set, defined by the
        slicing configuration self.slicer must be requested from the
        bunch (instance of the Particles class). """
        slice_set = bunch.get_slices(self.slicer)

        # Handle the different statistics quantities, which can
        # either be methods (like mean(), ...) or simply attributes
        # (macroparticlenumber or n_macroparticles_per_slice) of the bunch
        # or slice_set resp.

        # bunch-specific data.
        for stats in self.bunch_stats_to_store:
            evaluate_stats_bunch = getattr(bunch, stats)
            try:
                self.buffer_bunch[stats][0] = evaluate_stats_bunch()
            except TypeError:
                self.buffer_bunch[stats][0] = evaluate_stats_bunch
            self.buffer_bunch[stats] = np.roll(self.buffer_bunch[stats],
                                               shift=-1, axis=0)

        # slice_set-specific data.
        for stats in self.slice_stats_to_store:
            evaluate_stats_slice = getattr(slice_set, stats)
            try:
                self.buffer_slice[stats][:,0] = evaluate_stats_slice(bunch)
            except TypeError:
                self.buffer_slice[stats][:,0] = evaluate_stats_slice
            self.buffer_slice[stats] = np.roll(self.buffer_slice[stats],
                                               shift=-1, axis=1)

    def _write_buffer_to_file(self):
        """ Write buffer contents to the HDF5 file. The file is opened
        and closed each time the buffer is written to file to prevent
        from loss of data in case of a crash. """
        # Keep track of where to read from buffers and where to store
        # data in file.
        n_entries_in_buffer = min(self.i_steps+1, self.buffer_size)
        low_pos_in_buffer = self.buffer_size - n_entries_in_buffer
        low_pos_in_file = self.i_steps + 1 - n_entries_in_buffer
        up_pos_in_file = self.i_steps + 1

        # Try to write data to file. If file is not available, skip this
        # step and repeat it again after self.write_buffer_every. As
        # long as self.buffer_size is not exceeded, no data are lost.
        try:
            h5file = hp.File(self.filename + '.h5', 'a')
            h5group_bunch = h5file['Bunch']
            h5group_slice = h5file['Slices']
            for stats in self.bunch_stats_to_store:
                h5group_bunch[stats][low_pos_in_file:up_pos_in_file] = \
                    self.buffer_bunch[stats][low_pos_in_buffer:]
            for stats in self.slice_stats_to_store:
                h5group_slice[stats][:,low_pos_in_file:up_pos_in_file] = \
                    self.buffer_slice[stats][:,low_pos_in_buffer:]
            h5file.close()
        except:
            print ('Slice monitor file is temporarily unavailable. \n')


class ParticleMonitor(Monitor):
    """ Class to store particle-specific data to a HDF5 file, i.e. the
    coordinates and conjugate momenta as well as the id of individual
    macroparticles of a bunch. """

    def __init__(self, filename, stride=1, parameters_dict=None):
        """ Create an instance of a ParticleMonitor class. The HDF5 file
        is initialized, and if specified, the parameters_dict is written
        to file.

          filename:        Path and name of HDF5 file. Without file
                           extension.
          stride:          Only store data of macroparticles for which
                           id % stride == 0.
          parameters_dict: Metadata for HDF5 file containing main
                           simulation parameters. """
        self.quantities_to_store = [ 'x', 'xp', 'y', 'yp', 'z', 'dp', 'id' ]
        self.filename = filename
        self.stride = stride
        self.i_steps = 0

        self._create_file_structure(parameters_dict)

    def dump(self, bunch):
        """ Write particle data to file. See docstring of method
        self._write_data_to_file . """
        self._write_data_to_file(bunch)
        self.i_steps += 1

    def _create_file_structure(self, parameters_dict):
        """ Initialize HDF5 file. If specified by the user, write the
        contents of the parameters_dict as metadata (attributes)
        to the file. Maximum file compression is activated. """
        try:
            h5file = hp.File(self.filename + '.h5part', 'w')
            if parameters_dict:
                for key in parameters_dict:
                    h5file.attrs[key] = parameters_dict[key]
            h5file.close()
        except:
            print ('Creation of particle monitor file ' + self.filename +
                   'failed. \n')
            raise

    def _write_data_to_file(self, bunch):
        """ Write macroparticle data (x, xp, y, yp, z, dp, id) of a
        selection of particles to the HDF5 file. The file is opened and
        closed every time to prevent from loss of data in case of a
        crash.
        For each simulation step, a new group with name 'Step#..' is
        created. It contains one dataset for each of the quantities
        given in self.quantities_to_store. """
        h5file = hp.File(self.filename + '.h5part', 'a')
        h5group = h5file.create_group('Step#' + str(self.i_steps))
        dims = (bunch.macroparticlenumber // self.stride,)

        # resorting_indices = np.argsort(bunch.id)[::self.stride]
        for quant in self.quantities_to_store:
            quant_values = getattr(bunch, quant)
            h5group.create_dataset(quant, shape=dims, compression='gzip',
                compression_opts=9, dtype=quant_values.dtype)
            h5group[quant][:] = quant_values[::self.stride]
        h5file.close()
