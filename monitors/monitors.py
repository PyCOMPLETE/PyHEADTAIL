"""
@author Kevin Li, Michael Schenk, Stefan Hegglin
@date 11. February 2014
@brief Implementation of monitors to store bunch-, slice- or particle-
       specific data to a HDF5 file.
@copyright CERN
"""
from __future__ import division

from mpi4py import MPI
import h5py as hp
import numpy as np
import sys

from abc import ABCMeta, abstractmethod
from . import Printing


class Monitor(Printing):
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
    def __init__(self, filename, n_steps, parameters_dict=None,
                 write_buffer_every=512, buffer_size=4096,
                 mpi=False, filling_scheme=None, *args, **kwargs):

        stats_to_store = [
            'mean_x', 'mean_xp', 'mean_y', 'mean_yp', 'mean_z', 'mean_dp',
            'sigma_x', 'sigma_y', 'sigma_z', 'sigma_dp', 'epsn_x', 'epsn_y',
            'epsn_z', 'macroparticlenumber' ]
        self.stats_to_store = kwargs.pop('stats_to_store', stats_to_store)
        self.filename = filename
        self.n_steps = n_steps
        self.i_steps = 0
        self.mpi = mpi
        self.filling_scheme = filling_scheme
        self.local_bunch_ids = []

        self.buffer = None
        self.buffer_size = buffer_size
        self.write_buffer_every = write_buffer_every

        self._create_file_structure(parameters_dict)


    def _init_buffer(self):
        self.buffer = {}
        for bid in self.local_bunch_ids:
            self.buffer[bid] = {}
            for stats in self.stats_to_store:
                self.buffer[bid][stats] = np.zeros(self.buffer_size)


    def dump(self, bunches):
        """ Evaluate the statistics like mean and standard deviation for
        the given bunch and write the data to the HDF5 file. """

        bunch_list = bunches.split_to_views()
        if self.buffer is None:
            self.local_bunch_ids = [ b.bunch_id[0] for b in bunch_list ]
            self._init_buffer()

        self._write_data_to_buffer(bunches)
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
        Maximum file compression is activated only if not using MPI. """
        try:
            if self.mpi:
                h5file = hp.File(self.filename + '.h5', 'w', driver='mpio',
                                 comm=MPI.COMM_WORLD)
                kwargs_gr = {}
            else:
                h5file = hp.File(self.filename + '.h5', 'w')
                kwargs_gr = {'compression': 'gzip', 'compression_opts': 9 }

            if parameters_dict:
                for key in parameters_dict:
                    h5file.attrs[key] = parameters_dict[key]

            # INTERMEDIATE HACK - NOT ENTIRELY CONSISTENT. Should be fixed once
            # dependency to filling scheme is removed and taken during dump.
            if self.filling_scheme is not None:
                h5group = h5file.create_group('Bunches')
                for bid in self.filling_scheme:
                    gr = h5group.create_group(repr(bid))
                    for stats in sorted(self.stats_to_store):
                        gr.create_dataset(stats, shape=(self.n_steps,), **kwargs_gr)
            else:
                h5group = h5file.create_group('Bunch')
                gr = h5group
                for stats in sorted(self.stats_to_store):
                    gr.create_dataset(stats, shape=(self.n_steps,), **kwargs_gr)
            h5file.close()
        except:
            self.prints('Creation of bunch monitor file ' + self.filename +
                   'failed. \n')
            raise


    def _write_data_to_buffer(self, bunches):
        """ Store the data in the self.buffer dictionary before writing
        them to file. The buffer is implemented as a shift register. """

        bunch_list = bunches.split_to_views()
        for b in bunch_list:
            bid = b.bunch_id[0]
            for stats in self.stats_to_store:
                evaluate_stats = getattr(b, stats)

                # Handle the different statistics quantities, which can
                # either be methods (like mean(), ...) or simply attributes
                # (macroparticlenumber) of the bunch.
                write_pos = self.i_steps % self.buffer_size
                # Is needed just because of Particles.macroparticles -
                # this we want to keep as it may be used by other
                # classes
                try:
                    self.buffer[bid][stats][write_pos] = evaluate_stats()
                except TypeError:
                    self.buffer[bid][stats][write_pos] = evaluate_stats


    def _write_buffer_to_file(self):
        """ Write buffer contents to the HDF5 file. The file is opened and
        closed each time the buffer is written to file to prevent from
        loss of data in case of a crash.
        buffer_tmp is an extra buffer which is always on the CPU. If
        self.buffer is on the GPU, copy the data to buffer_tmp and write
        the result to the file."""

        buffer_tmp = {} # always on CPU
        for bid in self.local_bunch_ids:
            buffer_tmp[bid] = {}
        shift = - (self.i_steps + 1 % self.buffer_size)
        for bid in self.local_bunch_ids:
            for stats in self.stats_to_store:
                try:
                    buffer_tmp[bid][stats] = np.roll(self.buffer[bid][stats].get(),
                            shift=shift, axis=0)
                except:
                    buffer_tmp[bid][stats] = np.roll(self.buffer[bid][stats].copy(),
                            shift=shift, axis=0)
        n_entries_in_buffer = min(self.i_steps+1, self.buffer_size)
        low_pos_in_buffer = self.buffer_size - n_entries_in_buffer
        low_pos_in_file = self.i_steps + 1 - n_entries_in_buffer
        up_pos_in_file = self.i_steps + 1

        # Try to write data to file. If file is not available, skip this
        # step and repeat it again after self.write_buffer_every. As
        # long as self.buffer_size is not exceeded, no data are lost.
        try:
            if self.mpi:
                h5file = hp.File(self.filename + '.h5', 'a', driver='mpio',
                                 comm=MPI.COMM_WORLD)
            else:
                h5file = hp.File(self.filename + '.h5', 'a')

            # Note that in this case, local_bunch_ids should have only
            # one entry.
            for bid in self.local_bunch_ids:
                if self.filling_scheme is not None:
                    h5group = h5file['Bunches'][repr(bid)]
                else:
                    h5group = h5file['Bunch']
                for stats in self.stats_to_store:
                    h5group[stats][low_pos_in_file:up_pos_in_file] = \
                        buffer_tmp[bid][stats][low_pos_in_buffer:]
            h5file.close()
        except IOError:
            self.warns('Bunch monitor file is temporarily unavailable. \n')


class SliceMonitor(Monitor):
    """ Class to store bunch- and slice_set-specific data to a HDF5
    file. This monitor uses two buffers (shift registers) to reduce the
    number of writing operations to file. This also helps to avoid IO
    errors and loss of data when writing to a file that may become
    temporarily unavailable (e.g. if file is located on network) during
    the simulation. """

    def __init__(self, filename, n_steps, slicer, parameters_dict=None,
                 write_buffer_every=512, buffer_size=4096,
                 *args, **kwargs):
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
          buffer_size:        Number of steps to be buffered.

          optionally pass a list called bunch_stats_to_store or
          slice_stats_to_store which specifie
          which members/methods of the bunch will be called/stored.
        """
        bunch_stats_to_store = [
            'mean_x', 'mean_xp', 'mean_y', 'mean_yp', 'mean_z', 'mean_dp',
            'sigma_x', 'sigma_y', 'sigma_z', 'sigma_dp', 'epsn_x', 'epsn_y',
            'epsn_z', 'macroparticlenumber' ]
        slice_stats_to_store = [
            'mean_x', 'mean_xp', 'mean_y', 'mean_yp', 'mean_z', 'mean_dp',
            'sigma_x', 'sigma_y', 'sigma_z', 'sigma_dp', 'epsn_x', 'epsn_y',
            'epsn_z', 'n_macroparticles_per_slice' ]
        self.bunch_stats_to_store = kwargs.pop('bunch_stats_to_store',
                bunch_stats_to_store)
        self.slice_stats_to_store = kwargs.pop('slice_stats_to_store',
                slice_stats_to_store)

        self.filename = filename
        self.n_steps = n_steps
        self.i_steps = 0
        self.slicer = slicer

        # Prepare buffers.
        self.buffer_size = buffer_size
        self.write_buffer_every = write_buffer_every
        self.buffer_bunch = None
        self.buffer_slice = None
        self._create_file_structure(parameters_dict)

    def _init_buffer(self):
        self.buffer_bunch = {}
        self.buffer_slice = {}
        for stats in self.bunch_stats_to_store:
            self.buffer_bunch[stats] = np.zeros(self.buffer_size)
        for stats in self.slice_stats_to_store:
            self.buffer_slice[stats] = np.zeros((self.slicer.n_slices,
                                                 self.buffer_size))

    def dump(self, bunch):
        """ Evaluate the statistics like mean and standard deviation for
        the given slice_set and the bunch and write the data to the
        buffers and/or to the HDF5 file. The buffers are used to reduce
        the number of writing operations to file. This helps to avoid IO
        errors and loss of data when writing data to a file that may
        become temporarily unavailable (e.g. if file is on network)
        during the simulation. Buffer contents are written to file only
        every self.write_buffer_every steps. """
        if self.buffer_bunch is None:
            self._init_buffer()
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
            self.prints('Creation of slice monitor file ' + self.filename +
                        'failed. \n')
            raise

    def _write_data_to_buffer(self, bunch):
        """ Store the data in the self.buffer dictionary before writing
        them to file. The buffer is implemented as a shift register. To
        find the slice_set-specific data, a slice_set, defined by the
        slicing configuration self.slicer must be requested from the
        bunch (instance of the Particles class), including all the
        statistics that are to be saved. """
        
        z_delay = bunch.mean_z()
        bunch.z -= z_delay
        slice_set = bunch.get_slices(self.slicer, statistics=True)
        bunch.z += z_delay
        

        # Handle the different statistics quantities, which can
        # either be methods (like mean(), ...) or simply attributes
        # (macroparticlenumber or n_macroparticles_per_slice) of the bunch
        # or slice_set resp.

        # bunch-specific data.
        write_pos = self.i_steps % self.buffer_size
        for stats in self.bunch_stats_to_store:
            evaluate_stats_bunch = getattr(bunch, stats)
            try:
                self.buffer_bunch[stats][write_pos] = evaluate_stats_bunch()
            except TypeError:
                self.buffer_bunch[stats][write_pos] = evaluate_stats_bunch

        # slice_set-specific data.
        for stats in self.slice_stats_to_store:
            if stats == 'mean_z':
                self.buffer_slice[stats][:, write_pos] = getattr(slice_set, stats) + z_delay
            else:
                self.buffer_slice[stats][:, write_pos] = getattr(slice_set, stats)

    def _write_buffer_to_file(self):
        """ Write buffer contents to the HDF5 file. The file is opened
        and closed each time the buffer is written to file to prevent
        from loss of data in case of a crash. """
        buffer_tmp_bunch = {} # always on CPU
        buffer_tmp_slice = {}
        shift = - (self.i_steps + 1 % self.buffer_size)
        for stats in self.bunch_stats_to_store:
            try:
                buffer_tmp_bunch[stats] = np.roll(self.buffer_bunch[stats].get(),
                        shift=shift, axis=0)
            except:
                buffer_tmp_bunch[stats] = np.roll(self.buffer_bunch[stats].copy(),
                        shift=shift, axis=0)
        for stats in self.slice_stats_to_store:
            try:
                buffer_tmp_slice[stats] = np.roll(self.buffer_slice[stats].get(),
                        shift=shift, axis=1)
            except:
                buffer_tmp_slice[stats] = np.roll(self.buffer_slice[stats].copy(),
                        shift=shift, axis=1)

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
                    buffer_tmp_bunch[stats][low_pos_in_buffer:]
                    #self.buffer_bunch[stats][low_pos_in_buffer:]
            for stats in self.slice_stats_to_store:
                h5group_slice[stats][:,low_pos_in_file:up_pos_in_file] = \
                    buffer_tmp_slice[stats][:,low_pos_in_buffer:]
                    #self.buffer_slice[stats][:,low_pos_in_buffer:]
            h5file.close()
        except IOError:
            self.warns('Slice monitor file is temporarily unavailable. \n')


class ParticleMonitor(Monitor):
    """ Class to store particle-specific data to a HDF5 file, i.e. the
    coordinates and conjugate momenta as well as the id of individual
    macroparticles of a bunch. """

    def __init__(self, filename, stride=1, parameters_dict=None,
                 *args, **kwargs):
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

    def dump(self, bunch, arrays_dict=None):
        """ Write particle data to file. See docstring of method
        self._write_data_to_file . """
        self._write_data_to_file(bunch, arrays_dict)
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
            self.prints('Creation of particle monitor file ' +
                        self.filename + 'failed. \n')
            raise

    def _write_data_to_file(self, bunch, arrays_dict):
        """ Write macroparticle data (x, xp, y, yp, z, dp, id) of a
        selection of particles to the HDF5 file. Optionally, data in
        additional_quantities can also be added if provided in the
        constructor. The file is opened and closed every time to prevent
        from loss of data in case of a crash.
        For each simulation step, a new group with name 'Step#..' is
        created. It contains one dataset for each of the quantities
        given in self.quantities_to_store. """
        h5file = hp.File(self.filename + '.h5part', 'a')
        h5group = h5file.create_group('Step#' + str(self.i_steps))
        dims = (bunch.macroparticlenumber // self.stride,)
        dims = bunch.get_coords_n_momenta_dict().values()[0][::self.stride].shape # more robust implementation

        # resorting_indices = np.argsort(bunch.id)[::self.stride]
        all_quantities = {}
        for quant in self.quantities_to_store:
            quant_values = getattr(bunch, quant)
            all_quantities[quant] = quant_values

        if arrays_dict is not None:
            all_quantities.update(arrays_dict)

        for quant in all_quantities.keys():
            quant_values = all_quantities[quant]
            h5group.create_dataset(quant, shape=dims, compression='gzip',
                compression_opts=9, dtype=quant_values.dtype)
            h5group[quant][:] = quant_values[::self.stride]

        h5file.close()
