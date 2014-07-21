'''
@author Kevin Li
@date 20/06/2014
@copyright CERN
'''
from __future__ import division


import numpy as np
from scipy.special import k0
from scipy.constants import c, e
import pylab as plt


class TransferFunction(object):

    def __init__(self, slices): pass

    def one_pole_roll_off(self, frequency):

        def f(z):
            return np.exp(-2*np.pi*frequency * z/c)
        return f

    def convert(self): pass

    def convert_as_one_pole(self, x): pass

    def convert_as_table(self): pass


def one_pole(fr, xmax=0):
    def y(x):
        ix = np.where(x > xmax)
        # y = 0.113458834 * np.exp(2*np.pi*x*fr)
        y = np.exp(2*np.pi*x*fr)
        y[ix] = 0
        return y
    return y


def one_pole_symmetric(fr, xmax=0):
    def y(x):
        ix = np.where(x > xmax)
        y = 0.113458834 * np.exp(2*np.pi*x*fr)
        y[ix] = 0.113458834 * np.exp(-2*np.pi*x[ix]*fr)
        return y
    return y


def one_pole_symmetric_gerd(fr, xmax=0):
    def y(x):
        ix = np.where(x > xmax)
        y = k0(-2*np.pi*x*fr) #/ np.sum(k0(2*np.pi*x*fr))
        y[ix] = k0(2*np.pi*x[ix]*fr) #/ np.sum(k0(-2*np.pi*x*fr))
        return y
    return y


class Pickup(object):
    '''
    '''

    def __init__(self, slices, plane='y'):

        self.slices = slices
        self.plane = plane

    def track(self, beam):

        # slices = self.slices
        # slices.update_slices(beam)
        self.slices.compute_statistics(beam)

        if self.plane == 'x':
            self.y = np.copy(self.slices.mean_x)
            self.yin = self.y
            # self.yin = np.dot(self.transfer_function, self.x)
        elif self.plane == 'y':
            self.y = np.copy(self.slices.mean_y)
            self.yin = self.y
            # self.yin = np.dot(self.transfer_function, self.y)


class Kicker(object):
    '''
    '''

    def __init__(self, pickup, plane='y', transfer_function=None, filter_fir=[0], filter_iir=[1], closedloop=False, gain=0, slices=None):

        # TODO: better binding of self.slices
        self.pickup = pickup
        self.plane = plane
        self.transfer_function = transfer_function
        self.filter_fir = filter_fir
        self.filter_iir = filter_iir
        self.gain = gain

        self._prepare_registers(len(self.filter_fir), len(self.filter_iir), pickup.slices.n_slices)

        if slices:
            self.slices = slices
            if self.transfer_function == None:
                self.transfer_function = np.eye(slices.n_slices)

            # self.xout = np.zeros(self.transfer_function.shape[0])
            # self.yout = np.zeros(self.transfer_function.shape[0])

        # else:
        #     if self.transfer_function:
        #         self.xout = np.zeros(self.transfer_function.shape[0])
        #         self.yout = np.zeros(self.transfer_function.shape[0])

    def controller_fir(self, beam):

        # Fill shift register
        self.register_forward[:, 1:] = self.register_forward[:, :-1]
        self.register_forward[:, 0] = self.pickup.yin

        self.yout = np.dot(self.register_forward, self.filter_fir)

        # with the one zero implicit!
        # for (j=0; j<ntabsforward; j++)
        # {
        #     yout += FeedbackRegisterForward[j+1] * FilterControllerForward[j];
        # }

    def controller_iir(self, beam):

        tmpyout = np.dot(self.register_reverse, self.filter_iir)
        self.yout[:] -= tmpyout
        # for i in xrange(len(self.filter_iir)):
        #     self.xout[:] -= self.filter_iir[i] * self.register_forward_x[i, :]
        #     self.yout[:] -= self.filter_iir[i] * self.register_forward_y[i, :]

        # Fill shift register
        self.register_reverse[:, 1:] = self.register_reverse[:, :-1]
        self.register_reverse[:, 0] = self.yout

        # No delay!
        # for (j=0; j<ntabsreverse; j++)
        # {
        #     yout -= FeedbackRegisterReverse[j] * FilterControllerReverse[j];
        # }

    def kicker(self, beam):

        # for (i=0; i<n_channels; i++)
        #     vout[i] += AmplifierNoise[i][turnnumber];

        # // Saturation
        # if (saturation_out)
        # {
        #     for (i=0; i<n_channels; i++)
        #     {
        #         if (vout[i] < saturation_out_min)
        #             vout[i] = saturation_out_min;
        #         else if (vout[i] > saturation_out_max)
        #             vout[i] = saturation_out_max;
        #     }
        # }

        self.vout = self.gain * self.yout
        # Convolution here
        self.kick = np.dot(self.transfer_function, self.vout)

        p_absolute = (1+beam.dp) * beam.p0
        kick = self.kick[self.slices.slice_index_of_particle]
        if self.plane == 'x':
            beam.xp += kick * e/p_absolute
        elif self.plane == 'y':
            beam.yp += kick * e/p_absolute
        # for i in xrange(self.slices.n_slices):
        #     ix = np.s_[self.slices.z_index[i]:self.slices.z_index[i + 1]]

        #     beam.xp[ix] += self.kick_x[i] * e/p_absolute[ix]
        #     beam.yp[ix] += self.kick_y[i] * e/p_absolute[ix]

    def track(self, beam):

        self._check_slices(beam)

        if self.plane == 'x':
            self.x = np.copy(self.slices.mean_x)
        elif self.plane == 'y':
            self.y = np.copy(self.slices.mean_y)

        self.controller_fir(beam)
        self.controller_iir(beam)
        self.kicker(beam)

        # TODO: this can be modeled with gain=0; kind of redundant...
        # if self.closedloop:
        #     self.vout_x = self.gain * self.xout
        #     self.vout_y = self.gain * self.yout
        # else:
        #     self.vout_x = 0
        #     self.vout_y = 0

        # self.x = self.transfer_function * self.slices.mean_x
        # self.y = self.transfer_function * self.slices.mean_y
        # self.xp = self.pickup.x
        # self.yp = self.pickup.y

    def _prepare_registers(self, n_taps_forward, n_taps_reverse, n_slices):

        self.register_forward = np.zeros((n_slices, n_taps_forward))
        self.register_reverse = np.zeros((n_slices, n_taps_reverse))

    def _check_slices(self, beam):

        try:
            slices = self.slices
        except AttributeError:
            slices = beam.slices

            self._prepare_registers(len(self.filter_fir), len(self.filter_iir), slices.n_slices)

            if not self.transfer_function:
                self.transfer_function = np.eye(slices.n_slices)

            # self.xout = np.zeros(self.transfer_function.shape[0])
            # self.yout = np.zeros(self.transfer_function.shape[0])

            self.slices = slices

        slices.update_slices(beam)
        slices.compute_statistics(beam)

# // Functions
# void Feedback::init(CFG_IO &cfgfile)
# {
#     unsigned int i, j;
#     CFG_IO::expression exp = cfgfile.cfg_expression;

#     for (i=0; i<exp.attribute.size(); i++)
#     {
#         if (exp.attribute[i] == "gain")
#             gain = toNumber<double>(exp.value[i]);
#         else if (exp.attribute[i] == "ipickup")
#             i_pickup = toNumber<unsigned int>(exp.value[i]);
#         else if (exp.attribute[i] == "ikicker")
#             i_kicker = toNumber<unsigned int>(exp.value[i]);
#         else if (exp.attribute[i] == "loopmode")
#             loopmode = exp.value[i];
#         else if (exp.attribute[i] == "signalmode")
#             signalmode = exp.value[i];
#         else if (exp.attribute[i] == "noisein")
#             noise_in = toNumber<unsigned int>(exp.value[i]);
#         else if (exp.attribute[i] == "noiseout")
#             noise_out = toNumber<unsigned int>(exp.value[i]);
#         else if (exp.attribute[i] == "satin")
#             saturation_in = toNumber<unsigned int>(exp.value[i]);
#         else if (exp.attribute[i] == "satinmin")
#             saturation_in_min = toNumber<double>(exp.value[i]);
#         else if (exp.attribute[i] == "satinmax")
#             saturation_in_max = toNumber<double>(exp.value[i]);
#         else if (exp.attribute[i] == "satout")
#             saturation_out = toNumber<unsigned int>(exp.value[i]);
#         else if (exp.attribute[i] == "satoutmin")
#             saturation_out_min = toNumber<double>(exp.value[i]);
#         else if (exp.attribute[i] == "satoutmax")
#             saturation_out_max = toNumber<double>(exp.value[i]);

#         else if (exp.attribute[i] == "fcffile")
#             fcffile = exp.value[i];
#         else if (exp.attribute[i] == "fcrfile")
#             fcrfile = exp.value[i];
#         else if (exp.attribute[i] == "fkfile")
#             fkfile = exp.value[i];
#         else if (exp.attribute[i] == "frfile")
#             frfile = exp.value[i];
#         else if (exp.attribute[i] == "fnafile")
#             fnafile = exp.value[i];
#         else if (exp.attribute[i] == "fnrfile")
#             fnrfile = exp.value[i];
#         else if (exp.attribute[i] == "fnafile")
#             fnafile = exp.value[i];

#         else
#             std::cerr << "*** Unused attribute \"" << exp.attribute[i] << std::endl;
#     }

#     // Filter receiver: n_channels x n_slices
#     /////////////////////////////////////////
#     fbkfile.open(frfile.c_str());
#     if (fbkfile.is_open())
#     {
#         shape = import(fbkfile, FilterReceiver);
#         fbkfile.close();
#     }
#     else
#     {
#         std::cerr << "*** ERROR: file " << frfile << " not found!" << std::endl;
#         exit(-1);
#     }
#     n_channels = shape[0];
#     n_slices = shape[1];

#     // Noise receiver: n_channels x n_turns
#     ////////////////////////////////////////
#     fbkfile.open(fnrfile.c_str());
#     if (fbkfile.is_open())
#     {
#         shape = import(fbkfile, ReceiverNoise);
#         fbkfile.close();
#     }
#     else
#     {
#         std::cerr << "*** ERROR: file " << fnrfile << " not found!" << std::endl;
#         exit(-1);
#     }
#     if (noise_in)
#         n_turns = shape[1];
# //    {
# //        if (static_cast<unsigned int>(nturns) > shape[1])
# //            n_turns = shape[1];
# //        else
# //            n_turns = nturns;
# //    }

#     // Filter kicker: n_slices x n_channels
#     ///////////////////////////////////////
#     fbkfile.open(fkfile.c_str());
#     if (fbkfile.is_open())
#     {
#         shape = import(fbkfile, FilterKicker);
#         fbkfile.close();
#     }
#     else
#     {
#         std::cerr << "*** ERROR: file " << fkfile << " not found!" << std::endl;
#         exit(-1);
#     }

#     // Noise amplifier: n_channels x n_turns
#     //////////////////
#     fbkfile.open(fnafile.c_str());
#     if (fbkfile.is_open())
#     {
#         shape = import(fbkfile, AmplifierNoise);
#         fbkfile.close();
#     }
#     else
#     {
#         std::cerr << "*** ERROR: file " << fnafile << " not found!" << std::endl;
#         exit(-1);
#     }

#     // Filter controller forward ntabsforward x 1
#     ////////////////////////////
#     fbkfile.open(fcffile.c_str());
#     if (fbkfile.is_open())
#     {
#         shape = import(fbkfile, FilterControllerForward);
#         fbkfile.close();
#     }
#     else
#     {
#         std::cerr << "*** ERROR: file " << fcffile << " not found!" << std::endl;
#         exit(-1);
#     }
#     ntabsforward = shape[0];

#     // Filter controller reverse: ntabsreverse x 1
#     ////////////////////////////
#     fbkfile.open(fcrfile.c_str());
#     if (fbkfile.is_open())
#     {
#         shape = import(fbkfile, FilterControllerReverse);
#         fbkfile.close();
#     }
#     else
#     {
#         std::cerr << "*** ERROR: file " << fcrfile << " not found!" << std::endl;
#     }
#     ntabsreverse = shape[0];


#     // Reset external variables
# //         NBIN = n_slices;
# //         nturns = n_turns;


#     // Allocate memory
#     // Reallocate particular matrix to add one more register coefficient
# //         FilterControllerForward = new double*[ntabsforward+1];
# //         for (int i=0; i<ntabsforward+1; i++)
# //             FilterControllerForward[i] = new double[1];

#     FeedbackRegisterForward = new double*[n_channels];
#     for (i=0; i<n_channels; i++)
#         FeedbackRegisterForward[i] = new double[ntabsforward+1];

#     FeedbackRegisterReverse = new double*[n_channels];
#     for (i=0; i<n_channels; i++)
#         FeedbackRegisterReverse[i] = new double[ntabsreverse];

#     // Initialise all to 0
#     for (i=0; i<n_channels; i++)
#     {
#         for (j=0; j<ntabsforward+1; j++)
#             FeedbackRegisterForward[i][j] = 0;
#         for (j=0; j<ntabsreverse; j++)
#             FeedbackRegisterReverse[i][j] = 0;
#     }

#     // Allocate memory
#     xslice.resize(n_slices);
#     yslice.resize(n_slices);
#     kick_y.resize(n_slices);
#     yin.resize(n_channels);
#     yout.resize(n_channels);
#     vout.resize(n_channels);
# }


# ////////////////////////////////////////////////////////////////////////////////
# // (C) 2012 Kevin Li, CERN; Mauro Pivi, Claudio Rivetta, SLAC                 //
# // HeadTail                                                                   //
# // Class Implementation: Feedback                                             //
# ////////////////////////////////////////////////////////////////////////////////


# /*******************************************************************************
#  * INCLUDES
#  * ****************************************************************************/
# // #include "Beam.h"
# #include "CFG_IO.h"
# #include "Constants.h"
# #include "Feedback.h"
# #include "Setup.h"
# #include "Statistics.h"


# // Constructor
# Feedback::Feedback(const char *h5filename, CFG_IO &cfgfile)
# : h5filename(h5filename)
# {
#     std::cout << "\n--> Initializing Feedback." << std::endl;
#     feedbackflag = cfgfile.readlines("Feedback");
#     if (feedbackflag)
#     {
#         init(cfgfile);
#         hdf5init();
#     }
# }

# void Feedback::set_kicky(long turnnumber)
# {
#     unsigned int i, j;

#     /***************************************************************************
#     * RECEIVER
#     * ************************************************************************/
#     // Pickup
#     for (i=0; i<n_channels; i++)
#     {
#         yin[i] = 0; // rather vin...
#         for (j=0; j<n_slices; j++)
#         {
#             yin[i] += FilterReceiver[i][j]*yslice[j];
#         }
#     }

#     // Pickup noise
#     if (noise_in && turnnumber < n_turns)
#     {
#         for (i=0; i<n_channels; i++)
#             yin[i] += ReceiverNoise[i][turnnumber];
#     }

#     // Saturation
#     if (saturation_in)
#     {
#         for (i=0; i<n_channels; i++)
#         {
#             if (yin[i] < saturation_in_min)
#                 yin[i] = saturation_in_min;
#             else if (yin[i] > saturation_in_max)
#                 yin[i] = saturation_in_max;
#         }
#     }


# void Feedback::hdf5init()
# {
#     // Create H5 filename
#     std::string tmpfilename(mainfilename);
#     size_t pos1 = tmpfilename.find_last_of(".");
#     h5filename = tmpfilename.replace(tmpfilename.begin() + pos1,
#                                      tmpfilename.end(), ".fbk.h5").c_str();

#     // Create H5 file
#     h5file = H5Fcreate(h5filename, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

#     h5group = H5Gcreate(h5file, "Signals", 0);

#     dims[0] = nturns;
#     dims[1] = n_channels;
#     h5s2ddata = H5Screate_simple(2, dims, NULL);
#     h5dset = H5Dcreate(h5group, "yslice", H5T_NATIVE_DOUBLE, h5s2ddata, H5P_DEFAULT);
#     h5dset = H5Dcreate(h5group, "yin", H5T_NATIVE_DOUBLE, h5s2ddata, H5P_DEFAULT);
#     h5dset = H5Dcreate(h5group, "yout", H5T_NATIVE_DOUBLE, h5s2ddata, H5P_DEFAULT);
#     h5dset = H5Dcreate(h5group, "vout", H5T_NATIVE_DOUBLE, h5s2ddata, H5P_DEFAULT);
#     h5dset = H5Dcreate(h5group, "kicky", H5T_NATIVE_DOUBLE, h5s2ddata, H5P_DEFAULT);

#     h5group = H5Gcreate(h5file, "Registers", 0);

#     dims[0] = n_channels;
#     dims[1] = ntabsforward+1;
#     dims[2] = nturns;
#     h5s3ddata = H5Screate_simple(3, dims, NULL);
#     h5dset = H5Dcreate(h5group, "RegisterForward", H5T_NATIVE_DOUBLE, h5s3ddata, H5P_DEFAULT);
#     dims[0] = n_channels;
#     dims[1] = ntabsreverse;
#     dims[2] = nturns;
#     h5s3ddata = H5Screate_simple(3, dims, NULL);
#     h5dset = H5Dcreate(h5group, "RegisterReverse", H5T_NATIVE_DOUBLE, h5s3ddata, H5P_DEFAULT);
# }

# void Feedback::hdf5dump(long turnnumber)
# {
#     // Signal vectors
#     if (!yin.empty() && !yout.empty() && !vout.empty())
#     {
#         h5group = H5Gopen(h5file, "Signals");
#         dims[0] = n_channels;
#         h5smemory = H5Screate_simple(1, dims, NULL);

#         count[0] = 1;
#         count[1] = n_channels;
#         offset[0] = turnnumber;
#         offset[1] = 0;
#         status = H5Sselect_hyperslab(h5s2ddata, H5S_SELECT_SET, offset, NULL, count, NULL);
#         h5dset = H5Dopen(h5group, "yslice");
#         status = H5Dwrite(h5dset, H5T_NATIVE_DOUBLE, h5smemory, h5s2ddata, H5P_DEFAULT, &yslice[0]);
#         h5dset = H5Dopen(h5group, "yin");
#         status = H5Dwrite(h5dset, H5T_NATIVE_DOUBLE, h5smemory, h5s2ddata, H5P_DEFAULT, &yin[0]);
#         h5dset = H5Dopen(h5group, "yout");
#         status = H5Dwrite(h5dset, H5T_NATIVE_DOUBLE, h5smemory, h5s2ddata, H5P_DEFAULT, &yout[0]);
#         h5dset = H5Dopen(h5group, "vout");
#         status = H5Dwrite(h5dset, H5T_NATIVE_DOUBLE, h5smemory, h5s2ddata, H5P_DEFAULT, &vout[0]);
#         h5dset = H5Dopen(h5group, "kicky");
#         status = H5Dwrite(h5dset, H5T_NATIVE_DOUBLE, h5smemory, h5s2ddata, H5P_DEFAULT, &kick_y[0]);
#     }

#     // Register matrices
#     // Get contiguous data!
#     data.resize(n_channels * (ntabsforward + 1));
#     for (unsigned int i=0; i<n_channels; i++)
#         for (unsigned int j=0; j<ntabsforward+1; j++)
#             data[i * (ntabsforward+1) + j] = FeedbackRegisterForward[i][j];

#     h5group = H5Gopen(h5file, "Registers");
#     h5dset = H5Dopen(h5group, "RegisterForward");
#     h5s3ddata = H5Dget_space(h5dset);
#     dims[0] = n_channels;
#     dims[1] = ntabsforward + 1;
#     h5smemory = H5Screate_simple(2, dims, NULL);

#     count[0] = dims[0];
#     count[1] = dims[1];
#     count[2] = 1;
#     offset[0] = 0;
#     offset[1] = 0;
#     offset[2] = turnnumber;
#     status = H5Sselect_hyperslab(h5s3ddata, H5S_SELECT_SET, offset, NULL, count, NULL);
#     status = H5Dwrite(h5dset, H5T_NATIVE_DOUBLE, h5smemory, h5s3ddata, H5P_DEFAULT, &data[0]);

#     // Get contiguous data!
#     data.reserve(n_channels * ntabsreverse);
#     for (unsigned int i=0; i<n_channels; i++)
#         for (unsigned int j=0; j<ntabsreverse; j++)
#             data[i*ntabsforward + j] = FeedbackRegisterReverse[i][j];

#     h5dset = H5Dopen(h5group, "RegisterReverse");
#     h5s3ddata = H5Dget_space(h5dset);
#     dims[0] = n_channels;
#     dims[1] = ntabsreverse;
#     h5smemory = H5Screate_simple(2, dims, NULL);

#     count[0] = dims[0];
#     count[1] = dims[1];
#     count[2] = 1;
#     offset[0] = 0;
#     offset[1] = 0;
#     offset[2] = turnnumber;
#     status = H5Sselect_hyperslab(h5s3ddata, H5S_SELECT_SET, offset, NULL, count, NULL);
#     status = H5Dwrite(h5dset, H5T_NATIVE_DOUBLE, h5smemory, h5s3ddata, H5P_DEFAULT, &data[0]);
# }
