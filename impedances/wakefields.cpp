#include "wakefields.h"
#include <boost/python.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>

using namespace boost::python;


Wakefields::Wakefields(double Rs_r, double fr, double Q_r,
                       double Rs_z, double fz, double Q_z)
: Q_r(Q_r), Q_z(Q_z), Rs_r(Rs_r), Rs_z(Rs_z)
{
    // Taken from Alex Chao's resonator model (2.82)
    omega_r = 2 * pi * fr;
    alpha_r = omega_r / 2 / Q_r;
    omegabar_r = std::sqrt(omega_r * omega_r - alpha_r * alpha_r);

    omega_z = 2 * pi * fz;
    alpha_z = omega_z / 2 / Q_z;
    omegabar_z = std::sqrt(omega_z * omega_z - alpha_z * alpha_z);
}

double Wakefields::wake_rewall(double z)
{
    // Taken from Alex Chao's resisitve wall (2.53)
    double sigma = 5.4e17; // copper conductivity in CGS [1/s]
    double piper = 2e-3;
    double length = 6911;

    double field = -1 / (2 * pi * eps0) / (pi * piper * piper * piper)
                 * std::sqrt(c / sigma) * 1 / std::sqrt(-z) * length;

    return field;
}

double Wakefields::wake_resonator_r(double z)
{
    // Taken from Alex Chao's resonator model (2.82)
//    double field = c * Rs_r * omega_r * omega_r / Q_r / omegabar_r
//                 * exp(alpha_r * z / c) * sin(omegabar_r * z / c);
    double field = Rs_r * omega_r* omega_r / Q_r / omegabar_r
                 * std::exp(alpha_r * z / c) * std::sin(omegabar_r * z / c);

    return field;
}

double Wakefields::wake_resonator_z(double z)
{
    // Taken from Alex Chao's resonator model (2.82)
    double field = 2 * alpha_z * Rs_z * exp(alpha_z * z / c)
                 * (cos(omegabar_z * z / c)
                 + alpha_z / omegabar_z * sin(omegabar_z * z / c));

    return field;
}

void Wakefields::track(Bunch& bunch)
{
    std::ofstream out;
    out.precision(12);
    out.open("wakes.dat");

    double lambda_i;
    std::vector<int> index;

    double qp = bunch.intensity / bunch.get_nparticles();
    double scale = -bunch.charge * bunch.charge / (bunch.mass * bunch.gamma * c * c);

    for (size_t i=1; i<bunch.get_nslices() + 1; i++)
//    for (size_t i=bunch.get_nslices(); i>0; i--)
    {
        bunch.get_slice(i, lambda_i, index);
        int np_i = index.size();

        // Initialization
        kick_x = 0;
        kick_y = 0;
        kick_z = scale * qp * np_i * alpha_z * Rs_z; // Beam loading of self-slice

        // Interact with all sources
        for (size_t j=1; j<i; j++)
//        for (size_t j=bunch.get_nslices(); j>i; j--)
        {
            int np_j = bunch.slice_index[j].size();
            double zj = 1 / 2. * (bunch.slice_dz[j] - bunch.slice_dz[i]
                                + bunch.slice_dz[j + 1] - bunch.slice_dz[i + 1]);

//            double zj = 1 / 2. * (bunch.zbins[i] - bunch.zbins[j]
//                                + bunch.zbins[i + 1] - bunch.zbins[j + 1]);

            kick_x += scale * qp * np_j * bunch.mean_x[j] * wake_resonator_r(zj);
            kick_y += scale * qp * np_j * bunch.mean_y[j] * wake_resonator_r(zj);
            kick_z += scale * qp * np_j * wake_resonator_z(zj);

//            if (j==bunch.get_nslices())
            if (j==1)
            {
                bunch.mean_kx[i] = wake_resonator_r(zj);
                bunch.mean_ky[i] = kick_y;
                bunch.mean_kz[i] = wake_resonator_z(zj);
                out << std::scientific
                    << j << '\t' << kick_x << '\t'
                    << wake_resonator_r(zj) << '\t' << kick_z << std::endl;
            }
        }

        // Apply kicks
        for (int j=0; j<np_i; j++)
        {

//            if (i_pipe!=4)
//            {
//                if (i_pipe!=5)
//                {
//                    // Interact with all slices in front
//                    for (mmain=NBIN-1; mmain>jmain; mmain--)
//                    {
//                        zi = (jmain-mmain)*zstep;
//                        switch (i_pipe)
//                        {
//                        case 0:
//                            kick_x += wakefac*npr[mmain]*xs[mmain]*wake_func(zi);
//                            kick_y += wakefac*npr[mmain]*ys[mmain]*wake_func(zi);
//                            kick_z += wakefac*npr[mmain]*wake_funcz(zi);
//                            // printf("field= %13.8e \n",kick_x);
//                            break;
//
//                        case 1:
//                            kick_x += wakefac*npr[mmain]*0.41/1.23*(xs[mmain]-xpr[ind])*wake_func(zi);
//                            kick_y += wakefac*npr[mmain]*0.82/1.23*(ys[mmain]+0.5*ypr[ind])*wake_func(zi);
//                            kick_z += wakefac*npr[mmain]*wake_funcz(zi);
//                            break;
//
//                        case 2:
//                            kick_x += 0.;
//                            kick_y += wakefac*npr[mmain]*ys[mmain]*wake_func(zi);
//                            kick_z += wakefac*npr[mmain]*wake_funcz(zi);
//                            break;
//
//                        case 3:
//                            kick_x += wakefac*npr[mmain]*xs[mmain]*wake_reswall(zi);
//                            kick_y += wakefac*npr[mmain]*ys[mmain]*wake_reswall(zi);
//                            kick_z += wakefac*npr[mmain]*wake_funcz(zi);
//                            break;
//
//                        case 6:
//                            kick_x += wakefac*npr[mmain]*wake_reswall(zi)*(0.41*xs[mmain]-0.41*xpr[ind]);
//                            kick_y += wakefac*npr[mmain]*wake_reswall(zi)*(0.82*ys[mmain]+0.41*ypr[ind]);
//                            kick_z = 0.;       //prima era +=0 !!!!!!!!!
//                            break;
//                        }
//                }
//                else // (in the case i_pipe is equal to 5)
//                {
//                    ycoff =0.;     // b_coll/5.;
//                    kick_x = 0.;
//                    kick_y +=-wakefac*1/(4*PI*EPS0)*(npr[jmain]/zstep)*(theta_coll*4/b_coll)*((ys[jmain]+ycoff-ypr[ind])+PI*h_coll/2/b_coll*ypr[ind]);
//                    // kick_y=kick_y*1e4;
//                    // kick_y=1e-10;
//                    // kick_y=1e-9*rand()/(double)RAND_MAX;
//                    // kick_y=1e-9*2.*(rand()/(double)RAND_MAX - 0.5);
//                    kick_z =0.;
//                }
//            }
//            else //(in the case i_pipe is equal to 4)
//                kick_z = -9.57e-10 + 8.e-11*2.*(rand()/(double)RAND_MAX - 0.5);
//

            bunch.xp[index[j]] += kick_x;
            bunch.yp[index[j]] += kick_y;
            bunch.dp[index[j]] += kick_z;
        }
    }
}

BOOST_PYTHON_MODULE(libWakefields)
{
    class_<Wakefields>("Wakefields", init<double, double, double,
                                          double, double, double>())
        .def("track", &Wakefields::track)
    ;
}
