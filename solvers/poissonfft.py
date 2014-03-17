'''
Created on 08.01.2014

@author: Kevin Li
'''


import numpy as np


from solvers.grid import *


class PoissonFFT(UniformGrid):
    '''
    classdocs
    '''

    # def __init__(self, n_points):
    #     '''
    #     Constructor
    #     '''
    #     self.n_points = n_points

    #     self.fgreen = np.zeros(100)

    void set_integrated_green_circular(size_t i, size_t j, size_t k)
    {
        fgreen[k] += fgreenbase[j][i];
        fgreen[k] -= fgreenbase[j][i + 1];
        fgreen[k] -= fgreenbase[j + 1][i];
        fgreen[k] += fgreenbase[j + 1][i + 1];
    }
    void compute_integrated_green_base()
    {
        size_t nx = get_npoints_x();
        size_t ny = get_npoints_y();

        double dx = (mx.back() - mx.front()) / (nx - 1);
        double dy = (my.back() - my.front()) / (ny - 1);

        for (size_t j=0; j<ny + 1; j++)
            for (size_t i=0; i<nx + 1; i++)
            {
                double x = -dx / 2. + i * dx;
                double y = -dy / 2. + j * dy;
                double r2 = x * x + y * y;
                fgreenbase[j][i] = -1 / 2. * (-3 * x * y + x * y * log(r2)
                                 + x * x * atan(y / x) + y * y * atan(x / y));
//                                 * 2 / dx / dy;
            }
    }
    void compute_integrated_green_circular()
    {
        size_t nx = get_npoints_x();
        size_t ny = get_npoints_y();
        size_t np = 4 * nx * ny;

        // Initialise
        for (size_t i=0; i<np; i++)
            fgreen[i] = 0;

        // Base region
        for (size_t j=0; j<ny; j++)
            for (size_t i=0; i<nx; i++)
            {
                size_t k = j * 2 * nx + i;
                set_integrated_green_circular(i, j, k);
            }
        // Mirror x
        for (size_t j=0; j<ny; j++)
            for (size_t i=1; i<nx; i++)
            {
                size_t k = (j + 1) * 2 * nx - i;
                set_integrated_green_circular(i, j, k);
            }
        // Mirror y
        for (size_t j=1; j<ny; j++)
            for (size_t i=0; i<nx; i++)
            {
                size_t k = (2 * ny - j) * 2 * nx + i;
                set_integrated_green_circular(i, j, k);
            }
        // Mirror xy
        for (size_t j=1; j<ny; j++)
            for (size_t i=1; i<nx; i++)
            {
                size_t k = (2 * ny - (j - 1)) * 2 * nx - i;
                set_integrated_green_circular(i, j, k);
            }
    }
