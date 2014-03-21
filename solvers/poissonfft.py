from __future__ import division
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

    # @profile
    def __init__(self, *args, **kwargs):
        '''
        Constructor
        '''
        super(PoissonFFT, self).__init__(*args, **kwargs)

        self.fgreen = np.zeros((2 * self.nx, 2 * self.ny))

        # for j, y in enumerate(ay):
        #     for i, x in enumerate(ax):
        #         self.fgreen[j][i] = -1 / 2 * (-3 * x * y + x * y * np.log(x ** 2 + y ** 2)
        #                             + x * x * np.arctan(y / x) + y * y * np.arctan(x / y)) # * 2 / dx / dy

        # Antiderivative
        ax = -self.dx / 2 + np.arange(self.nx + 1) * self.dx
        ay = -self.dy / 2 + np.arange(self.ny + 1) * self.dy

        x, y = np.meshgrid(ax, ay)
        r2 = x ** 2 + y ** 2

        tmpfgreen = -1 / 2 * (-3 * x * y + x * y * np.log(r2)
                   + x * x * np.arctan(y / x) + y * y * np.arctan(x / y)) # * 2 / dx / dy

        # Integration and circular Green's function
        self.fgreen[:self.nx, :self.ny] = tmpfgreen[1:, 1:] + tmpfgreen[:-1, :-1] - tmpfgreen[1:, :-1] - tmpfgreen[:-1, 1:]
        self.fgreen[self.nx:, :self.ny] = self.fgreen[self.nx:0:-1, :self.ny]
        self.fgreen[:self.nx, self.ny:] = self.fgreen[:self.nx, self.ny:0:-1]
        self.fgreen[self.nx:, self.ny:] = self.fgreen[self.nx:0:-1, self.ny:0:-1]


        self.fgreen1 = np.zeros(4 * self.nx * self.ny)

        tmpfgreen1 = np.zeros((self.nx + 1, self.ny + 1))
        for i in range(self.nx + 1):
            for j in range(self.ny + 1):
                x = -self.dx / 2 + i * self.dx
                y = -self.dy / 2 + j * self.dy
                r2 = x ** 2 + y ** 2
                tmpfgreen1[i, j] = -1 / 2 * (-3 * x * y + x * y * np.log(r2)
                                  + x * x * np.arctan(y / x) + y * y * np.arctan(x / y))

        # Base region
        for i in range(self.nx):
            for j in range(self.ny):
                k = i * 2 * self.ny + j
                self.fgreen1[k] += tmpfgreen1[i, j]
                self.fgreen1[k] -= tmpfgreen1[i + 1, j]
                self.fgreen1[k] -= tmpfgreen1[i, j + 1]
                self.fgreen1[k] += tmpfgreen1[i + 1, j + 1]
        # Mirror x
        for i in range(self.nx):
            for j in range(1, self.ny):
                k = (2 * self.ny) * (i + 1)  - j
                self.fgreen1[k] += tmpfgreen1[i, j]
                self.fgreen1[k] -= tmpfgreen1[i + 1, j]
                self.fgreen1[k] -= tmpfgreen1[i, j + 1]
                self.fgreen1[k] += tmpfgreen1[i + 1, j + 1]
        # Mirror y
        for i in range(1, self.nx):
            for j in range(self.ny):
                k = ((2 * self.nx)
                   * (2 * self.ny - i) + j)
                self.fgreen1[k] += tmpfgreen1[i, j]
                self.fgreen1[k] -= tmpfgreen1[i + 1, j]
                self.fgreen1[k] -= tmpfgreen1[i, j + 1]
                self.fgreen1[k] += tmpfgreen1[i + 1, j + 1]
        # Mirror xy
        for i in range(1, self.nx):
            for j in range(1, self.ny):
                k = ((2 * self.nx)
                   * (2 * self.ny - (i - 1)) - (j))
                self.fgreen1[k] += tmpfgreen1[i, j]
                self.fgreen1[k] -= tmpfgreen1[i + 1, j]
                self.fgreen1[k] -= tmpfgreen1[i, j + 1]
                self.fgreen1[k] += tmpfgreen1[i + 1, j + 1]

        self.fgreen1 = np.reshape(self.fgreen1, (2 * self.nx, 2 * self.ny))

        # print self.fgreen[:self.nx, 0]
        # print self.fgreen[2 * self.nx:self.nx:-1, 0]
        # print self.fgreen[:self.nx, 0] - self.fgreen[2 * self.nx:self.nx:-1, 0]

#     void set_integrated_green_circular(size_t i, size_t j, size_t k)
#     {
#         fgreen[k] += fgreenbase[j][i];
#         fgreen[k] -= fgreenbase[j][i + 1];
#         fgreen[k] -= fgreenbase[j + 1][i];
#         fgreen[k] += fgreenbase[j + 1][i + 1];
#     }
#     void compute_integrated_green_base()
#     {
#         size_t nx = get_npoints_x();
#         size_t ny = get_npoints_y();

#         double dx = (mx.back() - mx.front()) / (nx - 1);
#         double dy = (my.back() - my.front()) / (ny - 1);

#         for (size_t j=0; j<ny + 1; j++)
#             for (size_t i=0; i<nx + 1; i++)
#             {
#                 double x = -dx / 2. + i * dx;
#                 double y = -dy / 2. + j * dy;
#                 double r2 = x * x + y * y;
#                 fgreenbase[j][i] = -1 / 2. * (-3 * x * y + x * y * log(r2)
#                                  + x * x * atan(y / x) + y * y * atan(x / y));
# //                                 * 2 / dx / dy;
#             }
#     }
#     void compute_integrated_green_circular()
#     {
#         size_t nx = get_npoints_x();
#         size_t ny = get_npoints_y();
#         size_t np = 4 * nx * ny;

#         // Initialise
#         for (size_t i=0; i<np; i++)
#             fgreen[i] = 0;

#         // Base region
#         for (size_t j=0; j<ny; j++)
#             for (size_t i=0; i<nx; i++)
#             {
#                 size_t k = j * 2 * nx + i;
#                 set_integrated_green_circular(i, j, k);
#             }
#         // Mirror x
#         for (size_t j=0; j<ny; j++)
#             for (size_t i=1; i<nx; i++)
#             {
#                 size_t k = (j + 1) * 2 * nx - i;
#                 set_integrated_green_circular(i, j, k);
#             }
#         // Mirror y
#         for (size_t j=1; j<ny; j++)
#             for (size_t i=0; i<nx; i++)
#             {
#                 size_t k = (2 * ny - j) * 2 * nx + i;
#                 set_integrated_green_circular(i, j, k);
#             }
#         // Mirror xy
#         for (size_t j=1; j<ny; j++)
#             for (size_t i=1; i<nx; i++)
#             {
#                 size_t k = (2 * ny - (j - 1)) * 2 * nx - i;
#                 set_integrated_green_circular(i, j, k);
#             }
#     }
