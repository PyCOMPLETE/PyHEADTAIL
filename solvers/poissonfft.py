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

        # Antiderivative
        ax = -self.dx / 2 + np.arange(self.nx + 1) * self.dx
        ay = -self.dy / 2 + np.arange(self.ny + 1) * self.dy

        x, y = np.meshgrid(ax, ay)
        r2 = x ** 2 + y ** 2

        tmpfgreen = -1 / 2 * (-3 * x * y + x * y * np.log(r2)
                   + x * x * np.arctan(y / x) + y * y * np.arctan(x / y)) # * 2 / dx / dy

        # Integration and circular Green's function
        self.fgreen[:self.nx, :self.ny] = tmpfgreen[1:, 1:] + tmpfgreen[:-1, :-1] - tmpfgreen[1:, :-1] - tmpfgreen[:-1, 1:]
        # # Would expect to be fully symmetric
        # self.fgreen[self.nx:, :self.ny] = self.fgreen[self.nx - 1::-1, :self.ny]
        # self.fgreen[:self.nx, self.ny:] = self.fgreen[:self.nx, self.ny - 1::-1]
        # self.fgreen[self.nx:, self.ny:] = self.fgreen[self.nx - 1::-1, self.ny - 1::-1]
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

    def compute_potential(self, bunch, i_slice):

        tmprho = np.zeros((2 * self.ny, 2 * self.nx))
        tmprho[:self.ny, :self.nx] = self.rho[:self.ny, :self.nx]

        fftphi = np.fft.fft2(tmprho) * np.fft.fft2(self.fgreen)
        
        tmpphi = np.fft.ifft2(fftphi)
        self.phi = tmpphi[:self.ny, :self.nx]

        # for (size_t j=0; j<np; j++)
        # {
        #     tmpphi[j] = std::sqrt(fftw_phi[j][0] * fftw_phi[j][0]
        #               + fftw_phi[j][1] * fftw_phi[j][1]);
        #     tmpphi[j] *= norm; // FFT specific
        # }

        # // Assign
        # for (size_t n=0; n<ny; n++)
        # {
        #     for (size_t m=0; m<nx; m++)
        #     {
        #         size_t k = n * nx + m;
        #         size_t l = n * 2 * nx + m;
        #         (*t.phi)[i_slice][k] = t.qsgn * tmpphi[l];
        #     }
        # }

#     void set_integrated_green_circular(size_t i, size_t j, size_t k)
#     {
#         fgreen[k] += fgreenbase[j][i];
#         fgreen[k] -= fgreenbase[j][i + 1];
#         fgreen[k] -= fgreenbase[j + 1][i];
#         fgreen[k] += fgreenbase[j + 1][i + 1];
#     }
