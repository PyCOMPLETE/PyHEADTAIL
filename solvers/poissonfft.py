from __future__ import division
'''
Created on 08.01.2014

@author: Kevin Li
'''


import numpy as np


from solvers.grid import *
from solvers.compute_potential_fgreenm2m import compute_potential_fgreenm2m


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

        mx = -self.dx / 2 + np.arange(self.nx + 1) * self.dx
        my = -self.dy / 2 + np.arange(self.ny + 1) * self.dy

        x, y = np.meshgrid(mx, my)
        r2 = x ** 2 + y ** 2

        # Antiderivative
        tmpfgreen = -1 / 2 * (-3 * x * y + x * y * np.log(r2)
                   + x * x * np.arctan(y / x) + y * y * np.arctan(x / y)) # * 2 / dx / dy

        # Integration and circular Green's function
        self.fgreen[:self.nx, :self.ny] = tmpfgreen[1:, 1:] + tmpfgreen[:-1, :-1] - tmpfgreen[1:, :-1] - tmpfgreen[:-1, 1:]
        self.fgreen[self.nx:, :self.ny] = self.fgreen[self.nx:0:-1, :self.ny]
        self.fgreen[:self.nx, self.ny:] = self.fgreen[:self.nx, self.ny:0:-1]
        self.fgreen[self.nx:, self.ny:] = self.fgreen[self.nx:0:-1, self.ny:0:-1]
        # # Would expect to be fully symmetric
        # self.fgreen[self.nx:, :self.ny] = self.fgreen[self.nx - 1::-1, :self.ny]
        # self.fgreen[:self.nx, self.ny:] = self.fgreen[:self.nx, self.ny - 1::-1]
        # self.fgreen[self.nx:, self.ny:] = self.fgreen[self.nx - 1::-1, self.ny - 1::-1]

        from types import MethodType
        PoissonFFT.compute_potential_fgreenm2m = MethodType(compute_potential_fgreenm2m, None, PoissonFFT)

        # self.fgreen1 = np.zeros(4 * self.nx * self.ny)

        # tmpfgreen1 = np.zeros((self.nx + 1, self.ny + 1))
        # for i in range(self.nx + 1):
        #     for j in range(self.ny + 1):
        #         x = -self.dx / 2 + i * self.dx
        #         y = -self.dy / 2 + j * self.dy
        #         r2 = x ** 2 + y ** 2
        #         tmpfgreen1[i, j] = -1 / 2 * (-3 * x * y + x * y * np.log(r2)
        #                           + x * x * np.arctan(y / x) + y * y * np.arctan(x / y))

        # # Base region
        # for i in range(self.nx):
        #     for j in range(self.ny):
        #         k = i * 2 * self.ny + j
        #         self.fgreen1[k] += tmpfgreen1[i, j]
        #         self.fgreen1[k] -= tmpfgreen1[i + 1, j]
        #         self.fgreen1[k] -= tmpfgreen1[i, j + 1]
        #         self.fgreen1[k] += tmpfgreen1[i + 1, j + 1]
        # # Mirror x
        # for i in range(self.nx):
        #     for j in range(1, self.ny):
        #         k = (2 * self.ny) * (i + 1)  - j
        #         self.fgreen1[k] += tmpfgreen1[i, j]
        #         self.fgreen1[k] -= tmpfgreen1[i + 1, j]
        #         self.fgreen1[k] -= tmpfgreen1[i, j + 1]
        #         self.fgreen1[k] += tmpfgreen1[i + 1, j + 1]
        # # Mirror y
        # for i in range(1, self.nx):
        #     for j in range(self.ny):
        #         k = ((2 * self.nx)
        #            * (2 * self.ny - i) + j)
        #         self.fgreen1[k] += tmpfgreen1[i, j]
        #         self.fgreen1[k] -= tmpfgreen1[i + 1, j]
        #         self.fgreen1[k] -= tmpfgreen1[i, j + 1]
        #         self.fgreen1[k] += tmpfgreen1[i + 1, j + 1]
        # # Mirror xy
        # for i in range(1, self.nx):
        #     for j in range(1, self.ny):
        #         k = ((2 * self.nx)
        #            * (2 * self.ny - (i - 1)) - (j))
        #         self.fgreen1[k] += tmpfgreen1[i, j]
        #         self.fgreen1[k] -= tmpfgreen1[i + 1, j]
        #         self.fgreen1[k] -= tmpfgreen1[i, j + 1]
        #         self.fgreen1[k] += tmpfgreen1[i + 1, j + 1]

        # self.fgreen1 = np.reshape(self.fgreen1, (2 * self.nx, 2 * self.ny))

        # print self.fgreen[:self.nx, 0]
        # print self.fgreen[2 * self.nx:self.nx:-1, 0]
        # print self.fgreen[:self.nx, 0] - self.fgreen[2 * self.nx:self.nx:-1, 0]

    # @profile
    def compute_potential(self):

        tmprho = np.zeros((2 * self.ny, 2 * self.nx))
        tmprho[:self.ny, :self.nx] = self.rho[:self.ny, :self.nx]

        fftphi = np.fft.fft2(tmprho) * np.fft.fft2(self.fgreen)

        tmpphi = np.fft.ifft2(fftphi)
        self.phi = np.abs(tmpphi[:self.ny, :self.nx])

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

    # @profile
    def py_green_m2m(self):

        self.phi = np.zeros((self.ny, self.nx))

        da = self.dx * self.dy
        x, y = self.x.flatten(), self.y.flatten()
        phi, rho = self.phi.flatten(), self.rho.flatten()

        for i in xrange(self.n_points):
            xi, yi = x[i], y[i]
            for j in xrange(self.n_points):
                xj, yj = x[j], y[j]
                r2 = (xi - xj) ** 2 + (yi - yj) ** 2 + 1e-6
                phi[i] += (-rho[j] * da * 1 / 2 * np.log(r2)) # * self.dx * self.dy / (2 * np.pi))

        self.phi = np.reshape(phi, (self.ny, self.nx))
        #     for j in points:
        #         r = j - i + 1e-6
        #         phi += np.log(r)

        # self.phi = 0

# class PoissonGreen(RectilinearGrid2D):

#     def __init__(self, n_points_x, n_points_y, dim_x, dim_y):
#         # Inheritance
#         RectilinearGrid2D.__init__(self, "regular", n_points_x, n_points_y,
#                                    dim_x, dim_y)

#         # Allocation
#         self.dx = 2 * abs(self.dim_x) / (self.n_points_x - 1)
#         self.dy = 2 * abs(self.dim_y) / (self.n_points_y - 1)
#         self.fgreen = np.zeros((4 * self.n_points_x * self.n_points_y))
#         self.fgreeni = np.zeros((4 * self.n_points_x * self.n_points_y))

#         # Initialisation
#         self.initd()
#         self.baseGreen()
#         self.transformedGreen()
#         self.integratedGreen()

#     def initd(self):
#         ''' Double grid initialisation'''
#         mx = [self.dim_x + (i + 1) * self.dx for i in range(self.n_points_x)]
#         my = [self.dim_y + (j + 1) * self.dy for j in range(self.n_points_y)]

#         self.mx = np.append(self.mx, mx)
#         self.my = np.append(self.my, my)

#     def baseGreen(self):
#         '''
#         Standard Green's function: log(r)
#         '''
#         tmpfgreen = np.zeros((self.n_points_x, self.n_points_y))
#         for i in range(self.n_points_x):
#             for j in range(self.n_points_y):
#                 if i != 0 or j != 0:
#                     x = i * self.dx
#                     y = j * self.dy
#                     r2 = x ** 2 + y ** 2
#                     tmpfgreen[i][j] = -1 / 2. * np.log(r2)
#         tmpfgreen[0][0] = tmpfgreen[1][0] / 2. + tmpfgreen[0][1] / 2.

#         # Base region
#         for i in range(self.n_points_x):
#             for j in range(self.n_points_y):
#                 k = 2 * self.n_points_y * i + j
#                 self.fgreen[k] = tmpfgreen[i][j]
#         # Mirror x
#         for i in range(self.n_points_x):
#             for j in range(1, self.n_points_y):
#                 k = (2 * self.n_points_y) * (i + 1) - j
#                 self.fgreen[k] = tmpfgreen[i][j]
#         # Mirror y
#         for i in range(1, self.n_points_x):
#             for j in range(self.n_points_y):
#                 k = ((2 * self.n_points_y)
#                    * (2 * self.n_points_x - i) + j)
#                 self.fgreen[k] = tmpfgreen[i][j]
#         # Mirror xy
#         for i in range(1, self.n_points_x):
#             for j in range(1, self.n_points_y):
#                 k = ((2 * self.n_points_y)
#                    * (2 * self.n_points_x - (i - 1)) - j)
#                 self.fgreen[k] = tmpfgreen[i][j]

#     def transformedGreen(self):
#         '''
#         Green's function in Fourier space: 1 / (k ** 2)
#         '''
#         self.G = np.zeros((4 * self.n_points_x * self.n_points_y))
#         kx, ky = np.pi / abs(self.dim_x), np.pi / abs(self.dim_y)

#         tmpfgreen = np.zeros((self.n_points_x, self.n_points_y))
#         for i in range(self.n_points_x):
#             for j in range(self.n_points_y):
#                 x = i * kx
#                 y = j * ky
#                 tmpfgreen[i, j] = -1. / np.sqrt(x ** 2 + y ** 2 + 1e-5 ** 2)

#         # Base region
#         for i in range(self.n_points_x):
#             for j in range(self.n_points_y):
#                 k = (2 * self.n_points_y - 1) * i + j
#                 self.G[k] = tmpfgreen[i][j]
#         # Mirror x
#         for i in range(self.n_points_x - 1):
#             for j in range(self.n_points_y):
#                 k = (2 * self.n_points_y - 1) \
#                   * (2 * self.n_points_x - 2 - i) + j
#                 self.G[k] = tmpfgreen[i][j]
#         # Mirror y
#         for i in range(self.n_points_x):
#             for j in range(self.n_points_y - 1):
#                 k = (2 * self.n_points_y - 1) * (i + 1) - j - 1
#                 self.G[k] = tmpfgreen[i][j]
#         # Mirror xy
#         for i in range(self.n_points_x - 1):
#             for j in range(self.n_points_y - 1):
#                 k = (2 * self.n_points_y - 1) \
#                   * (2 * self.n_points_x - 2 - i + 1) - j - 1
#                 self.G[k] = tmpfgreen[i][j]

#     def integratedGreen(self):

#         tmpfgreen = np.zeros((self.n_points_x + 1, self.n_points_y + 1))
#         for i in range(self.n_points_x + 1):
#             for j in range(self.n_points_y + 1):
#                 x = -self.dx / 2. + i * self.dx
#                 y = -self.dy / 2. + j * self.dy
#                 r2 = x ** 2 + y ** 2
#                 tmpfgreen[i][j] = -1 / 2. * (-3 * x * y + x * y * log(r2)
#                     + x * x * np.arctan(y / x) + y * y * np.arctan(x / y))

#         # Base region
#         for i in range(self.n_points_x):
#             for j in range(self.n_points_y):
#                 k = i * 2 * self.n_points_y + j
#                 self.fgreeni[k] += tmpfgreen[i][j]
#                 self.fgreeni[k] -= tmpfgreen[i + 1][j]
#                 self.fgreeni[k] -= tmpfgreen[i][j + 1]
#                 self.fgreeni[k] += tmpfgreen[i + 1][j + 1]
#         # Mirror x
#         for i in range(self.n_points_x):
#             for j in range(1, self.n_points_y):
#                 k = (2 * self.n_points_y) * (i + 1)  - j
#                 self.fgreeni[k] += tmpfgreen[i][j]
#                 self.fgreeni[k] -= tmpfgreen[i + 1][j]
#                 self.fgreeni[k] -= tmpfgreen[i][j + 1]
#                 self.fgreeni[k] += tmpfgreen[i + 1][j + 1]
#         # Mirror y
#         for i in range(1, self.n_points_x):
#             for j in range(self.n_points_y):
#                 k = ((2 * self.n_points_x)
#                    * (2 * self.n_points_y - i) + j)
#                 self.fgreeni[k] += tmpfgreen[i][j]
#                 self.fgreeni[k] -= tmpfgreen[i + 1][j]
#                 self.fgreeni[k] -= tmpfgreen[i][j + 1]
#                 self.fgreeni[k] += tmpfgreen[i + 1][j + 1]
#         # Mirror xy
#         for i in range(1, self.n_points_x):
#             for j in range(1, self.n_points_y):
#                 k = ((2 * self.n_points_x)
#                    * (2 * self.n_points_y - (i - 1)) - (j))
#                 self.fgreeni[k] += tmpfgreen[i][j]
#                 self.fgreeni[k] -= tmpfgreen[i + 1][j]
#                 self.fgreeni[k] -= tmpfgreen[i][j + 1]
#                 self.fgreeni[k] += tmpfgreen[i + 1][j + 1]
# #        # Mirror x
# #        for i in range(self.n_points_x):
# #            for j in range(self.n_points_y - 1):
# #                k = (2 * self.n_points_y) * (i + 1)  - (j + 1)
# #                self.fgreeni[k] += tmpfgreen[i][j]
# #                self.fgreeni[k] -= tmpfgreen[i + 1][j]
# #                self.fgreeni[k] -= tmpfgreen[i][j + 1]
# #                self.fgreeni[k] += tmpfgreen[i + 1][j + 1]
# #        # Mirror y
# #        for i in range(self.n_points_x - 1):
# #            for j in range(self.n_points_y):
# #                k = ((2 * self.n_points_y)
# #                   * (2 * self.n_points_x - (i + 1)) + j)
# #                self.fgreeni[k] += tmpfgreen[i][j]
# #                self.fgreeni[k] -= tmpfgreen[i + 1][j]
# #                self.fgreeni[k] -= tmpfgreen[i][j + 1]
# #                self.fgreeni[k] += tmpfgreen[i + 1][j + 1]
# #        # Mirror xy
# #        for i in range(self.n_points_x - 1):
# #            for j in range(self.n_points_y - 1):
# #                k = ((2 * self.n_points_y)
# #                   * (2 * self.n_points_x - i) - (j + 1))
# #                self.fgreeni[k] += tmpfgreen[i][j]
# #                self.fgreeni[k] -= tmpfgreen[i + 1][j]
# #                self.fgreeni[k] -= tmpfgreen[i][j + 1]
# #                self.fgreeni[k] += tmpfgreen[i + 1][j + 1]

#     def m2mGreen(self, b):
#         '''
#         Points-to-mesh solver
#         '''
#         phi = np.zeros((self.n_points_x, self.n_points_y))
#         rho = np.reshape(b, (self.n_points_x, self.n_points_y))

#         for i in range(self.n_points_x):
#             for j in range(self.n_points_y):
#                 x0 = -abs(self.dim_x) + i * self.dx
#                 y0 = -abs(self.dim_y) + j * self.dy
#                 for k in range(self.n_points_x):
#                     for l in range(self.n_points_y):
#                         x = -abs(self.dim_x) + k * self.dx
#                         y = -abs(self.dim_y) + l * self.dy
#                         r2 = (x0 - x) ** 2 + (y0 - y) ** 2 + 1e-3 ** 2
#                         phi[i, j] += (-rho[k, l] * 1 / 2. * np.log(r2)
#                             * self.dx * self.dy / (2 * np.pi))

#         # Plot
#         figure(10)
#         XX, YY = np.meshgrid(self.mx[:self.n_points_x],
#                              self.my[:self.n_points_y])
#         XX, YY = XX.T, YY.T
#         ct = contour(XX, YY, phi, 40)
# #        clabel(ct, manual=True)
#         gca().set_aspect('equal')

#         return phi

#     def p2mGreen(self, beam):
#         '''
#         Piont-to-point solver
#         '''
#         phi = np.zeros((self.n_points_x, self.n_points_y))

#         for i in range(self.n_points_y):
#             for j in range(self.n_points_y):
#                 x0 = -abs(self.dim_x) + i * self.dx
#                 y0 = -abs(self.dim_y) + j * self.dy
#                 for k in range(int(beam.n_particles)):
#                         xp = beam.x[k]
#                         yp = beam.y[k]
#                         r2 = (x0 - xp) ** 2 + (y0 - yp) ** 2
#                         phi[i, j] += -1 / 2. * np.log(r2)

#         # Plot
#         figure(11)
#         XX, YY = np.meshgrid(self.mx[:self.n_points_x],
#                              self.my[:self.n_points_y])
#         XX, YY = XX.T, YY.T
#         contour(XX, YY, phi, 40)
# #        gca().set_xlim(np.amin(self.mx), abs(np.amin(self.mx)))
# #        gca().set_ylim(np.amin(self.my), abs(np.amin(self.my)))
#         gca().set_aspect('equal')

#         return phi

#     def solveFFT(self, b):
#         b *= 0
#         b[528] = 96100
        
#         nx = 2 * self.n_points_x
#         ny = 2 * self.n_points_y

#         rho = np.zeros((nx, ny))
#         rho[:self.n_points_x, :self.n_points_y] = np.reshape(
#             b, (self.n_points_x, self.n_points_y))

#         fgreen = np.reshape(self.fgreen, (nx, ny))
#         fgreeni = np.reshape(self.fgreeni, (nx, ny)) * 1. / (self.dx * self.dy)
#         G = np.reshape(self.G, (nx, ny))

# #         Mysterious delete
# #        fgreeni = np.insert(fgreeni, self.n_points_x, 0, 0)
# #        fgreeni = np.insert(fgreeni, self.n_points_y, 0, 1)
# #        fgreeni = np.delete(fgreeni, fgreeni.shape[0] - 1, 0)
# #        fgreeni = np.delete(fgreeni, fgreeni.shape[1] - 1, 1)

# #        fgreen = np.delete(fgreen, self.n_points_x, 0)
# #        fgreen = np.delete(fgreen, fgreen.shape[0] - 1, 0)
# #        fgreen = np.delete(fgreen, self.n_points_y, 1)
# #        fgreen = np.delete(fgreen, fgreen.shape[1] - 1, 1)
# #
# #        fgreeni = np.delete(fgreeni, self.n_points_x, 0)
# #        fgreeni = np.delete(fgreeni, fgreeni.shape[0] - 1, 0)
# #        fgreeni = np.delete(fgreeni, self.n_points_y, 1)
# #        fgreeni = np.delete(fgreeni, fgreeni.shape[1] - 1, 1)
# #
# #        G = np.delete(G, self.n_points_x, 0)
# #        G = np.delete(G, G.shape[0] - 1, 0)
# #        G = np.delete(G, self.n_points_y, 1)
# #        G = np.delete(G, G.shape[1] - 1, 1)

#         # Perform FFT
#         rhofft = np.fft.fft2(rho)
#         fgreenfft = np.fft.fft2(fgreen)
#         fgreenffti = np.fft.fft2(fgreeni)

#         phi = np.fft.ifft2(rhofft * fgreenfft)#[:self.n_points_x, :self.n_points_y]
#         phii = np.fft.ifft2(rhofft * fgreenffti)#[:self.n_points_x, :self.n_points_y]
#         phiG = np.fft.ifft2(rhofft * G)#[:self.n_points_x, :self.n_points_y]
        
#         # Rescale for volume correction
#         phi *= self.dx * self.dy / (2 * np.pi)
#         phii *= self.dx * self.dy / (2 * np.pi)
#         phiG *= self.dx * self.dy / (2 * np.pi)

# #        Green = np.fft.ifft2(G)

#         # Plot
#         mx = self.mx
#         my = self.my
# #        mx = self.mx[:self.n_points_x]
# #        my = self.my[:self.n_points_y]
# #        [axvline(i, c='r', lw=0.5) for i in mx]
# #        [axhline(i, c='r', lw=0.5) for i in my]
#         XX, YY = np.meshgrid(mx, my)
#         XX, YY = XX.T, YY.T

# #        figure(1)
# #        ct = contour(XX, YY, abs(phi), 40)
# #        scatter(XX, YY, c=rho, s=0.001 * rho, lw=0)
# #        gca().set_xlim(np.amin(mx), abs(np.amin(mx)))
# #        gca().set_ylim(np.amin(my), abs(np.amin(my)))
# #        gca().set_aspect('equal')
# #        figure(2)
#         contour(XX, YY, abs(phii), 40)
# #        scatter(XX, YY, c=rho, s=0.001 * rho, lw=0)
# #        gca().set_xlim(np.amin(mx), abs(np.amin(mx)))
# #        gca().set_ylim(np.amin(my), abs(np.amin(my)))
# #        gca().set_aspect('equal')
# #        figure(3)
# #        contour(XX, YY, abs(fgreeni), 40)
# #        gca().set_xlim(np.amin(mx), abs(np.amin(mx)))
# #        gca().set_ylim(np.amin(my), abs(np.amin(my)))
# #        gca().set_aspect('equal')

# #        savetxt("myfgreen.dat", fgreeni)

#         return phii
