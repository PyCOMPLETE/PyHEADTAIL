

import numpy as np


def stationary_exponential(H, Hmax, H0, bunch):

    def psi(dz, dp):
        result = np.exp(H(dz, dp, bunch) / H0) - np.exp(Hmax / H0)
        return result

    return psi
