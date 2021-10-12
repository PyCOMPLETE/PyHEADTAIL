from abc import ABCMeta

import numpy as np
from scipy.constants import c, m_p


class KineticElement(object, metaclass=ABCMeta):
    def __init__(self):
        super(KineticElement, self).__init__()

    @property
    def p0(self):
        return self._p0

    @p0.setter
    def p0(self, value):
        self._p0 = value
        self._gamma = np.sqrt(1 + (value / m_p / c) ** 2)  # TODO adapt to ions
        self._beta = np.sqrt(1 - self._gamma ** -2)

    @property
    def gamma(self):
        return self._gamma

    @gamma.setter
    def gamma(self, value):
        self.p0 = m_p * c * np.sqrt(value ** 2 - 1)

    @property
    def beta(self):
        return self._beta

    @beta.setter
    def beta(self, value):
        self.p0 = m_p * c * np.sqrt(value ** 2 / (1 - value ** 2))
