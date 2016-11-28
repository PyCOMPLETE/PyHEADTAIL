import numpy as np
from PyHEADTAIL.general.utils import ListProxy
from scipy.constants import c

from abc import ABCMeta, abstractmethod


class KineticElement(object):
    """Documentation for KineticElement

    """
    __metaclass__ = ABCMeta

    def __init__(self, gamma, mass):
        self.gamma = gamma
        self.mass = mass

    @property
    def gamma(self):
        return self._gamma

    @gamma.setter
    def gamma(self, value):
        self._gamma = value
        self._beta = np.sqrt(1 - self.gamma**-2)
        self._betagamma = np.sqrt(self.gamma**2 - 1)
        self._p0 = self.betagamma * self.mass * c

    @property
    def beta(self):
        return self._beta

    @beta.setter
    def beta(self, value):
        self.gamma = 1. / np.sqrt(1 - value**2)

    @property
    def betagamma(self):
        return self._betagamma

    @betagamma.setter
    def betagamma(self, value):
        self.gamma = np.sqrt(value**2 + 1)

    @property
    def p0(self):
        return self._p0

    @p0.setter
    def p0(self, value):
        self.gamma = np.sqrt(1 + (value/(self.mass*c))**2)
        self.gamma = (1 / (c*self.mass) *
                      np.sqrt(value**2+self.mass**2*c**2))


class TrackingElement(object):
    """Documentation for TrackingElement

    """
    __metaclass__ = ABCMeta

    def __init__(self, args):
        super(TrackingElement, self).__init__()
        self.args = args

    @abstractmethod
    def track(self, arg):
        pass


class Detuner(object):
    """Documentation for Detuner

    """
    __metaclass__ = ABCMeta

    def __init__(self, args):
        super(Detuner, self).__init__()
        self.args = args

    @abstractmethod
    def generate(self, arg):
        pass

    @abstractmethod
    def detune(self, beam):
        pass


class Octupoles(Detuner):
    """Documentation for Octupoles

    """
    def __init__(self, ap_x, ap_y, ap_xy):
        super(Octupoles, self).__init__('args')
        self._ap_x = [ap_x]
        self._ap_y = [ap_y]
        self._ap_xy = [ap_xy]

        # self._ap_x = ListProxy(self, "ap_x")

    @property
    def ap_x(self):
        return self._ap_x[0]

    @ap_x.setter
    def ap_x(self, value):
        self._ap_x[0] = value

    def ap_y():
        doc = """Doc string"""
        def fget(self):
            return self._ap_y[0]

        def fset(self, value):
            self._ap_y[0] = value

        def fdel(self):
            del self._ap_y
        return locals()
    ap_y = property(**ap_y())

    def generate(self, arg):
        self.arg = arg

    def detune(self, beam):
        pass


class Segment(object):
    """Documentation for Segment

    """
    def __init__(self,
                 alpha_0, beta_0, alpha_1, beta_1,
                 dmu, detuner_list, plane, D_0=0, D_1=0):
        super(Segment, self).__init__()
        self.alpha_0 = alpha_0
        self.beta_0 = beta_0
        self.D_0 = D_0
        self.alpha_1 = alpha_1
        self.beta_1 = beta_1
        self.D_1 = D_1
        self.dmu = dmu
        self.detuner_list = detuner_list
        self.plane = plane

    def build_matrix(self, arg):

        B = [[np.sqrt(1/self.beta_0), 0],
             [-np.sqrt(self.alpha_0/self.beta_0), np.sqrt(self.beta_0)]]
        R = [[np.cos(self.dmu), np.sin(self.dmu)],
             [-np.sin(self.dmu), np.cos(self.dmu)]]
        B_inv = [[np.sqrt(self.beta_0), 0],
                 [np.sqrt(self.alpha_1/self.beta_1), np.sqrt(1/self.beta_1)]]

        I = [[1, 0],
             [0, 1]]
        S = [[0, 1],
             [-1, 0]]

        self.M = np.dot(B_inv, np.dot(R, B))
        self.C = np.dot(B_inv, np.dot(I, B))
        self.S = np.dot(B_inv, np.dot(S, B))

    def kick(self, beam):

        if self.plane == 'x':
            beam.x = (
                (self.C[0, 0]*np.cos(self.dmu) +
                 self.S[0, 0]*np.sin(self.dmu)) * beam.x +
                (self.C[0, 1]*np.cos(self.dmu) +
                 self.S[0, 1]*np.sin(self.dmu)) * beam.xp)
            beam.xp = (
                (self.C[1, 0]*np.cos(self.dmu) +
                 self.S[1, 0]*np.sin(self.dmu)) * beam.x +
                (self.C[1, 1]*np.cos(self.dmu) +
                 self.S[1, 1]*np.sin(self.dmu)) * beam.xp)

        if self.plane == 'y':
            beam.y = (
                (self.C[0, 0]*np.cos(self.dmu) +
                 self.S[0, 0]*np.sin(self.dmu)) * beam.y +
                (self.C[0, 1]*np.cos(self.dmu) +
                 self.S[0, 1]*np.sin(self.dmu)) * beam.yp)
            beam.yp = (
                (self.C[1, 0]*np.cos(self.dmu) +
                 self.S[1, 0]*np.sin(self.dmu)) * beam.y +
                (self.C[1, 1]*np.cos(self.dmu) +
                 self.S[1, 1]*np.sin(self.dmu)) * beam.yp)
        if self.plane == 'z':
            beam.z = (
                (self.C[0, 0]*np.cos(self.dmu) +
                 self.S[0, 0]*np.sin(self.dmu)) * beam.z +
                (self.C[0, 1]*np.cos(self.dmu) +
                 self.S[0, 1]*np.sin(self.dmu)) * beam.dp)
            beam.dp = (
                (self.C[1, 0]*np.cos(self.dmu) +
                 self.S[1, 0]*np.sin(self.dmu)) * beam.z +
                (self.C[1, 1]*np.cos(self.dmu) +
                 self.S[1, 1]*np.sin(self.dmu)) * beam.dp)
