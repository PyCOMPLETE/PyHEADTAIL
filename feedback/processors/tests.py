from linear_transform import LinearTransform
from addition import Addition
from multiplication import Multiplication
from convolution import FIRfilter
from ..core import debug_extension


class BypassLinearTransform(LinearTransform):
    """ A test processor for testing the abstract class of linear transform. The response function produces
        an unit matrix
    """
    def __init__(self,norm_type = None, norm_range = None):
        super(self.__class__, self).__init__(norm_type, norm_range)

    def response_function(self, ref_bin_mid, ref_bin_from, ref_bin_to, bin_mid, bin_from, bin_to):
        if ref_bin_mid == bin_mid:
            return 1
        else:
            return 0


class BypassMultiplication(Multiplication):
    """
    A test processor for testing the abstract class of multiplication
    """
    def __init__(self, normalization = 'maximum_weight'):
        super(self.__class__, self).__init__('signal', normalization)

    def multiplication_function(self,weight):
        return 1.


class BypassAddition(Addition):
    def __init__(self, normalization = 'maximum_weight'):
        super(self.__class__, self).__init__('signal', normalization)

    def addend_function(self,weight):
        return 0.


class BypassFIR(FIRfilter):

    def __init__(self):
        coefficients = [1.]

        super(self.__class__, self).__init__(coefficients)


