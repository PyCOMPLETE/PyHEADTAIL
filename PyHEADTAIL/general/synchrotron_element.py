from abc import ABCMeta, abstractmethod


class SynchrotronElement(object):
    __metaclass__ = ABCMeta

    def __init__(self):
        super(SynchrotronElement, self).__init__()

    @abstractmethod
    def track(self, bunches):
        pass
