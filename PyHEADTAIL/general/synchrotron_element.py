from abc import ABCMeta, abstractmethod


class SynchrotronElement(object, metaclass=ABCMeta):
    def __init__(self):
        super(SynchrotronElement, self).__init__()

    @abstractmethod
    def track(self, bunches):
        pass
