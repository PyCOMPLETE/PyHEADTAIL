'''
@authors: Adrian Oeftiger
@date:    12/09/2014

Provide abstract element as part of the tracking layout (e.g. circular
accelerator) for PyHeadtail. All implemented elements derive from this.
Can be used for implementing general features that every derived element
in PyHEADTAIL should have.
'''

from abc import ABCMeta, abstractmethod
from printers import ConsolePrinter

class Element(object):
    '''
    Abstract element as part of the tracking layout. Guarantees
    to have fulfill it's tracking contract via the method track(beam).
    Provides prints(output) method in order to communicate any output
    to the user. Use for instance

    >>> self.prints("Example message to console.")

    instead of

    >>> print ("Example message to console.")

    in order to obtain full flexibility over output channels.
    '''
    __metaclass__ = ABCMeta

    def __new__(cls, *args, **kwargs):
        '''
        Factory method makes sure that inheriting elements always
        have a Printer available for output redirection.
        If an inheriting element constructor gets the keyword argument
        'printer', an individual Printer as defined in the
        PyHEADTAIL.general.printers module can be attached to this
        instance. Standard is console output, i.e. ConsolePrinter.
        '''
        instance = object.__new__(cls)
        instance._printer = kwargs.get('printer', ConsolePrinter())
        return instance

    @abstractmethod
    def track(self, beam):
        '''
        Perform tracking of beam through this Element.
        '''
        pass

    def prints(self, output):
        '''
        Communicate any output to the user. Use for instance

        >>> self.prints("Example message to console.")

        instead of

        >>> print ("Example message to console.")

        in order to obtain full flexibility over output channels.
        '''
        self._printer.prints(output)
