'''
@authors: Adrian Oeftiger
@date:    12/09/2014

Provide abstract element as part of the tracking layout (e.g. circular
accelerator) for PyHeadtail. All implemented elements derive from this.
Can be used for implementing general features that every derived element
in PyHEADTAIL should have.
'''
from .printers import ConsolePrinter
from abc import ABCMeta, abstractmethod


class Printing(object):
    '''Provides prints(output) method in order to communicate any output
    to the user. Use for instance

    >>> self.prints("Example message to console.")

    instead of

    >>> print ("Example message to console.")

    in order to obtain full flexibility over output channels.
    '''

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
        instance._warningprinter = kwargs.get('warningprinter',
                                              ConsolePrinter())
        return instance

    def prints(self, output):
        '''
        Communicate any output to the user. Use for instance

        >>> self.prints("Example message to console.")

        instead of

        >>> print ("Example message to console.")

        in order to obtain full flexibility over output channels.
        '''
        self._printer.prints(output)

    def warns(self, output):
        '''
        Communicate warnings to the user. Use for instance

        >>> self.warns("Example warning to console.")

        instead of

        >>> print ("Example message to console.")
        '''
        self._warningprinter.prints("*** PyHEADTAIL WARNING! " + output)


class Element(Printing):
    '''
    Abstract element as part of the tracking layout. Guarantees
    to fulfil its tracking contract via the method track(beam).
    '''
    __metaclass__ = ABCMeta

    @abstractmethod
    def track(self, beam):
        '''
        Perform tracking of beam through this Element.
        '''
        pass
