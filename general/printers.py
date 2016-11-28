'''
@authors: Adrian Oeftiger
@date:    12/09/2014

Printer functionality provides different means to control the flow of
output streams.
'''

from abc import ABCMeta, abstractmethod


class Printer(object):
    '''
    A generic printer knows where to redirect text for print.
    Use Printer.prints(output) to print the output instead of
    using the standard keyword

    >>> print (output)

    in order to gain flexibility in redirecting output centrally.
    E.g. instead of directing output to console one could specify
    a file or use different streams for errors, warnings and content
    related output etc.
    '''
    __metaclass__ = ABCMeta

    @abstractmethod
    def prints(self, output):
        '''
        Direct the output to the internally defined printing stream.
        '''
        pass


class ConsolePrinter(Printer):
    '''
    Redirects to console, equivalent to the print statement

    >>> print (output)
    '''
    def prints(self, output):
        '''
        Directs the output to console.
        '''
        print (output)


class SilentPrinter(Printer):
    '''
    Mutes output.
    '''
    def prints(self, output):
        '''
        Accepts output and does nothing.
        '''
        pass


class AccumulatorPrinter(Printer):
    '''
    Accumulates all calls to prints in a list 'log'
    '''
    def __init__(self, *args, **kwargs):
        self.log = []

    def prints(self, output):
        '''Stores warnings in list log'''
        self.log.append(output)
