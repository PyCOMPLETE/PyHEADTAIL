"""
@class WakeField
@author Hannes Bartosik, Kevin Li, Giovanni Rumolo, Michael Schenk
@date March 2014
@brief Definition of a WakeField as a collection of WakeKick objects.
@copyright CERN
"""
from __future__ import division
import numpy as np


class WakeField(object):
    """
    A WakeField is defined by elementary WakeKick objects that may
    originate from different WakeSource objects. Usually, there is
    no need for the user to define more than one instance of the
    WakeField class in a simulation - except if one wants to use
    different slicing configurations (one WakeField object is allowed
    to have exactly one slicing configuration, i.e. one instance of the
    Slicer class).
    """

    def __init__(self, slicer, *wake_sources):
        """
        Accepts a list of WakeSource objects. Each WakeSource object
        knows how to generate its corresponding WakeKick objects. The
        collection of all the WakeKick objects of each of the passed
        WakeSource objects defines the WakeField.
        When instantiating the WakeField object, the WakeKick objects
        for each WakeSource defined in wake_sources are requested. The
        returned WakeKick lists are all stored in the
        WakeField.wake_kicks list. The WakeField itself forgets about
        the origin (WakeSource) of the kicks as soon as they have been
        generated.
        Exactly one instance of the Slicer class must be passed to the
        WakeField constructor. All the wake field components (kicks)
        hence use the same slicing and thus the same slice_set to
        calculate the strength of the kicks.
        """
        self.slicer = slicer

        self.wake_kicks = []
        for source in wake_sources:
            kicks = source.get_wake_kicks(self.slicer.mode)
            self.wake_kicks.extend(kicks)

    def track(self, bunch):
        """
        Calls the WakeKick.apply(bunch, slice_set) method of each of the
        WakeKick objects stored in self.wake_kicks. A slice_set is
        necessary to perform this operation. It is requested from the
        bunch (instance of the Particles class) using the
        Particles.get_slices(self.slicer) method, where self.slicer is
        the instance of the Slicer class used for this particluar
        WakeField object. A slice_set is returned according to the
        self.slicer configuration.
        """
        slice_set = bunch.get_slices(self.slicer)

        for kick in self.wake_kicks:
            kick.apply(bunch, slice_set)
