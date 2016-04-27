"""The module containes the classes WakeKicks, WakeSrouces and WakeFields with
the latter being the top level instance.

In this description, WakeFields have a collection of WakeSources which have a
collection of WakeKick objects. Calling the WakeField.track method calls all
WakeKick.apply methods which update the respective macroparticle momenta
according to the particular rules of the specific WakeKick implementation.

A WakeField is defined as a composition of the elementary WakeKick objects (see
.wake_kicks module). They originate from WakeSources, e.g. a WakeTable,
Resonator and/or a ResistiveWall. The WakeField does not directly accept the
WakeKick objects, but takes a list of WakeSources first (can be of different
kinds), each of which knows how to generate its WakeKick objects via the
factory method WakeSource.get_wake_kicks(..). The collection of WakeKicks from
all the WakeSources define the WakeField and are the elementary objects that
are stored, (i.e. the WakeField forgets about the origin of the WakeKicks once
they have been created).

"""
from .. import __version__
from .. import Element, Printing
