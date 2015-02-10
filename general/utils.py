'''
@authors: Kevin Li, Adrian Oeftiger
@date:    10/02/2015

Provide useful conceptual classes and logics for PyHEADTAIL.
'''


class ListProxy(object):
    """Is a list of object attributes. Accessing ListProxy entries
    evaluates the object attributes each time it is accessed,
    i.e. this list "proxies" the object attributes.
    """
    def __init__(self, list_of_objects, attr_name):
        """Provide a list of object instances and a name of a commonly
        shared attribute that should be proxied by this ListProxy
        instance.
        """
        self._list_of_objects = list_of_objects
        self._attr_name = attr_name

    def __getitem__(self, index):
        return getattr(self._list_of_objects[index], self._attr_name)

    def __setitem__(self, index, value):
        setattr(self._list_of_objects[index], self._attr_name, value)

    def __repr__(self):
        return repr(list(self))

    def __len__(self):
        return len(self._list_of_objects)
