'''
@authors: Kevin Li, Adrian Oeftiger
@date:    10/02/2015

Provide useful conceptual classes and logics for PyHEADTAIL.
'''

from .element import Printing


class ListProxy(Printing):
    """Is a list of object attributes. Accessing ListProxy entries
    evaluates the object attributes each time it is accessed,
    i.e. this list "proxies" the object attributes.

    Attention:
    If accessed via slicing, e.g.
    >>> original = ListProxy(...)
    >>> part = original[2:5]
    then part created a new list of references, a new _list_of_objects.
    Consequently, any change to the direct contents of
    original._list_of_objects (such as popping or adding elements)
    is not reflected in part.
    """
    def __init__(self, list_of_objects, attr_name):
        """Provide a list of object instances and a name of a commonly
        shared attribute that should be proxied by this ListProxy
        instance.
        """
        self._list_of_objects = list_of_objects
        self._attr_name = attr_name

    def __getitem__(self, index):
        '''Return a ListProxy for slice arguments,
        otherwise return the requested value at the given index.
        '''
        subject = self._list_of_objects[index]
        try:
            return getattr(subject, self._attr_name)
        except AttributeError as e:
            self.warns("ListProxy: the reference list of the sliced " +
                       "part is detached. Any change in the list_of_objects " +
                       "will not be consistent between the sliced part and " +
                       "the original ListProxy.\n" +
                       "See help(PyHEADTAIL.utils.ListProxy) " +
                       "for further info. ")
            # this following line breaks the consistency (uniqueness)
            # between the _list_of_objects lists.
            # (subject is a new list w.r.t. self._list_of_objects)
            return ListProxy(subject, self._attr_name)

            # possible solution: make subject a view on the top list,
            # in analogy to the numpy view method

    def __setitem__(self, index, value):
        self._rewritable_setitem(index, value)

    def _rewritable_setitem(self, index, value):
        """This setter method may be overwritten."""
        subject = self._list_of_objects[index]
        try:
            setattr(subject, self._attr_name, value)
        except AttributeError as e:
            for obj, v in zip(subject, value):
                setattr(obj, self._attr_name, v)

    def __repr__(self):
        return repr(list(self))

    def __len__(self):
        return len(self._list_of_objects)

    def pop(self, index):
        '''Remove the object from the internal list and return the
        corresponding attribute, analogous to list.pop .'''
        return getattr(self._list_of_objects.pop(index), self._attr_name)


class MutableNumber(object):
    """Documentation for MutableNumber

    """
    def __init__(self, value):
        super(MutableNumber, self).__init__()
        self.value = value
