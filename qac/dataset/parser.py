"""XML Reader utilities for the Stack Exchange dump.

Each file of the Stack Exchange dump (e.g, Posts.xml, Comments.xml, PostHistory.xml) can be read by
this module. See `dump schema <https://meta.stackexchange.com/a/2678>` for information on available
fields and properties.
"""
import xml.etree.ElementTree as etree


class dotdict(dict):
    """Dictionary wrapper that allows property access with dot notation."""
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __getattr__(self, name):
        return self[name]


class XMLParser(object):
    """XML parser class for reading an individual file of the Stack Exchange dump.

    The class is an iterable where each iteration corresponds to one row in the data file. The
    reader presumes the following XML file structure::

        <posts>
            <row attr1="val" attr2="val" />
            <row attr1="val" attr2="val" />
            ...
        </posts>
    """

    def __init__(self, file):
        self._context = iter(etree.iterparse(file, events=('start', 'end')))
        _, root = next(self._context)
        self._root = root

    def __iter__(self):
        return self

    def __next__(self):
        event, element = next(self._context)
        if element.tag != 'row':
            raise StopIteration

        if event == 'start':
            return self.__next__()

        if event == 'end' and element.tag == 'row':
            row = element.attrib
            # Remove intermediate elements from XML tree immediately as the SE dump is quite large
            self._root.clear()
            return dotdict(row)

        return {}
