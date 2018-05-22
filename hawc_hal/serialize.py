import os
from pandas import HDFStore


# This object is to decouple the serialization from any particular implementation. At the moment we use HDF5 through
# pandas but this might change in the future. Without changing the external API, only changes here will be necessary.
class Serialization(object):

    def __init__(self, filename):

        self._filename = filename

    def __enter__(self):

        self._store = HDFStore(self._filename, complib='blosc', complevel=9)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):

        self._store.close()

    @property
    def keys(self):

        return self._store.keys()

    def store_pandas_object(self, name, object, **metadata):

        self._store.put(name, object)

        self._store.get_storer(name).attrs.metadata = metadata

    def retrieve_pandas_object(self, name):

        # Get the metadata
        metadata = self._store.get_storer(name).attrs.metadata

        # Get the object
        obj = self._store[name]

        return obj, metadata
