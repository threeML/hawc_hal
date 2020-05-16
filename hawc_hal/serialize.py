from builtins import object
from pandas import HDFStore


# This object is to decouple the serialization from any particular implementation. At the moment we use HDF5 through
# pandas but this might change in the future. Without changing the external API, only changes here will be necessary.
class Serialization(object):

    def __init__(self, filename, mode='r', compress=True):

        self._filename = filename
        self._compress = compress
        self._mode = mode

    def __enter__(self):

        if self._compress:

            self._store = HDFStore(self._filename, complib='blosc:lz4', complevel=9, mode=self._mode)

        else:  # pragma: no cover

            self._store = HDFStore(self._filename, mode=self._mode)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):

        self._store.close()

    @property
    def keys(self):

        return list(self._store.keys())

    def store_pandas_object(self, path, obj, **metadata):

        self._store.put(path, obj, format='fixed')

        self._store.get_storer(path).attrs.metadata = metadata

    def retrieve_pandas_object(self, path):

        # Get the metadata
        metadata = self._store.get_storer(path).attrs.metadata

        # Get the object
        obj = self._store.get(path)

        return obj, metadata
