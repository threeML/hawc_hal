from builtins import object
import numpy as np
import healpy as hp
import pandas as pd
from ..special_values import UNSEEN


def _not_implemented():  # pragma: no cover

    raise RuntimeError("You cannot use the base class. Use the derived classes.")


class HealpixWrapperBase(object):
    """
    A class which wrap a numpy array containing an healpix map, in order to expose always the same interface
    independently of whether the underlying map is sparse or dense
    """

    def __init__(self, sparse, nside):

        self._nside = int(nside)
        self._npix = hp.nside2npix(self._nside)
        self._pixel_area = hp.nside2pixarea(self._nside, degrees=True)
        self._sparse = bool(sparse)

    @property
    def is_sparse(self):
        return self._sparse

    @property
    def nside(self):
        return self._nside

    @property
    def npix(self):
        """
        :return: total number of pixels for this nside. Note that mymap.npix is equivalent to
        healpy.nside2npix(mymap.nside)
        """
        return self._npix

    @property
    def pixel_area(self):
        """
        :return: area (solid angle) of the healpix pixel in sq. degrees
        """
        return self._pixel_area

    def as_dense(self):  # pragma: no cover

        return _not_implemented()

    def as_partial(self):  # pragma: no cover

        return _not_implemented()

    def to_pandas(self):
        """
        Returns a pandas Series with the dense representation of the data

        :return: pd.Series, type
        """

        return pd.Series(self.as_partial())


class SparseHealpix(HealpixWrapperBase):

    def __init__(self, partial_map, pixels_ids, nside, fill_value=UNSEEN):

        self._partial_map = partial_map
        self._pixels_ids = pixels_ids
        self._fill_value = fill_value

        super(SparseHealpix, self).__init__(sparse=True, nside=nside)

    def __add__(self, other_map):

        # Make sure they have the same pixels
        assert np.array_equal(self._pixels_ids, other_map.pixels_ids)

        added = self.as_partial() + other_map.as_partial()

        sparse_added = SparseHealpix(added, self._pixels_ids, self.nside)
        
        return sparse_added

    def __sub__(self, other_map):

        # Make sure they have the same pixels
        assert np.array_equal(self._pixels_ids, other_map.pixels_ids)

        subtraction = self.as_partial() - other_map.as_partial()

        sparse_subtracted = SparseHealpix(subtraction, self._pixels_ids, self.nside)

        return sparse_subtracted

    def as_dense(self):
        """
        Returns the dense (i.e., full sky) representation of the map. Note that this means unwrapping the map,
        and the memory usage increases a lot.

        :return: the dense map, suitable for use with healpy routine (among other uses)
        """

        # Make the full Healpix map
        new_map = np.full(self.npix, self._fill_value)

        # Assign the active pixels their values
        new_map[self._pixels_ids] = self._partial_map

        return new_map

    def as_partial(self):

        return self._partial_map

    def set_new_values(self, new_values):

        assert new_values.shape == self._partial_map.shape

        self._partial_map[:] = new_values

    @property
    def pixels_ids(self):
        return self._pixels_ids



class DenseHealpix(HealpixWrapperBase):
    """
    A dense (fullsky) healpix map. In this case partial and complete are the same map.

    """

    def __init__(self, healpix_array):

        self._dense_map = healpix_array

        super(DenseHealpix, self).__init__(nside=hp.npix2nside(healpix_array.shape[0]), sparse=False)

    def as_dense(self):
        """
        Returns the complete (i.e., full sky) representation of the map. Since this is a dense map, this is identical
        to the input map

        :return: the complete map, suitable for use with healpy routine (among other uses)
        """

        return self._dense_map

    def as_partial(self):

        return self._dense_map

    def set_new_values(self, new_values):

        assert new_values.shape == self._dense_map.shape

        self._dense_map[:] = new_values

