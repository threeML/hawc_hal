import pandas as pd
import numpy as np
import healpy as hp
from special_values import UNSEEN


def _not_implemented():

    raise RuntimeError("You cannot use the base class. Use the derived classes.")


class HealpixWrapperBase(object):
    """
    A class which wrap a numpy array containing an healpix map, in order to expose always the same interface
    independently of whether the underlying map is sparse or dense
    """

    def __init__(self, healpix_array, sparse):

        self._nside = hp.npix2nside(healpix_array.shape[0])
        self._npix = healpix_array.shape[0]
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
        :return: total number of pixels (if sparse, this is the *total* number of pixels as if the map was dense)
        """
        return self._npix

    @property
    def pixel_area(self):
        """
        :return: area (solid angle) of the healpix pixel in sq. degrees
        """
        return self._pixel_area

    def as_sparse(self):

        return _not_implemented()

    def as_dense(self):

        return _not_implemented()

    def as_array(self):

        return _not_implemented()


class SparseHealpix(HealpixWrapperBase):
    """
    A sparse healpix map, where elements (likely, many elements) are empty. Storing it like this preserves *a lot*
    of memory. Use copy=True to copy the array, otherwise the sparse array will only be a view.
    """

    def __init__(self, healpix_array, copy=False, fill_value=UNSEEN):

        self._sparse_map = pd.SparseArray(healpix_array, kind='block', fill_value=fill_value, copy=copy)

        # Compute indexes of pixels which are actually non-null
        self._active_pixels_ids = (healpix_array != UNSEEN)

        super(SparseHealpix, self).__init__(healpix_array, sparse=True)

        # This will keep a cached version of the array when needed
        self._cache = None

    def as_sparse(self):
        """
        Returns the sparse representation of the map. Unused pixels have a value of UNSEEN

        :return: sparse representation (note: the sparse representation is read-only)
        """
        return self._sparse_map

    def as_dense(self):
        """
        Returns the dense (i.e., full sky) representation of the map. Note that this means unwrapping the map,
        and the memory usage increases a lot.

        :return: the dense map, suitable for use with healpy routine (among other uses)
        """

        return self._sparse_map.to_dense()

    def as_array(self, cache=False):

        if cache:

            if self._cache is not None:

                return self._cache

            else:

                self._cache = np.array(self.as_sparse())

                return self._cache

        else:

            return np.array(self.as_sparse())


class DenseHealpix(HealpixWrapperBase):
    """
    A dense (fullsky) healpix map. In this case partial and complete are the same map.

    """

    def __init__(self, healpix_array):

        self._dense_map = healpix_array

        super(DenseHealpix, self).__init__(healpix_array, sparse=False)

    def as_sparse(self):
        """
        Since this is a dense map, the sparse map is identical to the dense map.

        :return: the healpix map as a np.ndarray
        """

        return self._dense_map

    def as_dense(self):
        """
        Returns the complete (i.e., full sky) representation of the map. Since this is a dense map, this is identical
        to the input map

        :return: the complete map, suitable for use with healpy routine (among other uses)
        """

        return self._dense_map

    def as_array(self):

        return self._dense_map
