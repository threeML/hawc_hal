# import pandas as pd
# from pandas.core.sparse.array import IntIndex
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

    def as_dense(self):

        return _not_implemented()

    def as_partial(self):

        return _not_implemented()


class SparseHealpix(HealpixWrapperBase):

    def __init__(self, partial_map, pixels_ids, nside, fill_value=UNSEEN):

        self._partial_map = partial_map
        self._pixels_ids = pixels_ids
        self._fill_value = fill_value

        super(SparseHealpix, self).__init__(sparse=True, nside=nside)

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


# class SparseHealpixOld(HealpixWrapperBase):
#     """
#     A sparse healpix map, where elements (likely, many elements) are empty. Storing it like this preserves *a lot*
#     of memory. Use copy=True to copy the array, otherwise the sparse array will only be a view.
#     """
#
#     def __init__(self, healpix_array, copy=True, fill_value=UNSEEN, non_null_elements=None):
#
#         if non_null_elements is not None:
#
#             # NOTE: non_null_elements contains the actual indexes, it is NOT a boolean mask. In other words,
#             # non_null_elements.shape != healpix_array.shape unless all pixels are non null (in which case
#             # there is no point in using a sparse healpix representation)
#
#             not_nans = healpix_array[non_null_elements]
#             index = IntIndex(healpix_array.shape[0], non_null_elements)
#
#             self._sparse_map = pd.SparseArray(not_nans, sparse_index=index)
#
#         else:
#
#             self._sparse_map = pd.SparseArray(healpix_array, kind='block', fill_value=fill_value, copy=copy)
#
#         super(SparseHealpixOld, self).__init__(healpix_array, sparse=True)
#
#     def as_sparse(self):
#         """
#         Returns the sparse representation of the map. Unused pixels have a value of UNSEEN
#
#         :return: sparse representation (note: the sparse representation is read-only)
#         """
#         return self._sparse_map
#
#     def as_dense(self):
#         """
#         Returns the dense (i.e., full sky) representation of the map. Note that this means unwrapping the map,
#         and the memory usage increases a lot.
#
#         :return: the dense map, suitable for use with healpy routine (among other uses)
#         """
#
#         return self._sparse_map.to_dense()
#
#     def as_partial(self):
#
#         return np.array(self.as_sparse())


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
