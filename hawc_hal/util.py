from __future__ import division
from builtins import range
from past.utils import old_div
import numpy as np


def cartesian(arrays):
    return np.dstack(
        np.meshgrid(*arrays, indexing='ij')
        ).reshape(-1, len(arrays))


def cartesian_(arrays, out=None):
    """
    Generate a cartesian product of input arrays.
    """

    # A re-entrant function

    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])

    # This happen only the first iteration
    if out is None:

        out = np.zeros([n, len(arrays)], dtype=dtype)

    m = old_div(n, arrays[0].size)
    out[:,0] = np.repeat(arrays[0], m)

    if arrays[1:]:

        cartesian_(arrays[1:], out=out[0:m, 1:])

        for j in range(1, arrays[0].size):

            out[j*m:(j+1)*m, 1:] = out[0:m, 1:]

    return out


def ra_to_longitude(ra):

    if ra > 180.0:

        longitude = -180 + (ra - 180.0)

    else:

        longitude = ra

    return longitude
