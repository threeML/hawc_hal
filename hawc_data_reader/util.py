import numpy as np


def cartesian(arrays, out=None):
    """
    Generate a cartesian product of input arrays.
    """

    # A re-entrant function

    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])

    # This happen only the first iteration
    if out is None:

        out = np.zeros([n, len(arrays)], dtype=dtype)

    m = n / arrays[0].size
    out[:,0] = np.repeat(arrays[0], m)

    if arrays[1:]:

        cartesian(arrays[1:], out=out[0:m, 1:])

        for j in range(1, arrays[0].size):

            out[j*m:(j+1)*m, 1:] = out[0:m, 1:]

    return out