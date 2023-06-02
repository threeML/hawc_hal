from numba import jit
import numpy as np


# A fast way to compute angular distances
@jit("float64[:](float64, float64, float64[:], float64[:])", nopython=True)
def sphere_dist(ra1, dec1, ra2, dec2):  # pragma: no cover
    """
    Compute angular distance using the Haversine formula. Use this one when you know you will never ask for points at
    their antipodes. If this is not the case, use the angular_distance function which is slower, but works also for
    antipodes.

    :param ra1: first RA (deg)
    :param dec1: first Dec (deg)
    :param ra2: second RA (deg)
    :param dec2: second Dec (deg)

    :returns: angular separation distance (deg)
    """

    deg2rad = 0.017453292519943295
    rad2deg = 57.29577951308232

    lon1 = ra1 * deg2rad
    lat1 = dec1 * deg2rad
    lon2 = ra2 * deg2rad
    lat2 = dec2 * deg2rad
    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon /2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))

    return c * rad2deg
