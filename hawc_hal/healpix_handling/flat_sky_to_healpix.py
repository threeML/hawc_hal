from builtins import object
import healpy as hp
import numpy as np
import six

from scipy.ndimage import map_coordinates

from astropy.coordinates import Galactic, ICRS
from astropy import units as u
from astropy.coordinates import UnitSphericalRepresentation
from astropy.wcs.utils import wcs_to_celestial_frame

from ..special_values import UNSEEN
from ..interpolation import FastBilinearInterpolation


ORDER = {}
ORDER['nearest-neighbor'] = 0
ORDER['bilinear'] = 1
ORDER['biquadratic'] = 2
ORDER['bicubic'] = 3


COORDSYS = {
    'g': Galactic(),
    'c': ICRS(),
    'icrs': ICRS(),
}


def _parse_coord_system(system):

    try:

        return COORDSYS[system.lower()]

    except KeyError:  # pragma: no cover

        raise ValueError("Coordinate system %s is not known" % system)


def _convert_world_coordinates(lon_in, lat_in, wcs_in, wcs_out):

    frame_in, lon_in_unit, lat_in_unit = wcs_in

    wcs_out = wcs_out.celestial
    frame_out = wcs_to_celestial_frame(wcs_out)
    lon_out_unit = u.Unit(wcs_out.wcs.cunit[0])
    lat_out_unit = u.Unit(wcs_out.wcs.cunit[1])

    data = UnitSphericalRepresentation(lon_in * lon_in_unit,
                                       lat_in * lat_in_unit)

    coords_in = frame_in.realize_frame(data)
    coords_out = coords_in.transform_to(frame_out)

    lon_out = coords_out.represent_as('unitspherical').lon.to(lon_out_unit).value
    lat_out = coords_out.represent_as('unitspherical').lat.to(lat_out_unit).value

    return lon_out, lat_out


class FlatSkyToHealpixTransform(object):
    """
    A class to perform transformation from a flat sky projection to Healpix optimized to be used for the same
    transformation over and over again.

    The constructor will pre-compute all needed quantities for the transformation, and the __call__ method just applies
    the transformation. This avoids to re-compute the same quantities over and over again.
    """

    def __init__(self, wcs_in, coord_system_out, nside, pixels_id, input_shape, order='bilinear', nested=False):

        # Look up lon, lat of pixels in output system and convert colatitude theta
        # and longitude phi to longitude and latitude.
        theta, phi = hp.pix2ang(nside, pixels_id, nested)

        lon_out = np.degrees(phi)
        lat_out = 90. - np.degrees(theta)

        # Convert between celestial coordinates
        coord_system_out = _parse_coord_system(coord_system_out)

        with np.errstate(invalid='ignore'):
            lon_in, lat_in = _convert_world_coordinates(lon_out, lat_out, (coord_system_out, u.deg, u.deg), wcs_in)

        # Look up pixels in input system
        yinds, xinds = wcs_in.wcs_world2pix(lon_in, lat_in, 0)

        self._coords = [xinds, yinds]

        # Interpolate

        if isinstance(order, six.string_types):
            order = ORDER[order]

        self._order = order

        self._interpolator = FastBilinearInterpolation(input_shape, self._coords)

    def __call__(self, data, fill_value=UNSEEN):

        # healpix_data = map_coordinates(data, self._coords,
        #                                 order=self._order,
        #                                 mode='constant', cval=fill_value)

        healpix_data = self._interpolator(data)

        return healpix_data
