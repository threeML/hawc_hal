"""
The code contained here is adapted from the "reproject" project under the BSD-3.0 license.

It has been adapted to partial healpix map (for speed)
"""

import healpy as hp
import numpy as np
import six

from scipy.ndimage import map_coordinates

from astropy.coordinates import Galactic, ICRS, BaseCoordinateFrame, frame_transform_graph
from astropy import units as u
from astropy.coordinates import UnitSphericalRepresentation
from astropy.wcs.utils import wcs_to_celestial_frame
from astropy.wcs import WCS

from hawc_hal.special_values import UNSEEN
from hawc_hal.interpolation.fast_linear_interpolator import FastLinearInterpolatorIrregularGrid
from hawc_hal.interpolation.fast_linear_interpolator import  FastBilinearInterpolation


ORDER = {}
ORDER['nearest-neighbor'] = 0
ORDER['bilinear'] = 1
ORDER['biquadratic'] = 2
ORDER['bicubic'] = 3


FRAMES = {
    'g': Galactic(),
    'c': ICRS()
}


def parse_coord_system(system):

    if isinstance(system, BaseCoordinateFrame):

        return system

    elif isinstance(system, six.string_types):

        system = system.lower()

        if system == 'e':
            raise ValueError("Ecliptic coordinate frame not yet supported")

        elif system in FRAMES:

            return FRAMES[system]

        else:

            system_new = frame_transform_graph.lookup_name(system)

            if system_new is None:

                raise ValueError("Could not determine frame for system={0}".format(system))

            else:

                return system_new()


def convert_world_coordinates(lon_in, lat_in, wcs_in, wcs_out):
    """
    Convert longitude/latitude coordinates from an input frame to an output
    frame.
    Parameters
    ----------
    lon_in, lat_in : `~numpy.ndarray`
        The longitude and latitude to convert
    wcs_in, wcs_out : tuple or `~astropy.wcs.WCS`
        The input and output frames, which can be passed either as a tuple of
        ``(frame, lon_unit, lat_unit)`` or as a `~astropy.wcs.WCS` instance.
    Returns
    -------
    lon_out, lat_out : `~numpy.ndarray`
        The output longitude and latitude
    """

    if isinstance(wcs_in, WCS):
        # Extract the celestial component of the WCS in (lon, lat) order
        wcs_in = wcs_in.celestial
        frame_in = wcs_to_celestial_frame(wcs_in)
        lon_in_unit = u.Unit(wcs_in.wcs.cunit[0])
        lat_in_unit = u.Unit(wcs_in.wcs.cunit[1])
    else:
        frame_in, lon_in_unit, lat_in_unit = wcs_in

    if isinstance(wcs_out, WCS):
        # Extract the celestial component of the WCS in (lon, lat) order
        wcs_out = wcs_out.celestial
        frame_out = wcs_to_celestial_frame(wcs_out)
        lon_out_unit = u.Unit(wcs_out.wcs.cunit[0])
        lat_out_unit = u.Unit(wcs_out.wcs.cunit[1])
    else:
        frame_out, lon_out_unit, lat_out_unit = wcs_out

    data = UnitSphericalRepresentation(lon_in * lon_in_unit,
                                       lat_in * lat_in_unit)

    coords_in = frame_in.realize_frame(data)
    coords_out = coords_in.transform_to(frame_out)

    lon_out = coords_out.represent_as('unitspherical').lon.to(lon_out_unit).value
    lat_out = coords_out.represent_as('unitspherical').lat.to(lat_out_unit).value

    return lon_out, lat_out


def image_to_healpix(data, wcs_in, coord_system_out,
                     nside, pixels_id, order='bilinear', nested=False,
                     fill_value=UNSEEN, pixels_to_be_zeroed=None, full=False):

    npix = hp.nside2npix(nside)

    # Look up lon, lat of pixels in output system and convert colatitude theta
    # and longitude phi to longitude and latitude.
    theta, phi = hp.pix2ang(nside, pixels_id, nested)

    lon_out = np.degrees(phi)
    lat_out = 90. - np.degrees(theta)

    # Convert between celestial coordinates
    coord_system_out = parse_coord_system(coord_system_out)

    with np.errstate(invalid='ignore'):
        lon_in, lat_in = convert_world_coordinates(lon_out, lat_out, (coord_system_out, u.deg, u.deg), wcs_in)

    # Look up pixels in input system
    yinds, xinds = wcs_in.wcs_world2pix(lon_in, lat_in, 0)

    # Interpolate

    if isinstance(order, six.string_types):
        order = ORDER[order]

    healpix_data_ = map_coordinates(data, [xinds, yinds],
                                    order=order,
                                    mode='constant', cval=fill_value)

    if not full:

        # Return partial map
        return healpix_data_

    else:

        # Return full healpix map

        healpix_data = np.full(npix, fill_value)

        healpix_data[pixels_id] = healpix_data_

        if pixels_to_be_zeroed is not None:

            healpix_data[pixels_to_be_zeroed] = np.where(np.isnan(healpix_data[pixels_to_be_zeroed]),
                                                         0.0,
                                                         healpix_data[pixels_to_be_zeroed])

        return healpix_data


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
        coord_system_out = parse_coord_system(coord_system_out)

        with np.errstate(invalid='ignore'):
            lon_in, lat_in = convert_world_coordinates(lon_out, lat_out, (coord_system_out, u.deg, u.deg), wcs_in)

        # Look up pixels in input system
        yinds, xinds = wcs_in.wcs_world2pix(lon_in, lat_in, 0)

        self._coords = [xinds, yinds]

        # Interpolate

        if isinstance(order, six.string_types):
            order = ORDER[order]

        self._order = order

        #self._interpolator = FastLinearInterpolatorIrregularGrid(input_shape, np.asarray(self._coords).T)
        self._interpolator = FastBilinearInterpolation(input_shape, self._coords)

    def __call__(self, data, fill_value=UNSEEN):

        # healpix_data = map_coordinates(data, self._coords,
        #                                 order=self._order,
        #                                 mode='constant', cval=fill_value)

        healpix_data = self._interpolator(data)

        return healpix_data
