from hawc_hal.healpix_handling.flat_sky_to_healpix import _parse_coord_system, _convert_world_coordinates, ORDER
from hawc_hal.special_values import UNSEEN
import healpy as hp
import numpy as np
import six
from scipy.ndimage import map_coordinates

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
    coord_system_out = _parse_coord_system(coord_system_out)

    with np.errstate(invalid='ignore'):
        lon_in, lat_in = _convert_world_coordinates(lon_out, lat_out, (coord_system_out, u.deg, u.deg), wcs_in)

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
