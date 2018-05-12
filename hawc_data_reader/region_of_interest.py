import healpy as hp
from healpix_utils import radec_to_vec

from astromodels.core.sky_direction import SkyDirection

import astropy.units as u
import numpy as np

from flat_sky_projection import FlatSkyProjection

_EQUATORIAL = 'equatorial'
_GALACTIC = 'galactic'
_RING = 'RING'
_NESTED = 'NESTED'


class HealpixROIBase(object):

    def active_pixels(self, nside, system=_EQUATORIAL, ordering=_RING):
        """
        Returns the non-zero elements, i.e., the pixels selected according to this Region Of Interest

        :param nside: the NSIDE of the healpix map
        :param system: the system of the Healpix map, either 'equatorial' or 'galactic' (default: equatorial)
        :param ordering: numbering scheme for Healpix. Either RING or NESTED (default: RING)
        :return: an array of pixels IDs (in healpix RING numbering scheme)
        """

        # Let's transform to lower case (so Equatorial will work, as well as EQuaTorial, or whatever)
        system = system.lower()

        assert system == _EQUATORIAL, "%s reference system not supported" % system

        assert ordering in [_RING, _NESTED], "Could not understand ordering %s. Must be %s or %s" % (ordering,
                                                                                                     _RING,
                                                                                                     _NESTED)

        return self._active_pixels(nside, ordering)

    # This is supposed to be overridden by child classes
    def _active_pixels(self, nside, ordering):

        raise NotImplementedError("You need to implement this")

    def active_pixels_for_convolution(self, nside, ordering=_RING):
        """
        Returns the non-zero elements, i.e., the pixels selected according to this Region Of Interest to be used
        for the convolution of extended sources

        :param nside: the NSIDE of the healpix map
        :param ordering: numbering scheme for Healpix. Either RING or NESTED (default: RING)
        :return: an array of pixels IDs (in healpix RING numbering scheme)
        """

        assert ordering in [_RING, _NESTED], "Could not understand ordering %s. Must be %s or %s" % (ordering,
                                                                                                     _RING,
                                                                                                     _NESTED)

        return self._active_pixels_for_convolution(nside, ordering)

    # This is supposed to be overridden by child classes
    def _active_pixels_for_convolution(self, nside, ordering):

        raise NotImplementedError("You need to implement this")

    def coordinates_for_convolution(self, pixel_size):
        """
        Returns a flat image and its WCS covering the convolution ROI (slightly larger than the actual ROI to cover
        leakage from the PSF)

        :param pixel_size: size of the pixel in the WCS image (quantity or float, which is assumed to represent degrees)
        :param system: the system of the Healpix map, either 'equatorial' or 'galactic' (default: equatorial)
        :return: (nhpix, nwpix, all RAs, all Decs)
        """

        pixel_size_rad = _get_radians(pixel_size)

        return self._coordinates_for_convolution(pixel_size_rad)

    def _coordinates_for_convolution(self, pixel_size_rad):

        raise NotImplementedError("You need to implement this")

    def display(self):

        raise NotImplementedError("You need to implement this")


def _get_radians(my_angle):

    if isinstance(my_angle, u.Quantity):

        my_angle_radians = my_angle.to(u.rad).value

    else:

        my_angle_radians = np.deg2rad(my_angle)

    return my_angle_radians


class HealpixConeROI(HealpixROIBase):

    def __init__(self, data_radius, model_radius, *args, **kwargs):
        """
        A cone Region of Interest defined by a center and a radius.

        Examples:

            ROI centered on (R.A., Dec) = (1.23, 4.56) in J2000 ICRS coordinate system, with a radius of 5 degrees:

            > roi = HealpixConeROI(5.0, ra=1.23, dec=4.56)

            ROI centered on (L, B) = (1.23, 4.56) (Galactic coordiantes) with a radius of 30 arcmin:

            > roi = HealpixConeROI(30.0 * u.arcmin, l=1.23, dec=4.56)

        :param data_radius: radius of the cone. Either an astropy.Quantity instance, or a float, in which case it is assumed
        to be the radius in degrees
        :param model_radius: radius of the model cone. Either an astropy.Quantity instance, or a float, in which case it
        is assumed to be the radius in degrees
        :param args: arguments for the SkyDirection class of astromodels
        :param kwargs: keywords for the SkyDirection class of astromodels
        """

        self._center = SkyDirection(*args, **kwargs)

        self._data_radius_radians = _get_radians(data_radius)
        self._model_radius_radians = _get_radians(model_radius)

        self._convolution_coordinates_cache = {}

    def display(self):

        print("%s: Center (R.A., Dec) = (%.3f, %.3f), data radius = %.3f deg, model radius: %.3f deg" %
              (type(self).__name__, self.ra_dec_center[0], self.ra_dec_center[1],
               self.data_radius.to(u.deg).value, self.model_radius.to(u.deg).value))

    @property
    def ra_dec_center(self):

        return self._get_ra_dec()

    @property
    def data_radius(self):

        return self._data_radius_radians * u.rad

    @property
    def model_radius(self):
        return self._model_radius_radians * u.rad

    def _get_ra_dec(self):

        lon, lat = self._center.get_ra(), self._center.get_dec()

        return lon, lat

    def _get_healpix_vec(self):

        lon, lat = self._get_ra_dec()

        vec = radec_to_vec(lon, lat)

        return vec

    def _active_pixels(self, nside, ordering):

        vec = self._get_healpix_vec()

        nest = ordering is _NESTED

        pixels_inside_cone = hp.query_disc(nside, vec, self._data_radius_radians, inclusive=False, nest=nest)

        return pixels_inside_cone

    def get_flat_sky_projection(self, pixel_size_deg):

        # Decide side for image

        # Compute number of pixels, making sure it is going to be even (by approximating up)
        npix_per_side = 2 * int(np.ceil(np.rad2deg(self._model_radius_radians) / pixel_size_deg))

        # Get lon, lat of center
        ra, dec = self._get_ra_dec()

        # This gets a list of all RA, Decs for an AIT-projected image of npix_per_size x npix_per_side
        flat_sky_proj = FlatSkyProjection(ra, dec, pixel_size_deg, npix_per_side, npix_per_side)

        return flat_sky_proj


