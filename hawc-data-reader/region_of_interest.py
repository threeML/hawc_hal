import numpy as np
import healpy as hp

from astromodels.core.sky_direction import SkyDirection

import astropy.units as u


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

        assert system in [_EQUATORIAL, _GALACTIC], "The reference system must be '%s' or '%s'" % (_EQUATORIAL,
                                                                                                  _GALACTIC)

        assert ordering in [_RING, _NESTED], "Could not understand ordering %s. Must be %s or %s" % (ordering,
                                                                                                     _RING,
                                                                                                     _NESTED)

        return self._active_pixels(nside, system, ordering)

    # This is supposed to be overridden by child classes
    def _active_pixels(self, nside, system, ordering):

        raise NotImplementedError("You need to implement this")


class HealpixConeROI(HealpixROIBase):

    def __init__(self, radius, *args, **kwargs):
        """
        A cone Region of Interest defined by a center and a radius.

        Examples:

            ROI centered on (R.A., Dec) = (1.23, 4.56) in J2000 ICRS coordinate system, with a radius of 5 degrees:

            > roi = HealpixConeROI(5.0, ra=1.23, dec=4.56)

            ROI centered on (L, B) = (1.23, 4.56) (Galactic coordiantes) with a radius of 30 arcmin:

            > roi = HealpixConeROI(30.0 * u.arcmin, l=1.23, dec=4.56)

        :param radius: radius of the cone. Either an astropy.Quantity instance, or a float, in which case it is assumed
        to be the radius in degrees
        :param args: arguments for the SkyDirection class of astromodels
        :param kwargs: keywords for the SkyDirection class of astromodels
        """

        self._center = SkyDirection(*args, **kwargs)

        if isinstance(radius, u.Quantity):

            self._radius_radians = radius.to(u.rad).value

        else:

            self._radius_radians = (float(radius) * u.deg).to(u.rad).value

    @property
    def center(self):

        return self._center

    @property
    def radius(self):

        return self._radius_radians * u.rad

    def _active_pixels(self, nside, system, ordering):

        if system == _EQUATORIAL:

            lon, lat = self._center.get_ra(), self._center.get_dec()

        else:

            lon, lat = self._center.get_l(), self._center.get_b()

        theta = 0.5 * np.pi - np.deg2rad(lat)
        phi = np.deg2rad(lon)

        ipix = hp.ang2pix(nside, theta, phi)

        nest = ordering is _NESTED

        vec = hp.pix2vec(nside, ipix, nest=nest)

        pixels_inside_cone = hp.query_disc(nside, vec, self._radius_radians, inclusive=False, nest=nest)

        return pixels_inside_cone