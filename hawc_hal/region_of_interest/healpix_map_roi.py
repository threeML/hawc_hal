import numpy as np
import astropy.units as u
import healpy as hp
from healpix_roi_base import HealpixROIBase, _RING, _NESTED
from astropy.io import fits

from astromodels.core.sky_direction import SkyDirection

from ..healpix_handling import radec_to_vec
from ..flat_sky_projection import FlatSkyProjection


def _get_radians(my_angle):

    if isinstance(my_angle, u.Quantity):

        my_angle_radians = my_angle.to(u.rad).value

    else:

        my_angle_radians = np.deg2rad(my_angle)

    return my_angle_radians


class HealpixMapROI(HealpixROIBase):

    def __init__(self, model_radius, map=None, file = None, *args, **kwargs):
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
 
        assert file is not kwargs or map is  not None, "Must supply either healpix map or fits file to create HealpixMapROI"


        self._center = SkyDirection(*args, **kwargs)

        self._model_radius_radians = _get_radians(model_radius)
        
        if file is not None:
            self._filename = file
            map =  hp.fitsfunc.read_map(self._filename)
        elif map is not None:
            map = map
            self._filename = None

        self._maps = {}

        self._original_nside = hp.npix2nside( map.shape[0] )
        self._maps[self._original_nside] = map

        self._threshold  = 0.5
        if "threshold" in kwargs:
            self._threshold = kwargs["threshold"]


    def to_dict(self):

        ra, dec = self.ra_dec_center

        s = {'ROI type': type(self).__name__.split(".")[-1],
             'ra': ra,
             'dec': dec,
             'model_radius_deg': np.rad2deg(self._model_radius_radians),
             'map': self._map,
             'threshold': self._threshold,
             'file': self._filename }

        return s

    @classmethod
    def from_dict(cls, data):

        return cls(data['data_radius_deg'], threshold = data['threshold'], map = data['map'], ra=data['ra'], dec=data['dec'], file=data['file'])

    def __str__(self):

        s = ("%s: Center (R.A., Dec) = (%.3f, %.3f), model radius: %.3f deg, threshold = %.2f" %
              (type(self).__name__, self.ra_dec_center[0], self.ra_dec_center[1],
               self.model_radius.to(u.deg).value, self._threshold))

        if self._filename is not None: 
            s  =  "%s,  data ROI from %s" %  (s, self._filename)
            
        return s

    def display(self):

        print(self)

    @property
    def ra_dec_center(self):

        return self._get_ra_dec()

    @property
    def model_radius(self):
        return self._model_radius_radians * u.rad

    @property
    def threshold(self):
        return self._threshold

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
            
        if not nside in self._maps:
          self._maps[nside] = hp.ud_grade(self._maps[self._original_nside], nside_out=nside)

        pixels_inside_map = np.where( self._maps[ nside ] >= self._threshold )[0]

        return pixels_inside_map

    def get_flat_sky_projection(self, pixel_size_deg):

        # Decide side for image

        # Compute number of pixels, making sure it is going to be even (by approximating up)
        npix_per_side = 2 * int(np.ceil(np.rad2deg(self._model_radius_radians) / pixel_size_deg))

        # Get lon, lat of center
        ra, dec = self._get_ra_dec()

        # This gets a list of all RA, Decs for an AIT-projected image of npix_per_size x npix_per_side
        flat_sky_proj = FlatSkyProjection(ra, dec, pixel_size_deg, npix_per_side, npix_per_side)

        return flat_sky_proj

