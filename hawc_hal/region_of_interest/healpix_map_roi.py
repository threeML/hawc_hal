import numpy as np
import astropy.units as u
import healpy as hp

from threeML.exceptions.custom_exceptions import custom_warnings
from astromodels.core.sky_direction import SkyDirection

from hawc_hal.region_of_interest.healpix_roi_base import HealpixROIBase
from hawc_hal.region_of_interest.healpix_cone_roi import HealpixConeROI, _get_radians
from ..flat_sky_projection import FlatSkyProjection


class HealpixMapROI(HealpixROIBase):

    def __init__(self, model_radius, data_radius, roimap=None, roifile=None, threshold=0.5, *args, **kwargs):
        """
        A cone Region of Interest defined by a healpix map (can be read from a fits file).
        User needs to supply a cone region (center and radius) defining the plane projection for the model map.

        Examples:

            Model map centered on (R.A., Dec) = (1.23, 4.56) in J2000 ICRS coordinate system,
            with a radius of 5 degrees, ROI defined in healpix map in fitsfile:

            > roi = HealpixMapROI(5.0, ra=1.23, dec=4.56, file = "myROI.fits" )

            Model map centered on (L, B) = (1.23, 4.56) (Galactic coordiantes)
            with a radius of 30 arcmin, ROI defined on-the-fly in healpix map:

            > roi = HealpixMapROI(30.0 * u.arcmin, l=1.23, dec=4.56, map = my_roi)

        :param model_radius: radius of the model cone. Either an astropy.Quantity instance, or a float, in which case it
        is assumed to be the radius in degrees
        :param data_radius: radius used for displaying maps. Either an astropy.Quantity instance, or a float, in which case it
        is assumed to be the radius in degrees
        :param map: healpix map containing the ROI.
        :param file: fits file containing a healpix map with the ROI.
        :param threshold: value below which pixels in the map will be set inactive (=not in ROI).
        :param args: arguments for the SkyDirection class of astromodels
        :param kwargs: keywords for the SkyDirection class of astromodels
        """
 
        assert roifile is not None or roimap is not None, "Must supply either healpix map or fitsfile to create HealpixMapROI"

        self._center = SkyDirection(*args, **kwargs)

        self._model_radius_radians = _get_radians(model_radius)

        self._data_radius_radians = _get_radians(data_radius)

        self._threshold = threshold

        self.read_map(roimap=roimap, roifile=roifile)


    def read_map(self, roimap=None, roifile=None):
        assert roifile is not None or roimap is  not None, \
                    "Must supply either healpix map or fits file"
 
        if roimap is not None:
            roimap = roimap
            self._filename = None

        elif roifile is not None:
            self._filename = roifile
            roimap =  hp.fitsfunc.read_map(self._filename)

        self._roimaps = {}

        self._original_nside = hp.npix2nside(roimap.shape[0])
        self._roimaps[self._original_nside] = roimap
        
        self.check_roi_inside_model()


    def check_roi_inside_model(self):

        active_pixels = self.active_pixels(self._original_nside)

        radius = np.rad2deg(self._model_radius_radians)
        ra, dec = self.ra_dec_center
        temp_roi =  HealpixConeROI(data_radius = radius , model_radius=radius, ra=ra, dec=dec)

        model_pixels = temp_roi.active_pixels( self._original_nside )

        if not all(p in model_pixels for p in active_pixels):
            custom_warnings.warn("Some pixels inside your ROI are not contained in the model map.")

    def to_dict(self):

        ra, dec = self.ra_dec_center

        s = {'ROI type': type(self).__name__.split(".")[-1],
             'ra': ra,
             'dec': dec,
             'model_radius_deg': np.rad2deg(self._model_radius_radians),
             'data_radius_deg': np.rad2deg(self._data_radius_radians),
             'roimap': self._roimaps[self._original_nside],
             'threshold': self._threshold,
             'roifile': self._filename }

        return s

    @classmethod
    def from_dict(cls, data):

        return cls(data['model_radius_deg'], data['data_radius_deg'], threshold=data['threshold'],
                   roimap=data['roimap'], ra=data['ra'],
                   dec=data['dec'], roifile=data['roifile'])

    def __str__(self):

        s = ("%s: Center (R.A., Dec) = (%.3f, %.3f), model radius: %.3f deg, display radius: %.3f deg, threshold = %.2f" %
              (type(self).__name__, self.ra_dec_center[0], self.ra_dec_center[1],
               self.model_radius.to(u.deg).value, self.data_radius.to(u.deg).value, self._threshold))

        if self._filename is not None: 
            s = "%s, data ROI from %s" % (s, self._filename)
            
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
    def data_radius(self):
        return self._data_radius_radians * u.rad

    @property
    def threshold(self):
        return self._threshold

    def _get_ra_dec(self):

        lon, lat = self._center.get_ra(), self._center.get_dec()

        return lon, lat

    def _active_pixels(self, nside, ordering):

        if not nside in self._roimaps:
          self._roimaps[nside] = hp.ud_grade(self._roimaps[self._original_nside], nside_out=nside)

        pixels_inside_roi = np.where(self._roimaps[nside] >= self._threshold)[0]

        return pixels_inside_roi

    def get_flat_sky_projection(self, pixel_size_deg):

        # Decide side for image

        # Compute number of pixels, making sure it is going to be even (by approximating up)
        npix_per_side = 2 * int(np.ceil(np.rad2deg(self._model_radius_radians) / pixel_size_deg))

        # Get lon, lat of center
        ra, dec = self._get_ra_dec()

        # This gets a list of all RA, Decs for an AIT-projected image of npix_per_size x npix_per_side
        flat_sky_proj = FlatSkyProjection(ra, dec, pixel_size_deg, npix_per_side, npix_per_side)

        return flat_sky_proj

