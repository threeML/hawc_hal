from __future__ import division
from __future__ import absolute_import
from builtins import object
from past.utils import old_div
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_area
import numpy as np

from .util import cartesian
from hawc_hal.sphere_dist import sphere_dist


_fits_header = """
NAXIS   =                    2
NAXIS1  =                   %i
NAXIS2  =                   %i
CTYPE1  = 'RA---AIT'
CRPIX1  =                   %i
CRVAL1  =                   %s
CDELT1  =                  -%f
CUNIT1  = 'deg     '
CTYPE2  = 'DEC--AIT'
CRPIX2  =                   %i
CRVAL2  =                   %s
CDELT2  =                   %f
CUNIT2  = 'deg     '
COORDSYS= '%s'
"""


def _get_header(ra, dec, pixel_size, coordsys, h, w):

    assert 0 <= ra <= 360

    header = fits.Header.fromstring(
        _fits_header
        % (
            h,
            w,
            old_div(h, 2),
            ra,
            pixel_size,
            old_div(w, 2),
            dec,
            pixel_size,
            coordsys,
        ),
        sep="\n",
    )

    return header


def _get_all_ra_dec(input_wcs, h, w):

    # An array of all the possible permutation of (i,j) for i=0..999 and j=0..999

    xx = np.arange(0.5, h + 0.5, 1, dtype=np.int16)
    yy = np.arange(0.5, w + 0.5, 1, dtype=np.int16)

    _ij_grid = cartesian((xx, yy))

    # Convert pixel coordinates to world coordinates
    world = input_wcs.all_pix2world(_ij_grid, 0, ra_dec_order=True)

    return world[:, 0], world[:, 1]


class FlatSkyProjection(object):
    def __init__(self, ra_center, dec_center, pixel_size_deg, npix_height, npix_width):

        assert npix_height % 2 == 0, "Number of height pixels must be even"
        assert npix_width % 2 == 0, "Number of width pixels must be even"

        if isinstance(npix_height, float):

            assert npix_height.is_integer(), "This is a bug"

        if isinstance(npix_width, float):

            assert npix_width.is_integer(), "This is a bug"

        self._npix_height = int(npix_height)
        self._npix_width = int(npix_width)

        assert 0 <= ra_center <= 360.0, "Right Ascension must be between 0 and 360"
        assert -90.0 <= dec_center <= 90.0, "Declination must be between -90.0 and 90.0"

        self._ra_center = float(ra_center)
        self._dec_center = float(dec_center)

        self._pixel_size_deg = float(pixel_size_deg)

        # Build projection, i.e., a World Coordinate System object

        self._wcs = WCS(
            _get_header(
                ra_center, dec_center, pixel_size_deg, "icrs", npix_height, npix_width
            )
        )

        # Pre-compute all R.A., Decs
        self._ras, self._decs = _get_all_ra_dec(self._wcs, npix_height, npix_width)

        # Make sure we have the right amount of coordinates
        assert self._ras.shape[0] == self._decs.shape[0]
        assert self._ras.shape[0] == npix_width * npix_height

        # Pre-compute pixel area
        self._pixel_area = proj_plane_pixel_area(self._wcs)

        # Pre-compute an oversampled version to be used for PSF integration
        # if oversample and pixel_size_deg > 0.025:
        #
        #     self._oversampled, self._oversample_factor = self._oversample(new_pixel_size=0.025)
        #
        # else:
        #
        #     self._oversampled = self
        #     self._oversample_factor = 1

        # Cache for angular distances from a point (see get_spherical_distances_from)
        self._distance_cache = {}

    # def _oversample(self, new_pixel_size):
    #     """Return a new instance oversampled by the provided factor"""
    #
    #     # Compute the oversampling factor (as a float because we need it for the division down)
    #     factor = float(np.ceil(self._pixel_size_deg / new_pixel_size))
    #
    #     if factor <= 1:
    #
    #         # The projection is already with a smaller pixel size than the oversampled version
    #         # No need to oversample
    #         return self, 1
    #
    #     else:
    #
    #         new_fp = FlatSkyProjection(self._ra_center, self._dec_center,
    #                                    self._pixel_size_deg / factor,
    #                                    self._npix_height * factor,
    #                                    self._npix_width * factor,
    #                                    oversample=False)
    #
    #         return new_fp, int(factor)

    # @property
    # def oversampled(self):
    #     return self._oversampled
    #
    # @property
    # def oversample_factor(self):
    #     return self._oversample_factor

    @property
    def ras(self):
        """
        :return: Right Ascension for all pixels
        """
        return self._ras

    @property
    def decs(self):
        """
        :return: Declination for all pixels
        """
        return self._decs

    @property
    def ra_center(self):
        """
        :return: R.A. for the center of the projection
        """
        return self._ra_center

    @property
    def dec_center(self):
        """
        :return: Declination for the center of the projection
        """
        return self._dec_center

    @property
    def pixel_size(self):
        """
        :return: size (in deg) of the pixel
        """
        return self._pixel_size_deg

    @property
    def wcs(self):
        """
        :return: World Coordinate System instance describing the projection
        """
        return self._wcs

    @property
    def npix_height(self):
        """
        :return: height of the projection in pixels
        """
        return self._npix_height

    @property
    def npix_width(self):
        """
        :return: width of the projection in pixels
        """
        return self._npix_width

    @property
    def project_plane_pixel_area(self):
        """
        :return: area of the pixels (remember, this is an equal-area projection so all pixels are equal)
        """
        return self._pixel_area

    # def get_spherical_distances_from(self, ra, dec, cutout_radius):
    #     """
    #     Returns the distances for all points in this grid from the given point
    #
    #     :param ra:
    #     :param dec:
    #     :param cutout_radius: do not consider elements beyond this radius (NOTE: we use a planar approximation on
    #     purpose, to make things fast, so the cut is not precise)
    #     :return: (angular distances of selected points from (ra, dec), selection indexes)
    #     """
    #
    #     # This is typically used sequentially on different energy bins, so we cache the result and re-use it
    #     # if we already computed it
    #
    #     key = (ra, dec, cutout_radius, self.ras.shape[0], self.decs.shape[0])
    #
    #     if key not in self._distance_cache:
    #
    #         # In order to gain speed, we use a planar approximation (instead of using the harversine formula we assume
    #         # plane geometry). This gets more and more unprecise the large the cutout radius, but we do not care here
    #
    #         selection_idx = (((ra - self.ras)**2 + (dec - self.decs)**2) <= (1.2*cutout_radius)**2)  # type: np.ndarray
    #
    #         ds = sphere_dist(ra, dec, self.ras[selection_idx], self.decs[selection_idx])
    #
    #         # Refine selection by putting to False all elements in the mask at a distance larger than the cutout
    #         # radius
    #         fine_selection_idx = (ds <= cutout_radius)
    #         selection_idx[selection_idx.nonzero()[0][~fine_selection_idx]] = False
    #
    #         # This is to make sure we only keep cached the last result, and the dictionary does not grow indefinitely
    #         self._distance_cache = {}
    #
    #         self._distance_cache[key] = (ds[fine_selection_idx], selection_idx)
    #
    #     return self._distance_cache[key]
