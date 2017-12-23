from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy.coordinates.angle_utilities import angular_separation
from astropy import wcs
from astropy.wcs.utils import proj_plane_pixel_area

import numpy as np
import healpy as hp
import pandas as pd

from partial_image_to_healpix import image_to_healpix
from special_values import UNSEEN
import healpix_utils
import sparse_healpix

import ROOT

_psf_stamp_npix = 1000

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

        for j in xrange(1, arrays[0].size):

            out[j*m:(j+1)*m, 1:] = out[0:m, 1:]

    return out


def _get_header(ra, dec, pixel_size, coordsys):

    assert 0 <= ra <= 360

    header = fits.Header.fromstring(_fits_header % (_psf_stamp_npix, _psf_stamp_npix,
                                                    _psf_stamp_npix / 2, ra, pixel_size,
                                                    _psf_stamp_npix / 2, dec, pixel_size,
                                                    coordsys),
                                    sep='\n')

    return header


# A function to get a grid of angular distances from the center of the stamp
# NOTE: the (ra, dec) are not important, as the center of the grid is transferred
# and therefore the distances are always the same, independently of the center

def _get_all_angular_distances(ra, dec, pixel_size, coordsys='icrs', input_wcs=None, fast=False):

    if input_wcs is None:

        target_header = _get_header(ra, dec, pixel_size, coordsys)

        # Compute separation of all points
        input_wcs = wcs.WCS(target_header)


    # An array of all the possible permutation of (i,j) for i=0..999 and j=0..999

    _ij_grid = cartesian((np.arange(0.5, _psf_stamp_npix + 0.5, 1, dtype=np.int16),
                          np.arange(0.5, _psf_stamp_npix + 0.5, 1, dtype=np.int16)))

    if not fast:

        # Use angular distance

        # Convert pixel coordinates to world coordinates
        world = np.deg2rad(input_wcs.wcs_pix2world(_ij_grid, 0, ra_dec_order=True))

        # Get angular distance
        d = np.rad2deg(angular_separation(np.deg2rad(ra), np.deg2rad(dec), world[:, 0], world[:, 1]))

    else:

        # Use flat approximation. This is good to better than 10% for a square up to 20 deg in size

        ij_center = input_wcs.wcs_world2pix([[ra, dec]], 0)[0]

        d = np.sqrt((ij_center[0] - _ij_grid[:, 0])**2 + (ij_center[1] - _ij_grid[:, 1])**2) * pixel_size

    return d


class TF1Wrapper(object):

    def __init__(self, tf1_instance):

        # Make a copy so that if the passed instance was a pointer from a TFile,
        # it will survive the closing of the associated TFile

        self._tf1 = ROOT.TF1(tf1_instance)

    def integral(self, *args, **kwargs):

        return self._tf1.Integral(*args, **kwargs)

    def __call__(self, *args, **kwargs):

        return self._tf1.Eval(*args, **kwargs)

    def _get_point_source_image_aitoff(self, ra, dec, coordsys='icrs'):

        # Get the differential PSF (using a truncation radius depending on the PSF, to avoid numerical problems)
        # from the stored TF1, which is the encircled energy function

        # We need the density, so we need to renormalize for the number of counts...
        truncation_radius = self._tf1.GetX(1e-20, 1e-3, 15.0)

        total_counts = self._tf1.Integral(0, truncation_radius)

        # Now decide the pixel size according to the truncation radius
        pixel_size = truncation_radius * 2 / _psf_stamp_npix

        # Get the WCS (we'll need it later for the pixel area and the transformation to healpix)
        target_header = _get_header(ra, dec, pixel_size, coordsys)

        target_wcs = wcs.WCS(target_header)

        interp_r = np.arange(0, truncation_radius * np.sqrt(2), pixel_size)

        interp_x = (interp_r[1:] + interp_r[:-1]) / 2.0

        # ... and for the inverse of the area of the pixel ...

        pixel_area_inv = proj_plane_pixel_area(target_wcs)  # square degrees

        renorm = pixel_area_inv / total_counts

        # ... and for the area of the ring between inner radius (a) and outer radius (b)

        interp_y = np.array(map(lambda (a, b): self._tf1.Integral(a, b) / (np.pi * (b ** 2 - a ** 2)) * renorm,
                             zip(interp_r[:-1], interp_r[1:])))

        # Let's interpolate

        # psf_diff = scipy.interpolate.InterpolatedUnivariateSpline(self._interp_x, self._interp_y, k=2, ext=0)
        #
        # densities = psf_diff(_angular_distances)

        angular_distances = _get_all_angular_distances(ra, dec, pixel_size, coordsys, input_wcs=target_wcs)

        densities = np.interp(angular_distances, interp_x, interp_y)

        assert np.alltrue(np.isfinite(densities))

        point_source_img_ait = densities.reshape((_psf_stamp_npix, _psf_stamp_npix))

        # Now the sum of the point source image should be close to 1.0, however it will not be
        # exactly one because of the interpolation and limited sampling. Let's check that we
        # are close enough, and then renormalize to make it exactly 1.0

        norm = point_source_img_ait.sum()

        # We accept a precision of 1 part in 100

        assert np.isclose(norm, 1.0, rtol=1e-2), "Point source image is not properly renormalized (norm = %s). " \
                                                 "This is a bug." % (norm)

        # Let's make it exactly 1.0

        point_source_img_ait /= norm

        return point_source_img_ait, target_wcs, interp_x, interp_y, truncation_radius

    def point_source_image(self, nside, ra, dec, coordsys='icrs', pixels_to_be_zeroed=None):
        """

        :param nside:
        :param ra:
        :param dec:
        :param coordsys:
        :param fill_value:
        :param pixels_to_be_zeroed: pixels IDS for pixels that will be 0.0 or the value of the PSF. This is useful
        to align the output sparse healpix map to another sparse healpix map
        :return:
        """

        point_source_img_ait, target_wcs, _, _, truncation_radius = self._get_point_source_image_aitoff(ra,
                                                                                                        dec,
                                                                                                        coordsys)

        # Now divide by the area before the interpolation
        point_source_img_ait /= proj_plane_pixel_area(target_wcs)

        # Now let's transport to healpix

        # First we need to find out which pixels in the future map are going to be affected by the
        # interpolation

        vec = healpix_utils.radec_to_vec(ra, dec)

        pixels_ids = hp.query_disc(nside, vec, np.deg2rad(truncation_radius), inclusive=True)

        assert len(pixels_ids) > 0

        healpix_img = image_to_healpix(point_source_img_ait, target_wcs,
                                       'icrs', nside, pixels_ids,
                                       order='bilinear', fill_value=UNSEEN,
                                       pixels_to_be_zeroed=pixels_to_be_zeroed)

        # Now let's account for the changed pixel area

        # NOTE: this "if" is only needed to support in the future if we change the value of UNSEEN
        # from np.nan to something else.
        if np.isnan(UNSEEN):

            idx = np.isfinite(healpix_img)

        else:

            idx = (healpix_img != UNSEEN)

        healpix_img[idx] *= hp.nside2pixarea(nside, degrees=True)

        # The interpolation slightly affects the normalization, let's restore it to exactly 1.0

        healpix_img[idx] /= np.sum(healpix_img[idx])

        # Make it sparse (to spare *a lot* of memory)
        healpix_img_sparse = sparse_healpix.SparseHealpix(healpix_img)

        return healpix_img_sparse

    def plot(self, ra=0.0, dec=0.0):

        point_source_img_ait, target_wcs, interp_x, interp_y, _ = self._get_point_source_image_aitoff(ra,
                                                                                                      dec,
                                                                                                      'icrs')

        fig = plt.figure()

        sub0 = plt.subplot(211)

        sub0.plot(interp_x, interp_y)
        sub0.set_xlabel("Radius (deg)")
        sub0.set_ylabel("Density")
        sub0.set_yscale("log")

        sub1 = plt.subplot(212, projection=target_wcs)

        print("Min: %s" % point_source_img_ait.min())
        print("Max: %s" % point_source_img_ait.max())

        sub1.imshow(point_source_img_ait, norm=LogNorm())

        sub1.coords.grid(color='white')
        sub1.coords.frame.set_color('none')

        plt.tight_layout()

        return fig, target_wcs
