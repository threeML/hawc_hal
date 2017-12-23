from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy.coordinates.angle_utilities import angular_separation
from astropy import wcs

import numpy as np
import pandas as pd

import healpy as hp
from healpix_utils import radec_to_vec, pixid_to_radec

import ROOT

from special_values import UNSEEN

_interpolation_nside = 1024 * 4


# A function to get a grid of angular distances from the center of the stamp

def _get_all_angular_distances(nside):

    # The center does not really matter as the grid is the same everywhere
    # and we are using a very high resolution
    ra, dec = (0.0, 0.0)

    # Select all healpix pixels within a disc of 10 deg radius
    radius = 15 # deg
    vec = radec_to_vec(ra, dec)

    pixel_ids = hp.query_disc(nside, vec, np.deg2rad(radius), inclusive=True)

    ras, decs = pixid_to_radec(nside, pixel_ids, nest=False)

    # Get angular distance
    d = np.rad2deg(angular_separation(np.deg2rad(ra), np.deg2rad(dec), np.deg2rad(ras), np.deg2rad(decs)))

    return d, pixel_ids


_angular_distances, _pixels_ids = _get_all_angular_distances(_interpolation_nside)


class TF1Wrapper(object):

    def __init__(self, tf1_instance):

        # Make a copy so that if the passed instance was a pointer from a TFile,
        # it will survive the closing of the associated TFile

        self._tf1 = ROOT.TF1(tf1_instance)

    def integral(self, *args, **kwargs):

        return self._tf1.Integral(*args, **kwargs)

    def __call__(self, *args, **kwargs):

        return self._tf1.Eval(*args, **kwargs)

    def point_source_image(self, nside, ra, dec, coordsys='icrs'):

        # Get the differential PSF (using a truncation radius depending on the PSF, to avoid numerical problems)
        # from the stored TF1, which is the encircled energy function

        # We need the density, so we need to renormalize for the number of counts...
        truncation_radius = self._tf1.GetX(1e-20, 1e-3, 10.0)

        total_counts = self._tf1.Integral(0, truncation_radius)

        interp_r = np.arange(0, 15.0, 0.001)

        self._interp_x = (interp_r[1:] + interp_r[:-1]) / 2.0

        # ... and for the inverse of the area of the pixel ...
        pixel_area = hp.nside2pixarea(_interpolation_nside, degrees=True)  # square degrees

        renorm = pixel_area / total_counts

        # ... and for the area of the ring between inner radius (a) and outer radius (b)

        self._interp_y = np.array(map(lambda (a, b): self._tf1.Integral(a, b) / (np.pi * (b**2-a**2)) * renorm,
                                  zip(interp_r[:-1],interp_r[1:])))

        # Let's interpolate

        # psf_diff = scipy.interpolate.InterpolatedUnivariateSpline(self._interp_x, self._interp_y, k=2, ext=0)
        #
        # densities = psf_diff(_angular_distances)

        densities = np.interp(_angular_distances, self._interp_x, self._interp_y, left=0)

        # Make a sparse healpix map

        _point_source_map_dense = np.zeros(hp.nside2npix(_interpolation_nside)) + UNSEEN

        _point_source_map_dense[_pixels_ids] = densities

        # Now resize the map
        _point_source_map_dense = hp.ud_grade(_point_source_map_dense, nside, power=-2,
                                              order_in='RING', order_out='RING', pess=False)

        # Now the sum of the point source image should be close to 1.0, however it will not be
        # exactly one because of the interpolation and limited sampling. Let's check that we
        # are close enough, and then renormalize to make it exactly 1.0

        new_pixels_ids = _point_source_map_dense != UNSEEN

        norm = _point_source_map_dense[new_pixels_ids].sum()

        # We accept a precision of 1 part in 1000

        assert np.isclose(norm, 1.0, rtol=1e-3), "Point source image is not properly renormalized (norm = %s). " \
                                                 "This is a bug." % (norm)

        # Let's make it exactly 1.0

        _point_source_map_dense[new_pixels_ids] /= norm

        self._point_source_map_sparse = pd.SparseArray(_point_source_map_dense, kind='block', fill_value=UNSEEN)

        return self._point_source_map_sparse

    def plot(self):

        self.point_source_image(1024, 0.0, 0.0, coordsys='icrs')

        fig = plt.figure()

        sub0 = plt.subplot(211)

        sub0.plot(self._interp_x, self._interp_y)
        sub0.set_xlabel("Radius (deg)")
        sub0.set_ylabel("Density")
        sub0.set_yscale("log")

        target_header = fits.Header.fromstring(_fits_header % (_psf_stamp_npix, _psf_stamp_npix,
                                                               _psf_stamp_npix / 2, 180.0, _pixel_size,
                                                               _psf_stamp_npix / 2, 0.0, _pixel_size,
                                                               'icrs'),
                                               sep='\n')

        # Compute separation of all points
        mywcs = wcs.WCS(target_header)

        sub1 = plt.subplot(212, projection=mywcs)

        print("Min: %s" % self._point_source_map_sparse.min())
        print("Max: %s" % self._point_source_map_sparse.max())

        sub1.imshow(self._point_source_map_sparse, norm=LogNorm())

        sub1.coords.grid(color='white')
        sub1.coords.frame.set_color('none')

        plt.tight_layout()

        return fig, target_header