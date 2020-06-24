from __future__ import division
from builtins import object
import numpy as np
import collections
from hawc_hal import flat_sky_projection
from hawc_hal.sphere_dist import sphere_dist
import reproject


def _divide_in_blocks(arr, nrows, ncols):
    """
    Return an array of shape (n, nrows, ncols) where
    n * nrows * ncols = arr.size

    If arr is a 2D array, the returned array should look like n subblocks with
    each subblock preserving the "physical" layout of arr.
    """
    h, w = arr.shape
    return (arr.reshape(h // nrows, nrows, -1, ncols)
            .swapaxes(1, 2)
            .reshape(-1, nrows, ncols))


class PSFInterpolator(object):

    def __init__(self, psf_wrapper, flat_sky_proj):

        self._psf = psf_wrapper

        self._flat_sky_p = flat_sky_proj

        # This will contain cached source images
        self._point_source_images = collections.OrderedDict()

    def _get_point_source_image_aitoff(self, ra, dec, psf_integration_method):

        # Get the density for the required (RA, Dec) points by interpolating the density profile

        # First we obtain an image with a flat sky projection centered exactly on the point source
        ancillary_image_pixel_size = 0.05
        pixel_side = 2 * int(np.ceil(self._psf.truncation_radius / ancillary_image_pixel_size))
        ancillary_flat_sky_proj = flat_sky_projection.FlatSkyProjection(ra, dec, ancillary_image_pixel_size,
                                                                        pixel_side, pixel_side)

        # Now we compute the angular distance of all pixels in the ancillary image with respect to the center
        angular_distances = sphere_dist(ra, dec, ancillary_flat_sky_proj.ras, ancillary_flat_sky_proj.decs)

        # Compute the brightness (i.e., the differential PSF)
        ancillary_brightness = self._psf.brightness(angular_distances).reshape((pixel_side, pixel_side)) * \
                               self._flat_sky_p.project_plane_pixel_area

        # Now reproject this brightness on the new image
        if psf_integration_method == 'exact':

            reprojection_method = reproject.reproject_exact
            additional_keywords = {'parallel': False}

        else:

            reprojection_method = reproject.reproject_interp
            additional_keywords = {}

        brightness, _ = reprojection_method((ancillary_brightness, ancillary_flat_sky_proj.wcs),
                                            self._flat_sky_p.wcs, shape_out=(self._flat_sky_p.npix_height,
                                                                             self._flat_sky_p.npix_width),
                                            **additional_keywords)

        brightness[np.isnan(brightness)] = 0.0

        # Now "integrate", i.e., multiply by pixel area
        point_source_img_ait = brightness

        # # First let's compute the core of the PSF, i.e., the central area with a radius of 0.5 deg,
        # # using a small pixel size
        #
        # # We use the oversampled version of the flat sky projection to make sure we compute an integral
        # # with the right pixelization
        #
        # oversampled = self._flat_sky_p.oversampled
        # oversample_factor = self._flat_sky_p.oversample_factor
        #
        # # Find bounding box for a faster selection
        #
        # angular_distances_core, core_idx = oversampled.get_spherical_distances_from(ra, dec, cutout_radius=5.0)
        #
        # core_densities = np.zeros(oversampled.ras.shape)
        #
        # core_densities[core_idx] = self._psf.brightness(angular_distances_core) * oversampled.project_plane_pixel_area
        #
        # # Now downsample by summing every "oversample_factor" pixels
        #
        # blocks = _divide_in_blocks(core_densities.reshape((oversampled.npix_height, oversampled.npix_width)),
        #                                oversample_factor,
        #                                oversample_factor)
        #
        # densities = blocks.flatten().reshape([-1, oversample_factor**2]).sum(axis=1)  # type: np.ndarray
        #
        # assert densities.shape[0] == (self._flat_sky_p.npix_height * self._flat_sky_p.npix_width)
        #
        # # Now that we have a "densities" array with the right (not oversampled) resolution,
        # # let's update the elements outside the core, which are still zero
        # angular_distances_, within_outer_radius = self._flat_sky_p.get_spherical_distances_from(ra, dec,
        #                                                                                         cutout_radius=10.0)
        #
        # # Make a vector with the same shape of "densities" (effectively a padding of what we just computed)
        # angular_distances = np.zeros_like(densities)
        # angular_distances[within_outer_radius] = angular_distances_
        #
        # to_be_updated = (densities == 0)
        # idx = (within_outer_radius & to_be_updated)
        # densities[idx] = self._psf.brightness(angular_distances[idx]) * self._flat_sky_p.project_plane_pixel_area
        #
        # # NOTE: we check that the point source image is properly normalized in the convolved point source class.
        # # There we know which source we are talking about and for which bin, so we can print a more helpful
        # # help message
        #
        # # Reshape to required shape
        # point_source_img_ait = densities.reshape((self._flat_sky_p.npix_height, self._flat_sky_p.npix_width)).T

        return point_source_img_ait

    def point_source_image(self, ra_src, dec_src, psf_integration_method='exact'):

        # Make a unique key for this request
        key = (ra_src, dec_src, psf_integration_method)

        # If we already computed this image, return it, otherwise compute it from scratch
        if key in self._point_source_images:

            point_source_img_ait = self._point_source_images[key]

        else:

            point_source_img_ait = self._get_point_source_image_aitoff(ra_src, dec_src, psf_integration_method)

            # Store for future use

            self._point_source_images[key] = point_source_img_ait

            # Limit the size of the cache. If we have exceeded 20 images, we drop the oldest 10
            if len(self._point_source_images) > 20:

                while len(self._point_source_images) > 10:
                    # FIFO removal (the oldest calls are removed first)
                    self._point_source_images.popitem(last=False)

        return point_source_img_ait
