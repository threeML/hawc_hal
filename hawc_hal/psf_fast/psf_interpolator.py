import numpy as np
import collections
from sphere_dist import sphere_dist


class PSFInterpolator(object):

    def __init__(self, psf_wrapper, flat_sky_proj):

        self._psf = psf_wrapper  # type: PSFWrapper

        self._flat_sky_p = flat_sky_proj

        # Now decide the bin size for the interpolation according to the truncation radius
        binsize = self._psf.truncation_radius * 2 / flat_sky_proj.npix_height

        # Let's interpolate. We want to interpolate the density as a function of the radius

        # Decide a bunch of radii bins
        interp_r = np.arange(0, self._psf.truncation_radius * np.sqrt(2), binsize)

        # Get the centers of the bins
        self._interp_x = (interp_r[1:] + interp_r[:-1]) / 2.0

        # To get the density we need to get the integral of the profile divided by the area of the circle between
        # radius a and b and the number of counts

        # Let's compute the pixel area and the overall normalization we will need to apply

        pixel_area = flat_sky_proj.project_plane_pixel_area  # square degrees

        renorm = pixel_area / self._psf.total_integral

        # Compute the density

        self._interp_y = np.array(map(lambda (a, b): self._psf.integral(a, b) / (np.pi * (b ** 2 - a ** 2)) * renorm,
                                      zip(interp_r[:-1], interp_r[1:])))

        # This will contain cached source images
        self._point_source_images = collections.OrderedDict()

    def _get_point_source_image_aitoff(self, ra, dec):

        # Get the density for the required (RA, Dec) points by interpolating the density profile

        angular_distances = sphere_dist(ra, dec, self._flat_sky_p.ras, self._flat_sky_p.decs)

        densities = np.interp(angular_distances, self._interp_x, self._interp_y)

        # Reshape to required shape
        point_source_img_ait = densities.reshape((self._flat_sky_p.npix_height, self._flat_sky_p.npix_width)).T

        return point_source_img_ait

    def point_source_image(self, ra_src, dec_src):

        # Make a unique key for this request
        key = (ra_src, dec_src)

        # If we already computed this image, return it, otherwise compute it from scratch
        if key in self._point_source_images:

            point_source_img_ait = self._point_source_images[key]

        else:

            point_source_img_ait = self._get_point_source_image_aitoff(ra_src, dec_src)

            # Store for future use

            self._point_source_images[key] = point_source_img_ait

            # Limit the size of the cache. If we have exceeded 20 images, we drop the oldest 10
            if len(self._point_source_images) > 20:

                while len(self._point_source_images) > 10:

                    # FIFO removal (the oldest calls are removed first)
                    self._point_source_images.popitem(last=False)

        return point_source_img_ait
