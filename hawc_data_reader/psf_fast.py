from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt

import numpy as np
from numpy.fft import rfftn, irfftn
from numba import jit, float64
import collections
import scipy.optimize
import scipy.interpolate
from tf1_wrapper import TF1Wrapper
import ROOT
ROOT.SetMemoryPolicy( ROOT.kMemoryStrict )


# A fast way to compute angular distances
@jit(float64[:](float64, float64, float64[:], float64[:]), nopython=True)
def sphere_dist(ra1, dec1, ra2, dec2):
    """
    Haversine formula for angular distance on a sphere: more stable at poles.
    This version uses arctan instead of arcsin and thus does better with sign
    conventions.

    :param ra1: first RA (deg)
    :param dec1: first Dec (deg)
    :param ra2: second RA (deg)
    :param dec2: second Dec (deg)

    :returns: angular separation distance (deg)
    """

    deg2rad = 0.017453292519943295
    rad2deg = 57.29577951308232

    ra1 = ra1 * deg2rad
    ra2 = ra2 * deg2rad
    dec1 = dec1 * deg2rad
    dec2 = dec2 * deg2rad

    numerator = np.sin((dec2 - dec1) / 2) ** 2 + np.cos(dec1) * np.cos(dec2) * np.sin((ra2 - ra1) / 2) ** 2

    dists = 2 * np.arctan2(numerator ** 0.5, (1 - numerator) ** 0.5)

    return dists * rad2deg


class PSFWrapper(TF1Wrapper):

    def __init__(self, tf1_instance):

        super(PSFWrapper, self).__init__(tf1_instance)

        # Make interpolation
        xs = np.logspace(-3, np.log10(30), 1000)
        ys = np.array(map(lambda x:self._tf1.Eval(x), xs), float)

        assert np.all(np.isfinite(ys))

        self._tf1_interpolated = scipy.interpolate.InterpolatedUnivariateSpline(xs, ys, k=2, ext=1)

        # Memorize the total integral (will use it for normalization)

        self._total_integral = self._tf1_interpolated.integral(1e-4, 30)

        # Now compute the truncation radius, which is a very conservative measurement
        # of the size of the PSF

        self._truncation_radius = self._find_eef_radius(0.9999)

        # Let's also compute another measurement more appropriate for convolution
        self._kernel_radius = self._find_eef_radius(0.999)

        assert self._kernel_radius < self._truncation_radius

        # print("Psf: %s, R_t = %.3f, R_k = %.3f" % (tf1_instance.GetName(), self._truncation_radius, self._kernel_radius))

    def _find_eef_radius(self, fraction):

        f = lambda r: fraction - self._tf1_interpolated.integral(1e-4, r) / self._total_integral

        radius, status = scipy.optimize.brentq(f, 0.005, 30, full_output = True)

        assert status.converged, "Brenq did not converged"

        return radius

    @property
    def truncation_radius(self):
        return self._truncation_radius

    @property
    def total_integral(self):
        return self._total_integral

    @property
    def kernel_radius(self):
        return self._kernel_radius


class PSFInterpolator(object):

    def __init__(self, tf1_wrapper, flat_sky_proj):

        self._tf1 = tf1_wrapper  # type: PSFWrapper

        self._flat_sky_p = flat_sky_proj

        # Now decide the bin size for the interpolation according to the truncation radius
        binsize = self._tf1.truncation_radius * 2 / flat_sky_proj.npix_height

        # Let's interpolate. We want to interpolate the density as a function of the radius

        # Decide a bunch of radii bins
        interp_r = np.arange(0, self._tf1.truncation_radius * np.sqrt(2), binsize)

        # Get the centers of the bins
        self._interp_x = (interp_r[1:] + interp_r[:-1]) / 2.0

        # To get the density we need to get the integral of the profile divided by the area of the circle between
        # radius a and b and the number of counts

        # Let's compute the pixel area and the overall normalization we will need to apply

        pixel_area = flat_sky_proj.project_plane_pixel_area  # square degrees

        renorm = pixel_area / self._tf1.total_integral

        # Compute the density

        self._interp_y = np.array(map(lambda (a, b): self._tf1.integral(a, b) / (np.pi * (b ** 2 - a ** 2)) * renorm,
                                zip(interp_r[:-1], interp_r[1:])))

        # This will contain cached source images
        self._point_source_images = collections.OrderedDict()

    def _get_point_source_image_aitoff(self, ra, dec):

        # Get the density for the required (RA, Dec) points by interpolating the density profile

        angular_distances = sphere_dist(ra, dec, self._flat_sky_p.ras, self._flat_sky_p.decs)

        densities = np.interp(angular_distances, self._interp_x, self._interp_y)

        # Reshape to required shape
        point_source_img_ait = densities.reshape((self._flat_sky_p.npix_height, self._flat_sky_p.npix_width))

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

    # def plot(self, ra=0.0, dec=0.0):
    #
    #     point_source_img_ait, target_wcs, interp_x, interp_y, _ = _get_point_source_image_aitoff(self._psf,
    #                                                                                              ra, dec,
    #                                                                                              'icrs')
    #
    #     fig = plt.figure()
    #
    #     sub0 = plt.subplot(211)
    #
    #     sub0.plot(interp_x, interp_y)
    #     sub0.set_xlabel("Radius (deg)")
    #     sub0.set_ylabel("Density")
    #     sub0.set_yscale("log")
    #
    #     sub1 = plt.subplot(212, projection=target_wcs)
    #
    #     print("Min: %s" % point_source_img_ait.min())
    #     print("Max: %s" % point_source_img_ait.max())
    #
    #     sub1.imshow(point_source_img_ait, norm=LogNorm())
    #
    #     sub1.coords.grid(color='white')
    #     sub1.coords.frame.set_color('none')
    #
    #     plt.tight_layout()
    #
    #     return fig, target_wcs


class PSFConvolutor(object):

    def __init__(self, psf_wrapper, flat_sky_proj):

        self._psf = psf_wrapper  # type: PSFWrapper
        self._flat_sky_proj = flat_sky_proj

        # Compute an image of the PSF on the current defined flat sky projection
        interpolator = PSFInterpolator(psf_wrapper, flat_sky_proj)
        psf_stamp = interpolator.point_source_image(flat_sky_proj.ra_center, flat_sky_proj.dec_center)

        # Crop the kernel at the appropriate radius for this PSF (making sure is divisible by 2)
        kernel_radius_px = int(np.ceil(self._psf.kernel_radius / flat_sky_proj.pixel_size))
        pixels_to_keep = kernel_radius_px * 2

        assert pixels_to_keep <= psf_stamp.shape[0] and \
               pixels_to_keep <= psf_stamp.shape[1], \
            "The kernel is too large with respect to the model image. Enlarge your model radius."

        xoff = (psf_stamp.shape[0] - pixels_to_keep) // 2
        yoff = (psf_stamp.shape[1] - pixels_to_keep) // 2

        self._kernel = psf_stamp[yoff:-yoff, xoff:-xoff]

        assert np.isclose(self._kernel.sum(), 1.0, rtol=1e-2), "Failed to generate proper kernel normalization"

        # Renormalize to exactly 1
        self._kernel = self._kernel / self._kernel.sum()

        self._expected_shape = (flat_sky_proj.npix_height, flat_sky_proj.npix_width)

        s1 = np.array(self._expected_shape)
        s2 = np.array(self._kernel.shape)

        shape = s1 + s2 - 1

        self._fshape = [_next_regular(int(d)) for d in shape]
        self._fslice = tuple([slice(0, int(sz)) for sz in shape])

        self._psf_fft = rfftn(self._kernel, self._fshape)

    @property
    def kernel(self):
        return self._kernel

    def extended_source_image(self, ideal_image):

        # Convolve

        assert np.alltrue(ideal_image.shape == self._expected_shape), "Shape of image to be convolved is not correct."

        ret = irfftn(rfftn(ideal_image, self._fshape) * self._psf_fft, self._fshape)[self._fslice].copy()

        return _centered(ret, self._expected_shape)


# Copied from scipy.signaltools.fftconvolve
def _next_regular(target):
    """
    Find the next regular number greater than or equal to target.
    Regular numbers are composites of the prime factors 2, 3, and 5.
    Also known as 5-smooth numbers or Hamming numbers, these are the optimal
    size for inputs to FFTPACK.
    Target must be a positive integer.
    """
    if target <= 6:
        return target

    # Quickly check if it's already a power of 2
    if not (target & (target-1)):
        return target

    match = float('inf')  # Anything found will be smaller
    p5 = 1
    while p5 < target:
        p35 = p5
        while p35 < target:
            # Ceiling integer division, avoiding conversion to float
            # (quotient = ceil(target / p35))
            quotient = -(-target // p35)

            # Quickly find next power of 2 >= quotient
            try:
                p2 = 2**((quotient - 1).bit_length())
            except AttributeError:
                # Fallback for Python <2.7
                p2 = 2**(len(bin(quotient - 1)) - 2)

            N = p2 * p35
            if N == target:
                return N
            elif N < match:
                match = N
            p35 *= 3
            if p35 == target:
                return p35
        if p35 < match:
            match = p35
        p5 *= 5
        if p5 == target:
            return p5
    if p5 < match:
        match = p5
    return match


def _centered(arr, newsize):
    # Return the center newsize portion of the array.
    newsize = np.asarray(newsize)
    currsize = np.array(arr.shape)
    startind = (currsize - newsize) // 2
    endind = startind + newsize
    myslice = [slice(startind[k], endind[k]) for k in range(len(endind))]
    return arr[tuple(myslice)]
