import numpy as np
from numpy.fft import rfftn, irfftn
#from scipy.signal import fftconvolve
from scipy.fftpack import helper


from psf_interpolator import PSFInterpolator
from psf_wrapper import PSFWrapper


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

        self._fshape = [helper.next_fast_len(int(d)) for d in shape]
        self._fslice = tuple([slice(0, int(sz)) for sz in shape])

        self._psf_fft = rfftn(self._kernel, self._fshape)

    @property
    def kernel(self):
        return self._kernel

    # def extended_source_image(self, ideal_image):
    #
    #     conv = fftconvolve(ideal_image, self._kernel, mode='same')
    #
    #     return conv

    def extended_source_image(self, ideal_image):

        # Convolve

        assert np.alltrue(ideal_image.shape == self._expected_shape), "Shape of image to be convolved is not correct."

        ret = irfftn(rfftn(ideal_image, self._fshape) * self._psf_fft, self._fshape)[self._fslice].copy()

        conv = _centered(ret, self._expected_shape)

        return conv


def _centered(arr, newsize):
    # Return the center newsize portion of the array.
    newsize = np.asarray(newsize)
    currsize = np.array(arr.shape)
    startind = (currsize - newsize) // 2
    endind = startind + newsize
    myslice = [slice(startind[k], endind[k]) for k in range(len(endind))]
    return arr[tuple(myslice)]