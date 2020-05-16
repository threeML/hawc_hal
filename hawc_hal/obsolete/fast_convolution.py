from builtins import range
from builtins import object
from numpy.fft import rfftn, irfftn
from numpy import array, asarray, alltrue
import numpy as np
from numba import jit, float64


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
    newsize = asarray(newsize)
    currsize = array(arr.shape)
    startind = (currsize - newsize) // 2
    endind = startind + newsize
    myslice = [slice(startind[k], endind[k]) for k in range(len(endind))]
    return arr[tuple(myslice)]


class FastConvolution(object):

    def __init__(self, ras, decs, nhpix, nwpix, psf_instance):

        # Make the PSF image
        psf_image = np.zeros((nhpix, nwpix))


        self._ras = ras
        self._decs = decs

        self._psf_fft = None
        self._psf_image = psf_image

        self._fshape = None
        self._fslice = None

        self._expected_shape = None

    def setup(self, expected_shape):
        """
        Setup the convolution with the given shape. The FFT of the PSF will be computed, as well as other small
        things that are needed during the convolution step but stay constant if the shape does not change.

        :param shape: the shape of the image that will be convoluted
        :return: None
        """

        self._expected_shape = expected_shape

        s1 = array(expected_shape)
        s2 = array(self._psf_image.shape)

        shape = s1 + s2 - 1

        self._fshape = [_next_regular(int(d)) for d in shape]
        self._fslice = tuple([slice(0, int(sz)) for sz in shape])

        self._psf_fft = rfftn(self._psf_image, self._fshape)

    def __call__(self, image):

        assert alltrue(image.shape == self._expected_shape), "Shape of image to be convolved is not correct. Re-run setup."

        ret = irfftn(rfftn(image, self._fshape) * self._psf_fft, self._fshape)[self._fslice].copy()

        return _centered(ret, self._expected_shape)


@jit(float64[:, :](float64[:, :], float64[:, :], float64[:, :]), nopython=True)
def brute_force_convolution(image, kernel, output):

    h, w = image.shape

    kh, kw = kernel.shape

    assert kh == kw, "Only squared kernels are supported"

    kernel_size = kh

    assert kernel_size % 2 != 0, "Kernel number of rows and columns must be odd"

    half_size_minus_one = (kernel_size - 1) // 2

    for i in range(half_size_minus_one, h-half_size_minus_one):

        for j in range(half_size_minus_one, w-half_size_minus_one):

            output[j, i] = np.sum(image[j - half_size_minus_one : j + half_size_minus_one + 1,
                                        i - half_size_minus_one : i + half_size_minus_one + 1]
                                  * kernel)

    return output