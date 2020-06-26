from __future__ import division
from builtins import zip
from builtins import object
from past.utils import old_div
import numpy as np
import scipy.interpolate
import scipy.optimize
import pandas as pd

_INTEGRAL_OUTER_RADIUS = 15.0


class InvalidPSFError(ValueError):

    pass


class PSFWrapper(object):

    def __init__(self, xs, ys, brightness_interp_x=None, brightness_interp_y=None):

        self._xs = xs
        self._ys = ys

        self._psf_interpolated = scipy.interpolate.InterpolatedUnivariateSpline(xs, ys, k=2,
                                                                                ext='raise',
                                                                                check_finite=True)

        # Memorize the total integral (will use it for normalization)

        self._total_integral = self._psf_interpolated.integral(self._xs[0], _INTEGRAL_OUTER_RADIUS)

        # Now compute the truncation radius, which is a very conservative measurement
        # of the size of the PSF

        self._truncation_radius = self.find_eef_radius(0.9999)

        # Let's also compute another measurement more appropriate for convolution
        self._kernel_radius = self.find_eef_radius(0.999)

        assert self._kernel_radius <= self._truncation_radius
        assert self._truncation_radius <= _INTEGRAL_OUTER_RADIUS

        # Prepare brightness interpolation

        if brightness_interp_x is None:

            brightness_interp_x, brightness_interp_y = self._prepare_brightness_interpolation_points()

        self._brightness_interp_x = brightness_interp_x
        self._brightness_interp_y = brightness_interp_y

        self._brightness_interpolation = scipy.interpolate.InterpolatedUnivariateSpline(brightness_interp_x,
                                                                                        brightness_interp_y,
                                                                                        k=2,
                                                                                        ext='extrapolate',
                                                                                        check_finite=True)

    def _prepare_brightness_interpolation_points(self):

        # Get the centers of the bins
        interp_x = (self._xs[1:] + self._xs[:-1]) / 2.0

        # Compute the density
        interp_y = np.array([(self.integral(a_b[0], a_b[1]) / (np.pi * (a_b[1] ** 2 - a_b[0] ** 2)) / self._total_integral) for a_b in zip(self._xs[:-1], self._xs[1:])])

        # Add zero at r = _INTEGRAL_OUTER_RADIUS so that the extrapolated values will be correct
        interp_x = np.append(interp_x, [_INTEGRAL_OUTER_RADIUS])
        interp_y = np.append(interp_y, [0.0])

        return interp_x, interp_y

    def find_eef_radius(self, fraction):

        f = lambda r: fraction - old_div(self.integral(1e-4, r), self._total_integral)

        radius, status = scipy.optimize.brentq(f, 0.005, _INTEGRAL_OUTER_RADIUS, full_output = True)

        assert status.converged, "Brentq did not converged"

        return radius

    def brightness(self, r):

        return self._brightness_interpolation(r)

    @property
    def xs(self):
        """
        X of the interpolation data
        """
        return self._xs

    @property
    def ys(self):
        """
        Y of the interpolation data
        """
        return self._ys

    def combine_with_other_psf(self, other_psf, w1, w2):
        """
        Return a PSF which is the linear interpolation between this one and the other one provided

        :param other_psf: another psf
        :param w1: weight for self (i.e., this PSF)
        :param w2: weight for the other psf
        :return: another PSF instance
        """

        if isinstance(self, InvalidPSF) or isinstance(other_psf, InvalidPSF):
            return InvalidPSF()

        # Weight the ys
        new_ys = w1 * self.ys + w2 * other_psf.ys

        # Also weight the brightness interpolation points
        new_br_interp_y = w1 * self._brightness_interp_y + w2 * other_psf._brightness_interp_y

        return PSFWrapper(self.xs, new_ys,
                          brightness_interp_x=self._brightness_interp_x,
                          brightness_interp_y=new_br_interp_y)

    def to_pandas(self):

        items = (('xs', self._xs), ('ys', self._ys))

        return pd.DataFrame.from_dict(dict(items))

    @classmethod
    def from_pandas(cls, df):

        # Check for an invalid PSF
        xs = df.loc[:, 'xs'].values
        ys = df.loc[:, 'ys'].values

        if len(xs) == 0:

            # Should never happen
            assert len(ys) == 0, "Corrupted response file? A PSF has 0 xs values but more than 0 ys values"

            # An invalid PSF
            return InvalidPSF()

        else:

            return cls(xs, ys)

    @classmethod
    def from_TF1(cls, tf1_instance):

        # Annoyingly, some PSFs for some Dec bins (at large Zenith angles) have
        # zero or negative integrals, i.e., they are useless. Return an unusable PSF
        # object in that case
        if tf1_instance.Integral(0, _INTEGRAL_OUTER_RADIUS) <= 0.0:

            return InvalidPSF()

        # Make interpolation
        xs = np.logspace(-3, np.log10(_INTEGRAL_OUTER_RADIUS), 500)
        ys = np.array([tf1_instance.Eval(x) for x in xs], float)

        assert np.all(np.isfinite(xs))
        assert np.all(np.isfinite(ys))

        instance = cls(xs, ys)

        instance._tf1 = tf1_instance.Clone()

        return instance

    def integral(self, a, b):

        return self._psf_interpolated.integral(a, b)

    @property
    def truncation_radius(self):
        return self._truncation_radius

    @property
    def total_integral(self):
        return self._total_integral

    @property
    def kernel_radius(self):
        return self._kernel_radius


# This is a class that, whatever you try to use it for, will raise an exception.
# This is to make sure that we never use an invalid PSF without knowing it
class InvalidPSF(object):

    # It can be useful to copy an invalid PSF. For instance HAL.get_simulated_dataset() makes a
    # copy of the HAL instance, including detector response, which can contain InvalidPSF (which
    # is fine as long as they are not used).
    def __deepcopy__(self, memo):
        return InvalidPSF()

    # This allow the Invalid PSF to be saved in the HDF file
    def to_pandas(self):

        items = (('xs', []), ('ys', []))

        return pd.DataFrame.from_dict(dict(items))

    def __getattribute__(self, item):

        # White list of available attributes
        if item in ["__deepcopy__", "to_pandas"]:

            return object.__getattribute__(self, item)

        raise InvalidPSFError("Trying to use an invalid PSF")
