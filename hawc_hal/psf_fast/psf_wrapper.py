import numpy as np
import scipy.interpolate
import scipy.optimize
import pandas as pd


class PSFWrapper(object):

    def __init__(self, xs, ys):

        self._xs = xs
        self._ys = ys

        self._tf1_interpolated = scipy.interpolate.InterpolatedUnivariateSpline(xs, ys, k=2, ext='raise')

        # Memorize the total integral (will use it for normalization)

        self._total_integral = self._tf1_interpolated.integral(0, 30)

        # Now compute the truncation radius, which is a very conservative measurement
        # of the size of the PSF

        self._truncation_radius = self.find_eef_radius(0.9999)

        # Let's also compute another measurement more appropriate for convolution
        self._kernel_radius = self.find_eef_radius(0.999)

        assert self._kernel_radius <= self._truncation_radius

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

        # We assume that the XS are the same
        assert np.allclose(self.xs, other_psf.xs)

        # Weight the ys
        new_ys = w1 * self.ys + w2 * other_psf.ys

        return PSFWrapper(self.xs, new_ys)

    def to_pandas(self):

        items = (('xs', self._xs), ('ys', self._ys))

        return pd.DataFrame.from_dict(dict(items))

    @classmethod
    def from_pandas(cls, df):

        return cls(df.loc[:, 'xs'], df.loc[:, 'ys'])

    @classmethod
    def from_TF1(cls, tf1_instance):

        # Make interpolation
        xs = np.logspace(-3, np.log10(30), 1000)
        ys = np.array(map(lambda x:tf1_instance.Eval(x), xs), float)

        assert np.all(np.isfinite(ys))

        return cls(xs, ys)

    def find_eef_radius(self, fraction):

        f = lambda r: fraction - self._tf1_interpolated.integral(1e-4, r) / self._total_integral

        radius, status = scipy.optimize.brentq(f, 0.005, 30, full_output = True)

        assert status.converged, "Brentq did not converged"

        return radius

    def integral(self, a, b):

        return self._tf1_interpolated.integral(a, b)

    @property
    def truncation_radius(self):
        return self._truncation_radius

    @property
    def total_integral(self):
        return self._total_integral

    @property
    def kernel_radius(self):
        return self._kernel_radius
