from builtins import object
import numpy as np
from numpy import log10
from math import log10 as mlog10
import scipy.interpolate


class LogLogInterpolator(object):  # pragma: no cover

    def __init__(self, x, y, k=2):

        y = y.astype(np.float64)
        y = np.clip(y, 2 * np.finfo(np.float64).tiny, None)

        logx = log10(x)
        logy = log10(y)

        self._interp = scipy.interpolate.InterpolatedUnivariateSpline(logx, logy, k=k)

    def __call__(self, x):

        return 10**self._interp(log10(x))

    def integral(self, a, b, n_points=100, k=1):

        # Make a second interpolator
        xx = np.logspace(mlog10(a), mlog10(b), n_points)

        yy = self.__call__(xx)

        int_interp = scipy.interpolate.InterpolatedUnivariateSpline(xx, yy, k=k)

        return int_interp.integral(a, b)