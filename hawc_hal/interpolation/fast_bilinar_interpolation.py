from __future__ import division
from builtins import object
from past.utils import old_div
import numpy as np

from numba import jit


class FastBilinearInterpolation(object):
    """
    A super fast bilinar interpolation implementation which exploits the fact that we are always interpolating in the
    same grid. For example, if we always go from the same flat sky projection to the same Healpix map, we can precompute
    the weights for the interpolation and then apply them to any new data instead of recomputing them every time.
    """

    def __init__(self, input_shape, new_points):

        self._gridx = np.arange(input_shape[0])
        self._gridy = np.arange(input_shape[1])

        self._x_bounds = [self._gridx.min(), self._gridx.max()]
        self._y_bounds = [self._gridy.min(), self._gridy.max()]

        self._data_shape = (self._gridx.shape[0], self._gridy.shape[0])

        self._bs, self._flat_points = self.compute_coefficients(new_points)

    @staticmethod
    def _find_bounding_box(xaxis, yaxis, xs, ys):
        # Find lower left corner of bounding box
        xidx = np.searchsorted(xaxis, xs, 'left') - 1
        yidx = np.searchsorted(yaxis, ys, 'left') - 1

        lower_left_x = xaxis[xidx]
        lower_left_y = yaxis[yidx]

        upper_left_x = xaxis[xidx]
        upper_left_y = yaxis[yidx + 1]

        upper_right_x = xaxis[xidx + 1]
        upper_right_y = yaxis[yidx + 1]

        lower_right_x = xaxis[xidx + 1]
        lower_right_y = yaxis[yidx]

        return (lower_left_x, lower_left_y,
                upper_left_x, upper_left_y,
                upper_right_x, upper_right_y,
                lower_right_x, lower_right_y)

    def compute_coefficients(self, p):

        xx = p[0]
        yy = p[1]

        # Find bounding boxes
        # We need a situation like this
        #    x1    x2
        # y1 q11   q21
        #
        # y2 q12   q22

        x1, y2, xx1, y1, x2, yy1, xx2, yy2 = self._find_bounding_box(self._gridx, self._gridy, xx, yy)

        bs = np.zeros((xx.shape[0], 4), np.float64)

        bs[:, 0] = old_div((x2 - xx) * (y2 - yy), (x2 - x1)) * (y2 - y1)
        bs[:, 1] = old_div((xx - x1) * (y2 - yy), (x2 - x1)) * (y2 - y1)
        bs[:, 2] = old_div((x2 - xx) * (yy - y1), (x2 - x1)) * (y2 - y1)
        bs[:, 3] = old_div((xx - x1) * (yy - y1), (x2 - x1)) * (y2 - y1)

        # Get the flat indexing for all the corners of the bounding boxes
        flat_upper_left = np.ravel_multi_index((x1, y1), self._data_shape)
        flat_upper_right = np.ravel_multi_index((x2, y1), self._data_shape)
        flat_lower_left = np.ravel_multi_index((x1, y2), self._data_shape)
        flat_lower_right = np.ravel_multi_index((x2, y2), self._data_shape)

        # Stack them so that they are in the right order, i.e.:
        # ul1, ur1, ll1, lr1, ul2, ur2, ll2, lr2 ... uln, urn, lln, lrn

        flat_points = np.vstack([flat_upper_left,
                                 flat_upper_right,
                                 flat_lower_left,
                                 flat_lower_right]).T.flatten()

        return bs, flat_points.astype(np.int64)

    def __call__(self, data):

        res = _apply_bilinar_interpolation(self._bs, self._flat_points, data)

        return res


@jit("float64[:](float64[:, :], int64[:], float64[:, :])", nopython=True)
def _apply_bilinar_interpolation(bs, points, data):  # pragma: no cover

    vs = data.ravel()[points]

    return np.sum(bs * vs.reshape(bs.shape), axis=1).flatten()