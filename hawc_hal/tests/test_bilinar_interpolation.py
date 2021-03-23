import numpy as np
import scipy.ndimage
from hawc_hal.util import cartesian
from hawc_hal.interpolation import fast_bilinar_interpolation


def test_fast_bilinear_interpolation():

    gridx = np.arange(10, dtype=int)
    gridy = np.arange(15, dtype=int)

    data = np.random.uniform(0, 1, size=(gridx.shape[0], gridy.shape[0]))

    new_x = np.random.uniform(min(gridx), max(gridx), 500)
    new_y = np.random.uniform(min(gridy), max(gridy), 500)
    new_coords = np.asarray(cartesian([new_x, new_y]))

    mfi = fast_bilinar_interpolation.FastBilinearInterpolation(data.shape, (new_coords[:, 0], new_coords[:, 1]))

    v1 = mfi(data)

    # Check against the slower scipy interpolator in map_coordinates
    v2 = scipy.ndimage.map_coordinates(data, np.array((new_coords[:, 0], new_coords[:, 1])), order=1)

    assert np.allclose(v1, v2)
