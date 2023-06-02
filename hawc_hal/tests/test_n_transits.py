import os

import pytest
from hawc_hal import HealpixConeROI
from hawc_hal.maptree.map_tree import map_tree_factory

from conftest import check_n_transits


def test_transits(maptree, roi):
    """Specify the number of transits to use with maptree

    Args:
        maptree (path.Path): Maptree in either ROOT or HDF5 format
        roi (roi): Region of interest for analysis
    """
    roi_ = roi

    # Case 1: specify number of transits

    n_transits = 777.7
    maptree_ntransits = map_tree_factory(maptree,roi_,n_transits)

    # does the maptree return the specified transits?
    check_n_transits(maptree_ntransits, n_transits)

    # Case 2: number of transits is not specified (use value from maptree)
    maptree_unspecifed = map_tree_factory(maptree, roi_)
    n_transits = maptree_unspecifed.n_transits

    check_n_transits(maptree_unspecifed, n_transits)

    