from hawc_hal.maptree.map_tree import map_tree_factory
from hawc_hal import HealpixConeROI
import os
import pytest
from conftest import check_n_transits


def test_transits(maptree,
                     roi):

    roi_ = roi
    
    #Case 1: not specifying transits

    # Dont specify n_transits, use value in maptree
    maptree_none = map_tree_factory(maptree, roi_)

    n_transits = maptree_none.n_transits

    #This is somewhat trivial, but something has definitely gone wrong if this fails
    check_n_transits(maptree_none,n_transits)

    
    #Case 2: specify transits

    # specify transits, in this case 777.7
    n_transits_specify = 777.7

    # specify n_transits
    maptree_specify = map_tree_factory(maptree, roi_, n_transits_specify)

    # Does maptree return specified transits?
    check_n_transits(maptree_specify,n_transits_specify)

