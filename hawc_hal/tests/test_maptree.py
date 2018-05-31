from hawc_hal.map_tree import map_tree_factory
from hawc_hal import HealpixConeROI
import os
import pytest
from conftest import check_map_trees


def test_constructor(maptree,
                     roi):

    roi_ = roi

    m = map_tree_factory(maptree, roi_)

    test_filename = "maptree.hd5"

    # Make sure it doesn't exist yet, if it does,remove it
    if os.path.exists(test_filename):
        os.remove(test_filename)

    m.write(test_filename)

    # Try to open and use it
    m2 = map_tree_factory(test_filename, roi_)

    # Check corner cases
    # This should issue a warning because we saved the maptree with a ROI and we try to use
    # without one
    with pytest.warns(UserWarning):
        _ = map_tree_factory(test_filename, None)

    # Now try to load with a different ROI than the one used for the file
    ra_c, dec_c = roi.ra_dec_center

    oroi = HealpixConeROI(10.0, 10.0, ra=ra_c, dec=dec_c)

    with pytest.raises(AssertionError):
        _ = map_tree_factory(test_filename, oroi)

    # This instead should work because the ROI is different, but contained in the ROI of the file
    smaller_roi = HealpixConeROI(data_radius=roi.data_radius / 2.0,
                                 model_radius=roi.model_radius,
                                 ra=ra_c - 0.2,
                                 dec=dec_c + 0.15)

    _ = map_tree_factory(test_filename, smaller_roi)

    check_map_trees(m, m2)

    # Now test reading without ROI
    m = map_tree_factory(maptree, None)

    os.remove(test_filename)