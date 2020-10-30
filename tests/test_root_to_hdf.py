from hawc_hal.maptree.map_tree import map_tree_factory
from hawc_hal.response import hawc_response_factory
import os
import pytest
from conftest import check_map_trees, check_responses

try:
    import ROOT
except:
    has_root = False
else:
    has_root = True

skip_if_ROOT_is_not_available = pytest.mark.skipif(
    not has_root, reason="No ROOT available"
)

@skip_if_ROOT_is_not_available
def test_root_to_hdf_response(response):

    r = hawc_response_factory(response)

    test_filename = "response.hd5"

    # Make sure it doesn't exist yet, if it does,remove it
    if os.path.exists(test_filename):
        os.remove(test_filename)

    r.write(test_filename)

    # Try to open and use it
    r2 = hawc_response_factory(test_filename)

    check_responses(r, r2)

    os.remove(test_filename)

def do_one_test_maptree(geminga_roi,
                        geminga_maptree,
                        fullsky=False):
    # Test both with a defined ROI and full sky (ROI is None)
    if fullsky:

        roi_ = None

    else:

        roi_ = geminga_roi

    m = map_tree_factory(geminga_maptree, roi_)

    test_filename = "maptree.hd5"

    # Make sure it doesn't exist yet, if it does,remove it
    if os.path.exists(test_filename):
        os.remove(test_filename)

    m.write(test_filename)

    # Try to open and use it
    m2 = map_tree_factory(test_filename, roi_)

    check_map_trees(m, m2)

    os.remove(test_filename)

@skip_if_ROOT_is_not_available
def test_root_to_hdf_maptree_roi(geminga_roi,
                                 geminga_maptree):
    do_one_test_maptree(geminga_roi,
                        geminga_maptree,
                        fullsky=False)

@skip_if_ROOT_is_not_available
def test_root_to_hdf_maptree_full_sky(geminga_roi,
                                      geminga_maptree):
    do_one_test_maptree(geminga_roi,
                        geminga_maptree,
                        fullsky=True)
