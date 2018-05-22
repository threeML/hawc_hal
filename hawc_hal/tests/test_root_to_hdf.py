from hawc_hal.map_tree import map_tree_factory
from hawc_hal import HealpixConeROI
import os
import numpy as np
from test_on_point_source import test_on_point_source
from conf_test import check_map_trees

test_data_path = os.environ['HAL_TEST_DATA']


def test_root_to_hdf_maptree():

    root_map_tree = os.path.join(test_data_path, "maptree_1024.root")

    ra_mkn421, dec_mkn421 = 166.113808, 38.208833
    data_radius = 3.0
    model_radius = 8.0

    roi = HealpixConeROI(data_radius=data_radius,
                         model_radius=model_radius,
                         ra=ra_mkn421,
                         dec=dec_mkn421)

    # Test both with a defined ROI and full sky (ROI is None)
    for roi_ in [roi, None]:

        m = map_tree_factory(root_map_tree, roi_)

        test_filename = "test.hd5"

        # Make sure it doesn't exist yet, if it does,remove it
        if os.path.exists(test_filename):

            os.remove(test_filename)

        m.write(test_filename)

        # Try to open and use it
        m2 = map_tree_factory(test_filename, roi_)

        check_map_trees(m, m2)

        # Make a simple analysis with both the original and the reloaded trees

        orig_par, orig_like = test_on_point_source(ra=ra_mkn421, dec=dec_mkn421,
                                                   maptree=root_map_tree, data_radius=data_radius,
                                                   model_radius=model_radius)

        new_par, new_like = test_on_point_source(ra=ra_mkn421, dec=dec_mkn421,
                                                 maptree=test_filename,
                                                 data_radius=data_radius, model_radius=model_radius)

        # Make sure the results are identical
        assert np.allclose(orig_par.loc[:, 'value'], new_par.loc[:, 'value'])
        assert np.allclose(orig_like.loc[:, '-log(likelihood)'], new_like.loc[:, '-log(likelihood)'])

        os.remove(test_filename)
