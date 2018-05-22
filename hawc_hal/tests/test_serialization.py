from hawc_hal import serialize
from hawc_hal import map_tree
from hawc_hal import HealpixConeROI
import pandas as pd
import numpy as np
import os
import pytest

test_data_path = os.environ['HAL_TEST_DATA']


def test_serialization():

    my_obj = pd.Series(range(100))
    meta = {'meta1': 1.0, 'meta2': 2.0}

    test_file = 'test.hd5'

    with serialize.Serialization(test_file) as store:

        store.store_pandas_object('test_obj', my_obj, **meta)

    assert os.path.exists(test_file)

    # Now retrieve the object
    with serialize.Serialization(test_file) as store:

        copy_obj, copy_meta = store.retrieve_pandas_object('test_obj')

    assert np.alltrue(my_obj == copy_obj)

    for k in meta:

        assert meta[k] == copy_meta[k]

    os.remove(test_file)


def test_serializing_maptree():

    ra_mkn421, dec_mkn421 = 166.113808, 38.208833
    data_radius = 3.0
    model_radius = 8.0

    roi = HealpixConeROI(data_radius=data_radius,
                         model_radius=model_radius,
                         ra=ra_mkn421,
                         dec=dec_mkn421)

    maptree_file = os.path.join(test_data_path, 'maptree_1024.root')

    maptree = map_tree.map_tree_factory(maptree_file, roi)

    test_file = 'test.hd5'

    maptree.write(test_file)

    assert os.path.exists(test_file)

    maptree2 = map_tree.map_tree_factory(test_file, roi)

    # Try to load with a different ROI
    oroi = HealpixConeROI(10.0, 10.0, ra=ra_mkn421, dec=dec_mkn421)

    with pytest.raises(AssertionError):

        _ = map_tree.map_tree_factory(test_file, oroi)

    # Check that all planes are the same
    for p1, p2 in zip(maptree, maptree2):

        assert np.allclose(p1.observation_map.as_partial(), p2.observation_map.as_partial())
        assert np.allclose(p1.background_map.as_partial(), p2.background_map.as_partial())

        assert p1.nside == p2.nside
        assert p1.n_transits == p2.n_transits



