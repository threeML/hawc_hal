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

    with serialize.Serialization(test_file, mode='w') as store:

        store.store_pandas_object('test_obj', my_obj, **meta)

    assert os.path.exists(test_file)

    # Now retrieve the object
    with serialize.Serialization(test_file, mode='r') as store:

        copy_obj, copy_meta = store.retrieve_pandas_object('test_obj')

    assert np.alltrue(my_obj == copy_obj)

    for k in meta:

        assert meta[k] == copy_meta[k]

    os.remove(test_file)




