from builtins import range
from hawc_hal import serialize
import pandas as pd
import numpy as np
import os


def test_serialization():

    my_obj = pd.Series(list(range(100)))
    my_ob2 = pd.DataFrame.from_dict({'one': np.random.uniform(0, 1, 10), 'two': np.random.uniform(0, 1, 10)})

    meta = {'meta1': 1.0, 'meta2': 2.0}

    test_file = 'test.hd5'

    with serialize.Serialization(test_file, mode='w') as store:

        store.store_pandas_object('test_obj', my_obj, **meta)
        store.store_pandas_object('test_df', my_ob2)

    assert os.path.exists(test_file)

    # Now retrieve the object
    with serialize.Serialization(test_file, mode='r') as store:

        copy_obj, copy_meta = store.retrieve_pandas_object('test_obj')
        copy_obj2, copy_meta2 = store.retrieve_pandas_object('test_df')

        assert set(store.keys) == set(('/test_obj','/test_df'))

    assert np.alltrue(my_obj == copy_obj)
    assert np.alltrue(my_ob2 == copy_obj2)

    for k in meta:

        assert meta[k] == copy_meta[k]

    assert len(copy_meta2)==0

    os.remove(test_file)




