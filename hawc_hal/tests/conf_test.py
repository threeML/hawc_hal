import numpy as np

def check_map_trees(m1, m2):

    for p1, p2 in zip(m1, m2):

        assert np.allclose(p1.observation_map.as_partial(), p2.observation_map.as_partial())
        assert np.allclose(p1.background_map.as_partial(), p2.background_map.as_partial())

        assert p1.nside == p2.nside
        assert p1.n_transits == p2.n_transits