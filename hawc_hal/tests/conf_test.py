import numpy as np


def check_map_trees(m1, m2):

    for p1, p2 in zip(m1, m2):

        assert np.allclose(p1.observation_map.as_partial(), p2.observation_map.as_partial())
        assert np.allclose(p1.background_map.as_partial(), p2.background_map.as_partial())

        assert p1.nside == p2.nside
        assert p1.n_transits == p2.n_transits


def check_responses(r1, r2):

    assert len(r1.response_bins) == len(r2.response_bins)

    for resp_key in r1.response_bins.keys():

        rbb1 = r1.response_bins[resp_key]
        rbb2 = r2.response_bins[resp_key]

        for rb1, rb2 in zip(rbb1, rbb2):

            assert rb1.name == rb2.name

            assert rb1.declination_boundaries == rb2.declination_boundaries

            assert rb1.declination_center == rb2.declination_center

            assert rb1.n_sim_signal_events == rb2.n_sim_signal_events

            assert rb1.n_sim_bkg_events == rb2.n_sim_bkg_events

            assert np.allclose(rb1.sim_energy_bin_centers, rb2.sim_energy_bin_centers)

            assert np.allclose(rb1.sim_differential_photon_fluxes, rb2.sim_differential_photon_fluxes)

            assert np.allclose(rb1.sim_signal_events_per_bin, rb2.sim_signal_events_per_bin)

            # Test PSF
            assert np.allclose(rb1.psf.xs, rb2.psf.xs)
            assert np.allclose(rb1.psf.ys, rb2.psf.ys)