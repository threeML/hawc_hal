import pytest
from hawc_hal import HealpixConeROI
from threeML import *
import os
from hawc_hal import HAL
import time


# Get data path
test_data_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'data')


def fit_point_source(roi,
                      maptree,
                      response,
                      point_source_model,
                      liff=False):
    data_radius = roi.data_radius.to("deg").value

    if not liff:

        # This is a 3ML plugin
        hawc = HAL("HAWC",
                   maptree,
                   response,
                   roi)

    else:

        from threeML import HAWCLike

        hawc = HAWCLike("HAWC",
                        maptree,
                        response,
                        fullsky=True)

        ra_roi, dec_roi = roi.ra_dec_center

        hawc.set_ROI(ra_roi, dec_roi, data_radius)

    hawc.set_active_measurements(1, 9)

    if not liff: hawc.display()

    data = DataList(hawc)

    jl = JointLikelihood(point_source_model, data, verbose=False)

    point_source_model.display(complete=True)

    beg = time.time()

    # jl.set_minimizer("ROOT")

    param_df, like_df = jl.fit()

    # _ = jl.get_errors()

    print("Fit time: %s" % (time.time() - beg))

    return param_df, like_df


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


@pytest.fixture(scope="session", autouse=True)
def roi():

    # zebra-source-injector -i /home/giacomov/science/hawc/data/zebra/combined_bin{0..9}.fits.gz -o bin{0..9}.fits.gz
    # -b {0..9}
    # -s CutoffPowerLaw,2.62e-11,2.29,42.7
    # --ra 83.6279 --dec 22.14
    # --dr zebra_response.root
    # --usebackground --padding 10 --pivot 1.0

    # skymaps-fits2maptree --input bin?.fits.gz -n 0 -o zebra_simulated_source_mt.root

    ra_sim_source, dec_sim_source = 83.6279, 22.14
    data_radius = 5.0
    model_radius = 10.0

    # NOTE: the center of the ROI is not exactly on the source. This is on purpose, to make sure that we are
    # doing the model map with the right orientation

    roi = HealpixConeROI(data_radius=data_radius,
                         model_radius=model_radius,
                         ra=ra_sim_source - 1.0,
                         dec=dec_sim_source + 0.5)

    return roi


@pytest.fixture(scope="session", autouse=True)
def geminga_roi():

    ra_c, dec_c, rad = 101.7, 16, 9.

    # NOTE: the center of the ROI is not exactly on the source. This is on purpose, to make sure that we are
    # doing the model map with the right orientation

    roi = HealpixConeROI(data_radius=rad,
                         model_radius=rad + 15.0,
                         ra=ra_c,
                         dec=dec_c)

    return roi


@pytest.fixture(scope="session", autouse=True)
def geminga_maptree():

    return os.path.join(test_data_path, 'geminga_maptree.root')


@pytest.fixture(scope="session", autouse=True)
def geminga_response():

    return os.path.join(test_data_path, 'geminga_response.root')


@pytest.fixture(scope="session", autouse=True)
def maptree():

    return os.path.join(test_data_path, 'zebra_simulated_source_mt_roi.hd5')


@pytest.fixture(scope="session", autouse=True)
def response():

    return os.path.join(test_data_path, 'detector_response.root')


@pytest.fixture(scope="function", autouse=True)
def point_source_model(ra=83.6279, dec=22.14):

    spectrum = Cutoff_powerlaw()

    source = PointSource("pts", ra=ra, dec=dec, spectral_shape=spectrum)

    # NOTE: if you use units, you have to set up the values for the parameters
    # AFTER you create the source, because during creation the function Log_parabola
    # gets its units

    source.position.ra.bounds = (ra - 0.5, ra + 0.5)
    source.position.dec.bounds = (dec - 0.5, dec + 0.5)
    source.position.ra.free = False
    source.position.dec.free = False

    spectrum.piv = 1 * u.TeV  # Pivot energy

    spectrum.piv.fix = True

    spectrum.K = 3.15e-11 / (u.TeV * u.cm ** 2 * u.s)  # norm (in 1/(keV cm2 s))
    spectrum.K.bounds = (1e-25, 1e-19)  # without units energies are in keV

    spectrum.index = -2.0
    spectrum.bounds = (-5, 0.0)

    spectrum.xc = 42.7 * u.TeV
    spectrum.xc.fix = False
    spectrum.xc.bounds = (1 * u.TeV, 100 * u.TeV)

    model = Model(source)

    return model