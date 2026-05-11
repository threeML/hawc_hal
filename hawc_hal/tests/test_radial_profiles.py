from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
import pytest
from threeML import *
from tqdm import tqdm

from hawc_hal import HAL


@pytest.fixture(scope="module")
def test_fit(roi, maptree, response, point_source_model):

    pts_model = point_source_model

    hawc = HAL("HAWC", maptree, response, roi)

    hawc.set_active_measurements(1, 9)

    hawc.display()

    hawc.get_saturated_model_likelihood()

    data = DataList(hawc)

    jl = JointLikelihood(pts_model, data, verbose=True)

    param_df, like_df = jl.fit()

    return jl, hawc, pts_model, param_df, like_df, data


def test_simulation(test_fit):

    jl, hawc, pts_model, param_df, like_df, data = test_fit

    sim = hawc.get_simulated_dataset("HAWCsim")
    sim.write("sim_resp.hd5", "sim_maptree.hd5")


def test_plots(test_fit):

    jl, hawc, pts_model, param_df, like_df, data = test_fit

    ra = pts_model.pts.position.ra.value
    dec = pts_model.pts.position.dec.value
    radius = 2.0

    bins = hawc._active_planes

    fig, table = hawc.plot_radial_profile(ra, dec, bins, radius, n_radial_bins=15)
    fig.savefig("radial_profile_all_bins.png")
    table.to_hdf("radial_profile_all_bins.hd5", key="radial")
    plt.close(fig)

    prog_bar = tqdm(total=len(hawc._active_planes), desc="Smoothing planes")
    for bin in hawc._active_planes:
        fig, table = hawc.plot_radial_profile(ra, dec, f"{bin}", radius, n_radial_bins=15)
        fig.savefig(f"radial_profile_bin{bin}.png")
        table.to_hdf(f"radial_profile_bin{bin}.hd5", key="radial")
        plt.close(fig)
        prog_bar.update(1)


# ---------------------------------------------------------------------------
# Sector radial profile tests
# ---------------------------------------------------------------------------

_PROFILE_KWARGS = dict(max_radius=2.0, n_radial_bins=10)


@pytest.fixture(scope="module")
def sector_hawc(roi, maptree, response, point_source_model):
    hawc = HAL("HAWC_sector", maptree, response, roi)
    hawc.set_active_measurements(1, 9)
    hawc.set_model(point_source_model)
    return hawc, point_source_model


def test_sector_profile_returns_correct_shape(sector_hawc):
    hawc, model = sector_hawc
    ra = model.pts.position.ra.value
    dec = model.pts.position.dec.value

    radii, excess_model, excess_data, excess_error, plane_ids = (
        hawc._get_radial_profile_sector(ra, dec, phi_min=0.0, phi_max=180.0, **_PROFILE_KWARGS)
    )

    n_bins = _PROFILE_KWARGS["n_radial_bins"]
    assert radii.shape == (n_bins,)
    assert excess_model.shape == (n_bins,)
    assert excess_data.shape == (n_bins,)
    assert excess_error.shape == (n_bins,)
    assert len(plane_ids) > 0


def test_full_sector_matches_full_profile(sector_hawc):
    """phi_min=0 / phi_max=360 excludes no pixels — result must equal full profile."""
    hawc, model = sector_hawc
    ra = model.pts.position.ra.value
    dec = model.pts.position.dec.value

    full = hawc._get_radial_profile(ra, dec, **_PROFILE_KWARGS)
    sector_full = hawc._get_radial_profile_sector(
        ra, dec, phi_min=0.0, phi_max=360.0, **_PROFILE_KWARGS
    )

    # radii, model, data, error should all be identical
    for arr_full, arr_sector in zip(full[:4], sector_full[:4]):
        np.testing.assert_allclose(arr_full, arr_sector)


def test_sector_profile_differs_from_full_profile(sector_hawc):
    """A half-sector must yield different excess values than the full profile."""
    hawc, model = sector_hawc
    ra = model.pts.position.ra.value
    dec = model.pts.position.dec.value

    _, _, data_full, _, _ = hawc._get_radial_profile(ra, dec, **_PROFILE_KWARGS)
    _, _, data_north, _, _ = hawc._get_radial_profile_sector(
        ra, dec, phi_min=0.0, phi_max=180.0, **_PROFILE_KWARGS
    )

    assert not np.allclose(data_full, data_north), (
        "North sector should differ from full profile"
    )


def test_complementary_sectors_differ(sector_hawc):
    """North (0–180°) and South (180–360°) sectors should give different profiles."""
    hawc, model = sector_hawc
    ra = model.pts.position.ra.value
    dec = model.pts.position.dec.value

    _, _, data_north, _, _ = hawc._get_radial_profile_sector(
        ra, dec, phi_min=0.0, phi_max=180.0, **_PROFILE_KWARGS
    )
    _, _, data_south, _, _ = hawc._get_radial_profile_sector(
        ra, dec, phi_min=180.0, phi_max=360.0, **_PROFILE_KWARGS
    )

    assert not np.allclose(data_north, data_south), (
        "North and South sector profiles should differ"
    )


def test_sector_plot_radial_profile(sector_hawc):
    """plot_radial_profile with phi_min/phi_max should return a Figure and DataFrame."""
    import pandas as pd
    from matplotlib.figure import Figure

    hawc, model = sector_hawc
    ra = model.pts.position.ra.value
    dec = model.pts.position.dec.value

    for phi_min, phi_max in [(0.0, 90.0), (90.0, 180.0), (180.0, 270.0), (270.0, 360.0)]:
        fig, df = hawc.plot_radial_profile(
            ra, dec, phi_min=phi_min, phi_max=phi_max, **_PROFILE_KWARGS
        )

        assert isinstance(fig, Figure)
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["Excess", "Error", "Model"]
        assert len(df) == _PROFILE_KWARGS["n_radial_bins"]

        fig.savefig(f"radial_profile_sector_phi{int(phi_min)}_to_{int(phi_max)}deg.png")
        df.to_hdf(f"radial_profile_sector_phi{int(phi_min)}_to_{int(phi_max)}deg.hd5", key="radial")
        plt.close(fig)
