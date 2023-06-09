from __future__ import print_function

import matplotlib.pyplot as plt
import pandas as pd
import pytest
from conftest import point_source_model
from threeML import *
from tqdm import tqdm

from hawc_hal import HAL


@pytest.fixture(scope="module")
def test_fit(roi, maptree, response, point_source_model):

    pts_model = point_source_model
    bins = ['1', '2', '3', '4', '5', '6', '7', '8', '9']
    hawc = HAL("HAWC", maptree, response, roi, bin_list=bins)

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
    radius = 1.5

    bins = hawc._active_planes

    fig, table = hawc.plot_radial_profile(
        ra,
        dec,
        bins,
        radius,
        n_radial_bins=25,
    )
    fig.savefig("hal_src_radial_profile.png")
    table.to_hdf("hal_src_radial_table.hd5", key="radial")

    prog_bar = tqdm(total=len(hawc._active_planes), desc="Smoothing planes")
    for bin in hawc._active_planes:

        fig, table = hawc.plot_radial_profile(ra, dec, f"{bin}", radius)
        fig.savefig(f"hal_src_radial_profile_bin{bin}.png")

        table.to_hdf(f"hal_src_radial_table_{bin}.hd5", key="radial")
        # NOTE: ensure figures are closed to maintain memory
        # usage low

        plt.close(fig)

        prog_bar.update(1)
