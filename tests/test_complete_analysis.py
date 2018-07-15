from hawc_hal import HAL
import matplotlib.pyplot as plt
from threeML import *
import pytest
from conftest import point_source_model


@pytest.fixture(scope='module')
def test_fit(roi, maptree, response):

    pts_model = point_source_model()

    hawc = HAL("HAWC",
               maptree,
               response,
               roi)

    # Use from bin 1 to bin 9
    hawc.set_active_measurements(1, 9)

    # Display information about the data loaded and the ROI
    hawc.display()

    # Get the likelihood value for the saturated model
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

    # See the model in counts space and the residuals
    fig = hawc.display_spectrum()
    # Save it to file
    fig.savefig("hal_src_residuals.png")

    # Look at the data
    fig = hawc.display_stacked_image(smoothing_kernel_sigma=0.17)
    # Save to file
    fig.savefig("hal_src_stacked_image.png")

    # Look at the different energy planes (the columns are model, data, residuals)
    fig = hawc.display_fit(smoothing_kernel_sigma=0.3)
    fig.savefig("hal_src_fit_planes.png")

    fig = hawc.display_fit(smoothing_kernel_sigma=0.3, display_colorbar=True)
    fig.savefig("hal_src_fit_planes_colorbar.png")


def test_compute_TS(test_fit):

    jl, hawc, pts_model, param_df, like_df, data = test_fit

    # Compute TS
    src_name = pts_model.pts.name
    jl.compute_TS(src_name, like_df)


def test_goodness(test_fit):

    jl, hawc, pts_model, param_df, like_df, data = test_fit

    # Compute goodness of fit with Monte Carlo
    gf = GoodnessOfFit(jl)
    gof, param, likes = gf.by_mc(10)
    print("Prob. of obtaining -log(like) >= observed by chance if null hypothesis is true: %.2f" % gof['HAWC'])

    # it is a good idea to inspect the results of the simulations with some plots
    # Histogram of likelihood values
    fig, sub = plt.subplots()
    likes.hist(ax=sub)
    # Overplot a vertical dashed line on the observed value
    plt.axvline(jl.results.get_statistic_frame().loc['HAWC', '-log(likelihood)'],
                color='black',
                linestyle='--')
    fig.savefig("hal_sim_all_likes.png")

    # Plot the value of beta for all simulations (for example)
    fig, sub = plt.subplots()
    param.loc[(slice(None), ['pts.spectrum.main.Cutoff_powerlaw.index']), 'value'].plot()
    fig.savefig("hal_sim_all_index.png")


def test_fit_with_free_position(test_fit):

    jl, hawc, pts_model, param_df, like_df, data = test_fit

    hawc.psf_integration_method = 'fast'

    # Free the position of the source
    pts_model.pts.position.ra.free = True
    pts_model.pts.position.dec.free = True

    # Set boundaries (no need to go further than this)
    ra = pts_model.pts.position.ra.value
    dec = pts_model.pts.position.dec.value
    pts_model.pts.position.ra.bounds = (ra - 0.5, ra + 0.5)
    pts_model.pts.position.dec.bounds = (dec - 0.5, dec + 0.5)

    # Fit with position free
    param_df, like_df = jl.fit()

    # Make localization contour

    # pts.position.ra(8.362 + / - 0.00028) x
    # 10
    # deg
    # pts.position.dec(2.214 + / - 0.00025) x
    # 10

    a, b, cc, fig = jl.get_contours(pts_model.pts.position.dec, 22.13, 22.1525, 5,
                                    pts_model.pts.position.ra, 83.615, 83.635, 5)

    plt.plot([ra], [dec], 'x')
    fig.savefig("hal_src_localization.png")

    hawc.psf_integration_method = 'exact'

    pts_model.pts.position.ra.free = False
    pts_model.pts.position.dec.free = False


def test_bayesian_analysis(test_fit):

    jl, hawc, pts_model, param_df, like_df, data = test_fit

    # Of course we can also do a Bayesian analysis the usual way

    pts_model.pts.position.ra.free = False
    pts_model.pts.position.dec.free = False

    # For this quick example, let's use a uniform prior for all parameters
    for parameter in pts_model.parameters.values():

        if parameter.fix:
            continue

        if parameter.is_normalization:

            parameter.set_uninformative_prior(Log_uniform_prior)

        else:

            parameter.set_uninformative_prior(Uniform_prior)

    # Let's execute our bayes analysis
    bs = BayesianAnalysis(pts_model, data)
    _ = bs.sample(30, 20, 20)
    fig = bs.corner_plot()

    fig.savefig("hal_corner_plot.png")

