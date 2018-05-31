from hawc_hal import HAL, HealpixConeROI
import matplotlib.pyplot as plt
from threeML import *
import argparse

test_data_path = os.environ['HAL_TEST_DATA']


def test_complete_analysis(roi,
                           maptree,
                           response,
                           point_source_model):
    # Define the ROI

    # Instance the plugin

    hawc = HAL("HAWC",
               maptree,
               response,
               roi)

    # Use from bin 1 to bin 9
    hawc.set_active_measurements(1, 9)

    # Display information about the data loaded and the ROI
    hawc.display()

    # Look at the data
    fig = hawc.display_stacked_image(smoothing_kernel_sigma=0.17)
    # Save to file
    fig.savefig("hal_src_stacked_image.png")

    data = DataList(hawc)

    jl = JointLikelihood(point_source_model, data, verbose=False)
    param_df, like_df = jl.fit()

    # See the model in counts space and the residuals
    fig = hawc.display_spectrum()
    # Save it to file
    fig.savefig("hal_src_residuals.png")

    # Look at the different energy planes (the columns are model, data, residuals)
    fig = hawc.display_fit(smoothing_kernel_sigma=0.3)
    fig.savefig("hal_src_fit_planes.png")

    # Compute TS
    src_name = point_source_model.pts.name
    jl.compute_TS(src_name, like_df)

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

    # Free the position of the source
    point_source_model.pts.position.ra.free = True
    point_source_model.pts.position.dec.free = True

    # Set boundaries (no need to go further than this)
    ra = point_source_model.pts.position.ra.value
    dec = point_source_model.pts.position.dec.value
    point_source_model.pts.position.ra.bounds = (ra - 0.5, ra + 0.5)
    point_source_model.pts.position.dec.bounds = (dec - 0.5, dec + 0.5)

    # Fit with position free
    param_df, like_df = jl.fit()

    # Make localization contour

    # pts.position.ra(8.362 + / - 0.00028) x
    # 10
    # deg
    # pts.position.dec(2.214 + / - 0.00025) x
    # 10

    a, b, cc, fig = jl.get_contours(point_source_model.pts.position.dec, 22.14, 22.16, 10,
                                    point_source_model.pts.position.ra, 83.60, 83.65, 10)

    plt.plot([ra], [dec], 'x')
    fig.savefig("hal_src_localization.png")

    # Of course we can also do a Bayesian analysis the usual way
    # NOTE: here the position is still free, so we are going to obtain marginals about that
    # as well
    # For this quick example, let's use a uniform prior for all parameters
    for parameter in point_source_model.parameters.values():

        if parameter.fix:
            continue

        if parameter.is_normalization:

            parameter.set_uninformative_prior(Log_uniform_prior)

        else:

            parameter.set_uninformative_prior(Uniform_prior)

    # Let's execute our bayes analysis
    bs = BayesianAnalysis(point_source_model, data)
    samples = bs.sample(30, 20, 20)
    fig = bs.corner_plot()

    fig.savefig("hal_corner_plot.png")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--maptree", help="Path to maptree", type=str,
                        default=os.path.join(test_data_path, "maptree_1024.root"))
    parser.add_argument("--response", help="Path to response", type=str,
                        default=os.path.join(test_data_path, "response.root"))
    args = parser.parse_args()

    test_complete_analysis(maptree=args.maptree, response=args.response)
