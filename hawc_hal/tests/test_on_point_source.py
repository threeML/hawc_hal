from hawc_hal import HAL, HealpixConeROI
from threeML import *
import time
import os
import argparse

test_data_path = os.environ['HAL_TEST_DATA']


ra_crab, dec_crab = 83.633083, 22.014500


def test_on_point_source(ra=ra_crab, dec=dec_crab, liff=False,
                         maptree=os.path.join(test_data_path, "maptree_1024.root"),
                         response=os.path.join(test_data_path, "response.root"),
                         data_radius=5.0, model_radius=10.0):

    if not liff:

        roi = HealpixConeROI(data_radius=data_radius,
                             model_radius=model_radius,
                             ra=ra,
                             dec=dec)

        # This is a 3ML plugin
        hawc = HAL("HAWC",
                   maptree,
                   response,
                   roi)

    else:

        from threeML import HAWCLike

        hawc = HAWCLike("HAWC",
                        os.path.join(test_data_path, "maptree_1024.root"),
                        os.path.join(test_data_path, "response.root"),
                        fullsky=True)

        hawc.set_ROI(ra, dec, data_radius)

    hawc.set_active_measurements(1, 9)

    if not liff: hawc.display()

    spectrum = Log_parabola()

    source = PointSource("pts", ra=ra, dec=dec, spectral_shape=spectrum)

    # NOTE: if you use units, you have to set up the values for the parameters
    # AFTER you create the source, because during creation the function Log_parabola
    # gets its units

    source.position.ra.bounds = (ra - 0.5, ra + 0.5)
    source.position.dec.bounds = (dec - 0.5, dec + 0.5)

    if ra==ra_crab:

        spectrum.piv = 10 * u.TeV  # Pivot energy as in the paper of the Crab

    else:

        spectrum.piv = 1 * u.TeV  # Pivot energy

    spectrum.piv.fix = True

    spectrum.K = 1e-14 / (u.TeV * u.cm ** 2 * u.s)  # norm (in 1/(keV cm2 s))
    spectrum.K.bounds = (1e-25, 1e-19)  # without units energies are in keV

    spectrum.beta = 0  # log parabolic beta
    spectrum.beta.bounds = (-4., 2.)

    spectrum.alpha = -2.5  # log parabolic alpha (index)
    spectrum.alpha.bounds = (-4., 2.)

    model = Model(source)

    data = DataList(hawc)

    jl = JointLikelihood(model, data, verbose=False)

    beg = time.time()

    param_df, like_df = jl.fit()

    print("Fit time: %s" % (time.time() - beg))

    return param_df, like_df

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--liff", help="Use LIFF instead of HAL (for benchmarking)", action="store_true")
    parser.add_argument("--ra", help="RA of source", type=float, default=ra_crab)
    parser.add_argument("--dec", help="Dec of source", type=float, default=dec_crab)
    args = parser.parse_args()

    test_on_point_source(ra=args.ra, dec=args.dec, liff=args.liff)
