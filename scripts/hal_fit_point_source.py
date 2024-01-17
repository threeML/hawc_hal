#!/usr/bin/env python

# If ROOT is active, we need to skip its own command line parsing
from builtins import map, range

try:
    import ROOT

    ROOT.PyConfig.IgnoreCommandLineOptions = True

except ImportError:
    pass

try:
    from hawc import hawcnest

    hawcnest.SetLoggingLevel(2, True)

except ImportError:
    pass

import argparse

import astromodels
import astropy.units as u
import pandas as pd
from astromodels import Model, PointSource
from threeML import plot_point_source_spectra

from hawc_hal import HealpixConeROI
from hawc_hal.convenience_functions.fit_point_source import fit_point_source

pd.options.display.max_columns = 80


def hal_fit_point_source(args: argparse.Namespace) -> None:
    if args.ra_roi is None:
        args.ra_roi = args.ra

    if args.dec_roi is None:
        args.dec_roi = args.dec

    roi = HealpixConeROI(
        args.data_radius, args.model_radius, ra=args.ra_roi, dec=args.dec_roi
    )

    # Build point source model

    # Get spectrum function
    spectrum = astromodels.get_function_class(args.spectrum)()

    src = PointSource("pts", ra=args.ra, dec=args.dec, spectral_shape=spectrum)

    model = Model(src)

    # Parse parameters
    tokens = args.params.split(",")

    for token in tokens:
        key, value = token.split("=")

        # The path of this parameter in the final model will be:
        path = "pts.spectrum.main.%s.%s" % (args.spectrum, key.strip())

        assert path in model, "Could not find parameter %s in function %s" % (
            key,
            args.spectrum,
        )

        p = model[path]

        p.value = u.Quantity(value)

        if p.is_normalization:
            p.bounds = (p.value / 100.0, p.value * 100)

    if args.display_model:
        model.display()

    param_df, like_df, ci, results = fit_point_source(
        roi,
        args.maptree,
        args.response,
        model,
        args.bin_list,
        args.confidence_intervals,
        args.use_liff,
        args.pixel_size,
        args.verbose,
    )

    if args.spectrum_plot is not None:
        fig = plot_point_source_spectra(
            results,
            ene_min=0.1 * u.TeV,
            ene_max=100 * u.TeV,
            flux_unit="erg/(cm2 s)",
            show_legend=False,
        )

        fig.savefig(args.spectrum_plot)


def main() -> None:
    help = """
    
    Example:
        
        - Fit the Crab with a power law spectrum starting from a normalization of 2.6e-11 TeV^-1 cm^-2 s^-1 and a 
          photon index of -2.0:
            
            > hal_fit_point_source.py --ra 83.633080 --dec 22.01450 --spectrum Powerlaw --params 'K=2.6e-11 1 / (TeV cm2 s), index=-2.0'
    
    """
    parser = argparse.ArgumentParser(description="bla", epilog=help)

    parser.add_argument(
        "--ra_roi", help="R.A. of center of ROI", type=float, required=False, default=None
    )
    parser.add_argument(
        "--dec_roi", help="Dec of center of ROI", type=float, required=False, default=None
    )
    parser.add_argument(
        "--ra", help="R.A. of source", type=float, required=True, default=None
    )
    parser.add_argument(
        "--dec", help="Dec of source", type=float, required=True, default=None
    )
    parser.add_argument(
        "--data_radius",
        help="Radius for data selection",
        type=float,
        required=False,
        default=3.0,
    )
    parser.add_argument(
        "--model_radius",
        help="Radius for model computation",
        type=float,
        required=False,
        default=8.0,
    )
    parser.add_argument(
        "--pixel_size",
        help="Size of the 'flat sky' pixels",
        type=float,
        required=False,
        default=0.17,
    )
    parser.add_argument(
        "--maptree", help="Path to maptree (root or hd5)", type=str, required=True
    )
    parser.add_argument(
        "--response", help="Path to response (root or hd5)", type=str, required=True
    )
    parser.add_argument(
        "--spectrum",
        help="Spectral model, for example Powerlaw. "
        "Any astromodels function is accepted",
        required=False,
        default="Power_law",
    )
    parser.add_argument(
        "--params",
        help="Parameter specification. For example, for the Power_law spectrum, "
        "you can use: '2e.5e-11 1/(TeV cm2 s)' '-2.0'",
        required=True,
    )
    parser.add_argument(
        "--bin_list",
        help="Bin list to use for analysis",
        required=False,
        default=list(map(str, list(range(1, 10)))),
        nargs="+",
    )
    parser.add_argument(
        "--display_model",
        help="Whether to display or not the model before fitting",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--use_liff", help="Use LiFF instead of HAL", action="store_true", default=False
    )
    parser.add_argument(
        "--confidence_intervals",
        help="Compute confidence intervals with the profile likelihood method",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--verbose",
        help="Print all the steps of the likelihood maximization",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--spectrum_plot",
        help="Name for output spectrum file (.png). If not provided, no plot will be made",
        type=str,
        required=False,
        default=None,
    )

    configured_args = parser.parse_args()
    hal_fit_point_source(configured_args)


if __name__ == "__main__":
    main()
