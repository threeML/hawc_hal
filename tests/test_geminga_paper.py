from __future__ import print_function
from hawc_hal import HAL, HealpixConeROI

try:
    import ROOT
    ROOT.PyConfig.IgnoreCommandLineOptions = True
except:
    pass

from threeML import *
import argparse
from collections import namedtuple
import pytest

def test_geminga_paper(geminga_maptree, geminga_response):

    Args_fake = namedtuple('args', 'mtfile,rsfile,startBin,stopBin,RA,Dec,uratio,delta,ROI,output,plugin')

    args = Args_fake(geminga_maptree, geminga_response, 1, 9, 98.5, 17.76, 1.12, 0.3333, 0, 'output.txt', 'new')

    param, like_df, TS = go(args)

    # Geminga...rdiff0                                    5.400 +/- 0.6              deg
    # Geminga.spectrum.main.Powerlaw.K      (1.40 -0.17 +0.20) x 10^-23  1 / (cm2 keV s)
    # Geminga.spectrum.main.Powerlaw.index              -2.340 +/- 0.07
    # B0656.spectrum.main.Powerlaw.K           (6.0 -1.3 +1.7) x 10^-24  1 / (cm2 keV s)
    # B0656.spectrum.main.Powerlaw.index                -2.140 +/- 0.17

    assert np.allclose(param.loc[:, 'value'].values,
                       [5.38278446e+00, 1.40122099e-23, -2.34261469e+00, 5.98438658e-24, -2.13846297e+00],
                       rtol=5e-2)


def go(args):

    spectrum = Powerlaw()
    shape = Continuous_injection_diffusion_legacy()

    source = ExtendedSource("Geminga",
                            spatial_shape=shape,
                            spectral_shape=spectrum)

    fluxUnit = 1. / (u.TeV * u.cm ** 2 * u.s)

    # Set spectral parameters
    spectrum.K = 1e-14 * fluxUnit
    spectrum.K.bounds = (1e-16 * fluxUnit, 1e-12 * fluxUnit)

    spectrum.piv = 20 * u.TeV
    spectrum.piv.fix = True

    spectrum.index = -2.4
    spectrum.index.bounds = (-4., -1.)

    # Set spatial parameters
    shape.lon0 = args.RA * u.degree
    shape.lon0.fix = True

    shape.lat0 = args.Dec * u.degree
    shape.lat0.fix = True

    shape.rdiff0 = 6.0 * u.degree
    shape.rdiff0.fix = False
    shape.rdiff0.max_value = 12.

    shape.delta.min_value = 0.1
    shape.delta = args.delta
    shape.delta.fix = True

    shape.uratio = args.uratio
    shape.uratio.fix = True

    shape.piv = 2e10
    shape.piv.fix = True

    shape.piv2 = 1e9
    shape.piv2.fix = True

    spectrum2 = Powerlaw()
    shape2 = Continuous_injection_diffusion_legacy()

    source2 = ExtendedSource("B0656",
                             spatial_shape=shape2,
                             spectral_shape=spectrum2)

    fluxUnit = 1. / (u.TeV * u.cm ** 2 * u.s)

    # Set spectral parameters for the 2nd source
    spectrum2.K = 1e-14 * fluxUnit
    spectrum2.K.bounds = (1e-16 * fluxUnit, 1e-12 * fluxUnit)

    spectrum2.piv = 20 * u.TeV
    spectrum2.piv.fix = True

    spectrum2.index = -2.2
    # spectrum2.index.fix = True
    spectrum2.index.bounds = (-4., -1.)

    # Set spatial parameters for the 2nd source
    shape2.lon0 = 104.95 * u.degree
    shape2.lon0.fix = True

    shape2.lat0 = 14.24 * u.degree
    shape2.lat0.fix = True

    shape2.rdiff0 = 6.0 * u.degree
    shape2.rdiff0.fix = False
    shape2.rdiff0.max_value = 12.

    shape2.delta.min_value = 0.2
    shape2.delta = args.delta
    shape2.delta.fix = True

    shape2.uratio = args.uratio
    shape2.uratio.fix = True

    shape2.piv = 2e10
    shape2.piv.fix = True

    shape2.piv2 = 1e9
    shape2.piv2.fix = True

    # Set up a likelihood model using the source
    if args.ROI == 0:
        lm = Model(source, source2)
    elif args.ROI == 1 or args.ROI == 2:
        lm = Model(source)
    elif args.ROI == 3 or args.ROI == 4:
        lm = Model(source2)

    ra_c, dec_c, rad = (None, None, None)

    if args.ROI == 0:
        ra_c, dec_c, rad = 101.7, 16, 9.
        # llh.set_ROI(101.7, 16, 9., True)
    elif args.ROI == 1:
        ra_c, dec_c, rad = 98.5, 17.76, 4.5
        # llh.set_ROI(98.5, 17.76, 4.5, True)
    elif args.ROI == 2:
        ra_c, dec_c, rad = 97, 18.5, 6
        # llh.set_ROI(97, 18.5, 6, True)
    elif args.ROI == 3:
        ra_c, dec_c, rad = 104.95, 14.24, 3.
        # llh.set_ROI(104.95, 14.24, 3., True)
    elif args.ROI == 4:
        ra_c, dec_c, rad = 107, 13, 5.4
        # llh.set_ROI(107, 13, 5.4, True)

    if args.plugin == 'old':

        llh = HAWCLike("Geminga", args.mtfile, args.rsfile, fullsky=True)
        llh.set_active_measurements(args.startBin, args.stopBin)
        llh.set_ROI(ra_c, dec_c, rad, True)

    else:

        roi = HealpixConeROI(data_radius=rad,
                             model_radius=rad + 10.0,
                             ra=ra_c, dec=dec_c)
    
        llh = HAL("HAWC",
                  args.mtfile,
                  args.rsfile,
                  roi)
            
        llh.set_active_measurements(args.startBin, args.stopBin)

    print(lm)

    # we fit a common diffusion coefficient so parameters are linked
    if "B0656" in lm and "Geminga" in lm:
        law = Line(a=250. / 288., b=0.)
        lm.link(lm.B0656.spatial_shape.rdiff0,
                lm.Geminga.spatial_shape.rdiff0,
                law)
        lm.B0656.spatial_shape.rdiff0.Line.a.fix = True
        lm.B0656.spatial_shape.rdiff0.Line.b.fix = True

    # Double check the free parameters
    print("Likelihood model:\n")
    print(lm)

    # Set up the likelihood and run the fit
    TS = 0.

    try:

        lm.Geminga.spatial_shape.rdiff0 = 5.5
        lm.Geminga.spatial_shape.rdiff0.fix = False
        lm.Geminga.spectrum.main.Powerlaw.K = 1.36e-23
        lm.Geminga.spectrum.main.Powerlaw.index = -2.34

    except:

        pass

    try:

        lm.B0656.spectrum.main.Powerlaw.K = 5.7e-24
        lm.B0656.spectrum.main.Powerlaw.index = -2.14

    except:

        pass

    print("Performing likelihood fit...\n")
    datalist = DataList(llh)
    jl = JointLikelihood(lm, datalist, verbose=True)

    try:
        jl.set_minimizer("minuit")
        param, like_df = jl.fit(compute_covariance=False)
    except AttributeError:
        jl.set_minimizer("minuit")
        param, like_df = jl.fit(compute_covariance=False)

    # Print the TS, significance, and fit parameters, and then plot stuff
    print("\nTest statistic:")

    if args.plugin == 'old':

        TS = llh.calc_TS()

    else:
    
        if "B0656" in lm and "Geminga" in lm:
            lm.unlink(lm.B0656.spatial_shape.rdiff0)
        
        TS_df = jl.compute_TS("Geminga", like_df)
        TS = TS_df.loc['HAWC', 'TS']
        
        TS_df2 = jl.compute_TS("B0656", like_df)
        TS2 = TS_df2.loc['HAWC', 'TS']
        
        print (TS_df)
        print (TS_df2)

    print("Test statistic: %g" % TS)

    freepars = []
    fixedpars = []
    for p in lm.parameters:
        par = lm.parameters[p]
        if par.free:
            freepars.append("%-45s %35.6g %s" % (p, par.value, par._unit))
        else:
            fixedpars.append("%-45s %35.6g %s" % (p, par.value, par._unit))

    # Output TS, significance, and parameter values to a file
    with open(args.output, "w") as f:
        f.write("Test statistic: %g\n" % TS)

        f.write("\nFree parameters:\n")
        for l in freepars:
            f.write("%s\n" % l)
        f.write("\nFixed parameters:\n")
        for l in fixedpars:
            f.write("%s\n" % l)

    return param, like_df, TS


if __name__ == "__main__":

    p = argparse.ArgumentParser(description="Example spectral fit with LiFF")
    p.add_argument("-m", "--maptreefile", dest="mtfile",
                   help="LiFF MapTree ROOT file", default="./maptree.root")
    p.add_argument("-r", "--responsefile", dest="rsfile",
                   help="LiFF detector response ROOT file", default="./response.root")
    p.add_argument("--startBin", dest="startBin", default=1, type=int,
                   help="Starting analysis bin [0..9]")
    p.add_argument("--stopBin", dest="stopBin", default=9, type=int,
                   help="Stopping analysis bin [0..9]")
    p.add_argument("--RA", default=98.5, type=float,
                   help="Source RA in degrees (Geminga default)")
    p.add_argument("--Dec", default=17.76, type=float,
                   help="Source Dec in degrees (Geminga default)")
    p.add_argument("--uratio", dest="uratio", default=1.12, type=float,
                   help="the ratio of energy density between CMB and B. 1.12 means B=3uG and CMB=0.25")
    p.add_argument("--delta", dest="delta", default=0.3333, type=float,
                   help="Diffusion spectral index (0.3 to 0.6)")
    p.add_argument("--ROI", dest="ROI", default=0, type=int)
    p.add_argument("-o", "--output", dest="output", default="output.txt",
                   help="Parameter output file.")
    p.add_argument("--plugin", dest='plugin', default='old', type=str,
                   help="Old or new", choices=['new', 'old'])

    args = p.parse_args()

    go(args)
