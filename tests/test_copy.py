import copy
import pytest

from hawc_hal import HAL, HealpixConeROI
from threeML import *
import astromodels

from conftest import maptree, response

def deepcopy_hal(theMaptree, theResponse, extended=False):

    src_ra, src_dec = 82.628, 22.640
    src_name = 'test_source'

    roi = HealpixConeROI(data_radius=5., model_radius=8., ra=src_ra, dec=src_dec)

    hawc = HAL('HAWC', theMaptree, theResponse, roi)
    hawc.set_active_measurements(1, 9)
    data = DataList(hawc)

    # Define model
    spectrum = Log_parabola()
    if not extended:
        source = PointSource(src_name, ra=src_ra, dec=src_dec, spectral_shape=spectrum)
    else:
        shape = astromodels.Gaussian_on_sphere()
        source = ExtendedSource(src_name, spatial_shape=shape, spectral_shape=spectrum)

    model = Model(source)

    jl = JointLikelihood(model, data, verbose=False)

    hawc_copy = copy.deepcopy(hawc)

def test_deepcopy_point_source(maptree, response):
    deepcopy_hal(maptree, response, extended=False)

@pytest.mark.xfail
def test_deepcopy_extended_source(maptree, response):
    deepcopy_hal(maptree, response, extended=True)
