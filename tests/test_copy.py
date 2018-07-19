import copy

from hawc_hal import HAL, HealpixConeROI
from threeML import *
import astromodels

from conftest import point_source_model, maptree, response

def deepcopy_hal(extended=False):

    src_ra, src_dec = 82.628, 22.640
    src_name = 'test_source'

    roi = HealpixConeROI(data_radius=5., model_radius=8., ra=src_ra, dec=src_dec)

    hawc = HAL('HAWC', maptree(), response(), roi)
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

def test_deepcopy_point_source():
    deepcopy_hal(extended=False)

def test_deepcopy_extended_source():
    deepcopy_hal(extended=True)
