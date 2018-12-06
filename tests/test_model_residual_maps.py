from hawc_hal import HAL, HealpixConeROI
import matplotlib.pyplot as plt
from threeML import *
#import os
from os.path import isfile, dirname, realpath
from hawc_hal.maptree import map_tree_factory



test_file = realpath(__file__)

file_dir = dirname(test_file)

output = "{0}/data".format(file_dir)

# Define the ROI
ra_src, dec_src = 101.75, 16.0

data_radius = 5.0
model_radius = 7.0

roi = HealpixConeROI(data_radius=data_radius,
                     model_radius=model_radius,
                     ra=ra_src,
                     dec=dec_src)

maptree = "{0}/geminga_maptree.root".format(output)
response = "{0}/detector_response.root".format(output)

assert isfile(maptree) == True
assert isfile(response) == True

hawc = HAL("HAWC",
           maptree,
           response,
           roi)

# Use from bin 1 to bin 9
hawc.set_active_measurements(1, 9)

# Display information about the data loaded and the ROI
hawc.display()

'''
Define model: Two sources, 1 point, 1 extended

Same declination, but offset in RA

Different spectral index, but both power laws

'''
pt_shift=3.0
ext_shift = 1.0

# First source
spectrum1 = Powerlaw()
source1 = PointSource("point", ra=ra_src + pt_shift, dec=dec_src, spectral_shape=spectrum1)

spectrum1.K = 1e-12 / (u.TeV * u.cm ** 2 * u.s)
spectrum1.piv = 1 * u.TeV
spectrum1.index = -2.3

spectrum1.piv.fix = True
spectrum1.K.fix = True
spectrum1.index.fix = True

# Second source
shape = Gaussian_on_sphere(lon0=ra_src - ext_shift, lat0=dec_src, sigma=0.3)
spectrum2 = Powerlaw()
source2 = ExtendedSource("extended", spatial_shape=shape, spectral_shape=spectrum2)

spectrum2.K = 1e-12 / (u.TeV * u.cm ** 2 * u.s) 
spectrum2.piv = 1 * u.TeV
spectrum2.index = -2.0  

shape.lon0.fix=True
shape.lat0.fix=True
shape.sigma.fix=True
spectrum2.piv.fix = True
spectrum2.K.fix = True
spectrum2.index.fix = True

# Define model with both sources
model = Model(source1, source2)

# Define the data we are using
data = DataList(hawc)

# Define the JointLikelihood object (glue the data to the model)
jl = JointLikelihood(model, data, verbose=False)

# This has the effect of loading the model cache 
fig = hawc.display_spectrum()

# the test file names
model_file_name = "{0}/test_model.hdf5".format(output)
residual_file_name = "{0}/test_residual.hdf5".format(output)

# Write the map trees for testing
model_map_tree = hawc.write_model_map(model_file_name, poisson_fluctuate=True, test_return_map=True)
residual_map_tree = hawc.write_residual_map(residual_file_name, test_return_map=True)

# Read the maps back in
hawc_model = map_tree_factory(model_file_name,roi)
hawc_residual = map_tree_factory(residual_file_name,roi)

# Check to see if it worked!
for model_bin_name, residual_bin_name in zip(hawc_model.analysis_bins_labels, hawc_residual.analysis_bins_labels):

    assert model_bin_name == residual_bin_name

    '''
    Check the model maps!
    '''
    model_bin = hawc_model[model_bin_name]

    check_model = model_map_tree[model_bin_name]

    # good model file
    assert all( model_bin.observation_map.as_partial() == check_model.observation_map.as_partial() )
    assert all( model_bin.background_map.as_partial() == check_model.background_map.as_partial() )
    #good pixel ids (defined as same for both bkg and obs, so we only test one
    assert all( model_bin.observation_map.pixels_ids == check_model.observation_map.pixels_ids )

    '''
    Now check the residual maps
    '''
    residual_bin = hawc_residual[residual_bin_name]

    check_residual = residual_map_tree[residual_bin_name]

    # good residual file
    assert all( residual_bin.observation_map.as_partial() == check_residual.observation_map.as_partial() )
    assert all( residual_bin.background_map.as_partial() == check_residual.background_map.as_partial() )
    #good pixel ids (defined as same for both bkg and obs, so we only test one
    assert all( residual_bin.observation_map.pixels_ids == check_residual.observation_map.pixels_ids )
