![CI](https://github.com/threeML/hawc_hal/workflows/CI/badge.svg)
[![codecov](https://codecov.io/gh/giacomov/hawc_hal/branch/master/graph/badge.svg)](https://codecov.io/gh/giacomov/hawc_hal)
[![Maintainability](https://api.codeclimate.com/v1/badges/7a1c8e60a5cde4275292/maintainability)](https://codeclimate.com/github/giacomov/hawc_hal/maintainability)
[![Scrutinizer Code Quality](https://scrutinizer-ci.com/g/giacomov/hawc_hal/badges/quality-score.png?b=master)](https://scrutinizer-ci.com/g/giacomov/hawc_hal/?branch=master)

# The HAWC Accelerated Likelihood (HAL) framework

## Installation

`hawc_hal` depends on `astromodels`, `threeML` as well as some additional packages (`numba`, `root`, `root_numpy`). 

If you don't have `mamba`,install mamba according to the [instruction](https://github.com/mamba-org/mamba) into the `base` environment.

For install in a new conda environment, we recommend to use the following precoedure:


```
mamba create --name new_hal -c conda-forge -c threeml numpy scipy matplotlib ipython numba reproject "astromodels>=2" "threeml>=2" root
conda activate new_hal
pip install --no-binary :all: root_numpy
pip install git+https://github.com/threeml/hawc_hal.git
```

**For the time being, we recommend updating to the master version of astromodels and threeML from github:**

```
pip install --upgrade git+https://github.com/threeml/astromodels.git
pip install --upgrade git+https://github.com/threeml/threeML.git
```

In particular, we recommend not to install the `root_numpy` binaries via conda or pip. 

The above will install a new python 3 environment. There seem to be version conflicts that currently prevent installing `hawc_hal` with the newer (>=2.0) versions of `threeML` and `astromodels`.

You can also add `hawc_hal` to an existing environment. If you have `conda` installed, it is highly reccomended that you install `numba` through conda like this (simply skip this step if you are not running in a `conda` environment):

```bash
> conda install -c conda-forge numba
```

You also need `root` (whether installed through conda or not) and `threeML`/`astromodels` and their dependencies.

Then:

```bash
> pip install --no-binary :all: root_numpy 
> pip uninstall hawc_hal -y ; pip install git+https://github.com/torresramiro350/hawc_hal.git
```

## Check installation

Use the following commands to check if your installation was successful. You should be inside your conda environment for this.

* To test threeML: `pytest --pyargs threeML`
* To test astromodels: `pytest --pyargs astromodels`
* To test HAL:  `pytest --pyargs hawc_hal`

If you are interested in more detailed output from the tests, learn more about pytest command line options [here](https://docs.pytest.org/en/reorganize-docs/new-docs/user/commandlineuseful.html#).



## Examples

You can find a worked example relying only on publicly accessible data on the [threeML documentation](https://threeml.readthedocs.io/en/latest/notebooks/hal_example.html) 
(or download the [notebook](https://github.com/threeML/threeML/blob/master/docs/notebooks/hal_example.ipynb)).

### Mrk 421 analysis example

(This example assumes you have access to an all-sky HAWC dataset)

```python
from hawc_hal import HAL, HealpixConeROI
import matplotlib.pyplot as plt
from threeML import *

# Define the ROI
ra_mkn421, dec_mkn421 = 166.113808, 38.208833
data_radius = 3.0
model_radius = 8.0

roi = HealpixConeROI(data_radius=data_radius,
                     model_radius=model_radius,
                     ra=ra_mkn421,
                     dec=dec_mkn421)

# Instance the plugin

maptree = ... # This can be either a ROOT or a hdf5 file
response = ... # This can be either a ROOT or hdf5 file

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
fig.savefig("hal_mkn421_stacked_image.png")

# If you want, you can save the data *within this ROI* and the response
# in hd5 files that can be used again with HAL
# (this is useful if you want to publish only the ROI you
# used for a given paper)
hawc.write("my_response.hd5", "my_maptree.hd5")

# Define model as usual
spectrum = Log_parabola()
source = PointSource("mkn421", ra=ra_mkn421, dec=dec_mkn421, spectral_shape=spectrum)

spectrum.piv = 1 * u.TeV
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
jl.set_minimizer("ROOT")
param_df, like_df = jl.fit()

# See the model in counts space and the residuals
fig = hawc.display_spectrum()
# Save it to file
fig.savefig("hal_mkn421_residuals.png")

# See the spectrum fit
fig = plot_point_source_spectra(jl.results,
                                ene_min=0.1,
                                ene_max=100,
                                num_ene=50,
                                energy_unit='TeV',
                                flux_unit='TeV/(s cm2)')
fig.savefig("hal_mkn421_fit_spectrum.png")

# Look at the different energy planes (the columns are model, data, residuals)
fig = hawc.display_fit(smoothing_kernel_sigma=0.3)
fig.savefig("hal_mkn421_fit_planes.png")

# Compute TS
jl.compute_TS("mkn421", like_df)

# Compute goodness of fit with Monte Carlo
gf = GoodnessOfFit(jl)
gof, param, likes = gf.by_mc(100)
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
param.loc[(slice(None), ['mkn421.spectrum.main.Log_parabola.beta']), 'value'].plot()
fig.savefig("hal_sim_all_beta.png")

# Free the position of the source
source.position.ra.free = True
source.position.dec.free = True

# Set boundaries (no need to go further than this)
source.position.ra.bounds = (ra_mkn421 - 0.5, ra_mkn421 + 0.5)
source.position.dec.bounds = (dec_mkn421 - 0.5, dec_mkn421 + 0.5)

# Fit with position free
param_df, like_df = jl.fit()

# Make localization contour
a, b, cc, fig = jl.get_contours(model.mkn421.position.dec, 38.15, 38.22, 10,
                                model.mkn421.position.ra, 166.08, 166.18, 10, )

plt.plot([ra_mkn421], [dec_mkn421], 'x')
fig.savefig("hal_mkn421_localization.png")

# Of course we can also do a Bayesian analysis the usual way
# NOTE: here the position is still free, so we are going to obtain marginals about that
# as well
# For this quick example, let's use a uniform prior for all parameters
for parameter in model.parameters.values():

    if parameter.fix:
        continue

    if parameter.is_normalization:

        parameter.set_uninformative_prior(Log_uniform_prior)

    else:

        parameter.set_uninformative_prior(Uniform_prior)

# Let's execute our bayes analysis
bs = BayesianAnalysis(model, data)
samples = bs.sample(30, 100, 100)
fig = bs.results.corner_plot()

fig.savefig("hal_corner_plot.png")
```

### Convert ROOT maptree to hdf5 maptree

```python
from hawc_hal.maptree import map_tree_factory
from hawc_hal import HealpixConeROI

root_map_tree = "maptree_1024.root" # path to your ROOT maptree

# Export the entire map tree (full sky)
m = map_tree_factory(root_map_tree, None)
m.write("full_sky_maptree.hd5")

# Export only the ROI. This is a file only a few Mb in size
# that can be provided as dataset to journals, for example
ra_mkn421, dec_mkn421 = 166.113808, 38.208833
data_radius = 3.0
model_radius = 8.0

roi = HealpixConeROI(data_radius=data_radius,
                     model_radius=model_radius,
                     ra=ra_mkn421,
                     dec=dec_mkn421)

m = map_tree_factory(root_map_tree, roi)
m.write("roi_maptree.hd5")                

```
