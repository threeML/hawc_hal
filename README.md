# The HAWC Accelerated Likelihood (HAL) framework

## Installation

```bash
> pip install git+https://github.com/giacomov/hawc_hal.git --upgrade
```

## Example

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

param_df, like_df = jl.fit()

# See the model in counts space and the residuals
fig = hawc.display_spectrum()
# Save it to file
fig.savefig("hal_mkn421_residuals.png")

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
fig = bs.corner_plot()

fig.savefig("hal_corner_plot.png")
```